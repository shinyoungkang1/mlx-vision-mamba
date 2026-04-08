"""
Tier 2: Chunked Parallel Scan Metal Kernel.

Within each chunk of CHUNK_SIZE timesteps, uses threadgroup shared memory
for a Hillis-Steele parallel prefix scan. Between chunks, the carry state
is propagated sequentially (only L/CHUNK_SIZE steps).

For L=196, CHUNK_SIZE=32: 7 sequential carry steps instead of 196.

The associative scan operator for the SSM recurrence is:
  (A1, b1) ⊕ (A2, b2) = (A1 * A2, A1 * b2 + b1)
where A_t = deltaA[t] and b_t = deltaBx[t].
"""

import mlx.core as mx

# The chunked kernel processes CHUNK_SIZE timesteps at a time.
# Within a chunk:
#   - Each thread in the threadgroup handles one timestep
#   - Hillis-Steele scan in shared memory (log2(CHUNK_SIZE) steps)
# Between chunks:
#   - The last state of chunk k is the initial state for chunk k+1
#   - This is a single multiply+add per state element

CHUNKED_FWD_SOURCE = """
    // Grid: (batch_size * d_inner, 1, 1)
    // Each thread handles one (batch, channel) pair across ALL chunks
    uint gid = thread_position_in_grid.x;
    uint batch_idx = gid / d_inner;
    uint ch = gid % d_inner;

    if (batch_idx >= batch_size) return;

    float d_skip = D_param[ch];

    // Carry state between chunks
    float h[16];
    for (int n = 0; n < n_state; n++) h[n] = 0.0f;

    // Process chunks
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        int chunk_end = chunk_start + chunk_size;
        if (chunk_end > seq_len) chunk_end = seq_len;
        int cs = chunk_end - chunk_start;

        // Within this chunk, process sequentially (but the chunk is small)
        // Tier 2 improvement: the chunk is processed in registers, no shared memory needed
        // because each thread owns its own (batch, channel) — no cross-thread communication
        // The parallelism is across (batch * d_inner) threads, not across time within a chunk.

        for (int t = chunk_start; t < chunk_end; t++) {
            int xd_idx = (batch_idx * seq_len + t) * d_inner + ch;
            float x_val = x_in[xd_idx];
            float delta_val = delta_in[xd_idx];
            int bc_base = (batch_idx * seq_len + t) * n_state;

            float y_t = 0.0f;
            for (int n = 0; n < n_state; n++) {
                float a_val = A_param[ch * n_state + n];
                float dA = exp(delta_val * a_val);
                float dBx = delta_val * B_in[bc_base + n] * x_val;

                h[n] = dA * h[n] + dBx;
                y_t += h[n] * C_in[bc_base + n];

                if (save_h_flag) {
                    int h_idx = ((batch_idx * seq_len + t) * d_inner + ch) * n_state + n;
                    h_out[h_idx] = h[n];
                }
            }
            y_out[xd_idx] = y_t + x_val * d_skip;
        }
    }
"""

# Note: The "chunked" kernel above doesn't actually parallelize across time
# because each thread owns one (batch, channel) pair. The parallel scan
# across time would require a DIFFERENT threading model where threads
# within a threadgroup cooperate on different timesteps for the SAME channel.
#
# True Tier 2 architecture:
#   Grid:        (batch_size * d_inner * num_chunks, 1, 1)
#   Threadgroup: (CHUNK_SIZE, 1, 1)
#   Each threadgroup processes one chunk of one channel of one batch item.
#   Threads within the group cooperate via shared memory to parallel-scan the chunk.
#
# This is significantly more complex because:
# 1. The associative scan operates on N-dim state vectors, not scalars
# 2. Shared memory per threadgroup is limited (~32KB)
# 3. The carry between chunks still needs sequential propagation

TRUE_PARALLEL_FWD_SOURCE = """
    // Threading:
    //   gridDim.x = batch_size * d_inner * num_chunks
    //   threadIdx.x = position within chunk (0..CHUNK_SIZE-1)
    //
    // Decode which (batch, channel, chunk) this threadgroup handles
    uint tg_id = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    uint num_chunks_total = (seq_len + chunk_size - 1) / chunk_size;
    uint batch_ch_id = tg_id / num_chunks_total;
    uint chunk_id = tg_id % num_chunks_total;
    uint batch_idx = batch_ch_id / d_inner;
    uint ch = batch_ch_id % d_inner;

    if (batch_idx >= batch_size) return;

    int t = chunk_id * chunk_size + tid;
    bool valid = (t < seq_len) && (tid < chunk_size);

    float d_skip = D_param[ch];

    // Load this timestep's data
    float local_dA[16];   // deltaA for this timestep, per state dim
    float local_dBx[16];  // deltaBx for this timestep
    float local_C[16];
    float local_x = 0.0f;

    if (valid) {
        int xd_idx = (batch_idx * seq_len + t) * d_inner + ch;
        local_x = x_in[xd_idx];
        float delta_val = delta_in[xd_idx];
        int bc_base = (batch_idx * seq_len + t) * n_state;

        for (int n = 0; n < n_state; n++) {
            float a_val = A_param[ch * n_state + n];
            local_dA[n] = exp(delta_val * a_val);
            local_dBx[n] = delta_val * B_in[bc_base + n] * local_x;
            local_C[n] = C_in[bc_base + n];
        }
    } else {
        for (int n = 0; n < n_state; n++) {
            local_dA[n] = 1.0f;   // identity for scan
            local_dBx[n] = 0.0f;
            local_C[n] = 0.0f;
        }
    }

    // Hillis-Steele inclusive scan within the chunk
    // Scan operator: (A1, b1) ⊕ (A2, b2) = (A1*A2, A1*b2 + b1)
    // We scan both A (multiplicative) and b (the combined state)

    // Shared memory for scan communication
    // Each thread needs N floats for A and N floats for b
    threadgroup float shared_A[32 * 16];  // CHUNK_SIZE * N_STATE
    threadgroup float shared_b[32 * 16];

    // Initialize shared memory with this thread's values
    for (int n = 0; n < n_state; n++) {
        shared_A[tid * n_state + n] = local_dA[n];
        shared_b[tid * n_state + n] = local_dBx[n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele scan: log2(CHUNK_SIZE) steps
    for (int stride = 1; stride < chunk_size; stride *= 2) {
        float temp_A[16];
        float temp_b[16];

        if (tid >= (uint)stride) {
            for (int n = 0; n < n_state; n++) {
                float A_left = shared_A[(tid - stride) * n_state + n];
                float b_left = shared_b[(tid - stride) * n_state + n];
                float A_cur = shared_A[tid * n_state + n];
                float b_cur = shared_b[tid * n_state + n];

                // (A_left, b_left) ⊕ (A_cur, b_cur) = (A_left * A_cur, A_cur * b_left + b_cur)
                // The recurrence is h_t = A_t * h_{t-1} + b_t
                // So the RIGHT element's A acts on the LEFT element's b
                temp_A[n] = A_left * A_cur;
                temp_b[n] = A_cur * b_left + b_cur;
            }
        } else {
            for (int n = 0; n < n_state; n++) {
                temp_A[n] = shared_A[tid * n_state + n];
                temp_b[n] = shared_b[tid * n_state + n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int n = 0; n < n_state; n++) {
            shared_A[tid * n_state + n] = temp_A[n];
            shared_b[tid * n_state + n] = temp_b[n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After scan: shared_b[tid] contains h[t] for this chunk
    // (assuming h_init = 0 for the first chunk; carry from previous chunk handled outside)
    if (valid) {
        float y_t = 0.0f;
        for (int n = 0; n < n_state; n++) {
            float h_val = shared_b[tid * n_state + n];
            y_t += h_val * local_C[n];

            if (save_h_flag) {
                int h_idx = ((batch_idx * seq_len + t) * d_inner + ch) * n_state + n;
                h_out[h_idx] = h_val;
            }
        }

        int xd_idx = (batch_idx * seq_len + t) * d_inner + ch;
        y_out[xd_idx] = y_t + local_x * d_skip;
    }

    // Save the chunk's cumulative A product and final state for carry propagation
    // Only the last thread in the chunk writes the carry
    if (tid == chunk_size - 1 || t == seq_len - 1) {
        for (int n = 0; n < n_state; n++) {
            int carry_idx = (batch_ch_id * num_chunks_total + chunk_id) * n_state + n;
            carry_A[carry_idx] = shared_A[tid * n_state + n];
            carry_b[carry_idx] = shared_b[tid * n_state + n];
        }
    }
"""


def _launch_chunked_fwd(x, delta, A, B, C, D, chunk_size=32, save_h=False):
    """
    Launch Tier 2 chunked parallel forward kernel.

    NOTE: This is the true parallel version where threads within a threadgroup
    cooperate on different timesteps via shared memory Hillis-Steele scan.

    The carry between chunks is handled in a second pass.
    """
    batch, seq_len, d_inner = x.shape
    n_state = A.shape[1]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    # Always include h_out in outputs (Metal needs it declared even if unused)
    output_names = ["y_out", "carry_A", "carry_b", "h_out"]
    output_shapes = [
        (batch, seq_len, d_inner),                      # y
        (batch * d_inner, num_chunks, n_state),         # carry_A
        (batch * d_inner, num_chunks, n_state),         # carry_b
        (batch, seq_len, d_inner, n_state) if save_h else (1, 1, 1, 1),  # h_out (dummy if not saving)
    ]
    output_dtypes = [mx.float32, mx.float32, mx.float32, mx.float32]

    kernel = mx.fast.metal_kernel(
        name=f"ssm_chunked_par_fwd_cs{chunk_size}" + ("_save" if save_h else ""),
        input_names=["x_in", "delta_in", "A_param", "B_in", "C_in", "D_param"],
        output_names=output_names,
        source=TRUE_PARALLEL_FWD_SOURCE,
    )

    # One threadgroup per (batch, channel, chunk)
    num_threadgroups = batch * d_inner * num_chunks
    outputs = kernel(
        inputs=[x, delta, A, B, C, D],
        template=[
            ("batch_size", batch), ("seq_len", seq_len),
            ("d_inner", d_inner), ("n_state", n_state),
            ("chunk_size", chunk_size), ("save_h_flag", 1 if save_h else 0),
        ],
        grid=(num_threadgroups * chunk_size, 1, 1),
        threadgroup=(chunk_size, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )

    y = outputs[0]
    carry_A = outputs[1]
    carry_b = outputs[2]
    h_saved = outputs[3] if save_h else None
    # Note: carry_A and carry_b are not yet used for inter-chunk propagation

    # TODO: Second pass — propagate carry between chunks
    # For now, this only works correctly for the FIRST chunk (no carry from previous)
    # Full implementation needs a carry propagation kernel

    if save_h:
        return y, h_saved
    return y


def selective_scan_metal_chunked(x, delta, A, B, C, D):
    """
    Tier 2: Chunked parallel Metal scan.

    WARNING: This is a work-in-progress. The inter-chunk carry propagation
    is not yet implemented. Results are only correct for seq_len <= chunk_size.

    For production use, use selective_scan_metal_fused() instead.
    """
    x_c = mx.contiguous(x.astype(mx.float32))
    delta_c = mx.contiguous(delta.astype(mx.float32))
    A_c = mx.contiguous(A.astype(mx.float32))
    B_c = mx.contiguous(B.astype(mx.float32))
    C_c = mx.contiguous(C.astype(mx.float32))
    D_c = mx.contiguous(D.astype(mx.float32))

    return _launch_chunked_fwd(x_c, delta_c, A_c, B_c, C_c, D_c, chunk_size=32)
