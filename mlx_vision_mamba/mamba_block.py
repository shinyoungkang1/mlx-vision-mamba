"""
Mamba (Selective State Space Model) block in MLX.

Implements the core Mamba block from Gu & Dao (2023):
  in_proj -> conv1d -> SiLU -> selective_scan -> gate -> out_proj

Supports bidirectional scanning for vision applications.
Supports multi-directional scanning (VMamba-style) for 2D/3D/4D inputs.

Reference: https://arxiv.org/abs/2312.00752
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def selective_scan(
    x: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
) -> mx.array:
    """
    Selective scan (S6) — the core SSM recurrence.

    Sequential implementation (no fused kernel). Correct for training via
    MLX autograd. For large sequences, this is the bottleneck.

    Args:
        x:     (B, L, D_inner)  — input after conv1d + SiLU
        delta: (B, L, D_inner)  — discretization step sizes
        A:     (D_inner, N)     — state transition (log-space, negative)
        B:     (B, L, N)        — input-to-state projection
        C:     (B, L, N)        — state-to-output projection
        D:     (D_inner,)       — skip connection

    Returns:
        y: (B, L, D_inner)
    """
    batch, seq_len, d_inner = x.shape
    n_state = A.shape[1]

    # Discretize: deltaA = exp(delta * A), deltaB_x = delta * B * x
    # delta: (B, L, D_inner) -> (B, L, D_inner, 1)
    # A:     (D_inner, N)    -> (1, 1, D_inner, N)
    delta_exp = mx.expand_dims(delta, axis=-1)      # (B, L, D_inner, 1)
    A_exp = mx.expand_dims(mx.expand_dims(A, 0), 0)  # (1, 1, D_inner, N)
    deltaA = mx.exp(delta_exp * A_exp)               # (B, L, D_inner, N)

    # B: (B, L, N) -> (B, L, 1, N)
    # x: (B, L, D_inner) -> (B, L, D_inner, 1)
    B_exp = mx.expand_dims(B, axis=2)                # (B, L, 1, N)
    x_exp = mx.expand_dims(x, axis=-1)               # (B, L, D_inner, 1)
    deltaB_x = delta_exp * B_exp * x_exp              # (B, L, D_inner, N)

    # Sequential scan over time
    h = mx.zeros((batch, d_inner, n_state))
    ys = []
    for t in range(seq_len):
        h = deltaA[:, t] * h + deltaB_x[:, t]        # (B, D_inner, N)
        # Output: sum over state dim
        y_t = (h * mx.expand_dims(C[:, t], axis=1)).sum(axis=-1)  # (B, D_inner)
        ys.append(y_t)

    y = mx.stack(ys, axis=1)  # (B, L, D_inner)

    # Skip connection
    y = y + x * D
    return y


class MambaBlock(nn.Module):
    """
    Single Mamba block: linear projections + conv1d + selective scan + gating.

    Input/output: (B, L, D) -> (B, L, D)

    Args:
        scan_mode: "sequential" (default, safe), "chunked" (faster),
                   or "parallel" (experimental). See mamba_fast.py.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        scan_mode: str = "sequential",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.scan_mode = scan_mode

        # Select scan function
        if scan_mode == "sequential":
            self._scan_fn = selective_scan
        elif scan_mode == "chunked":
            from mlx_vision_mamba.mamba_fast import selective_scan_chunked
            self._scan_fn = selective_scan_chunked
        elif scan_mode == "parallel":
            from mlx_vision_mamba.mamba_fast import selective_scan_parallel
            self._scan_fn = selective_scan_parallel
        elif scan_mode == "metal":
            from mlx_vision_mamba.mamba_metal import selective_scan_metal
            self._scan_fn = selective_scan_metal
        elif scan_mode == "metal_train":
            from mlx_vision_mamba.mamba_metal import selective_scan_metal_trainable
            self._scan_fn = selective_scan_metal_trainable
        elif scan_mode == "metal_fused":
            from mlx_vision_mamba.mamba_metal_fused import selective_scan_metal_fused
            self._scan_fn = selective_scan_metal_fused
        else:
            self._scan_fn = selective_scan

        # Input projection: d_model -> 2 * d_inner (for x and gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise conv1d on the x branch
        # Causal padding: pad left by (d_conv - 1), no right padding
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )

        # SSM parameter projections (input-dependent / "selective")
        # x -> (delta, B, C) projections
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Delta projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: initialized as negative log-spaced values (will stay negative via -exp)
        A = mx.repeat(
            mx.expand_dims(mx.arange(1, d_state + 1, dtype=mx.float32), axis=0),
            repeats=self.d_inner, axis=0,
        )  # (d_inner, N)
        self.A_log = mx.log(A)

        # D: skip connection parameter
        self.D = mx.ones((self.d_inner,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        # Project input to x_branch and gate
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_branch = xz[:, :, :self.d_inner]            # (B, L, d_inner)
        z = xz[:, :, self.d_inner:]                    # (B, L, d_inner)

        # Conv1d (causal: trim right padding)
        x_branch = self.conv1d(x_branch)[:, :L, :]    # (B, L, d_inner)
        x_branch = nn.silu(x_branch)

        # SSM parameters from input (selective mechanism)
        x_dbc = self.x_proj(x_branch)                 # (B, L, dt_rank + 2*N)
        dt = x_dbc[:, :, :self.dt_rank]                # (B, L, dt_rank)
        B_proj = x_dbc[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, N)
        C_proj = x_dbc[:, :, self.dt_rank + self.d_state:]              # (B, L, N)

        # Delta: project and softplus
        delta = self.dt_proj(dt)                       # (B, L, d_inner)
        delta = mx.log(1.0 + mx.exp(delta))           # softplus

        # A is kept negative
        A = -mx.exp(self.A_log)                        # (d_inner, N)

        # Run selective scan
        y = self._scan_fn(x_branch, delta, A, B_proj, C_proj, self.D)

        # Gate and project out
        y = y * nn.silu(z)
        y = self.out_proj(y)                           # (B, L, D)
        return y


class ResidualMambaBlock(nn.Module):
    """Mamba block with pre-norm and residual connection."""

    def __init__(self, d_model: int = 384, d_state: int = 16,
                 d_conv: int = 4, expand_factor: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand_factor)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.mamba(self.norm(x))


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for vision.

    Runs two independent Mamba blocks: one on the forward sequence,
    one on the reversed sequence. Outputs are summed.

    This gives the SSM access to both past and future context,
    which is essential for spatial data (patches have no causal order).
    """

    def __init__(self, d_model: int = 384, d_state: int = 16,
                 d_conv: int = 4, expand_factor: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand_factor)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand_factor)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)

        # Forward scan
        y_fwd = self.forward_mamba(h)

        # Backward scan (reverse sequence, scan, reverse back)
        h_rev = h[:, ::-1, :]
        y_bwd = self.backward_mamba(h_rev)[:, ::-1, :]

        # Merge via summation + residual
        return x + y_fwd + y_bwd


# ---------------------------------------------------------------------------
# Scan-order permutation helpers for MultiDirectionalMambaBlock
# ---------------------------------------------------------------------------

def _flatten_row_major(h: mx.array, grid_shape: Tuple[int, ...]) -> mx.array:
    """Identity: row-major order is already the default flattening."""
    return h


def _unflatten_row_major(h: mx.array, grid_shape: Tuple[int, ...]) -> mx.array:
    """Identity inverse."""
    return h


def _permute_2d_col_major(h: mx.array, grid_shape: Tuple[int, ...]) -> mx.array:
    """
    Reorder a flat (B, H*W, D) sequence from row-major to column-major.
    Row-major index (r, c) -> col-major index (c, r).
    """
    B, N, D = h.shape
    H, W = grid_shape
    # Reshape to grid, transpose spatial dims, flatten back
    h = h.reshape(B, H, W, D)
    h = mx.transpose(h, axes=(0, 2, 1, 3))   # (B, W, H, D)
    return h.reshape(B, N, D)


def _unpermute_2d_col_major(h: mx.array, grid_shape: Tuple[int, ...]) -> mx.array:
    """Inverse of column-major permutation."""
    B, N, D = h.shape
    H, W = grid_shape
    # Currently in col-major (W, H) order; transpose back to (H, W)
    h = h.reshape(B, W, H, D)
    h = mx.transpose(h, axes=(0, 2, 1, 3))   # (B, H, W, D)
    return h.reshape(B, N, D)


def _permute_3d_axis(h: mx.array, grid_shape: Tuple[int, ...],
                     axes_order: Tuple[int, ...]) -> mx.array:
    """
    Reorder a flat (B, Dx*Dy*Dz, D) sequence according to a spatial axis
    permutation. axes_order is a permutation of (0, 1, 2) specifying the
    new traversal order of the spatial dims.
    """
    B, N, D = h.shape
    Dx, Dy, Dz = grid_shape
    h = h.reshape(B, Dx, Dy, Dz, D)
    # Build full transpose axes: batch + permuted spatial + channel
    perm = (0,) + tuple(a + 1 for a in axes_order) + (4,)
    h = mx.transpose(h, axes=perm)
    return h.reshape(B, N, D)


def _unpermute_3d_axis(h: mx.array, grid_shape: Tuple[int, ...],
                       axes_order: Tuple[int, ...]) -> mx.array:
    """Inverse of _permute_3d_axis."""
    B, N, D = h.shape
    Dx, Dy, Dz = grid_shape
    # After the forward permutation, the spatial dims are in axes_order.
    # Compute the permuted shape.
    permuted_shape = tuple(grid_shape[a] for a in axes_order)
    h = h.reshape(B, *permuted_shape, D)
    # Compute inverse permutation
    inv_perm = [0] * 3
    for i, a in enumerate(axes_order):
        inv_perm[a] = i
    full_inv = (0,) + tuple(a + 1 for a in inv_perm) + (4,)
    h = mx.transpose(h, axes=full_inv)
    return h.reshape(B, N, D)


def _permute_4d_axis(h: mx.array, grid_shape: Tuple[int, ...],
                     axes_order: Tuple[int, ...]) -> mx.array:
    """
    Reorder a flat (B, T*Dx*Dy*Dz, D) sequence according to a spatial axis
    permutation. axes_order is a permutation of (0, 1, 2, 3) specifying the
    new traversal order of the 4 spatiotemporal dims.
    """
    B, N, D = h.shape
    h = h.reshape(B, *grid_shape, D)
    ndim_spatial = len(grid_shape)
    perm = (0,) + tuple(a + 1 for a in axes_order) + (ndim_spatial + 1,)
    h = mx.transpose(h, axes=perm)
    return h.reshape(B, N, D)


def _unpermute_4d_axis(h: mx.array, grid_shape: Tuple[int, ...],
                       axes_order: Tuple[int, ...]) -> mx.array:
    """Inverse of _permute_4d_axis."""
    B, N, D = h.shape
    ndim_spatial = len(grid_shape)
    permuted_shape = tuple(grid_shape[a] for a in axes_order)
    h = h.reshape(B, *permuted_shape, D)
    inv_perm = [0] * ndim_spatial
    for i, a in enumerate(axes_order):
        inv_perm[a] = i
    full_inv = (0,) + tuple(a + 1 for a in inv_perm) + (ndim_spatial + 1,)
    h = mx.transpose(h, axes=full_inv)
    return h.reshape(B, N, D)


class MultiDirectionalMambaBlock(nn.Module):
    """
    Multi-directional Mamba block for vision (VMamba-style).

    Creates K independent MambaBlock instances, each scanning the input
    sequence in a different spatial direction. Outputs are summed with a
    residual connection.

    Supported configurations:
        K=2: forward + backward (equivalent to BidirectionalMambaBlock)
        K=4: 2D — forward-row, backward-row, forward-col, backward-col
        K=6: 3D — scan along +X, -X, +Y, -Y, +Z, -Z
        K=8: 4D — scan along +T, -T, +X, -X, +Y, -Y, +Z, -Z

    The block receives flat patch sequences (B, N, D) and uses grid_shape
    to determine how to reorder patches for each scan direction.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Convolution kernel size.
        expand_factor: Inner dimension expansion factor.
        num_directions: Number of scan directions K (2, 4, 6, or 8).
        grid_shape: Spatial grid shape, e.g. (14, 14) for 2D, (8, 8, 8)
                     for 3D, (4, 8, 8, 8) for 4D. Required when K > 2.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        num_directions: int = 2,
        grid_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        assert num_directions in (2, 4, 6, 8), (
            f"num_directions must be 2, 4, 6, or 8, got {num_directions}"
        )
        if num_directions > 2:
            assert grid_shape is not None, (
                "grid_shape is required when num_directions > 2"
            )
        if num_directions == 4:
            assert grid_shape is not None and len(grid_shape) == 2, (
                f"K=4 requires 2D grid_shape (H, W), got {grid_shape}"
            )
        elif num_directions == 6:
            assert grid_shape is not None and len(grid_shape) == 3, (
                f"K=6 requires 3D grid_shape (Dx, Dy, Dz), got {grid_shape}"
            )
        elif num_directions == 8:
            assert grid_shape is not None and len(grid_shape) == 4, (
                f"K=8 requires 4D grid_shape (T, Dx, Dy, Dz), got {grid_shape}"
            )

        self.num_directions = num_directions
        self.grid_shape = grid_shape

        self.norm = nn.LayerNorm(d_model)
        self.mamba_blocks = [
            MambaBlock(d_model, d_state, d_conv, expand_factor)
            for _ in range(num_directions)
        ]

    def _apply_direction(self, h: mx.array, direction_idx: int) -> mx.array:
        """
        Apply the appropriate scan-order permutation for direction_idx,
        run through the corresponding MambaBlock, then undo the permutation.

        For K=2:
            0: forward (identity)
            1: backward (reverse)

        For K=4 (2D):
            0: forward along rows (identity)
            1: backward along rows (reverse)
            2: forward along columns (transpose to col-major)
            3: backward along columns (transpose to col-major + reverse)

        For K=6 (3D):
            0: +X scan — axis order (0,1,2) = identity
            1: -X scan — identity + reverse
            2: +Y scan — axis order (1,0,2) = Y-first
            3: -Y scan — Y-first + reverse
            4: +Z scan — axis order (2,0,1) = Z-first
            5: -Z scan — Z-first + reverse

        For K=8 (4D):
            0: +T scan — axis order (0,1,2,3) = identity
            1: -T scan — identity + reverse
            2: +X scan — axis order (1,0,2,3) = X-first
            3: -X scan — X-first + reverse
            4: +Y scan — axis order (2,0,1,3) = Y-first
            5: -Y scan — Y-first + reverse
            6: +Z scan — axis order (3,0,1,2) = Z-first
            7: -Z scan — Z-first + reverse
        """
        mamba = self.mamba_blocks[direction_idx]
        K = self.num_directions

        if K == 2:
            if direction_idx == 0:
                return mamba(h)
            else:
                out = mamba(h[:, ::-1, :])
                return out[:, ::-1, :]

        elif K == 4:
            # 2D directions
            pair_idx = direction_idx // 2   # 0=row, 1=col
            is_reverse = (direction_idx % 2 == 1)

            if pair_idx == 0:
                # Row-major (identity permutation)
                seq = h
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                return out
            else:
                # Column-major
                seq = _permute_2d_col_major(h, self.grid_shape)
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                out = _unpermute_2d_col_major(out, self.grid_shape)
                return out

        elif K == 6:
            # 3D directions: pairs of (forward, backward) along each axis
            # Axis permutations that put the target axis first
            axis_perms = [
                (0, 1, 2),  # X-first (identity)
                (1, 0, 2),  # Y-first
                (2, 0, 1),  # Z-first
            ]
            pair_idx = direction_idx // 2
            is_reverse = (direction_idx % 2 == 1)
            perm = axis_perms[pair_idx]

            if perm == (0, 1, 2):
                # Identity permutation, no reshape needed
                seq = h
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                return out
            else:
                seq = _permute_3d_axis(h, self.grid_shape, perm)
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                out = _unpermute_3d_axis(out, self.grid_shape, perm)
                return out

        elif K == 8:
            # 4D directions: pairs of (forward, backward) along each axis
            axis_perms = [
                (0, 1, 2, 3),  # T-first (identity)
                (1, 0, 2, 3),  # X-first
                (2, 0, 1, 3),  # Y-first
                (3, 0, 1, 2),  # Z-first
            ]
            pair_idx = direction_idx // 2
            is_reverse = (direction_idx % 2 == 1)
            perm = axis_perms[pair_idx]

            if perm == (0, 1, 2, 3):
                seq = h
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                return out
            else:
                seq = _permute_4d_axis(h, self.grid_shape, perm)
                if is_reverse:
                    seq = seq[:, ::-1, :]
                out = mamba(seq)
                if is_reverse:
                    out = out[:, ::-1, :]
                out = _unpermute_4d_axis(out, self.grid_shape, perm)
                return out

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, N, D) — flat patch sequence.
        Returns:
            (B, N, D) — output with residual.

        When N != prod(grid_shape) (e.g., context encoder processes a subset
        of patches), spatial permutations are invalid. Falls back to K=2
        bidirectional scanning (forward + backward only, using first 2 blocks).
        """
        B, N, D = x.shape
        h = self.norm(x)

        expected_n = 1
        for g in self.grid_shape:
            expected_n *= g

        if N != expected_n and self.num_directions > 2:
            # Subset of patches — can't do spatial permutations.
            # Fall back to forward + backward (K=2) using first two blocks.
            y_fwd = self.mamba_blocks[0](h)
            y_bwd = self.mamba_blocks[1](h[:, ::-1, :])[:, ::-1, :]
            return x + y_fwd + y_bwd

        # Full grid — run all K directions with spatial permutations
        y = self._apply_direction(h, 0)
        for k in range(1, self.num_directions):
            y = y + self._apply_direction(h, k)

        return x + y
