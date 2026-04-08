# MLX Vision Mamba

**First Metal-accelerated Vision Mamba for Apple Silicon with 2D/3D/4D support and training VJP.**

A complete Mamba (Selective State Space Model) implementation for MLX that supports multi-dimensional vision inputs. Includes custom Metal kernels for the selective scan that provide **3.9x training speedup** over Python loops.

## Features

- **Multi-dimensional Vision Mamba**: supports 2D images, 3D volumes (CT/MRI), and 4D spatiotemporal data (cine MRI, video)
- **Multi-directional scanning**: K=4 (2D), K=6 (3D), K=8 (4D) scan directions with VMamba-style spatial permutations
- **Custom Metal kernels**: fused selective scan in Metal Shading Language with full training VJP via `mx.custom_function`
- **3.9x training speedup**: over Python sequential scan on Apple Silicon (M5 Pro benchmarked)
- **Drop-in ViT replacement**: same `(B, H, W, C) -> (B, N, D)` interface as Vision Transformer
- **Bidirectional scanning**: forward + backward Mamba blocks for non-causal spatial data

## Benchmark

Tested on Apple M5 Pro, MLX 0.31, batch=2, seq_len=196, d_model=768, d_state=16:

| Mode | Scan Time | Full Training Step | vs Python |
|---|---|---|---|
| Python sequential | 16.4 ms | 0.393 s | 1.0x |
| Metal (forward only) | 2.0 ms | 0.173 s | 2.3x |
| Metal (fwd + VJP) | 1.5 ms | 0.106 s | 3.7x |
| **Metal fused (BEST)** | **1.2 ms** | **0.035 s** | **~4x** |

All gradients verified correct against Python reference (max diff < 1e-6).

## Installation

```bash
pip install mlx
# Clone this repo
git clone https://github.com/YOUR_USERNAME/mlx-vision-mamba.git
cd mlx-vision-mamba
```

## Quick Start

### Basic Mamba Block

```python
import mlx.core as mx
from mlx_vision_mamba import MambaBlock

# Single Mamba block (1D sequence)
block = MambaBlock(d_model=384, d_state=16, scan_mode="metal_fused")
x = mx.random.normal((2, 196, 384))  # (batch, seq_len, dim)
y = block(x)  # (2, 196, 384)
```

### Vision Mamba Encoder (Drop-in ViT Replacement)

```python
from mlx_vision_mamba import VisionMamba

# 2D images (X-ray, natural images)
encoder_2d = VisionMamba(img_size=224, patch_size=16, embed_dim=384, depth=6, input_dim=2)
images = mx.random.normal((2, 224, 224, 1))  # (B, H, W, C)
features = encoder_2d(images)  # (2, 196, 384)

# 3D volumes (CT, MRI) — K=6 multi-directional scanning
encoder_3d = VisionMamba(img_size=64, patch_size=16, embed_dim=384, depth=6, input_dim=3)
volumes = mx.random.normal((1, 64, 64, 64, 1))  # (B, D, H, W, C)
features_3d = encoder_3d(volumes)  # (1, 64, 384)

# 4D spatiotemporal (cine MRI, video) — K=8 directions
encoder_4d = VisionMamba(
    img_size=32, patch_size=16, embed_dim=384, depth=6,
    input_dim=4, num_temporal_frames=4, temporal_patch_size=1,
)
video = mx.random.normal((1, 4, 32, 32, 32, 1))  # (B, T, D, H, W, C)
features_4d = encoder_4d(video)  # (1, 32, 384)
```

### Metal Acceleration

```python
from mlx_vision_mamba import MambaBlock

# Inference (fastest, no autograd)
block = MambaBlock(d_model=384, scan_mode="metal")

# Training with gradients (Metal forward + Metal backward VJP)
block = MambaBlock(d_model=384, scan_mode="metal_fused")
```

### Scan Modes

| Mode | Use Case | Speed |
|---|---|---|
| `sequential` | Safe default, debugging | 1.0x |
| `chunked` | Slightly faster Python scan | ~1.1x |
| `metal` | Inference / frozen layers (no grad) | ~9x scan |
| `metal_train` | Training (Metal fwd + Metal bwd VJP) | ~3.7x |
| `metal_fused` | Training (fused discretization + scan) | **~3.9x** |

## Architecture

### Multi-Directional Scanning

Standard Mamba processes sequences left-to-right (causal). For spatial data with no causal ordering, we use multi-directional scanning:

- **2D (K=4)**: forward-row, backward-row, forward-column, backward-column
- **3D (K=6)**: scan along +X, -X, +Y, -Y, +Z, -Z
- **4D (K=8)**: scan along +T, -T, +X, -X, +Y, -Y, +Z, -Z

Outputs from all directions are summed with a residual connection.

When processing a subset of patches (e.g., JEPA context encoder), automatically falls back to K=2 (forward + backward) since spatial permutations require the full grid.

### Metal Kernel Design

The selective scan recurrence `h = A*h + b; y = C*h + D*x` is the bottleneck — it's inherently sequential over time. Our Metal kernel:

1. **One GPU thread per (batch, channel)** — embarrassingly parallel across B*D_inner
2. **SSM state in thread registers** — N=16 floats (64 bytes) per thread, no shared memory needed
3. **Single kernel launch** — replaces ~200 Python loop iterations with one Metal dispatch
4. **Fused discretization** — exp(delta*A) and delta*B*x computed inline, no intermediate tensors
5. **Custom VJP** via `mx.custom_function` — backward pass is a reverse scan in the same pattern

## Files

```
mlx_vision_mamba/
  __init__.py              — Public API
  mamba_block.py           — MambaBlock, BidirectionalMambaBlock, MultiDirectionalMambaBlock
  vision_mamba.py          — VisionMamba encoder (2D/3D/4D)
  mamba_metal.py           — Metal kernel (forward + VJP)
  mamba_metal_fused.py     — Fused Metal kernel (discretize + scan in one kernel)
  mamba_fast.py            — Python-level optimizations (chunked, parallel)
  vit.py                   — PatchEmbed, PatchEmbed3D (shared with ViT)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kang2026mlx_vision_mamba,
  author = {Kang, Shinyoung},
  title = {MLX Vision Mamba: Metal-Accelerated State Space Models for Multi-Dimensional Vision},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/mlx-vision-mamba},
  license = {Apache-2.0},
}
```

## Acknowledgments

- [Mamba](https://github.com/state-spaces/mamba) by Albert Gu and Tri Dao
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [VMamba](https://github.com/MzeroMiko/VMamba) for multi-directional scanning
- Built with assistance from Claude (Anthropic)

## License

Apache License 2.0. See [LICENSE](LICENSE).
