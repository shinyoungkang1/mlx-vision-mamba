"""
MLX Vision Mamba: Metal-accelerated State Space Models for 2D/3D/4D vision.

First Mamba implementation on MLX with custom Metal kernels and training VJP.
"""

from mlx_vision_mamba.mamba_block import (
    MambaBlock,
    BidirectionalMambaBlock,
    MultiDirectionalMambaBlock,
    selective_scan,
)
from mlx_vision_mamba.vision_mamba import VisionMamba
from mlx_vision_mamba.vit import PatchEmbed, PatchEmbed3D
from mlx_vision_mamba.mamba_metal import (
    selective_scan_metal,
    selective_scan_metal_trainable,
)
from mlx_vision_mamba.mamba_metal_fused import selective_scan_metal_fused

__version__ = "0.1.1"
__all__ = [
    "MambaBlock",
    "BidirectionalMambaBlock",
    "MultiDirectionalMambaBlock",
    "VisionMamba",
    "PatchEmbed",
    "PatchEmbed3D",
    "selective_scan",
    "selective_scan_metal",
    "selective_scan_metal_trainable",
    "selective_scan_metal_fused",
]
