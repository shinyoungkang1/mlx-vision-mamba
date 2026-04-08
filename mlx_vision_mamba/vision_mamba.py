"""
Multi-dimensional Vision Mamba encoder for JEPA.

Drop-in replacement for ViT — same interface:
  Input: (B, H, W, C) image  OR  (B, N, D) pre-embedded patches
  Output: (B, N, D) patch representations

Supports 2D, 3D, and 4D inputs:
  2D: (B, H, W, C) images — K=4 multi-directional scanning
  3D: (B, D, H, W, C) volumes — K=6 multi-directional scanning
  4D: (B, T, D, H, W, C) spatiotemporal — K=8 multi-directional scanning

Uses MultiDirectionalMambaBlock with VMamba-style scan ordering.
Supports processing a subset of patches (for JEPA context encoder efficiency).
"""

import math
import mlx.core as mx
import mlx.nn as nn

from mlx_vision_mamba.mamba_block import BidirectionalMambaBlock, MultiDirectionalMambaBlock
from mlx_vision_mamba.vit import PatchEmbed, PatchEmbed3D


class PatchEmbed4D(nn.Module):
    """
    Convert 4D spatiotemporal data to patch embeddings.

    Since MLX does not have Conv4d, we use a workaround:
      1. Reshape (B, T, D, H, W, C) to (B*T, D, H, W, C)
      2. Apply Conv3d to get spatial patch embeddings
      3. Reshape back and flatten all spatiotemporal dims

    The temporal dimension is handled by grouping every temporal_patch_size
    frames together. If temporal_patch_size > 1, we reshape T into
    (T // tp, tp) and fold tp into the channel dim before the Conv3d.

    Args:
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        patch_size: Spatial patch size (applied to D, H, W).
        temporal_patch_size: Temporal patch size (default 1 = no temporal
                            downsampling, each frame becomes its own set
                            of spatial patches).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 384,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        # Conv3d input channels: in_channels * temporal_patch_size
        # (temporal frames are folded into channels)
        conv_in_channels = in_channels * temporal_patch_size
        self.proj = nn.Conv3d(
            conv_in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, D, H, W, C) — 4D spatiotemporal volume, channel-last.
        Returns:
            (B, num_patches, embed_dim) — flattened patch embeddings.
        """
        B, T, D, H, W, C = x.shape
        tp = self.temporal_patch_size

        if tp > 1:
            # Group temporal frames: (B, T//tp, tp, D, H, W, C)
            T_out = T // tp
            x = x.reshape(B, T_out, tp, D, H, W, C)
            # Fold tp into channel dim: (B, T_out, D, H, W, tp*C)
            x = mx.transpose(x, axes=(0, 1, 3, 4, 5, 2, 6))  # (B, T_out, D, H, W, tp, C)
            x = x.reshape(B, T_out, D, H, W, tp * C)
        else:
            T_out = T

        # Reshape to (B*T_out, D, H, W, tp*C) for Conv3d
        x = x.reshape(B * T_out, D, H, W, -1)

        # Apply Conv3d: (B*T_out, D/p, H/p, W/p, embed_dim)
        x = self.proj(x)
        _, Dp, Hp, Wp, E = x.shape

        # Reshape back: (B, T_out * Dp * Hp * Wp, embed_dim)
        x = x.reshape(B, T_out * Dp * Hp * Wp, E)
        return x


class VisionMamba(nn.Module):
    """
    Multi-dimensional Vision Mamba encoder.

    Architecture:
        PatchEmbed -> [MultiDirectionalMambaBlock x depth] -> LayerNorm

    Supports 2D, 3D, and 4D input via the input_dim parameter:
        input_dim=2: 2D images (B, H, W, C) — K=4 directions
        input_dim=3: 3D volumes (B, D, H, W, C) — K=6 directions
        input_dim=4: 4D spatiotemporal (B, T, D, H, W, C) — K=8 directions

    No CLS token (I-JEPA operates on patch tokens only).
    Supports processing a subset of patches via patch_indices.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        input_dim: int = 2,
        temporal_patch_size: int = 1,
        num_temporal_frames: int = 1,
    ):
        """
        Args:
            img_size: Spatial dimension size (assumed square/cubic).
            patch_size: Spatial patch size.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            depth: Number of Mamba blocks.
            d_state: SSM state dimension.
            d_conv: Convolution kernel size.
            expand_factor: Inner dimension expansion factor.
            input_dim: Spatial dimensionality (2, 3, or 4).
            temporal_patch_size: Temporal patch size for 4D input (default 1).
            num_temporal_frames: Number of temporal frames T for 4D input.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.img_size = img_size
        self.patch_size = patch_size

        # Compute grid shape and num_patches based on dimensionality
        spatial_grid = img_size // patch_size  # patches per spatial dim

        if input_dim == 2:
            self.grid_shape = (spatial_grid, spatial_grid)
            self.num_patches = spatial_grid ** 2
            num_directions = 4
            self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        elif input_dim == 3:
            self.grid_shape = (spatial_grid, spatial_grid, spatial_grid)
            self.num_patches = spatial_grid ** 3
            num_directions = 6
            self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)

        elif input_dim == 4:
            temporal_grid = num_temporal_frames // temporal_patch_size
            self.grid_shape = (
                temporal_grid, spatial_grid, spatial_grid, spatial_grid
            )
            self.num_patches = temporal_grid * (spatial_grid ** 3)
            num_directions = 8
            self.patch_embed = PatchEmbed4D(
                in_channels, embed_dim, patch_size, temporal_patch_size
            )
        else:
            raise ValueError(
                f"input_dim must be 2, 3, or 4, got {input_dim}"
            )

        # Learned positional embeddings
        scale = 1.0 / math.sqrt(embed_dim)
        self.pos_embed = mx.random.normal(
            shape=(1, self.num_patches, embed_dim)
        ) * scale

        # Multi-directional Mamba blocks
        self.blocks = [
            MultiDirectionalMambaBlock(
                embed_dim, d_state, d_conv, expand_factor,
                num_directions=num_directions,
                grid_shape=self.grid_shape,
            )
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array, patch_indices: mx.array = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Image/volume tensor or pre-embedded patches.
               2D: (B, H, W, C) or (B, N, embed_dim)
               3D: (B, D, H, W, C) or (B, N, embed_dim)
               4D: (B, T, D, H, W, C) or (B, N, embed_dim)
            patch_indices: optional (N,) int array of patch indices to process.
                          If provided, only those patches get positional
                          embeddings. Used by context encoder.
        Returns:
            (B, num_tokens, embed_dim) patch representations.
        """
        # Patch embed if not already embedded
        # Pre-embedded patches always come as (B, N, D) i.e. ndim == 3
        if x.ndim != 3:
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        if patch_indices is not None:
            pos = self.pos_embed[:, patch_indices, :]
            x = x + pos
        else:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def embed_patches(self, images: mx.array) -> mx.array:
        """Just do patch embedding without positional encoding or backbone."""
        return self.patch_embed(images)
