"""
ViT-Small encoder for I-JEPA in MLX.

Architecture: patch_size=16, dim=384, depth=6, heads=6
Input: 224x224 grayscale -> 14x14 = 196 patch tokens (no CLS token)
"""

import math
import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    """Standard transformer MLP: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHA -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dims=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class PatchEmbed(nn.Module):
    """Convert 2D image to patch embeddings using Conv2d.

    MLX Conv2d expects NHWC input and produces NHWC output.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 384,
                 patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) in MLX channel-last format
        x = self.proj(x)  # (B, H/P, W/P, embed_dim)
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1])
        return x


class PatchEmbed3D(nn.Module):
    """Convert 3D volume to patch embeddings using Conv3d.

    Input: (B, D, H, W, C) — MLX channel-last.
    Output: (B, num_patches, embed_dim)

    MLX Conv3d expects (B, D, H, W, C) and produces (B, D/P, H/P, W/P, embed_dim).
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 384,
                 patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, D, H, W, C)
        x = self.proj(x)  # (B, D/P, H/P, W/P, embed_dim)
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1])
        return x


class VisionTransformer(nn.Module):
    """
    ViT-Small for I-JEPA.

    No CLS token (I-JEPA operates on patch tokens only).
    Supports processing a subset of patches (for context encoder efficiency).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 196

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Learned positional embeddings for all 196 positions.
        # Initialize with small random values (truncated normal style).
        scale = 1.0 / math.sqrt(embed_dim)
        self.pos_embed = mx.random.normal(
            shape=(1, self.num_patches, embed_dim)
        ) * scale

        self.blocks = [
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array, patch_indices: mx.array = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: (B, H, W, C) image tensor, OR (B, N, embed_dim)
               if patches are already embedded.
            patch_indices: optional (N,) int array of patch indices to process.
                          If provided, only those patches get their positional
                          embeddings. Used by context encoder.
        Returns:
            (B, num_tokens, embed_dim) patch representations.
        """
        if x.ndim == 4:
            x = self.patch_embed(x)  # (B, 196, 384)

        if patch_indices is not None:
            pos = self.pos_embed[:, patch_indices, :]  # (1, N, D)
            x = x + pos
        else:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def embed_patches(self, images: mx.array) -> mx.array:
        """Just do patch embedding without positional encoding or transformer."""
        return self.patch_embed(images)  # (B, 196, 384)
