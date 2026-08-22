"""UWCodec V3 Decoder — Transformer-based decoder.

Decodes 64 1D latent tokens back to a 128×128 RGB image.
Follows the TiTok architecture (ByteDance).

Architecture:
    64 latent tokens (from VQ)
    → Concatenate with 64 learned mask/patch tokens
    → Add positional embeddings
    → Transformer decoder (6 blocks)
    → Extract 64 patch tokens
    → Reshape to 8×8 spatial grid
    → Convolutional pixel decoder (upsample 8×8 → 128×128)
    → RGB output
"""

from __future__ import annotations

import torch
import torch.nn as nn

from uwcodec.models.v3_encoder import TransformerBlock


class PixelDecoder(nn.Module):
    """Upsample 8×8 sequence of patches to 128×128 RGB.
    
    A lightweight CNN that refines the Transformer patch outputs.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        # Input: (B, 256, 8, 8)
        self.net = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            # Output
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 8, 8) → (B, 3, 128, 128)"""
        return self.net(x)


class V3Decoder(nn.Module):
    """Transformer decoder mapping 1D tokens to 2D image.

    Args:
        embed_dim: Token dimension.
        num_latent_tokens: Number of 1D latents (64).
        input_size: Target image size (128).
        patch_size: Resolution of the transformer patch grid (16).
        depth: Number of transformer blocks (6).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_latent_tokens: int = 64,
        input_size: int = 128,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latent_tokens = num_latent_tokens
        self.input_size = input_size
        self.patch_size = patch_size

        self.num_patches = (input_size // patch_size) ** 2  # 64 for 128/16
        self.grid_size = input_size // patch_size           # 8 for 128/16

        # Learned mask tokens to represent the target image patches
        self.mask_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Positional embeddings
        total_tokens = self.num_patches + num_latent_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Lightweight CNN to go from 8x8 -> 128x128
        self.pixel_decoder = PixelDecoder(in_channels=embed_dim, out_channels=3)

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens to image.

        Args:
            latent_tokens: (B, num_latent_tokens, D) quantized features.

        Returns:
            (B, 3, 128, 128) reconstructed image (unconstrained float, usually sigmoided by codec)
        """
        B = latent_tokens.shape[0]

        # Expand mask tokens for batch
        mask_tokens = self.mask_tokens.expand(B, -1, -1)  # (B, 64, D)

        # Concatenate: [mask_tokens, latent_tokens] → (B, 128, D)
        # Note: TiTok usually puts mask tokens first or second; we put mask first.
        tokens = torch.cat([mask_tokens, latent_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Extract the mask tokens which now hold the patch information
        patch_tokens = tokens[:, :self.num_patches, :]  # (B, 64, D)

        # Reshape to spatial grid: (B, 64, D) -> (B, D, 8, 8)
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        # Decode to pixels: (B, D, 8, 8) -> (B, 3, 128, 128)
        img = self.pixel_decoder(spatial_features)

        return img

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
