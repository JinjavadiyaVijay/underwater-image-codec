"""UWCodec V3 Encoder — ViT-based 1D tokenizer encoder.

Encodes 128×128 RGB images into 64 learned 1D latent tokens.
Inspired by the TiTok architecture (ByteDance).

Architecture:
    128×128 RGB
    → Patch embedding (16×16 patches → 64 patch tokens, dim=256)
    → Concatenate with 64 learned latent tokens
    → Transformer encoder (6 blocks)
    → Extract latent tokens (discard patch tokens)
    → 64 × latent_dim output

Design constraints:
    - Must fit in RTX 3050 6GB during training
    - Latent tokens carry all information for reconstruction
    - Patch tokens are auxiliary (discarded after encoding)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings.
    
    128×128 image with 16×16 patches → 8×8 = 64 patch tokens.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, embed_dim)."""
        x = self.proj(x)           # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with multi-head self-attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class V3Encoder(nn.Module):
    """ViT-based encoder that produces 1D latent tokens.

    Architecture:
        1. Patch embedding: image → 64 patch tokens
        2. Concatenate: 64 patch tokens + 64 learned latent tokens → 128 tokens
        3. Add positional embeddings
        4. Transformer blocks (self-attention across all 128 tokens)
        5. Extract the 64 latent tokens (discard patch tokens)

    The latent tokens learn to query and aggregate information from the
    patch tokens through self-attention, producing a compact 1D representation.
    """

    def __init__(
        self,
        input_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_latent_tokens: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_latent_tokens = num_latent_tokens

        num_patches = (input_size // patch_size) ** 2  # 64 for 128/16
        self.num_patches = num_patches

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # Learned latent tokens (the core 1D representation)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim) * 0.02)

        # Positional embeddings for all tokens (patches + latents)
        total_tokens = num_patches + num_latent_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent tokens.

        Args:
            x: (B, 3, H, W) float [0, 1]

        Returns:
            (B, num_latent_tokens, embed_dim) latent token features
        """
        B = x.shape[0]

        # Patch embedding → (B, 64, D)
        patch_tokens = self.patch_embed(x)

        # Expand latent tokens for batch → (B, 64, D)
        latent_tokens = self.latent_tokens.expand(B, -1, -1)

        # Concatenate: [patch_tokens, latent_tokens] → (B, 128, D)
        tokens = torch.cat([patch_tokens, latent_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Extract latent tokens (last num_latent_tokens)
        latent_out = tokens[:, self.num_patches:, :]  # (B, 64, D)

        return latent_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
