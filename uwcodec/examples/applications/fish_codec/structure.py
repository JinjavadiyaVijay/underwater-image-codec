"""Structure encoder: silhouette/edge/shape representation.

Encodes the shape/structure of the detected fish/lobster into a compact
byte representation. Operates alongside the appearance encoder to ensure
the decoder preserves body outline, fin positions, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureEncoder(nn.Module):
    """Encode structure information (silhouette/edges) into compact latent.

    Dual-purpose:
    1. Produces a compact binary structure mask for payload (explicit channel).
    2. Produces a latent structure embedding to condition the decoder.

    Input: RGB image (B, 3, H, W)
    Outputs:
        - structure_logits: (B, 1, grid_h, grid_w) binary mask predictions
        - structure_embedding: (B, embed_dim) for decoder conditioning
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        grid_size: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Simple CNN to extract structure features
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Structure mask prediction (binary silhouette at grid_size × grid_size)
        self.mask_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Conv2d(hidden_dim, 1, 1),
        )

        # Structure embedding (global shape descriptor for decoder conditioning)
        self.embed_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode structure.

        Args:
            x: (B, 3, H, W) input image.

        Returns:
            structure_logits: (B, 1, grid_size, grid_size) mask logits.
            structure_embedding: (B, embed_dim) global shape descriptor.
        """
        feats = self.backbone(x)
        mask_logits = self.mask_head(feats)
        embedding = self.embed_head(feats)
        return mask_logits, embedding

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Predict binary structure mask.

        Args:
            x: (B, 3, H, W) input image.

        Returns:
            (B, grid_size, grid_size) binary mask (0 or 1).
        """
        logits, _ = self.forward(x)
        return (logits.squeeze(1) > 0).float()

    def mask_to_bytes(self, mask: torch.Tensor) -> list[bytes]:
        """Convert predicted mask to byte representation for payload.

        Args:
            mask: (B, grid_size, grid_size) binary mask.

        Returns:
            List of bytes (one per batch item), each grid_size*grid_size/8 bytes.
        """
        B = mask.shape[0]
        result = []
        for b in range(B):
            bits = mask[b].flatten().bool().tolist()
            num_bytes = (len(bits) + 7) // 8
            packed = bytearray(num_bytes)
            for i, bit in enumerate(bits):
                if bit:
                    packed[i // 8] |= 1 << (7 - (i % 8))
            result.append(bytes(packed))
        return result
