"""Rate controller: token prioritization and adaptive rate control.

Manages how VQ tokens are allocated within the byte budget.
Uses learned importance scores — NOT arbitrary truncation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from uwcodec.core.config import PayloadConfig


class RateController(nn.Module):
    """Learned rate controller for adaptive token allocation.

    Assigns importance scores to each VQ token position. When the byte budget
    is tight, low-importance tokens are dropped (set to zero/default).

    This ensures that prefix truncation degrades gracefully rather than
    arbitrarily cutting the most recently encoded tokens.
    """

    def __init__(
        self,
        num_tokens: int = 16,  # total spatial positions (e.g., 4×4 from encoder)
        num_groups: int = 4,   # product quantization groups
        latent_dim: int = 64,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_groups = num_groups

        # Importance scorer: maps latent features to per-token importance
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Learned position-level prior importance (positional bias)
        self.position_bias = nn.Parameter(torch.zeros(num_tokens))

    def compute_importance(self, z: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for each spatial token.

        Args:
            z: (B, D, H, W) latent features.

        Returns:
            (B, H*W) importance scores in [0, 1].
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N, D)
        scores = self.scorer(z_flat).squeeze(-1)  # (B, N)

        # Add positional bias
        N = scores.shape[1]
        bias = self.position_bias[:N]
        scores = scores + bias.unsqueeze(0)

        return torch.sigmoid(scores)

    def select_tokens(
        self,
        indices: list[torch.Tensor],  # per-group VQ indices
        importance: torch.Tensor,      # (B, N) importance scores
        budget_tokens: int,            # max tokens to keep
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Select top-k tokens based on importance scores.

        Args:
            indices: List of (B, H, W) index tensors per VQ group.
            importance: (B, N) importance scores.
            budget_tokens: Maximum number of spatial positions to keep.

        Returns:
            masked_indices: Same shape as input, but low-importance tokens zeroed.
            mask: (B, N) binary mask of selected tokens.
        """
        B, N = importance.shape
        k = min(budget_tokens, N)

        # Top-k selection
        _, top_k_indices = importance.topk(k, dim=1)
        mask = torch.zeros(B, N, device=importance.device)
        mask.scatter_(1, top_k_indices, 1.0)

        # Apply mask to indices
        masked_indices = []
        for group_idx in indices:
            B_g, H, W = group_idx.shape
            flat = group_idx.reshape(B_g, -1)  # (B, N)
            flat = flat * mask.long()
            masked_indices.append(flat.reshape(B_g, H, W))

        return masked_indices, mask

    def compute_budget_tokens(
        self,
        payload_config: PayloadConfig,
        max_bytes: int,
        bytes_per_token: int = 1,
    ) -> int:
        """Compute how many VQ tokens fit in the residual byte budget.

        Args:
            payload_config: Payload configuration.
            max_bytes: Total byte budget (64, 96, or 124).
            bytes_per_token: Bytes per VQ token index (typically 1 for 256-entry codebook).

        Returns:
            Maximum number of token positions.
        """
        residual_bytes = payload_config.residual_bytes(max_bytes)
        # Each position needs bytes_per_token * num_groups bytes
        tokens = residual_bytes // (bytes_per_token * self.num_groups)
        return max(1, tokens)


def serialize_indices(
    indices: list[torch.Tensor],
    mask: torch.Tensor | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """Serialize VQ indices into bytes for the payload.

    Packs indices from most important to least important positions.

    Args:
        indices: List of (B, H, W) per-group index tensors. Only B=0 is used.
        mask: (B, N) importance mask. If None, all tokens are included.
        max_bytes: Maximum bytes to output.

    Returns:
        Serialized bytes (interleaved group indices for each spatial position).
    """
    # Take first batch element
    flat_groups = []
    for group_idx in indices:
        flat_groups.append(group_idx[0].flatten().cpu().numpy().astype(np.uint8))

    num_positions = flat_groups[0].shape[0]
    num_groups = len(flat_groups)

    # Determine ordering (by importance if mask provided)
    if mask is not None:
        importance = mask[0].cpu().numpy()
        order = np.argsort(-importance)  # descending
    else:
        order = np.arange(num_positions)

    # Interleave: for each position, emit all group indices
    buf = bytearray()
    for pos in order:
        for g in range(num_groups):
            buf.append(int(flat_groups[g][pos]) & 0xFF)
        if max_bytes is not None and len(buf) >= max_bytes:
            break

    if max_bytes is not None:
        buf = buf[:max_bytes]

    return bytes(buf)


def deserialize_indices(
    data: bytes,
    num_groups: int = 4,
    spatial_size: tuple[int, int] = (4, 4),
) -> list[torch.Tensor]:
    """Deserialize bytes back into VQ indices.

    Args:
        data: Raw bytes from payload.
        num_groups: Number of product quantization groups.
        spatial_size: (H, W) spatial dimensions of the feature map.

    Returns:
        List of (1, H, W) index tensors per group.
    """
    H, W = spatial_size
    total_positions = H * W

    # Parse interleaved indices
    group_indices = [np.zeros(total_positions, dtype=np.int64) for _ in range(num_groups)]

    pos = 0
    byte_idx = 0
    while byte_idx < len(data) and pos < total_positions:
        for g in range(num_groups):
            if byte_idx < len(data):
                group_indices[g][pos] = data[byte_idx]
                byte_idx += 1
        pos += 1

    result = []
    for g in range(num_groups):
        idx_tensor = torch.from_numpy(group_indices[g]).reshape(1, H, W)
        result.append(idx_tensor)

    return result
