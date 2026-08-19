"""Vector Quantization: VQ, Product-VQ, Residual VQ.

Quantizes continuous latent features into discrete indices for ultra-compact
byte-stream encoding. General-purpose — works on any image domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Standard Vector Quantization with learned codebook.

    Uses EMA codebook update (no codebook loss gradient) for stable training.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        codebook_dim: int = 64,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("_ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("_ema_w", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Quantize latent features.

        Args:
            z: (B, D, H, W) continuous latent features.

        Returns:
            z_q: (B, D, H, W) quantized features (straight-through).
            info: Dict with indices, commitment_loss, codebook_loss, perplexity.
        """
        B, D, H, W = z.shape
        # (B, H, W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Find nearest codebook entries
        dists = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dists.argmin(dim=1)  # (B*H*W,)

        # Quantize
        z_q_flat = self.embedding(indices)  # (B*H*W, D)

        # EMA codebook update (training only)
        if self.training:
            encodings = F.one_hot(indices, self.codebook_size).float()
            self._ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            dw = encodings.t() @ z_flat
            self._ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )
            self.embedding.weight.data.copy_(self._ema_w / cluster_size.unsqueeze(1))

        # Losses
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Perplexity (codebook utilization)
        avg_probs = F.one_hot(indices, self.codebook_size).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        info = {
            "indices": indices.reshape(B, H, W),
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "vq_loss": codebook_loss + self.commitment_weight * commitment_loss,
        }

        return z_q, info

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries by index.

        Args:
            indices: (B, H, W) or (N,) integer indices.

        Returns:
            Codebook vectors corresponding to each index.
        """
        return self.embedding(indices)


class ProductQuantizer(nn.Module):
    """Product Quantization: split latent into sub-vectors, quantize each independently.

    Reduces codebook lookup cost and allows finer control over bits per sub-vector.
    """

    def __init__(
        self,
        codebook_size: int = 256,
        codebook_dim: int = 64,
        num_groups: int = 4,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        assert codebook_dim % num_groups == 0, \
            f"codebook_dim ({codebook_dim}) must be divisible by num_groups ({num_groups})"

        self.num_groups = num_groups
        self.sub_dim = codebook_dim // num_groups

        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, self.sub_dim, commitment_weight)
            for _ in range(num_groups)
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Quantize using product quantization.

        Args:
            z: (B, D, H, W) continuous latent features.

        Returns:
            z_q: (B, D, H, W) quantized features.
            info: Dict with per-group indices and aggregate losses.
        """
        B, D, H, W = z.shape
        sub_z = z.chunk(self.num_groups, dim=1)

        z_q_parts = []
        all_indices = []
        total_vq_loss = 0.0
        total_perplexity = 0.0

        for i, (sub, vq) in enumerate(zip(sub_z, self.quantizers)):
            z_q_sub, info = vq(sub)
            z_q_parts.append(z_q_sub)
            all_indices.append(info["indices"])
            total_vq_loss = total_vq_loss + info["vq_loss"]
            total_perplexity = total_perplexity + info["perplexity"]

        z_q = torch.cat(z_q_parts, dim=1)

        info = {
            "indices": all_indices,  # list of (B, H, W) per group
            "vq_loss": total_vq_loss / self.num_groups,
            "perplexity": total_perplexity / self.num_groups,
        }

        return z_q, info

    def flatten_indices(self, indices_list: list[torch.Tensor]) -> torch.Tensor:
        """Flatten product indices into a single index per spatial position.

        For serialization into the byte payload.
        """
        # Each group has codebook_size options → interleave indices
        # Returns (B, num_groups, H, W)
        return torch.stack(indices_list, dim=1)


class ResidualVQ(nn.Module):
    """Residual Vector Quantization: multi-level refinement.

    Each level quantizes the residual from the previous level.
    Enables progressive quality: early truncation gives coarse result,
    later tokens refine detail. This is key for rate adaptation.
    """

    def __init__(
        self,
        codebook_size: int = 256,
        codebook_dim: int = 64,
        num_levels: int = 2,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, codebook_dim, commitment_weight)
            for _ in range(num_levels)
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Multi-level residual quantization.

        Args:
            z: (B, D, H, W) continuous latent features.

        Returns:
            z_q: (B, D, H, W) sum of all quantized residuals.
            info: Dict with per-level indices and losses.
        """
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        total_vq_loss = 0.0
        total_perplexity = 0.0

        for level, vq in enumerate(self.quantizers):
            z_q_level, info = vq(residual)
            z_q_sum = z_q_sum + z_q_level
            residual = residual - z_q_level.detach()  # detach to prevent gradient through residual
            all_indices.append(info["indices"])
            total_vq_loss = total_vq_loss + info["vq_loss"]
            total_perplexity = total_perplexity + info["perplexity"]

        info = {
            "indices": all_indices,  # list of (B, H, W) per level
            "vq_loss": total_vq_loss / self.num_levels,
            "perplexity": total_perplexity / self.num_levels,
        }

        return z_q_sum, info

    def decode_partial(self, indices_list: list[torch.Tensor], num_levels: int | None = None) -> torch.Tensor:
        """Decode from partial indices (for rate control / prefix truncation).

        Args:
            indices_list: List of index tensors per level.
            num_levels: Number of levels to use (None = all).

        Returns:
            Partial reconstruction from the first num_levels levels.
        """
        if num_levels is None:
            num_levels = len(indices_list)

        z_q_sum = None
        for level in range(min(num_levels, len(indices_list))):
            codes = self.quantizers[level].indices_to_codes(indices_list[level])
            # Rearrange if needed
            if codes.dim() == 3:  # (B, H*W, D)
                B, N, D = codes.shape
                H = W = int(N ** 0.5)
                codes = codes.reshape(B, H, W, D).permute(0, 3, 1, 2)
            if z_q_sum is None:
                z_q_sum = codes
            else:
                z_q_sum = z_q_sum + codes

        return z_q_sum
