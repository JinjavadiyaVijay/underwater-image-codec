"""Multi-term loss functions for codec training.

L = λ_rate × bitrate_penalty
  + λ_pixel × L1(recon, target)
  + λ_perceptual × LPIPS(recon, target)
  + λ_structure × structure_loss(recon, target)
  + λ_teacher × BioCLIP_feature_distance
  + λ_color × color_consistency_loss

Species classification is SEPARATE and FROZEN — NOT in this loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from uwcodec.core.config import TrainingConfig


class PixelLoss(nn.Module):
    """Combined L1 + L2 pixel reconstruction loss."""

    def __init__(self, l1_weight: float = 0.8, l2_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(recon, target)
        l2 = F.mse_loss(recon, target)
        return self.l1_weight * l1 + self.l2_weight * l2


class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss.

    Uses pre-trained VGG features to measure perceptual similarity.
    Falls back to L1 if lpips is not installed.
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        self._net_name = net
        self._lpips_model = None
        self._available = None

    def _init_lpips(self, device: torch.device) -> bool:
        if self._available is not None:
            return self._available
        try:
            import lpips
            self._lpips_model = lpips.LPIPS(net=self._net_name, verbose=False).to(device)
            self._lpips_model.eval()
            for p in self._lpips_model.parameters():
                p.requires_grad = False
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._init_lpips(recon.device):
            # LPIPS expects [-1, 1] range
            recon_scaled = recon * 2 - 1
            target_scaled = target * 2 - 1
            return self._lpips_model(recon_scaled, target_scaled).mean()
        else:
            # Fallback to L1
            return F.l1_loss(recon, target)


class StructureLoss(nn.Module):
    """Structure preservation loss: edge + silhouette consistency.

    Encourages the decoder to preserve the subject's body outline,
    fin positions, and overall shape.
    """

    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        structure_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute structure loss.

        Args:
            recon: (B, 3, H, W) reconstructed image.
            target: (B, 3, H, W) target image.
            structure_mask: (B, 1, H, W) optional structure mask for weighted loss.
        """
        # Convert to grayscale
        recon_gray = recon.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        # Compute edges
        recon_edges = self._sobel(recon_gray)
        target_edges = self._sobel(target_gray)

        # Edge matching loss
        edge_loss = F.l1_loss(recon_edges, target_edges)

        # Masked pixel loss (focus on foreground structure)
        if structure_mask is not None:
            # Resize mask to match image size
            if structure_mask.shape[2:] != recon.shape[2:]:
                structure_mask = F.interpolate(
                    structure_mask, size=recon.shape[2:], mode="nearest"
                )
            masked_loss = (F.l1_loss(recon, target, reduction="none") * structure_mask).mean()
            return edge_loss + masked_loss

        return edge_loss

    def _sobel(self, gray: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class TeacherLoss(nn.Module):
    """BioCLIP-2 feature distillation loss.

    BioCLIP-2 is used ONLY as a frozen training/evaluation teacher.
    Encourages the reconstruction to preserve biologically-relevant features.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        recon_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine distance between teacher features.

        Args:
            recon_features: (B, D) BioCLIP features of reconstruction.
            target_features: (B, D) BioCLIP features of target.
        """
        recon_norm = F.normalize(recon_features, dim=1)
        target_norm = F.normalize(target_features, dim=1)
        return 1.0 - (recon_norm * target_norm).sum(dim=1).mean()


class ColorConsistencyLoss(nn.Module):
    """Color distribution consistency loss.

    Encourages the reconstruction to have a similar color histogram
    to the original, preserving the characteristic coloration.
    """

    def __init__(self, num_bins: int = 16):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute color consistency via channel-wise mean/std matching.

        Uses differentiable statistics rather than histograms.
        """
        # Channel-wise mean matching
        mean_loss = F.l1_loss(recon.mean(dim=(2, 3)), target.mean(dim=(2, 3)))

        # Channel-wise std matching
        std_loss = F.l1_loss(recon.std(dim=(2, 3)), target.std(dim=(2, 3)))

        return mean_loss + std_loss


class RatePenalty(nn.Module):
    """Bitrate penalty to encourage compact representations.

    Penalizes non-zero VQ tokens when the importance mask drops them.
    """

    def forward(
        self,
        importance: torch.Tensor,
        budget_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Compute rate penalty.

        Args:
            importance: (B, N) importance scores in [0, 1].
            budget_ratio: Target fraction of tokens to keep (0-1).
        """
        # Encourage importance distribution to match budget
        mean_importance = importance.mean(dim=1)
        target = torch.full_like(mean_importance, budget_ratio)
        return F.mse_loss(mean_importance, target)


@dataclass
class LossComponents:
    """Container for individual loss terms."""
    total: torch.Tensor
    pixel: torch.Tensor
    perceptual: torch.Tensor
    structure: torch.Tensor
    teacher: torch.Tensor
    color: torch.Tensor
    vq: torch.Tensor
    rate: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total.item(),
            "pixel": self.pixel.item(),
            "perceptual": self.perceptual.item(),
            "structure": self.structure.item(),
            "teacher": self.teacher.item(),
            "color": self.color.item(),
            "vq": self.vq.item(),
            "rate": self.rate.item(),
        }


class CodecLoss(nn.Module):
    """Combined multi-term codec training loss."""

    def __init__(self, config: TrainingConfig | None = None):
        super().__init__()
        if config is None:
            config = TrainingConfig()

        self.config = config
        self.pixel_loss = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.structure_loss = StructureLoss()
        self.teacher_loss = TeacherLoss()
        self.color_loss = ColorConsistencyLoss()
        self.rate_penalty = RatePenalty()

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        vq_loss: torch.Tensor | None = None,
        importance: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        recon_teacher_features: torch.Tensor | None = None,
        target_teacher_features: torch.Tensor | None = None,
        budget_ratio: float = 1.0,
    ) -> LossComponents:
        """Compute combined loss.

        Args:
            recon: (B, 3, H, W) reconstructed image [0, 1].
            target: (B, 3, H, W) target image [0, 1].
            vq_loss: VQ commitment + codebook loss.
            importance: (B, N) token importance scores.
            structure_mask: (B, 1, H, W) structure mask.
            recon_teacher_features: BioCLIP features of recon.
            target_teacher_features: BioCLIP features of target.
            budget_ratio: Target token keep ratio.
        """
        cfg = self.config
        device = recon.device
        zero = torch.tensor(0.0, device=device)

        pixel = self.pixel_loss(recon, target)
        perceptual = self.perceptual_loss(recon, target)
        structure = self.structure_loss(recon, target, structure_mask)
        color = self.color_loss(recon, target)

        teacher = zero
        if recon_teacher_features is not None and target_teacher_features is not None:
            teacher = self.teacher_loss(recon_teacher_features, target_teacher_features)

        vq = vq_loss if vq_loss is not None else zero

        rate = zero
        if importance is not None:
            rate = self.rate_penalty(importance, budget_ratio)

        total = (
            cfg.lambda_pixel * pixel
            + cfg.lambda_perceptual * perceptual
            + cfg.lambda_structure * structure
            + cfg.lambda_teacher * teacher
            + cfg.lambda_color * color
            + vq
            + cfg.lambda_rate * rate
        )

        return LossComponents(
            total=total,
            pixel=pixel,
            perceptual=perceptual,
            structure=structure,
            teacher=teacher,
            color=color,
            vq=vq,
            rate=rate,
        )
