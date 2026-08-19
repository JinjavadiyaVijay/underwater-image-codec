"""MobileNetV3-Small species classifier with pose/orientation head.

Distilled from BioCLIP-2 (frozen teacher). Includes confidence gating:
below threshold → fallback to genus/bbox-only mode.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeciesClassifier(nn.Module):
    """Lightweight species classifier for underwater organisms.

    Architecture: MobileNetV3-Small backbone with dual heads:
    1. Species classification head (num_species classes)
    2. Pose/orientation head (regression, 0-1 normalized angle)

    Designed to be distilled from BioCLIP-2 frozen teacher.
    """

    def __init__(
        self,
        num_species: int = 157,
        confidence_threshold: float = 0.6,
        input_size: int = 128,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_species = num_species
        self.confidence_threshold = confidence_threshold

        # MobileNetV3-Small backbone
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            feature_dim = 576  # MobileNetV3-Small output channels
        except Exception:
            # Fallback: simple CNN if torchvision models unavailable
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 256

        # Species classification head
        self.species_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_species + 1),  # +1 for "unknown" class (ID=0)
        )

        # Pose/orientation head
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # output [0, 1] representing normalized angle
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image, normalized to [0, 1].

        Returns:
            Dict with keys:
            - logits: (B, num_species+1) class logits
            - pose: (B, 1) pose prediction [0, 1]
            - features: (B, feature_dim) backbone features (for teacher loss)
        """
        feats = self.features(x)
        pooled = self.avgpool(feats).flatten(1)

        logits = self.species_head(pooled)
        pose = self.pose_head(pooled)

        return {
            "logits": logits,
            "pose": pose,
            "features": pooled,
        }

    def predict(
        self,
        x: torch.Tensor,
        top_k: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Predict with confidence gating.

        Args:
            x: (B, 3, H, W) input image.
            top_k: Number of top predictions to return.

        Returns:
            Dict with species_id, confidence, top_k_ids, top_k_probs, pose.
        """
        out = self.forward(x)
        probs = F.softmax(out["logits"], dim=1)

        top_probs, top_ids = probs.topk(top_k, dim=1)

        species_id = top_ids[:, 0]
        confidence = top_probs[:, 0]

        # Confidence gating: below threshold → fallback to unknown (ID=0)
        below_threshold = confidence < self.confidence_threshold
        species_id = torch.where(below_threshold, torch.zeros_like(species_id), species_id)

        return {
            "species_id": species_id,
            "confidence": confidence,
            "top_k_ids": top_ids,
            "top_k_probs": top_probs,
            "pose": out["pose"],
            "features": out["features"],
        }

    def classify_numpy(
        self,
        image: "np.ndarray",
        top_k: int = 5,
    ) -> tuple[list[int], list[float]]:
        """Classify a numpy image (for use in bio_metrics).

        Args:
            image: RGB uint8 (H, W, 3).
            top_k: Number of top predictions.

        Returns:
            (top_k_ids, top_k_probs) as Python lists.
        """
        import numpy as np

        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        with torch.no_grad():
            result = self.predict(x, top_k=top_k)

        return (
            result["top_k_ids"][0].cpu().tolist(),
            result["top_k_probs"][0].cpu().tolist(),
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "SpeciesClassifier":
        """Load a trained classifier from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint.get("config", kwargs))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save(self, path: str | Path, config: dict | None = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config or {"num_species": self.num_species},
        }, path)
