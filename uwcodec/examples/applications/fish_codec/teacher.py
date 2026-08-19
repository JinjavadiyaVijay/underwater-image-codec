"""BioCLIP-2 frozen teacher wrapper.

BioCLIP-2 is used ONLY as a frozen training/evaluation teacher.
NEVER deployed on the target device. Provides:
1. Feature vectors for distillation loss
2. Classification targets for soft-label training
3. Evaluation metric (feature similarity between original and reconstruction)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BioCLIPTeacher:
    """Frozen BioCLIP-2 teacher for feature extraction and distillation.

    This class wraps the BioCLIP-2 model (or open_clip equivalent) and
    provides a simple API for:
    - Extracting feature vectors from images
    - Computing soft classification targets
    - Computing similarity between two images in feature space

    CRITICAL: This model is NEVER part of the deployed codec.
    It is used only during training and evaluation.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:imageomics/bioclip",
        device: str = "auto",
        weights_path: str | Path | None = None,
    ):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._model_name = model_name
        self._weights_path = weights_path
        self._available = None

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load(self) -> bool:
        """Lazy-load the BioCLIP model."""
        if self._available is not None:
            return self._available

        try:
            import open_clip

            if self._weights_path:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained=str(self._weights_path)
                )
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self._model_name
                )

            model = model.to(self._device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self._model_name)
            self._available = True
            print(f"Loaded BioCLIP teacher: {self._model_name}")
        except (ImportError, Exception) as e:
            print(f"WARNING: BioCLIP teacher not available: {e}")
            print("Teacher distillation and BioCLIP similarity metrics will be skipped.")
            self._available = False

        return self._available

    @property
    def is_available(self) -> bool:
        return self._load()

    def extract_features(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract feature vector from an image.

        Args:
            image: RGB image as uint8 numpy array (H, W, 3) or
                   torch tensor (B, 3, H, W) in [0, 1].

        Returns:
            (D,) or (B, D) normalized feature vector.
        """
        if not self.is_available:
            return torch.zeros(512)

        if isinstance(image, np.ndarray):
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        else:
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = F.normalize(features, dim=-1)

        if features.shape[0] == 1:
            return features.squeeze(0)
        return features

    def compute_similarity(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two images in BioCLIP feature space.

        Args:
            image1: First RGB image (H, W, 3) uint8.
            image2: Second RGB image (H, W, 3) uint8.

        Returns:
            Cosine similarity in [-1, 1]. Higher is better.
        """
        if not self.is_available:
            return 0.0

        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        return float(torch.dot(feat1, feat2).item())

    def extract_features_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for a batch of images.

        Args:
            images: (B, 3, H, W) tensor in [0, 1].

        Returns:
            (B, D) normalized feature vectors.
        """
        if not self.is_available:
            return torch.zeros(images.shape[0], 512, device=images.device)

        images = images.to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(images)
            features = F.normalize(features, dim=-1)
        return features

    def get_feature_fn(self):
        """Return a callable for use with bio_metrics.compute_bioclip_similarity.

        Returns:
            Function that takes numpy image → numpy feature vector, or None.
        """
        if not self.is_available:
            return None

        def _fn(image: np.ndarray) -> np.ndarray:
            feat = self.extract_features(image)
            return feat.cpu().numpy()

        return _fn
