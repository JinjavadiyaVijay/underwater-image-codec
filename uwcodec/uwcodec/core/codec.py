"""UWCodec: General-purpose underwater image codec.

Clean public API — no fish dependencies, no labels, no YOLO.

Usage:
    from uwcodec import UWCodec

    codec = UWCodec.load("path/to/model.pt")

    # Encode any RGB image
    payload = codec.encode(image, max_bytes=124)
    assert len(payload) <= 124  # always enforced

    # Decode on receiver
    reconstructed = codec.decode(payload)

    # Rate-distortion sweep
    for budget in [64, 96, 124, 256, 512, 1024, 2048, 4096]:
        p = codec.encode(image, max_bytes=budget)
        r = codec.decode(p)
        # measure PSNR/SSIM here
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uwcodec.core.config import UWCodecConfig, ModelConfig
from uwcodec.codecs.vqvae_codec import MinimalVQVAE


class UWCodec:
    """General-purpose underwater image codec.

    Compresses ANY RGB image to ≤64/96/124 bytes for BLE transmission.

    This is extreme semantic compression — NOT lossless. At 64-124 bytes,
    reconstruction quality is coarse. See evaluation results for honest numbers.

    The hard payload limit is ALWAYS enforced:
        assert len(codec.encode(image, max_bytes)) <= max_bytes
    """

    def __init__(self, model: MinimalVQVAE, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def encode(self, image: np.ndarray, max_bytes: int = 124) -> bytes:
        """Encode an RGB image to a byte payload.

        Args:
            image: (H, W, 3) uint8 RGB image (any resolution).
            max_bytes: Byte budget. Payload will be EXACTLY this many bytes.
                       Supported: 64, 96, 124, 256, 512, 1024, 2048, 4096, or any ≥3.

        Returns:
            bytes of exactly max_bytes length.

        Raises:
            AssertionError: If payload exceeds max_bytes (should never happen).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) uint8 ndarray, got shape {getattr(image, 'shape', '?')}")

        payload = self.model.encode(image, max_bytes)
        result = payload.raw_bytes

        # Double-check hard limit
        assert len(result) <= max_bytes, (
            f"HARD LIMIT VIOLATION: encode() produced {len(result)}B for budget {max_bytes}B"
        )
        return result

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode a byte payload to an RGB image.

        Args:
            payload: Raw bytes from encode() (any length).

        Returns:
            (output_size, output_size, 3) uint8 RGB image.
        """
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"payload must be bytes, got {type(payload)}")
        return self.model.decode(payload)

    def encode_decode(self, image: np.ndarray, max_bytes: int = 124) -> tuple[bytes, np.ndarray]:
        """Encode then decode in one call (for evaluation).

        Returns:
            (payload_bytes, reconstructed_image)
        """
        payload = self.encode(image, max_bytes)
        recon = self.decode(payload)
        return payload, recon

    def rate_distortion_sweep(
        self,
        image: np.ndarray,
        budgets: list[int] | None = None,
        metric_fn: callable | None = None,
    ) -> list[dict]:
        """Encode/decode at multiple budgets and measure quality.

        Args:
            image: Input RGB image.
            budgets: List of byte budgets to test.
            metric_fn: Optional callable(original, recon) → dict.
                       If None, returns payload sizes only.

        Returns:
            List of dicts with 'budget', 'bytes', and optional metric keys.
        """
        if budgets is None:
            budgets = [64, 96, 124, 256, 512, 1024, 2048, 4096]

        results = []
        for budget in budgets:
            payload, recon = self.encode_decode(image, max_bytes=budget)
            entry = {
                "budget": budget,
                "bytes": len(payload),
                "bpp_128": len(payload) * 8 / (128 * 128),
            }
            if metric_fn is not None:
                metrics = metric_fn(image, recon)
                entry.update(metrics)
            results.append(entry)
        return results

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "UWCodec":
        """Load a trained codec from a checkpoint.

        Args:
            path: Path to .pt checkpoint saved by MinimalVQVAE.save().
            device: "cuda", "cpu", or "auto".

        Returns:
            UWCodec ready for encode/decode.
        """
        model = MinimalVQVAE.load(path)
        return cls(model, device=device)

    @classmethod
    def from_config(cls, config: UWCodecConfig, device: str = "auto") -> "UWCodec":
        """Create a new (untrained) codec from config.

        Used for training. Load a checkpoint with UWCodec.load() for inference.
        """
        mc = config.model
        model = MinimalVQVAE(
            input_size=mc.input_size,
            encoder_channels=mc.encoder_channels,
            latent_dim=mc.encoder_latent_dim,
            codebook_size=mc.codebook_size,
            decoder_channels=mc.decoder_channels,
            output_size=mc.output_size,
        )
        return cls(model, device=device)

    def model_info(self) -> dict:
        """Return model architecture info."""
        params = self.model.count_parameters()
        return {
            "input_size": self.model.input_size,
            "output_size": self.model.output_size,
            "codebook_size": self.model.codebook_size,
            "spatial_grid": f"{self.model.spatial_h}×{self.model.spatial_w}",
            "num_spatial_positions": self.model.num_spatial_positions,
            "parameters": params,
            "device": self.device,
            "budget_info": {
                b: f"{self.model._payload_format.config.vq_bytes(b)}B VQ / {b}B total"
                for b in [64, 96, 124, 256, 512]
            },
        }
