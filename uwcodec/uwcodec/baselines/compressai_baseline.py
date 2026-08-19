"""CompressAI baseline: neural image compression at 256B/512B/1KB+.

Shows where traditional learned compression becomes viable.
At 64-124B, these models also produce poor results — our domain-specific
approach is needed precisely because general-purpose codecs fail here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressAIResult:
    """Result from CompressAI baseline."""
    image: np.ndarray
    actual_bytes: int
    model_name: str
    quality: int
    bpp: float


def compressai_encode_decode(
    image: np.ndarray,
    model_name: str = "bmshj2018-hyperprior",
    quality: int = 1,
) -> CompressAIResult:
    """Encode/decode with CompressAI neural codec.

    Requires the compressai package.
    """
    try:
        import torch
        import compressai
        from compressai.zoo import models

        # Load model
        model = models[model_name](quality=quality, weights="DEFAULT")
        model.eval()
        model.update()

        # Prepare input
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Ensure dimensions are multiples of 64
        h, w = x.shape[2], x.shape[3]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # Compress
        with torch.no_grad():
            out = model.compress(x)
            rec = model.decompress(out["strings"], out["shape"])

        # Compute actual bytes
        total_bytes = sum(len(s[0]) for s in out["strings"])

        # Crop padding and convert
        recon = rec["x_hat"][0, :, :h, :w]
        recon = (recon.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        bpp = total_bytes * 8 / (h * w)

        return CompressAIResult(
            image=recon,
            actual_bytes=total_bytes,
            model_name=model_name,
            quality=quality,
            bpp=bpp,
        )

    except ImportError:
        # CompressAI not installed — return input as-is with large byte count
        return CompressAIResult(
            image=image.copy(),
            actual_bytes=image.nbytes,
            model_name=f"{model_name} (NOT INSTALLED)",
            quality=quality,
            bpp=24.0,
        )


def run_compressai_sweep(
    image: np.ndarray,
    model_name: str = "bmshj2018-hyperprior",
    qualities: list[int] | None = None,
) -> list[CompressAIResult]:
    """Run CompressAI baseline at multiple quality levels.

    Args:
        image: RGB uint8 image.
        model_name: CompressAI model name.
        qualities: Quality levels to test.

    Returns:
        List of CompressAIResult, one per quality level.
    """
    if qualities is None:
        qualities = [1, 2, 3, 4]

    results = []
    for q in qualities:
        results.append(compressai_encode_decode(image, model_name, q))

    return results
