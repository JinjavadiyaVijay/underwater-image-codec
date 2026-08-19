"""JPEG/WebP baseline: compress at minimum practical sizes.

Shows what traditional codecs achieve at comparable file sizes.
Spoiler: they produce garbage at 64-124 bytes. That's the whole point.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class BaselineResult:
    """Result of a baseline codec encoding."""
    image: np.ndarray  # reconstructed RGB
    actual_bytes: int
    quality_param: int | float
    codec_name: str


def jpeg_encode_decode(
    image: np.ndarray,
    target_bytes: int | None = None,
    quality: int = 1,
) -> BaselineResult:
    """Encode/decode with JPEG.

    If target_bytes is specified, binary-search for the quality factor
    that produces output closest to that size.
    """
    if target_bytes is not None:
        quality = _binary_search_quality(image, target_bytes, "JPEG")

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    compressed = buf.getvalue()

    recon = np.array(Image.open(io.BytesIO(compressed)).convert("RGB"))
    return BaselineResult(
        image=recon,
        actual_bytes=len(compressed),
        quality_param=quality,
        codec_name="JPEG",
    )


def webp_encode_decode(
    image: np.ndarray,
    target_bytes: int | None = None,
    quality: int = 1,
) -> BaselineResult:
    """Encode/decode with WebP."""
    if target_bytes is not None:
        quality = _binary_search_quality(image, target_bytes, "WEBP")

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="WEBP", quality=quality)
    compressed = buf.getvalue()

    recon = np.array(Image.open(io.BytesIO(compressed)).convert("RGB"))
    return BaselineResult(
        image=recon,
        actual_bytes=len(compressed),
        quality_param=quality,
        codec_name="WebP",
    )


def _binary_search_quality(
    image: np.ndarray,
    target_bytes: int,
    fmt: str,
    lo: int = 1,
    hi: int = 95,
    max_iters: int = 20,
) -> int:
    """Binary search for quality factor closest to target byte size."""
    best_q = lo
    best_diff = float("inf")

    for _ in range(max_iters):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format=fmt, quality=mid)
        size = buf.tell()

        diff = abs(size - target_bytes)
        if diff < best_diff:
            best_diff = diff
            best_q = mid

        if size > target_bytes:
            hi = mid - 1
        elif size < target_bytes:
            lo = mid + 1
        else:
            return mid

    return best_q


def run_baseline_sweep(
    image: np.ndarray,
    budgets: list[int] | None = None,
) -> dict[str, list[BaselineResult]]:
    """Run JPEG and WebP baselines at all target byte budgets.

    Args:
        image: RGB uint8 image.
        budgets: List of byte budgets. Defaults to [64, 96, 124, 256, 512, 1024].

    Returns:
        Dict mapping codec name → list of results.
    """
    if budgets is None:
        budgets = [64, 96, 124, 256, 512, 1024]

    results = {"JPEG": [], "WebP": []}

    for budget in budgets:
        results["JPEG"].append(jpeg_encode_decode(image, target_bytes=budget))
        results["WebP"].append(webp_encode_decode(image, target_bytes=budget))

    return results
