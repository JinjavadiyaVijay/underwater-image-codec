"""Semantic-only baseline: species + mean color → per-species mean image.

The minimal information baseline (< 10 bytes). If this is already good enough,
the entire learned codec is unnecessary.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def semantic_only_reconstruction(
    species_id: int,
    species_means: dict[int, np.ndarray],
    mean_color: np.ndarray | None = None,
    output_size: int = 128,
) -> np.ndarray:
    """Reconstruct from species ID + optional mean color only.

    Payload cost: 1 byte (species ID) + 3 bytes (mean RGB) = 4 bytes total.

    Args:
        species_id: Species classification.
        species_means: Dict mapping species_id → mean training image.
        mean_color: Optional (3,) mean RGB to color-shift the mean image.
        output_size: Output resolution.

    Returns:
        Reconstructed RGB image.
    """
    base = species_means.get(species_id)

    if base is None:
        # Unknown species — return solid color
        if mean_color is not None:
            img = np.full((output_size, output_size, 3), 0, dtype=np.uint8)
            img[:] = mean_color.astype(np.uint8)
            return img
        return np.full((output_size, output_size, 3), 128, dtype=np.uint8)

    # Resize to output
    if base.shape[0] != output_size or base.shape[1] != output_size:
        base = np.array(
            Image.fromarray(base).resize((output_size, output_size), Image.LANCZOS)
        )

    if mean_color is not None:
        # Color-shift: adjust the mean image toward the transmitted color
        current_mean = base.astype(np.float32).mean(axis=(0, 1))
        target_mean = mean_color.astype(np.float32)
        shift = target_mean - current_mean
        result = np.clip(base.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        return result

    return base.copy()


def semantic_only_baseline(
    species_id: int,
    species_means: dict[int, np.ndarray],
    mean_color: np.ndarray | None = None,
    output_size: int = 128,
) -> tuple[np.ndarray, int]:
    """Full semantic-only baseline.

    Returns:
        (reconstructed_image, payload_bytes)
    """
    image = semantic_only_reconstruction(
        species_id, species_means, mean_color, output_size
    )
    payload_bytes = 1 + (3 if mean_color is not None else 0)
    return image, payload_bytes
