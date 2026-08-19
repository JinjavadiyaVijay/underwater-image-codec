"""Prototype retrieval baseline: species ID → nearest training image.

No encoder training needed. Pure retrieval establishes the floor for
\"how good is species-prior alone?\"
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def prototype_retrieval(
    species_id: int,
    training_images: dict[int, list[np.ndarray]],
    target_color: np.ndarray | None = None,
    output_size: int = 128,
) -> np.ndarray:
    """Retrieve the best matching prototype for a given species.

    Args:
        species_id: Species to retrieve.
        training_images: Dict mapping species_id → list of training crops.
        target_color: Optional target mean color (3,) to match against.
        output_size: Output image resolution.

    Returns:
        Best matching training image (or mean image if no match).
    """
    candidates = training_images.get(species_id, [])

    if not candidates:
        # No training data for this species — return gray
        return np.full((output_size, output_size, 3), 128, dtype=np.uint8)

    if target_color is not None and len(candidates) > 1:
        # Find closest color match
        best_idx = 0
        best_dist = float("inf")
        for idx, cand in enumerate(candidates):
            mean_color = cand.astype(np.float32).mean(axis=(0, 1))
            dist = np.sum((mean_color - target_color.astype(np.float32)) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        result = candidates[best_idx]
    else:
        # Return first (or random) candidate
        result = candidates[0]

    # Resize to output
    if result.shape[0] != output_size or result.shape[1] != output_size:
        result = np.array(
            Image.fromarray(result).resize((output_size, output_size), Image.LANCZOS)
        )

    return result


def prototype_retrieval_baseline(
    species_id: int,
    training_images: dict[int, list[np.ndarray]],
    output_size: int = 128,
) -> tuple[np.ndarray, int]:
    """Full prototype baseline: species ID alone → image, measure payload.

    Payload = 1 byte (species ID). The absolute minimum.

    Returns:
        (reconstructed_image, payload_bytes=1)
    """
    image = prototype_retrieval(species_id, training_images, output_size=output_size)
    return image, 1  # Just the species ID
