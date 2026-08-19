"""Structure preservation metrics: silhouette IoU, edge overlap, pose preservation."""

from __future__ import annotations

import numpy as np


def compute_silhouette_iou(
    original: np.ndarray,
    reconstructed: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union of binary silhouettes.

    Args:
        original: Original image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).
        threshold: Binarization threshold (0-1 of max).

    Returns:
        IoU in [0, 1]. Higher is better.
    """
    from uwcodec.data.preprocessing import extract_silhouette

    mask_orig = extract_silhouette(original) > 127
    mask_recon = extract_silhouette(reconstructed) > 127

    intersection = np.logical_and(mask_orig, mask_recon).sum()
    union = np.logical_or(mask_orig, mask_recon).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_edge_preservation(
    original: np.ndarray,
    reconstructed: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """Compute edge overlap between original and reconstruction.

    Measures how well structural edges (fins, body outline) are preserved.

    Args:
        original: Original image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).

    Returns:
        Edge F1 score in [0, 1]. Higher is better.
    """
    from uwcodec.data.preprocessing import extract_edge_map

    edges_orig = extract_edge_map(original, threshold=threshold) > 127
    edges_recon = extract_edge_map(reconstructed, threshold=threshold) > 127

    # Dilate edges slightly for tolerance (3x3 neighborhood)
    edges_orig_dilated = _dilate(edges_orig, radius=1)
    edges_recon_dilated = _dilate(edges_recon, radius=1)

    # Precision: what fraction of recon edges are near original edges
    if edges_recon.sum() == 0:
        precision = 1.0 if edges_orig.sum() == 0 else 0.0
    else:
        precision = float(np.logical_and(edges_recon, edges_orig_dilated).sum() / edges_recon.sum())

    # Recall: what fraction of original edges are near recon edges
    if edges_orig.sum() == 0:
        recall = 1.0
    else:
        recall = float(np.logical_and(edges_orig, edges_recon_dilated).sum() / edges_orig.sum())

    # F1
    if precision + recall < 1e-8:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Simple binary dilation (no OpenCV dependency)."""
    result = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            result = np.logical_or(result, shifted)
    return result


def compute_structure_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    """Compute all structure preservation metrics.

    Args:
        original: Original image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).

    Returns:
        Dict with silhouette_iou, edge_f1, and aspect_ratio_error.
    """
    metrics = {
        "silhouette_iou": compute_silhouette_iou(original, reconstructed),
        "edge_f1": compute_edge_preservation(original, reconstructed),
    }

    # Aspect ratio preservation (from silhouettes)
    from uwcodec.data.preprocessing import extract_silhouette

    mask_orig = extract_silhouette(original) > 127
    mask_recon = extract_silhouette(reconstructed) > 127

    ar_orig = _aspect_ratio_from_mask(mask_orig)
    ar_recon = _aspect_ratio_from_mask(mask_recon)

    if ar_orig > 0:
        metrics["aspect_ratio_error"] = abs(ar_recon - ar_orig) / ar_orig
    else:
        metrics["aspect_ratio_error"] = 0.0

    return metrics


def _aspect_ratio_from_mask(mask: np.ndarray) -> float:
    """Compute aspect ratio of the bounding box of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return 0.0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h = rmax - rmin + 1
    w = cmax - cmin + 1

    return w / max(h, 1)
