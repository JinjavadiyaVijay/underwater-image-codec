"""Biology-specific evaluation metrics.

Species agreement, BioCLIP feature similarity, confidence change,
hallucination/failure rate. These are FIRST-CLASS metrics — PSNR/SSIM
alone are NOT sufficient for biological image reconstruction.

A generic beautiful fish of the correct species is NOT considered successful
reconstruction if it does not preserve information about the actual photographed fish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BioMetricsResult:
    """Biology-specific metrics for reconstruction quality."""

    species_top1_match: bool = False
    species_top5_match: bool = False
    species_top1_confidence_original: float = 0.0
    species_top1_confidence_reconstructed: float = 0.0
    confidence_delta: float = 0.0  # recon_conf - orig_conf (negative = lost confidence)
    bioclip_cosine_similarity: float = 0.0  # if BioCLIP available
    hallucination_detected: bool = False
    failure_mode: str = ""  # "wrong_species", "low_confidence", "no_detection", ""

    def summary(self) -> str:
        lines = [
            "=== Bio Metrics ===",
            f"  Species Top-1 Match:    {self.species_top1_match}",
            f"  Species Top-5 Match:    {self.species_top5_match}",
            f"  Orig Confidence:        {self.species_top1_confidence_original:.3f}",
            f"  Recon Confidence:       {self.species_top1_confidence_reconstructed:.3f}",
            f"  Confidence Delta:       {self.confidence_delta:+.3f}",
            f"  BioCLIP Similarity:     {self.bioclip_cosine_similarity:.4f}",
            f"  Hallucination:          {self.hallucination_detected}",
            f"  Failure Mode:           {self.failure_mode or 'none'}",
        ]
        return "\n".join(lines)


def compute_species_agreement(
    original_species_id: int,
    reconstructed_image: np.ndarray,
    classifier_fn: Any = None,
    top_k: int = 5,
) -> BioMetricsResult:
    """Evaluate whether species identity is preserved through encode/decode.

    This is THE critical experiment: re-classify the reconstruction and check
    if it still reads as the correct species.

    Args:
        original_species_id: Ground-truth species ID.
        reconstructed_image: Reconstructed RGB image (H, W, 3) uint8.
        classifier_fn: Callable that takes an image and returns
                       (top_k_ids: list[int], top_k_probs: list[float]).
                       If None, returns empty metrics.
        top_k: Number of top predictions to check.

    Returns:
        BioMetricsResult with species agreement data.
    """
    result = BioMetricsResult()

    if classifier_fn is None:
        result.failure_mode = "no_classifier"
        return result

    try:
        top_ids, top_probs = classifier_fn(reconstructed_image)
    except Exception as e:
        result.failure_mode = f"classifier_error: {e}"
        return result

    if not top_ids:
        result.failure_mode = "no_detection"
        return result

    # Top-1 match
    result.species_top1_match = top_ids[0] == original_species_id
    result.species_top1_confidence_reconstructed = top_probs[0] if top_probs else 0.0

    # Top-5 match
    result.species_top5_match = original_species_id in top_ids[:top_k]

    # Confidence delta (requires original confidence)
    # This will be filled in by the caller if available

    # Hallucination detection
    if not result.species_top1_match:
        result.hallucination_detected = True
        result.failure_mode = "wrong_species"
    elif result.species_top1_confidence_reconstructed < 0.3:
        result.failure_mode = "low_confidence"

    return result


def compute_bioclip_similarity(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    bioclip_fn: Any = None,
) -> float:
    """Compute BioCLIP-2 feature similarity between original and reconstruction.

    BioCLIP-2 is used ONLY as a frozen evaluation teacher — never deployed.

    Args:
        original_image: Original RGB crop (H, W, 3) uint8.
        reconstructed_image: Reconstructed RGB crop (H, W, 3) uint8.
        bioclip_fn: Callable that takes an image and returns a feature vector.
                    If None, returns 0.0.

    Returns:
        Cosine similarity in [-1, 1]. Higher is better.
    """
    if bioclip_fn is None:
        return 0.0

    try:
        feat_orig = bioclip_fn(original_image)
        feat_recon = bioclip_fn(reconstructed_image)

        # Cosine similarity
        feat_orig = np.array(feat_orig, dtype=np.float64).flatten()
        feat_recon = np.array(feat_recon, dtype=np.float64).flatten()

        norm_orig = np.linalg.norm(feat_orig)
        norm_recon = np.linalg.norm(feat_recon)

        if norm_orig < 1e-8 or norm_recon < 1e-8:
            return 0.0

        return float(np.dot(feat_orig, feat_recon) / (norm_orig * norm_recon))
    except Exception:
        return 0.0


def compute_hallucination_rate(
    results: list[BioMetricsResult],
) -> dict[str, float]:
    """Compute aggregate hallucination statistics.

    Args:
        results: List of per-image BioMetricsResult.

    Returns:
        Dict with hallucination statistics.
    """
    if not results:
        return {"hallucination_rate": 0.0, "failure_rate": 0.0}

    n = len(results)
    n_hallucinated = sum(1 for r in results if r.hallucination_detected)
    n_failed = sum(1 for r in results if r.failure_mode != "")
    n_top1 = sum(1 for r in results if r.species_top1_match)
    n_top5 = sum(1 for r in results if r.species_top5_match)

    conf_deltas = [r.confidence_delta for r in results if r.confidence_delta != 0.0]

    return {
        "hallucination_rate": n_hallucinated / n,
        "failure_rate": n_failed / n,
        "species_top1_accuracy": n_top1 / n,
        "species_top5_accuracy": n_top5 / n,
        "mean_confidence_delta": np.mean(conf_deltas) if conf_deltas else 0.0,
        "median_confidence_delta": np.median(conf_deltas) if conf_deltas else 0.0,
        "total_images": n,
    }
