"""Full evaluation benchmark runner."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run_benchmark(
    codec,
    dataset,
    byte_budgets: list[int] | None = None,
    output_dir: str | Path = "outputs/benchmark",
    max_samples: int = 100,
) -> dict:
    """Run full evaluation benchmark.

    Args:
        codec: Codec instance with encode() and decode() methods.
        dataset: FishCropDataset instance.
        byte_budgets: Budgets to test.
        output_dir: Output directory for results.
        max_samples: Maximum samples to evaluate.

    Returns:
        Dict with all benchmark results.
    """
    from uwcodec.evaluation.metrics import compute_all_metrics
    from uwcodec.evaluation.structure_metrics import compute_structure_metrics
    from uwcodec.evaluation.visualize import create_comparison_grid

    if byte_budgets is None:
        byte_budgets = [64, 96, 124]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(len(dataset), max_samples)
    samples = [dataset[i] for i in range(num_samples)]

    results = {}

    for budget in byte_budgets:
        print(f"\n=== Benchmark @ {budget}B ===")
        budget_results = []

        originals = []
        reconstructions = []

        for i, sample in enumerate(samples):
            image = sample["image"]
            species_id = sample["species_id"]

            t0 = time.perf_counter()
            payload = codec.encode(image, species_id=species_id, max_bytes=budget)
            enc_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            result = codec.decode(payload)
            dec_ms = (time.perf_counter() - t0) * 1000

            if hasattr(result, 'image'):
                recon = result.image
            else:
                recon = result

            # Compute metrics
            metrics = compute_all_metrics(
                image, recon,
                payload_bytes=budget,
                encode_time_ms=enc_ms,
                decode_time_ms=dec_ms,
                compute_lpips_flag=False,
            )

            struct = compute_structure_metrics(image, recon)

            budget_results.append({
                "sample_idx": i,
                "species_id": species_id,
                "species_name": sample["species_name"],
                "metrics": metrics.metrics,
                "structure": struct,
                "encode_ms": enc_ms,
                "decode_ms": dec_ms,
            })

            originals.append(image)
            reconstructions.append(recon)

        # Aggregate
        all_psnr = [r["metrics"].get("psnr", 0) for r in budget_results if r["metrics"].get("psnr", 0) != float("inf")]
        all_ssim = [r["metrics"].get("ssim", 0) for r in budget_results]
        all_iou = [r["structure"].get("silhouette_iou", 0) for r in budget_results]

        agg = {
            "budget_bytes": budget,
            "num_samples": num_samples,
            "mean_psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
            "mean_ssim": float(np.mean(all_ssim)),
            "mean_silhouette_iou": float(np.mean(all_iou)),
            "mean_encode_ms": float(np.mean([r["encode_ms"] for r in budget_results])),
            "mean_decode_ms": float(np.mean([r["decode_ms"] for r in budget_results])),
        }

        results[f"{budget}B"] = {"aggregate": agg, "per_sample": budget_results}
        print(f"  PSNR: {agg['mean_psnr']:.2f} | SSIM: {agg['mean_ssim']:.4f} | IoU: {agg['mean_silhouette_iou']:.4f}")

        # Visualization
        create_comparison_grid(
            originals[:8],
            {f"{budget}B": reconstructions[:8]},
            labels=[s["species_name"] for s in samples[:8]],
            title=f"Benchmark @ {budget}B",
            save_path=output_dir / f"benchmark_{budget}B.png",
        )

    # Save results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({k: v["aggregate"] for k, v in results.items()}, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {output_dir}")
    return results
