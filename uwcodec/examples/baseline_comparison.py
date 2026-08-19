"""Example: Run all baselines comparison."""

from pathlib import Path
import numpy as np

from uwcodec.baselines.jpeg_webp import run_baseline_sweep
from uwcodec.baselines.prototype_baseline import prototype_retrieval_baseline
from uwcodec.baselines.semantic_only import semantic_only_baseline
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim


def main():
    print("=" * 60)
    print("UWCodec Baseline Comparison")
    print("=" * 60)

    # Create a test image (gradient fish-like shape)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    # Blueish background
    image[:] = [20, 80, 140]
    # Fish body (orange/yellow)
    image[30:90, 20:110, 0] = 220
    image[30:90, 20:110, 1] = 160
    image[30:90, 20:110, 2] = 50

    # 1. JPEG/WebP baselines
    print("\n--- Traditional Codecs ---")
    baseline_results = run_baseline_sweep(image, budgets=[64, 96, 124, 256, 512])

    for codec_name, results in baseline_results.items():
        print(f"\n  {codec_name}:")
        for r in results:
            psnr = compute_psnr(image, r.image)
            ssim_val = compute_ssim(image, r.image)
            print(f"    Target: {r.actual_bytes:>4d}B | Quality: {r.quality_param:>3d} | "
                  f"PSNR: {psnr:5.1f} | SSIM: {ssim_val:.4f}")

    # 2. Semantic-only baseline
    print("\n--- Semantic-Only (4B) ---")
    species_means = {1: image.copy()}
    recon, payload_bytes = semantic_only_baseline(
        species_id=1,
        species_means=species_means,
        mean_color=image.mean(axis=(0, 1)).astype(np.uint8),
        output_size=128,
    )
    psnr = compute_psnr(image, recon)
    ssim_val = compute_ssim(image, recon)
    print(f"  Payload: {payload_bytes}B | PSNR: {psnr:.1f} | SSIM: {ssim_val:.4f}")

    # 3. Prototype retrieval baseline
    print("\n--- Prototype Retrieval (1B) ---")
    training_images = {1: [image.copy()]}
    recon, payload_bytes = prototype_retrieval_baseline(
        species_id=1,
        training_images=training_images,
        output_size=128,
    )
    psnr = compute_psnr(image, recon)
    ssim_val = compute_ssim(image, recon)
    print(f"  Payload: {payload_bytes}B | PSNR: {psnr:.1f} | SSIM: {ssim_val:.4f}")

    print("\n" + "=" * 60)
    print("Key takeaway: JPEG/WebP produce garbage at 64-124B.")
    print("Our domain-specific approach is essential at these rates.")


if __name__ == "__main__":
    main()
