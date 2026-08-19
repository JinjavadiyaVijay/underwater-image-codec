"""Run the General Oracle on REAL underwater images.

This script tests the theoretical limits of 64/96/124B budgets using
non-learned strategies (pixel grid, DCT, etc.) on real dataset images.
"""

import argparse
from pathlib import Path
import time
import csv

import numpy as np
from PIL import Image

from uwcodec.data.dataset import MultiDatasetLoader, find_images
from uwcodec.codecs.general_oracle import GeneralOracle
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim, compute_uiqm, compute_uciqe


def parse_args():
    p = argparse.ArgumentParser(description="Run General Oracle on Real Images")
    p.add_argument("--dataset", type=str, default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--num-images", type=int, default=20, help="Number of images to test")
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124, 256, 512, 1024])
    p.add_argument("--output-dir", type=Path, default=Path("oracle_results/real"))
    return p.parse_args()


def create_comparison_grid(
    original: np.ndarray,
    recons: dict[str, dict[int, np.ndarray]],
    budgets: list[int],
    out_path: Path,
):
    """Save a grid comparing strategies across budgets."""
    h, w, c = original.shape
    strategies = list(recons.keys())
    
    # Grid: Rows = strategies + 1 (original), Cols = budgets
    grid_h = h * len(strategies)
    grid_w = w * (len(budgets) + 1)
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # First column: original
    for i in range(len(strategies)):
        grid[i*h:(i+1)*h, 0:w] = original
        
    # Other columns: reconstructions
    for i, strategy in enumerate(strategies):
        for j, budget in enumerate(budgets):
            if budget in recons[strategy]:
                grid[i*h:(i+1)*h, (j+1)*w:(j+2)*w] = recons[strategy][budget]
                
    Image.fromarray(grid).save(out_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.dataset.upper()} {args.split}...")
    loader = MultiDatasetLoader(args.datasets_root)
    try:
        ds = loader.get_dataset(args.dataset, args.split, input_size=128)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is downloaded and extracted to the correct path.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    num_to_test = min(args.num_images, len(ds))
    if num_to_test == 0:
        print("No images found in dataset.")
        return
        
    print(f"Testing {num_to_test} images across {len(args.budgets)} budgets...")
    
    oracle = GeneralOracle()
    strategies = ["tiny_pixel_grid", "dct_coefficients", "mean_color_blocks"]
    
    results = []
    
    for i in range(num_to_test):
        item = ds[i]
        # Dataset returns float tensor [0, 1] (C, H, W)
        img_tensor = item["image"]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        recons = {s: {} for s in strategies}
        
        print(f"[{i+1}/{num_to_test}] Image {Path(item['path']).name}")
        
        for strategy in strategies:
            for budget in args.budgets:
                payload, recon = oracle.encode_decode(img_np, max_bytes=budget, strategy=strategy)
                recons[strategy][budget] = recon
                
                # Metrics
                psnr = compute_psnr(img_np, recon)
                ssim = compute_ssim(img_np, recon)
                
                results.append({
                    "image_id": i,
                    "image_name": Path(item['path']).name,
                    "strategy": strategy,
                    "budget": budget,
                    "psnr": psnr,
                    "ssim": ssim,
                })
                
        # Save visual grid for this image
        grid_path = args.output_dir / f"grid_{i:03d}.jpg"
        create_comparison_grid(img_np, recons, args.budgets, grid_path)
        
    # Aggregate and save metrics
    metrics_path = args.output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nDone! Results saved to {args.output_dir}")
    print("Aggregate PSNR by strategy and budget:")
    
    # Print summary
    for strategy in strategies:
        print(f"\n--- {strategy} ---")
        for budget in args.budgets:
            psnrs = [r["psnr"] for r in results if r["strategy"] == strategy and r["budget"] == budget]
            mean_psnr = sum(psnrs) / len(psnrs)
            print(f"  {budget}B: {mean_psnr:.2f} dB")


if __name__ == "__main__":
    main()
