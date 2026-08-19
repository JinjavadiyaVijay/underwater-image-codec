"""Compare UWCodec against standard baselines (JPEG, WebP).

This runs a rate-distortion study across a dataset to evaluate how traditional
codecs fail at 64-124B budgets, and how UWCodec compares.
"""

import argparse
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch

from uwcodec.data.dataset import MultiDatasetLoader
from uwcodec.core.codec import UWCodec
from uwcodec.baselines.jpeg_webp import run_baseline_sweep
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim


def parse_args():
    p = argparse.ArgumentParser(description="Run Baselines & Rate-Distortion Study")
    p.add_argument("--dataset", type=str, default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124, 256, 512, 1024])
    p.add_argument("--model", type=Path, default=None, help="Path to trained UWCodec model")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = MultiDatasetLoader(args.datasets_root)
    try:
        ds = loader.get_dataset(args.dataset, args.split, input_size=128)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    num_to_test = min(args.num_images, len(ds))
    if num_to_test == 0:
        print("No images found.")
        return
        
    print(f"Testing {num_to_test} images...")
    
    uwcodec = None
    if args.model and args.model.exists():
        print(f"Loading UWCodec from {args.model}")
        uwcodec = UWCodec.load(args.model)
    else:
        print("No UWCodec model provided/found. Testing JPEG/WebP only.")
        
    results = []
    
    for i in range(num_to_test):
        item = ds[i]
        img_np = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_name = Path(item["path"]).name
        print(f"[{i+1}/{num_to_test}] {img_name}")
        
        # 1. Traditional baselines
        baseline_res = run_baseline_sweep(img_np, budgets=args.budgets)
        
        # JPEG
        for budget, res_list in zip(args.budgets, [baseline_res.get("JPEG", [])]):
            pass  # handled below
        
        for codec_name, res_list in baseline_res.items():
            for budget, res in zip(args.budgets, res_list):
                if res.image is not None:
                    psnr = compute_psnr(img_np, res.image)
                    ssim = compute_ssim(img_np, res.image)
                else:
                    psnr, ssim = 0.0, 0.0
                results.append({
                    "image": img_name,
                    "codec": codec_name,
                    "budget": budget,
                    "actual_bytes": res.actual_bytes,
                    "psnr": psnr,
                    "ssim": ssim,
                })
            
        # 2. UWCodec
        if uwcodec is not None:
            for budget in args.budgets:
                try:
                    payload, recon = uwcodec.encode_decode(img_np, max_bytes=budget)
                    psnr = compute_psnr(img_np, recon)
                    ssim = compute_ssim(img_np, recon)
                    results.append({
                        "image": img_name,
                        "codec": "UWCodec",
                        "budget": budget,
                        "actual_bytes": len(payload),
                        "psnr": psnr,
                        "ssim": ssim,
                    })
                except Exception as e:
                    print(f"  UWCodec failed at {budget}B: {e}")
                    
    # Save metrics
    metrics_path = args.output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nDone! Results saved to {metrics_path}")
    
    # Print summary
    print("\nMean PSNR by Codec & Budget:")
    codecs = set(r["codec"] for r in results)
    for codec in sorted(codecs):
        print(f"\n--- {codec} ---")
        for budget in args.budgets:
            recs = [r for r in results if r["codec"] == codec and r["budget"] == budget and r["psnr"] > 0]
            if recs:
                mean_psnr = sum(r["psnr"] for r in recs) / len(recs)
                mean_bytes = sum(r["actual_bytes"] for r in recs) / len(recs)
                print(f"  {budget:4d}B target -> ~{mean_bytes:4.0f}B actual: {mean_psnr:.2f} dB")
            else:
                print(f"  {budget:4d}B target -> FAILED to compress")


if __name__ == "__main__":
    main()
