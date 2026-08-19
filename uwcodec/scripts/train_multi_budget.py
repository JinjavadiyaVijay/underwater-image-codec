"""Train multiple models across different byte budgets.

This script sequentially trains MinimalVQVAE models at 64B, 96B, and 124B budgets
and saves their checkpoints in separate directories.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Multi-budget training runner")
    p.add_argument("--dataset", type=str, default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p.add_argument("--num-images", type=int, default=2000, help="Number of train images to use")
    p.add_argument("--output-base", type=Path, default=Path("outputs/multi_budget"))
    return p.parse_args()


def main():
    args = parse_args()
    
    # Verify dataset exists
    dataset_dir = args.datasets_root / args.dataset.upper()
    if not dataset_dir.exists():
        print(f"Error: Dataset {args.dataset.upper()} not found at {dataset_dir}")
        print("Please ensure it is downloaded and setup first.")
        sys.exit(1)
        
    print(f"Starting multi-budget training for budgets: {args.budgets}")
    print(f"Dataset: {args.dataset.upper()}")
    
    for budget in args.budgets:
        print("\n" + "="*60)
        print(f"TRAINING BUDGET: {budget} BYTES")
        print("="*60)
        
        output_dir = args.output_base / f"budget_{budget}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            "-m", "uwcodec.training.train_codec",
            "--dataset", args.dataset,
            "--datasets-root", str(args.datasets_root),
            "--train-budget", str(budget),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--output-dir", str(output_dir),
            "--device", args.device,
            "--num-images", str(args.num_images),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Training for {budget}B completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] Training for {budget}B failed with code {e.returncode}")
            sys.exit(e.returncode)
            
    print("\n" + "="*60)
    print("MULTI-BUDGET TRAINING COMPLETE")
    print(f"Checkpoints saved in: {args.output_base}")
    print("="*60)


if __name__ == "__main__":
    main()
