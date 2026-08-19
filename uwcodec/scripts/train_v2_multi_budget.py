"""Multi-budget runner for UWCodec v2.

Automatically schedules training runs for budgets 64, 96, 124, 128.
Skips budgets that already have a complete checkpoint.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BUDGETS = [128, 124, 96, 64]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-root", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/v2"))
    return p.parse_args()

def main():
    args = parse_args()
    
    for budget in BUDGETS:
        out_dir = args.output_dir / f"budget_{budget}"
        if (out_dir / "best.pt").exists():
            print(f"Skipping budget {budget}B (best.pt already exists)")
            continue
            
        print(f"\n" + "="*60)
        print(f"Starting training for {budget}B budget")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "uwcodec.training.train_v2",
            "--dataset", "euvp",
            "--datasets-root", str(args.datasets_root),
            "--budget", str(budget),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--output-dir", str(out_dir),
            "--eval-every", "5",
            "--save-every", "10",
        ]
        
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Training failed for budget {budget}B! Stopping.")
            sys.exit(1)

if __name__ == "__main__":
    main()
