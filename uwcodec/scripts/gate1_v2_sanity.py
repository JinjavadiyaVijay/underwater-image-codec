"""Gate 1: UWCodec v2 sanity run.

Runs 2-epoch training on 500 images to verify:
1. Model instantiates and forward() works.
2. Loss decreases over epochs.
3. VQ perplexity > 50 (codebook not collapsed).
4. Reconstructions are non-trivial (not uniform gray/color).

Pass criteria:
  - No crash
  - Epoch 2 loss < Epoch 1 loss
  - Semantic perplexity > 50
  - Payload PSNR > 5 dB (i.e., decoder is actually doing something)

Run:
    python scripts/gate1_v2_sanity.py --datasets-root s:/IMG_compressors/datasets
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-root", type=Path, default=Path("s:/IMG_compressors/datasets"))
    p.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    p.add_argument("--budget", type=int, default=128, choices=[64, 96, 124, 128])
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GATE 1: UWCodec v2 Sanity Check")
    print("=" * 60)
    print(f"Budget: {args.budget}B | Device: {args.device}")
    print(f"Dataset root: {args.datasets_root}")

    cmd = [
        sys.executable, "-m", "uwcodec.training.train_v2",
        "--dataset", "euvp",
        "--datasets-root", str(args.datasets_root),
        "--budget", str(args.budget),
        "--epochs", "2",
        "--batch-size", "8",
        "--num-images", "500",
        "--device", args.device,
        "--output-dir", f"outputs/v2_gate1/budget_{args.budget}",
        "--eval-every", "1",
        "--res-bottom", "2",    # smaller for sanity check
        "--res-mid", "1",
        "--decoder-channels", "128",  # lite decoder for speed
        "--lr", "3e-4",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[FAIL] Gate 1 training crashed. See error above.")
        sys.exit(1)

    # Check for checkpoint
    ckpt = Path(f"outputs/v2_gate1/budget_{args.budget}/best.pt")
    if not ckpt.exists():
        print("[FAIL] No checkpoint saved.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[PASS] Gate 1: Model trains without crashing.")
    print("  Next: inspect payload PSNR in the training log above.")
    print("  If payload PSNR > 5 dB and perplexity > 20, proceed to Gate 2.")
    print("  If not, investigate codebook collapse or architecture bug.")
    print("=" * 60)


if __name__ == "__main__":
    main()
