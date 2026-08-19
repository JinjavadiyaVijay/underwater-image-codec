"""Run Ablation studies on UWCodec.

This script tests different architecture configurations (channel sizes, number of
resnet blocks) at a fixed budget (124B) to find the optimal trade-off between
model size and reconstruction quality.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Run Ablation Studies")
    p.add_argument("--dataset", type=str, default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--budget", type=int, default=124)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-base", type=Path, default=Path("outputs/ablations"))
    return p.parse_args()


def get_ablation_configs():
    """Define the variants to test."""
    return {
        "baseline": {}, # Default config
        "more_channels": {"--hidden-channels": "64"}, # Default is 32
        "fewer_channels": {"--hidden-channels": "16"},
        "more_resblocks": {"--num-res-blocks": "4"}, # Default is 2
        "no_perceptual_loss": {"--lambda-perceptual": "0.0"},
    }


def main():
    args = parse_args()
    
    # Verify dataset exists
    dataset_dir = args.datasets_root / args.dataset.upper()
    if not dataset_dir.exists():
        print(f"Error: Dataset {args.dataset.upper()} not found at {dataset_dir}")
        print("Please ensure it is downloaded and setup first.")
        sys.exit(1)
        
    print(f"Starting Ablation Study (Budget: {args.budget}B)")
    configs = get_ablation_configs()
    
    for name, config_args in configs.items():
        print("\n" + "="*60)
        print(f"ABLATION VARIANT: {name}")
        print("="*60)
        
        output_dir = args.output_base / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            "-m", "uwcodec.training.train_codec",
            "--dataset", args.dataset,
            "--datasets-root", str(args.datasets_root),
            "--train-budget", str(args.budget),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--output-dir", str(output_dir),
            "--device", args.device,
            "--num-images", str(args.num_images),
        ]
        
        for k, v in config_args.items():
            cmd.extend([k, v])
            
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Ablation {name} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] Ablation {name} failed with code {e.returncode}")
            sys.exit(e.returncode)
            
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved in: {args.output_base}")
    print("Compare the validation loss and perceptual quality across variants.")
    print("="*60)


if __name__ == "__main__":
    main()
