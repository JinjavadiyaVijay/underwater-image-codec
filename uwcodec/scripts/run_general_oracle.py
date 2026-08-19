"""Fail-fast oracle experiment: run BEFORE any training.

Tests whether 64-124 bytes can encode useful visual information
without ANY learning. This is the mandatory experimental gate.

Usage:
    # With real data:
    python scripts/run_general_oracle.py --data-dir path/to/underwater/images

    # With synthetic data (for quick test):
    python scripts/run_general_oracle.py --synthetic

    # Custom budgets:
    python scripts/run_general_oracle.py --data-dir ./data --budgets 64 96 124 256 512

Output: Results table + verdict (PASS/MARGINAL/FAIL per strategy+budget).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def load_test_images(
    data_dir: Path | None,
    synthetic: bool,
    num_images: int,
    image_size: int,
) -> list[np.ndarray]:
    """Load test images for the oracle experiment."""

    if synthetic or data_dir is None:
        print(f"Generating {num_images} synthetic underwater images...")
        from uwcodec.data.dataset import create_synthetic_dataset
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        create_synthetic_dataset(tmp, num_images=num_images, image_size=image_size)
        data_dir = tmp

    from uwcodec.data.dataset import find_images
    paths = find_images(data_dir)[:num_images]
    if not paths:
        raise ValueError(f"No images found in {data_dir}")

    images = []
    for p in paths:
        try:
            img = np.array(Image.open(p).convert("RGB"))
            images.append(img)
        except Exception as e:
            print(f"  Skip {p.name}: {e}")

    print(f"Loaded {len(images)} images from {data_dir}")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="General image oracle experiment (fail-fast gate before training)"
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing underwater images")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic underwater images (for testing)")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of images to test (default: 20)")
    parser.add_argument("--output-size", type=int, default=128,
                        help="Resize images to this square size (default: 128)")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[64, 96, 124, 256, 512, 1024],
                        help="Byte budgets to test")
    parser.add_argument("--strategies", nargs="+",
                        default=["tiny_pixel_grid", "dct_coefficients", "mean_color_blocks"],
                        help="Oracle strategies to test")
    parser.add_argument("--save-images", type=Path, default=None,
                        help="Save comparison images to this directory")
    args = parser.parse_args()

    if not args.synthetic and args.data_dir is None:
        print("WARNING: No --data-dir provided. Using synthetic images.")
        print("For real evaluation, run: python scripts/run_general_oracle.py --data-dir <path>")
        args.synthetic = True

    # Load images
    images = load_test_images(
        args.data_dir, args.synthetic, args.num_images, args.output_size
    )

    # Run oracle
    from uwcodec.codecs.general_oracle import run_oracle_experiment
    results = run_oracle_experiment(
        images=images,
        budgets=args.budgets,
        strategies=args.strategies,
        output_size=args.output_size,
        verbose=True,
    )

    # Optionally save visual comparisons
    if args.save_images and images:
        args.save_images.mkdir(parents=True, exist_ok=True)
        from uwcodec.codecs.general_oracle import GeneralOracle
        oracle = GeneralOracle(output_size=args.output_size)
        test_img = images[0]

        print(f"\nSaving visual comparisons to {args.save_images}...")
        orig_resized = np.array(
            Image.fromarray(test_img).resize((args.output_size, args.output_size), Image.LANCZOS)
        )
        Image.fromarray(orig_resized).save(args.save_images / "original.png")

        for strategy in args.strategies:
            for budget in [64, 124]:
                if budget in args.budgets:
                    try:
                        _, recon = oracle.encode_decode(test_img, budget, strategy=strategy)
                        name = f"oracle_{strategy}_{budget}B.png"
                        Image.fromarray(recon).save(args.save_images / name)
                    except Exception as e:
                        print(f"  Could not save {strategy}/{budget}B: {e}")

        print(f"Saved to {args.save_images}/")

    # Return exit code based on overall verdict
    all_fail = all(r.verdict == "FAIL" for r in results if r.budget <= 124)
    if all_fail:
        print("\nEXIT 1: All strategies fail at ≤124B. Consider larger budgets or report limitation.")
        sys.exit(1)
    else:
        print("\nEXIT 0: At least one strategy shows useful reconstruction at ≤124B.")
        sys.exit(0)


if __name__ == "__main__":
    main()
