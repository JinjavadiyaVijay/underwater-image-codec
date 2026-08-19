"""Deployment profiling script for UWCodec.

Measures encoder/decoder latency and model memory footprint on CPU
(the target deployment scenario for embedded receivers).

Usage:
    python scripts/profile_deployment.py --model outputs/train/best.pt
    python scripts/profile_deployment.py --model outputs/train/best.pt --budgets 64 96 124
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Profile UWCodec Deployment Performance")
    p.add_argument("--model", type=Path, required=True, help="Trained codec model (.pt)")
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/profiling"))
    return p.parse_args()


def make_test_image(size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def profile_budget(codec, img: np.ndarray, budget: int, num_warmup: int, num_iters: int) -> dict:
    """Profile encode+decode for one budget."""
    # Warmup
    for _ in range(num_warmup):
        payload = codec.encode(img, max_bytes=budget)
        _ = codec.decode(payload)

    # Encode timing
    encode_times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        payload = codec.encode(img, max_bytes=budget)
        encode_times.append((time.perf_counter() - t0) * 1000)

    # Decode timing
    decode_times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = codec.decode(payload)
        decode_times.append((time.perf_counter() - t0) * 1000)

    enc_ms = float(np.median(encode_times))
    dec_ms = float(np.median(decode_times))
    total_ms = enc_ms + dec_ms

    return {
        "budget_bytes": budget,
        "payload_bytes": len(payload),
        "encode_ms_median": enc_ms,
        "decode_ms_median": dec_ms,
        "total_ms": total_ms,
        "throughput_fps": 1000.0 / max(total_ms, 1e-6),
        "encode_ms_p95": float(np.percentile(encode_times, 95)),
        "decode_ms_p95": float(np.percentile(decode_times, 95)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first: python -m uwcodec.training.train_codec --synthetic")
        return

    print("=" * 60)
    print("UWCodec Deployment Profiling")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Iterations: {args.num_iters} (after {args.num_warmup} warmup)")
    print(f"Platform:   CPU (deployment target)")

    from uwcodec.core.codec import UWCodec
    import torch

    codec = UWCodec.load(args.model, device="cpu")
    info = codec.model_info()

    print(f"\nModel Info:")
    print(f"  Params:        {info['parameters']['total']:,}")
    print(f"  Spatial grid:  {info['spatial_grid']}")
    print(f"  Codebook:      {info['codebook_size']}")

    # Estimate model size in MB
    model_size_mb = sum(
        p.numel() * p.element_size()
        for p in codec.model.parameters()
    ) / (1024 ** 2)
    print(f"  Model size:    {model_size_mb:.2f} MB (fp32)")
    print(f"  Model size:    {model_size_mb / 4:.2f} MB (int8 quantized, estimated)")

    img = make_test_image(args.image_size)

    print(f"\nProfiling across budgets: {args.budgets}")
    print("-" * 60)
    print(f"{'Budget':>8} | {'Payload':>8} | {'Encode':>10} | {'Decode':>10} | {'Total':>10} | {'FPS':>8}")
    print("-" * 60)

    all_results = []
    for budget in args.budgets:
        r = profile_budget(codec, img, budget, args.num_warmup, args.num_iters)
        all_results.append(r)
        print(
            f"  {budget:4d}B   | {r['payload_bytes']:4d}B    | "
            f"{r['encode_ms_median']:6.1f} ms  | "
            f"{r['decode_ms_median']:6.1f} ms  | "
            f"{r['total_ms']:6.1f} ms  | "
            f"{r['throughput_fps']:5.1f} fps"
        )

    print("-" * 60)

    # P95 latency
    print("\nP95 Latency (worst-case):")
    for r in all_results:
        print(f"  {r['budget_bytes']:3d}B: encode {r['encode_ms_p95']:.1f}ms | decode {r['decode_ms_p95']:.1f}ms")

    # Save results
    import json
    results_path = args.output_dir / "profiling_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": str(args.model),
            "image_size": args.image_size,
            "model_size_mb_fp32": model_size_mb,
            "model_params": info["parameters"]["total"],
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
