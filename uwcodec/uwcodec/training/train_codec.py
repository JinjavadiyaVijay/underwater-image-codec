"""Training script for UWCodec minimal VQ-VAE.

IMPORTANT: Run run_general_oracle.py FIRST to verify the byte budget
can convey useful information before investing in training.

Usage:
    # With real underwater images:
    python -m uwcodec.training.train_codec --data-dir path/to/images

    # Quick smoke test on synthetic data:
    python -m uwcodec.training.train_codec --synthetic --epochs 5 --num-images 200

    # Full training run:
    python -m uwcodec.training.train_codec --data-dir ./data --epochs 50 --batch-size 16

Training stops early if loss does not decrease for 10 epochs.
Results are reported honestly: no fabricated metrics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train UWCodec VQ-VAE")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    p.add_argument("--num-images", type=int, default=500, help="Max images to use")
    p.add_argument("--input-size", type=int, default=128, help="Encoder input resolution")
    p.add_argument("--output-size", type=int, default=128, help="Decoder output resolution")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-pixel", type=float, default=1.0)
    p.add_argument("--lambda-perceptual", type=float, default=0.5)
    p.add_argument("--lambda-vq", type=float, default=1.0)
    p.add_argument("--train-budget", type=int, default=124,
                   help="Primary byte budget for training")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=15)
    return p.parse_args()


def build_dataset(args):
    """Build train/val datasets."""
    from uwcodec.data.dataset import UnderwaterImageDataset, create_synthetic_dataset

    if args.synthetic or args.data_dir is None:
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "synthetic"
        print(f"Generating synthetic dataset ({args.num_images} images)...")
        create_synthetic_dataset(tmp, num_images=args.num_images, image_size=args.input_size * 2)
        data_dir = tmp
    else:
        data_dir = args.data_dir

    train_ds = UnderwaterImageDataset.from_directory(
        root=data_dir,
        input_size=args.input_size,
        split="train",
        augment=True,
        verbose=True,
    )
    val_ds = UnderwaterImageDataset.from_directory(
        root=data_dir,
        input_size=args.input_size,
        split="val",
        augment=False,
        verbose=False,
    )
    return train_ds, val_ds


def compute_perceptual_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple perceptual-like loss using gradient magnitude.

    This avoids the VGG dependency while still penalizing blurring.
    Sobol edge detection in x and y.
    """
    def gradient_magnitude(img):
        # Simple finite difference gradients
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        # Pad to original size
        dy = nn.functional.pad(dy, (0, 0, 0, 1))
        dx = nn.functional.pad(dx, (0, 1, 0, 0))
        return torch.sqrt(dx**2 + dy**2 + 1e-8)

    grad_recon = gradient_magnitude(recon)
    grad_target = gradient_magnitude(target)
    return nn.functional.l1_loss(grad_recon, grad_target)


def train_one_epoch(model, loader, optimizer, device, args) -> dict:
    model.train()
    total_pixel = 0.0
    total_perceptual = 0.0
    total_vq = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()
        out = model(images)

        recon = out["reconstruction"]
        pixel_loss = nn.functional.l1_loss(recon, images)
        perceptual_loss = compute_perceptual_loss(recon, images)
        vq_loss = out["vq_loss"]

        loss = (
            args.lambda_pixel * pixel_loss
            + args.lambda_perceptual * perceptual_loss
            + args.lambda_vq * vq_loss
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_pixel += pixel_loss.item()
        total_perceptual += perceptual_loss.item()
        total_vq += vq_loss.item()
        total_loss += loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "pixel": total_pixel / n,
        "perceptual": total_perceptual / n,
        "vq": total_vq / n,
        "perplexity": out["perplexity"].item() if n_batches > 0 else 0.0,
    }


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    total_pixel = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        total_pixel += nn.functional.l1_loss(out["reconstruction"], images).item()
        n_batches += 1

    n = max(n_batches, 1)
    return {"val_pixel_loss": total_pixel / n}


@torch.no_grad()
def compute_psnr_sample(model, loader, device, num_samples: int = 4) -> float:
    """Compute PSNR on a few validation samples."""
    from uwcodec.evaluation.metrics import compute_psnr
    model.eval()
    psnrs = []

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        recon = out["reconstruction"]

        for i in range(min(len(images), num_samples - len(psnrs))):
            orig_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            recon_np = (recon[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            p = compute_psnr(orig_np, recon_np)
            if not (np.isinf(p) or np.isnan(p)):
                psnrs.append(p)

        if len(psnrs) >= num_samples:
            break

    return float(np.mean(psnrs)) if psnrs else 0.0


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("\nBuilding dataset...")
    train_ds, val_ds = build_dataset(args)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: Empty training set. Check --data-dir.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    from uwcodec.codecs.vqvae_codec import MinimalVQVAE
    model = MinimalVQVAE(
        input_size=args.input_size,
        output_size=args.output_size,
    ).to(device)

    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"  Encoder:  {params['encoder']:,}")
    print(f"  Quantizer:{params['quantizer']:,}")
    print(f"  Decoder:  {params['decoder']:,}")
    print(f"  Total:    {params['total']:,}")

    # Budget info
    from uwcodec.core.config import PayloadConfig
    pc = PayloadConfig()
    print(f"\nByte budget info:")
    for b in [64, 96, 124, 256]:
        vq_b = pc.vq_bytes(b)
        n_pos = model.num_spatial_positions
        print(f"  {b:4d}B: {vq_b}B VQ, {n_pos} spatial positions, "
              f"{'fits' if vq_b >= n_pos else 'TRUNCATED — only ' + str(vq_b) + '/' + str(n_pos) + ' positions'}")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training: {args.epochs} epochs")
    print("=" * 70)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        entry = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": lr_now,
        }
        history.append(entry)

        # Console output
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss={train_metrics['loss']:.4f} | "
            f"Pixel={train_metrics['pixel']:.4f} | "
            f"VQ={train_metrics['vq']:.4f} | "
            f"Perplexity={train_metrics['perplexity']:.1f} | "
            f"Val={val_metrics['val_pixel_loss']:.4f} | "
            f"LR={lr_now:.6f} | {elapsed:.1f}s"
        )

        # Early stopping on val loss
        val_loss = val_metrics["val_pixel_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            model.save(args.output_dir / "best.pt")
        else:
            patience_counter += 1

        # Periodic save + PSNR check
        if epoch % args.save_every == 0:
            model.save(args.output_dir / f"epoch_{epoch:03d}.pt")
            if val_loader and len(val_ds) > 0:
                psnr = compute_psnr_sample(model, val_loader, device)
                print(f"  → Val PSNR (sample): {psnr:.2f} dB")
                entry["val_psnr"] = psnr

        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping: val loss did not improve for {args.early_stop_patience} epochs.")
            break

    # Final save
    model.save(args.output_dir / "final.pt")

    # Save loss history
    import json
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best val pixel loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {args.output_dir}/best.pt")
    print(f"  History saved to: {args.output_dir}/training_history.json")
    print("\nNEXT: Run evaluation:")
    print(f"  python scripts/evaluate.py --model {args.output_dir}/best.pt --data-dir <path>")


if __name__ == "__main__":
    main()
