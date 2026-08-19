"""UWCodec v2 Training Script.

Loss: L1(pixel) + MS-SSIM + optional LPIPS + VQ commitment (semantic + detail).
Evaluation: full encode→payload→decode path (no metric mismatch possible).

Usage:
    python -m uwcodec.training.train_v2 \
        --dataset euvp --datasets-root datasets \
        --budget 128 --epochs 50 --batch-size 16 \
        --output-dir outputs/v2/budget_128

Optional dependencies:
    pip install lpips pytorch-msssim
    (Both degrade gracefully to L1+SSIM if unavailable.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# ---- Optional dependencies ----
try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    HAS_MSSSIM = True
except ImportError:
    raise RuntimeError("pytorch_msssim is required for training v2. Please run: pip install pytorch-msssim")

try:
    import lpips as _lpips_lib
    _LPIPS_MODEL: Any = None  # lazy-loaded
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    _LPIPS_MODEL = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UWCodec v2")

    # Dataset
    p.add_argument("--dataset", default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--num-images",    type=int, default=None,
                   help="Limit training images (None = use all)")
    p.add_argument("--input-size",    type=int, default=128)

    # Model
    p.add_argument("--budget",       type=int, default=128, choices=[64, 96, 124, 128])
    p.add_argument("--sem-dim",      type=int, default=64)
    p.add_argument("--det-dim",      type=int, default=32)
    p.add_argument("--decoder-channels", type=int, default=256,
                   help="Base channels in V2Decoder (256=full, 128=lite)")
    p.add_argument("--res-bottom",   type=int, default=4,
                   help="Residual blocks at 4×4 (deepest)")
    p.add_argument("--res-mid",      type=int, default=2,
                   help="Residual blocks at each intermediate scale")

    # Training
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs",type=int, default=3)

    # Loss weights
    p.add_argument("--lambda-l1",      type=float, default=1.0)
    p.add_argument("--lambda-msssim",  type=float, default=0.84)
    p.add_argument("--lambda-lpips",   type=float, default=0.1)
    p.add_argument("--lambda-sem-vq",  type=float, default=0.25)
    p.add_argument("--lambda-det-vq",  type=float, default=0.25)

    # Infrastructure
    p.add_argument("--device",       type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers",  type=int, default=0)
    p.add_argument("--output-dir",   type=Path, default=Path("outputs/v2/budget_128"))
    p.add_argument("--eval-every",   type=int, default=5)
    p.add_argument("--save-every",   type=int, default=10)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device(requested: str) -> torch.device:
    """Resolve device, fail fast if CUDA explicitly requested but unavailable."""
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is not available.\n"
                "Check your PyTorch installation. See docs/GPU_SETUP.md."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("DEVICE DIAGNOSTICS")
    print("=" * 60)
    if device.type == "cuda":
        print(f"Device:      CUDA — {torch.cuda.get_device_name(0)}")
        print(f"CUDA:        {torch.version.cuda}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:        {mem_gb:.1f} GB")
    else:
        print("Device:      CPU")
        print("WARNING: CPU training will be slow. See docs/GPU_SETUP.md.")
    print("=" * 60)
    return device


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _get_lpips_model(device: torch.device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None and HAS_LPIPS:
        _LPIPS_MODEL = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    sem_vq_loss: torch.Tensor,
    det_vq_loss: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the total v2 training loss.

    Returns (total_loss, breakdown_dict).
    """
    breakdown: dict[str, float] = {}

    # L1 pixel loss
    l1 = F.l1_loss(recon, target)
    total = args.lambda_l1 * l1
    breakdown["l1"] = float(l1.detach())

    # MS-SSIM (minimize 1 - MS_SSIM since MS_SSIM ∈ [0,1], higher is better)
    if HAS_MSSSIM and args.lambda_msssim > 0:
        # ms_ssim requires minimum spatial size; skip if too small
        if recon.shape[-1] >= 32:
            ms = _ms_ssim(recon, target, data_range=1.0, size_average=True, win_size=7)
            ms_loss = 1.0 - ms
            total = total + args.lambda_msssim * ms_loss
            breakdown["ms_ssim_loss"] = float(ms_loss.detach())
    else:
        # Fall back to gradient-based structural loss
        dx_r = recon[:, :, :, 1:] - recon[:, :, :, :-1]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_r = recon[:, :, 1:, :] - recon[:, :, :-1, :]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = F.l1_loss(dx_r, dx_t) + F.l1_loss(dy_r, dy_t)
        total = total + args.lambda_msssim * grad_loss
        breakdown["grad_loss"] = float(grad_loss.detach())

    # LPIPS perceptual loss
    if HAS_LPIPS and args.lambda_lpips > 0:
        lpips_model = _get_lpips_model(device)
        # LPIPS expects input in [-1, 1]
        r_lpips = recon  * 2.0 - 1.0
        t_lpips = target * 2.0 - 1.0
        with torch.no_grad() if False else torch.enable_grad():
            lp = lpips_model(r_lpips, t_lpips).mean()
        total = total + args.lambda_lpips * lp
        breakdown["lpips"] = float(lp.detach())

    # VQ commitment losses
    total = total + args.lambda_sem_vq * sem_vq_loss + args.lambda_det_vq * det_vq_loss
    breakdown["sem_vq"] = float(sem_vq_loss.detach())
    breakdown["det_vq"] = float(det_vq_loss.detach())
    breakdown["total"]  = float(total.detach())

    return total, breakdown


# ---------------------------------------------------------------------------
# PSNR util
# ---------------------------------------------------------------------------

def psnr_batch(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Mean PSNR over a batch. Tensors in [0, 1]."""
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=(1, 2, 3))  # (B,)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return float(psnr.mean())


# ---------------------------------------------------------------------------
# Epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: "UWCodecV2",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    t0 = time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(images)
            loss, breakdown = compute_loss(
                out["reconstruction"], images,
                out["sem_vq_loss"], out["det_vq_loss"],
                args, device,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    elapsed = time.time() - t0
    return {k: v / max(n_batches, 1) for k, v in totals.items()} | {"epoch_secs": elapsed}


@torch.no_grad()
def validate(
    model: "UWCodecV2",
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate via direct forward PSNR and pixel loss."""
    model.eval()
    total_psnr = 0.0
    total_l1   = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        total_psnr += psnr_batch(out["reconstruction"], images)
        total_l1   += float(F.l1_loss(out["reconstruction"], images))
        n += 1

    n = max(n, 1)
    return {"val_psnr": total_psnr / n, "val_l1": total_l1 / n}


@torch.no_grad()
def validate_payload_psnr(
    model: "UWCodecV2",
    loader: DataLoader,
    device: torch.device,
    budget: int,
    num_samples: int = 20,
) -> float:
    """Validate via the FULL encode→payload→decode path.

    This measures the REAL deployment PSNR (no metric mismatch).
    Slower (CPU NumPy round-trip) — run every few epochs.
    """
    from uwcodec.evaluation.metrics import compute_psnr

    model.eval()
    psnrs: list[float] = []

    for batch in loader:
        for img_t in batch["image"]:
            if len(psnrs) >= num_samples:
                break
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            try:
                payload = model.encode(img_np, budget)
                recon_np = model.decode(payload)
                p = compute_psnr(img_np, recon_np)
                if not (np.isinf(p) or np.isnan(p)):
                    psnrs.append(p)
            except Exception:
                pass
        if len(psnrs) >= num_samples:
            break

    return float(np.mean(psnrs)) if psnrs else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Need these imports here to avoid circular imports
    from uwcodec.codecs.v2_codec import UWCodecV2
    from uwcodec.data.dataset import MultiDatasetLoader

    device = setup_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    print(f"\nLoading {args.dataset.upper()} from {args.datasets_root}...")
    loader_obj = MultiDatasetLoader(args.datasets_root)

    train_ds = loader_obj.get_dataset(
        args.dataset, split="train", input_size=args.input_size,
        augment=True,
    )
    val_ds = loader_obj.get_dataset(
        args.dataset, split="val", input_size=args.input_size,
        augment=False,
    )

    # Optional image count limit (subsetting for quick experiments)
    if args.num_images is not None and len(train_ds) > args.num_images:
        from torch.utils.data import Subset
        indices = list(range(args.num_images))
        train_ds = Subset(train_ds, indices)
        val_n = max(50, args.num_images // 10)
        val_indices = list(range(min(val_n, len(val_ds))))
        val_ds = Subset(val_ds, val_indices)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Model ----
    model = UWCodecV2(
        budget=args.budget,
        sem_dim=args.sem_dim,
        det_dim=args.det_dim,
        decoder_base_channels=args.decoder_channels,
        num_res_blocks_bottom=args.res_bottom,
        num_res_blocks_mid=args.res_mid,
        input_size=args.input_size,
        output_size=args.input_size,
    ).to(device)

    model.print_summary()

    if not HAS_MSSSIM:
        raise RuntimeError("pytorch_msssim is required but HAS_MSSSIM is False.")
    if not HAS_LPIPS:
        print("[WARN] lpips not installed — LPIPS loss disabled.")
        print("       Install: pip install lpips")

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs,
    )
    from torch.amp import GradScaler
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- Training loop ----
    best_val_psnr = 0.0
    history: list[dict] = []

    print(f"\nStarting training: {args.epochs} epochs | Budget: {args.budget}B")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs:
            optimizer.step()  # step first to avoid skipping lr schedule initial value
            warmup.step()
        else:
            scheduler.step()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch,
        )
        val_metrics = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("total", 0),
            "train_l1": train_metrics.get("l1", 0),
            "sem_perp": train_metrics.get("sem_vq", 0),
            **val_metrics,
            "lr": lr,
            "secs": train_metrics.get("epoch_secs", 0),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss={row['train_loss']:.4f} | "
            f"L1={row['train_l1']:.4f} | "
            f"Val_PSNR={val_metrics['val_psnr']:.2f}dB | "
            f"LR={lr:.2e} | "
            f"{row['secs']:.1f}s"
        )

        # Full payload PSNR every eval_every epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            payload_psnr = validate_payload_psnr(
                model, val_loader, device, args.budget, num_samples=20,
            )
            print(f"  [Payload PSNR on 20 samples: {payload_psnr:.2f} dB]")
            row["payload_psnr"] = payload_psnr

        # Checkpoint
        is_best = val_metrics["val_psnr"] > best_val_psnr
        if is_best:
            best_val_psnr = val_metrics["val_psnr"]
            model.save(args.output_dir / "best.pt")
            print(f"  [Saved best: val_psnr={best_val_psnr:.2f} dB]")

        if epoch % args.save_every == 0:
            model.save(args.output_dir / f"epoch_{epoch:03d}.pt")

    # Final checkpoint
    model.save(args.output_dir / "final.pt")

    # Save history
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best val PSNR:    {best_val_psnr:.2f} dB")
    print(f"  Checkpoint:       {args.output_dir / 'best.pt'}")
    print(f"  History:          {args.output_dir / 'training_history.json'}")
    print("\nNEXT: Run full evaluation:")
    print(f"  python scripts/eval_v2.py --model {args.output_dir}/best.pt \\")
    print(f"      --datasets-root {args.datasets_root} --budget {args.budget}")
    print("=" * 60)


if __name__ == "__main__":
    main()
