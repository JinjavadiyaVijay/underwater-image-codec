"""UWCodec v3 Training Script (TiTok-style 1D tokenizer).

Loss: L1(pixel) + MS-SSIM + optional LPIPS + VQ commitment.
Evaluation: full encode→payload→decode path (no metric mismatch possible).

Usage:
    python -m scripts.train_v3 \
        --dataset euvp --datasets-root datasets \
        --epochs 100 --batch-size 16 \
        --output-dir outputs/v3/run1
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
from torch.utils.data import DataLoader

# Optional dependencies
try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    HAS_MSSSIM = True
except ImportError:
    raise RuntimeError("pytorch_msssim is required. Run: pip install pytorch-msssim")

try:
    import lpips as _lpips_lib
    _LPIPS_MODEL: Any = None
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    _LPIPS_MODEL = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UWCodec v3")

    # Dataset
    p.add_argument("--dataset", default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--num-images",    type=int, default=None)
    p.add_argument("--input-size",    type=int, default=128)

    # Model
    p.add_argument("--embed-dim",         type=int, default=256)
    p.add_argument("--num-latent-tokens", type=int, default=64)
    p.add_argument("--codebook-size",     type=int, default=4096)
    p.add_argument("--encoder-depth",     type=int, default=6)
    p.add_argument("--decoder-depth",     type=int, default=6)

    # Training
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs",type=int, default=5)

    # Loss weights
    p.add_argument("--lambda-l1",      type=float, default=1.0)
    p.add_argument("--lambda-msssim",  type=float, default=0.84)
    p.add_argument("--lambda-lpips",   type=float, default=0.1)
    p.add_argument("--lambda-vq",      type=float, default=0.25)

    # Infrastructure
    p.add_argument("--device",       type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--output-dir",   type=Path, default=Path("outputs/v3/default"))
    p.add_argument("--eval-every",   type=int, default=5)
    p.add_argument("--save-every",   type=int, default=10)
    p.add_argument("--resume",       type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print(f"Device: {device}")
    print("=" * 60)
    return device


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _get_lpips_model(device: torch.device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None and HAS_LPIPS:
        _LPIPS_MODEL = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad = False
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    vq_loss: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    breakdown: dict[str, torch.Tensor] = {}

    l1 = F.l1_loss(recon, target)
    total = args.lambda_l1 * l1
    breakdown["l1"] = l1.item()

    if HAS_MSSSIM and args.lambda_msssim > 0:
        ms = _ms_ssim(recon, target, data_range=1.0, size_average=True, win_size=7)
        ms_loss = 1.0 - ms
        total = total + args.lambda_msssim * ms_loss
        breakdown["ms_ssim_loss"] = ms_loss.item()

    if HAS_LPIPS and args.lambda_lpips > 0:
        lpips_model = _get_lpips_model(device)
        r_lpips = recon * 2.0 - 1.0
        t_lpips = target * 2.0 - 1.0
        with torch.enable_grad():
            lp = lpips_model(r_lpips, t_lpips).mean()
        total = total + args.lambda_lpips * lp
        breakdown["lpips"] = lp.item()

    total = total + args.lambda_vq * vq_loss
    breakdown["vq"] = vq_loss.item()
    breakdown["total"] = total.item()

    return total, breakdown


def psnr_batch(recon: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return float(psnr.mean())


# ---------------------------------------------------------------------------
# Epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: "UWCodecV3",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
            recon, info = model(images)
            loss, breakdown = compute_loss(recon, images, info["vq_loss"], args, device)
            
            breakdown["perplexity"] = info["perplexity"].item()
            breakdown["active_codes"] = info["active_codes"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        del images, recon, info, loss, breakdown

    elapsed = time.time() - t0
    return {k: float(v) / max(n_batches, 1) for k, v in totals.items()} | {"epoch_secs": elapsed}


@torch.no_grad()
def validate(model: "UWCodecV3", loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_psnr = 0.0
    total_l1 = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        recon, _ = model(images)
        total_psnr += psnr_batch(recon, images)
        total_l1 += float(F.l1_loss(recon, images))
        n += 1

    n = max(n, 1)
    return {"val_psnr": total_psnr / n, "val_l1": total_l1 / n}


@torch.no_grad()
def validate_payload_psnr(model: "UWCodecV3", loader: DataLoader, num_samples: int = 20) -> float:
    from uwcodec.evaluation.metrics import compute_psnr

    model.eval()
    psnrs: list[float] = []

    for batch in loader:
        for img_t in batch["image"]:
            if len(psnrs) >= num_samples:
                break
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            try:
                # Encode handles conversion internally
                payload = model.encode(img_np, max_bytes=128)
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

    from uwcodec.codecs.v3_codec import UWCodecV3
    from uwcodec.data.dataset import MultiDatasetLoader

    device = setup_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader_obj = MultiDatasetLoader(args.datasets_root)
    train_ds = loader_obj.get_dataset(args.dataset, split="train", input_size=args.input_size, augment=True)
    val_ds = loader_obj.get_dataset(args.dataset, split="val", input_size=args.input_size, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = UWCodecV3(
        input_size=args.input_size,
        embed_dim=args.embed_dim,
        num_latent_tokens=args.num_latent_tokens,
        codebook_size=args.codebook_size,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_psnr = 0.0
    start_epoch = 1
    history: list[dict] = []

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            if "train_state" in ckpt:
                ts = ckpt["train_state"]
                optimizer.load_state_dict(ts["optimizer"])
                scheduler.load_state_dict(ts["scheduler"])
                warmup.load_state_dict(ts["warmup"])
                scaler.load_state_dict(ts["scaler"])
                start_epoch = ts["epoch"] + 1
                best_val_psnr = ts.get("best_val_psnr", 0.0)
                history = ts.get("history", [])
                print(f"Resumed from epoch {start_epoch} (Best PSNR: {best_val_psnr:.2f})")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            optimizer.step()
            warmup.step()
        else:
            scheduler.step()

        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_metrics = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("total", 0),
            "train_l1": train_metrics.get("l1", 0),
            "perplexity": train_metrics.get("perplexity", 0),
            "active_codes": train_metrics.get("active_codes", 0),
            **val_metrics,
            "lr": lr,
            "secs": train_metrics.get("epoch_secs", 0),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss={row['train_loss']:.4f} | "
            f"Val_PSNR={val_metrics['val_psnr']:.2f}dB | "
            f"Perp={row['perplexity']:.1f} | "
            f"Act={row['active_codes']:.1f} | "
            f"LR={lr:.2e} | "
            f"{row['secs']:.1f}s"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            payload_psnr = validate_payload_psnr(model, val_loader, num_samples=20)
            print(f"  [Payload PSNR on 20 samples: {payload_psnr:.2f} dB]")
            row["payload_psnr"] = payload_psnr

        def _save_ckpt(path: Path):
            train_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "warmup": warmup.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_psnr": best_val_psnr,
                "history": history,
            }
            model.save(path, train_state=train_state)

        is_best = val_metrics["val_psnr"] > best_val_psnr
        if is_best:
            best_val_psnr = val_metrics["val_psnr"]
            _save_ckpt(args.output_dir / "best.pt")

        if epoch % args.save_every == 0:
            _save_ckpt(args.output_dir / f"epoch_{epoch:03d}.pt")

    _save_ckpt(args.output_dir / "final.pt")

    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTRAINING COMPLETE")
    print(f"Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
