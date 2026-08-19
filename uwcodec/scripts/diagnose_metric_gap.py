"""Diagnostic: verify train-forward vs. real encode→decode payload metric discrepancy.

This is the FIRST step before any redesign. We measure:
  1. model.forward(images) → direct PSNR (what training sees)
  2. image → encode() → bytes → decode() → PSNR (what real deployment sees)

These MUST agree. If they differ, we have a bug in the eval pipeline.
Run BEFORE claiming any PSNR numbers.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to best.pt checkpoint")
    p.add_argument("--datasets-root", type=Path, default=Path("s:/IMG_compressors/datasets"))
    p.add_argument("--dataset", type=str, default="euvp")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--save-grid", type=Path, default=Path("outputs/diagnostic_grid.png"))
    return p.parse_args()


def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0**2 / mse))


def main():
    args = parse_args()

    from uwcodec.codecs.vqvae_codec import MinimalVQVAE
    from uwcodec.data.dataset import MultiDatasetLoader

    print(f"Loading model from {args.model}")
    model = MinimalVQVAE.load(args.model)
    model.eval()
    device = "cpu"

    print(f"  target_vq_tokens = {model.target_vq_tokens}")
    print(f"  spatial_h={model.spatial_h}, spatial_w={model.spatial_w}")
    print(f"  num_spatial_positions = {model.num_spatial_positions}")

    loader = MultiDatasetLoader(args.datasets_root)
    ds = loader.get_dataset(args.dataset, split="val", input_size=128, augment=False)
    n = min(args.num_images, len(ds))

    psnr_forward = []  # what training sees: model.forward() without any serialization
    psnr_payload  = []  # what deployment sees: encode→bytes→decode

    print(f"\nRunning diagnostic on {n} images from {args.dataset.upper()} val set...")
    print(f"Budget: {args.budget}B\n")

    grid_originals = []
    grid_forward   = []
    grid_payload   = []

    for i in range(n):
        item = ds[i]
        x_t = item["image"].unsqueeze(0).to(device)  # (1, 3, 128, 128)

        # --- Path 1: Direct forward (what training loss uses) ---
        with torch.no_grad():
            if model.budget_proj_enc is not None:
                B, D, H, W = model.encoder(x_t).shape
                z = model.encoder(x_t)
                z_flat = z.view(B, D, H * W)
                z_proj = model.budget_proj_enc(z_flat)
                z_for_vq = z_proj.unsqueeze(-1)
            else:
                z = model.encoder(x_t)
                z_for_vq = z
                B, D, H, W = z.shape

            z_q, vq_info = model.quantizer(z_for_vq)

            if model.budget_proj_dec is not None:
                z_q_flat = z_q.squeeze(-1)
                z_q_spatial = model.budget_proj_dec(z_q_flat)
                z_q_dec = z_q_spatial.view(B, D, model.spatial_h, model.spatial_w)
            else:
                z_q_dec = z_q

            recon_fwd = model.decoder(z_q_dec)

        orig_np = (x_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        recon_fwd_np = (recon_fwd[0].permute(1, 2, 0).cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8)
        p_fwd = psnr(orig_np, recon_fwd_np)
        psnr_forward.append(p_fwd)

        # --- Path 2: Full encode→payload→decode (what deployment uses) ---
        payload = model.encode(orig_np, args.budget)
        recon_payload_np = model.decode(payload)
        p_pay = psnr(orig_np, recon_payload_np)
        psnr_payload.append(p_pay)

        if i < 4:
            grid_originals.append(orig_np)
            grid_forward.append(recon_fwd_np)
            grid_payload.append(recon_payload_np)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{n}] forward={p_fwd:6.2f}dB  payload={p_pay:6.2f}dB  diff={p_pay-p_fwd:+.2f}dB")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Direct forward PSNR (training path):  {np.mean(psnr_forward):6.2f} ± {np.std(psnr_forward):.2f} dB")
    print(f"  Full payload PSNR  (deploy path):     {np.mean(psnr_payload):6.2f} ± {np.std(psnr_payload):.2f} dB")
    print(f"  Mean discrepancy:                     {np.mean(np.array(psnr_payload) - np.array(psnr_forward)):+.2f} dB")

    if abs(np.mean(psnr_payload) - np.mean(psnr_forward)) > 1.0:
        print("\n[WARNING] >1 dB discrepancy between forward and payload paths!")
        print("  This means eval metrics are NOT measuring what training optimizes.")
        print("  The train/eval mismatch is CONFIRMED. Must fix before longer training.")
    else:
        print("\n[OK] Forward and payload paths agree within 1 dB.")

    # Save visual grid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_grid = len(grid_originals)
        fig, axes = plt.subplots(n_grid, 3, figsize=(9, n_grid * 3))
        if n_grid == 1:
            axes = [axes]

        for r in range(n_grid):
            axes[r][0].imshow(grid_originals[r]); axes[r][0].set_title("Original"); axes[r][0].axis("off")
            p_f = psnr(grid_originals[r], grid_forward[r])
            axes[r][1].imshow(grid_forward[r]); axes[r][1].set_title(f"Forward {p_f:.1f}dB"); axes[r][1].axis("off")
            p_p = psnr(grid_originals[r], grid_payload[r])
            axes[r][2].imshow(grid_payload[r]); axes[r][2].set_title(f"Payload {p_p:.1f}dB"); axes[r][2].axis("off")

        plt.suptitle(f"Diagnostic: {args.budget}B payload | Forward vs Payload paths", fontsize=12)
        plt.tight_layout()
        args.save_grid.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_grid, dpi=150)
        print(f"\nGrid saved to {args.save_grid}")
        plt.close()
    except Exception as e:
        print(f"  (Could not save grid: {e})")


if __name__ == "__main__":
    main()
