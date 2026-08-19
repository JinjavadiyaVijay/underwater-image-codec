import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from uwcodec.codecs.vqvae_codec import MinimalVQVAE
from uwcodec.data.dataset import MultiDatasetLoader
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim, compute_uciqe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("outputs/multi_budget"),
                   help="Directory containing budget_64, budget_96, etc.")
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124])
    p.add_argument("--dataset", type=str, default="euvp")
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--num-val", type=int, default=200)
    p.add_argument("--output-img", type=Path, default=Path("outputs/eval_grid.png"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    model.eval()
    psnrs = []
    ssims = []
    uciqes = []
    perplexities = []
    
    for batch in val_loader:
        images = batch["image"].to(device)
        
        # Forward pass to get metrics
        z = model.encoder(images)
        B, D, H, W = z.shape
        if model.budget_proj_enc is not None:
            z_flat = z.view(B, D, H * W)
            z_proj = model.budget_proj_enc(z_flat)
            z_for_vq = z_proj.unsqueeze(-1)
        else:
            z_for_vq = z
            
        z_q, vq_info = model.quantizer(z_for_vq)
        perplexities.append(vq_info["perplexity"].item())
        
        if model.budget_proj_dec is not None:
            z_q_flat = z_q.squeeze(-1)
            z_q_spatial = model.budget_proj_dec(z_q_flat)
            z_q = z_q_spatial.view(B, D, H, W)
            
        recon = model.decoder(z_q)
        
        # Calculate metrics
        for i in range(len(images)):
            orig_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            recon_np = (recon[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            
            p = compute_psnr(orig_np, recon_np)
            s = compute_ssim(orig_np, recon_np)
            u = compute_uciqe(recon_np)
            
            if not np.isinf(p) and not np.isnan(p):
                psnrs.append(p)
            if not np.isnan(s):
                ssims.append(s)
            if not np.isnan(u):
                uciqes.append(u)
                
    return {
        "psnr": np.mean(psnrs) if psnrs else 0.0,
        "ssim": np.mean(ssims) if ssims else 0.0,
        "uciqe": np.mean(uciqes) if uciqes else 0.0,
        "perplexity": np.mean(perplexities) if perplexities else 0.0,
    }


def main():
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    
    # Load dataset
    loader = MultiDatasetLoader(args.datasets_root)
    val_ds = loader.get_dataset(args.dataset, split="val", input_size=128, augment=False)
    
    if len(val_ds) > args.num_val:
        val_ds.paths = val_ds.paths[:args.num_val]
        
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # Keep 4 random images for the grid
    np.random.seed(42)
    grid_indices = np.random.choice(len(val_ds), min(4, len(val_ds)), replace=False)
    grid_images = [val_ds[i]["image"] for i in grid_indices]
    grid_images_t = torch.stack(grid_images).to(device)
    
    # Original images for grid
    grid_np_orig = [(img.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8) for img in grid_images_t]
    
    results = {}
    reconstructions = {b: [] for b in args.budgets}
    
    print("=" * 60)
    print(f"EVALUATION GATE: {args.dataset.upper()} VAL SET ({len(val_ds)} images)")
    print("=" * 60)
    
    for budget in args.budgets:
        ckpt_path = args.model_dir / f"budget_{budget}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[{budget}B] SKIP: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"Evaluating {budget}B model...")
        model = MinimalVQVAE.load(ckpt_path).to(device)
        model.eval()
        
        # Get metrics
        metrics = evaluate_model(model, val_loader, device)
        results[budget] = metrics
        
        # Get reconstructions for grid
        with torch.no_grad():
            out = model(grid_images_t)
            recon = out["reconstruction"]
            for i in range(len(recon)):
                r_np = (recon[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                reconstructions[budget].append(r_np)
                
    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Budget':>8} | {'PSNR':>6} | {'SSIM':>6} | {'UCIQE':>6} | {'Perplex':>7}")
    print("-" * 60)
    for budget in args.budgets:
        if budget in results:
            m = results[budget]
            print(f"{budget:>6} B | {m['psnr']:6.2f} | {m['ssim']:6.3f} | {m['uciqe']:6.2f} | {m['perplexity']:7.1f}")
        else:
            print(f"{budget:>6} B | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>7}")
            
    # Generate visual grid
    valid_budgets = [b for b in args.budgets if b in results]
    if not valid_budgets:
        print("No models evaluated, skipping visual grid.")
        return
        
    num_cols = 1 + len(valid_budgets)
    num_rows = len(grid_indices)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    for r in range(num_rows):
        # Original
        ax = axes[r, 0] if num_rows > 1 else axes[0]
        ax.imshow(grid_np_orig[r])
        ax.axis('off')
        if r == 0:
            ax.set_title("Original")
            
        # Reconstructions
        for c, budget in enumerate(valid_budgets):
            ax = axes[r, c+1] if num_rows > 1 else axes[c+1]
            ax.imshow(reconstructions[budget][r])
            ax.axis('off')
            if r == 0:
                ax.set_title(f"{budget}B")
                
    plt.tight_layout()
    args.output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_img)
    print(f"\nVisual grid saved to {args.output_img}")
    
if __name__ == "__main__":
    main()
