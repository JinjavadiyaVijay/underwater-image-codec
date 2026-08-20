"""Evaluation script for UWCodec v2 across test datasets."""
import argparse
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.data.dataset import MultiDatasetLoader
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim, compute_uciqe

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--datasets-root", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--dataset", default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, help="JSON output path")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda") else "cpu")
    
    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.dataset} | Budget: {args.budget}B | Device: {device}")
    
    # Load model
    codec = UWCodecV2.load(args.model, device=device)
    codec.eval()
    
    # Load data
    loader_obj = MultiDatasetLoader(args.datasets_root)
    # the multi dataset loader get_dataset method ignores split for uieb and uses "images"
    ds = loader_obj.get_dataset(args.dataset, split="val", augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    psnrs, ssims, uciqes, dists_scores = [], [], [], []
    
    # Initialize DISTS
    try:
        from DISTS_pytorch import DISTS
        dists_model = DISTS().to(device)
    except ImportError:
        dists_model = None

    for batch in tqdm(loader, desc="Evaluating"):
        x_np = batch[0]

        if isinstance(x_np, torch.Tensor):
            x = x_np[0].cpu().numpy()
        elif isinstance(x_np, np.ndarray):
            x = x_np[0] if x_np.ndim == 4 else x_np
        elif isinstance(x_np, (str, Path)):
            from PIL import Image
            x = np.array(Image.open(x_np).convert("RGB"))
        else:
            raise TypeError(f"Unsupported dataset image type: {type(x_np)}")
        
        # Encode -> payload -> Decode
        payload = codec.encode(x)
        recon = codec.decode(payload.raw_bytes)
        
        # Metrics
        psnrs.append(image_psnr(x, recon))
        ssims.append(image_ssim(x, recon))
        uciqes.append(image_uciqe(recon))
        
        if dists_model is not None:
            # DISTS requires torch tensors (B, C, H, W) in [0, 1]
            x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            recon_t = torch.from_numpy(recon).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            score = dists_model(x_t, recon_t).item()
            dists_scores.append(score)
        
    res = {
        "dataset": args.dataset,
        "budget": args.budget,
        "model": str(args.model),
        "metrics": {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "uciqe": float(np.mean(uciqes)),
        }
    }
    
    if dists_scores:
        res["metrics"]["dists"] = float(np.mean(dists_scores))
    
    print("\nResults:")
    print(json.dumps(res, indent=2))
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
            
if __name__ == "__main__":
    main()
