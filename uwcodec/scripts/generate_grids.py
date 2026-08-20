"""Generate visual comparison grids for UWCodec v2 evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.data.dataset import MultiDatasetLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--datasets-root", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--dataset", default="euvp", choices=["euvp", "suim", "uieb"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, required=True, help="Output image file (e.g., grid.png)")
    p.add_argument("--num-images", type=int, default=16, help="Number of images to include in the grid")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda") else "cpu")
    
    print(f"Loading model: {args.model}")
    codec = UWCodecV2.load(args.model, device=device)
    codec.eval()
    
    loader_obj = MultiDatasetLoader(args.datasets_root)
    ds = loader_obj.get_dataset(args.dataset, split="val", augment=False)
    
    # Shuffle or just take the first N
    # We will just take the first N for reproducibility on the test set
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    pairs = []
    
    for i, batch in enumerate(loader):
        if i >= args.num_images:
            break
            
        x_t = batch["image"]
        if x_t.ndim == 4: x_t = x_t[0]
        x = (x_t.permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        
        # Encode -> payload -> Decode (The EXACT evaluated path)
        payload = codec.encode(x)
        assert len(payload) == args.budget, f"Payload is {len(payload)}B, expected {args.budget}B"
        recon = codec.decode(payload)
        
        pairs.append((x, recon))
        
    if not pairs:
        print("No images found.")
        return
        
    # Build grid
    # pairs[i][0] is original, pairs[i][1] is recon
    # Let's stack them horizontally (orig | recon)
    # And vertically for multiple images
    
    H, W, _ = pairs[0][0].shape
    
    # Grid dimensions
    cols = 4 # 4 pairs per row
    rows = int(np.ceil(len(pairs) / cols))
    
    # Add a border between orig and recon, and between pairs
    border_px = 4
    pair_w = W * 2 + border_px
    pair_h = H
    
    grid_w = cols * pair_w + (cols - 1) * border_px
    grid_h = rows * pair_h + (rows - 1) * border_px
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    
    for i, (orig, recon) in enumerate(pairs):
        r = i // cols
        c = i % cols
        
        y_offset = r * (pair_h + border_px)
        x_offset = c * (pair_w + border_px)
        
        img_o = Image.fromarray(orig)
        img_r = Image.fromarray(recon)
        
        grid_img.paste(img_o, (x_offset, y_offset))
        grid_img.paste(img_r, (x_offset + W + border_px, y_offset))
        
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(args.output)
    print(f"Grid saved to {args.output}")

if __name__ == "__main__":
    main()
