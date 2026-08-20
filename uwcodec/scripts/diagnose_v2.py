import torch
from pathlib import Path
import numpy as np
from PIL import Image
from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.data.dataset import MultiDatasetLoader
import sys

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path("outputs/v2/budget_128_final/best.pt")
    if not model_path.exists():
        print("Model not found")
        return
        
    codec = UWCodecV2.load(model_path, device=device)
    codec.eval()
    
    loader_obj = MultiDatasetLoader("S:/IMG_compressors/datasets")
    ds = loader_obj.get_dataset("euvp", split="val", augment=False)
    
    # Get 1 image
    batch = ds[0]
    x_t = batch["image"].unsqueeze(0).to(device)  # (1, 3, H, W) in [0,1]
    
    print("--- 1. Direct Forward ---")
    with torch.no_grad():
        out = codec(x_t)
        recon_direct = out["reconstruction"]
        print(f"sem_vq_loss: {out['sem_vq_loss'].item():.4f}")
        print(f"det_vq_loss: {out['det_vq_loss'].item():.4f}")
        print(f"sem_perplexity: {out['sem_perplexity'].item():.4f}")
        print(f"det_perplexity: {out['det_perplexity'].item():.4f}")
        
    print("\n--- 2. Serialize / Deserialize Path ---")
    # x is numpy uint8 RGB
    x_np = (x_t[0].permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    
    payload = codec.encode(x_np)
    print(f"Payload length: {len(payload)}B")
    
    recon_serialized_np = codec.decode(payload)
    
    # Convert recon_direct to numpy for comparison
    recon_direct_np = codec._postprocess(recon_direct)
    
    diff = np.abs(recon_direct_np.astype(int) - recon_serialized_np.astype(int))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max pixel difference between forward() and encode->decode: {max_diff}")
    print(f"Mean pixel difference: {mean_diff:.4f}")
    
    print("\n--- 3. Check Spatial Ordering & Padding ---")
    print("Semantic positions:", 4*4)
    print("Detail L1 token allocation:", codec.det_l1_tokens)
    print("Detail L2 token allocation:", codec.det_l2_tokens)
    
    # Manually check encode output
    sem_z = codec.sem_encoder(x_t)
    sem_q, sem_info = codec.sem_rvq(sem_z)
    
    det_z = codec.det_encoder(x_t)
    det_q1, det_info1 = codec.det_vq1(det_z)
    
    print("\nSemantic indices shape (per level):", [idx.shape for idx in sem_info["indices"]])
    print("Detail L1 indices shape:", det_info1["indices"].shape)
    
    if codec.det_vq2:
        det_residual = det_z - det_q1
        det_q2, det_info2 = codec.det_vq2(det_residual)
        print("Detail L2 indices shape:", det_info2["indices"].shape)
        
    print("Done")

if __name__ == '__main__':
    main()
