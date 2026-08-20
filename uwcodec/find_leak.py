import torch
import torch.nn.functional as F
from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.training.train_v2 import compute_loss
import argparse
import gc
import sys

device = torch.device('cpu')
model = UWCodecV2(budget=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
args = argparse.Namespace(
    lambda_l1=1.0, lambda_msssim=0.84, lambda_lpips=0.1,
    lambda_sem_vq=0.25, lambda_det_vq=0.25
)

def track_mem(prefix=""):
    import psutil, os
    process = psutil.Process(os.getpid())
    print(f"{prefix} Alloc: {process.memory_info().rss/1e6:.1f} MB")

history = []

for epoch in range(5):
    totals = {}
    for i in range(20):
        images = torch.rand(8, 3, 128, 128, device=device)
        out = model(images)
        loss, bd = compute_loss(out['reconstruction'], images, out['sem_vq_loss'], out['det_vq_loss'], args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v
            
        del images, out, loss, bd
        
    train_metrics = {k: float(v) / 20 for k, v in totals.items()}
        
    # Simulating metrics collection
    row = {
        "epoch": epoch,
        "train_loss": train_metrics.get("total", 0),
        "sem_perp": train_metrics.get("sem_vq", 0)
    }
    history.append(row)
    
    gc.collect()
    track_mem(f"Epoch {epoch} End:")
