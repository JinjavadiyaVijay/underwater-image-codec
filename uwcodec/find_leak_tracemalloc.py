import torch
import gc
import tracemalloc
from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.training.train_v2 import compute_loss
import argparse

device = torch.device('cpu')
model = UWCodecV2(budget=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
args = argparse.Namespace(
    lambda_l1=1.0, lambda_msssim=0.84, lambda_lpips=0.1,
    lambda_sem_vq=0.25, lambda_det_vq=0.25
)

def run_epochs(n_epochs):
    tracemalloc.start()
    
    # run 1 epoch warmup
    for i in range(10):
        images = torch.rand(8, 3, 128, 128, device=device)
        out = model(images)
        loss, bd = compute_loss(out['reconstruction'], images, out['sem_vq_loss'], out['det_vq_loss'], args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        del images, out, loss, bd
        
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    
    for epoch in range(n_epochs):
        totals = {}
        for i in range(10):
            images = torch.rand(8, 3, 128, 128, device=device)
            out = model(images)
            loss, bd = compute_loss(out['reconstruction'], images, out['sem_vq_loss'], out['det_vq_loss'], args, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            for k, v in bd.items():
                totals[k] = totals.get(k, 0.0) + v
                
            del images, out, loss, bd
            
        train_metrics = {k: float(v) / 10 for k, v in totals.items()}
        
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 Memory Differences ]")
    for stat in top_stats[:10]:
        print(stat)

run_epochs(5)
