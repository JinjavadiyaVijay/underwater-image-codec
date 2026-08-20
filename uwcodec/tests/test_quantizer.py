import torch
import pytest
from uwcodec.models.quantizer import VectorQuantizer, ProductQuantizer, ResidualVQ

def test_vector_quantizer_ema_and_dead_code():
    vq = VectorQuantizer(codebook_size=10, codebook_dim=16, decay=0.5)
    
    # 1. embedding.weight receives no gradient
    assert vq.embedding.weight.requires_grad == False
    
    # Simulate dead code revival
    vq.train()
    
    z = torch.randn(2, 16, 4, 4, requires_grad=True) # Batch 2, Dim 16
    
    # Run once
    z_q, info = vq(z)
    assert not vq.embedding.weight.requires_grad
    
    # Check that backward doesn't fail and doesn't populate embedding.weight.grad
    loss = z_q.sum() + info["vq_loss"]
    loss.backward()
    
    assert z.grad is not None
    assert vq.embedding.weight.grad is None
    
    # 2. Check EMA changes codebook weights and dead codes are revived
    # With a batch of 2*4*4=32 vectors and 10 codebook entries, some might be dead if initialization was bad.
    # But let's force a dead code situation.
    
    vq._ema_cluster_size.zero_() # force dead codes
    
    z2 = torch.randn(2, 16, 4, 4)
    z_q2, info2 = vq(z2)
    
    # After forward, dead codes should be revived. 
    # Because we forced cluster size to 0, all 10 codes were dead and got replaced.
    assert (vq._ema_cluster_size >= 1.0).all(), "Dead codes were not revived"
    
    assert "active_codes" in info2
    assert "perplexity" in info2
    
def test_residual_vq():
    rvq = ResidualVQ(codebook_size=10, codebook_dim=16, num_levels=2)
    
    z = torch.randn(2, 16, 4, 4)
    z_q, info = rvq(z)
    
    assert z_q.shape == z.shape
    assert "active_codes" in info
    assert "perplexity" in info

def test_product_vq():
    pvq = ProductQuantizer(codebook_size=10, codebook_dim=16, num_groups=2)
    
    z = torch.randn(2, 16, 4, 4)
    z_q, info = pvq(z)
    
    assert z_q.shape == z.shape
    assert "active_codes" in info
    assert "perplexity" in info

if __name__ == "__main__":
    pytest.main([__file__])
