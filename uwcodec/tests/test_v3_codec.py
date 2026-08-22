"""Unit tests for the V3 TiTok-style Codec."""

import numpy as np
import pytest
import torch

from uwcodec.codecs.v3_codec import UWCodecV3
from uwcodec.core.payload import EncodedPayload

@pytest.fixture
def codec():
    return UWCodecV3(
        input_size=128,
        embed_dim=128,          # smaller for faster testing
        num_latent_tokens=64,
        codebook_size=4096,
        encoder_depth=2,
        decoder_depth=2,
    )

def test_v3_forward(codec):
    """Test standard forward pass during training."""
    x = torch.rand(2, 3, 128, 128)
    recon, info = codec(x)
    
    assert recon.shape == (2, 3, 128, 128)
    assert "vq_loss" in info
    assert "indices" in info
    assert info["indices"].shape == (2, 64)

def test_v3_encode_decode(codec):
    """Test full 128B encode/decode serialization path."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    payload = codec.encode(img, max_bytes=128)
    
    assert isinstance(payload, EncodedPayload)
    assert len(payload.raw_bytes) == 128
    
    recon = codec.decode(payload)
    
    assert isinstance(recon, np.ndarray)
    assert recon.shape == (128, 128, 3)
    assert recon.dtype == np.uint8

def test_v3_packing(codec):
    """Test 12-bit packing logic."""
    # Max size
    indices = torch.randint(0, 4096, (64,), dtype=torch.long)
    
    packed = codec.pack_12bit(indices)
    assert len(packed) == 96
    
    unpacked = codec.unpack_12bit(packed)
    assert unpacked.shape == (64,)
    
    assert torch.equal(indices, unpacked)
