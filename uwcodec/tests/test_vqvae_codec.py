"""Tests for the minimal VQ-VAE codec end-to-end."""

import pytest
import torch
import numpy as np

from uwcodec.codecs.vqvae_codec import MinimalVQVAE


class TestMinimalVQVAE:
    @pytest.fixture
    def codec(self):
        # Create a small version for fast tests
        return MinimalVQVAE(
            input_size=64,
            output_size=64,
            encoder_channels=[16, 32],
            decoder_channels=[32, 16],
            latent_dim=16,
            codebook_size=64,
        )

    def test_forward_pass(self, codec):
        images = torch.randn(2, 3, 64, 64).clamp(0, 1)
        out = codec(images)
        assert out["reconstruction"].shape == images.shape
        assert "vq_loss" in out
        assert "perplexity" in out

    def test_encode(self, codec):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        payload = codec.encode(image, max_bytes=124)
        assert len(payload.raw_bytes) == 124
        assert payload.max_bytes == 124

    @pytest.mark.parametrize("budget", [64, 96, 124])
    def test_encode_decode_roundtrip(self, codec, budget):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        payload = codec.encode(image, max_bytes=budget)
        recon = codec.decode(payload)
        assert recon.shape == (64, 64, 3)
        assert recon.dtype == np.uint8

    def test_gradient_flow(self, codec):
        images = torch.randn(2, 3, 64, 64, requires_grad=True).clamp(0, 1)
        # Need to detach and re-create since clamp breaks grad
        images_detached = images.detach().requires_grad_(True)
        out = codec(images_detached)
        loss = out["reconstruction"].mean() + out["vq_loss"]
        loss.backward()
        # Check that encoder has gradients
        assert any(p.grad is not None for p in codec.encoder.parameters())
