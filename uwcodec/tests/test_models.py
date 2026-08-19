"""Tests for learned model components: encoder, decoder, quantizer."""

import pytest
import torch

from uwcodec.models.encoder import AppearanceEncoder
from uwcodec.models.decoder import ImageDecoder
from uwcodec.models.quantizer import VectorQuantizer, ProductQuantizer, ResidualVQ


class TestAppearanceEncoder:
    @pytest.mark.parametrize("input_size", [64, 96, 128])
    def test_forward_shape(self, input_size):
        enc = AppearanceEncoder(input_size=input_size)
        x = torch.randn(2, 3, input_size, input_size)
        z = enc(x)
        assert z.dim() == 4
        assert z.shape[0] == 2
        assert z.shape[1] == 64  # latent_dim

    def test_compute_output_shape(self):
        enc = AppearanceEncoder(input_size=128)
        h, w = enc.compute_output_shape(128)
        assert h > 0 and w > 0
        assert h == w  # square input → square output

    def test_param_count_reasonable(self):
        enc = AppearanceEncoder()
        params = enc.count_parameters()
        assert 10_000 < params < 5_000_000  # Not too tiny, not too large


class TestVectorQuantizer:
    def test_forward_shape(self):
        vq = VectorQuantizer(codebook_size=256, codebook_dim=64)
        z = torch.randn(2, 64, 4, 4)
        z_q, info = vq(z)
        assert z_q.shape == z.shape
        assert info["indices"].shape == (2, 4, 4)

    def test_perplexity_positive(self):
        vq = VectorQuantizer(codebook_size=256, codebook_dim=64)
        z = torch.randn(4, 64, 4, 4)
        _, info = vq(z)
        assert info["perplexity"] > 0


class TestProductQuantizer:
    def test_forward_shape(self):
        pq = ProductQuantizer(codebook_size=256, codebook_dim=64, num_groups=4)
        z = torch.randn(2, 64, 4, 4)
        z_q, info = pq(z)
        assert z_q.shape == z.shape
        assert len(info["indices"]) == 4


class TestResidualVQ:
    def test_forward_shape(self):
        rvq = ResidualVQ(codebook_size=256, codebook_dim=64, num_levels=2)
        z = torch.randn(2, 64, 4, 4)
        z_q, info = rvq(z)
        assert z_q.shape == z.shape
        assert len(info["indices"]) == 2


class TestImageDecoder:
    def test_forward_shape(self):
        dec = ImageDecoder(latent_dim=64, output_size=128)
        z_q = torch.randn(2, 64, 8, 8)
        recon = dec(z_q)
        assert recon.shape == (2, 3, 128, 128)
        assert (recon >= 0).all()
        assert (recon <= 1).all()
