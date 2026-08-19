"""Tests for the main UWCodec API (general-purpose codec)."""

import pytest
import numpy as np

from uwcodec.core.config import UWCodecConfig
from uwcodec.core.codec import UWCodec


class TestUWCodec:
    @pytest.fixture
    def codec(self):
        config = UWCodecConfig()
        config.model.input_size = 64
        config.model.output_size = 64
        config.model.encoder_channels = [16, 32]
        config.model.decoder_channels = [32, 16]
        config.model.codebook_size = 64
        # Use CPU for tests
        return UWCodec.from_config(config, device="cpu")

    def test_encode_hard_limit(self, codec):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test across multiple budgets
        for budget in [64, 96, 124, 256, 512]:
            payload = codec.encode(image, max_bytes=budget)
            assert isinstance(payload, bytes)
            assert len(payload) == budget, f"Budget {budget} violated: got {len(payload)}"

    def test_decode(self, codec):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        payload = codec.encode(image, max_bytes=124)
        
        recon = codec.decode(payload)
        assert isinstance(recon, np.ndarray)
        assert recon.shape == (64, 64, 3)  # Matches model output size
        assert recon.dtype == np.uint8

    def test_encode_decode_roundtrip(self, codec):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        payload, recon = codec.encode_decode(image, max_bytes=124)
        assert len(payload) == 124
        assert recon.shape == (64, 64, 3)

    def test_rate_distortion_sweep(self, codec):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        budgets = [64, 96, 124]
        
        # Test without metric fn
        results = codec.rate_distortion_sweep(image, budgets=budgets)
        assert len(results) == 3
        assert results[0]["budget"] == 64
        assert results[0]["bytes"] == 64
        
        # Test with metric fn
        def mock_metric(orig, rec):
            return {"mock_score": 1.0}
            
        results = codec.rate_distortion_sweep(image, budgets=budgets, metric_fn=mock_metric)
        assert len(results) == 3
        assert "mock_score" in results[0]
        assert results[0]["mock_score"] == 1.0

    def test_bad_image_input(self, codec):
        with pytest.raises(ValueError):
            codec.encode(np.zeros((128, 128), dtype=np.uint8), max_bytes=124)  # Missing channel dim
            
    def test_bad_payload_input(self, codec):
        with pytest.raises(TypeError):
            codec.decode("not a byte string")
