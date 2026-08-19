"""Tests for minimal payload pack/unpack roundtrip."""

import pytest

from uwcodec.core.config import PayloadConfig
from uwcodec.core.payload import PayloadFormat, PROTOCOL_VERSION


class TestPayloadRoundtrip:
    """Test that pack → unpack preserves VQ data exactly."""

    def setup_method(self):
        self.config = PayloadConfig()
        self.fmt = PayloadFormat(self.config)

    @pytest.mark.parametrize("max_bytes", [64, 96, 124, 256])
    def test_roundtrip(self, max_bytes):
        # Generate some dummy VQ bytes (N-2 bytes)
        vq_budget = self.config.vq_bytes(max_bytes)
        vq_data = bytes([i % 256 for i in range(vq_budget)])

        payload = self.fmt.pack(vq_data, max_bytes, version=PROTOCOL_VERSION)
        
        # Hard assertion test
        assert len(payload.raw_bytes) == max_bytes
        assert payload.actual_bytes == max_bytes

        # Unpack
        unpacked_data, unpacked_version = self.fmt.unpack(payload.raw_bytes)
        
        assert unpacked_version == PROTOCOL_VERSION
        assert unpacked_data == vq_data

    def test_pack_truncates_long_data(self):
        max_bytes = 64
        vq_budget = self.config.vq_bytes(max_bytes)
        # Supply way too much data
        vq_data = b"\xAA" * (vq_budget + 100)

        payload = self.fmt.pack(vq_data, max_bytes)
        assert len(payload.raw_bytes) == max_bytes
        
        unpacked_data, _ = self.fmt.unpack(payload.raw_bytes)
        assert len(unpacked_data) == vq_budget
        assert unpacked_data == b"\xAA" * vq_budget

    def test_pack_pads_short_data(self):
        max_bytes = 64
        vq_budget = self.config.vq_bytes(max_bytes)
        # Supply too little data
        vq_data = b"\xBB" * (vq_budget - 10)

        payload = self.fmt.pack(vq_data, max_bytes)
        assert len(payload.raw_bytes) == max_bytes
        
        unpacked_data, _ = self.fmt.unpack(payload.raw_bytes)
        assert len(unpacked_data) == vq_budget
        assert unpacked_data[:len(vq_data)] == vq_data
        assert unpacked_data[len(vq_data):] == b"\x00" * 10

    def test_crc_corruption_detected(self):
        vq_data = b"\x01" * 62
        payload = self.fmt.pack(vq_data, 64)

        # Corrupt a byte
        corrupted = bytearray(payload.raw_bytes)
        corrupted[5] ^= 0xFF
        corrupted = bytes(corrupted)

        with pytest.raises(ValueError, match="CRC"):
            self.fmt.unpack(corrupted)


class TestPayloadConfig:
    def test_summary(self):
        config = PayloadConfig()
        summary = config.summary(124)
        assert summary["total"] == 124
        assert summary["version"] == 1
        assert summary["crc"] == 1
        assert summary["vq_tokens"] == 122

    def test_budget_too_small(self):
        config = PayloadConfig()
        with pytest.raises(ValueError):
            config.vq_bytes(2)  # Too small for header+vq
