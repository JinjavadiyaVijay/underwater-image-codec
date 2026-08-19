"""Tests for BLE packet format and CRC."""

import pytest

from uwcodec.ble.crc import crc8
from uwcodec.ble.packet import packetize, reassemble, serialize_packet, deserialize_packet
from uwcodec.ble.mtu import validate_mtu, recommended_mtu, max_payload_for_mtu


class TestCRC8:
    def test_empty(self):
        assert crc8(b"") == 0

    def test_deterministic(self):
        data = b"hello world"
        assert crc8(data) == crc8(data)

    def test_different_data(self):
        assert crc8(b"aaa") != crc8(b"bbb")

    def test_byte_range(self):
        for i in range(256):
            result = crc8(bytes([i]))
            assert 0 <= result <= 255


class TestBLEPacket:
    def test_single_packet(self):
        payload = b"\x00" * 124
        packets = packetize(payload, mtu=244)
        assert len(packets) == 1
        assert packets[0].payload == payload

    def test_fragmentation(self):
        payload = b"\x00" * 200
        packets = packetize(payload, mtu=50)
        assert len(packets) > 1
        reassembled = reassemble(packets)
        assert reassembled == payload

    def test_serialize_deserialize(self):
        payload = b"\xAA\xBB\xCC"
        packets = packetize(payload, mtu=244)
        serialized = serialize_packet(packets[0])
        deserialized = deserialize_packet(serialized)
        assert deserialized.payload == payload

    def test_all_codec_budgets_single_packet(self):
        for size in [64, 96, 124]:
            payload = b"\x00" * size
            packets = packetize(payload, mtu=244)
            assert len(packets) == 1, f"Budget {size}B should fit single packet"


class TestMTU:
    def test_validate_mtu_130(self):
        result = validate_mtu(130)
        assert result["supports_124b"] is True
        assert result["recommended"] is True

    def test_validate_mtu_23(self):
        result = validate_mtu(23)
        assert result["supports_124b"] is False
        assert result["recommended"] is False

    def test_recommended_mtu(self):
        mtu = recommended_mtu(124)
        assert mtu >= 127

    def test_max_payload(self):
        assert max_payload_for_mtu(244) == 241
