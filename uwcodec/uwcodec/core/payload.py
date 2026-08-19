"""Minimal payload format: version(1B) + VQ_data(N-2B) + CRC(1B).

No fish metadata. No species/bbox/pose. Every byte is image data.

Format (max_bytes total):
  Byte 0:          version (protocol version, future flags)
  Bytes 1..N-2:    raw VQ token bytes (indices packed as uint8)
  Byte N-1:        CRC-8 (computed over bytes 0..N-2)

The HARD RULE: assert len(payload) == max_bytes always.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uwcodec.ble.crc import crc8
from uwcodec.core.config import PayloadConfig


# Current protocol version
PROTOCOL_VERSION = 1


@dataclass
class EncodedPayload:
    """An encoded payload ready for BLE transmission."""

    raw_bytes: bytes       # Exactly max_bytes bytes
    max_bytes: int         # The budget this was encoded at
    vq_bytes: bytes        # The VQ token data (without header/CRC)
    version: int = PROTOCOL_VERSION

    def __post_init__(self):
        # HARD ENFORCEMENT: payload must be exactly max_bytes
        assert len(self.raw_bytes) == self.max_bytes, (
            f"Payload size violation: got {len(self.raw_bytes)}B, expected {self.max_bytes}B"
        )

    @property
    def actual_bytes(self) -> int:
        return len(self.raw_bytes)

    def report(self) -> dict:
        return {
            "total_bytes": self.actual_bytes,
            "budget": self.max_bytes,
            "vq_data_bytes": len(self.vq_bytes),
            "overhead_bytes": self.max_bytes - len(self.vq_bytes),
            "version": self.version,
        }


class PayloadFormat:
    """Pack and unpack minimal codec payloads.

    Format:
        [version:1][vq_data:N-2][crc8:1]

    The version byte encodes:
        bits 7-4: protocol version (0-15)
        bits 3-0: flags (reserved, 0 for now)
    """

    def __init__(self, config: PayloadConfig | None = None):
        self.config = config or PayloadConfig()

    def pack(self, vq_data: bytes, max_bytes: int, version: int = PROTOCOL_VERSION) -> EncodedPayload:
        """Pack VQ token bytes into a fixed-size payload.

        Args:
            vq_data: Raw VQ index bytes (will be truncated or zero-padded to fit).
            max_bytes: Target payload size (hard limit enforced).
            version: Protocol version byte.

        Returns:
            EncodedPayload with exactly max_bytes bytes.
        """
        vq_budget = self.config.vq_bytes(max_bytes)

        buf = bytearray(max_bytes)

        # Byte 0: version
        buf[0] = version & 0xFF

        # Bytes 1..N-2: VQ data (truncate if too long, zero-pad if too short)
        data = vq_data[:vq_budget]
        buf[1:1 + len(data)] = data
        # Remaining positions already zero (bytearray default)

        # Byte N-1: CRC over everything except the CRC byte itself
        buf[-1] = crc8(bytes(buf[:-1]))

        payload = bytes(buf)
        assert len(payload) == max_bytes, f"BUG: pack produced {len(payload)}B, expected {max_bytes}B"

        return EncodedPayload(
            raw_bytes=payload,
            max_bytes=max_bytes,
            vq_bytes=bytes(buf[1:1 + len(data)]),
            version=version,
        )

    def unpack(self, data: bytes) -> tuple[bytes, int]:
        """Unpack a payload into (vq_data, version).

        Args:
            data: Raw bytes received (any length accepted; CRC verified).

        Returns:
            (vq_data, version) where vq_data is the raw VQ token bytes.

        Raises:
            ValueError: CRC mismatch.
        """
        if len(data) < self.config.fixed_overhead + 1:
            raise ValueError(f"Payload too short: {len(data)}B (minimum {self.config.fixed_overhead + 1}B)")

        # Verify CRC
        expected_crc = data[-1]
        actual_crc = crc8(data[:-1])
        if expected_crc != actual_crc:
            raise ValueError(
                f"CRC mismatch: expected 0x{expected_crc:02X}, got 0x{actual_crc:02X}"
            )

        version = data[0] & 0xFF
        vq_data = bytes(data[1:-1])  # everything between version and CRC

        return vq_data, version

    def inspect(self, data: bytes) -> str:
        """Human-readable inspection of a payload."""
        lines = [f"=== Payload ({len(data)} bytes) ==="]
        try:
            vq_data, version = self.unpack(data)
            lines.append(f"  CRC:         OK")
            lines.append(f"  Version:     {version}")
            lines.append(f"  VQ data:     {len(vq_data)} bytes")
            lines.append(f"  Overhead:    {len(data) - len(vq_data)} bytes ({(len(data) - len(vq_data))/len(data)*100:.1f}%)")
            if vq_data:
                hex_preview = vq_data[:32].hex(" ")
                lines.append(f"  VQ hex[0:32]: {hex_preview}{'...' if len(vq_data) > 32 else ''}")
        except ValueError as e:
            lines.append(f"  CRC:         FAILED — {e}")
        return "\n".join(lines)
