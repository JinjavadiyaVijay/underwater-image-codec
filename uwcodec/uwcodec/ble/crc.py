"""CRC-8 implementation for payload integrity checking.

Uses the CRC-8/MAXIM polynomial (x^8 + x^5 + x^4 + 1, poly=0x31)
which is well-suited for short payloads and commonly used in embedded systems.
"""

# CRC-8/MAXIM lookup table (polynomial 0x31)
_CRC8_TABLE = None


def _build_table() -> list[int]:
    """Build CRC-8 lookup table for polynomial 0x31."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


def crc8(data: bytes, init: int = 0x00) -> int:
    """Compute CRC-8/MAXIM checksum.

    Args:
        data: Input bytes to compute checksum over.
        init: Initial CRC value (default 0x00).

    Returns:
        Single-byte CRC checksum (0-255).
    """
    global _CRC8_TABLE
    if _CRC8_TABLE is None:
        _CRC8_TABLE = _build_table()

    crc = init
    for byte in data:
        crc = _CRC8_TABLE[crc ^ byte]
    return crc


def verify_crc8(data: bytes) -> bool:
    """Verify CRC-8 of a payload where the last byte is the CRC.

    Args:
        data: Payload bytes with CRC as the last byte.

    Returns:
        True if CRC is valid.
    """
    if len(data) < 2:
        return False
    payload = data[:-1]
    expected = data[-1]
    return crc8(payload) == expected
