"""MTU negotiation helpers for BLE connections.

For the UWCodec, MTU negotiation above 130 bytes is REQUIRED to transmit
payloads of 64-124 bytes in a single notification. This module provides
helpers and validation.
"""

from __future__ import annotations


# BLE constants
DEFAULT_MTU = 23       # Minimum BLE MTU (BLE 4.0)
ATT_OVERHEAD = 3       # ATT protocol overhead
MINIMUM_CODEC_MTU = 130  # Required for single-packet 124B payload

# Common MTU values by BLE version
COMMON_MTUS = {
    "4.0": 23,
    "4.1": 23,
    "4.2": 247,   # With DLE (Data Length Extension)
    "5.0": 247,   # Standard with DLE
    "5.1": 247,
    "5.2": 247,
}


def validate_mtu(mtu: int) -> dict[str, bool | int]:
    """Validate an MTU for codec use.

    Args:
        mtu: Negotiated MTU value.

    Returns:
        Dict with validation results.
    """
    effective_payload = mtu - ATT_OVERHEAD
    return {
        "mtu": mtu,
        "effective_payload": effective_payload,
        "supports_64b": effective_payload >= 64,
        "supports_96b": effective_payload >= 96,
        "supports_124b": effective_payload >= 124,
        "single_packet_124b": effective_payload >= 124,
        "recommended": mtu >= MINIMUM_CODEC_MTU,
    }


def recommended_mtu(payload_size: int = 124) -> int:
    """Return the minimum recommended MTU for a given payload size.

    Args:
        payload_size: Target payload size in bytes.

    Returns:
        Minimum MTU that supports single-packet transmission.
    """
    return payload_size + ATT_OVERHEAD + 3  # +3 for safety margin


def max_payload_for_mtu(mtu: int) -> int:
    """Return the maximum payload bytes for a given MTU.

    Args:
        mtu: Negotiated MTU.

    Returns:
        Maximum payload size in bytes.
    """
    return max(0, mtu - ATT_OVERHEAD)
