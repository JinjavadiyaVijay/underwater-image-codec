"""BLE packet format: MTU-aware packetization.

64-124 byte payloads fit in a single BLE notification (with MTU ≥ 130).
This module handles packetization for cases where MTU is smaller or
payloads need fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from uwcodec.ble.crc import crc8


# BLE packet header flags
FLAG_SINGLE_PACKET = 0x00
FLAG_FIRST_FRAGMENT = 0x01
FLAG_MIDDLE_FRAGMENT = 0x02
FLAG_LAST_FRAGMENT = 0x03

PACKET_HEADER_SIZE = 3  # flags(1) + seq(1) + total_fragments(1)


@dataclass
class BLEPacket:
    """A single BLE notification packet."""
    flags: int       # packet type flags
    sequence: int    # sequence number (0-based)
    total: int       # total fragments in this payload
    payload: bytes   # packet payload data


def packetize(
    payload: bytes,
    mtu: int = 244,
) -> list[BLEPacket]:
    """Split a payload into BLE-sized packets.

    For our typical payloads (64-124 bytes), this almost always produces
    a single packet. Fragmentation is only needed if MTU is unusually small.

    Args:
        payload: Complete codec payload bytes.
        mtu: Maximum Transmission Unit (negotiated BLE MTU minus ATT overhead).
             Default 244 = typical BLE 5.0 with DLE. Minimum useful = 23.

    Returns:
        List of BLEPacket objects.
    """
    # ATT overhead: 3 bytes for ATT header
    max_payload = mtu - 3
    # Reserve header space
    max_data = max_payload - PACKET_HEADER_SIZE

    if max_data <= 0:
        raise ValueError(f"MTU {mtu} too small for any payload (need > {PACKET_HEADER_SIZE + 3})")

    if len(payload) <= max_data:
        # Single packet — the common case
        return [BLEPacket(
            flags=FLAG_SINGLE_PACKET,
            sequence=0,
            total=1,
            payload=payload,
        )]

    # Fragment
    packets = []
    offset = 0
    total_fragments = (len(payload) + max_data - 1) // max_data

    for seq in range(total_fragments):
        chunk = payload[offset:offset + max_data]
        offset += max_data

        if seq == 0:
            flags = FLAG_FIRST_FRAGMENT
        elif seq == total_fragments - 1:
            flags = FLAG_LAST_FRAGMENT
        else:
            flags = FLAG_MIDDLE_FRAGMENT

        packets.append(BLEPacket(
            flags=flags,
            sequence=seq,
            total=total_fragments,
            payload=chunk,
        ))

    return packets


def reassemble(packets: list[BLEPacket]) -> bytes:
    """Reassemble fragmented BLE packets into complete payload.

    Args:
        packets: List of BLEPacket objects (must be in order).

    Returns:
        Complete reassembled payload bytes.

    Raises:
        ValueError: If packets are missing or out of order.
    """
    if len(packets) == 1 and packets[0].flags == FLAG_SINGLE_PACKET:
        return packets[0].payload

    # Sort by sequence
    packets = sorted(packets, key=lambda p: p.sequence)

    # Validate
    if packets[0].flags != FLAG_FIRST_FRAGMENT:
        raise ValueError("Missing first fragment")
    if packets[-1].flags != FLAG_LAST_FRAGMENT:
        raise ValueError("Missing last fragment")

    expected_total = packets[0].total
    if len(packets) != expected_total:
        raise ValueError(f"Expected {expected_total} fragments, got {len(packets)}")

    # Reassemble
    payload = b"".join(p.payload for p in packets)
    return payload


def serialize_packet(packet: BLEPacket) -> bytes:
    """Serialize a BLEPacket to bytes for transmission.

    Format: [flags:1][sequence:1][total:1][payload:N]
    """
    header = bytes([packet.flags, packet.sequence, packet.total])
    return header + packet.payload


def deserialize_packet(data: bytes) -> BLEPacket:
    """Deserialize bytes into a BLEPacket.

    Args:
        data: Raw bytes received from BLE notification.

    Returns:
        BLEPacket object.
    """
    if len(data) < PACKET_HEADER_SIZE:
        raise ValueError(f"Packet too short: {len(data)} bytes")

    return BLEPacket(
        flags=data[0],
        sequence=data[1],
        total=data[2],
        payload=data[PACKET_HEADER_SIZE:],
    )


def estimate_transmission_time(
    payload_bytes: int,
    mtu: int = 244,
    ble_version: str = "5.0",
    connection_interval_ms: float = 7.5,
) -> dict[str, float]:
    """Estimate BLE transmission time.

    Args:
        payload_bytes: Total payload size.
        mtu: Negotiated MTU.
        ble_version: BLE version (affects PHY data rate).
        connection_interval_ms: BLE connection interval.

    Returns:
        Dict with estimated timing.
    """
    # PHY data rates
    rates = {
        "4.2": 1_000_000,   # 1 Mbps LE 1M
        "5.0": 2_000_000,   # 2 Mbps LE 2M
        "5.1": 2_000_000,
        "5.2": 2_000_000,
    }

    phy_rate = rates.get(ble_version, 1_000_000)
    packets = packetize(b"\x00" * payload_bytes, mtu)
    num_packets = len(packets)

    # Time per packet (rough estimate)
    bits_per_packet = (mtu + 14) * 8  # payload + BLE overhead
    time_per_packet_us = bits_per_packet / (phy_rate / 1_000_000)

    # Total time = packet transmission + connection intervals
    total_ms = (num_packets * connection_interval_ms +
                num_packets * time_per_packet_us / 1000)

    return {
        "num_packets": num_packets,
        "estimated_ms": total_ms,
        "phy_rate_mbps": phy_rate / 1_000_000,
        "payload_bytes": payload_bytes,
    }
