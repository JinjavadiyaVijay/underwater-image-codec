"""BLE pipeline integration test.

Simulates the full encode -> BLE packetize -> reassemble -> decode round-trip
without real hardware. Validates that the codec payload round-trips correctly
through BLE fragmentation logic at all MTU sizes and budgets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uwcodec.ble.packet import (
    packetize,
    reassemble,
    serialize_packet,
    deserialize_packet,
    estimate_transmission_time,
)
from uwcodec.ble.crc import crc8
from uwcodec.evaluation.metrics import compute_psnr, compute_ssim


def parse_args():
    p = argparse.ArgumentParser(description="BLE Pipeline Integration Test")
    p.add_argument("--model", type=Path, default=None, help="Trained codec model (.pt)")
    p.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 124])
    p.add_argument("--mtus", type=int, nargs="+", default=[23, 64, 128, 244],
                   help="BLE MTU sizes to test fragmentation")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-images", type=int, default=3)
    return p.parse_args()


def make_test_image(size: int) -> np.ndarray:
    """Create a synthetic underwater-like test image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 20
    img[:, :, 1] = 80
    img[:, :, 2] = 140
    img[size//4:3*size//4, size//4:3*size//4, 0] = 220
    img[size//4:3*size//4, size//4:3*size//4, 1] = 160
    img[size//4:3*size//4, size//4:3*size//4, 2] = 50
    return img


def test_ble_round_trip(payload: bytes, mtu: int) -> bool:
    """Simulate BLE transmission: packetize -> serialize -> deserialize -> reassemble."""
    packets = packetize(payload, mtu=mtu)
    serialized = [serialize_packet(p) for p in packets]
    deserialized = [deserialize_packet(s) for s in serialized]
    recovered = reassemble(deserialized)
    return recovered == payload


def main():
    args = parse_args()

    print("=" * 60)
    print("UWCodec BLE Pipeline Integration Test")
    print("=" * 60)

    # --- Phase 1: BLE Packetization Tests ---
    print("\n[1] BLE Packetization & Reassembly Tests")
    print("-" * 40)

    all_pass = True
    for budget in args.budgets:
        for mtu in args.mtus:
            payload = bytes(range(budget % 256)) * (budget // 256 + 1)
            payload = payload[:budget]

            try:
                ok = test_ble_round_trip(payload, mtu=mtu)
                status = "PASS" if ok else "FAIL"
                packets = packetize(payload, mtu=mtu)
                print(f"  {budget:3d}B payload | MTU={mtu:3d} | Fragments={len(packets)} | [{status}]")
                if not ok:
                    all_pass = False
            except Exception as e:
                print(f"  {budget:3d}B payload | MTU={mtu:3d} | [ERROR]: {e}")
                all_pass = False

    # --- Phase 2: CRC Validation ---
    print("\n[2] CRC-8 Validation")
    print("-" * 40)

    test_data = b"UWCodec payload test data"
    crc = crc8(test_data)
    print(f"  CRC-8 of test data: 0x{crc:02X}")
    print(f"  CRC-8 of empty:     0x{crc8(b''):02X}")
    tampered = bytes([test_data[0] ^ 0xFF]) + test_data[1:]
    crc_tampered = crc8(tampered)
    crc_ok = (crc != crc_tampered)
    print(f"  CRC detects tamper: {'YES [PASS]' if crc_ok else 'NO [FAIL]'}")

    # --- Phase 3: Transmission Time Estimates ---
    print("\n[3] Transmission Time Estimates")
    print("-" * 40)

    for budget in args.budgets:
        for ble_ver in ["4.2", "5.0"]:
            timing = estimate_transmission_time(budget, ble_version=ble_ver)
            print(
                f"  {budget:3d}B | BLE {ble_ver} | "
                f"{timing['num_packets']} packet(s) | "
                f"~{timing['estimated_ms']:.1f} ms"
            )

    # --- Phase 4: Codec Round-Trip (if model provided) ---
    if args.model and args.model.exists():
        print("\n[4] Full Codec + BLE Round-Trip")
        print("-" * 40)

        from uwcodec.core.codec import UWCodec
        codec = UWCodec.load(args.model)

        for i in range(args.num_images):
            img = make_test_image(args.image_size)

            for budget in args.budgets:
                payload = codec.encode(img, max_bytes=budget)
                recon = codec.decode(payload)
                ble_ok = test_ble_round_trip(payload, mtu=244)
                psnr = compute_psnr(img, recon)
                ssim = compute_ssim(img, recon)
                ble_status = "PASS" if ble_ok else "FAIL"
                print(
                    f"  img{i} | {budget:3d}B | "
                    f"BLE:[{ble_status}] | "
                    f"PSNR={psnr:.1f}dB | SSIM={ssim:.4f}"
                )
    else:
        print("\n[4] Full Codec Round-Trip: SKIPPED (no --model provided)")
        print("    Run with: python scripts/test_ble_pipeline.py --model outputs/train/best.pt")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] ALL BLE PACKETIZATION TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED -- check output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
