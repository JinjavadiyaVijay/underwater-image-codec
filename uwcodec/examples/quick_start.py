"""Example: Quick start with synthetic data."""

from pathlib import Path
import numpy as np

from uwcodec.core.config import UWCodecConfig
from uwcodec.core.codec import UWCodec
from uwcodec.codecs.oracle import OracleCodec, build_oracle_training_data
from uwcodec.data.dataset import create_synthetic_dataset, FishCropDataset
from uwcodec.evaluation.metrics import compute_all_metrics


def main():
    print("=" * 60)
    print("UWCodec Quick Start — Synthetic Data Demo")
    print("=" * 60)

    # 1. Create synthetic dataset
    data_dir = Path("outputs/quick_start/data")
    create_synthetic_dataset(data_dir, num_species=3, images_per_species=10)
    dataset = FishCropDataset(root=data_dir, crop_size=128)
    print(f"\nCreated dataset: {len(dataset)} samples, {dataset.species_mapping.num_species} species")

    # 2. Build oracle training data
    oracle_data = build_oracle_training_data(dataset, output_size=128)
    codec = OracleCodec(oracle_data, output_size=128)

    # 3. Encode/decode at each budget
    sample = dataset[0]
    image = sample["image"]
    species_id = sample["species_id"]
    species_name = sample["species_name"]

    print(f"\nTest image: species={species_name} (ID={species_id})")

    for budget in [64, 96, 124]:
        payload, recon = codec.encode_decode(image, species_id, max_bytes=budget)
        metrics = compute_all_metrics(image, recon, payload_bytes=budget, compute_lpips_flag=False)

        psnr = metrics.metrics.get("psnr", 0)
        ssim = metrics.metrics.get("ssim", 0)
        print(f"\n  {budget}B payload:")
        print(f"    Size:  {len(payload.raw_bytes)} bytes")
        print(f"    PSNR:  {psnr:.2f} dB")
        print(f"    SSIM:  {ssim:.4f}")

    # 4. Payload inspection
    from uwcodec.core.payload import PayloadFormat
    fmt = PayloadFormat()
    print(f"\n  Payload breakdown (124B):")
    payload_124, _ = codec.encode_decode(image, species_id, max_bytes=124)
    print(fmt.inspect(payload_124.raw_bytes))

    # 5. BLE transmission estimate
    from uwcodec.ble.packet import estimate_transmission_time
    timing = estimate_transmission_time(124)
    print(f"\n  BLE transmission estimate (124B):")
    print(f"    Packets:    {timing['num_packets']}")
    print(f"    Est. time:  {timing['estimated_ms']:.1f} ms")

    print("\n" + "=" * 60)
    print("Quick start complete!")


if __name__ == "__main__":
    main()
