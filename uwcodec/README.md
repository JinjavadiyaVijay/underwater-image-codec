# UWCodec

**General-purpose underwater image codec for extreme low-bitrate BLE transmission.**

> **Honest caveat**: At 64–124 bytes, this is *extreme semantic reconstruction*, not lossless compression. A 500KB image compressed to 124 bytes (0.06 bpp) cannot be reconstructed with high fidelity. UWCodec uses a learned prior to produce a *plausible, visually coherent* underwater image — not a pixel-accurate copy.

---

## What It Does

```
ANY RGB underwater image (~500KB)
    → UWCodec encoder (~3–4 ms)
    → 64 / 96 / 124 bytes
    → BLE (single packet at MTU ≥ 130)
    → receiver with preinstalled decoder (~9 ms)
    → visible RGB reconstruction
```

The **decoder is preinstalled** on the receiver. Only the per-image byte payload is transmitted over BLE.

## Architecture

UWCodec uses a **Minimal VQ-VAE** (Vector-Quantized Variational Autoencoder):

- **Encoder**: MobileNet-style depthwise-separable CNN → 8×8 spatial grid of latent codes
- **Quantizer**: 256-entry codebook (1 byte per code) → exactly 64 codes per image
- **Decoder**: Unconditional CNN upsampler → 128×128 RGB output
- **Payload**: `[version:1B][crc8:1B][vq_indices:62-122B]` = exactly 64–124 bytes

### Key Design Principle: Shared Decoder

The decoder is preinstalled on receivers. No class labels, species IDs, or lookup tables are transmitted. The model itself encodes all visual priors.

---

## Quick Start

### Install

```bash
cd uwcodec
pip install -e .
```

### Smoke Test (No Data Required)

```bash
# 2-epoch training on synthetic underwater images
python -m uwcodec.training.train_codec --synthetic --epochs 5 --num-images 200
```

### Encode / Decode

```python
from uwcodec import UWCodec
import numpy as np

# Load a trained model
codec = UWCodec.load("outputs/train/best.pt")

# Encode any RGB image (any resolution)
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # simulated frame
payload = codec.encode(image, max_bytes=124)

assert len(payload) <= 124  # always enforced

# Decode on receiver
reconstructed = codec.decode(payload)  # returns (128, 128, 3) uint8
```

---

## Training

### 1. Download Datasets

See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md) for download instructions.

| Dataset | Role | Source |
|---|---|---|
| EUVP | Primary training | [UMN IRL Lab](https://irvlab.cs.umn.edu/resources/euvp-dataset) |
| SUIM | Held-out validation | [UMN IRL Lab](https://irvlab.cs.umn.edu/resources/suim-dataset) |
| UIEB | Independent evaluation (never trained on) | [UIEB Project Page](https://li-chongyi.github.io/proj_benchmark.html) |

### 2. Train

```bash
# Train at 124B budget on EUVP (CPU: ~2-4h, GPU: ~20min)
python -m uwcodec.training.train_codec \
    --dataset euvp \
    --datasets-root datasets/ \
    --train-budget 124 \
    --epochs 50

# Train all three budgets (64, 96, 124B) sequentially
python scripts/train_multi_budget.py --dataset euvp --datasets-root datasets/
```

See [`docs/TRAINING.md`](docs/TRAINING.md) for full hyperparameter reference and troubleshooting.

---

## Evaluation

### Oracle Baseline (non-learned upper bound)

```bash
python scripts/run_oracle_real.py --datasets-root datasets/
```

Typical result: ~26–28 dB PSNR at 64–124B using DCT/pixel-grid methods.

### Baselines vs. UWCodec

```bash
python scripts/run_baselines.py \
    --datasets-root datasets/ \
    --model outputs/train/best.pt
```

> **Note**: JPEG and WebP *fail* at 64B (their minimum file size exceeds the budget due to headers). At 124B they produce blocky/blurry garbage. UWCodec's learned semantic prior outperforms all traditional methods at these extreme rates.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for full methodology.

### Ablation Study

```bash
python scripts/run_ablation.py --datasets-root datasets/
```

---

## BLE Integration Test

```bash
# Test packetization, CRC, and timing without hardware
python scripts/test_ble_pipeline.py

# End-to-end with trained model
python scripts/test_ble_pipeline.py --model outputs/train/best.pt
```

At BLE 5.0 with MTU=244, all three budgets (64/96/124B) fit in **a single BLE notification** with ~8.5ms estimated transmission time.

See [`docs/BLE.md`](docs/BLE.md) for the GATT service design and payload format.

---

## Deployment Profiling

```bash
python scripts/profile_deployment.py --model outputs/train/best.pt
```

**Measured CPU performance** (no GPU, 128×128 images):

| Budget | Encode | Decode | Total | FPS |
|---|---|---|---|---|
| 64B  | 3.9 ms | 9.2 ms | 13.1 ms | ~76 |
| 96B  | 3.4 ms | 9.0 ms | 12.3 ms | ~81 |
| 124B | 3.0 ms | 8.8 ms | 11.8 ms | ~85 |

Model: ~2.85M parameters, ~10.9 MB fp32 / ~2.7 MB int8. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Repository Structure

```
uwcodec/
├── uwcodec/
│   ├── core/           # Public API (UWCodec class, config)
│   ├── codecs/         # MinimalVQVAE model
│   ├── training/       # Training loop
│   ├── data/           # Dataset loading (EUVP, SUIM, UIEB)
│   ├── evaluation/     # PSNR, SSIM, UCIQE, UIQM metrics
│   ├── baselines/      # JPEG, WebP, oracle baselines
│   ├── ble/            # BLE packetization, CRC, MTU utilities
│   └── deployment/     # ONNX export, profiling, STM32 export
├── scripts/
│   ├── run_oracle_real.py       # Phase 2: Non-learned baseline
│   ├── run_baselines.py         # Phase 6: JPEG/WebP comparison
│   ├── run_ablation.py          # Phase 7: Architecture search
│   ├── train_multi_budget.py    # Phase 5: Multi-budget training
│   ├── test_ble_pipeline.py     # Phase 10: BLE round-trip test
│   └── profile_deployment.py   # Phase 11: Latency profiling
├── docs/
│   ├── DATASET_SETUP.md         # Download & setup instructions
│   ├── TRAINING.md              # Training guide & hyperparameters
│   ├── EVALUATION.md            # Evaluation methodology
│   ├── RESULTS.md               # Measured results
│   ├── BLE.md                   # BLE integration & GATT service
│   └── DEPLOYMENT.md            # Deployment guide & profiling
└── datasets/                    # (gitignored) EUVP, SUIM, UIEB
```

---

## Design Constraints

| Constraint | Value | Enforced? |
|---|---|---|
| Max payload (64B mode) | 64 bytes | Hard assertion in `encode()` |
| Max payload (96B mode) | 96 bytes | Hard assertion in `encode()` |
| Max payload (124B mode) | 124 bytes | Hard assertion in `encode()` |
| Overhead (version+CRC) | 2 bytes | Always |
| Codebook entries | 256 (1 byte each) | Fixed |
| Requires YOLO/labels | NO | By design |
| Requires species ID | NO | By design |
| Complete frame codec | YES | By design |

---

## Limitations & Honest Expectations

- At 64–124 bytes, **pixel-accurate reconstruction is impossible**. PSNR ~20–28 dB depending on image complexity.
- The learned decoder produces a *plausible* reconstruction, not a copy.
- Models trained on EUVP will degrade on very different domains (e.g. coral reefs vs. open ocean).
- CUDA is not required but significantly speeds up training.
- The current architecture has not been optimized for MCU deployment — ONNX/TFLite export is experimental.
