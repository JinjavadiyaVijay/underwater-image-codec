# UWCodec — Finish Project Implementation Plan

## Repository Audit Summary

The current codebase is in good shape after the general-purpose redesign:

| Component | Status | Notes |
|---|---|---|
| `core/codec.py` (UWCodec API) | ✅ Working | Clean encode/decode, hard limit enforcement |
| `core/config.py` | ✅ Working | 2-byte overhead, no fish fields |
| `core/payload.py` | ✅ Working | version + VQ data + CRC |
| `models/encoder.py` | ✅ Working | MobileNet-style depthwise separable CNN |
| `models/decoder.py` | ✅ Working | Unconditional CNN upsampler |
| `models/quantizer.py` | ✅ Working | VQ, Product-VQ, Residual-VQ |
| `codecs/vqvae_codec.py` | ✅ Working | MinimalVQVAE end-to-end |
| `codecs/general_oracle.py` | ✅ Working | 3 strategies, honest verdicts |
| `data/dataset.py` | ⚠️ Needs update | Supports flat dirs, needs EUVP/SUIM/UIEB awareness |
| `training/train_codec.py` | ⚠️ Needs update | Works but needs EUVP integration, proper validation |
| `training/losses.py` | ✅ Working | Pixel, perceptual, VQ losses |
| `evaluation/metrics.py` | ✅ Working | PSNR, SSIM, UCIQE, UIQM |
| `ble/` | ✅ Working | Packet, CRC, MTU |
| `baselines/` | ✅ Exists | JPEG, WebP, CompressAI, prototype |
| `deployment/` | ✅ Exists | ONNX export, STM32, profiling stubs |
| Tests | ✅ 51 passing | Core tests cover all components |

---

## Phase 1: Dataset Setup (EUVP, SUIM, UIEB)

### What
Set up the three required datasets external to the repository.

### Directory structure
```
S:\IMG_compressors\
├── uwcodec\                    # repository (no large data here)
└── datasets\
    ├── EUVP\                   # PRIMARY TRAINING
    │   ├── train\
    │   ├── val\
    │   └── test\
    ├── SUIM\                   # validation/generalization
    │   ├── train\
    │   ├── val\
    │   └── test\
    ├── UIEB\                   # independent evaluation
    │   ├── images\
    │   └── references\
    └── MB1854B_test\           # future: real camera frames
```

### Files to create/modify

#### [NEW] `docs/DATASET_SETUP.md`
Document exact download URLs, license requirements, and setup instructions for each dataset. Do NOT fabricate download URLs.

#### [MODIFY] `uwcodec/data/dataset.py`
Add `MultiDatasetLoader` class that:
- Accepts named dataset configs (EUVP, SUIM, UIEB) with explicit split assignments
- Prevents train/test leakage (no image from SUIM/UIEB appears in training)
- Generates a dataset manifest CSV

#### [NEW] `uwcodec/data/manifest.py`
Dataset manifest generator containing: image path, dataset name, split, resolution, format, perceptual hash.

#### [MODIFY] `.gitignore`
Add `datasets/` entry.

---

## Phase 2: Oracle on Real Images

### What
Run the existing `general_oracle.py` on real EUVP images (not synthetic).

### Files to create

#### [NEW] `scripts/run_oracle_real.py`
Wrapper that:
1. Loads 20-50 real EUVP images
2. Runs all 3 oracle strategies at 64/96/124/256/512/1024B
3. Saves comparison image grids
4. Saves metrics CSV
5. Generates a text report

#### [NEW] `docs/RESULTS.md`
Start with oracle results. Append measured results from each subsequent phase.

### Gate
The oracle result is a BASELINE only. Even if all strategies fail at 64B, we proceed to the learned codec. The oracle just sets expectations.

---

## Phase 3: Minimal Learned Codec Training (5-10 epochs)

### What
Train the existing `MinimalVQVAE` on 1,000-2,000 EUVP images for 5-10 epochs at 124B budget. This is the fail-fast learning gate.

### Files to modify

#### [MODIFY] `uwcodec/training/train_codec.py`
- Add `--dataset` flag (euvp/suim/uieb)
- Add `--dataset-root` flag pointing to `S:\IMG_compressors\datasets`
- Add per-epoch reconstruction image saving
- Add PSNR/SSIM tracking per epoch

### Commands
```bash
python -m uwcodec.training.train_codec \
  --dataset euvp \
  --dataset-root S:\IMG_compressors\datasets \
  --input-size 128 \
  --train-budget 124 \
  --epochs 10 \
  --batch-size 16 \
  --num-images 2000 \
  --output-dir outputs/phase3_minimal
```

### Gate
Does reconstruction loss decrease over epochs? Are reconstruction images improving? If yes → proceed. If no → diagnose before scaling.

---

## Phase 4: Inspect & Fix

### What
Visually inspect Phase 3 reconstructions. Fix any issues before scaling.

### Checks
- Does loss decrease?
- Do reconstructions show scene structure?
- Is VQ codebook utilized (perplexity > 1)?
- Are colors vaguely correct?

---

## Phase 5: Multi-Budget Training (64/96/124B)

### What
Train separate models or a budget-aware model for each target budget.

### Files to create

#### [NEW] `scripts/train_multi_budget.py`
Trains at 64B, 96B, and 124B. Saves separate checkpoints. Compares quality.

### Gate
Compare 64B vs 96B vs 124B reconstruction quality. Document rate-distortion curve.

---

## Phase 6: Baselines & Rate-Distortion Study

### What
Run JPEG, WebP, JPEG XL, CompressAI at equal byte budgets. Generate rate-distortion table.

### Files to create/modify

#### [NEW] `scripts/run_baselines.py`
- Tests at 64/96/124/256/512/1K/2K/4K bytes
- Measures PSNR, SSIM, MS-SSIM, LPIPS, UCIQE, UIQM, edge preservation
- Saves comparison grids
- Generates CSV + markdown table

#### [NEW] `docs/EVALUATION.md`
Rate-distortion results, visual comparisons, methodology.

---

## Phase 7: Ablation Improvements (One at a Time)

### What
After the baseline VQ-VAE works, test improvements individually:
1. Product VQ
2. Residual VQ
3. Entropy coding
4. Rate controller
5. Stronger perceptual loss (LPIPS)
6. Underwater-specific loss (color correction aware)
7. Optional BioCLIP teacher (NOT a core dependency)

### Files to create

#### [NEW] `scripts/run_ablation.py`
Trains one variant, evaluates, records in ablation table.

#### [NEW] `docs/RESULTS.md` (append)
Ablation table showing metric delta for each change.

---

## Phase 8: Held-Out Evaluation (SUIM, UIEB)

### What
Evaluate on SUIM and UIEB images that were NEVER used in training.

### Key checks
- Does the codec preserve the actual scene vs generating generic underwater?
- Major objects present?
- Colors/geometry approximately correct?
- Hallucinated objects?

---

## Phase 9: MB1854B Test Set

### What
Create `datasets/MB1854B_test/` with real camera frames. Evaluate.

> [!IMPORTANT]
> This requires real hardware capture. Document the procedure but do not fabricate frames.

---

## Phase 10: BLE Integration Test

### What
Test the full pipeline: `image → encode → packetize → BLE → reassemble → decode`

### Files to create

#### [NEW] `scripts/test_ble_pipeline.py`
Simulated BLE round-trip including packet loss, corruption, and retransmission.

#### [NEW] `docs/BLE.md`
BLE test results: MTU, latency, packet loss resilience.

---

## Phase 11: Deployment Profiling

### What
ONNX export, INT8 quantization, memory/latency profiling.

> [!WARNING]
> Do NOT claim embedded deployment until measured on actual hardware.

#### [NEW] `docs/DEPLOYMENT.md`
Model size, latency, memory, STM32N6570 feasibility assessment.

---

## Phase 12-15: Documentation & Final Integration

### What
Update all documentation. Create clear labels: IMPLEMENTED / TRAINED / MEASURED / ESTIMATED / FUTURE.

### Files to create/modify

#### [NEW] `docs/TRAINING.md`
#### [MODIFY] `README.md`
#### [MODIFY] walkthrough.md

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/ -v
```

### Manual Verification
- Phase 3: Inspect reconstruction images visually
- Phase 5: Compare 64/96/124B reconstructions side by side
- Phase 6: Review rate-distortion table against baselines
- Phase 8: Human assessment of held-out SUIM/UIEB results

---

## Open Questions

> [!IMPORTANT]
> **Dataset Download**: EUVP, SUIM, and UIEB require manual download (they have license agreements). I will document the download procedure but cannot auto-download them. Do you already have any of these datasets downloaded?

> [!IMPORTANT]
> **MB1854B Frames**: Phase 9 requires real camera frames. Do you have any captured MB1854B frames available, or should I create a placeholder directory and document the capture procedure?

> [!IMPORTANT]
> **GPU Availability**: The training phases (3, 5, 7) benefit from GPU. Is CUDA available on this machine, or should I optimize for CPU-only training?
