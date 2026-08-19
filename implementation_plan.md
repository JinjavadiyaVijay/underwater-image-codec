# UWCodec Redesign Plan
## From: Fish/Object Codec → To: General-Purpose Underwater Image Codec

---

## Audit Findings

### ✅ Keep (reuse directly)
| Component | File | Why |
|-----------|------|-----|
| Encoder CNN | `models/encoder.py` | Good MobileNet-style architecture, no fish coupling |
| VQ / PQ / RVQ | `models/quantizer.py` | Solid, general-purpose |
| Rate controller | `codecs/rate_controller.py` | General token serialization/importance scoring |
| Losses (pixel, perceptual, structure, color) | `training/losses.py` | Keep, remove species/teacher terms |
| BLE packet/MTU/CRC | `ble/` | Fully general, keep as-is |
| ONNX export / profiling | `deployment/` | General, keep as-is |
| Metrics | `evaluation/metrics.py`, `structure_metrics.py` | General, keep |
| Preprocessing (color correction, resize) | `data/preprocessing.py` | Keep |
| JPEG/WebP baselines | `baselines/jpeg_webp.py` | Keep |
| CompressAI baseline | `baselines/compressai_baseline.py` | Keep |
| Tests for BLE, metrics, preprocessing | `tests/` | Keep and extend |

### ❌ Remove from core (move to `examples/applications/fish_codec/`)
| Component | File | Reason |
|-----------|------|--------|
| Species classifier | `models/classifier.py` | Fish-specific |
| YOLO detector | `models/detector.py` | Fish-specific |
| BioCLIP teacher | `models/teacher.py` | Optional, fish-specific |
| `SpeciesMapping` class | `data/dataset.py` | Fish-specific |
| `FishCropDataset` class | `data/dataset.py` | Fish-specific |
| Species embedding in decoder | `models/decoder.py` | Fish-specific conditioning |
| Oracle with species prior | `codecs/oracle.py` | Fish-specific strategy |
| Species ID in payload | `core/config.py`, `core/payload.py` | Fish-specific fields |
| `UWCodec` class (old) | `core/codec.py` | Entire class is fish-coupled |
| VQVAECodec with species conditioning | `codecs/vqvae_codec.py` | Rewrite without species |

### 🔄 Rewrite
| Component | New Location | What Changes |
|-----------|-------------|--------------|
| `PayloadConfig` | `core/config.py` | Remove species/bbox/pose/shape fields; just version + CRC + VQ tokens |
| `PayloadFields` | `core/payload.py` | Remove species/confidence/mode fields |
| `ConditionalDecoder` | `models/decoder.py` | Replace species conditioning with unconditional decoder |
| `VQVAECodec` | `codecs/vqvae_codec.py` | Remove species, structure encoder, rate controller complexity; start minimal |
| Dataset | `data/dataset.py` | New `UnderwaterImageDataset` for arbitrary unlabeled images |
| Oracle | `codecs/oracle.py` | New image-level oracle (no species): transmit tiny DCT/pixel summary → decode |
| `UWCodec` | `core/codec.py` | Clean API: `encode(image, max_bytes) → bytes`; `decode(bytes) → RGB image` |
| Config | `core/config.py` | Remove all fish fields; add resolution sweep config |

---

## New Payload Layout

Fish fields wasted precious bytes. New minimal layout:

| Field | Bytes | Purpose |
|-------|-------|---------|
| version | 1 | Protocol version + mode flags |
| CRC-8 | 1 | Integrity |
| VQ tokens | max_bytes - 2 | Everything else is latent data |

**At 64B: 62 bytes for VQ tokens.**
**At 124B: 122 bytes for VQ tokens.**

> [!IMPORTANT]
> This is a radical simplification. The old layout wasted 34 bytes on species/bbox/pose/shape/structure/colormap headers that are meaningless for general images.

---

## Information-Theoretic Reality Check

| Budget | Bits available | Pixels reconstructed | Bits/pixel (128×128) |
|--------|---------------|---------------------|----------------------|
| 64B | 512 bits | 16,384 pixels (128²) | 0.03 bpp |
| 96B | 768 bits | 16,384 pixels | 0.047 bpp |
| 124B | 992 bits | 16,384 pixels | 0.060 bpp |

**JPEG starts working around 0.1-0.5 bpp. We are 2-10× below that.** Reconstruction will be coarse/semantic. This must be documented honestly.

---

## Development Order (Fail-Fast)

### Step 1 — Dataset loader (no labels needed)
`data/dataset.py`: `UnderwaterImageDataset` — scans any directory for `*.jpg/*.png`, resizes to target resolution, no labels.

### Step 2 — Clean payload (no fish metadata)
Rewrite `core/config.py` + `core/payload.py` with minimal 2-byte overhead.

### Step 3 — Clean decoder (unconditional)
Replace species-conditioned decoder with a simple upsampling CNN decoder. No FiLM, no species embedding.

### Step 4 — Minimal VQ-VAE codec (NEW `codecs/vqvae_codec.py`)
```
encoder(3,H,W) → z(D,h,w) → VQ(codebook=256) → indices → pack(≤124B)
                                                           ↑
                             serialize(indices, budget)
```
No rate controller, no structure encoder, no species. Prove it works first.

### Step 5 — General image oracle (fail-fast experiment)
Before training: encode image as tiny DCT/pixel block at target budget → decode.
If the oracle output is visually complete rubbish → report limit honestly.
If it shows structure → proceed to learned training.

### Step 6 — Tiny training run
500-2,000 real underwater images × 5-10 epochs.
Loss: L1 + perceptual + VQ commitment.
Measure: PSNR, SSIM, LPIPS. If no learning → stop and report.

### Step 7+ — Iterative improvements (only if Step 6 passes)
Product VQ → Residual VQ → entropy coding → rate controller → stronger losses.

---

## Architecture (Post-Redesign)

```
Input RGB (H, W, 3) — any size
    ↓ resize to target (128×128, 256×256, etc.)
    ↓ optional color correction
CNN Encoder (MobileNet-style, ~150K params)
    ↓ z: (D, h, w) e.g. (64, 8, 8)
Vector Quantizer (256-entry codebook)
    ↓ indices: (h*w,) integers 0-255
Serialize → exactly max_bytes - 2 bytes
    + 1B version
    + 1B CRC-8
= exactly max_bytes bytes ✅

BLE TX → reassemble → verify CRC

Decoder CNN (larger, receiver-side)
    ↓ lookup codebook → z_q: (D, h, w)
    ↓ upsample to (H, W)
RGB image out
```

---

## New API

```python
from uwcodec import UWCodec

codec = UWCodec.load("model_path")

# Encode any RGB image
payload = codec.encode(image, max_bytes=124)
assert len(payload) <= 124  # HARD ENFORCEMENT

# Decode on receiver
reconstructed = codec.decode(payload)  # returns np.ndarray (H, W, 3) uint8

# Rate-distortion sweep
for budget in [64, 96, 124, 256, 512, 1024, 2048, 4096]:
    p = codec.encode(image, max_bytes=budget)
    r = codec.decode(p)
    # evaluate PSNR/SSIM/LPIPS
```

---

## Files That Will Change

### New files
- `data/dataset.py` → rewrite (keep filename, replace content)
- `codecs/general_oracle.py` → new general image oracle
- `codecs/vqvae_codec.py` → rewrite (minimal, no species)
- `core/codec.py` → rewrite (clean UWCodec API)
- `core/config.py` → rewrite (remove fish fields)
- `core/payload.py` → rewrite (minimal 2B overhead)
- `models/decoder.py` → rewrite (unconditional decoder)
- `training/train_codec.py` → rewrite (no species collate)
- `scripts/run_oracle.py` → rewrite (general oracle experiment)
- `scripts/run_quick_experiment.py` → new fail-fast experiment
- `tests/test_general_codec.py` → new general image tests
- `examples/applications/fish_codec/` → move all fish code here

### Files unchanged
- `models/encoder.py` — keep
- `models/quantizer.py` — keep
- `codecs/rate_controller.py` — keep
- `ble/` — keep
- `deployment/` — keep
- `evaluation/` — keep
- `data/preprocessing.py` — keep
- `data/palette.py` — keep
- `baselines/` — keep
- `training/losses.py` — keep (remove species/teacher references)
- `training/schedulers.py` — keep

---

## Open Questions (No Action Needed — Just Documenting)

1. **Target resolution**: Start with 128×128. Then experimentally test 256×256, 320×240, 320×320 once basic codec works.
2. **Codebook size**: Start with 256 (1B/index). Test 512/1024 after baseline.
3. **VQ indices at 64B budget**: 62 indices max (1B each). At 8×8 spatial = 64 positions → just barely fits with 1-2 residual levels.
4. **Do real underwater datasets exist locally?** Need user to point to data directory or we use a synthetic/placeholder dataset for the fail-fast experiment.

---

## What I Will NOT Do Yet
- No 200-epoch training
- No Product VQ / Residual VQ (Phase 2+)
- No BioCLIP teacher
- No entropy coding
- No STM32 ONNX export
- No rate controller complexity
- No fabricated benchmark numbers
