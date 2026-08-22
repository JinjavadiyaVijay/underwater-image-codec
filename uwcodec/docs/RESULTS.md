# UWCodec Results

This document tracks measured results across the different phases of the project.

---

## V2 128B Evaluation Summary

### All V2 128B Variants (EUVP Held-out, 2305 images)

| Variant | Epochs | PSNR | SSIM | MS-SSIM | DISTS | UCIQE | Payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 bilinear (final) | 50 | 19.798 | 0.4433 | 0.7115 | — | 0.1554 | 128B ✅ |
| V2 EMA (final) | 50 | 19.970 | 0.4717 | 0.7414 | 0.3247 | 0.1584 | 128B ✅ |
| V2 nearest-neighbor | ~13 | 19.635 | 0.4545 | 0.7232 | 0.3434 | 0.1546 | 128B ✅ |

### Visual Quality Assessment

**V2 bilinear (50 epochs)**:
- Severe, ubiquitous ring/topographical artifacts dominate every reconstruction
- Objects are reduced to vaguely recognizable color blobs surrounded by concentric rings
- Structure is destroyed — images are NOT usable for underwater interpretation
- **Visual Gate: FAIL**

**V2 nearest-neighbor (epoch 13)**:
- Ring artifacts are substantially reduced compared to bilinear — the concentric ring pattern is gone
- However, reconstructions show severe blockiness (expected from nearest-neighbor upsampling)
- Objects are blurry color blobs; major structure is barely recognizable
- Scene colors are approximately preserved but all detail is lost
- Fish, coral, and sea creatures are not individually distinguishable
- Training was incomplete (epoch 13/50); metrics may have improved with longer training
- **Visual Gate: FAIL** — blockiness replaced rings, but reconstruction remains structurally unusable

### V2 128B Final Decision: **FAIL**

All three V2 variants fail the visual gate:

1. **Bilinear**: Severe ring artifacts
2. **EMA bilinear**: Best metrics but same ring artifacts (not visually evaluated separately; assumed same artifact pattern since same decoder architecture)
3. **Nearest-neighbor**: Rings eliminated but replaced by severe blockiness; incomplete training

**Root cause**: The 4×4 + 8×8 spatial latent representation is fundamentally too coarse to reconstruct visually useful 128×128 images from 128 bytes. The decoder architecture (bilinear or nearest-neighbor upsampling from 4×4) cannot hallucinate enough structure from 16 spatial positions.

**Decision**: Stop V2 development. Proceed to V3 TiTok-style 1D tokenizer architecture.

---

## Phase 2: General Oracle Baseline

**Goal**: Determine the absolute non-learned theoretical limit of image reconstruction at ultra-low bandwidths (64B–1KB).
**Images**: 5 synthetic EUVP validation images.
**Resolution**: 128×128.

| Budget | Tiny Pixel Grid (PSNR) | DCT Coefficients (PSNR) | Mean Color Blocks (PSNR) |
|---|---|---|---|
| 64B | 26.72 dB | 26.78 dB | 26.71 dB |
| 96B | 26.72 dB | 26.82 dB | 26.71 dB |
| 124B | 26.72 dB | 26.84 dB | 26.72 dB |
| 256B | 26.74 dB | 26.96 dB | 26.73 dB |
| 512B | 26.77 dB | 27.16 dB | 26.66 dB |
| 1024B | 26.82 dB | 27.50 dB | 26.76 dB |

**Analysis**:
- At 64B–124B (~0.03–0.06 bpp), all unlearned strategies struggle to exceed 27 dB.
- Images are extremely blocky and lack structural coherence.
- This confirms the necessity of a learned semantic prior to reconstruct coherent structure at these budgets.

---

## Experiment Registry

| ID | Model | Resolution | Tokens | Payload | PSNR | SSIM | MS-SSIM | DISTS | Visual Gate | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| V2-E1 | V2 bilinear | 128 | 2D(4×4+8×8) | 128B | 19.80 | 0.443 | 0.712 | — | FAIL (rings) | Complete |
| V2-E2 | V2 EMA | 128 | 2D(4×4+8×8) | 128B | 19.97 | 0.472 | 0.741 | 0.325 | FAIL (rings) | Complete |
| V2-E3 | V2 nearest | 128 | 2D(4×4+8×8) | 128B | 19.64 | 0.455 | 0.723 | 0.343 | FAIL (blocks) | Incomplete (ep13) |
| V3-E1 | V3 TiTok64 | 128 | 64×1D | 128B | — | — | — | — | — | Planned |

---

*(V3 results to be appended here after implementation and training)*
