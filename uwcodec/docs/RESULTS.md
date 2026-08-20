# UWCodec Results

This document tracks measured results across the different phases of the project.

## Phase 2: General Oracle Baseline
*Note: Due to datasets currently being downloaded, these initial numbers are from a small synthetic validation subset. This section will be updated when the real EUVP dataset is processed.*

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
- As expected, traditional signal-processing approaches like DCT show marginal improvements as the budget scales up to 1KB.
- However, at the extreme constraint of 64B–124B (approx. 0.03–0.06 bpp), all unlearned strategies struggle to exceed 27 dB. 
- The images look extremely blocky and lack structural coherence.
- This confirms the necessity of a learned semantic prior (VQ-VAE) to reconstruct coherent structure at these budgets.

---

*(Future results for Baselines, Ablations, and BLE to be appended here)*

## UWCodec v2 128B Final Evaluation (Phase 1)
**Goal**: Evaluate the trained 128B checkpoint (outputs/v2/budget_128_final/best.pt) on the EUVP held-out set using the exact encode -> bytes -> decode path.

### Metrics (EUVP Held-out, 2305 images)
- **PSNR**: 19.7981 dB
- **SSIM**: 0.4433
- **MS-SSIM**: 0.7115
- **UCIQE**: 0.1554
- **Payload Size**: Exactly 128 bytes verified.

### Visual Quality Assessment
A 40-image visual comparison grid was generated (`outputs/v2/budget_128_final/visual_grid.png`) and manually inspected:
- **Structural Preservation**: Very poor. While general color layouts are preserved, fine structure is completely lost.
- **Object Shape**: Fish, starfish, and corals are reduced to vaguely recognizable color blobs.
- **Hallucinations**: There are severe, ubiquitous "ring-like" artifacts dominating the reconstructions.
- **Conclusion**: The current 128B model fails the visual gate. It is structurally unusable for downstream detailed tasks.

**Status**: **FAIL** (Stopped at Phase 1. Diagnosis required before proceeding to smaller budgets or generalization tests).
