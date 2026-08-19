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

*(Future results for Training, Baselines, Ablations, and BLE to be appended here)*
