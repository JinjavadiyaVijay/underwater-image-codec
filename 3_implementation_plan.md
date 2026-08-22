# UWCodec — Repository Audit & Implementation Plan

## 1. Audit Findings

### 1.1 Repository State
- **Git**: Clean on `main` branch (single branch). 8 commits. No uncommitted changes to tracked files.
- **Untracked external**: `../Autoencode-D64-v0.1.0/` and `../__MACOSX/` outside the repo (not our concern).
- **No active training process** — the only Python process is the Jedi LSP (IDE language server, PID 15952).

### 1.2 Nearest-Neighbor Experiment Status — ⚠️ INCOMPLETE

The `budget_128_nearest_final` experiment **did NOT complete 50 epochs**:

| Evidence | Finding |
|---|---|
| Checkpoints | `best.pt` (epoch 13), `epoch_010.pt` only |
| `training_history.json` | **Missing** |
| `evaluation_euvp.json` | **Missing** |
| `final.pt` | **Missing** |
| `visual_grid.png` | **Missing** |
| Running process | **None** |

**Conclusion**: Training crashed or was interrupted around epoch 13. The `best.pt` checkpoint at epoch 13 is usable for evaluation but the experiment is incomplete.

### 1.3 V2 Experiment Inventory

| Directory | Epochs | Has Eval | Has Grid | Status |
|---|---|---|---|---|
| `v2/budget_128` | Short run | ❌ | ❌ | Early experiment |
| `v2/budget_128_final` | 50 (complete) | ✅ (PSNR 19.80) | ✅ | **FAIL** — ring artifacts |
| `v2/budget_128_ema_final` | 50 (complete) | ✅ (PSNR 19.97) | ❌ | EMA variant |
| `v2/budget_128_nearest_final` | ~13 (incomplete) | ❌ | ❌ | Crashed/interrupted |
| `v2/quantizer_fix_128` | Short run | ❌ | ❌ | Diagnostic |

### 1.4 Other Output Directories

| Directory | Contents | Classification |
|---|---|---|
| `outputs/ablations/` | 5 ablation subdirs (baseline, fewer_channels, etc.) | KEEP (v1 experiments) |
| `outputs/baselines/` | `metrics.csv` (24KB) | KEEP |
| `outputs/multi_budget/` | budget_64, budget_96, budget_124 checkpoints (v1) | KEEP |
| `outputs/smoke_test/` | best.pt, final.pt | KEEP (but can archive) |
| `outputs/train/` | best.pt, final.pt (v1) | KEEP (but can archive) |
| `outputs/profiling/` | profiling_results.json | KEEP |
| `outputs/v2_gate1/` | budget_128 subdir | KEEP |
| `oracle_results/` | Oracle images + real subdir | KEEP |

### 1.5 Package Structure — File Classification

#### Active V2 Code (KEEP)
| File | Used By | Status |
|---|---|---|
| [v2_encoder.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/v2_encoder.py) | v2_codec, train_v2 | ✅ Active |
| [v2_decoder.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/v2_decoder.py) | v2_codec, train_v2 | ✅ Active |
| [quantizer.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/quantizer.py) | v2_codec, vqvae_codec | ✅ Active |
| [v2_codec.py](file:///S:/IMG_compressors/uwcodec/uwcodec/codecs/v2_codec.py) | train_v2, diagnostic | ✅ Active |
| [train_v2.py](file:///S:/IMG_compressors/uwcodec/uwcodec/training/train_v2.py) | CLI entry point | ✅ Active |
| [payload.py](file:///S:/IMG_compressors/uwcodec/uwcodec/core/payload.py) | v2_codec, vqvae_codec | ✅ Active |
| [config.py](file:///S:/IMG_compressors/uwcodec/uwcodec/core/config.py) | vqvae_codec, codec.py | ✅ Active |
| [dataset.py](file:///S:/IMG_compressors/uwcodec/uwcodec/data/dataset.py) | train_v2, train_codec | ✅ Active |
| [losses.py](file:///S:/IMG_compressors/uwcodec/uwcodec/training/losses.py) | train_codec | ✅ Active |
| [metrics.py](file:///S:/IMG_compressors/uwcodec/uwcodec/evaluation/metrics.py) | eval scripts | ✅ Active |
| BLE module (crc, mtu, packet) | payload.py | ✅ Active |

#### V1/Legacy Code (KEEP but clarify role)
| File | Used By | Status |
|---|---|---|
| [encoder.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/encoder.py) | vqvae_codec, __init__.py, tests | V1 encoder |
| [decoder.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/decoder.py) | vqvae_codec, __init__.py, tests | V1 decoder |
| [vqvae_codec.py](file:///S:/IMG_compressors/uwcodec/uwcodec/codecs/vqvae_codec.py) | core/codec.py, train_codec, tests | V1 codec |
| [codec.py](file:///S:/IMG_compressors/uwcodec/uwcodec/core/codec.py) | uwcodec __init__.py (public API) | V1 API wrapper |
| [train_codec.py](file:///S:/IMG_compressors/uwcodec/uwcodec/training/train_codec.py) | scripts, docs | V1 training |
| [general_oracle.py](file:///S:/IMG_compressors/uwcodec/uwcodec/codecs/general_oracle.py) | oracle scripts | Diagnostic tool |
| [rate_controller.py](file:///S:/IMG_compressors/uwcodec/uwcodec/codecs/rate_controller.py) | **UNUSED** (0 imports) | Dead code candidate |

#### Supporting Code (KEEP)
| File | Notes |
|---|---|
| Evaluation module (bio_metrics, structure_metrics, benchmark, visualize) | All useful |
| Baselines module (jpeg_webp, compressai, prototype, semantic_only) | All useful |
| Deployment module (export_onnx, export_stm32, profile) | Future use |
| Examples (quick_start, baseline_comparison, fish_codec) | Reference |

#### Root-Level Artifacts (CLEAN UP)
| File | Classification |
|---|---|
| [diagnostic.py](file:///S:/IMG_compressors/uwcodec/diagnostic.py) | **ARCHIVE** — one-off diagnostic script, hardcoded paths |
| `diagnostic_comparison.png` (375KB) | **ARCHIVE** — generated diagnostic image |
| `diagnostic_grid.png` (2.8MB) | **ARCHIVE** — generated diagnostic image |
| `uwcodec.egg-info/` | **DELETE** — regenerated by pip install |

### 1.6 Key Findings

1. **`rate_controller.py`** has zero imports anywhere — dead code.
2. **Root `diagnostic.py`** and PNG files are one-off debugging artifacts with hardcoded paths.
3. V1 code (`encoder.py`, `decoder.py`, `vqvae_codec.py`, `train_codec.py`, `codec.py`) is still referenced by tests and the public API (`UWCodec` class). It should be preserved but marked as V1.
4. `models/__init__.py` only exports V1 models — needs updating.
5. `examples/applications/fish_codec/` has legacy fish-specific code (classifier, detector, teacher, fish_oracle) — outdated.
6. No `ARCHITECTURE.md` doc exists yet.
7. The `pyproject.toml` has stale script entry points (`scripts.run_oracle`, `scripts.train`, `scripts.evaluate`) that likely don't work.

---

## 2. User Review Required

> [!IMPORTANT]
> **Nearest-neighbor experiment is INCOMPLETE.** The training stopped at ~epoch 13. We have two options:
> 1. **Evaluate the epoch-13 checkpoint** as-is (best checkpoint available) to make the V2 PASS/FAIL decision now.
> 2. **Re-run the nearest-neighbor training** for the full 50 epochs before evaluation.
> 
> Given that the V2 architecture has already been diagnosed as fundamentally limited by the 4×4/8×8 spatial representation, and the previous complete V2 run (`budget_128_final`, 50 epochs) already FAILED the visual gate with PSNR 19.80 and severe ring artifacts, I recommend **Option 1**: evaluate epoch-13 nearest-neighbor checkpoint now. If it also fails (likely), we proceed to V3 without wasting further GPU time on V2.

> [!WARNING]
> **Fish-specific legacy code** exists in `examples/applications/fish_codec/` (classifier, detector, teacher, fish_oracle). This appears to be from an earlier fish-species-specific design. Should this be:
> - **Archived** to `examples/archive/fish_codec/`
> - **Deleted** entirely
> - **Kept** as-is

---

## 3. Proposed Changes

### Phase 1: Safe Cleanup (No GPU, No Training)

---

#### Root Directory

##### [ARCHIVE] diagnostic.py, diagnostic_comparison.png, diagnostic_grid.png
- Move to `outputs/archive/v2_diagnostics/`

##### [DELETE] uwcodec.egg-info/
- Regenerated automatically; no value in keeping.

---

#### uwcodec/codecs/

##### [ARCHIVE] rate_controller.py
- Zero imports. Move to `outputs/archive/v1_code/` or delete.

##### [MODIFY] [__init__.py](file:///S:/IMG_compressors/uwcodec/uwcodec/codecs/__init__.py)
- Update docstring to reflect current state.

---

#### uwcodec/models/

##### [MODIFY] [__init__.py](file:///S:/IMG_compressors/uwcodec/uwcodec/models/__init__.py)
- Add V2 model exports alongside V1.

---

#### outputs/

##### [ARCHIVE] outputs/smoke_test/, outputs/train/
- Move to `outputs/archive/v1_training/`

##### [ARCHIVE] outputs/v2/budget_128/, outputs/v2/quantizer_fix_128/
- Move to `outputs/archive/v2_early/` (early/diagnostic runs)

---

#### docs/

##### [NEW] docs/ARCHITECTURE.md
- Document V2 architecture, known limitations, and V3 design rationale.

##### [MODIFY] [docs/RESULTS.md](file:///S:/IMG_compressors/uwcodec/docs/RESULTS.md)
- Add V2 EMA results, nearest-neighbor evaluation results (once evaluated).

---

#### scripts/

- Keep all current scripts. They are diagnostic/utility scripts with valid uses.
- `__pycache__/` in scripts will be removed.

---

#### pyproject.toml

##### [MODIFY] [pyproject.toml](file:///S:/IMG_compressors/uwcodec/pyproject.toml)
- Fix stale entry points or remove them.

---

### Phase 2: V2 Nearest-Neighbor Evaluation Gate

Run the following on the epoch-13 nearest-neighbor checkpoint:

1. Load `outputs/v2/budget_128_nearest_final/best.pt`
2. Encode → exact 128 bytes → decode all 2305 EUVP validation images
3. Compute: PSNR, SSIM, MS-SSIM, DISTS, UCIQE, payload size
4. Generate visual grid
5. Assess: ring artifacts, blocking, structural quality
6. Save results to `outputs/v2/budget_128_nearest_final/`
7. Make PASS/FAIL decision
8. Update `docs/RESULTS.md`

---

### Phase 3: V3 Design (if V2 FAIL)

Begin V3 TiTok-style 1D tokenizer architecture per the specification:
- 128×128 → 64 latent tokens → 4096-entry codebook → 128B
- ViT/hybrid encoder → 1D VQ → Transformer decoder → pixel decoder
- Separate experiment: `configs/v3_128_titok64.yaml`

> [!NOTE]
> Phase 3 will only be executed after Phase 2 evaluation and explicit PASS/FAIL decision.

---

## 4. Verification Plan

### Phase 1 Verification
- All existing tests pass: `conda run -n uwcodec_gpu pytest tests/`
- All imports resolve correctly
- No files are broken by archival

### Phase 2 Verification
- Payload size is exactly 128B for every image
- Visual grid generated and manually inspectable
- Metrics JSON saved
- RESULTS.md updated

### Phase 3 Verification
- V3 model forward pass produces correct shapes
- V3 serialization produces exactly 128B
- V3 training runs without OOM on RTX 3050 6GB
