# UWCodec Architecture

This document describes the architecture of UWCodec — an ultra-low-bandwidth learned underwater image codec.

## Overview

UWCodec compresses a 128×128 RGB image into a fixed-size byte payload (64–128 bytes) using learned Vector Quantization, enabling real-time underwater image transmission over extremely bandwidth-constrained links (BLE, acoustic modem).

```
Input (128×128 RGB)
    → Encoder
    → Latent representation
    → Vector Quantization
    → Discrete token IDs
    → Fixed-size serialization
    → Byte payload (64–128B)
    → Deserialization
    → Codebook lookup
    → Decoder
    → Output (128×128 RGB)
```

---

## V1 Architecture (MinimalVQVAE)

Single-branch VQ-VAE. Simple and general.

| Component | Description | Parameters |
|---|---|---|
| Encoder | MobileNet-style CNN → (B, D, H', W') | ~100K |
| Quantizer | Single VQ, 256-entry codebook | ~16K |
| Decoder | CNN upsampler with residual blocks | ~400K |

Payload: VQ indices packed as uint8. Budget = 2B overhead + N bytes VQ data.

**Status**: Baseline. Works for evaluation but reconstruction quality is extremely coarse at 64–128B.

---

## V2 Architecture (UWCodecV2)

Dual-branch semantic + detail encoder with FiLM-conditioned UNet decoder.

### Encoders

| Branch | Output | Stages | Parameters | Purpose |
|---|---|---|---|---|
| Semantic | (B, 64, 4, 4) | 5× stride-2 (128→4) | ~550K | Global structure, scene geometry |
| Detail | (B, 32, 8, 8) | 4× stride-2 (128→8) | ~35K | Edges, textures, local features |

Both use MobileNet-style depthwise separable convolutions (ReLU6, BatchNorm).

### Quantization

| Branch | Method | Codebook | Bytes per level |
|---|---|---|---|
| Semantic | Residual VQ (2–3 levels) | 256 entries, dim=64 | 16B per level (4×4 grid) |
| Detail L1 | VQ | 256 entries, dim=32 | 30–64B (subset of 8×8) |
| Detail L2 | VQ (residual) | 256 entries, dim=32 | 0–14B |

**Budget allocation** (2B header: version + CRC):

| Budget | Sem RVQ | Det L1 | Det L2 | Data | Total |
|---|---|---|---|---|---|
| 64B | 2×16B | 30B | 0B | 62B | 64B |
| 96B | 2×16B | 62B | 0B | 94B | 96B |
| 124B | 3×16B | 64B | 10B | 122B | 124B |
| 128B | 3×16B | 64B | 14B | 126B | 128B |

### Decoder (~5.9M parameters)

UNet-style generative decoder with FiLM conditioning:

```
Semantic (4×4, 256ch) → 4 ResBlocks
    → Upsample → 8×8, 128ch
    → FiLM conditioning from detail (8×8)
    → 2 ResBlocks
    → Upsample → 16×16 → 32×32 → 64×64 → 128×128
    → Output head → RGB [0,1]
```

- GroupNorm + SiLU activations (stable for small batches)
- FiLM (Feature-wise Linear Modulation) injects detail features as per-channel scale+shift
- FiLM initialized to identity (gamma≈0, beta≈0) for stable early training
- Nearest-neighbor upsampling (bilinear produced ring artifacts)

### Known V2 Limitations

1. **Ring/topographical artifacts**: Bilinear decoder interpolation produced severe repeating ring patterns. Diagnosed to be decoder-side, NOT codebook or serialization related.
2. **Fundamental representation limit**: The 4×4 + 8×8 spatial latent grid is extremely coarse. Even with perfect quantization, reconstructing 128×128 from 16 + 64 spatial positions is extremely aggressive.
3. **EMA + Adam conflict**: Previously diagnosed. VQ codebook must use EMA updates with detached tensors; simultaneous Adam optimization of codebook weights causes instability.

**V2 Status**: 128B final evaluation FAILED visual gate (PSNR 19.80, severe ring artifacts). Nearest-neighbor decoder variant being evaluated.

---

## V3 Architecture (Planned — TiTok-inspired 1D tokenizer)

If V2 fails the visual gate, V3 adopts a 1D token representation inspired by the TiTok family of image tokenizers.

### Core Design

```
128×128 RGB
    → Patch embedding
    → Vision Transformer / hybrid visual encoder
    → 64 learned 1D latent tokens
    → Vector Quantization (4096-entry codebook)
    → 64 discrete token IDs
    → Serialization: 64 × uint16 = 128 bytes
    → Deserialization
    → Codebook lookup
    → Transformer decoder
    → Learned pixel decoder
    → 128×128 RGB
```

### Key Differences from V2

| Aspect | V2 | V3 |
|---|---|---|
| Latent shape | 2D spatial (4×4 + 8×8) | 1D tokens (64 tokens) |
| Codebook size | 256 (8-bit) | 4096 (12-bit) |
| Token ID size | uint8 (1B) | uint16 (2B) |
| Encoder | CNN (MobileNet-style) | ViT / hybrid |
| Decoder | UNet + FiLM | Transformer + pixel decoder |

### Future Optimization: 12-bit Packing

Since log₂(4096) = 12 bits, packed serialization yields:

```
64 tokens × 12 bits = 768 bits = 96 bytes
```

Leaving 32B for FEC/checksum/metadata. This is a later optimization after 128B quality is proven with uint16.

---

## Payload Format

### Header (2 bytes)

```
Byte 0:  Version (protocol version, future flags)
Byte N-1: CRC-8 (integrity check over bytes 0..N-2)
```

### Data (N-2 bytes)

VQ token indices packed sequentially. Format depends on codec version:

- **V1/V2**: uint8 indices (codebook size 256)
- **V3**: uint16 indices (codebook size 4096)

### Hard Rule

```python
assert len(payload) == budget  # ALWAYS enforced
```

---

## Training

See [TRAINING.md](TRAINING.md) for training commands and configuration.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for metrics and evaluation protocol.
