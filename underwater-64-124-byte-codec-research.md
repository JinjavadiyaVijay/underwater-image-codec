# 64–124 Byte Underwater Image Transmission: Feasibility, Architecture, and Research Plan

**Prepared for:** FishVision-AI / STM32N6570-DK + MB1854B + BLE fish/lobster ID system
**Scope:** Research synthesis + engineering recommendation, grounded in 2024–2026 literature.

---

## Part 0 — The One Number That Determines Everything

Everything in this report hinges on one ratio: **bits per pixel (bpp)**, where `bpp = payload_bits / pixels_to_reconstruct`. The size of the *original* 500 KB image is irrelevant — sensor resolution, JPEG artifacts, file headers, none of that matters. What matters is how many pixels you're asking the decoder to fill in, because that's what fixes how many bits per pixel you get to spend.

At **100 bytes = 800 bits**, here's what bpp you get depending on what you choose to reconstruct:

| Target output | Pixels | bpp at 100 B | bpp at 64 B | bpp at 124 B | Where this sits vs. published SOTA |
|---|---|---|---|---|---|
| Full scene, 768×512 (Kodak-standard) | 393,216 | 0.0020 | 0.0013 | 0.0025 | **Below** the lowest bpp anyone has published real per-pixel reconstruction at. Text+Sketch (2023) — the most extreme case in the literature — operates at ~0.003 bpp and produces a *semantically similar*, not pixel-similar, image.<cite index="37-1">Text and sketch–based compression can represent an image using an extremely low bitrate below 0.003 bits-per-pixel while still preserving meaningful semantics.</cite> |
| Full scene, 512×512 | 262,144 | 0.0031 | 0.0020 | 0.0038 | Same territory — caption-level only |
| Fish crop, 256×256 | 65,536 | 0.0122 | 0.0078 | 0.0151 | Right at the frontier. SOTA generative codecs (GLC, DLF, PerCo, MISC) report <cite index="16-1">high visual quality with less than 0.04 bpp on natural images and less than 0.01 bpp on facial images</cite> and <cite index="5-1">impressive reconstruction quality even below 0.01 bpp</cite>, but these use full Stable-Diffusion-scale decoders (~1–2 GB) and are demonstrated on curated benchmarks (Kodak, CLIC, faces) — not underwater fish crops. Achievable in principle; is genuinely research-frontier. |
| Fish crop, 128×128 | 16,384 | 0.049 | 0.031 | 0.061 | **Comfortably inside** the demonstrated range of ultra-low-bitrate generative codecs (0.02–0.06 bpp is where MISC, PerCo, SEDIC, DiffEIC routinely operate).<cite index="79-1">SEDIC operates in the 0.02–0.05 bpp range depending on configuration.</cite> |
| Fish crop, 64×64 | 4,096 | 0.195 | 0.125 | 0.242 | Well within reach of ordinary learned image compression (CompressAI-class models, non-generative), which routinely operates at 0.1–0.5 bpp with good fidelity. This is an *easy* regime. |

**This table is the whole feasibility argument in one place.** Trying to reconstruct a full scene at 64–124 bytes is asking for compression ~10× more extreme than anything demonstrated in the literature, and what *is* demonstrated at that extremity (Text+Sketch, MISC) explicitly produces a different image that merely shares a caption with the original — not a trustworthy reconstruction of *this specific fish*. Cropping to the subject and reconstructing a small canonical patch moves you into a regime that multiple 2024–2026 papers have actually built and measured.

**This is why the object-centric / hybrid-structured architecture (your options G and I) is not just "also worth considering" — it's the only path in which "64–124 bytes → visually useful, biologically trustworthy RGB image" is not simply false advertising.**

---

## Part 1 — Feasibility: The Honest Answer

**Can a useful RGB underwater image be reconstructed from 64–124 bytes? Yes, but only if "image" means "a canonical crop of the subject," and only if "reconstructed" means "regenerated from strong priors plus a small amount of true per-image information," not "recovered."**

### What's theoretically possible
- Shannon's source-coding theorem doesn't forbid this — it just says you can't losslessly encode more information than your entropy allows. A 64×64 RGB fish crop has *far* less true entropy than its 12,288 raw bytes (8bpp) would suggest: fish shape is heavily constrained by taxonomy, color palettes are heavily constrained by species and depth/turbidity, and most of the pixel-level variance across a species' population is redundant with a shared prior. If the receiver already "knows" what fish tend to look like (a decoder trained on thousands of images), the transmitter only needs to send the *residual* — which species, what pose, what's different about this individual — not the pixels.
- This is exactly the logic behind every generative low-bitrate codec in Part 2: the model is the majority of the "information," and it's paid for once (shipped with firmware), not per image.

### What's impossible
- **True lossless or near-lossless reconstruction of an arbitrary 500 KB image from 100 bytes is impossible**, full stop — that's a ~5,000:1 ratio with no prior strong enough to make up the gap for *arbitrary* content. Nothing in the literature claims this, and nothing should.
- **Reconstructing information that isn't correlated with the prior** (a scar, a tag, an anomaly, a novel unidentified species, debris in frame, a diver's hand) is impossible — the decoder will either drop it or hallucinate something plausible-but-wrong in its place. This is the central risk for a biological application (Part 6).
- **Full-scene reconstruction with individually-correct background content** (substrate, other fish, water column texture) at this rate is impossible; at best you get a *generic underwater backdrop*, not *this dive site's* backdrop.

### What can only be reconstructed because of a strong prior
- Fin shape, body proportions, typical coloration, typical markings *for the identified species* — all of this comes from the shared decoder's training data, not from the transmitted bytes. The transmitted bytes are essentially a **key into that prior**, plus a small correction term.
- This is precisely why species ID must be treated as **explicit, verified, transmitted data** (an integer, chosen by your existing classifier) rather than something the decoder infers from a fuzzy embedding — see Part 3 and Part 6.

### What happens when the input isn't represented by the prior
- Out-of-distribution inputs (a species outside your 157-class set, an unusual pose, a heavily occluded animal, non-fish content) will produce a reconstruction that looks like *the nearest thing the decoder knows how to draw*, with a plausibility that has zero correlation to correctness. This is a known, documented failure mode of every generative low-bitrate codec surveyed here (Careil et al. call it "compromising the fidelity of the reconstructions" due to the randomness introduced when initiating denoising from pure noise <cite index="1-1">initiating the denoising process from pure noise introduces significant randomness, compromising the fidelity of the reconstructions</cite>), and it's the reason you need explicit confidence signaling, not just image output, at the receiver.

### What is fundamentally, unavoidably lost
- Exact pixel values, camera noise texture, precise water-column effects, anything not correlated with species/pose/coarse-color, exact individual markings beyond what a small residual code can capture, background scene content, EXIF/context. This should be treated as *lost by design*, not as a bug to eventually fix.

### What "reconstruction" actually means at 64–124 bytes
It means: **"a plausible, species-correct, roughly-posed, roughly-colored image consistent with the transmitted category and coarse parameters — not a recovery of the original pixels."** This is closer to *procedural regeneration conditioned on a semantic key* than to *compression* in the classical sense. Framing it this way to any downstream user of the reconstructed image (and in your own documentation/paper) is not just intellectually honest, it's a safety requirement given Part 6.

### Distinguishing the terms you asked about (Section 22)
| Term | What it is | Applies here? |
|---|---|---|
| **Embedding** | A vector (e.g. CLIP, 512–768-dim float) representing semantic content; not designed to be compact or to reconstruct pixels | A *component* — but a raw CLIP embedding is ~1,500–3,000 bytes even at low quantization, so a full CLIP vector doesn't fit your budget (CoCliCo quantizes to a handful of bits/dim — see Part 2) |
| **Latent representation** | Compressed intermediate representation from an encoder, typically still needs the paired decoder to have any meaning | The VQ token indices in Part 3 are this |
| **Compressed bitstream** | Entropy-coded, typically variable-length, designed for exact (or near-exact) reconstruction under a specific decoder | Not what we're building — 64–124 B is a **fixed-size structured record**, not a general bitstream |
| **Semantic representation** | Category/attribute-level description (species, pose, color) rather than pixel-level | The species ID + shape/pose/color fields in Part 3 |
| **Image reconstruction** | Actual pixel array output | The final decoder output, produced by combining all of the above with a shared generative prior |

Our recommended design (Part 3) is explicitly a **semantic + latent hybrid**, not a compressed bitstream of the image. This must be represented accurately in any documentation — calling it "image compression at 5000:1" without this caveat would be misleading.

---

## Part 2 — Existing Approaches: Research Survey

### A/B/C. Learned compression, VAEs, VQ-VAE (classical + hyperprior)
Ballé-style hyperprior models (the basis of CompressAI) are extremely well-understood but were designed for the 0.1–2 bpp range. Below ~0.1 bpp they degrade badly — <cite index="7-1">at extremely low bitrates below 0.1 bpp, VAE-based methods tend to produce severe blurriness, while GAN-based methods can introduce erroneous textures</cite>. **CompressAI/hyperprior alone cannot reach 64–124 bytes on any image larger than a small icon.** They're the right tool for your *256B–2KB* operating points (Part 5), not for 64–124B directly, but useful as an encoder backbone feeding into a generative decoder (see D/E below).

### D/E. Semantic communication, generative reconstruction — this is where the real 2024–2026 progress is
The dominant 2024–2026 paradigm at extreme bitrates is: **compress to a semantic/structural key + tiny side-information, decode with a large pretrained generative prior.** Representative systems:

| System | Year | What's transmitted | Reported bpp | Decoder | Code/models | Hallucination handling |
|---|---|---|---|---|---|---|
| **Text+Sketch** | 2023 | CLIP-space text description + binary edge sketch | <cite index="37-1">below 0.003 bpp</cite> | Pretrained text-to-image diffusion (Stable Diffusion) | [Open](https://github.com/leieric/Text-Sketch) <cite index="32-1">GitHub - leieric/Text-Sketch: Code for Text + Sketch: Image Compression at Ultra Low Rates</cite> | Weak — text alone is very lossy for a specific instance |
| **CoCliCo** | 2024 | CLIP latent vector (quantized) + tiny color map | <cite index="64-1">~10⁻² bpp</cite> | Conditional diffusion, with explicit color guidance | Inria, not fully open | Color map materially reduces hallucination vs. text-only <cite index="61-1">keeps most of the high-level information and a good level of realism</cite> |
| **MISC** | 2024 | LMM (GPT-4V) caption + spatial/positional map + extremely-compressed base image | <cite index="80-1">achieves optimal consistency and perception while saving 50% bitrate vs. prior methods, at bitrates below ~0.024 bpp</cite> | LIC (Cheng2020) + 4-step diffusion refinement | [To be released](https://github.com/lcysyzxdxc/MISC) | Best-in-class consistency because it keeps a real (if tiny) image, not text alone |
| **PerCo / PerCoV2** | 2023–2024 | VQ latent + short caption | <cite index="68-1">competitive at <0.04 bpp</cite> | Latent diffusion (Stable Diffusion) | Open (PerCoV2) | Moderate |
| **GLC (Generative Latent Coding)** | 2024 | Transform-coded VQ-VAE generative latent | <cite index="16-1">less than 0.04 bpp on natural images, less than 0.01 bpp on facial images</cite> | VQ-VAE decoder | CVPR 2024, code referenced | Moderate — closed-domain (faces) performs best, hinting that *narrow domain = better fidelity at extreme bitrate*, which matters for you |
| **SEDIC** | 2025 | LMM-disentangled semantic streams (object text + masks + reference) | <cite index="79-1">0.02–0.05 bpp</cite> | ControlNet + diffusion | Open (arXiv 2503.00399) | Explicitly designed to reduce "natural-looking but source-inconsistent" hallucination via disambiguation |
| **DLF (Dual-generative Latent Fusion)** | 2025 | Semantic tokens + detail tokens (two branches) | <cite index="5-1">below 0.01 bpp</cite> | Dual-branch generative decoder | ICCV 2025 | Detail branch specifically targets "perceptually critical details" to reduce hallucination |
| **AEIC (shallow encoder)** | 2025 | Small-encoder latent | <cite index="8-1">below 0.05 bpp, designed for weak/edge sender devices</cite> | One-step diffusion | arXiv 2512.12229 | Directly relevant: this is the first paper explicitly targeting **cheap encoders** (your STM32N6 constraint) rather than assuming a beefy transmitter |

**Key finding relevant to your architecture:** independent groups converge on the same recipe — *a compact semantic/category signal + a small amount of explicit structural/color side-information + a shared generative decoder* — and every paper that adds explicit structure (color map, mask, edge sketch) over pure-text/pure-latent approaches reports **better consistency and less hallucination**, at the cost of a few more bytes. This directly supports your instinct in Section 8I (hybrid) over 8E (pure generative) or 8D (pure semantic).

**Important nuance from the most recent work (2026):** a newer paper (CoD, arXiv 2511.18706) found that **text conditioning is not always beneficial for compression** — <cite index="33-1">fine-tuning Stable Diffusion to discard text priors can further improve compression efficiency, and text-conditioned diffusion models are not naturally suitable for compression</cite>. This matters for you: a free-text caption is the wrong semantic key for a closed-set problem. A **categorical species ID is strictly better** than a text caption here — it's unambiguous, costs ~1 byte instead of dozens, and doesn't inherit CLIP/LLM captioning's general-purpose imprecision. This is the basis of the architecture in Part 3.

### C (continued). VQ-VAE / codebook specifically
GLC <cite index="16-1">performs transform coding in the latent space of a generative vector-quantized variational auto-encoder (VQ-VAE) instead of in pixel space, which is more sparse, more semantic, and better aligned with human perception</cite> — this is a strong argument for VQ-token-based residual coding (used in Part 3) over raw pixel-space latents.

### F. Retrieval / prototype+residual
No paper in this survey does *exactly* "nearest-prototype ID + residual" for images the way you describe, but it is the implicit mechanism inside GLC, CoCliCo, and DLF's "semantic branch" — they all learn a discrete/quantized semantic code that functions as a soft prototype index. **Your explicit species-ID version of this idea (Part 3) is a harder, more literal, and more interpretable version of what these papers do implicitly** — this is a genuine point of novelty (Part 12).

### G. Object-centric transmission
No 2024–2026 paper surveyed here does "detect object, transmit only object-scale compact code, reconstruct object-only" for *extreme* (<0.1bpp) rates specifically — the closest analogues are ROI-based codecs (region-adaptive diffusion compression <cite index="71-1">a region-adaptive diffusion-based image codec that allocates more representational capacity to perceptually important regions</cite>) which spend *more* bits on the region of interest within a full-image budget, rather than *discarding* the rest of the scene entirely. Discarding the background outright — appropriate for your use case, where the fish/lobster is the payload and the background is not — is the more aggressive and more appropriate move for a 64–124B budget, and is the single biggest lever available to you (see Part 0 table: 0.002 bpp full-scene → 0.2 bpp crop is a **100× improvement**, for free, just by cropping).

### H. Neural rendering / implicit representations (INR)
COIN <cite index="141-1">overfits an image with a neural network mapping pixel locations to RGB values, then quantizes the weights and transmits them</cite>, achieving <cite index="143-1">performance that outperforms JPEG at low bitrates using MLPs with as few as 8,000 parameters (~14 KB at 0.3 bpp)</cite>. That's still ~100× your byte budget even in its best configuration, and COIN's per-image weights approach requires no shared decoder — the opposite trade you want (you *have* a shared decoder budget: the STM32N6 firmware). COIN++ improves this by meta-learning a shared base network and transmitting only small *modulation* vectors <cite index="155-1">store modulations applied to a meta-learned base network as a compressed code for the data, then quantize and entropy code these modulations, leading to large compression gains while reducing encoding time by two orders of magnitude</cite> — this modulation-vector idea is architecturally close to your target byte budget and is worth prototyping as an alternative appearance-code mechanism, but it has **not been demonstrated at 64–124 bytes** in the literature; treat it as a secondary experiment, not the primary path.

### I. Underwater-specific compression
This is the thinnest part of the literature. Findings:
- **EUICN** (TCSVT 2024) is an underwater-specific compression network but targets normal (not extreme) bitrates — general-purpose underwater compression, no evidence of sub-0.1bpp operation.
- **"Deep underwater image compression for enhanced machine vision applications"** (Frontiers Marine Science 2024) is the most relevant hit: it <cite index="59-1">proposes a task-driven feature decomposition fusion module that enables the network to understand and preserve machine-friendly information during compression, prioritizing task relevance over human visual perception, and effectively preserves machine-friendly features at low bit rate across object detection, semantic segmentation, and saliency detection</cite>. This validates the *principle* (task-aware loss > pixel loss for underwater compression) but again at normal, not extreme, bitrates, and it targets machine features, not human-viewable reconstruction.
- **No paper found combines** (a) extreme (<0.1bpp) bitrate, (b) underwater imagery, (c) species-level biological fidelity constraints. **This is a genuine, currently-open gap** — see Part 12.
- Underwater physical degradations (blue/green cast, backscatter, low contrast, depth-dependent attenuation) are well studied for *enhancement* (dozens of GAN/diffusion papers, e.g. UIEB/SUIM/LSUI benchmarks), but none of that literature intersects with extreme compression. Practically: you should apply underwater color-correction *before* your compression encoder (as preprocessing), and let your codec's shared color prior assume "already white-balanced" input — trying to jointly learn compression + color correction inside a 64–124B budget wastes bits on something a cheap classical preprocessing step (or ST's onboard ISP, see Part 10) already solves well.

### Comparison Table (Section 11)

| Approach | Payload @ target quality | Compression ratio (vs 500KB) | RGB reconstruction? | Visual quality | Fish/species accuracy | Hallucination risk | Decoder size | Compute (decode) | Training difficulty | BLE suitability | Embedded suitability | Open source | Recommended? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JPEG/WebP/JPEG-XL | ~5–15 KB min. useful | ~30–100× | Yes (real) | Good | N/A (not designed for it) | None | N/A | Trivial | None | Poor (too big for target) | Trivial | Yes | No — can't reach target |
| CompressAI hyperprior | ~1–4 KB | ~150–500× | Yes (real) | Good→fair | Preserved reasonably well | Low | 5–20 MB | Low-med | Moderate | Marginal | Feasible on STM32N6 | Yes | Good for the 1–8 KB tier only |
| VAE (plain, pixel-space) | Degrades badly <0.1bpp | high | Yes but blurry | Poor at target rate | Degraded | Low (just blurry) | Small | Low | Low | Fits at 64–124B but useless quality | Feasible | Yes | No — blur destroys ID-relevant detail |
| VQ-VAE / Generative Latent Coding | 64–124B for small crops | 5,000×+ (crop) | Yes (generative) | Good (species-plausible) | Good if species conditioned | Moderate | 20–200 MB | Med-high (decode) | High | Fits | Decoder likely needs cloud/phone, not MCU | Partial | **Yes — as decoder architecture, hosted off-MCU** |
| Semantic communication / JSCC | Tunable, incl. 64–124B | Very high | Yes (learned joint) | Fair-good | Depends on task loss | Moderate | Small-med (10–50MB) | Low-med | High | Excellent fit (designed for lossy/constrained channels) | Feasible w/ lightweight backbone (DeepJSCC-T) | Partial | Good complementary layer, not primary strategy |
| Underwater-specific (EUICN, FDFM) | Normal bitrates only | Moderate | Yes | Good | Not evaluated at extreme rates | Low | Small-med | Low | Moderate | No (not extreme enough) | Feasible | Partial | Use ideas (task-aware loss), not the codec itself |
| GAN reconstruction | 64–124B feasible | Very high | Yes (generative) | Good but artifact-prone | Risky — GANs invent texture | **High** | Med | Med | High | Fits | Marginal | Mixed | No — worse hallucination profile than diffusion+structure |
| Diffusion reconstruction (pure latent) | 64–124B feasible | Very high | Yes (generative) | Very good visually | Risky without explicit structure | **High** without side-info, moderate with it | Large (100MB–2GB) | High (iterative denoising) | High | Fits | **Not feasible on-MCU; needs phone/cloud decoder** | Partial | Only as decoder, with mandatory structural conditioning |
| Retrieval/codebook (prototype+residual) | 64–124B natural fit | Extreme | Yes (generative, prototype-based) | Good, honest about limits | **Good — species is explicit, not inferred** | **Low if species transmitted explicitly** | Small-med codebook | Low | Moderate | Fits well | Feasible | N/A (custom) | **Yes — core of recommendation** |
| Object-centric transmission | Enables everything above | 100×+ extra, free | N/A (enabler) | N/A | N/A | Reduces risk (less to hallucinate) | N/A | Low (crop+resize) | Low | Essential | Cheap on NPU | N/A | **Yes — mandatory first step** |
| **Hybrid (structured + generative, this report's recommendation)** | 64–124B | ~5,000–20,000× (crop) | Yes (semantic-key generative) | Good, species-correct | **Best of all options — explicit ID + explicit shape/pose + learned residual** | **Lowest of the generative options** | Med (residual decoder) | Low-med | High (multi-stage) | Fits by design | Encoder on MCU, decoder on phone/edge/cloud | Custom, plan to open-source | **Yes** |

---

## Part 3 — Recommended Architecture

### Rejecting the pure pixel-compression framing
Given Part 0/Part 1, "compress the image 5,000×" is the wrong mental model. The right mental model is: **"transmit the minimum information needed to pick a point in a learned, species-structured appearance manifold, then let a shared decoder render it."** This reframes the problem from *compression* to *parametric encoding against a domain-specific generative prior* — closer to how a vector graphic or a 3D character rig is "transmitted" as a small parameter vector, not as pixels.

### Why not just use CoCliCo/MISC/PerCo as-is
These are excellent, and their mechanism (semantic key + tiny side info + generative decoder) is exactly right — but they're built for **open-vocabulary natural images**, where a CLIP embedding is the cheapest way to specify "what is in the image" because the space of possible images is unbounded. **Your problem is closed-set**: 157 known species (fits in 1 byte) plus a handful of pose/shape/color parameters. Spending bytes on a general-purpose CLIP vector when you already have a fine-tuned, accurate BioCLIP classifier is wasteful — replace the expensive general semantic key with a **cheap, exact, domain-specific categorical key**, and reinvest every byte saved into the appearance residual, where it actually improves per-individual fidelity. This substitution is the core novel move of this design (see Part 12).

### Architecture: Object-Centric Structured-Semantic Codec (OC-SSC)

```
UNDERWATER FRAME (MB1854B, via STM32N6 ISP)
        │
        ▼
  Underwater color correction (classical / ISP, not learned — cheap, well-solved)
        │
        ▼
  YOLOv8n/v11n fish+lobster DETECTION  ──────────────► bbox (x,y,w,h) [4 B, quantized]
        │
        ▼
  Crop + resize to canonical patch (64×64 or 96×96)
        │
        ▼
  Lightweight species classifier (MobileNetV3/EfficientNet-Lite,
  distilled FROM your existing fine-tuned BioCLIP-2)  ────────► species_id [1 B] + confidence [1 B]
        │
        ▼
  Shape/pose head (small CNN head sharing the classifier backbone)  ─► pose/orientation [1 B],
                                                                         aspect/curvature [1 B]
        │
        ▼
  Appearance encoder → residual VQ tokens
  (small CNN → 16–32 codebook indices, product-quantized
   against a SHARED, per-species-conditioned codebook)  ────────► residual tokens [32–72 B]
        │
        ▼
  Tiny color map (4×4 or 6×6 downsampled, palette-indexed,
  akin to CoCliCo's color side-channel)                 ────────► color map [8–16 B]
        │
        ▼
  Pack into fixed-size record + 1-byte mode/version + 1-byte CRC8
        │
        ▼
  64–124 BYTE PAYLOAD  ──────────────────────────────►  BLE (single notification, no fragmentation
        │                                                needed once MTU ≥ 130 — see Part 9)
        ▼
  RECEIVER (phone / gateway / cloud — NOT the STM32N6 for the decode side)
        │
        ▼
  Unpack record → species_id selects SHARED per-species generative prior
        │
        ▼
  Conditional decoder (small diffusion-lite or VQ-VAE decoder,
  conditioned on species_id + pose + color map + residual tokens)
        │
        ▼
  RECONSTRUCTED RGB FISH/LOBSTER CROP (64×64–256×256, upsampled for display)
        │
        ▼
  (Optional) same species classifier re-run on reconstruction → confidence-check report
```

### Example 100-byte record layout

| Field | Bytes | Purpose |
|---|---|---|
| Version/mode flag | 1 | Protocol versioning, visual-vs-AI mode switch (Part 8J) |
| Species ID | 1 | Index into 157(+reserve) known species — **explicit, not inferred by decoder** |
| Classifier confidence | 1 | Quantized 0–255; low confidence flags the record for human review downstream |
| Bounding box (in source frame) | 4 | Quantized x, y, w, h — lets a full-frame viewer place the crop back in context if ever needed |
| Pose/heading | 1 | Discretized orientation (e.g. 32 directions × left/right flip) |
| Shape/aspect parameters | 2 | Body elongation, fin-spread proxy — coarse morphology beyond what species-ID implies |
| Tiny color map | 16 | 4×4 grid, palette-indexed (8-bit indices into a shared 256-color underwater palette) |
| Appearance residual (VQ tokens) | 64 | 32 tokens × 2 bytes (16-bit codebook, per-species-conditioned) — carries individual markings/texture beyond the species prior |
| CRC-8 | 1 | Payload integrity check |
| **Total** | **91** | Fits inside the 64–124 B budget with headroom for a 64 B "coarse mode" (drop residual tokens to 8) and a 124 B "fine mode" (residual tokens up to 48) |

This directly operationalizes Section 8I's hybrid concept and Section 15's "shared knowledge is allowed" principle: the codebook, the per-species decoder, and the color palette are **shipped once with firmware/app update**, never transmitted per-image.

### Two-mode protocol (Section 8J, resolved)
You should build **both modes**, switched by 1 bit in the version/mode field:
- **AI mode** (default, lowest bytes): species_id + confidence + bbox + pose only (~8–10 bytes). No visual reconstruction attempted; downstream system just logs "Cod detected at (x,y), 96% confidence." This is the common case for a monitoring buoy sending hundreds of detections.
- **Visual mode** (on demand, or for low-confidence/anomalous detections worth a human look): full 91–124 byte record with residual tokens and color map, enabling image reconstruction for human verification.

This hybrid protocol is *more* useful than committing to always-reconstruct-an-image, and it's nearly free to implement since AI mode is a strict subset of visual mode's fields.

---

## Part 4 — Recommended Bitrate

**Don't fix on one number — but if forced to pick a default, use ~100–124 bytes, with adaptive fallback down to 64.**

Reasoning:
- 64 bytes is enough for AI-mode-plus-coarse-color, but the residual token budget (Part 3) shrinks to ~8 tokens, which the literature (Part 2) suggests will produce recognizable-species-but-generic-individual reconstructions — fine for a "detection confirmed" thumbnail, not for anything claiming per-individual fidelity.
- 124 bytes buys roughly 4× the residual tokens of 64 bytes (proportionally, once you subtract the ~15-byte fixed header), which based on the VQ-token literature (GLC, DLF) is roughly where "recognizable and plausible" starts becoming "recognizable and individually distinctive."
- BLE-wise, both fit in a **single ATT notification** once MTU is negotiated above ~130 bytes (very achievable — Part 9), so there's no packetization cost difference between 64 and 124; **the constraint is genuinely about information content, not radio overhead.** Given that, default to the larger budget you have available and use 64 B only as a fallback (weak link, congested channel, battery-critical mode).
- Recommend implementing `codec.encode(image, max_bytes=N)` as **fully rate-adaptive** (Section 19) rather than committing to one N at design time — the architecture in Part 3 supports this natively by just dropping residual tokens from the end.

---

## Part 5 — Visual Quality: What to Actually Expect

| Byte budget | What kind of output | Basis |
|---|---|---|
| 32 B | Category tag only — no meaningful image; "species X detected here" | Estimated (below anything demonstrated with real reconstruction) |
| 64 B | Species-correct **prototype** image — right species, right rough pose/color, individual details are synthetic/generic | Estimated, extrapolating from CoCliCo/GLC's sub-0.01bpp facial-image results scaled to a 64×64 crop |
| 96 B | Species-correct image with **some real per-individual texture/marking cues** captured by residual tokens | Estimated, interpolating between GLC/DLF's demonstrated 0.01–0.04bpp results and your smaller, closed-set decoding target |
| 124 B | As above, modestly higher fidelity; closest to "fish-preserving reconstruction" in your terminology | Estimated |
| 256 B | Real per-pixel learned codec (non-generative, e.g. small hyperprior) becomes viable at 64×64 (≈0.5 bpp) — first budget where "near-original" pixel fidelity, not just plausibility, is realistic | Estimated from standard CompressAI rate-distortion curves |
| 512 B | Same, ≈1 bpp at 64×64, or ≈0.25 bpp at 128×128 — solidly good classical/learned compression territory | Estimated, well-supported by CompressAI literature |
| 1 KB | Near-original quality crop at 128×128 (≈0.5bpp) achievable with standard learned codecs, **experimentally well-documented** in the general (non-underwater) compression literature | Grounded — this is squarely inside CompressAI's normal operating range |
| 2 KB | Near-original 128×128, or good-quality 256×256 | Grounded |
| 4–8 KB | High-quality crop reconstruction, or a modest low/mid-quality full scene via JPEG/WebP | Grounded (this is where "just use JPEG" starts being genuinely reasonable, per your own instruction not to assume that below this range) |

**Labeling discipline** (as you requested): everything at 64–124 B in this report is labeled "estimated/extrapolated" because **no published paper has evaluated exactly this configuration** (closed-set species-conditioned generative decoding of underwater fish crops at 64–124 B). The 256B+ rows are "grounded" because they sit inside ranges directly measured in cited papers. Your Part 11 baseline experiment (below) is what converts the 64–124B row from "estimated" to "measured" for your specific case.

---

## Part 6 — Fish/Lobster Preservation Strategy

The single biggest lever against hallucination is architectural, not a loss function: **make species identity an explicit, transmitted, verifiable field — never something the decoder has to infer from a fuzzy continuous code.** A wrong species ID is then a detectable *upstream classification error* (which you can monitor, threshold on confidence, and flag), not a silent, undetectable *decoder hallucination*. This is a categorical improvement over pure-latent generative codecs, where a bad reconstruction and a good one are indistinguishable without the original to compare against.

Beyond that:
- **Freeze the species classifier during codec training**, at least initially. Train the appearance residual encoder/decoder to minimize reconstruction loss *conditioned on* the (frozen, correct) species label, so the residual pathway never learns to "help" the classification task by drifting the species-typical shape — it should only encode *individual deviation from typical*.
- **Explicitly penalize shape/keypoint drift** between original and reconstruction (fin position, tail shape, eye position) using a lightweight keypoint or segmentation-mask loss, not just pixel/perceptual loss — this operationalizes Section 7's "not just PSNR/SSIM" requirement directly.
- **Run your existing fine-tuned BioCLIP-2 classifier on both original and reconstruction during training and evaluation**, and treat *agreement* as a first-class metric (Section 21's critical experiment) — Top-1 species match rate, confidence delta, and confusion-matrix drift between original and reconstructed images.
- **Confidence gating at the encoder, not just the decoder**: if your on-device classifier's confidence is below a threshold, don't claim a species ID at all — fall back to a coarser record (genus-level ID, or "unidentified fish, bbox only") rather than transmitting a possibly-wrong species that then drives a confident-looking-but-wrong reconstruction downstream.

---

## Part 7 — Training Strategy

### Loss function (your proposed formula, revised)
Your proposed loss is a reasonable starting point but needs two changes: (1) treat species-ID as a *hard classification target with its own frozen/near-frozen pathway*, not a soft loss term competing with pixel/perceptual terms, and (2) add an explicit structure-preservation term separate from generic "perceptual" loss.

```
Total codec loss (appearance-residual pathway only; species/pose heads trained separately) =

  λ_rate      × bitrate_penalty(residual_tokens)
+ λ_pixel     × L1(reconstruction, target_crop)
+ λ_perceptual× LPIPS(reconstruction, target_crop)
+ λ_structure × keypoint_or_mask_loss(reconstruction, target_crop)   # fins, eye, tail, outline
+ λ_teacher   × ||BioCLIP_features(reconstruction) - BioCLIP_features(original)||   # see below
+ λ_color     × underwater_color_consistency_loss
```

Drop the classification loss from this joint objective — classification is handled by a **separately trained, frozen** species head so it can never be "gamed" by the reconstruction pathway.

### Yes — use your fine-tuned BioCLIP-2 as a frozen teacher
This is a good idea and directly supported by how GLC and DLF's "semantic branch" losses work in the literature — they all use a frozen, pretrained feature extractor as a perceptual/semantic anchor. Concretely: run both the original crop and the reconstructed crop through your **frozen** BioCLIP-2, and minimize the distance between their feature embeddings as one loss term. This transfers BioCLIP's rich, fish-specific notion of "what matters visually for this species" into the tiny deployed codec, without deploying BioCLIP itself at inference time — exactly the "shared knowledge, not shared compute" pattern from Section 15, except here the sharing happens at *training time* rather than at *inference time*.

### Datasets
- **Your existing FishVision-AI dataset** (157 species, 55K+ images) — primary source, already labeled, already has verified BioCLIP-2 embeddings you can distill from.
- **OzFish, Fish4Knowledge, SUIM** — supplementary, for species/pose/background diversity and to sanity-check generalization beyond your own capture conditions.
- **Real MB1854B captures** — essential before deployment claims; every other dataset differs in optics/color response from your actual sensor.

### Model architecture
- Detector: YOLOv8n/v11n (already NPU-friendly, already in your stack).
- Species/pose head: MobileNetV3-Small or EfficientNet-Lite0, distilled from BioCLIP-2 — small enough for the STM32N6's 4.2 MB SRAM budget.
- Appearance residual encoder: small CNN (4–6 conv layers) → product-quantized VQ tokens against a per-species-conditioned codebook (learned, e.g. 512–1024 entries per species cluster, shared across similar species to keep codebook size bounded).
- Decoder (receiver-side, NOT on the STM32N6): small conditional generator (VQ-VAE decoder or lightweight conditional diffusion/one-step distilled diffusion, cf. AEIC's "shallow encoder + rich decoder" asymmetry <cite index="8-1">AEIC employs moderate or even shallow encoder networks, while leveraging a one-step diffusion decoder to maintain high-fidelity and high-realism reconstructions under extreme bitrates</cite>) — this asymmetric-compute pattern is exactly right for your MCU-encoder / phone-or-cloud-decoder split.

### Training stages
1. Pretrain/confirm species+pose heads (distill from BioCLIP-2, freeze).
2. Train VQ codebook + appearance encoder/decoder end-to-end, species-conditioned, with the loss above.
3. Fine-tune jointly with a rate-distortion sweep across your target byte budgets (64/96/124B) to produce one adaptive model rather than three separate ones.
4. Domain-adapt on real MB1854B captures.

### Evaluation protocol (Sections 20–21, operationalized)
Run every test image through: encode → decode → (a) PSNR/SSIM/MS-SSIM/LPIPS, (b) UCIQE/UIQM underwater-specific quality, (c) re-run YOLO + species classifier on the reconstruction and diff against ground truth (**this is the critical experiment** — Top-1/Top-5 species match rate and confidence delta between original and reconstruction, not just image-quality metrics), (d) manual audit of a sample for the specific failure modes in Section 7 (false fins, wrong markings, wrong shape) using your keypoint/mask annotations as ground truth. Report all of these per byte-budget (64/96/124B) so the byte/accuracy trade-off curve is explicit, not a single number.

---

## Part 8 — Python Library Design

```python
from uwcodec import UWCodec

codec = UWCodec.load("fishvision_codec_v1.pt")   # ships with species codebook + decoder weights

# Encoding (runs on-device / STM32N6, or in Python for prototyping)
payload = codec.encode(
    image,                 # full-frame RGB array (post color-correction)
    max_bytes=100,          # 64 / 96 / 124 / 256 / 1024 ... adaptive
    mode="visual",          # "ai" | "visual"
    min_confidence=0.6,     # falls back to genus-level / bbox-only record if classifier is below this
)

# payload.bytes            -> bytes object, ready for BLE
# payload.report            -> dict: species, confidence, bbox, bytes_used, mode

# Decoding (runs on receiver: phone / gateway / cloud — not the MCU)
result = codec.decode(payload.bytes)

# result.image              -> reconstructed RGB array
# result.species            -> predicted species (from transmitted ID, not re-inferred)
# result.confidence         -> transmitted classifier confidence
# result.metrics            -> populated only if codec.decode(..., ground_truth=original) is called,
#                               e.g. during evaluation: PSNR, SSIM, LPIPS, species_match, confidence_delta
```

Key design choices:
- `encode()` and `decode()` are cleanly separable — encode can run standalone on embedded C (after export) while decode stays Python/PyTorch for now, matching the STM32N6 (encoder) / phone-or-cloud (decoder) split from Part 3.
- `max_bytes` drives adaptive rate control (Section 19) by truncating the residual-token list — implemented as a strict prefix code (drop lowest-priority tokens first), so a single trained model serves all byte budgets rather than needing separate models per target size.
- The library should expose a `codec.rate_search(image, targets=[64,96,124], quality_targets={"species_top1":0.95})` helper implementing Section 19's adaptive search directly.

---

## Part 9 — BLE Integration

Grounded facts about BLE payload limits: <cite index="121-1">the default ATT_MTU is 23 bytes, of which 3 bytes go to the ATT header, leaving 20 bytes of usable payload per notification until MTU is negotiated higher; most stacks cap negotiated ATT_MTU at 247 bytes, giving 244 bytes of usable payload once negotiated</cite>, and <cite index="128-1">the Bluetooth spec maximum is 517 bytes MTU (512 bytes usable payload after overhead)</cite>, though <cite index="121-1">iOS in particular typically lands at 185 bytes negotiated MTU (182 bytes usable), and app protocols targeting iOS should design around that number rather than assuming the full 244</cite>.

**Practical implication for you: your entire 64–124 byte payload fits inside a single BLE notification once MTU negotiation completes (even the conservative iOS ceiling of ~182 usable bytes is comfortably above 124).** No fragmentation, sequence numbers, or reassembly logic is required for the payload itself — this is one of the nicer properties of committing to a small, fixed byte budget.

Recommended design:
```python
payload = codec.encode(image, max_bytes=100)     # codec is BLE-agnostic
packets = ble.packetize(payload.bytes)             # trivially: [payload.bytes] — one packet, given MTU>124
send(packets)

# Receiver
payload_bytes = ble.reassemble(packets)             # trivial reassembly (single packet)
result = codec.decode(payload_bytes)
```
- Still negotiate MTU explicitly at connection time (don't rely on the 20-byte default) and confirm it via the `att_mtu_updated`-style callback before sending, per the "gotcha" documented in BLE MTU guides <cite index="121-1">even after the exchange completes, in-flight packets already on the air use the old size, so bulk transfers should be gated on the MTU-updated callback rather than the connection callback</cite>.
- Add a 1-byte CRC (already in the Part 3 record) rather than relying solely on the link layer's own CRC, since a single bit error in a 100-byte semantic record could flip the species ID silently — cheap insurance for a biologically-sensitive payload.
- Use **notifications, not indications**, for the actual transfer (fire-and-forget, lower latency, no ack round-trip) since a missed detection can simply be re-sent on the next cycle — <cite index="132-1">write commands and notifications are asynchronous, allowing them to be sent back-to-back within a single connection event, while write requests and indications are synchronous and require a response before the next command, reducing throughput</cite>.
- Design the payload format to be **forward-fragmentable anyway** (a 1-byte "more fragments" flag) even though you won't need it at 64–124B — cheap insurance if you ever add an optional larger "high-fidelity" mode later.

---

## Part 10 — Embedded Deployment

### STM32N6570-DK reality check (important correction to the assumed architecture)
The STM32N6570-DK's <cite index="104-1">Neural-ART Accelerator provides 600 GOPS at roughly 3 TOPS/W, with 4.2 MB of contiguous SRAM, on a Cortex-M55 core clocked up to 800 MHz</cite> — genuinely capable for running your YOLOv8n/v11n detector and a small MobileNetV3/EfficientNet-Lite species+pose head on-device in real time. **However, based on available documentation, the STM32N6570-DK does not include a built-in BLE radio** — its wireless connectivity is exposed only via <cite index="106-1">flexible extension connectors intended for adding wireless connectivity, analog applications, and sensors</cite>, not an onboard BLE chip. In practice this means your architecture needs a companion BLE radio (e.g., an ST STM32WB/STM32WBA module, or a discrete BLE chip such as nRF52 or ESP32-as-BLE-peripheral) bridged to the STM32N6 over UART/SPI. This doesn't change any of the codec design above (the codec is BLE-agnostic by design, per Part 9), but it does change your bill-of-materials and firmware architecture — worth confirming with your MB1854B/board documentation directly before finalizing hardware, since this detail isn't fully certain from public docs alone.

### What runs where
| Component | Where | Why |
|---|---|---|
| Underwater color correction | STM32N6 ISP (hardware) or lightweight classical algorithm | ST's ISP already does black-level/exposure/color correction in hardware — reuse it rather than a learned module |
| YOLOv8n/v11n detection | STM32N6 Neural-ART NPU | Purpose-built for exactly this; 600 GOPS is comfortably sufficient for nano-scale YOLO at low-res input |
| Species/pose classification head | STM32N6 Neural-ART NPU | Small distilled model (MobileNetV3-Small class), fits in SRAM alongside the detector |
| Appearance residual encoder | STM32N6 Neural-ART NPU (small CNN) | Keep this deliberately tiny — a handful of conv layers, per AEIC's "shallow encoder" philosophy |
| Payload packing + BLE transmission | Cortex-M55 host CPU + companion BLE radio | Trivial bit-packing, no NPU needed |
| **Generative decoder** | **Phone app, gateway, or cloud — not the STM32N6** | The decoder (VQ-VAE or lightweight diffusion) is the heavy part; even "shallow-encoder" papers keep the decoder off the constrained device. This is the correct asymmetry, not a limitation to work around. |
| BioCLIP-2 (full) | Training-time only (teacher), never deployed | Confirmed by Part 7 — distillation target, not a runtime dependency |

If a fully-standalone underwater unit needs to *display* a reconstruction without a phone/gateway in the loop, that's a separate, harder requirement (on-device generative decoding) — flag this explicitly if it's actually needed, since it changes the NPU budget calculus substantially (a real-time generative decoder is a much heavier NPU workload than a nano-YOLO detector).

---

## Part 11 — Baseline Experiment (fail-fast, no weeks of training)

Goal: determine in **under a day** whether the core premise (species-conditioned tiny-code reconstruction) has legs, before investing in the full training pipeline.

1. **Take 100–200 existing FishVision-AI crops** (already labeled with species, already have BioCLIP-2 embeddings).
2. **Skip training an encoder entirely for this test.** Instead, for each image: (a) look up its true species ID (1 byte, "free" — you already have ground truth), (b) compute a *very coarse* color descriptor by hand (e.g., k-means with k=4 on the crop, 4 dominant RGB colors ≈ 12 bytes), (c) that's your entire "payload" — no residual tokens yet, deliberately testing the cheapest possible version of the architecture.
3. **Build a tiny decoder fast**: fine-tune a small pretrained image generator (e.g., a small class-conditional GAN, or even a nearest-neighbor "retrieve the closest training image of that species + apply the coarse color palette as a filter") conditioned on species_id + the 4-color palette. This can be as crude as: for each species, precompute a mean/prototype image, then color-grade it toward the transmitted palette. It doesn't need to be good — it needs to tell you whether the *approach* has signal.
4. **Evaluate**: run your existing YOLO+BioCLIP-2 pipeline on the "reconstructions." If species re-classification accuracy on the crude reconstructions is meaningfully above chance and visually the outputs are at least *species-plausible* to a human, the core hypothesis holds and full training is worth the investment. If reconstructions are indistinguishable across species even with ground-truth species ID handed to the decoder, that's a fast, cheap signal that the appearance/color side-channel needs fundamental rethinking before you invest further.
5. **Total cost**: no GPU training run, no new model architecture beyond a tiny conditional colorizer — this can realistically be a single afternoon, and it directly tests the riskiest assumption in the whole design (that species + coarse color + a shared prior is enough to produce something recognizable).

---

## Part 12 — Research Novelty

**Is this already solved?** No — not at this exact intersection. Extreme (<0.05bpp) generative compression is an active, hot research area (2024–2026, dozens of papers, Part 2), but every example found targets **general natural images, faces, or remote sensing**, not underwater biological subjects with a **closed taxonomic label space**.

**What's closest?**
- CoCliCo/MISC/SEDIC/GLC (semantic-key + generative decoder) — closest *mechanism*.
- The Frontiers 2024 task-driven underwater compression paper (FDFM) — closest *domain*, but not extreme-rate, not generative, not biology-fidelity-focused.
- Nothing combines both.

**What would actually be novel:**
1. **Replacing a general open-vocabulary semantic key (CLIP embedding, ~100s of bytes) with a closed-set categorical key (1 byte) for a domain where you have a strong existing classifier** — this is a genuinely underexplored simplification; most of the surveyed literature assumes open-vocabulary content because their target domain (natural images) has no equivalent closed taxonomy.
2. **Using a fine-tuned biological foundation model (your BioCLIP-2) as a frozen distillation teacher for an extreme-compression codec** — teacher-distillation into learned codecs exists generically, but not with a domain-specialized biological vision-language model as the teacher.
3. **A hallucination-safety framing specific to biological identification** (explicit-ID-vs-inferred-ID as the core mitigation, confidence-gated fallback modes, species-match-rate as a first-class eval metric alongside PSNR/LPIPS) — this evaluation discipline doesn't have a direct precedent in the compression literature surveyed, which optimizes for FID/LPIPS/ClipSIM, not task-specific-taxonomic-correctness.
4. **The whole system, end to end, on a real BLE link with a real MB1854B/STM32N6 embedded encoder** — the compression papers are all pure ML research (GPU inference, no radio, no MCU); the underwater-compression papers don't target extreme rates; nobody has built and measured this specific pipeline in the real world.

**What's just engineering, not novel:** the BLE packetization, the STM32N6 NPU deployment of YOLO+MobileNet (well-trodden), classical underwater color correction, the Python library API design.

**Publication/open-source potential:** Point 1–3 above, demonstrated with real numbers on your FishVision-AI dataset (species-match-rate vs. byte-budget curves, compared honestly against a CoCliCo-style CLIP-embedding baseline as the control), is a legitimate short paper or strong workshop submission (e.g., a computer-vision-for-conservation or efficient-ML workshop) — the "closed-set category key beats open-vocabulary embedding for extreme-rate domain-specific compression" finding, if it holds up experimentally, is a clean, falsifiable, novel-enough contribution. It's also a strong open-source project on its own even without a paper: "the first extreme-low-bitrate codec benchmarked specifically against species/task accuracy rather than pixel metrics" is a genuinely useful artifact for the underwater-CV community, independent of whether it's "novel" in the strict academic sense.

---

## Summary

| Question | Answer |
|---|---|
| Can you reconstruct a full RGB *scene* from 64–124 bytes? | No — that's ~10× beyond anything demonstrated in the literature (Part 0). |
| Can you reconstruct a useful, species-plausible *fish/lobster crop*? | Yes, in the same regime as multiple published 2024–2026 extreme-compression systems, **if you crop first**. |
| Is the right approach "compress the image"? | No — reframe as "transmit a species key + shape/pose/color/residual parameters into a shared generative prior." |
| Is your hybrid instinct (8I) correct? | Yes, and the literature independently converges on the same recipe (semantic key + tiny structural side-info + shared decoder). |
| Should species be inferred by the decoder or transmitted explicitly? | **Transmitted explicitly** — this is the single biggest hallucination-safety win available, and it's nearly free (1 byte). |
| Recommended default byte budget | ~100–124 B, adaptive down to 64 B, single-packet over BLE (no fragmentation needed). |
| Where does the decoder run? | Phone/gateway/cloud, not the STM32N6 — encoder-heavy/decoder-light asymmetry, matching 2025's "shallow encoder" literature. |
| Biggest open gap in prior work | No one has combined extreme-rate + underwater + closed-set biological fidelity — this is your opening. |
| First thing to actually build | The one-afternoon baseline in Part 11 — before any training pipeline. |
