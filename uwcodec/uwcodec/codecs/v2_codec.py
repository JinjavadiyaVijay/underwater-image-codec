"""UWCodec v2 — Dual-Branch Semantic+Detail Codec.

Core codec class. Combines:
  - SemanticEncoder (4×4 latent) + ResidualVQ (2-3 levels)
  - DetailEncoder (8×8 latent) + VQ (1-2 levels)
  - V2Decoder (UNet-style, FiLM conditioning)

Byte allocation (header = 1B version + 1B CRC = 2B fixed overhead):

  Budget | Sem (RVQ) | Det L1 | Det L2 | Data | Total
  -------|-----------|--------|--------|------|------
    64B  |   32B(2×) |   30B  |    0B  |  62B |   64B
    96B  |   32B(2×) |   62B  |    0B  |  94B |   96B
   124B  |   48B(3×) |   64B  |   10B  | 122B |  124B
   128B  |   48B(3×) |   64B  |   14B  | 126B |  128B

Payload layout within data bytes:
  [sem_lvl1: 16B][sem_lvl2: 16B][sem_lvl3?: 16B][det_l1: N1B][det_l2?: N2B]

This is the COMPLETE encode→VQ→serialize→deserialize→decode path.
train_v2.py uses the forward() method for the differentiable training path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from uwcodec.models.v2_encoder import SemanticEncoder, DetailEncoder
from uwcodec.models.v2_decoder import V2Decoder
from uwcodec.models.quantizer import ResidualVQ, VectorQuantizer
from uwcodec.core.payload import PayloadFormat


# ---------------------------------------------------------------------------
# Budget allocation table
# ---------------------------------------------------------------------------

# Key: total budget in bytes
# Value: dict with sem_levels, det_l1, det_l2 (all in bytes / token counts)
BUDGET_ALLOC: dict[int, dict[str, int]] = {
    64:  dict(sem_levels=2, det_l1=30, det_l2=0),
    96:  dict(sem_levels=2, det_l1=62, det_l2=0),
    124: dict(sem_levels=3, det_l1=64, det_l2=10),
    128: dict(sem_levels=3, det_l1=64, det_l2=14),
}
# Verify byte totals are correct
_HEADER = 2  # version(1) + CRC(1)
for _b, _a in BUDGET_ALLOC.items():
    _data = _a["sem_levels"] * 16 + _a["det_l1"] + _a["det_l2"]
    assert _data + _HEADER == _b, (
        f"Budget {_b}B: {_a['sem_levels']}×16 + {_a['det_l1']} + {_a['det_l2']} + {_HEADER} = {_data + _HEADER} ≠ {_b}"
    )

SUPPORTED_BUDGETS = sorted(BUDGET_ALLOC.keys())

# Spatial grid sizes
SEM_H, SEM_W = 4, 4          # 16 positions total (semantic)
DET_H, DET_W = 8, 8          # 64 positions total (detail)
SEM_POSITIONS = SEM_H * SEM_W  # 16
DET_POSITIONS = DET_H * DET_W  # 64
CODEBOOK_SIZE = 256            # 1 byte per index


# ---------------------------------------------------------------------------
# Main codec
# ---------------------------------------------------------------------------

class UWCodecV2(nn.Module):
    """UWCodec v2: dual-branch semantic+detail underwater image codec.

    Instantiate with a fixed byte budget. To train multiple budgets, run
    separate instances (via train_v2_multi_budget.py).

    Usage:
        codec = UWCodecV2(budget=128)

        # Training (differentiable):
        out = codec(images)  # dict with 'reconstruction', 'vq_loss', ...
        loss = out['loss']

        # Inference:
        payload = codec.encode(image_np, budget=128)  # bytes
        recon   = codec.decode(payload)               # np.ndarray uint8
    """

    BUDGET_ALLOC = BUDGET_ALLOC

    def __init__(
        self,
        budget: int = 128,
        # Encoder config
        sem_dim: int = 64,
        det_dim: int = 32,
        # Decoder config
        decoder_base_channels: int = 256,
        num_res_blocks_bottom: int = 4,
        num_res_blocks_mid: int = 2,
        # I/O
        input_size: int = 128,
        output_size: int = 128,
    ):
        super().__init__()

        if budget not in BUDGET_ALLOC:
            raise ValueError(f"Budget {budget}B not supported. Choose from {SUPPORTED_BUDGETS}")

        self.budget = budget
        self.sem_dim = sem_dim
        self.det_dim = det_dim
        self.input_size = input_size
        self.output_size = output_size

        # Store for checkpoint save/load
        self._decoder_base_channels = decoder_base_channels
        self._num_res_blocks_bottom = num_res_blocks_bottom
        self._num_res_blocks_mid    = num_res_blocks_mid

        alloc = BUDGET_ALLOC[budget]
        self.sem_levels: int  = alloc["sem_levels"]
        self.det_l1_tokens: int = alloc["det_l1"]
        self.det_l2_tokens: int = alloc["det_l2"]

        # ---- Encoders ----
        self.sem_encoder = SemanticEncoder(sem_dim=sem_dim)
        self.det_encoder = DetailEncoder(det_dim=det_dim)

        # ---- Quantizers ----
        # Semantic: multi-level residual VQ  (sem_levels × 16 positions)
        self.sem_rvq = ResidualVQ(
            codebook_size=CODEBOOK_SIZE,
            codebook_dim=sem_dim,
            num_levels=self.sem_levels,
            commitment_weight=0.25,
        )

        # Detail L1: single VQ over 8×8 = 64 positions
        self.det_vq1 = VectorQuantizer(
            codebook_size=CODEBOOK_SIZE,
            codebook_dim=det_dim,
            commitment_weight=0.25,
        )

        # Detail L2: residual VQ on det_l1 residual (only for 124B and 128B)
        if self.det_l2_tokens > 0:
            self.det_vq2: VectorQuantizer | None = VectorQuantizer(
                codebook_size=CODEBOOK_SIZE,
                codebook_dim=det_dim,
                commitment_weight=0.25,
            )
        else:
            self.det_vq2 = None

        # ---- Decoder ----
        self.decoder = V2Decoder(
            sem_dim=sem_dim,
            det_dim=det_dim,
            base_channels=decoder_base_channels,
            num_res_blocks_bottom=num_res_blocks_bottom,
            num_res_blocks_mid=num_res_blocks_mid,
            output_size=output_size,
        )

        # Payload format utility (shared header/CRC logic)
        self._payload_fmt = PayloadFormat()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_det_mask(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, 1, 8, 8) float mask: first n_tokens positions=1, rest=0 (raster order)."""
        flat = torch.zeros(DET_POSITIONS, device=device)
        flat[:n_tokens] = 1.0
        return flat.view(1, 1, DET_H, DET_W)

    @staticmethod
    def _indices_to_bytes(indices: torch.Tensor, n: int) -> bytes:
        """(1, H, W) or (H, W) long tensor → first n bytes (raster order, uint8)."""
        flat = indices.view(-1).to(torch.int16).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return bytes(flat[:n])

    @staticmethod
    def _bytes_to_indices(data: bytes, n_pos: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Bytes → (1, h, w) long index tensor, zero-padding to h*w positions."""
        full = list(data) + [0] * (n_pos - len(data))
        full = full[:n_pos]
        return torch.tensor(full, dtype=torch.long, device=device).view(1, h, w)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) → float32 (1, 3, input_size, input_size) in [0, 1]."""
        pil = Image.fromarray(image).resize(
            (self.input_size, self.input_size), Image.LANCZOS
        )
        t = torch.from_numpy(np.array(pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t

    def _postprocess(self, t: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) float [0,1] → uint8 (H, W, 3)."""
        return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Differentiable forward (used during training)
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """Differentiable encode-decode for training.

        Args:
            images: (B, 3, H, W) float in [0, 1]

        Returns:
            dict with keys:
              reconstruction  (B, 3, H, W)  sigmoid output
              sem_vq_loss     scalar
              det_vq_loss     scalar
              vq_loss         scalar  (sum of sem + det)
              sem_perplexity  scalar
              det_perplexity  scalar
        """
        device = images.device

        # --- Encode ---
        sem_z = self.sem_encoder(images)   # (B, sem_dim, 4, 4)
        det_z = self.det_encoder(images)   # (B, det_dim, 8, 8)

        # --- Quantize semantic (RVQ) ---
        sem_q, sem_info = self.sem_rvq(sem_z)  # sem_q: (B, sem_dim, 4, 4)
        sem_vq_loss   = sem_info["vq_loss"]
        sem_perplexity = sem_info["perplexity"]

        # --- Quantize detail L1 ---
        det_q1, det_info1 = self.det_vq1(det_z)   # (B, det_dim, 8, 8)
        det_vq_loss   = det_info1["vq_loss"]
        det_perplexity = det_info1["perplexity"]

        # Apply budget mask: zero positions not transmitted
        det_mask = self._make_det_mask(self.det_l1_tokens, device)  # (1,1,8,8)
        det_q_masked = det_q1 * det_mask

        # --- Quantize detail L2 (for 124B/128B budgets) ---
        if self.det_vq2 is not None:
            det_residual = (det_z - det_q1).detach()
            det_q2, det_info2 = self.det_vq2(det_residual)
            det_vq_loss = det_vq_loss + det_info2["vq_loss"]
            det_perplexity = (det_perplexity + det_info2["perplexity"]) / 2

            # Mask: only first det_l2_tokens positions of L2 are transmitted
            det_mask_l2 = self._make_det_mask(self.det_l2_tokens, device)
            det_q_masked = det_q_masked + det_q2 * det_mask_l2

        # --- Decode ---
        recon = self.decoder(sem_q, det_q_masked)

        return {
            "reconstruction": recon,
            "sem_vq_loss":    sem_vq_loss,
            "det_vq_loss":    det_vq_loss,
            "vq_loss":        sem_vq_loss + det_vq_loss,
            "sem_perplexity": sem_perplexity,
            "det_perplexity": det_perplexity,
        }

    # ------------------------------------------------------------------
    # Encode / Decode (inference, exact payload)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, image: np.ndarray, budget: int | None = None) -> bytes:
        """Encode image to exactly `budget` bytes.

        Args:
            image:  uint8 (H, W, 3) RGB numpy array.
            budget: Byte budget. Must match this codec's training budget.

        Returns:
            bytes of exactly `budget` length (version + VQ data + CRC).
        """
        if budget is not None and budget != self.budget:
            raise ValueError(
                f"This codec was trained for {self.budget}B; cannot encode at {budget}B. "
                f"Load the correct checkpoint."
            )
        budget = self.budget
        device = next(self.parameters()).device

        x = self._preprocess(image).to(device)

        # Encode
        sem_z = self.sem_encoder(x)
        det_z = self.det_encoder(x)

        # Quantize semantic (all levels)
        _, sem_info = self.sem_rvq(sem_z)
        sem_indices_list: list[torch.Tensor] = sem_info["indices"]  # list[(1,4,4)]

        # Quantize detail L1
        det_q1, det_info1 = self.det_vq1(det_z)
        det_indices_l1: torch.Tensor = det_info1["indices"]  # (1, 8, 8)

        # Quantize detail L2 (if applicable)
        det_indices_l2: torch.Tensor | None = None
        if self.det_vq2 is not None:
            det_residual = det_z - det_q1
            _, det_info2 = self.det_vq2(det_residual)
            det_indices_l2 = det_info2["indices"]  # (1, 8, 8)

        # Serialize to bytes
        data = bytearray()
        for lvl_idx in sem_indices_list:
            data += self._indices_to_bytes(lvl_idx, SEM_POSITIONS)  # 16B per level

        data += self._indices_to_bytes(det_indices_l1, self.det_l1_tokens)

        if det_indices_l2 is not None and self.det_l2_tokens > 0:
            data += self._indices_to_bytes(det_indices_l2, self.det_l2_tokens)

        # Verify data length
        expected_data = self.sem_levels * SEM_POSITIONS + self.det_l1_tokens + self.det_l2_tokens
        assert len(data) == expected_data, (
            f"Bug: serialized {len(data)}B data, expected {expected_data}B"
        )

        # Pack with header + CRC
        payload = self._payload_fmt.pack(bytes(data), budget)
        assert len(payload.raw_bytes) == budget, (
            f"Bug: payload is {len(payload.raw_bytes)}B, expected {budget}B"
        )
        return payload.raw_bytes

    @torch.no_grad()
    def decode(self, payload: bytes | None = None, *, raw_bytes: bytes | None = None) -> np.ndarray:
        """Decode payload bytes to RGB image.

        Args:
            payload:   Raw bytes (exactly `budget` bytes, including header/CRC).
            raw_bytes: Alias for payload (for compatibility).

        Returns:
            uint8 (output_size, output_size, 3) RGB numpy array.
        """
        data_bytes = payload if payload is not None else raw_bytes
        if data_bytes is None:
            raise ValueError("Must provide payload bytes")

        device = next(self.parameters()).device

        # Unpack header + CRC
        vq_data, _ = self._payload_fmt.unpack(data_bytes)

        offset = 0

        # Parse semantic indices
        sem_indices_list: list[torch.Tensor] = []
        for _ in range(self.sem_levels):
            chunk = vq_data[offset : offset + SEM_POSITIONS]
            offset += SEM_POSITIONS
            sem_indices_list.append(
                self._bytes_to_indices(chunk, SEM_POSITIONS, SEM_H, SEM_W, device)
            )

        # Parse detail L1 indices (zero-pad to 64 positions)
        det_l1_chunk = vq_data[offset : offset + self.det_l1_tokens]
        offset += self.det_l1_tokens
        det_indices_l1 = self._bytes_to_indices(det_l1_chunk, DET_POSITIONS, DET_H, DET_W, device)

        # Parse detail L2 indices (zero-pad to 64 positions)
        det_indices_l2: torch.Tensor | None = None
        if self.det_l2_tokens > 0:
            det_l2_chunk = vq_data[offset : offset + self.det_l2_tokens]
            offset += self.det_l2_tokens
            det_indices_l2 = self._bytes_to_indices(det_l2_chunk, DET_POSITIONS, DET_H, DET_W, device)

        # Reconstruct semantic latent from codes
        sem_q = torch.zeros(1, self.sem_dim, SEM_H, SEM_W, device=device)
        for lvl, idx in enumerate(sem_indices_list):
            emb = self.sem_rvq.quantizers[lvl].embedding(idx)  # (1, 4, 4, D)
            emb = emb.permute(0, 3, 1, 2)                      # (1, D, 4, 4)
            sem_q = sem_q + emb

        # Reconstruct detail latent
        det_emb_l1 = self.det_vq1.embedding(det_indices_l1)    # (1, 8, 8, D)
        det_q = det_emb_l1.permute(0, 3, 1, 2)                 # (1, D, 8, 8)

        if det_indices_l2 is not None and self.det_vq2 is not None:
            det_emb_l2 = self.det_vq2.embedding(det_indices_l2)
            # Only positions 0..det_l2_tokens-1 are valid; rest are already zeroed by indexing 0
            det_mask_l2 = self._make_det_mask(self.det_l2_tokens, device)
            det_q = det_q + det_emb_l2.permute(0, 3, 1, 2) * det_mask_l2

        # Decode
        recon = self.decoder(sem_q, det_q)
        return self._postprocess(recon)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path, train_state: dict | None = None) -> None:
        """Save model weights + config to a single checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "state_dict": self.state_dict(),
            "config": {
                "budget":                 self.budget,
                "sem_dim":                self.sem_dim,
                "det_dim":                self.det_dim,
                "input_size":             self.input_size,
                "output_size":            self.output_size,
                "decoder_base_channels":  self._decoder_base_channels,
                "num_res_blocks_bottom":  self._num_res_blocks_bottom,
                "num_res_blocks_mid":     self._num_res_blocks_mid,
            },
        }
        if train_state is not None:
            save_dict["train_state"] = train_state
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "UWCodecV2":
        """Load codec from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        model = cls(
            budget=cfg["budget"],
            sem_dim=cfg["sem_dim"],
            det_dim=cfg["det_dim"],
            input_size=cfg.get("input_size", 128),
            output_size=cfg.get("output_size", 128),
            decoder_base_channels=cfg.get("decoder_base_channels", 256),
            num_res_blocks_bottom=cfg.get("num_res_blocks_bottom", 4),
            num_res_blocks_mid=cfg.get("num_res_blocks_mid", 2),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)

    def parameter_summary(self) -> dict[str, int]:
        """Return parameter counts by component."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        sem_rvq_params = count(self.sem_rvq)
        det_vq_params  = count(self.det_vq1) + (count(self.det_vq2) if self.det_vq2 else 0)

        d = {
            "sem_encoder":  count(self.sem_encoder),
            "det_encoder":  count(self.det_encoder),
            "sem_rvq":      sem_rvq_params,
            "det_vq":       det_vq_params,
            "decoder":      count(self.decoder),
        }
        d["total"] = sum(d.values())
        return d

    def print_summary(self) -> None:
        """Print model summary to stdout."""
        alloc = BUDGET_ALLOC[self.budget]
        p = self.parameter_summary()
        print("=" * 60)
        print(f"UWCodec v2  |  Budget: {self.budget}B")
        print("=" * 60)
        print(f"  Sem encoder:  {p['sem_encoder']:>10,} params  (4×4 latent, {self.sem_levels}-level RVQ)")
        print(f"  Det encoder:  {p['det_encoder']:>10,} params  (8×8 latent, VQ)")
        print(f"  Sem RVQ:      {p['sem_rvq']:>10,} params  ({self.sem_levels} × {SEM_POSITIONS}B = {self.sem_levels*SEM_POSITIONS}B)")
        print(f"  Det VQ:       {p['det_vq']:>10,} params  (L1={alloc['det_l1']}B, L2={alloc['det_l2']}B)")
        print(f"  Decoder:      {p['decoder']:>10,} params")
        print(f"  Total:        {p['total']:>10,} params")
        print("-" * 60)
        data_bytes = self.sem_levels * SEM_POSITIONS + self.det_l1_tokens + self.det_l2_tokens
        print(f"  Payload:  {_HEADER}B header + {data_bytes}B data = {self.budget}B total")
        print(f"  bpp (128×128): {self.budget*8/(128*128):.4f}")
        print("=" * 60)
