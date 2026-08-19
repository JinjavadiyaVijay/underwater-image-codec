"""Minimal VQ-VAE codec: encoder → VQ → serialize → BLE → deserialize → decoder → RGB.

No species, no labels, no metadata. Works on ANY RGB image.

Information budget (honest):
    64B:  62 bytes VQ data = 496 bits for 128×128 = 0.031 bpp
    96B:  94 bytes VQ data = 752 bits for 128×128 = 0.046 bpp
    124B: 122 bytes VQ data = 976 bits for 128×128 = 0.060 bpp

For reference: JPEG starts looking reasonable at ~0.1 bpp.
These rates are 2-10× below JPEG. Expect coarse/semantic reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from uwcodec.core.config import UWCodecConfig, ModelConfig, PayloadConfig
from uwcodec.core.payload import PayloadFormat, EncodedPayload, PROTOCOL_VERSION
from uwcodec.models.encoder import AppearanceEncoder
from uwcodec.models.decoder import ImageDecoder
from uwcodec.models.quantizer import VectorQuantizer


def _serialize_indices(indices: torch.Tensor, num_bytes: int) -> bytes:
    """Serialize VQ indices (long tensor) to bytes.

    Args:
        indices: (B, H, W) or (H, W) integer tensor, values 0-255.
        num_bytes: Maximum number of bytes to produce (hard limit).

    Returns:
        bytes of length min(H*W, num_bytes).
    """
    if indices.dim() == 3:
        indices = indices[0]  # Take first batch item
    flat = indices.flatten().to(torch.uint8).cpu().numpy().tobytes()
    return flat[:num_bytes]  # Hard truncation to budget


def _deserialize_indices(
    data: bytes, spatial_size: tuple[int, int]
) -> torch.Tensor:
    """Deserialize VQ index bytes to spatial tensor.

    Args:
        data: Raw bytes (may be shorter than spatial_size[0]*spatial_size[1]).
        spatial_size: (H, W) of the spatial grid.

    Returns:
        (1, H, W) long tensor with index values.
    """
    H, W = spatial_size
    total = H * W
    arr = np.frombuffer(data, dtype=np.uint8)

    if len(arr) < total:
        # Pad with zeros (codebook entry 0) if data is shorter than needed
        arr = np.pad(arr, (0, total - len(arr)), constant_values=0)
    else:
        arr = arr[:total]

    return torch.from_numpy(arr.reshape(1, H, W).astype(np.int64))


class MinimalVQVAE(nn.Module):
    """Minimal VQ-VAE for extreme image compression.

    Encoder: MobileNet-style CNN → latent map
    Quantizer: Single VQ codebook (256 entries, 1 byte per index)
    Decoder: Unconditional CNN upsampler

    Training forward pass: images → encoder → VQ → decoder → reconstruction
    Inference: encode to bytes / decode from bytes
    """

    def __init__(
        self,
        input_size: int = 128,
        encoder_channels: list[int] | None = None,
        latent_dim: int = 64,
        codebook_size: int = 256,
        decoder_channels: list[int] | None = None,
        output_size: int = 128,
        num_res_blocks: int = 2,
        target_vq_tokens: int | None = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.codebook_size = codebook_size

        # Encoder
        self.encoder = AppearanceEncoder(
            channels=encoder_channels or [32, 64, 128, 256],
            latent_dim=latent_dim,
            input_size=input_size,
        )

        # Compute spatial dimensions after encoding
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            enc_out = self.encoder(dummy)
        self.spatial_h, self.spatial_w = enc_out.shape[2], enc_out.shape[3]
        self.num_spatial_positions = self.spatial_h * self.spatial_w
        
        # Budget projection if needed
        self.target_vq_tokens = target_vq_tokens
        if target_vq_tokens is not None and target_vq_tokens != self.num_spatial_positions:
            self.budget_proj_enc = nn.Linear(self.num_spatial_positions, target_vq_tokens)
            self.budget_proj_dec = nn.Linear(target_vq_tokens, self.num_spatial_positions)
        else:
            self.budget_proj_enc = None
            self.budget_proj_dec = None
            self.target_vq_tokens = self.num_spatial_positions

        # Vector quantizer (codebook_size=256 → 1 byte per index)
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
        )

        # Decoder
        self.decoder = ImageDecoder(
            latent_dim=latent_dim,
            channels=decoder_channels or [256, 128, 64, 32],
            output_size=output_size,
            num_res_blocks=num_res_blocks,
        )

        self._payload_format = PayloadFormat()

    @property
    def max_indices_for_budget(self) -> dict[int, int]:
        """Map from byte budget → max VQ indices that fit."""
        pc = PayloadConfig()
        budgets = [64, 96, 124, 256, 512, 1024, 2048, 4096]
        return {b: pc.vq_bytes(b) for b in budgets}

    def forward(self, images: torch.Tensor) -> dict:
        """Training forward pass.

        Args:
            images: (B, 3, H, W) in [0, 1].

        Returns:
            Dict with reconstruction, vq_loss, perplexity.
        """
        z = self.encoder(images)  # (B, D, H', W')
        B, D, H, W = z.shape
        
        # Apply budget projection if needed
        if self.budget_proj_enc is not None:
            # Flatten spatial dims: (B, D, H*W) -> (B, D, N)
            z_flat = z.view(B, D, H * W)
            # Project to target tokens: (B, D, T)
            z_proj = self.budget_proj_enc(z_flat)
            # Reshape back to pseudo-spatial for Quantizer (B, D, T, 1)
            z_for_vq = z_proj.unsqueeze(-1)
        else:
            z_for_vq = z
            
        z_q_proj, vq_info = self.quantizer(z_for_vq)
        
        # Inverse projection
        if self.budget_proj_dec is not None:
            # z_q_proj is (B, D, T, 1) -> flat (B, D, T)
            z_q_flat = z_q_proj.squeeze(-1)
            # Project back to spatial: (B, D, H*W)
            z_q_spatial = self.budget_proj_dec(z_q_flat)
            z_q = z_q_spatial.view(B, D, H, W)
        else:
            z_q = z_q_proj

        recon = self.decoder(z_q)

        return {
            "reconstruction": recon,
            "vq_loss": vq_info["vq_loss"],
            "perplexity": vq_info["perplexity"],
            "indices": vq_info["indices"],
            "z": z,
            "z_q": z_q,
        }

    def encode(self, image: np.ndarray, max_bytes: int) -> EncodedPayload:
        """Encode a single RGB image to a byte payload.

        Args:
            image: (H, W, 3) uint8 RGB.
            max_bytes: Byte budget. Payload will be EXACTLY this many bytes.

        Returns:
            EncodedPayload with exactly max_bytes bytes.
        """
        self.eval()
        device = next(self.parameters()).device

        # Preprocess: resize to input_size, normalize to [0, 1]
        img_pil = Image.fromarray(image).resize(
            (self.input_size, self.input_size), Image.LANCZOS
        )
        x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            z = self.encoder(x)
            B, D, H, W = z.shape
            
            # Apply budget projection if needed
            if self.budget_proj_enc is not None:
                z_flat = z.view(B, D, H * W)
                z_proj = self.budget_proj_enc(z_flat)
                z_for_vq = z_proj.unsqueeze(-1)
            else:
                z_for_vq = z
                
            z_q, vq_info = self.quantizer(z_for_vq)

        indices = vq_info["indices"]  # (1, T, 1) or (1, H', W')
        vq_budget = PayloadConfig().vq_bytes(max_bytes)
        
        # Serialize exactly target_vq_tokens or vq_budget
        vq_data = _serialize_indices(indices, vq_budget)

        payload = self._payload_format.pack(vq_data, max_bytes)

        # HARD ENFORCEMENT
        assert len(payload.raw_bytes) == max_bytes, (
            f"encode() produced {len(payload.raw_bytes)}B, budget was {max_bytes}B"
        )
        return payload

    def decode(self, payload: EncodedPayload | bytes) -> np.ndarray:
        """Decode a payload to an RGB image.

        Args:
            payload: EncodedPayload or raw bytes.

        Returns:
            (output_size, output_size, 3) uint8 RGB image.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(payload, bytes):
            vq_data, _ = self._payload_format.unpack(payload)
        elif isinstance(payload, EncodedPayload):
            vq_data = payload.vq_bytes
        else:
            raise TypeError(f"Expected bytes or EncodedPayload, got {type(payload)}")

        if self.budget_proj_dec is not None:
            spatial_shape = (self.target_vq_tokens, 1)
        else:
            spatial_shape = (self.spatial_h, self.spatial_w)

        indices = _deserialize_indices(
            vq_data, spatial_shape
        ).to(device)

        with torch.no_grad():
            z_q_proj = self.quantizer.indices_to_codes(indices)
            
            # Reshape to (B, D, H', W') or (B, D, T, 1)
            B, H, W = indices.shape
            D = self.quantizer.codebook_dim
            z_q_proj = z_q_proj.reshape(B, H, W, D).permute(0, 3, 1, 2)
            
            if self.budget_proj_dec is not None:
                # z_q_proj is (B, D, T, 1) -> flat (B, D, T)
                z_q_flat = z_q_proj.squeeze(-1)
                # Project back to spatial: (B, D, H*W)
                z_q_spatial = self.budget_proj_dec(z_q_flat)
                z_q = z_q_spatial.view(B, D, self.spatial_h, self.spatial_w)
            else:
                z_q = z_q_proj
                
            recon = self.decoder(z_q)

        img = (recon[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return img

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "codebook_size": self.codebook_size,
                "spatial_h": self.spatial_h,
                "spatial_w": self.spatial_w,
                "target_vq_tokens": self.target_vq_tokens,
            },
        }, path)
        print(f"Saved codec to {path}")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "MinimalVQVAE":
        """Load codec from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model = cls(
            input_size=config.get("input_size", 128),
            output_size=config.get("output_size", 128),
            codebook_size=config.get("codebook_size", 256),
            target_vq_tokens=config.get("target_vq_tokens", None),
            **kwargs,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        vq = sum(p.numel() for p in self.quantizer.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "quantizer": vq, "decoder": dec, "total": enc + vq + dec}
