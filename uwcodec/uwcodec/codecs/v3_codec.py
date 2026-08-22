"""UWCodec V3 Codec wrapper.

Orchestrates the 1D tokenizer architecture:
V3Encoder → VectorQuantizer (4096 entries) → Serialization (12-bit packing) → V3Decoder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uwcodec.models.v3_encoder import V3Encoder
from uwcodec.models.v3_decoder import V3Decoder
from uwcodec.models.quantizer import VectorQuantizer
from uwcodec.core.payload import PayloadFormat, EncodedPayload


class UWCodecV3(nn.Module):
    """TiTok-style 1D Tokenizer for underwater image compression.
    
    Compresses 128x128 images into exactly 128 bytes.
    Actually, 64 tokens of 12-bits = 96 bytes.
    We pack them into the available payload space.
    """

    def __init__(
        self,
        input_size: int = 128,
        embed_dim: int = 256,
        num_latent_tokens: int = 64,
        codebook_size: int = 4096,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_latent_tokens = num_latent_tokens
        self.codebook_size = codebook_size
        
        self.encoder = V3Encoder(
            input_size=input_size,
            embed_dim=embed_dim,
            num_latent_tokens=num_latent_tokens,
            depth=encoder_depth,
        )
        
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=embed_dim,
        )
        
        self.decoder = V3Decoder(
            embed_dim=embed_dim,
            num_latent_tokens=num_latent_tokens,
            input_size=input_size,
            depth=decoder_depth,
        )
        
        self._payload_format = PayloadFormat()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Training forward pass.
        
        Args:
            x: (B, 3, H, W) float [0, 1]
            
        Returns:
            recon: (B, 3, H, W) reconstructed image
            info: dict with vq_loss, perplexity, etc.
        """
        # Encode: (B, 3, H, W) -> (B, num_latent_tokens, D)
        latent = self.encoder(x)
        
        # Quantize. VectorQuantizer expects (B, D, H, W)
        # Reshape to (B, D, num_latent_tokens, 1)
        latent_reshape = latent.transpose(1, 2).unsqueeze(-1)
        z_q, info = self.quantizer(latent_reshape)
        
        # Reshape back to (B, num_latent_tokens, D)
        z_q = z_q.squeeze(-1).transpose(1, 2)
        
        # We also need to fix the indices shape in info
        # Info indices is (B, num_latent_tokens, 1), make it (B, num_latent_tokens)
        info["indices"] = info["indices"].squeeze(-1)
        
        # Decode: (B, num_latent_tokens, D) -> (B, 3, H, W)
        recon = self.decoder(z_q)
        recon = torch.sigmoid(recon)
        
        return recon, info

    def pack_12bit(self, indices: torch.Tensor) -> bytes:
        """Pack 64 12-bit integers into 96 bytes.
        
        indices: (64,) uint16 tensor with values in [0, 4095]
        """
        # Convert to numpy array of uint16
        idx_np = indices.cpu().numpy().astype(np.uint16)
        
        # We have 64 elements of 12 bits each = 768 bits = 96 bytes.
        # Pack 2 12-bit integers into 3 bytes.
        # 64 / 2 = 32 pairs. 32 * 3 = 96 bytes.
        packed = bytearray(96)
        
        for i in range(32):
            val1 = idx_np[2*i]
            val2 = idx_np[2*i + 1]
            
            # val1 takes byte 0 and top 4 bits of byte 1
            # val2 takes bottom 4 bits of byte 1 and byte 2
            packed[3*i]     = (val1 >> 4) & 0xFF
            packed[3*i + 1] = ((val1 & 0x0F) << 4) | ((val2 >> 8) & 0x0F)
            packed[3*i + 2] = val2 & 0xFF
            
        return bytes(packed)
        
    def unpack_12bit(self, packed: bytes) -> torch.Tensor:
        """Unpack 96 bytes into 64 12-bit integers."""
        assert len(packed) >= 96, f"Packed data too short: {len(packed)} < 96"
        
        indices = np.zeros(64, dtype=np.uint16)
        for i in range(32):
            b0 = packed[3*i]
            b1 = packed[3*i + 1]
            b2 = packed[3*i + 2]
            
            val1 = (b0 << 4) | (b1 >> 4)
            val2 = ((b1 & 0x0F) << 8) | b2
            
            indices[2*i] = val1
            indices[2*i + 1] = val2
            
        return torch.from_numpy(indices).long()

    def encode(self, image: np.ndarray, max_bytes: int = 128) -> EncodedPayload:
        """Encode RGB image to payload."""
        device = next(self.parameters()).device
        x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            latent = self.encoder(x)
            latent_reshape = latent.transpose(1, 2).unsqueeze(-1)
            _, info = self.quantizer(latent_reshape)
            indices = info["indices"].squeeze().cpu() # (64,)
            
        vq_data = self.pack_12bit(indices)
        return self._payload_format.pack(vq_data, max_bytes=max_bytes)
        
    def decode(self, payload: bytes | EncodedPayload) -> np.ndarray:
        """Decode payload to RGB image."""
        if isinstance(payload, EncodedPayload):
            payload = payload.raw_bytes
            
        vq_data, version = self._payload_format.unpack(payload)
        
        device = next(self.parameters()).device
        indices = self.unpack_12bit(vq_data).unsqueeze(0).to(device) # (1, 64)
        
        with torch.no_grad():
            z_q = self.quantizer.indices_to_codes(indices) # (1, 64, D)
            recon = self.decoder(z_q)
            recon = torch.sigmoid(recon)
            
        recon_np = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return recon_np

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
        
    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "UWCodecV3":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        
        model = cls(
            input_size=config["input_size"],
            embed_dim=config.get("embed_dim", 256),
            num_latent_tokens=config.get("num_latent_tokens", 64),
            codebook_size=config.get("codebook_size", 4096),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
        
    def save(self, path: str | Path, train_state: dict | None = None):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        ckpt = {
            "config": {
                "input_size": self.input_size,
                "embed_dim": self.encoder.embed_dim,
                "num_latent_tokens": self.num_latent_tokens,
                "codebook_size": self.codebook_size,
            },
            "state_dict": self.state_dict(),
            "train_state": train_state or {},
        }
        torch.save(ckpt, path)
