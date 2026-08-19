"""UWCodec v2 Generative UNet Decoder.

Architecture:
  (B, sem_dim, 4, 4)  semantic quantized codes
  (B, det_dim, 8, 8)  detail quantized codes (may have zero-padding for untrasmitted tokens)
  
  → Input projection (4×4, 256ch)
  → Residual blocks at 4×4                     [deepest — 4 blocks]
  → Upsample2x → 8×8, 128ch
  → FiLM conditioning from detail (8×8)
  → Residual blocks at 8×8                     [2 blocks]
  → Upsample2x → 16×16, 64ch
  → Residual blocks at 16×16                   [2 blocks]
  → Upsample2x → 32×32, 32ch
  → Residual blocks at 32×32                   [2 blocks]
  → Upsample2x → 64×64, 16ch
  → Upsample2x → 128×128, 8ch
  → Output conv → (B, 3, 128, 128) in [0, 1]

Key design decisions:
  - GroupNorm throughout (stable for small batch sizes and generation)
  - SiLU activations (smooth gradient flow)
  - FiLM (Feature-wise Linear Modulation) for detail injection at 8×8
  - Receiver-side: can be large (3-7M params). Only the encoder runs on MCU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with automatic group count reduction for small channel counts."""
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    """Residual block: GroupNorm + SiLU + Conv3×3 × 2, with skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = _gn(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Upsample2x(nn.Module):
    """Bilinear upsample ×2 then Conv3×3 + GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = _gn(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.silu(self.norm(self.conv(x)))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for detail injection.

    Learns per-channel scale (gamma) and shift (beta) from the detail feature
    map, then applies them to the semantic feature map at the same resolution:

        output = (1 + gamma(detail)) * x + beta(detail)

    The additive residual around 1 ensures stable initialization: at init,
    gamma ≈ 0 and beta ≈ 0, so the FiLM layer is effectively an identity.
    This prevents catastrophic interference early in training.
    """

    def __init__(self, detail_dim: int, target_channels: int):
        super().__init__()
        # Two separate 1×1 convs: one for scale modulation, one for shift
        self.gamma_proj = nn.Conv2d(detail_dim, target_channels, 1, bias=True)
        self.beta_proj  = nn.Conv2d(detail_dim, target_channels, 1, bias=True)

        # Stable initialization: identity FiLM at start
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C, H, W)     semantic features at current scale
            detail: (B, D_det, H, W) detail features (same spatial size)

        Returns:
            (B, C, H, W) modulated features
        """
        gamma = 1.0 + self.gamma_proj(detail)   # additive around 1
        beta  = self.beta_proj(detail)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

class V2Decoder(nn.Module):
    """Generative UNet-style decoder for UWCodec v2.

    Upsampling path:  4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128.
    Detail codes are injected at the 8×8 scale via FiLM conditioning.

    The decoder is intentionally larger than the encoder since it runs on
    the receiver (phone/PC/gateway), not the embedded MCU.

    Parameter count scales with base_channels:
      base_channels=256: ~6-7M params (default, good quality)
      base_channels=128: ~1.5M params (lighter, faster)
    """

    def __init__(
        self,
        sem_dim: int = 64,
        det_dim: int = 32,
        base_channels: int = 256,
        num_res_blocks_bottom: int = 4,
        num_res_blocks_mid: int = 2,
        output_size: int = 128,
    ):
        super().__init__()
        self.output_size = output_size
        self.sem_dim = sem_dim
        self.det_dim = det_dim

        C = base_channels

        # ---- 4×4 entry ----
        # Project semantic codes to decoder channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(sem_dim, C, 1, bias=False),
            _gn(C),
            nn.SiLU(),
        )

        # Residual processing at the deepest scale (most parameters here)
        self.bottom_blocks = nn.Sequential(
            *[ResBlock(C) for _ in range(num_res_blocks_bottom)]
        )

        # ---- 4×4 → 8×8 ----
        self.up1 = Upsample2x(C, C // 2)          # → C/2 = 128

        # FiLM conditioning from detail (at 8×8)
        self.film_8x8 = FiLMLayer(det_dim, C // 2)
        self.mid_8x8 = nn.Sequential(
            *[ResBlock(C // 2) for _ in range(num_res_blocks_mid)]
        )

        # ---- 8×8 → 16×16 ----
        self.up2 = Upsample2x(C // 2, C // 4)     # → C/4 = 64
        self.mid_16x16 = nn.Sequential(
            *[ResBlock(C // 4) for _ in range(num_res_blocks_mid)]
        )

        # ---- 16×16 → 32×32 ----
        self.up3 = Upsample2x(C // 4, C // 8)     # → C/8 = 32
        self.mid_32x32 = nn.Sequential(
            *[ResBlock(C // 8) for _ in range(num_res_blocks_mid)]
        )

        # ---- 32×32 → 64×64 ----
        self.up4 = Upsample2x(C // 8, C // 16)    # → C/16 = 16

        # ---- 64×64 → 128×128 ----
        self.up5 = Upsample2x(C // 16, C // 32)   # → C/32 = 8

        # Output convolution: two-layer head for smooth output
        self.output_head = nn.Sequential(
            nn.Conv2d(C // 32, 16, 3, padding=1, bias=False),
            _gn(16),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sem_q: torch.Tensor, det_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to RGB.

        Args:
            sem_q: (B, sem_dim, 4, 4)  semantic quantized codes (always full 16 positions)
            det_q: (B, det_dim, 8, 8)  detail quantized codes (zero-padded for untrasmitted tokens)

        Returns:
            (B, 3, output_size, output_size)  RGB in [0, 1]
        """
        # 4×4: project and process semantic
        x = self.input_proj(sem_q)
        x = self.bottom_blocks(x)

        # 4×4 → 8×8: upsample + inject detail via FiLM
        x = self.up1(x)
        x = self.film_8x8(x, det_q)
        x = self.mid_8x8(x)

        # 8×8 → 16×16 → 32×32
        x = self.up2(x)
        x = self.mid_16x16(x)
        x = self.up3(x)
        x = self.mid_32x32(x)

        # 32×32 → 64×64 → 128×128
        x = self.up4(x)
        x = self.up5(x)

        # Final resize if spatial mismatch (insurance for non-power-of-2 sizes)
        if x.shape[-1] != self.output_size or x.shape[-2] != self.output_size:
            x = F.interpolate(
                x, size=(self.output_size, self.output_size),
                mode="bilinear", align_corners=False,
            )

        return self.output_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
