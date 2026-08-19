"""UWCodec v2 Dual-Branch Encoder.

Semantic branch: deep encoder → 4×4 global latent (scene/structure/geometry).
Detail branch:   shallow encoder → 8×8 local latent (edges/texture/residual).

Both branches share the same input (128×128 RGB) and are lightweight.
Semantic branch is deeper (5 stride-2 stages) to achieve global compression.
Detail branch is shallower (4 stride-2 stages) for local structure.

Parameter counts (approximate):
  SemanticEncoder: ~200K  (channels [32, 64, 128, 192, 256])
  DetailEncoder:   ~100K  (channels [32, 64, 128, 128])

These must eventually be deployable on STM32N6570 (Cortex-M55 + NPU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DSConv(nn.Module):
    """Depthwise separable convolution (MobileNet-style).

    in_ch → depthwise 3×3 → BatchNorm → ReLU6 → pointwise 1×1 → BatchNorm → ReLU6.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.dw(x)), inplace=True)
        return F.relu6(self.bn2(self.pw(x)), inplace=True)


class InvResBlock(nn.Module):
    """MobileNetV2 inverted residual block with optional skip connection."""

    def __init__(self, in_ch: int, out_ch: int, expansion: int = 4, stride: int = 1):
        super().__init__()
        mid = in_ch * expansion
        self.use_skip = (stride == 1 and in_ch == out_ch)

        layers: list[nn.Module] = []
        if expansion != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_skip:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Semantic Encoder
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """Global semantic encoder: (B, 3, 128, 128) → (B, sem_dim, 4, 4).

    5 stride-2 downsampling stages:  128 → 64 → 32 → 16 → 8 → 4.
    Captures scene geometry, dominant colors, coarse structure.

    Designed to be lightweight (< 250K params) for embedded deployment.
    The 4×4 = 16 spatial positions form the primary information budget:
      - 2-level RVQ → 32 bytes  (used for 64B and 96B budgets)
      - 3-level RVQ → 48 bytes  (used for 124B and 128B budgets)
    """

    def __init__(
        self,
        sem_dim: int = 64,
        channels: list[int] | None = None,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 192, 256]

        if len(channels) != 5:
            raise ValueError(f"SemanticEncoder requires exactly 5 channel stages, got {len(channels)}")

        # Stage 0: 128 → 64 (standard conv for first layer)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
        )

        # Stage 1: 64 → 32
        self.stage1 = DSConv(channels[0], channels[1], stride=2)

        # Stage 2: 32 → 16  (add inverted residual for capacity)
        self.stage2 = nn.Sequential(
            DSConv(channels[1], channels[2], stride=2),
            InvResBlock(channels[2], channels[2]),
        )

        # Stage 3: 16 → 8
        self.stage3 = nn.Sequential(
            DSConv(channels[2], channels[3], stride=2),
            InvResBlock(channels[3], channels[3]),
        )

        # Stage 4: 8 → 4  (deepest global features)
        self.stage4 = DSConv(channels[3], channels[4], stride=2)

        # Final projection to sem_dim
        self.project = nn.Conv2d(channels[4], sem_dim, 1, bias=False)
        self.sem_dim = sem_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 128, 128) → (B, sem_dim, 4, 4)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.project(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Detail Encoder
# ---------------------------------------------------------------------------

class DetailEncoder(nn.Module):
    """Local detail encoder: (B, 3, 128, 128) → (B, det_dim, 8, 8).

    4 stride-2 downsampling stages:  128 → 64 → 32 → 16 → 8.
    Captures edges, textures, local object boundaries.
    Intentionally shallower than SemanticEncoder.

    The 8×8 = 64 spatial positions form the secondary information budget:
      - Subset of 30/62/64 bytes sent depending on total byte budget.
      - Positions sent in raster-scan order (top-left first).
      - Untransmitted positions are zero-padded at the decoder.
    """

    def __init__(
        self,
        det_dim: int = 32,
        channels: list[int] | None = None,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 128]

        if len(channels) != 4:
            raise ValueError(f"DetailEncoder requires exactly 4 channel stages, got {len(channels)}")

        # Stage 0: 128 → 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
        )

        # Stages 1-3: 64→32→16→8
        self.stage1 = DSConv(channels[0], channels[1], stride=2)
        self.stage2 = DSConv(channels[1], channels[2], stride=2)
        self.stage3 = DSConv(channels[2], channels[3], stride=2)

        # Final projection to det_dim
        self.project = nn.Conv2d(channels[3], det_dim, 1, bias=False)
        self.det_dim = det_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 128, 128) → (B, det_dim, 8, 8)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.project(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
