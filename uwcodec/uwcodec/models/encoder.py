"""Lightweight appearance encoder: CNN → latent features for VQ.

Architecture: MobileNet-style depthwise separable convolutions.
Must be small enough for eventual STM32N6570 deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (MobileNet-style)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x


class AppearanceEncoder(nn.Module):
    """Lightweight CNN encoder producing latent features for VQ.

    Input: RGB crop (B, 3, H, W) — 64×64, 96×96, or 128×128.
    Output: Latent feature map (B, latent_dim, H', W') for quantization.

    Design constraints:
    - Small enough for STM32N6570 (Cortex-M55 + NPU, ~2MB SRAM)
    - MobileNet-style depthwise separable convolutions
    - 4-6 conv layers, ~50-200K params
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        latent_dim: int = 64,
        input_size: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers = []
        prev_ch = in_channels

        # Initial standard conv
        layers.append(nn.Conv2d(prev_ch, channels[0], 3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels[0]))
        layers.append(nn.ReLU6(inplace=True))
        prev_ch = channels[0]

        # Depthwise separable blocks
        for ch in channels[1:]:
            layers.append(DepthwiseSeparableConv(prev_ch, ch, stride=2))
            prev_ch = ch

        self.features = nn.Sequential(*layers)

        # Project to latent dimension
        self.project = nn.Conv2d(prev_ch, latent_dim, 1, bias=False)

        self._input_size = input_size
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent features.

        Args:
            x: (B, 3, H, W) float tensor, normalized to [0, 1].

        Returns:
            (B, latent_dim, H', W') latent features.
        """
        z = self.features(x)
        z = self.project(z)
        return z

    def compute_output_shape(self, input_size: int | None = None) -> tuple[int, int]:
        """Compute the spatial size of the output feature map."""
        s = input_size or self._input_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, s, s)
            out = self.forward(dummy)
        return out.shape[2], out.shape[3]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
