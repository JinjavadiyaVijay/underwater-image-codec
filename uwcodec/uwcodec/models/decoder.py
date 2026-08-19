"""Unconditional VQ-VAE decoder: quantized latent → RGB image.

No species conditioning. No FiLM. No metadata.
Just: VQ codes → CNN upsampler → RGB.

This is the receiver-side decoder. It can be larger than the encoder
since it runs on phone/PC/gateway, not the embedded MCU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block for the decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class UpsampleBlock(nn.Module):
    """Upsample by 2× + convolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ImageDecoder(nn.Module):
    """Unconditional CNN decoder: VQ latent → RGB image.

    Input:  (B, latent_dim, H', W') quantized latent feature map.
    Output: (B, 3, output_size, output_size) RGB image in [0, 1].

    Architecture:
        initial_proj → [ResidualBlock] × num_res_blocks → [UpsampleBlock × n] → output_conv

    The number of upsample blocks is determined by the ratio output_size / spatial_size.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        channels: list[int] | None = None,
        output_size: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        if channels is None:
            channels = [256, 128, 64, 32]

        self.output_size = output_size

        # Project from latent_dim to first decoder channel
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_dim, channels[0], 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual blocks at lowest resolution
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels[0]) for _ in range(num_res_blocks)]
        )

        # Upsample blocks (progressively larger)
        self.upsample_blocks = nn.ModuleList()
        prev_ch = channels[0]
        for ch in channels[1:]:
            self.upsample_blocks.append(UpsampleBlock(prev_ch, ch))
            prev_ch = ch

        # Final output (sigmoid for [0,1] output)
        self.output_conv = nn.Sequential(
            nn.Conv2d(prev_ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to RGB.

        Args:
            z_q: (B, latent_dim, H', W') quantized feature map.

        Returns:
            (B, 3, output_size, output_size) in [0, 1].
        """
        x = self.input_proj(z_q)
        x = self.res_blocks(x)

        for block in self.upsample_blocks:
            x = block(x)

        # Final resize to exact output size if spatial mismatch
        if x.shape[2] != self.output_size or x.shape[3] != self.output_size:
            x = F.interpolate(
                x, size=(self.output_size, self.output_size),
                mode="bilinear", align_corners=False,
            )

        return self.output_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
