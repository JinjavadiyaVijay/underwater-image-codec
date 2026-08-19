"""Configuration system for UWCodec (general-purpose underwater image codec).

No fish/species/label dependencies. Supports arbitrary RGB images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PayloadConfig:
    """Byte layout for the codec payload.

    Minimal overhead: just version (1B) + CRC (1B) = 2 bytes fixed.
    All remaining bytes go to VQ latent tokens.

    At 64B:  62 bytes for VQ data (496 bits)
    At 96B:  94 bytes for VQ data (752 bits)
    At 124B: 122 bytes for VQ data (976 bits)
    """

    # Fixed overhead (minimal — every byte saved helps at 64B budgets)
    version_bytes: int = 1   # protocol version
    crc_bytes: int = 1       # CRC-8 integrity

    @property
    def fixed_overhead(self) -> int:
        """Total non-VQ bytes (version + CRC)."""
        return self.version_bytes + self.crc_bytes

    def vq_bytes(self, max_bytes: int) -> int:
        """Bytes available for VQ token data at a given budget.

        Args:
            max_bytes: Total payload budget (64, 96, 124, 256, 512, ...).

        Returns:
            Number of bytes available for VQ indices.

        Raises:
            ValueError: If budget is below minimum.
        """
        avail = max_bytes - self.fixed_overhead
        if avail < 1:
            raise ValueError(
                f"Budget {max_bytes}B is too small. Minimum is {self.fixed_overhead + 1}B."
            )
        return avail

    def summary(self, max_bytes: int = 124) -> dict[str, int]:
        """Return byte allocation breakdown."""
        vq = self.vq_bytes(max_bytes)
        return {
            "version": self.version_bytes,
            "crc": self.crc_bytes,
            "vq_tokens": vq,
            "total": max_bytes,
            "overhead_pct": round(self.fixed_overhead / max_bytes * 100, 1),
        }


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Designed for general underwater images at extreme compression ratios.
    """

    # Input resolution (encoder operates at this resolution)
    # Experimentally test: 128, 256, 320
    input_size: int = 128

    # Encoder CNN channels (MobileNet-style)
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_latent_dim: int = 64  # feature dimension per spatial position

    # VQ codebook
    # At 1B/index (256-entry codebook):
    #   128×128 → 8×8 spatial → 64 positions → 64B of indices
    #   So 64B budget = 62B VQ = 62 indices max (need to fit in spatial grid)
    codebook_size: int = 256    # 1 byte per index
    codebook_dim: int = 64      # must match encoder_latent_dim

    # Decoder CNN channels (receiver-side, can be larger)
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])

    # Output resolution (can differ from input_size for reconstruction)
    output_size: int = 128


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    seed: int = 42
    epochs: int = 50           # start small; extend only if learning confirmed
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 3

    # Loss weights
    lambda_pixel: float = 1.0
    lambda_perceptual: float = 0.5
    lambda_vq: float = 1.0      # VQ commitment loss weight

    # Target byte budgets (used for rate-distortion evaluation)
    target_budgets: list[int] = field(
        default_factory=lambda: [64, 96, 124, 256, 512, 1024, 2048, 4096]
    )

    # Primary training budget
    train_budget: int = 124

    # Checkpointing
    save_every: int = 10
    eval_every: int = 5

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class DataConfig:
    """Dataset configuration.

    No labels required. Supports any directory of RGB images.
    """

    # Path to local dataset directory (set by user)
    data_root: str = ""

    # Supported image extensions
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])

    # Splits
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10

    # Near-duplicate prevention: images within this hash distance are deduplicated
    dedup_threshold: int = 8  # perceptual hash distance

    # Preprocessing
    color_correction: str = "gray_world"  # gray_world | histogram_eq | none
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation (training only)
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.1
    rotation_degrees: int = 10


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_ms_ssim: bool = True
    compute_lpips: bool = True     # requires lpips package
    compute_uciqe: bool = True
    compute_uiqm: bool = True
    compute_edge_preservation: bool = True

    # Rate-distortion sweep budgets
    rd_budgets: list[int] = field(
        default_factory=lambda: [64, 96, 124, 256, 512, 1024, 2048, 4096]
    )

    # Visual comparison samples
    num_vis_samples: int = 8


@dataclass
class OracleConfig:
    """Oracle experiment configuration.

    The general image oracle uses NO learning. It tests the hypothesis:
    can we encode ANYTHING useful in 64-124 bytes?
    """

    # Strategies to test (in order of information preserved)
    strategies: list[str] = field(
        default_factory=lambda: ["tiny_pixel_grid", "dct_coefficients", "mean_color_blocks"]
    )

    # Budgets to test
    budgets: list[int] = field(
        default_factory=lambda: [64, 96, 124, 256, 512, 1024, 2048]
    )

    # Pass/fail criteria (honest about extreme compression)
    min_psnr_pass: float = 15.0   # >15dB means some structure preserved
    min_ssim_pass: float = 0.3    # >0.3 means broadly similar


@dataclass
class UWCodecConfig:
    """Top-level configuration for UWCodec (general-purpose)."""

    payload: PayloadConfig = field(default_factory=PayloadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)

    output_dir: str = "outputs"
    experiment_name: str = "uwcodec_v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UWCodecConfig":
        """Load config from YAML, merging with defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        config = cls()
        _recursive_update(config, overrides)
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=False)

    def print_budget_summary(self) -> None:
        """Print byte allocation for all target budgets."""
        print("=" * 50)
        print("UWCodec Payload Budget Summary")
        print("=" * 50)
        for budget in self.training.target_budgets:
            s = self.payload.summary(budget)
            bar_vq = "█" * min(s["vq_tokens"], 50)
            print(f"\n  {budget:4d}B: overhead={s['overhead_pct']}% | "
                  f"VQ={s['vq_tokens']}B  {bar_vq}")


def _recursive_update(obj: Any, overrides: dict) -> None:
    """Recursively update dataclass fields from dict."""
    import dataclasses
    if not dataclasses.is_dataclass(obj):
        return
    for key, value in overrides.items():
        if hasattr(obj, key):
            current = getattr(obj, key)
            if dataclasses.is_dataclass(current) and isinstance(value, dict):
                _recursive_update(current, value)
            else:
                setattr(obj, key, value)
