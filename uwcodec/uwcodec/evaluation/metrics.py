"""Image quality metrics: PSNR, SSIM, MS-SSIM, LPIPS, UCIQE, UIQM.

Supports both numpy arrays and torch tensors. LPIPS requires the optional
`lpips` package. MS-SSIM uses `pytorch_msssim` if available, falls back
to a numpy implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MetricsResult:
    """Container for computed metrics."""

    metrics: dict[str, float] = field(default_factory=dict)
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    payload_bytes: int = 0
    compression_ratio: float = 0.0

    def summary(self) -> str:
        lines = ["=== Metrics Summary ==="]
        for k, v in sorted(self.metrics.items()):
            if isinstance(v, float):
                lines.append(f"  {k:>30s}: {v:.4f}")
            else:
                lines.append(f"  {k:>30s}: {v}")
        if self.payload_bytes > 0:
            lines.append(f"  {'payload_bytes':>30s}: {self.payload_bytes}")
            lines.append(f"  {'compression_ratio':>30s}: {self.compression_ratio:.1f}x")
        if self.encode_time_ms > 0:
            lines.append(f"  {'encode_time_ms':>30s}: {self.encode_time_ms:.2f}")
        if self.decode_time_ms > 0:
            lines.append(f"  {'decode_time_ms':>30s}: {self.decode_time_ms:.2f}")
        return "\n".join(lines)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        original: Reference image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).

    Returns:
        PSNR in dB. Higher is better. Returns inf for identical images.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def compute_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 11,
    C1: float = 6.5025,  # (0.01 * 255)^2
    C2: float = 58.5225,  # (0.03 * 255)^2
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Numpy implementation — no external dependencies.

    Args:
        original: Reference image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).

    Returns:
        SSIM value in [0, 1]. Higher is better.
    """
    img1 = original.astype(np.float64)
    img2 = reconstructed.astype(np.float64)

    # Compute per-channel and average
    ssim_vals = []
    for c in range(min(img1.shape[2], 3)):
        ch1 = img1[:, :, c]
        ch2 = img2[:, :, c]
        ssim_vals.append(_ssim_single_channel(ch1, ch2, window_size, C1, C2))

    return float(np.mean(ssim_vals))


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    C1: float,
    C2: float,
) -> float:
    """SSIM for a single channel."""
    from scipy.ndimage import uniform_filter

    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 * img1, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, window_size) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def compute_ms_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Multi-Scale SSIM (MS-SSIM).

    Uses pytorch_msssim if available, otherwise falls back to single-scale SSIM.
    """
    try:
        from pytorch_msssim import ms_ssim
        import torch

        img1 = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2 = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # MS-SSIM requires minimum size for multi-scale
        min_dim = min(img1.shape[2], img1.shape[3])
        if min_dim < 64:
            # Fall back to single-scale SSIM for tiny images
            return compute_ssim(original, reconstructed)

        return float(ms_ssim(img1, img2, data_range=1.0))
    except ImportError:
        # Fallback to standard SSIM
        return compute_ssim(original, reconstructed)


def compute_lpips(
    original: np.ndarray,
    reconstructed: np.ndarray,
    net: str = "alex",
    _model_cache: dict = {},
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Requires the `lpips` package. Lower is better.

    Args:
        original: Reference image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).
        net: Network backbone ("alex", "vgg", "squeeze").

    Returns:
        LPIPS distance. Lower is better (0 = identical).
    """
    try:
        import lpips
        import torch

        if net not in _model_cache:
            _model_cache[net] = lpips.LPIPS(net=net, verbose=False)
            if torch.cuda.is_available():
                _model_cache[net] = _model_cache[net].cuda()

        model = _model_cache[net]

        img1 = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img2 = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        with torch.no_grad():
            dist = model(img1, img2)

        return float(dist.item())
    except ImportError:
        return float("nan")


def compute_uciqe(image: np.ndarray) -> float:
    """Compute Underwater Color Image Quality Evaluation (UCIQE).

    UCIQE = c1 * σ_c + c2 * con_l + c3 * μ_s

    Where:
    - σ_c = standard deviation of chroma
    - con_l = contrast of luminance
    - μ_s = average of saturation

    Higher is better for underwater images.
    """
    img = image.astype(np.float64) / 255.0

    # Convert to Lab-like space (simplified)
    # Using a simple approximation without full Lab conversion
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Luminance
    l = 0.299 * r + 0.587 * g + 0.114 * b

    # Chroma (simplified as color saturation in RGB space)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c

    # Saturation
    saturation = np.where(max_c > 1e-6, chroma / max_c, 0.0)

    # UCIQE components
    sigma_c = np.std(chroma)
    con_l = np.std(l)  # luminance contrast
    mu_s = np.mean(saturation)

    # Standard UCIQE weights
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s

    return float(uciqe)


def compute_uiqm(image: np.ndarray) -> float:
    """Compute Underwater Image Quality Measure (UIQM).

    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM

    Where:
    - UICM = underwater image colorfulness measure
    - UISM = underwater image sharpness measure
    - UIConM = underwater image contrast measure

    Higher is better for underwater images.
    """
    img = image.astype(np.float64)

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # UICM: Colorfulness measure
    rg = r - g
    yb = (r + g) / 2.0 - b
    uicm = np.sqrt(np.mean(rg**2) + np.mean(yb**2)) + 0.3 * (np.std(rg) + np.std(yb))

    # UISM: Sharpness measure (using gradient magnitude)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gradient_mag = np.sqrt(gx**2 + gy**2)
    eme = np.mean(gradient_mag)
    uism = np.log(1 + eme)

    # UIConM: Contrast measure
    # Using local contrast (standard deviation in small windows)
    h, w = gray.shape
    block_size = max(4, min(h, w) // 8)
    contrasts = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y : y + block_size, x : x + block_size]
            if block.std() > 0:
                contrasts.append(block.std())
    uiconm = np.mean(contrasts) if contrasts else 0.0

    # Standard UIQM weights
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm

    return float(uiqm)


def compute_color_distribution_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bins: int = 32,
) -> float:
    """Compute Earth Mover's Distance between color distributions.

    Measures how well the reconstruction preserves the original's color palette.

    Returns:
        EMD distance. Lower is better.
    """
    from scipy.stats import wasserstein_distance

    distances = []
    for c in range(3):
        hist_orig, _ = np.histogram(original[:, :, c], bins=bins, range=(0, 256))
        hist_recon, _ = np.histogram(reconstructed[:, :, c], bins=bins, range=(0, 256))

        # Normalize
        hist_orig = hist_orig.astype(np.float64) / max(hist_orig.sum(), 1)
        hist_recon = hist_recon.astype(np.float64) / max(hist_recon.sum(), 1)

        distances.append(wasserstein_distance(hist_orig, hist_recon))

    return float(np.mean(distances))


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    payload_bytes: int = 0,
    encode_time_ms: float = 0.0,
    decode_time_ms: float = 0.0,
    compute_lpips_flag: bool = True,
) -> MetricsResult:
    """Compute all available image quality metrics.

    Args:
        original: Reference image, uint8 (H, W, 3).
        reconstructed: Reconstructed image, uint8 (H, W, 3).
        payload_bytes: Size of encoded payload.
        encode_time_ms: Encoding time in milliseconds.
        decode_time_ms: Decoding time in milliseconds.
        compute_lpips_flag: Whether to compute LPIPS (requires lpips package).

    Returns:
        MetricsResult with all computed metrics.
    """
    # Ensure same size
    if original.shape != reconstructed.shape:
        from PIL import Image

        recon_pil = Image.fromarray(reconstructed)
        recon_pil = recon_pil.resize(
            (original.shape[1], original.shape[0]), Image.LANCZOS
        )
        reconstructed = np.array(recon_pil)

    metrics = {}

    # PSNR
    metrics["psnr"] = compute_psnr(original, reconstructed)

    # SSIM
    metrics["ssim"] = compute_ssim(original, reconstructed)

    # MS-SSIM
    metrics["ms_ssim"] = compute_ms_ssim(original, reconstructed)

    # LPIPS
    if compute_lpips_flag:
        metrics["lpips"] = compute_lpips(original, reconstructed)

    # Underwater-specific
    metrics["uciqe_original"] = compute_uciqe(original)
    metrics["uciqe_reconstructed"] = compute_uciqe(reconstructed)
    metrics["uciqe_delta"] = metrics["uciqe_reconstructed"] - metrics["uciqe_original"]

    metrics["uiqm_original"] = compute_uiqm(original)
    metrics["uiqm_reconstructed"] = compute_uiqm(reconstructed)
    metrics["uiqm_delta"] = metrics["uiqm_reconstructed"] - metrics["uiqm_original"]

    # Color distribution
    metrics["color_emd"] = compute_color_distribution_distance(original, reconstructed)

    # Compression ratio
    original_bytes = original.nbytes  # H * W * 3
    compression_ratio = original_bytes / max(payload_bytes, 1) if payload_bytes > 0 else 0.0

    # bpp (bits per pixel)
    num_pixels = original.shape[0] * original.shape[1]
    if payload_bytes > 0:
        metrics["bpp"] = (payload_bytes * 8) / num_pixels

    return MetricsResult(
        metrics=metrics,
        encode_time_ms=encode_time_ms,
        decode_time_ms=decode_time_ms,
        payload_bytes=payload_bytes,
        compression_ratio=compression_ratio,
    )
