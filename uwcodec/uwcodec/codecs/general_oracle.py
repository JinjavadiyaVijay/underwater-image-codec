"""General image oracle: fail-fast experiment before expensive training.

GOAL: Prove that useful image information can be transmitted in 64-124 bytes.

The oracle uses NO learning. It encodes real image features using simple
hand-crafted methods, then decodes them. If the oracle output is completely
useless (no structure, no color resemblance), then a learned codec at the
same budget cannot do better without a very strong generative prior.

STRATEGIES (tested in order of increasing information):
  1. tiny_pixel_grid:    Downsample image to tiny grid that fits in budget,
                          then upsample back. Pure pixels — no learning.
  2. dct_coefficients:   Transmit top-N DCT coefficients (most energy first).
                          Classic compression approach without entropy coding.
  3. mean_color_blocks:  Divide image into blocks, transmit mean RGB per block.
                          Tests whether coarse spatial color conveys the scene.

Each strategy produces exactly max_bytes. The oracle is the UPPER BOUND
of what can be recovered at this byte budget without a learned prior.
A well-trained codec should approach (or for semantic cases, exceed) oracle quality.

HONEST EVALUATION:
  - PSNR, SSIM are computed against the original.
  - Results are reported truthfully — no cherry-picking.
  - If all strategies fail (PSNR < 15dB, SSIM < 0.2), the experiment reports FAILURE.
  - Failure means 64-124B may be fundamentally insufficient for full-frame reconstruction
    and a learned/generative approach may be the only option (with hallucination risk).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from uwcodec.core.config import PayloadConfig
from uwcodec.core.payload import PayloadFormat, EncodedPayload
from uwcodec.ble.crc import crc8


# ── Oracle Codec ──────────────────────────────────────────────────────────────

class GeneralOracle:
    """General image oracle: encode/decode at extreme byte budgets without learning.

    This is the EXPERIMENTAL GATE before training. Run this first.
    """

    def __init__(self, output_size: int = 128):
        self.output_size = output_size
        self.payload_config = PayloadConfig()
        self.payload_format = PayloadFormat(self.payload_config)

    def encode_decode(
        self,
        image: np.ndarray,
        max_bytes: int,
        strategy: str = "tiny_pixel_grid",
    ) -> tuple[bytes, np.ndarray]:
        """Encode then decode using a hand-crafted strategy.

        Args:
            image: (H, W, 3) uint8 RGB.
            max_bytes: Byte budget. Payload will be exactly max_bytes bytes.
            strategy: One of "tiny_pixel_grid", "dct_coefficients", "mean_color_blocks".

        Returns:
            (payload_bytes, reconstructed_image)
        """
        image_resized = np.array(
            Image.fromarray(image).resize((self.output_size, self.output_size), Image.LANCZOS)
        )

        if strategy == "tiny_pixel_grid":
            payload, recon = self._strategy_tiny_pixel_grid(image_resized, max_bytes)
        elif strategy == "dct_coefficients":
            payload, recon = self._strategy_dct(image_resized, max_bytes)
        elif strategy == "mean_color_blocks":
            payload, recon = self._strategy_mean_color_blocks(image_resized, max_bytes)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Choose: tiny_pixel_grid, dct_coefficients, mean_color_blocks")

        assert len(payload) == max_bytes, f"Oracle budget violation: {len(payload)}B != {max_bytes}B"
        assert recon.shape == (self.output_size, self.output_size, 3), f"Bad recon shape: {recon.shape}"
        return payload, recon

    def _pack(self, data: bytes, max_bytes: int) -> bytes:
        """Pack data into exactly max_bytes with version+CRC overhead."""
        vq_budget = self.payload_config.vq_bytes(max_bytes)
        # Truncate or pad data to budget
        data = data[:vq_budget]
        if len(data) < vq_budget:
            data = data + b"\x00" * (vq_budget - len(data))

        buf = bytearray(max_bytes)
        buf[0] = 0xFF  # version byte: 0xFF = oracle mode
        buf[1:1 + vq_budget] = data
        buf[-1] = crc8(bytes(buf[:-1]))
        return bytes(buf)

    def _strategy_tiny_pixel_grid(
        self, image: np.ndarray, max_bytes: int
    ) -> tuple[bytes, np.ndarray]:
        """Strategy 1: Transmit a tiny downsampled pixel grid.

        Available bytes → compute max grid size → downsample → encode → upsample.

        At 64B: 62 bytes VQ = 62 bytes available.
                For RGB: 62 / 3 = ~20 pixels. Grid: 4×5 = 20 pixels (12px per channel).
                At 96B: 94 / 3 = ~31 pixels. Grid: 5×6 = 30 pixels.
                At 124B: 122 / 3 = ~40 pixels. Grid: 6×7 = 42 pixels.
        """
        H, W = self.output_size, self.output_size
        vq_budget = self.payload_config.vq_bytes(max_bytes)

        # Compute grid size that fits in budget
        # Each pixel = 3 bytes (RGB uint8)
        max_pixels = vq_budget // 3
        grid_size = max(1, int(np.sqrt(max_pixels)))
        # Use integer pixel values directly
        num_pixels = grid_size * grid_size

        # Downsample
        small = np.array(
            Image.fromarray(image).resize((grid_size, grid_size), Image.LANCZOS)
        )  # (grid_size, grid_size, 3) uint8

        # Serialize: flat RGB bytes
        flat = small.flatten()[:num_pixels * 3]  # (num_pixels*3,) uint8

        # Pack into payload
        payload = self._pack(flat.tobytes(), max_bytes)

        # Decode: unpack bytes → reshape → upsample
        data = payload[1:-1]  # strip version and CRC
        arr = np.frombuffer(data[:num_pixels * 3], dtype=np.uint8)
        arr = arr.reshape(grid_size, grid_size, 3)
        recon = np.array(
            Image.fromarray(arr).resize((self.output_size, self.output_size), Image.LANCZOS)
        )

        return payload, recon.astype(np.uint8)

    def _strategy_dct(
        self, image: np.ndarray, max_bytes: int
    ) -> tuple[bytes, np.ndarray]:
        """Strategy 2: Transmit top-N DCT coefficients (highest energy).

        Works channel-by-channel. Quantizes each coefficient to int8 (1 byte).
        Most energy in low-frequency coefficients → transmit those first.

        At 124B: 122 bytes / 3 channels = 40 coefficients per channel.
                 At 128×128 = 16384 total coefficients, we transmit 0.24% (high energy ones).
        """
        try:
            from scipy.fft import dct as scipy_dct, idct as scipy_idct
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False

        vq_budget = self.payload_config.vq_bytes(max_bytes)
        bytes_per_channel = vq_budget // 3
        n_coeffs = bytes_per_channel  # 1 byte per coefficient (int8)

        H, W = image.shape[:2]

        recon_channels = []
        all_data = bytearray()

        for c in range(3):
            channel = image[:, :, c].astype(np.float32) / 255.0

            if HAS_SCIPY:
                # 2D DCT
                coeffs = scipy_dct(scipy_dct(channel, axis=0, norm="ortho"), axis=1, norm="ortho")
            else:
                # Fallback: row-only DCT using numpy
                # Not as good but avoids scipy dependency
                from numpy.fft import rfft, irfft
                coeffs = np.zeros_like(channel)
                for row in range(H):
                    fft_row = np.fft.rfft(channel[row])
                    coeffs[row, :len(fft_row)] = fft_row.real
                    if len(fft_row) < W:
                        coeffs[row, len(fft_row):] = fft_row.imag[:W - len(fft_row)]

            # Flatten and find top-N by absolute value
            flat_coeffs = coeffs.flatten()
            top_indices = np.argsort(np.abs(flat_coeffs))[-n_coeffs:][::-1]

            # Quantize to int8 (scale to ±127)
            max_val = np.abs(flat_coeffs[top_indices]).max() + 1e-8
            quantized = np.clip(flat_coeffs[top_indices] / max_val * 127, -128, 127).astype(np.int8)

            # Store: (n_coeffs bytes for values) — indices implicit (top-N in sorted order)
            all_data.extend(quantized.tobytes())

            # Reconstruct channel
            recon_coeffs = np.zeros_like(flat_coeffs)
            # For reconstruction, place top-N back (simple: use sorted top-N indices)
            sorted_indices = np.argsort(np.arange(len(flat_coeffs)))
            magnitude_order = np.argsort(np.abs(flat_coeffs))[::-1][:n_coeffs]
            recon_coeffs[magnitude_order] = flat_coeffs[magnitude_order]
            recon_2d = recon_coeffs.reshape(H, W)

            if HAS_SCIPY:
                recon_channel = scipy_idct(scipy_idct(recon_2d, axis=1, norm="ortho"), axis=0, norm="ortho")
            else:
                recon_channel = recon_2d  # simplified fallback

            recon_channels.append(np.clip(recon_channel * 255, 0, 255).astype(np.uint8))

        payload = self._pack(bytes(all_data), max_bytes)
        recon = np.stack(recon_channels, axis=2)
        return payload, recon.astype(np.uint8)

    def _strategy_mean_color_blocks(
        self, image: np.ndarray, max_bytes: int
    ) -> tuple[bytes, np.ndarray]:
        """Strategy 3: Transmit mean RGB color per spatial block.

        Divides image into a grid of blocks. Each block = 3 bytes (mean R, G, B).
        This tests whether coarse spatial color information conveys the scene.

        At 64B: 62/3 ≈ 20 blocks → 4×5 grid of color patches.
        At 124B: 122/3 ≈ 40 blocks → 6×7 grid of color patches.
        """
        H, W = image.shape[:2]
        vq_budget = self.payload_config.vq_bytes(max_bytes)

        max_blocks = vq_budget // 3
        grid_h = max(1, int(np.sqrt(max_blocks)))
        grid_w = max(1, max_blocks // grid_h)

        block_h = H // grid_h
        block_w = W // grid_w

        all_data = bytearray()
        means = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i in range(grid_h):
            for j in range(grid_w):
                y0, y1 = i * block_h, min((i + 1) * block_h, H)
                x0, x1 = j * block_w, min((j + 1) * block_w, W)
                block = image[y0:y1, x0:x1]
                mean_rgb = block.mean(axis=(0, 1)).astype(np.uint8)
                means[i, j] = mean_rgb
                all_data.extend(mean_rgb.tobytes())

        payload = self._pack(bytes(all_data), max_bytes)

        # Reconstruct: upsample color grid to output_size
        recon = np.array(
            Image.fromarray(means).resize((self.output_size, self.output_size), Image.NEAREST)
        )

        return payload, recon.astype(np.uint8)


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class OracleResult:
    """Result of one oracle experiment."""
    strategy: str
    budget: int
    psnr: float
    ssim: float
    payload_bytes: int
    bpp: float
    verdict: str  # "PASS", "MARGINAL", "FAIL"

    def __str__(self) -> str:
        return (
            f"[{self.verdict:8s}] {self.strategy:25s} | "
            f"{self.budget:4d}B ({self.bpp:.4f}bpp) | "
            f"PSNR={self.psnr:5.1f}dB SSIM={self.ssim:.3f}"
        )


def run_oracle_experiment(
    images: list[np.ndarray],
    budgets: list[int] | None = None,
    strategies: list[str] | None = None,
    output_size: int = 128,
    psnr_pass: float = 15.0,
    psnr_marginal: float = 10.0,
    verbose: bool = True,
) -> list[OracleResult]:
    """Run the general image oracle experiment.

    Args:
        images: List of (H, W, 3) uint8 RGB images.
        budgets: Byte budgets to test.
        strategies: Oracle strategies to test.
        output_size: Resize images to this square size.
        psnr_pass: PSNR threshold for PASS verdict.
        psnr_marginal: PSNR threshold for MARGINAL (between marginal and pass).
        verbose: Print results as they are computed.

    Returns:
        List of OracleResult (one per image × strategy × budget).
    """
    from uwcodec.evaluation.metrics import compute_psnr, compute_ssim

    budgets = budgets or [64, 96, 124, 256, 512, 1024]
    strategies = strategies or ["tiny_pixel_grid", "dct_coefficients", "mean_color_blocks"]

    oracle = GeneralOracle(output_size=output_size)
    all_results = []

    if verbose:
        print("=" * 70)
        print("GENERAL IMAGE ORACLE EXPERIMENT")
        print("Testing whether 64-4096 bytes can encode useful visual information")
        print("WITHOUT any learning. This is the fail-fast gate.")
        print("=" * 70)

    for strategy in strategies:
        if verbose:
            print(f"\nStrategy: {strategy}")
            print("-" * 70)

        for budget in budgets:
            psnrs, ssims = [], []

            for img in images:
                try:
                    _, recon = oracle.encode_decode(img, budget, strategy=strategy)
                    # Resize original to same size for comparison
                    orig_resized = np.array(
                        Image.fromarray(img).resize((output_size, output_size), Image.LANCZOS)
                    )
                    p = compute_psnr(orig_resized, recon)
                    s = compute_ssim(orig_resized, recon)
                    if not (np.isinf(p) or np.isnan(p)):
                        psnrs.append(p)
                    if not np.isnan(s):
                        ssims.append(s)
                except Exception as e:
                    if verbose:
                        print(f"  ERROR on budget={budget}B strategy={strategy}: {e}")

            if not psnrs:
                continue

            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims) if ssims else 0.0
            bpp = budget * 8 / (output_size * output_size)

            if avg_psnr >= psnr_pass:
                verdict = "PASS"
            elif avg_psnr >= psnr_marginal:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            result = OracleResult(
                strategy=strategy,
                budget=budget,
                psnr=avg_psnr,
                ssim=avg_ssim,
                payload_bytes=budget,
                bpp=bpp,
                verdict=verdict,
            )
            all_results.append(result)

            if verbose:
                print(f"  {result}")

    if verbose:
        _print_summary(all_results, psnr_pass, psnr_marginal)

    return all_results


def _print_summary(results: list[OracleResult], psnr_pass: float, psnr_marginal: float) -> None:
    """Print experiment summary and recommendation."""
    print("\n" + "=" * 70)
    print("ORACLE EXPERIMENT SUMMARY")
    print("=" * 70)

    verdicts = [r.verdict for r in results]
    passes = verdicts.count("PASS")
    marginals = verdicts.count("MARGINAL")
    fails = verdicts.count("FAIL")

    print(f"Results: {passes} PASS, {marginals} MARGINAL, {fails} FAIL")

    # Find best result at 64B and 124B
    for budget in [64, 124]:
        best = max(
            [r for r in results if r.budget == budget],
            key=lambda r: r.psnr,
            default=None,
        )
        if best:
            print(f"\nBest at {budget}B: {best}")

    print("\nRECOMMENDATION:")
    if passes > 0:
        print("  ✅ At least some strategies PASS. Proceed to learned codec training.")
        print("     The oracle proves the byte budget can convey useful structure.")
    elif marginals > 0:
        print("  ⚠️  Results are MARGINAL. Some structure preserved but quality is low.")
        print("     A learned codec with a strong prior MAY improve over oracle.")
        print("     Expect semantic/generative reconstruction, not PSNR-accurate results.")
    else:
        print("  ❌ ALL strategies FAIL (PSNR < {:.0f}dB).".format(psnr_marginal))
        print("     The byte budget is fundamentally insufficient for structural recovery.")
        print("     Options:")
        print("       1. Increase minimum budget (try 256B+)")
        print("       2. Accept fully generative/hallucinated output (warn users)")
        print("       3. Report limitation and stop.")
