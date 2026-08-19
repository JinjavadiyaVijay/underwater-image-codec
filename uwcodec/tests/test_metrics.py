"""Tests for image quality metrics."""

import pytest
import numpy as np

from uwcodec.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_uciqe,
    compute_uiqm,
    compute_all_metrics,
)


class TestPSNR:
    def test_identical_images(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        assert compute_psnr(img, img) == float("inf")

    def test_different_images(self):
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.full((64, 64, 3), 128, dtype=np.uint8)
        psnr = compute_psnr(img1, img2)
        assert 0 < psnr < 40

    def test_small_noise(self):
        img1 = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        noise = np.random.randint(-5, 5, img1.shape, dtype=np.int16)
        img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        psnr = compute_psnr(img1, img2)
        assert psnr > 30  # Small noise = high PSNR


class TestSSIM:
    def test_identical_images(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim > 0.99

    def test_different_images(self):
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.full((64, 64, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(img1, img2)
        assert ssim < 0.5


class TestUnderwaterMetrics:
    def test_uciqe_range(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        uciqe = compute_uciqe(img)
        assert uciqe >= 0

    def test_uiqm_range(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        uiqm = compute_uiqm(img)
        assert isinstance(uiqm, float)


class TestAllMetrics:
    def test_compute_all(self):
        img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = compute_all_metrics(img1, img2, payload_bytes=124, compute_lpips_flag=False)
        assert "psnr" in result.metrics
        assert "ssim" in result.metrics
        assert "uciqe_original" in result.metrics
        assert result.payload_bytes == 124
