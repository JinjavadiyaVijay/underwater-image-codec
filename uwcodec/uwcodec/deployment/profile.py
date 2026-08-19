"""Latency and memory profiling tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProfilingResult:
    """Profiling measurement results."""
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    model_params: int = 0
    model_size_mb: float = 0.0
    payload_bytes: int = 0
    throughput_fps: float = 0.0

    def summary(self) -> str:
        return (
            f"=== Profiling Results ===\n"
            f"  Encode:     {self.encode_time_ms:.2f} ms\n"
            f"  Decode:     {self.decode_time_ms:.2f} ms\n"
            f"  Total:      {self.total_time_ms:.2f} ms\n"
            f"  Throughput: {self.throughput_fps:.1f} FPS\n"
            f"  Memory:     {self.peak_memory_mb:.1f} MB\n"
            f"  Params:     {self.model_params:,}\n"
            f"  Model Size: {self.model_size_mb:.2f} MB\n"
            f"  Payload:    {self.payload_bytes} bytes"
        )


def profile_codec(
    codec,
    image: np.ndarray,
    species_id: int = 0,
    max_bytes: int = 124,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> ProfilingResult:
    """Profile codec encode/decode performance.

    Args:
        codec: UWCodec or OracleCodec instance.
        image: Test image (H, W, 3) uint8.
        species_id: Species for encoding.
        max_bytes: Byte budget.
        num_warmup: Warmup iterations.
        num_iterations: Measurement iterations.

    Returns:
        ProfilingResult with timing and memory data.
    """
    # Warmup
    for _ in range(num_warmup):
        payload = codec.encode(image, species_id=species_id, max_bytes=max_bytes)
        _ = codec.decode(payload)

    # Measure encoding
    encode_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        payload = codec.encode(image, species_id=species_id, max_bytes=max_bytes)
        t1 = time.perf_counter()
        encode_times.append((t1 - t0) * 1000)

    # Measure decoding
    decode_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = codec.decode(payload)
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000)

    enc_ms = np.median(encode_times)
    dec_ms = np.median(decode_times)
    total = enc_ms + dec_ms

    # Model stats
    params = 0
    size_mb = 0.0
    try:
        import torch
        if hasattr(codec, 'parameters'):
            params = sum(p.numel() for p in codec.parameters())
            size_mb = sum(p.numel() * p.element_size() for p in codec.parameters()) / (1024 ** 2)
    except (ImportError, AttributeError):
        pass

    # Memory tracking
    peak_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            payload = codec.encode(image, species_id=species_id, max_bytes=max_bytes)
            _ = codec.decode(payload)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except (ImportError, RuntimeError):
        pass

    payload_bytes = len(payload.raw_bytes) if hasattr(payload, 'raw_bytes') else 0

    return ProfilingResult(
        encode_time_ms=enc_ms,
        decode_time_ms=dec_ms,
        total_time_ms=total,
        peak_memory_mb=peak_mb,
        model_params=params,
        model_size_mb=size_mb,
        payload_bytes=payload_bytes,
        throughput_fps=1000.0 / max(total, 0.001),
    )
