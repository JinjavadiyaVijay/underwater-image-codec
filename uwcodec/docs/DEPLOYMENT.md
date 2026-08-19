# UWCodec Deployment Guide

This document covers performance characteristics, hardware requirements, and optimization strategies for deploying UWCodec in an embedded/edge underwater camera system.

## Target Hardware

| Component | Role | Notes |
|---|---|---|
| MB1854B (or similar) | Capture + Encoding | ARM Cortex-M or similar MCU with camera interface |
| BLE module | Transmission | BLE 4.2+ required, BLE 5.0 recommended |
| Receiver device | Decoding | Phone, tablet, or embedded Linux board |

The **encoder** runs on the camera MCU. The **decoder** runs on the receiver (phone/PC/embedded board). The decoder is preinstalled — only the per-image byte payload is transmitted.

## CPU Performance (Measured)

Results measured on CPU-only deployment (no GPU). See `outputs/profiling/profiling_results.json` for raw numbers.

| Budget | Encode Time (median) | Decode Time (median) | Total | Throughput |
|---|---|---|---|---|
| 64B  | 3.9 ms | 9.2 ms | 13.1 ms | ~76 fps |
| 96B  | 3.4 ms | 9.0 ms | 12.3 ms | ~81 fps |
| 124B | 3.0 ms | 8.8 ms | 11.8 ms | ~85 fps |

**P95 worst-case latency** (measured over 30 iterations, CPU-only):

| Budget | Encode P95 | Decode P95 |
|---|---|---|
| 64B  | 13.7 ms | 10.9 ms |
| 96B  | 6.6 ms | 14.4 ms |
| 124B | 4.5 ms | 10.6 ms |

## Model Footprint

| Metric | Value |
|---|---|
| Total parameters | ~2.85M |
| Model size (fp32) | ~10.9 MB |
| Model size (int8, quantized) | ~2.7 MB (estimated) |
| Codebook size | 256 entries |
| Spatial positions (128x128) | 64 |

## Running the Profiler

```bash
# Full profiling across all budgets
python scripts/profile_deployment.py --model outputs/train/best.pt

# Custom iteration count (for more accurate measurements)
python scripts/profile_deployment.py --model outputs/train/best.pt --num-iters 100
```

## Memory Optimization Strategies

For MCU deployment (encoder only), consider:

1. **Quantization**: Convert the encoder to int8 using PyTorch dynamic quantization:
   ```python
   import torch
   encoder_quantized = torch.quantization.quantize_dynamic(
       model.encoder, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
   )
   ```

2. **ONNX Export**: Export the encoder for use with ONNX Runtime on embedded Linux:
   ```bash
   python -m uwcodec.deployment.export_onnx --model outputs/train/best.pt
   ```

3. **STM32/MCU Export**: For raw C deployment (experimental):
   ```bash
   python -m uwcodec.deployment.export_stm32 --model outputs/train/best.pt
   ```

## BLE Payload Timing Budget

At 7.5ms BLE connection interval, the full encode+transmit pipeline fits in one connection event for all three budgets at MTU >= 130 bytes. See `docs/BLE.md` for detailed BLE integration information.
