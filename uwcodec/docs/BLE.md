# UWCodec BLE Integration

UWCodec is designed for **Bluetooth Low Energy (BLE)** transmission of compressed underwater images. This document describes the BLE architecture, payload format, and integration guidelines.

## Architecture Overview

```
[Camera/Encoder MCU]                    [Receiver]
        |                                    |
  RGB image (any size)              Preinstalled decoder
        |                                    |
  UWCodec.encode()                  UWCodec.decode()
        |                                    |
  64 / 96 / 124 bytes               Reconstructed RGB image
        |                                    |
  BLE Notification ------------->  BLE Notification received
```

The key insight is that the **decoder is preinstalled** on the receiver. Only the per-image payload needs to be transmitted over BLE.

## Payload Size vs. BLE MTU

| Budget | Typical Use Case | BLE Packets (MTU=244) | BLE Packets (MTU=64) |
|---|---|---|---|
| 64B | Minimal, lowest latency | 1 | 2 |
| 96B | Balanced | 1 | 2 |
| 124B | Best quality | 1 | 3 |

With BLE 5.0 + Data Length Extension (DLE), an MTU of 244 bytes is typical, meaning **all three budgets fit in a single BLE notification**.

## Packet Format

The BLE packet format is defined in `uwcodec/ble/packet.py`.

**Single Packet (most common case):**
```
[flags:1B][seq:1B][total:1B][UWCodec payload: N bytes]
```

**Fragmented (rare, only for small MTUs):**
```
Packet 0: [0x01:1B][0:1B][N:1B][chunk_0]
Packet 1: [0x02:1B][1:1B][N:1B][chunk_1]
...
Packet N-1: [0x03:1B][N-1:1B][N:1B][chunk_N-1]
```

## UWCodec Payload Format

Inside the BLE packet, the UWCodec payload is structured as:

```
[version:1B][crc8:1B][vq_indices: N-2 bytes]
```

- **version**: Codec version byte (currently `0x01`). Used for future compatibility.
- **crc8**: CRC-8 checksum for payload integrity verification.
- **vq_indices**: Compressed image data as VQ codebook indices (1 byte each, 256-entry codebook).

## Estimated Transmission Latency

| Budget | BLE 4.2 (1Mbps) | BLE 5.0 (2Mbps) |
|---|---|---|
| 64B | ~9.6 ms | ~8.5 ms |
| 96B | ~9.6 ms | ~8.5 ms |
| 124B | ~9.6 ms | ~8.5 ms |

At a standard 7.5ms connection interval, all payloads fit within a single connection event.

## Integration Test

Run the BLE pipeline integration test to validate packetization, CRC, and timing:

```bash
python scripts/test_ble_pipeline.py

# With a trained model for full end-to-end test:
python scripts/test_ble_pipeline.py --model outputs/train/best.pt
```

## BLE GATT Service Design (Reference)

For hardware integration, a minimal GATT service is recommended:

```
Service UUID: [custom]
  Characteristic: IMAGE_PAYLOAD (Notify)
    UUID: [custom]
    Properties: NOTIFY
    Value: UWCodec payload bytes (64, 96, or 124 bytes)

  Characteristic: BUDGET_CONFIG (Write)
    UUID: [custom]
    Properties: WRITE
    Value: 1 byte (0x40=64B, 0x60=96B, 0x7C=124B)
```

The sender writes `BUDGET_CONFIG` once during connection setup. The receiver uses the same budget to configure the decoder's output size expectations.
