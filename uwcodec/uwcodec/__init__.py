"""UWCodec: General-purpose semantic codec for underwater image transmission.

Compresses ANY RGB image to extremely low bitrates (≤64-124 bytes) for BLE 
transmission. Uses a learned unconditional VQ-VAE for semantic reconstruction.

Usage:
    from uwcodec import UWCodec

    codec = UWCodec.load("model_path.pt")
    
    # Payload is exactly 124 bytes
    payload_bytes = codec.encode(image, max_bytes=124)
    
    # Reconstructed image
    result = codec.decode(payload_bytes)
"""

from uwcodec.version import __version__

# Lazy import to avoid circular dependency issues during development
def __getattr__(name):
    if name == "UWCodec":
        from uwcodec.core.codec import UWCodec
        return UWCodec
    raise AttributeError(f"module 'uwcodec' has no attribute {name}")

__all__ = ["UWCodec", "__version__"]
