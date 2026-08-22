"""Models subpackage: encoder, decoder, quantizer.

V1 models: AppearanceEncoder, ImageDecoder (used by MinimalVQVAE / UWCodec v1).
V2 models: SemanticEncoder, DetailEncoder, V2Decoder (used by UWCodecV2).
Quantizers: VectorQuantizer, ProductQuantizer, ResidualVQ (shared).
"""
# V1 models
from uwcodec.models.encoder import AppearanceEncoder
from uwcodec.models.decoder import ImageDecoder

# V2 models
from uwcodec.models.v2_encoder import SemanticEncoder, DetailEncoder
from uwcodec.models.v2_decoder import V2Decoder

# Quantizers (shared across versions)
from uwcodec.models.quantizer import VectorQuantizer, ProductQuantizer, ResidualVQ
