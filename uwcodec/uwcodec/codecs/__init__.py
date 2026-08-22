"""Codecs subpackage.

V1: MinimalVQVAE (single-branch VQ-VAE).
V2: UWCodecV2 (dual-branch semantic+detail with RVQ).
V3: UWCodecV3 (TiTok-style 1D tokenizer with Transformer).
Oracle: GeneralOracle (non-learned baselines).
"""
from uwcodec.codecs.v3_codec import UWCodecV3
