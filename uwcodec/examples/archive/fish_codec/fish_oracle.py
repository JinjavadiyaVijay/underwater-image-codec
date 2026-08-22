"""Oracle codec: ground-truth features → payload → decoder → RGB reconstruction.

CRITICAL EXPERIMENTAL GATE: If the oracle cannot produce a useful visible fish
image from ground-truth species + simple structure/color info, STOP and report
failure before investing in expensive learned codec training.

The oracle deliberately uses NO learned encoder. Features are handcrafted from
ground truth to test the core hypothesis: that species + coarse color + structure
+ a shared prior is enough to produce something recognizable.

Oracle decoder strategies (tested in order of simplicity):
1. Nearest-neighbor retrieval: find closest training image of same species, apply color grading
2. Mean-image: per-species mean image, color-shifted to match transmitted palette
3. Template warp: species template warped to match transmitted silhouette + colored
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from uwcodec.core.config import PayloadConfig, OracleConfig
from uwcodec.core.payload import PayloadFormat, PayloadFields, EncodedPayload
from uwcodec.data.palette import (
    DEFAULT_PALETTE,
    extract_color_map,
    decode_color_map,
    extract_dominant_colors,
)
from uwcodec.data.preprocessing import (
    extract_silhouette,
    downsample_structure,
    upsample_structure,
    canonical_resize,
)


@dataclass
class OracleTrainingData:
    """Pre-computed per-species data for oracle decoding.

    This represents the "shared prior" that the decoder has access to.
    In the real codec, this is learned. In the oracle, it's computed from
    the training set.
    """

    # Per-species image collections
    species_images: dict[int, list[np.ndarray]] = field(default_factory=dict)
    # Per-species mean images
    species_means: dict[int, np.ndarray] = field(default_factory=dict)
    # Per-species color histograms
    species_color_stats: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_species(self) -> int:
        return len(self.species_images)


class OracleEncoder:
    """Oracle encoder: extracts ground-truth features (no learning).

    Uses true labels and hand-computed features to test the reconstruction
    hypothesis at its theoretical best.
    """

    def __init__(
        self,
        payload_config: PayloadConfig | None = None,
        palette: np.ndarray | None = None,
        structure_grid_size: int = 8,
    ):
        self.payload_config = payload_config or PayloadConfig()
        self.payload_format = PayloadFormat(self.payload_config)
        self.palette = palette if palette is not None else DEFAULT_PALETTE
        self.structure_grid_size = structure_grid_size

    def encode(
        self,
        image: np.ndarray,
        species_id: int,
        confidence: float = 1.0,
        max_bytes: int = 124,
        mode: str = "visual",
        bbox: tuple[int, int, int, int] = (0, 0, 255, 255),
    ) -> EncodedPayload:
        """Encode using ground-truth species and hand-computed features.

        Args:
            image: RGB crop, uint8 (H, W, 3). Already preprocessed.
            species_id: Ground-truth species ID.
            confidence: Ground-truth confidence (1.0 for oracle).
            max_bytes: Target byte budget.
            mode: "visual" or "ai".
            bbox: Bounding box in source frame (quantized 0-255).

        Returns:
            EncodedPayload ready for decoding.
        """
        t_start = time.perf_counter()

        # Extract structure: binary silhouette
        silhouette = extract_silhouette(image)
        structure_bytes = downsample_structure(silhouette, self.structure_grid_size)

        # Extract color map: palette-indexed spatial colors
        grid_h = self.payload_config.color_map_grid_h
        grid_w = self.payload_config.color_map_grid_w
        color_map_bytes = extract_color_map(
            image,
            palette=self.palette,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        # Compute pose/orientation from silhouette
        pose = self._estimate_pose(silhouette)

        # Compute shape parameters
        shape = self._estimate_shape(silhouette)

        # For the oracle, residual tokens are just downsampled image bytes
        # (testing whether the format can carry enough info — in the real codec,
        # these would be learned VQ indices)
        residual_budget = self.payload_config.residual_bytes(max_bytes)
        residual_tokens = self._compute_oracle_residual(image, residual_budget)

        # Pack
        fields = PayloadFields(
            version=1,
            species_id=species_id,
            confidence=int(np.clip(confidence * 255, 0, 255)),
            bbox=tuple(int(v) & 0xFF for v in bbox),
            pose=pose,
            shape=shape,
            structure=structure_bytes,
            color_map=color_map_bytes,
            residual_tokens=residual_tokens,
            mode=mode,
        )

        payload = self.payload_format.pack(fields, max_bytes)
        encode_time = (time.perf_counter() - t_start) * 1000

        return payload

    def _estimate_pose(self, silhouette: np.ndarray) -> int:
        """Estimate orientation from silhouette (0-255 quantized)."""
        # Find center of mass and principal axis
        coords = np.argwhere(silhouette > 127)
        if len(coords) < 10:
            return 0

        center = coords.mean(axis=0)
        centered = coords - center

        # PCA for orientation
        cov = np.cov(centered.T)
        if cov.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.arctan2(eigvecs[0, 1], eigvecs[1, 1])
            # Quantize to 0-255
            return int(((angle + np.pi) / (2 * np.pi)) * 255) & 0xFF
        return 0

    def _estimate_shape(self, silhouette: np.ndarray) -> tuple[int, int]:
        """Estimate shape parameters from silhouette.

        Returns (elongation, compactness) each 0-255.
        """
        coords = np.argwhere(silhouette > 127)
        if len(coords) < 10:
            return (128, 128)

        # Bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h = max(y_max - y_min, 1)
        w = max(x_max - x_min, 1)

        # Elongation: w/h ratio mapped to 0-255
        elongation = np.clip(w / h, 0.2, 5.0)
        elongation_byte = int(((elongation - 0.2) / 4.8) * 255) & 0xFF

        # Compactness: area / (perimeter^2) * 4pi, mapped to 0-255
        area = len(coords)
        # Simple perimeter estimate
        filled = silhouette > 127
        eroded = filled.copy()
        eroded[1:-1, 1:-1] = (
            filled[1:-1, 1:-1]
            & filled[:-2, 1:-1]
            & filled[2:, 1:-1]
            & filled[1:-1, :-2]
            & filled[1:-1, 2:]
        )
        perimeter = (filled.astype(int) - eroded.astype(int)).sum()
        perimeter = max(perimeter, 1)

        compactness = min(4 * np.pi * area / (perimeter**2), 1.0)
        compactness_byte = int(compactness * 255) & 0xFF

        return (elongation_byte, compactness_byte)

    def _compute_oracle_residual(
        self,
        image: np.ndarray,
        budget: int,
    ) -> bytes:
        """Compute oracle residual: heavily downsampled image data.

        For the oracle, we pack as much raw pixel information as possible
        into the residual budget. This tests the upper bound of what the
        format can carry.
        """
        if budget <= 0:
            return b""

        # Strategy: downsample to tiny resolution, flatten RGB, subsample
        # Each pixel = 3 bytes, so budget / 3 = max pixels
        max_pixels = budget // 3
        if max_pixels < 1:
            # Just pack grayscale samples
            gray = np.mean(image, axis=2).astype(np.uint8)
            h, w = gray.shape
            # Sample evenly across the image
            indices = np.linspace(0, h * w - 1, budget, dtype=int)
            return bytes(gray.flatten()[indices])

        # Find nearest square resolution
        side = int(np.sqrt(max_pixels))
        side = max(2, min(side, 16))

        # Downsample
        tiny = np.array(
            Image.fromarray(image).resize((side, side), Image.LANCZOS),
            dtype=np.uint8,
        )

        flat = tiny.flatten()
        # Truncate to exactly budget bytes
        return bytes(flat[:budget])


class OracleDecoder:
    """Oracle decoder: reconstruct from payload using training data as prior.

    Tests multiple decoding strategies to find the best approach.
    """

    def __init__(
        self,
        training_data: OracleTrainingData,
        palette: np.ndarray | None = None,
        output_size: int = 128,
    ):
        self.training_data = training_data
        self.palette = palette if palette is not None else DEFAULT_PALETTE
        self.output_size = output_size

    def decode(
        self,
        payload: EncodedPayload | bytes,
        strategy: str = "nearest_neighbor",
        payload_config: PayloadConfig | None = None,
    ) -> np.ndarray:
        """Decode a payload into an RGB reconstruction.

        Args:
            payload: EncodedPayload or raw bytes.
            strategy: Decoding strategy to use.
            payload_config: Config for unpacking (if payload is raw bytes).

        Returns:
            Reconstructed RGB image, uint8 (output_size, output_size, 3).
        """
        # Unpack fields
        if isinstance(payload, bytes):
            fmt = PayloadFormat(payload_config or PayloadConfig())
            fields = fmt.unpack(payload)
        else:
            fields = payload.fields

        # AI mode: no visual reconstruction
        if fields.is_ai_mode:
            # Return a simple info image
            return self._render_ai_mode_info(fields)

        # Dispatch to strategy
        if strategy == "nearest_neighbor":
            return self._decode_nearest_neighbor(fields)
        elif strategy == "mean_image":
            return self._decode_mean_image(fields)
        elif strategy == "template_warp":
            return self._decode_template_warp(fields)
        else:
            raise ValueError(f"Unknown oracle strategy: {strategy}")

    def _decode_nearest_neighbor(self, fields: PayloadFields) -> np.ndarray:
        """Strategy 1: Find closest training image, apply color grading.

        Uses species ID to narrow search, then matches on color/structure.
        """
        species_id = fields.species_id
        candidates = self.training_data.species_images.get(species_id, [])

        if not candidates:
            # Fallback: use mean image strategy
            return self._decode_mean_image(fields)

        # Decode transmitted color map
        color_map = decode_color_map(
            fields.color_map, self.palette, output_size=self.output_size
        )

        # Find closest candidate by color distribution
        best_idx = 0
        best_dist = float("inf")

        for idx, candidate in enumerate(candidates):
            # Simple color histogram distance
            hist_cand = np.histogram(candidate.flatten(), bins=32, range=(0, 256))[0]
            hist_cand = hist_cand.astype(np.float32) / max(hist_cand.sum(), 1)

            # Use color map as target histogram proxy
            hist_target = np.histogram(color_map.flatten(), bins=32, range=(0, 256))[0]
            hist_target = hist_target.astype(np.float32) / max(hist_target.sum(), 1)

            dist = np.sum((hist_cand - hist_target) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        result = candidates[best_idx].copy()

        # Apply color grading from transmitted color map
        result = self._apply_color_grading(result, color_map)

        # Apply structure mask
        result = self._apply_structure(result, fields)

        # Blend with residual data if available
        result = self._blend_residual(result, fields)

        return result

    def _decode_mean_image(self, fields: PayloadFields) -> np.ndarray:
        """Strategy 2: Per-species mean image, color-shifted."""
        species_id = fields.species_id
        mean_img = self.training_data.species_means.get(species_id)

        if mean_img is None:
            # No mean available — generate from solid color
            color_map = decode_color_map(
                fields.color_map, self.palette, output_size=self.output_size
            )
            return color_map

        result = mean_img.copy()

        # Apply color grading from transmitted color map
        color_map = decode_color_map(
            fields.color_map, self.palette, output_size=self.output_size
        )
        result = self._apply_color_grading(result, color_map)

        # Apply structure
        result = self._apply_structure(result, fields)

        # Blend residual
        result = self._blend_residual(result, fields)

        return result

    def _decode_template_warp(self, fields: PayloadFields) -> np.ndarray:
        """Strategy 3: Species template warped to match transmitted silhouette + colored."""
        # Start with mean image
        result = self._decode_mean_image(fields)

        # Apply stronger structure enforcement
        if fields.structure:
            structure_mask = upsample_structure(
                fields.structure,
                grid_size=8,  # default oracle grid
                output_size=self.output_size,
            )

            # Mask background to underwater blue
            bg_color = np.array([20, 80, 120], dtype=np.uint8)
            mask_3d = np.stack([structure_mask] * 3, axis=-1) / 255.0

            result = (result * mask_3d + bg_color * (1 - mask_3d)).astype(np.uint8)

        return result

    def _apply_color_grading(
        self,
        image: np.ndarray,
        color_map: np.ndarray,
        strength: float = 0.5,
    ) -> np.ndarray:
        """Apply color grading from the transmitted color map.

        Blends the image's colors toward the target color map.
        """
        img = image.astype(np.float32)
        cmap = color_map.astype(np.float32)

        # Ensure same size
        if img.shape[:2] != cmap.shape[:2]:
            cmap = np.array(
                Image.fromarray(color_map).resize(
                    (img.shape[1], img.shape[0]), Image.BILINEAR
                ),
                dtype=np.float32,
            )

        # Blend
        result = img * (1 - strength) + cmap * strength
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_structure(
        self,
        image: np.ndarray,
        fields: PayloadFields,
    ) -> np.ndarray:
        """Apply structure mask to enforce silhouette."""
        if not fields.structure:
            return image

        structure_mask = upsample_structure(
            fields.structure, grid_size=8, output_size=self.output_size
        )

        # Soft masking: dim background, keep foreground
        mask_float = structure_mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = mask_float[:, :, np.newaxis]

        # Background: darken but don't zero (looks more natural)
        bg_factor = 0.3
        result = image.astype(np.float32) * (mask_float + bg_factor * (1 - mask_float))
        return np.clip(result, 0, 255).astype(np.uint8)

    def _blend_residual(
        self,
        image: np.ndarray,
        fields: PayloadFields,
    ) -> np.ndarray:
        """Blend residual data into the reconstruction.

        For the oracle, residual tokens are downsampled image bytes.
        """
        if not fields.residual_tokens or len(fields.residual_tokens) < 12:
            return image

        residual = fields.residual_tokens
        num_pixels = len(residual) // 3
        if num_pixels < 4:
            return image

        side = int(np.sqrt(num_pixels))
        side = max(2, side)
        total = side * side * 3

        if total > len(residual):
            side = max(2, int(np.sqrt(len(residual) // 3)))
            total = side * side * 3

        if total > len(residual):
            return image

        # Reconstruct tiny image from residual
        tiny_data = np.frombuffer(residual[:total], dtype=np.uint8)
        tiny = tiny_data.reshape(side, side, 3)

        # Upsample
        tiny_upsampled = np.array(
            Image.fromarray(tiny).resize(
                (self.output_size, self.output_size), Image.BILINEAR
            ),
            dtype=np.float32,
        )

        # Blend: residual carries real per-pixel info, weight it higher
        residual_weight = min(0.7, num_pixels / 64.0)  # more pixels = more weight
        result = image.astype(np.float32) * (1 - residual_weight) + tiny_upsampled * residual_weight

        return np.clip(result, 0, 255).astype(np.uint8)

    def _render_ai_mode_info(self, fields: PayloadFields) -> np.ndarray:
        """Render a simple info display for AI mode (no reconstruction)."""
        img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        img[:] = [30, 30, 40]  # dark background

        # Just return a labeled placeholder
        # Real implementation would overlay text, but we keep this dependency-free
        # by just using the color map
        if fields.color_map:
            color_map = decode_color_map(
                fields.color_map, self.palette, output_size=self.output_size
            )
            # Dim it to indicate AI mode
            img = (color_map.astype(np.float32) * 0.3).astype(np.uint8)

        return img


def build_oracle_training_data(
    dataset,  # FishCropDataset
    max_per_species: int = 50,
    output_size: int = 128,
) -> OracleTrainingData:
    """Build oracle decoder's training data (shared prior) from a dataset.

    Args:
        dataset: FishCropDataset instance.
        max_per_species: Maximum images to keep per species.
        output_size: Resolution for stored images.

    Returns:
        OracleTrainingData ready for the decoder.
    """
    data = OracleTrainingData()

    for i in range(len(dataset)):
        sample = dataset[i]
        species_id = sample["species_id"]
        image = sample["image"]

        # Resize to output_size
        if image.shape[0] != output_size or image.shape[1] != output_size:
            image = canonical_resize(image, output_size)

        if species_id not in data.species_images:
            data.species_images[species_id] = []

        if len(data.species_images[species_id]) < max_per_species:
            data.species_images[species_id].append(image)

    # Compute mean images
    for species_id, images in data.species_images.items():
        stack = np.stack(images, axis=0).astype(np.float32)
        data.species_means[species_id] = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)

    return data


class OracleCodec:
    """Complete oracle codec combining encoder and decoder.

    Provides the same API as the full UWCodec for easy comparison.
    """

    def __init__(
        self,
        training_data: OracleTrainingData,
        payload_config: PayloadConfig | None = None,
        palette: np.ndarray | None = None,
        output_size: int = 128,
        strategy: str = "nearest_neighbor",
    ):
        self.payload_config = payload_config or PayloadConfig()
        self.palette = palette if palette is not None else DEFAULT_PALETTE
        self.output_size = output_size
        self.strategy = strategy

        self.encoder = OracleEncoder(
            payload_config=self.payload_config,
            palette=self.palette,
        )
        self.decoder = OracleDecoder(
            training_data=training_data,
            palette=self.palette,
            output_size=output_size,
        )

    def encode(
        self,
        image: np.ndarray,
        species_id: int,
        max_bytes: int = 124,
        mode: str = "visual",
        confidence: float = 1.0,
    ) -> EncodedPayload:
        """Oracle encode: extract ground-truth features into payload."""
        return self.encoder.encode(
            image=image,
            species_id=species_id,
            confidence=confidence,
            max_bytes=max_bytes,
            mode=mode,
        )

    def decode(
        self,
        payload: EncodedPayload | bytes,
        strategy: str | None = None,
    ) -> np.ndarray:
        """Oracle decode: reconstruct from payload."""
        return self.decoder.decode(
            payload=payload,
            strategy=strategy or self.strategy,
            payload_config=self.payload_config,
        )

    def encode_decode(
        self,
        image: np.ndarray,
        species_id: int,
        max_bytes: int = 124,
        mode: str = "visual",
        strategy: str | None = None,
    ) -> tuple[EncodedPayload, np.ndarray]:
        """Encode then decode in one call (for evaluation)."""
        t_enc_start = time.perf_counter()
        payload = self.encode(image, species_id, max_bytes, mode)
        t_enc = (time.perf_counter() - t_enc_start) * 1000

        t_dec_start = time.perf_counter()
        reconstruction = self.decode(payload, strategy)
        t_dec = (time.perf_counter() - t_dec_start) * 1000

        return payload, reconstruction
