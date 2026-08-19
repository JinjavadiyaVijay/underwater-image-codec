"""Dataset loading for UWCodec — general-purpose, no labels required.

Supports any directory of RGB images. No species, no crops, no YOLO.

Directory structure (any of these work):
    data_root/
        *.jpg, *.png, ...      # flat directory
    data_root/
        subdir1/*.jpg          # nested directories
        subdir2/*.png

No labels needed. The codec is self-supervised (reconstruction objective).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _image_hash(path: Path, size: int = 8) -> str:
    """Compute a perceptual-ish hash for near-duplicate detection.

    Uses a simple 8x8 grayscale downsample + binarize approach.
    Fast enough to run on thousands of images.
    """
    try:
        img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
        arr = np.array(img)
        return "".join("1" if px > arr.mean() else "0" for px in arr.flatten())
    except Exception:
        # Return file size hash as fallback
        return hashlib.md5(str(path.stat().st_size).encode()).hexdigest()


def _hamming(h1: str, h2: str) -> int:
    """Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def find_images(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    """Recursively find all images under root directory.

    Args:
        root: Directory to search.
        extensions: Set of lowercase extensions (default: SUPPORTED_EXTENSIONS).

    Returns:
        Sorted list of image paths.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    exts = extensions or SUPPORTED_EXTENSIONS
    paths = []
    for ext in exts:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))

    return sorted(set(paths))


def deduplicate_images(
    paths: list[Path],
    threshold: int = 8,
    verbose: bool = False,
) -> list[Path]:
    """Remove near-duplicate images using perceptual hashing.

    Args:
        paths: Image paths to deduplicate.
        threshold: Hamming distance threshold (0=identical, 8=quite similar).
        verbose: Print progress.

    Returns:
        Deduplicated list of paths.
    """
    if not paths:
        return paths

    hashes: list[tuple[str, Path]] = []
    kept: list[Path] = []

    for i, path in enumerate(paths):
        if verbose and i % 100 == 0:
            print(f"  Hashing {i}/{len(paths)}...")

        h = _image_hash(path)
        is_dup = any(_hamming(h, existing_h) <= threshold for existing_h, _ in hashes)

        if not is_dup:
            hashes.append((h, path))
            kept.append(path)

    if verbose:
        removed = len(paths) - len(kept)
        print(f"  Deduplicated: kept {len(kept)}/{len(paths)} (removed {removed} near-duplicates)")

    return kept


def split_paths(
    paths: list[Path],
    train: float = 0.80,
    val: float = 0.10,
    test: float = 0.10,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split paths into train/val/test with a fixed seed.

    Args:
        paths: All image paths (after deduplication).
        train, val, test: Fractions (must sum to 1.0).
        seed: Random seed for reproducible splits.

    Returns:
        (train_paths, val_paths, test_paths)
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)

    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


class UnderwaterImageDataset(TorchDataset):
    """General-purpose dataset for unlabeled underwater images.

    No species labels. No crops. Just RGB images.

    Returns dicts with:
        - "image": float32 tensor (3, H, W) in [0, 1]
        - "path": str (for debugging)
    """

    def __init__(
        self,
        paths: list[Path],
        input_size: int = 128,
        augment: bool = False,
        color_correction: str = "gray_world",
        transform: Callable | None = None,
    ):
        self.paths = paths
        self.input_size = input_size
        self.augment = augment
        self.color_correction = color_correction
        self.transform = transform

        if not HAS_TORCH:
            raise ImportError("PyTorch required for UnderwaterImageDataset")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        # Load image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a random solid color on load failure (keep training stable)
            img = Image.fromarray(
                np.random.randint(50, 200, (self.input_size, self.input_size, 3), dtype=np.uint8)
            )

        # Resize
        img = img.resize((self.input_size, self.input_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)

        # Color correction
        if self.color_correction == "gray_world":
            arr = _gray_world(arr)
        elif self.color_correction == "histogram_eq":
            arr = _histogram_equalize(arr)

        # Augmentation
        if self.augment:
            arr = _augment(arr)

        # Convert to tensor [0, 1]
        if self.transform is not None:
            tensor = self.transform(arr)
        else:
            import torch
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        return {"image": tensor, "path": str(path)}

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        input_size: int = 128,
        split: str = "train",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        dedup: bool = True,
        dedup_threshold: int = 8,
        seed: int = 42,
        augment: bool | None = None,
        color_correction: str = "gray_world",
        verbose: bool = True,
    ) -> "UnderwaterImageDataset":
        """Build dataset from a directory of images.

        Args:
            root: Root directory containing images.
            input_size: Resize all images to this square size.
            split: "train", "val", or "test".
            dedup: Remove near-duplicate images.
            verbose: Print dataset statistics.

        Returns:
            UnderwaterImageDataset for the requested split.
        """
        root = Path(root)
        if verbose:
            print(f"Scanning {root}...")

        all_paths = find_images(root)
        if verbose:
            print(f"  Found {len(all_paths)} images")

        if not all_paths:
            raise ValueError(f"No images found in {root}. Check path and extensions.")

        if dedup:
            all_paths = deduplicate_images(all_paths, threshold=dedup_threshold, verbose=verbose)

        train_paths, val_paths, test_paths = split_paths(
            all_paths, train=train_frac, val=val_frac, test=test_frac, seed=seed
        )

        split_map = {"train": train_paths, "val": val_paths, "test": test_paths}
        if split not in split_map:
            raise ValueError(f"split must be 'train'/'val'/'test', got '{split}'")

        paths = split_map[split]

        if verbose:
            print(f"  Split '{split}': {len(paths)} images "
                  f"(train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)})")

        do_augment = augment if augment is not None else (split == "train")

        return cls(
            paths=paths,
            input_size=input_size,
            augment=do_augment,
            color_correction=color_correction,
        )


def create_synthetic_dataset(
    output_dir: str | Path,
    num_images: int = 100,
    image_size: int = 256,
    seed: int = 42,
) -> Path:
    """Create a synthetic underwater-like dataset for testing.

    Generates random images with underwater color palette (blue/green tints,
    some with simulated particles/haze). NOT a substitute for real data —
    used only for unit tests and the fail-fast experiment when no real data is available.

    Args:
        output_dir: Where to save images.
        num_images: Number of synthetic images to generate.
        image_size: Image resolution.
        seed: Random seed.

    Returns:
        Path to the created directory.
    """
    import random as std_random
    rng = np.random.default_rng(seed)
    std_rng = std_random.Random(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Random underwater scene type
        scene_type = std_rng.choice(["blue_water", "green_water", "turbid", "coral", "dark_deep"])

        h, w = image_size, image_size
        img = np.zeros((h, w, 3), dtype=np.float32)

        if scene_type == "blue_water":
            # Blue-dominant, clear water
            img[:, :, 0] = rng.uniform(0.05, 0.3, (h, w))   # R
            img[:, :, 1] = rng.uniform(0.15, 0.5, (h, w))   # G
            img[:, :, 2] = rng.uniform(0.4, 0.9, (h, w))    # B
        elif scene_type == "green_water":
            # Green-tinted (coastal / algae)
            img[:, :, 0] = rng.uniform(0.05, 0.25, (h, w))
            img[:, :, 1] = rng.uniform(0.3, 0.7, (h, w))
            img[:, :, 2] = rng.uniform(0.2, 0.5, (h, w))
        elif scene_type == "turbid":
            # Murky / low visibility
            base = rng.uniform(0.2, 0.5, (h, w))
            img[:, :, 0] = base * rng.uniform(0.8, 1.2, (h, w))
            img[:, :, 1] = base * rng.uniform(0.7, 1.0, (h, w))
            img[:, :, 2] = base * rng.uniform(0.5, 0.8, (h, w))
        elif scene_type == "coral":
            # Colorful reef: bright spots
            img[:, :, 0] = rng.uniform(0.1, 0.6, (h, w))
            img[:, :, 1] = rng.uniform(0.1, 0.7, (h, w))
            img[:, :, 2] = rng.uniform(0.1, 0.8, (h, w))
            # Add some bright spots (coral/fish)
            for _ in range(rng.integers(3, 15)):
                cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
                r = rng.integers(5, 30)
                color = rng.uniform(0.5, 1.0, 3)
                yy, xx = np.ogrid[:h, :w]
                circle = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
                img[circle] = color
        elif scene_type == "dark_deep":
            # Deep water — very dark
            img[:, :, 0] = rng.uniform(0.0, 0.1, (h, w))
            img[:, :, 1] = rng.uniform(0.0, 0.15, (h, w))
            img[:, :, 2] = rng.uniform(0.02, 0.2, (h, w))

        # Add subtle noise
        img += rng.uniform(-0.02, 0.02, img.shape)
        img = np.clip(img, 0, 1)

        arr = (img * 255).astype(np.uint8)
        Image.fromarray(arr).save(output_dir / f"synthetic_{i:04d}.jpg", quality=90)

    print(f"Created {num_images} synthetic underwater images in {output_dir}")
    return output_dir


# ── Internal helpers ──────────────────────────────────────────────────────────

def _gray_world(img: np.ndarray) -> np.ndarray:
    """Gray world white balance for underwater color correction."""
    img_f = img.astype(np.float32)
    means = img_f.mean(axis=(0, 1))
    global_mean = means.mean()
    if global_mean < 1e-6:
        return img
    scale = global_mean / (means + 1e-8)
    corrected = img_f * scale[np.newaxis, np.newaxis, :]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _histogram_equalize(img: np.ndarray) -> np.ndarray:
    """Per-channel histogram equalization."""
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
        n = channel.size
        lut = np.round(
            (cdf - cdf_min) / max(n - cdf_min, 1) * 255
        ).astype(np.uint8)
        result[:, :, c] = lut[channel]
    return result


def _augment(img: np.ndarray) -> np.ndarray:
    """Simple augmentation: horizontal flip + mild color jitter."""
    rng = np.random.default_rng()

    # Horizontal flip
    if rng.random() > 0.5:
        img = np.fliplr(img).copy()

    # Mild brightness/contrast jitter
    img_f = img.astype(np.float32)
    brightness = rng.uniform(0.85, 1.15)
    img_f = img_f * brightness
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    return img
