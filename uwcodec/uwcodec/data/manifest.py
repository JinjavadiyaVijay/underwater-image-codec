"""Dataset manifest generator for UWCodec."""

from __future__ import annotations

import csv
from pathlib import Path

from uwcodec.data.dataset import _image_hash, find_images


def generate_manifest(
    datasets_root: str | Path,
    output_file: str | Path,
    verbose: bool = True,
) -> None:
    """Generate a CSV manifest of all images in the datasets directory.
    
    Includes dataset name, split, resolution, format, and perceptual hash
    to detect cross-dataset duplicates/leakage.
    """
    root = Path(datasets_root)
    out_path = Path(output_file)
    
    if not root.exists():
        raise FileNotFoundError(f"Datasets root not found: {root}")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_images = find_images(root)
    
    if verbose:
        print(f"Found {len(all_images)} images. Generating manifest...")
        
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "dataset", "split", "width", "height", "format", "phash"])
        
        for i, img_path in enumerate(all_images):
            if verbose and i % 500 == 0:
                print(f"  Processed {i}/{len(all_images)}...")
                
            # Determine dataset and split from path relative to root
            try:
                rel_path = img_path.relative_to(root)
                parts = rel_path.parts
                dataset_name = parts[0] if len(parts) > 0 else "unknown"
                
                # Try to guess split
                split = "unknown"
                for s in ["train", "val", "test", "images", "references"]:
                    if s in parts:
                        split = s
                        break
            except ValueError:
                dataset_name = "unknown"
                split = "unknown"
                
            # Get image metadata
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    w, h = img.size
                    fmt = img.format or img_path.suffix[1:].upper()
            except Exception:
                w, h, fmt = 0, 0, "ERROR"
                
            # Get perceptual hash
            phash = _image_hash(img_path)
            
            # Write row
            writer.writerow([
                str(img_path.absolute()),
                dataset_name,
                split,
                w,
                h,
                fmt,
                phash
            ])
            
    if verbose:
        print(f"Manifest saved to {out_path}")


if __name__ == "__main__":
    generate_manifest("datasets", "datasets/manifest.csv")
