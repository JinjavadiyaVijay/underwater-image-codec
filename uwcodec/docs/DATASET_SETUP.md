# UWCodec Dataset Setup

The UWCodec project requires three primary external datasets for training and evaluation. To prevent repository bloat and ensure clear separation of data, these datasets are stored **outside** the python package in the `datasets/` directory.

## Directory Structure
The expected structure for the datasets directory (`S:\IMG_compressors\datasets\`) is:

```text
datasets/
├── EUVP/                   # Primary training data
│   ├── train/
│   ├── val/
│   └── test/
├── SUIM/                   # Validation and generalization tests
│   ├── train/
│   ├── val/
│   └── test/
├── UIEB/                   # Independent evaluation
│   ├── images/             # Raw underwater images
│   └── references/         # (Optional) Enhanced references
└── MB1854B_test/           # Hardware-specific capture test set
```

## 1. EUVP (Enhancing Underwater Visual Perception)
**Role:** Primary training data for the codec.
**Download:** Requires filling out a Google Form from the authors. 
- [EUVP Dataset Project Page](https://irvlab.cs.umn.edu/resources/euvp-dataset)
**Setup:** 
1. Download the archive and extract it.
2. We only need the raw/distorted underwater images for the codec, not the enhanced pairs.
3. Distribute the raw images into `datasets/EUVP/train`, `datasets/EUVP/val`, and `datasets/EUVP/test`.

## 2. SUIM (Semantic Underwater Image Dataset)
**Role:** Generalization evaluation and held-out validation.
**Download:** Available from the authors' repository.
- [SUIM Dataset Project Page](https://irvlab.cs.umn.edu/resources/suim-dataset)
**Setup:**
1. Download the archive.
2. We only need the RGB images, not the semantic segmentation masks.
3. Place them into `datasets/SUIM/train`, `datasets/SUIM/val`, and `datasets/SUIM/test`.

## 3. UIEB (Underwater Image Enhancement Benchmark)
**Role:** Independent third-party evaluation. NEVER used for training.
**Download:** Requires permission/form submission from the authors.
- [UIEB Dataset Project Page](https://li-chongyi.github.io/proj_benchmark.html)
**Setup:**
1. Download the UIEB dataset.
2. Place the raw uncorrected images into `datasets/UIEB/images`.
3. (Optional) Place the reference images into `datasets/UIEB/references`.

## 4. MB1854B Test Set
**Role:** Target hardware evaluation.
**Download:** N/A (must be physically captured).
**Setup:** 
1. Capture raw RGB frames using the MB1854B camera module.
2. Place these frames in `datasets/MB1854B_test/`.
3. These will be used to evaluate the final deployment fidelity.

## Manifest Generation
After placing the images, you can generate a CSV manifest to verify the contents and check for cross-dataset near-duplicates (leakage):

```bash
python -m uwcodec.data.manifest
```
This script will produce `datasets/manifest.csv` containing paths, resolutions, and perceptual hashes of all your images.
