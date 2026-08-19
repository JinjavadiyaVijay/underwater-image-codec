# UWCodec: Enable GPU Training on Windows

## Problem
PyTorch ships a CPU-only build (`torch 2.x+cpu`) by default from PyPI when running Python 3.13 on Windows.
The CUDA-enabled wheels are not yet published for Python 3.13.

## Solution: Downgrade to Python 3.12

The RTX 3050 is fully supported. To use it, you need Python 3.12 and a CUDA wheel:

### Option A: Create a conda environment with Python 3.12

```bash
conda create -n uwcodec_gpu python=3.12 -y
conda activate uwcodec_gpu

# Install PyTorch with CUDA 12.1 (for your CUDA driver 13.3 which is backward-compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Option B: Use conda to install torch-cuda directly

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### After setting up, run full training with GPU

```bash
# Install project dependencies
cd s:\IMG_compressors\uwcodec
pip install -e .

# Verify CUDA smoke test
python -m uwcodec.training.train_codec --synthetic --epochs 1 --device cuda --num-images 100

# Run full multi-budget training
python scripts/train_multi_budget.py \
  --dataset euvp \
  --datasets-root "s:\IMG_compressors\datasets" \
  --device cuda \
  --num-images 2000 \
  --epochs 50
```

### Expected GPU Performance (RTX 3050, 6GB VRAM)
- Per-epoch time: ~30-60s (vs ~300s on CPU)
- Full 50-epoch training: ~30-60 min per budget
- Total (64B + 96B + 124B): ~2-3 hours

## Current Status
Currently running on CPU (Python 3.13 environment). CPU training at 50 epochs × 2000 images
takes ~4-5 hours per budget. The model architecture and code are GPU-ready — all tensors and
the GradScaler are configured for CUDA. Just switch the environment.
