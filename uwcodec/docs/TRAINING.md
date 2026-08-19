# UWCodec Training Guide

This document explains how to train UWCodec models and tune hyperparameters for different deployment constraints.

## Prerequisites

1. **Download datasets** — see `docs/DATASET_SETUP.md` for instructions.
2. **Install dependencies** — `pip install -e ".[dev]"` from the `uwcodec/` directory.
3. **Verify setup** — run the oracle first to confirm the byte budgets carry useful information.

## Step-by-Step Training Workflow

### Step 1: Verify Oracle Baseline

Before investing in training, confirm the information-theoretic limits:

```bash
# Run non-learned oracle on your validation set
python scripts/run_oracle_real.py --datasets-root ../datasets --dataset euvp
```

This will show PSNR ~26-28 dB for all non-learned strategies at 64-124B. This is the baseline that a trained model must beat.

### Step 2: Quick Smoke Test (Synthetic Data)

Verify the training pipeline works end-to-end before using real data:

```bash
# 2-epoch sanity check on 50 synthetic images
python -m uwcodec.training.train_codec --synthetic --epochs 2 --num-images 50

# Expected output: loss decreasing, perplexity > 20, no errors
```

### Step 3: Train on EUVP (Primary Training)

```bash
# Train at 124B budget for 50 epochs on EUVP
python -m uwcodec.training.train_codec \
    --dataset euvp \
    --datasets-root ../datasets \
    --train-budget 124 \
    --epochs 50 \
    --batch-size 16 \
    --output-dir outputs/euvp_124b

# On CPU: ~2-4 hours depending on dataset size
# On GPU: ~15-30 minutes
```

**Loss targets** (rough guides, not hard rules):
- After epoch 5: pixel loss < 0.12
- After epoch 20: pixel loss < 0.09
- After epoch 50: pixel loss < 0.07

### Step 4: Multi-Budget Training (Optional)

Train separate models for each byte budget:

```bash
python scripts/train_multi_budget.py \
    --dataset euvp \
    --datasets-root ../datasets \
    --budgets 64 96 124 \
    --epochs 50
```

Checkpoints are saved to `outputs/multi_budget/budget_64/`, `budget_96/`, `budget_124/`.

### Step 5: Evaluate Against Baselines

```bash
python scripts/run_baselines.py \
    --dataset euvp \
    --datasets-root ../datasets \
    --model outputs/euvp_124b/best.pt
```

### Step 6: Generalization Test (Held-Out)

```bash
# Test on SUIM (never seen during training)
python scripts/run_baselines.py \
    --dataset suim \
    --datasets-root ../datasets \
    --model outputs/euvp_124b/best.pt

# Test on UIEB
python scripts/run_baselines.py \
    --dataset uieb \
    --datasets-root ../datasets \
    --model outputs/euvp_124b/best.pt
```

## Hyperparameter Reference

| Argument | Default | Notes |
|---|---|---|
| `--train-budget` | 124 | Byte budget the model trains for |
| `--input-size` | 128 | Encoder input resolution |
| `--hidden-channels` | 32 | Base channel width (32→64→128→256) |
| `--epochs` | 10 | For CPU: use 50; for GPU: use 100+ |
| `--batch-size` | 16 | Reduce to 8 if OOM on CPU |
| `--lr` | 3e-4 | Learning rate |
| `--lambda-pixel` | 1.0 | Weight for pixel-level L1 loss |
| `--lambda-perceptual` | 0.5 | Weight for gradient-based perceptual loss |
| `--lambda-vq` | 1.0 | Weight for VQ commitment loss |
| `--early-stop-patience` | 15 | Epochs to wait before early stopping |

## Loss Components

The total training loss is:

```
loss = lambda_pixel * L1(recon, target)
     + lambda_perceptual * GradientLoss(recon, target)
     + lambda_vq * VQ_commitment_loss
```

**`L1 pixel loss`**: Drives reconstruction fidelity. Should dominate.  
**`Gradient loss`**: Penalizes blurry reconstructions by comparing edge magnitudes. Avoids VGG dependency.  
**`VQ commitment loss`**: Forces encoder output to stay close to codebook entries. Monitors with `perplexity` — higher is better (means more codebook entries are used).

## Monitoring Training

Key metrics to watch:
- **pixel loss**: Main signal, should decrease monotonically.
- **vq loss**: Should decrease then stabilize.
- **perplexity**: Should be > 50 (out of 256) for good codebook utilization. If < 20, codebook is collapsing.
- **val pixel loss**: Must track train loss — if diverging, reduce learning rate or add regularization.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Loss not decreasing | LR too high or too low | Try `--lr 1e-4` or `--lr 1e-3` |
| Perplexity < 10 | Codebook collapse | Increase `--lambda-vq` to 2.0 |
| OOM error on CPU | Batch too large | Reduce `--batch-size` to 8 or 4 |
| PSNR lower than oracle | Model underfitting | More epochs, more data, larger model |
| Training loss << val loss | Overfitting | Use full EUVP dataset, add augmentation |
