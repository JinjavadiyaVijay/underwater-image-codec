"""LR scheduling and warmup utilities for codec training."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_epochs.
    2. Cosine decay from base_lr to min_lr over remaining epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 200,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


class WarmupStepScheduler(_LRScheduler):
    """Step decay with linear warmup.

    Useful for fine-tuning after initial pretraining.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        step_size: int = 50,
        gamma: float = 0.5,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            effective_epoch = self.last_epoch - self.warmup_epochs
            decay = self.gamma ** (effective_epoch // self.step_size)
            return [base_lr * decay for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    warmup_epochs: int = 5,
    total_epochs: int = 200,
    **kwargs,
) -> _LRScheduler:
    """Build a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_name: "cosine" or "step".
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
    """
    if scheduler_name == "cosine":
        return WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs,
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_name == "step":
        return WarmupStepScheduler(
            optimizer, warmup_epochs=warmup_epochs,
            step_size=kwargs.get("step_size", 50),
            gamma=kwargs.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
