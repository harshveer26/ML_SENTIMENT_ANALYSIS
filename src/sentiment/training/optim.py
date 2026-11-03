"""Optimizer, scheduler, loss-weight, and scaler helpers for fine-tuning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sentiment.models.factory import AmpPolicy


def build_adamw(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW with no decay on bias and normalization parameters."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    norm_names = ("layernorm", "layer_norm", "layer_norm", ".norm.")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        if name.endswith("bias") or any(token in lower_name for token in norm_names):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable model parameters were found for AdamW.")
    return AdamW(groups, lr=learning_rate)


def build_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    """Create a linear warmup then linear decay learning-rate scheduler."""
    if total_steps < 1 or not 0 <= warmup_ratio <= 1:
        raise ValueError("total_steps must be positive and warmup_ratio must be in [0, 1].")
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(0, total_steps - step)
        decay_steps = max(1, total_steps - warmup_steps)
        return float(remaining) / float(decay_steps)

    return LambdaLR(optimizer, schedule)


def balanced_class_weights(labels: Iterable[int], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency balanced weights using only supplied training labels."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    values = np.asarray(list(labels))
    if values.ndim != 1 or not len(values):
        raise ValueError("labels must be a non-empty one-dimensional sequence.")
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError("labels must contain integer class ids.")
    if not np.isin(values, np.arange(num_classes)).all():
        raise ValueError(f"labels must be in the range 0–{num_classes - 1}.")
    counts = Counter(values.astype(int).tolist())
    missing = [label for label in range(num_classes) if counts[label] == 0]
    if missing:
        raise ValueError(f"Cannot balance training labels; missing class id(s): {missing}.")
    total = len(values)
    weights = [total / (num_classes * counts[label]) for label in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def make_grad_scaler(policy: AmpPolicy) -> torch.amp.GradScaler:
    """Create a CUDA GradScaler only for the explicitly selected fp16 policy."""
    try:
        return torch.amp.GradScaler("cuda", enabled=policy.use_grad_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=policy.use_grad_scaler)
