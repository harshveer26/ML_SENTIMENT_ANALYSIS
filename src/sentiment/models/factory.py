"""Model loading, backbone resolution, freezing, and mixed-precision policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification

from sentiment.constants import LABEL2ID, LABELS
from sentiment.safeguards import assert_finite_model

LOGGER = logging.getLogger(__name__)

SUPPORTED_BACKBONES = {
    "mdeberta": "microsoft/mdeberta-v3-base",
    "xlmr": "xlm-roberta-base",
}


@dataclass(frozen=True)
class AmpPolicy:
    """Resolved autocast and scaler options for a device."""

    mode: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool


def resolve_backbone(model_name_or_path: str | Path) -> str:
    """Resolve a short backbone name or pass through a model identifier/path."""
    name = str(model_name_or_path)
    return SUPPORTED_BACKBONES.get(name.lower(), name)


def load_classifier(
    model_name_or_path: str | Path,
    num_labels: int = 3,
    dtype: torch.dtype = torch.float32,
    *,
    local_files_only: bool = False,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load a Hugging Face sequence classifier and reject non-finite weights."""
    if num_labels != len(LABELS):
        raise ValueError(f"num_labels must be {len(LABELS)} for this benchmark.")
    resolved_name = resolve_backbone(model_name_or_path)
    label2id = dict(LABEL2ID)
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    assert_finite_model(model)
    return model


def freeze_encoder(model: torch.nn.Module) -> None:
    """Freeze the backbone while leaving the sequence-classification head trainable."""
    prefix = getattr(model, "base_model_prefix", "")
    if not prefix:
        raise ValueError(f"Cannot identify the encoder prefix for {type(model).__name__}.")
    matched_encoder = False
    for name, parameter in model.named_parameters():
        is_encoder_parameter = name.startswith(f"{prefix}.") or name == prefix
        parameter.requires_grad = not is_encoder_parameter
        matched_encoder = matched_encoder or is_encoder_parameter
    if not matched_encoder:
        raise ValueError(f"No parameters matched the model's encoder prefix '{prefix}'.")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("No classification head parameters remain trainable after freezing.")


def count_trainable(model: torch.nn.Module) -> dict[str, int]:
    """Count and log trainable, frozen, and total parameter values."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    counts = {"trainable": trainable, "frozen": total - trainable, "total": total}
    LOGGER.info(
        "Parameters: %s trainable / %s total (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        (100 * trainable / total) if total else 0.0,
    )
    return counts


def mixed_precision_policy(
    device: str | torch.device,
    amp: str = "auto",
    *,
    bf16_supported: bool | None = None,
) -> AmpPolicy:
    """Choose a safe autocast policy without casting FP32 master model weights.

    mDeBERTa can be unstable in fp16; that mode is enabled only when explicitly selected.
    The default auto policy uses bf16 when supported and otherwise stays in full precision.
    """
    device_type = torch.device(device).type
    if amp not in {"auto", "none", "bf16", "fp16"}:
        raise ValueError("amp must be one of: auto, none, bf16, fp16.")
    if device_type != "cuda" or amp == "none":
        return AmpPolicy(mode="none", autocast_dtype=None, use_grad_scaler=False)
    supported = (
        torch.cuda.is_bf16_supported()
        if bf16_supported is None and torch.cuda.is_available()
        else bool(bf16_supported)
    )
    if amp == "auto":
        if supported:
            return AmpPolicy(mode="bf16", autocast_dtype=torch.bfloat16, use_grad_scaler=False)
        return AmpPolicy(mode="none", autocast_dtype=None, use_grad_scaler=False)
    if amp == "bf16":
        if not supported:
            raise ValueError(
                "bf16 was requested, but the selected CUDA device does not support it."
            )
        return AmpPolicy(mode="bf16", autocast_dtype=torch.bfloat16, use_grad_scaler=False)
    return AmpPolicy(mode="fp16", autocast_dtype=torch.float16, use_grad_scaler=True)
