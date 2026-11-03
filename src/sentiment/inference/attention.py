"""Diagnostic CLS attention export; attention is not a causal explanation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sentiment.data.dataset import format_text
from sentiment.safeguards import assert_finite_array

if TYPE_CHECKING:
    from sentiment.inference.predictor import Predictor

DISCLAIMER = "Attention weights are a diagnostic, not a faithful causal explanation."


def explain_attention(
    predictor: Predictor,
    text: str,
    language: str | None = None,
) -> dict[str, Any]:
    """Return tokens and normalized final-layer head-averaged attention from CLS."""
    if predictor.kind not in {"full", "lora", "probe"}:
        raise ValueError("Attention export is unavailable for a majority classifier.")
    formatted = format_text(
        text,
        "und" if language is None else language,
        predictor.use_lang_tag,
        allow_und=True,
    )
    encoded = predictor.tokenizer(
        formatted,
        truncation=True,
        max_length=predictor.max_length,
        padding=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(predictor.device) for key, value in encoded.items()}
    model = predictor.encoder if predictor.kind == "probe" else predictor.model
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, return_dict=True)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "This model or attention backend did not return attention weights; "
            "use a backend that supports output_attentions=True."
        ) from exc
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "Attention weights were not returned by the model/backend; "
            "enable output_attentions or use an eager attention implementation."
        )
    last_layer = attentions[-1]
    cls_attention = last_layer[0, :, 0, :].mean(dim=0)
    mask = encoded.get("attention_mask")
    valid_length = int(mask[0].sum().item()) if mask is not None else int(cls_attention.shape[0])
    cls_attention = cls_attention[:valid_length].float()
    assert_finite_array(cls_attention.detach().cpu().numpy(), "attention weights")
    total = cls_attention.sum()
    if not torch.isfinite(total).item() or total.item() <= 0:
        raise ValueError("CLS attention has a non-positive or non-finite normalization sum.")
    normalized = cls_attention / total
    token_ids = encoded["input_ids"][0, :valid_length].detach().cpu().tolist()
    tokens = predictor.tokenizer.convert_ids_to_tokens(token_ids)
    values = normalized.detach().cpu().tolist()
    return {
        "tokens": tokens,
        "attention": [float(value) for value in values],
        "disclaimer": DISCLAIMER,
    }
