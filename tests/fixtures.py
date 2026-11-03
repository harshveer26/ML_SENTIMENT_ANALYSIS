"""Small randomly initialized model fixtures; these never download or train real weights."""

from pathlib import Path
from typing import Any

from transformers import DebertaV2Config, DebertaV2ForSequenceClassification


def tiny_model_and_tokenizer(directory: str | Path, tokenizer: Any) -> tuple[Path, Any]:
    """Save a two-layer, 32-hidden-unit random DeBERTa model and local tokenizer."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )
    model = DebertaV2ForSequenceClassification(config)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    return target, tokenizer
