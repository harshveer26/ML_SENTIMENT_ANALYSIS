"""Tests for sentiment dataset, language tags, and dynamic batch padding."""

import pytest

from sentiment.data.collate import DynamicPaddingCollator
from sentiment.data.dataset import (
    SentimentDataset,
    format_text,
    make_dataloader,
    normalize_language,
)


def _records() -> list[dict[str, object]]:
    return [
        {"sample_id": "en-0", "language": "en", "text": "good movie", "label": 2},
        {"sample_id": "fr-0", "language": "fr", "text": "bon film nice", "label": 1},
        {"sample_id": "en-1", "language": "en", "text": "bad", "label": 0},
        {"sample_id": "en-2", "language": "en", "text": "very long text review", "label": 2},
    ]


def test_language_normalization_and_tag_modes() -> None:
    assert normalize_language("FR-FR") == "fr"
    assert normalize_language("ZH") == "zh"
    assert format_text("review", "fr-FR", True) == "[LANG=fr] review"
    assert format_text("review", "FR", False) == "review"
    assert format_text("review", "xx", True, allow_und=True) == "[LANG=und] review"
    with pytest.raises(ValueError, match="Unsupported language"):
        format_text("review", "xx", True)


def test_dataset_truncation_and_dynamic_padding(tiny_tokenizer) -> None:
    dataset = SentimentDataset(_records(), tiny_tokenizer, max_length=4, use_lang_tag=True)
    items = [dataset[0], dataset[1]]
    assert all(len(item["input_ids"]) <= 4 for item in items)
    assert "[LANG=en]" in tiny_tokenizer.convert_ids_to_tokens(items[0]["input_ids"])
    batch = DynamicPaddingCollator(tiny_tokenizer)(items)
    assert batch["input_ids"].shape == (2, max(len(item["input_ids"]) for item in items))
    assert batch["sample_ids"] == ["en-0", "fr-0"]
    assert batch["languages"] == ["en", "fr"]
    assert batch["labels"].tolist() == [2, 1]


def test_pad_to_multiple_of_and_invalid_lengths(tiny_tokenizer) -> None:
    dataset = SentimentDataset(_records(), tiny_tokenizer, max_length=8, use_lang_tag=False)
    batch = DynamicPaddingCollator(tiny_tokenizer, pad_to_multiple_of=8)([dataset[0], dataset[2]])
    assert batch["input_ids"].shape[1] == 8
    with pytest.raises(ValueError, match="pad_token_id"):
        DynamicPaddingCollator(type("TokenizerWithoutPad", (), {"pad_token_id": None})())
    with pytest.raises(ValueError, match="max_length"):
        SentimentDataset(_records(), tiny_tokenizer, max_length=513, use_lang_tag=False)


def test_dataloader_seed_order_and_metadata_alignment(tiny_tokenizer) -> None:
    dataset = SentimentDataset(_records(), tiny_tokenizer, max_length=8, use_lang_tag=False)
    collator = DynamicPaddingCollator(tiny_tokenizer)

    def order(seed: int) -> tuple[list[str], list[int]]:
        sample_ids = []
        labels = []
        loader = make_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=seed,
            collate_fn=collator,
        )
        for batch in loader:
            sample_ids.extend(batch["sample_ids"])
            labels.extend(batch["labels"].tolist())
        return sample_ids, labels

    first_ids, first_labels = order(21)
    assert order(21) == (first_ids, first_labels)
    assert order(22)[0] != first_ids
    expected_labels = {record["sample_id"]: record["label"] for record in _records()}
    assert [expected_labels[sample_id] for sample_id in first_ids] == first_labels
