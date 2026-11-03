"""Dynamic batch padding for tokenized sentiment examples."""

from __future__ import annotations

from typing import Any

import torch

from sentiment.safeguards import validate_label


class DynamicPaddingCollator:
    """Pad a batch to its longest sequence and retain aligned metadata lists."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id for dynamic padding.")
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool)
            or not isinstance(pad_to_multiple_of, int)
            or pad_to_multiple_of < 1
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer or None.")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad token arrays and return tensors plus metadata in matching order."""
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if max_length < 1:
            raise ValueError("Every input sequence must contain at least one token.")
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[int] = []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            if len(ids) != len(mask):
                raise ValueError("Feature input_ids and attention_mask lengths differ.")
            padding = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_masks.append(mask + [0] * padding)
            labels.append(validate_label(feature["labels"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_ids": [feature["sample_id"] for feature in features],
            "languages": [feature["language"] for feature in features],
        }
