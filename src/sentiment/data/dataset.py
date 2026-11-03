"""Lazy tokenization and deterministic PyTorch data loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset

from sentiment.constants import LANGUAGES
from sentiment.safeguards import validate_record
from sentiment.seed import make_generator, seed_worker


def normalize_language(language: str, *, allow_und: bool = False) -> str:
    """Normalize language tags such as FR or fr-FR to a supported base code."""
    if not isinstance(language, str) or not language.strip():
        if allow_und:
            return "und"
        raise ValueError("language must be a non-empty language code.")
    normalized = language.strip().lower().replace("_", "-").split("-", maxsplit=1)[0]
    if normalized in LANGUAGES:
        return normalized
    if allow_und:
        return "und"
    raise ValueError(f"Unsupported language {language!r}; choose one of {LANGUAGES}.")


def format_text(text: str, language: str, use_lang_tag: bool, *, allow_und: bool = False) -> str:
    """Optionally prefix a review with the normalized `[LANG=xx]` language tag."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")
    normalized = normalize_language(language, allow_und=allow_und)
    return f"[LANG={normalized}] {text}" if use_lang_tag else text


class SentimentDataset(Dataset[dict[str, Any]]):
    """A benchmark dataset that tokenizes each review only when it is requested."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
        use_lang_tag: bool,
    ) -> None:
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 1 <= max_length <= 512
        ):
            raise ValueError("max_length must be an integer between 1 and 512.")
        self.records = [dict(record) for record in records]
        for record in self.records:
            validate_record(record)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_lang_tag = use_lang_tag

    def __len__(self) -> int:
        """Return the number of stored benchmark records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Tokenize one record without padding and retain evaluation metadata."""
        record = self.records[index]
        text = format_text(record["text"], record["language"], self.use_lang_tag)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        if len(input_ids) > self.max_length:
            raise RuntimeError(
                f"Tokenizer returned {len(input_ids)} tokens, "
                f"exceeding max_length={self.max_length}."
            )
        if len(attention_mask) != len(input_ids):
            raise RuntimeError(
                "Tokenizer returned mismatched input_ids and attention_mask lengths."
            )
        return {
            "input_ids": list(input_ids),
            "attention_mask": list(attention_mask),
            "labels": int(record["label"]),
            "sample_id": record["sample_id"],
            "language": normalize_language(record["language"]),
        }


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    collate_fn: Any = None,
) -> DataLoader:
    """Build a DataLoader with a seeded sampler and deterministic worker seeds."""
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
