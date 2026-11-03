"""Validate benchmark rows, manifest counts/hashes, and split isolation."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from sentiment.constants import ID2LABEL, LANGUAGES, SPLIT_SIZES
from sentiment.io_utils import read_json, read_jsonl, sha256_file
from sentiment.safeguards import SplitOverlapError, validate_record

LOGGER = logging.getLogger(__name__)


def validate_benchmark(directory: str | Path = "data/amazon_reviews_benchmark") -> dict[str, Any]:
    """Validate benchmark contents and return a compact report, raising on any mismatch."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    languages = manifest.get("languages")
    if not isinstance(languages, list) or not languages or not set(languages).issubset(LANGUAGES):
        raise ValueError("Benchmark manifest has missing or invalid language codes.")
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    normalized_texts: dict[str, dict[str, str]] = {}
    report: dict[str, Any] = {"split_counts": {}, "class_counts": {}}

    for split in SPLIT_SIZES:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark split: {path}")
        expected_hash = manifest.get("files", {}).get(path.name)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Manifest SHA-256 mismatch for {path}.")
        records = read_jsonl(path)
        for record in records:
            validate_record(record)
            sample_id = record["sample_id"]
            if sample_id in all_ids:
                raise ValueError(f"Duplicate sample_id across benchmark: {sample_id}")
            all_ids.add(sample_id)
            normalized = record["text"].strip()
            normalized_texts.setdefault(split, {})[normalized] = sample_id
        expected_count = manifest.get("split_counts", {}).get(split)
        if len(records) != expected_count:
            raise ValueError(
                f"{split} count is {len(records)}, but manifest records {expected_count}."
            )
        configured_size = manifest.get("split_sizes_per_language", {}).get(split)
        if configured_size is None:
            raise ValueError(f"Manifest is missing the configured size for split {split}.")
        expected_total = configured_size * len(languages)
        if len(records) != expected_total:
            raise ValueError(
                f"{split} has {len(records)} records; expected {expected_total} "
                f"({configured_size} per language across {len(languages)} languages)."
            )
        records_by_split[split] = records
        report["split_counts"][split] = len(records)

    overlap_examples: list[str] = []
    overlap_count = 0
    split_names = list(SPLIT_SIZES)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            common = normalized_texts[left].keys() & normalized_texts[right].keys()
            overlap_count += len(common)
            overlap_examples.extend(
                f"{left}/{normalized_texts[left][text]} == {right}/{normalized_texts[right][text]}"
                for text in list(common)[:5]
            )
    if overlap_count:
        shown = "; ".join(overlap_examples[:10])
        raise SplitOverlapError(
            f"Found {overlap_count} exact-text overlap(s) across splits. Examples: {shown}"
        )

    for split, records in records_by_split.items():
        configured_size = manifest.get("split_sizes_per_language", {}).get(split)
        counts_by_language: dict[str, dict[str, int]] = {}
        for language in languages:
            counts = Counter(
                record["label"] for record in records if record["language"] == language
            )
            if sum(counts.values()) != configured_size:
                raise ValueError(
                    f"{split}/{language} has {sum(counts.values())} records; "
                    f"expected {configured_size}."
                )
            counts_by_language[language] = {
                str(label): counts.get(label, 0) for label in range(len(ID2LABEL))
            }
            total = sum(counts.values())
            neutral_share = counts.get(1, 0) / total if total else 0
            if neutral_share < 0.1:
                LOGGER.warning(
                    "Neutral share is below 10%% for %s/%s (%.1f%%).",
                    language,
                    split,
                    neutral_share * 100,
                )
        report["class_counts"][split] = counts_by_language
        expected_classes = manifest.get("class_counts", {}).get(split)
        if counts_by_language != expected_classes:
            raise ValueError(f"Class counts for {split} do not match the manifest.")

    table_lines = ["split language Negative Neutral Positive"]
    for split, languages_counts in report["class_counts"].items():
        for language, counts in languages_counts.items():
            table_lines.append(
                f"{split:10} {language:8} {counts['0']:8} {counts['1']:7} {counts['2']:8}"
            )
    LOGGER.info("Benchmark class balance:\n%s", "\n".join(table_lines))
    return report
