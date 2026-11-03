"""Deterministically sample and de-duplicate the multilingual benchmark."""

from __future__ import annotations

import json
import logging
import random
import zlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentiment.constants import (
    DATASET_NAME,
    DEFAULT_SEED_BENCHMARK,
    ID2LABEL,
    LANGUAGES,
    SPLIT_SIZES,
    STAR_TO_LABEL,
)
from sentiment.env_info import capture_env
from sentiment.io_utils import sha256_file, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def hash_salt(language: str, split: str) -> int:
    """Return a stable unsigned CRC32 salt for a language and split."""
    return zlib.crc32(f"{language}:{split}".encode()) & 0xFFFFFFFF


def _read_source(path: Path, language: str, split: str) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path} line {line_number}.")
            if record.get("language") != language or not isinstance(record.get("text"), str):
                raise ValueError(f"Bad language/text fields in {path} line {line_number}.")
            stars = record.get("stars")
            if isinstance(stars, bool) or not isinstance(stars, int) or stars not in STAR_TO_LABEL:
                raise ValueError(f"Bad 0–4 stars value in {path} line {line_number}: {stars!r}.")
            if not record["text"].strip():
                raise ValueError(f"Empty review text in {path} line {line_number}.")
            if not isinstance(record.get("sample_id"), str) or not record["sample_id"].strip():
                raise ValueError(f"Missing sample_id in {path} line {line_number}.")
            yield record


def _reservoir_sample(
    candidates: Iterable[dict[str, Any]], size: int, rng: random.Random, excluded_texts: set[str]
) -> tuple[list[dict[str, Any]], int]:
    reservoir: list[dict[str, Any]] = []
    eligible_count = 0
    for record in candidates:
        if record["text"].strip() in excluded_texts:
            continue
        eligible_count += 1
        if len(reservoir) < size:
            reservoir.append(record)
        else:
            index = rng.randrange(eligible_count)
            if index < size:
                reservoir[index] = record
    if len(reservoir) < size:
        raise ValueError(
            f"Could sample only {len(reservoir)} of {size} requested records after split-overlap "
            f"exclusion; {eligible_count} eligible candidates were available."
        )
    return reservoir, eligible_count


def _load_source_manifest(source_dir: Path) -> dict[str, Any]:
    path = source_dir / "download_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Source manifest must be a JSON object: {path}")
    return manifest


def _check_source_hash(path: Path, source_manifest: dict[str, Any]) -> None:
    relative = path.relative_to(path.parents[1]).as_posix()
    entry = source_manifest.get("files", {}).get(relative)
    if not entry:
        raise ValueError(f"Source manifest has no hash entry for {relative}.")
    actual = sha256_file(path)
    if actual != entry.get("sha256"):
        raise ValueError(
            f"Source hash mismatch for {path}; rerun download or restore the verified file."
        )


def prepare_benchmark(
    source_dir: str | Path = "data/amazon_reviews_multi",
    output_dir: str | Path = "data/amazon_reviews_benchmark",
    *,
    languages: Iterable[str] = LANGUAGES,
    seed: int = DEFAULT_SEED_BENCHMARK,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Create train/validation/test samples with protected split precedence and a manifest."""
    selected_languages = list(languages)
    sizes = dict(SPLIT_SIZES if split_sizes is None else split_sizes)
    if not selected_languages or len(set(selected_languages)) != len(selected_languages):
        raise ValueError("languages must be non-empty and unique.")
    if not set(selected_languages).issubset(LANGUAGES):
        raise ValueError(f"languages must be chosen from {LANGUAGES}.")
    if set(sizes) != set(SPLIT_SIZES) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes.values()
    ):
        raise ValueError(
            f"split_sizes must contain positive integer sizes for {sorted(SPLIT_SIZES)}."
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    source = Path(source_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Source data directory not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = _load_source_manifest(source)
    sampled: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_SIZES}
    protected_texts: set[str] = set()
    salts: dict[str, dict[str, int]] = {language: {} for language in selected_languages}
    eligible_counts: dict[str, dict[str, int]] = {language: {} for language in selected_languages}

    for split in ("test", "validation", "train"):
        split_texts: set[str] = set()
        for language in selected_languages:
            relative = Path(language) / f"{split}.jsonl"
            source_path = source / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source split: {source_path}")
            if source_manifest:
                _check_source_hash(source_path, source_manifest)
            salt = hash_salt(language, split)
            salts[language][split] = salt
            rng = random.Random(seed ^ salt)
            chosen, eligible_count = _reservoir_sample(
                _read_source(source_path, language, split), sizes[split], rng, protected_texts
            )
            eligible_counts[language][split] = eligible_count
            for record in chosen:
                stars = record["stars"]
                sampled[split].append(
                    {
                        "sample_id": record["sample_id"],
                        "language": language,
                        "text": record["text"],
                        "label": STAR_TO_LABEL[stars],
                        "stars": stars,
                    }
                )
                split_texts.add(record["text"].strip())
        protected_texts.update(split_texts)

    class_counts: dict[str, dict[str, dict[str, int]]] = {}
    split_counts: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for split, records in sampled.items():
        shuffle_seed = seed ^ hash_salt("all", split)
        random.Random(shuffle_seed).shuffle(records)
        ids = [record["sample_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Source sample_id values are not unique within {split}.")
        path = destination / f"{split}.jsonl"
        write_jsonl(path, records)
        split_counts[split] = len(records)
        file_hashes[path.name] = sha256_file(path)
        class_counts[split] = {}
        for language in selected_languages:
            counts = Counter(
                record["label"] for record in records if record["language"] == language
            )
            class_counts[split][language] = {
                str(label): counts.get(label, 0) for label in range(len(ID2LABEL))
            }

    train_label_counts = Counter(record["label"] for record in sampled["train"])
    majority_label = min(range(3), key=lambda label: (-train_label_counts[label], label))
    test_accuracy = sum(record["label"] == majority_label for record in sampled["test"]) / len(
        sampled["test"]
    )
    manifest = {
        "dataset": source_manifest.get("dataset", DATASET_NAME),
        "dataset_revision": source_manifest.get("revision", "unavailable"),
        "seed": seed,
        "languages": selected_languages,
        "split_sizes_per_language": sizes,
        "split_counts": split_counts,
        "sampling_salts_crc32": salts,
        "eligible_candidate_counts": eligible_counts,
        "label_mapping": {str(stars): label for stars, label in STAR_TO_LABEL.items()},
        "label_names": {str(label): name for label, name in ID2LABEL.items()},
        "class_counts": class_counts,
        "files": file_hashes,
        "sampling_algorithm": (
            "Reservoir sampling per language/split; exact-text strip() duplicates across splits "
            "excluded in priority order test, validation, train; output shuffled by seeded RNG."
        ),
        "majority_baseline": {
            "training_majority_label": majority_label,
            "training_majority_name": ID2LABEL[majority_label],
            "test_accuracy": test_accuracy,
        },
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": capture_env(),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest
