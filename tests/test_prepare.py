"""Tests for deterministic reservoir sampling and benchmark validation."""

import subprocess
import sys

import pytest

from sentiment.constants import STAR_TO_LABEL
from sentiment.data.prepare import hash_salt, prepare_benchmark
from sentiment.data.validate import validate_benchmark
from sentiment.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from sentiment.safeguards import SplitOverlapError


def _make_source(root, *, duplicate_across_splits: bool = False) -> None:
    for split in ("train", "validation", "test"):
        records = []
        for index in range(30):
            text = f"{split} review {index}"
            if duplicate_across_splits and split == "validation" and index == 0:
                text = "test review 0"
            records.append(
                {
                    "sample_id": f"en-{split}-{index}",
                    "language": "en",
                    "text": text,
                    "stars": index % 5,
                }
            )
        write_jsonl(root / "en" / f"{split}.jsonl", records)


def _prepare(source, destination, seed=17):
    return prepare_benchmark(
        source,
        destination,
        languages=["en"],
        seed=seed,
        split_sizes={"train": 5, "validation": 4, "test": 3},
    )


def test_same_seed_has_byte_identical_splits_and_exact_counts(tmp_path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = _prepare(source, first)
    _prepare(source, second)
    for split in ("train", "validation", "test"):
        assert sha256_file(first / f"{split}.jsonl") == sha256_file(second / f"{split}.jsonl")
        records = read_jsonl(first / f"{split}.jsonl")
        assert len(records) == manifest["split_sizes_per_language"][split]
    assert manifest["split_counts"] == {"train": 5, "validation": 4, "test": 3}
    assert validate_benchmark(first)["split_counts"] == manifest["split_counts"]


def test_different_seed_changes_sample(tmp_path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare(source, first, seed=17)
    _prepare(source, second, seed=18)
    assert sha256_file(first / "train.jsonl") != sha256_file(second / "train.jsonl")


def test_split_overlap_is_reported(tmp_path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    output = tmp_path / "benchmark"
    _prepare(source, output)
    train = read_jsonl(output / "train.jsonl")
    validation = read_jsonl(output / "validation.jsonl")
    validation[0]["text"] = train[0]["text"]
    write_jsonl(output / "validation.jsonl", validation)
    manifest = read_json(output / "manifest.json")
    manifest["files"]["validation.jsonl"] = sha256_file(output / "validation.jsonl")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(SplitOverlapError, match="overlap"):
        validate_benchmark(output)


def test_star_mapping_for_all_values(tmp_path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    output = tmp_path / "benchmark"
    prepare_benchmark(
        source,
        output,
        languages=["en"],
        seed=17,
        split_sizes={"train": 10, "validation": 10, "test": 10},
    )
    records = []
    for split in ("train", "validation", "test"):
        records.extend(read_jsonl(output / f"{split}.jsonl"))
    assert all(record["label"] == STAR_TO_LABEL[record["stars"]] for record in records)
    assert set(STAR_TO_LABEL.values()) == {0, 1, 2}


def test_salt_is_stable_across_processes() -> None:
    code = "from sentiment.data.prepare import hash_salt; print(hash_salt('zh','validation'))"
    first = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert first.stdout == second.stdout
    assert int(first.stdout.strip()) == hash_salt("zh", "validation")
