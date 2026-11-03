"""Download and normalize the raw Amazon Reviews Multi JSONL files."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from sentiment.constants import DATASET_NAME, LANGUAGES
from sentiment.io_utils import sha256_file, write_json

LOGGER = logging.getLogger(__name__)
_SPLITS = ("train", "validation", "test")
_REPO_URL = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/{revision}/{language}/{split}.jsonl?download=true"


def detect_schema(columns: Iterable[str], sample: dict[str, Any]) -> dict[str, str]:
    """Detect a unique text column and an integer 0–4 rating column."""
    available = set(columns)
    text_candidates = [name for name in ("text", "review_body") if name in available]
    if len(text_candidates) != 1:
        raise ValueError(
            "Could not uniquely detect a review text column; expected one of "
            "text/review_body, "
            f"found {text_candidates}."
        )
    if not isinstance(sample.get(text_candidates[0]), str):
        raise ValueError(f"Detected text column '{text_candidates[0]}' is not a string.")

    rating_candidates = []
    for name in ("stars", "rating", "label"):
        value = sample.get(name)
        is_integer_rating = isinstance(value, int) and not isinstance(value, bool)
        if name in available and is_integer_rating and 0 <= value <= 4:
            rating_candidates.append(name)
    if len(rating_candidates) != 1:
        raise ValueError(
            "Could not uniquely detect an integer rating column in the range 0–4; "
            f"found {rating_candidates}."
        )
    return {"text": text_candidates[0], "stars": rating_candidates[0]}


def convert_record(
    record: dict[str, Any], *, language: str, split: str, index: int, schema: dict[str, str]
) -> dict[str, Any]:
    """Map one raw source record to the canonical source JSONL schema."""
    text = record.get(schema["text"])
    stars = record.get(schema["stars"])
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{language}/{split} row {index} has missing or empty review text.")
    if isinstance(stars, bool) or not isinstance(stars, int) or stars not in range(5):
        raise ValueError(f"{language}/{split} row {index} has invalid 0–4 rating: {stars!r}.")
    return {
        "sample_id": f"{language}-{split}-{index}",
        "language": language,
        "text": text,
        "stars": stars,
    }


def _library_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest_path(output: Path) -> Path:
    return output / "download_manifest.json"


def _load_existing_manifest(output: Path) -> dict[str, Any]:
    path = _manifest_path(output)
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring unreadable download manifest %s: %s", path, exc)
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _matching_existing(path: Path, relative: str, previous: dict[str, Any]) -> bool:
    expected = previous.get("files", {}).get(relative, {}).get("sha256")
    return bool(expected and path.is_file() and sha256_file(path) == expected)


def _read_online_file(
    url: str, output_file: Path, *, language: str, split: str
) -> tuple[dict[str, str], str | None, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    row_count = 0
    schema: dict[str, str] | None = None
    revision: str | None = None
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            revision = response.headers.get("x-repo-commit")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_file.parent,
                prefix=f".{output_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid source JSON in {language}/{split} row {row_count}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Source {language}/{split} row {row_count} is not an object."
                        )
                    if schema is None:
                        schema = detect_schema(record.keys(), record)
                        LOGGER.info(
                            "Detected %s schema for %s/%s: %s",
                            DATASET_NAME,
                            language,
                            split,
                            schema,
                        )
                    canonical = convert_record(
                        record, language=language, split=split, index=row_count, schema=schema
                    )
                    handle.write(json.dumps(canonical, ensure_ascii=False, allow_nan=False) + "\n")
                    row_count += 1
                handle.flush()
                os.fsync(handle.fileno())
        if row_count == 0 or schema is None:
            raise ValueError(
                f"Source {language}/{split} was empty; refusing to write an empty data file."
            )
        os.replace(temp_name, output_file)
        temp_name = None
        return schema, revision, row_count
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def download_dataset(
    languages: Iterable[str] = LANGUAGES,
    output_dir: str | Path = "data/amazon_reviews_multi",
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Download selected language splits and write normalized files and a manifest."""
    selected = list(languages)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(LANGUAGES):
        raise ValueError(f"languages must be unique values chosen from {LANGUAGES}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous = _load_existing_manifest(output)
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    file_info: dict[str, dict[str, Any]] = {}
    split_sizes: dict[str, dict[str, int]] = {}
    schemas: dict[str, dict[str, str]] = {}
    revision: str | None = None

    for language in selected:
        split_sizes[language] = {}
        schemas[language] = {}
        for split in _SPLITS:
            relative = f"{language}/{split}.jsonl"
            destination = output / relative
            if not force and _matching_existing(destination, relative, previous):
                count = sum(1 for line in destination.open(encoding="utf-8") if line.strip())
                split_sizes[language][split] = count
                file_info[relative] = {"sha256": sha256_file(destination), "rows": count}
                schemas[language][split] = (
                    previous.get("schema", {})
                    .get(language, {})
                    .get(split, previous.get("schema", {}).get(language, {}))
                )
                LOGGER.info("Reusing verified %s", destination)
                continue
            if offline:
                raise RuntimeError(
                    f"HF_HUB_OFFLINE=1 and verified data is missing for {relative}; "
                    "disable offline mode or download the file first."
                )
            url = _REPO_URL.format(revision="main", language=language, split=split)
            schema, found_revision, count = _read_online_file(
                url, destination, language=language, split=split
            )
            revision = found_revision or revision
            split_sizes[language][split] = count
            schemas[language][split] = schema
            file_info[relative] = {"sha256": sha256_file(destination), "rows": count}
            LOGGER.info("Downloaded %s rows to %s", count, destination)

    manifest = {
        "dataset": DATASET_NAME,
        "revision": revision or previous.get("revision") or "main (commit unavailable)",
        "languages": selected,
        "split_sizes": split_sizes,
        "schema": schemas,
        "files": file_info,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "datasets": _library_version("datasets"),
            "huggingface_hub": _library_version("huggingface_hub"),
            "requests": _library_version("requests"),
        },
        "loading_method": (
            "raw repository JSONL fallback; installed datasets rejects the repository script"
        ),
    }
    write_json(_manifest_path(output), manifest)
    return manifest
