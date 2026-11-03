"""Offline unit tests for raw data schema detection and conversion."""

import pytest

from sentiment.data.download import convert_record, detect_schema


def test_detects_current_mteb_jsonl_schema() -> None:
    row = {"id": "en-1", "text": "review", "label": 4, "label_text": "4"}
    assert detect_schema(row.keys(), row) == {"text": "text", "stars": "label"}


def test_detects_legacy_amazon_schema() -> None:
    row = {"review_body": "review", "stars": 4}
    assert detect_schema(row.keys(), row) == {"text": "review_body", "stars": "stars"}


def test_schema_detection_fails_on_ambiguous_or_bad_rating() -> None:
    with pytest.raises(ValueError, match="uniquely detect"):
        detect_schema(
            ["text", "review_body", "stars"],
            {"text": "a", "review_body": "b", "stars": 1},
        )
    with pytest.raises(ValueError, match="rating column"):
        detect_schema(["text", "label"], {"text": "review", "label": "4"})


def test_record_conversion_produces_canonical_fields() -> None:
    result = convert_record(
        {"text": "很好", "label": 4},
        language="zh",
        split="test",
        index=3,
        schema={"text": "text", "stars": "label"},
    )
    assert result == {"sample_id": "zh-test-3", "language": "zh", "text": "很好", "stars": 4}
