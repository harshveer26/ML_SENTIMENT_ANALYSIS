"""Tests for atomic JSON output and stable content hashes."""

from sentiment.io_utils import read_json, sha256_file, sha256_text, write_json


def test_atomic_json_write_and_hash_are_stable(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"
    payload = {"text": "café", "value": 3}
    write_json(path, payload)
    first_hash = sha256_file(path)
    assert read_json(path) == payload

    write_json(path, payload)
    assert sha256_file(path) == first_hash
    assert sha256_text("same") == sha256_text("same")
