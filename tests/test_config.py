"""Tests for strict run configuration loading."""

import pytest

from sentiment.config import load_config


def test_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  unknown_option: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(path)


def test_config_accepts_nested_overrides(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    config = load_config(path, {"model.max_length": 64, "train.seed": 7})
    assert config.model.max_length == 64
    assert config.train.seed == 7


def test_config_rejects_invalid_ranges(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  max_length: 513\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_length"):
        load_config(path)
