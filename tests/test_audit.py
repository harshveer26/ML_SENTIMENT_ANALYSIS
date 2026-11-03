"""Tests for checkpoint finiteness auditing."""

import torch

from sentiment.audit import audit_checkpoint


def test_audit_marks_corrupted_checkpoint(tmp_path) -> None:
    torch.save({"weight": torch.tensor([float("nan")])}, tmp_path / "bad.pt")
    assert audit_checkpoint(tmp_path) == 1
    assert (tmp_path / "INVALID.md").is_file()
    assert (tmp_path / "invalid_metrics.json").is_file()


def test_audit_leaves_clean_checkpoint_unmarked(tmp_path) -> None:
    torch.save({"weight": torch.tensor([1.0, 2.0])}, tmp_path / "good.pt")
    assert audit_checkpoint(tmp_path) == 0
    assert not (tmp_path / "INVALID.md").exists()
