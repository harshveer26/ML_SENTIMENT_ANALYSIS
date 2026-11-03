"""CLI and run-comparison smoke tests using local tiny fixtures."""

import json
import subprocess
import sys

import pytest

from sentiment.evaluation.compare import compare_runs
from tests.fixtures import tiny_model_and_tokenizer


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sentiment", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_commands_and_examples() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    for command in ("probe", "train", "evaluate", "predict", "compare"):
        assert command in result.stdout
    subcommand = _run_cli("predict", "--help")
    assert subcommand.returncode == 0
    assert "Example:" in subcommand.stdout


def test_cli_predict_uses_local_tiny_checkpoint(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "best_model", tiny_tokenizer)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # The checkpoint directory is also the run's best_model directory.
    model_path.rename(run_dir / "best_model")
    result = _run_cli("predict", "--run", str(run_dir), "--text", "good movie", "--lang", "en")
    assert result.returncode == 0, result.stderr
    assert '"probabilities"' in result.stdout


def test_cli_test_evaluation_enforces_selection_lock(tmp_path) -> None:
    run_dir = tmp_path / "unselected"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"kind": "full", "selection_complete": False}), encoding="utf-8"
    )
    result = _run_cli("evaluate", "--run", str(run_dir), "--split", "test")
    assert result.returncode == 2
    assert "locked until checkpoint selection is complete" in result.stderr


def test_compare_marks_pending_and_refuses_manifest_mismatch(tmp_path) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "run_status.json").write_text(json.dumps({"kind": "probe"}), encoding="utf-8")
    (pending / "dataset_manifest_ref.json").write_text(
        json.dumps({"sha256": "abc"}), encoding="utf-8"
    )
    result = compare_runs([pending], tmp_path / "reports")
    assert result["runs"][0]["status"] == "PENDING"
    assert "PENDING" in (tmp_path / "reports" / "comparison.csv").read_text()

    other = tmp_path / "other"
    other.mkdir()
    (other / "dataset_manifest_ref.json").write_text(
        json.dumps({"sha256": "different"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different dataset manifest hashes"):
        compare_runs([pending, other], tmp_path / "bad-report")
