"""Tests for metrics, finite output checks, and test-set locking."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sentiment.evaluation.evaluate import evaluate_model
from sentiment.evaluation.metrics import compute_metrics
from sentiment.safeguards import NonFiniteError, TestSetLockedError


def _batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[0], [1], [2], [3]]),
        "attention_mask": torch.ones((4, 1), dtype=torch.long),
        "labels": torch.tensor([0, 1, 2, 2]),
        "languages": ["en", "en", "fr", "fr"],
        "sample_ids": ["a", "b", "c", "d"],
    }


class TableModel(torch.nn.Module):
    """Return fixed logits indexed by the first input token."""

    def __init__(self, *, non_finite: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.non_finite = non_finite
        self.calls = 0

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask
        self.calls += 1
        if self.non_finite:
            logits = torch.full((len(labels), 3), float("nan"), device=input_ids.device)
        else:
            logits = torch.tensor(
                [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
                device=input_ids.device,
            )
            logits = logits.index_select(0, input_ids[:, 0])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


class PoisonLoader:
    def __iter__(self):
        raise AssertionError("locked test data loader was touched")


def test_hand_computed_metrics_and_neutral_never_predicted() -> None:
    result = compute_metrics([0, 1, 2, 2], [0, 0, 2, 2], ["en", "en", "fr", "fr"])
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx(5 / 9)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    assert result["neutral_recall"] == 0.0
    assert result["neutral_predicted_count"] == 0
    assert result["neutral_never_predicted"] is True
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 2]]


def test_per_language_metrics_on_two_languages() -> None:
    result = compute_metrics([0, 2, 1, 2], [0, 0, 2, 2], ["en", "en", "fr", "fr"])
    assert result["per_language"]["en"]["accuracy"] == 0.5
    assert result["per_language"]["fr"]["accuracy"] == 0.5
    assert result["per_language"]["fr"]["support"] == 2


def test_metrics_reject_non_integer_labels() -> None:
    with pytest.raises(ValueError, match="integer"):
        compute_metrics([0.5, 1.0], [0, 1], ["en", "fr"])


def test_non_finite_logits_write_reason_only(tmp_path) -> None:
    output = tmp_path / "eval_validation"
    with pytest.raises(NonFiniteError, match="non-finite"):
        evaluate_model(
            TableModel(non_finite=True),
            [_batch()],
            "cpu",
            output_dir=output,
            metadata={"dataset_manifest_hash": "test-hash"},
        )
    assert (output / "evaluation_invalid.json").is_file()
    assert not (output / "metrics.json").exists()
    invalid = json.loads((output / "evaluation_invalid.json").read_text(encoding="utf-8"))
    assert "reason" in invalid
    assert "accuracy" not in invalid


def test_test_evaluation_obeys_selection_lock_and_force(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status_path = run_dir / "run_status.json"
    status_path.write_text('{"selection_complete": false}\n', encoding="utf-8")
    model = TableModel()
    with pytest.raises(TestSetLockedError, match="selection is complete"):
        evaluate_model(model, PoisonLoader(), "cpu", run_dir=run_dir, split="test")
    assert model.calls == 0

    status_path.write_text('{"selection_complete": true}\n', encoding="utf-8")
    first = evaluate_model(
        model,
        [_batch()],
        "cpu",
        run_dir=run_dir,
        split="test",
        metadata={"dataset_manifest_hash": "same-data", "seed": 42},
    )
    assert first["metadata"]["test_evaluation_count"] == 1
    lock = json.loads((run_dir / "test_evaluated.lock").read_text(encoding="utf-8"))
    assert lock["test_evaluation_count"] == 1
    with pytest.raises(TestSetLockedError, match="already been run"):
        evaluate_model(model, [_batch()], "cpu", run_dir=run_dir, split="test")
    with pytest.warns(UserWarning, match="FORCE"):
        second = evaluate_model(model, [_batch()], "cpu", run_dir=run_dir, split="test", force=True)
    assert second["metadata"]["test_evaluation_count"] == 2
    saved = json.loads((run_dir / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["test_evaluation_count"] == 2
    assert np.isfinite(saved["accuracy"])
