"""Demo UI helper and Streamlit AppTest coverage."""

import json
from pathlib import Path

from app.ui_helpers import discover_runs, load_comparison, neutral_recall_statement


def test_discover_runs_marks_invalid_as_unselectable(tmp_path) -> None:
    runs = tmp_path / "runs"
    good = runs / "good"
    bad = runs / "bad"
    (good / "best_model").mkdir(parents=True)
    (good / "run_status.json").write_text(json.dumps({"kind": "probe"}), encoding="utf-8")
    (bad / "best_model").mkdir(parents=True)
    (bad / "INVALID.md").write_text("non-finite weights", encoding="utf-8")
    found = discover_runs(runs)
    assert [run["run_id"] for run in found] == ["bad", "good"]
    assert found[0]["invalid"] is True
    assert "non-finite" in found[0]["invalid_reason"]
    assert found[1]["invalid"] is False


def test_load_comparison_keeps_pending_and_filters_invalid(tmp_path) -> None:
    pending = tmp_path / "pending"
    invalid = tmp_path / "invalid"
    pending.mkdir()
    invalid.mkdir()
    (invalid / "INVALID").write_text("bad", encoding="utf-8")
    report = tmp_path / "comparison.json"
    report.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "pending", "run_dir": str(pending), "status": "PENDING"},
                    {"run_id": "invalid", "run_dir": str(invalid), "status": "DONE"},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_comparison(report)
    assert [row["run_id"] for row in loaded["runs"]] == ["pending"]
    assert loaded["runs"][0]["status"] == "PENDING"


def test_neutral_recall_callout_is_computed() -> None:
    assert "improved" in neutral_recall_statement({"neutral_recall": 0.5}, {"neutral_recall": 0.3})
    assert "did not improve" in neutral_recall_statement(
        {"neutral_recall": 0.2}, {"neutral_recall": 0.3}
    )


def test_streamlit_app_starts_without_runs(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENTIMENT_RUNS_DIR", str(tmp_path / "empty-runs"))
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert not app.exception
    assert any("Research prototype" in warning.value for warning in app.warning)
    assert len(app.tabs) == 4


def test_streamlit_tiny_model_predicts_and_invalid_is_excluded(
    tiny_tokenizer, tmp_path, monkeypatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from tests.fixtures import tiny_model_and_tokenizer

    runs = tmp_path / "runs"
    model_dir = runs / "tiny-full" / "best_model"
    model_dir.parent.mkdir(parents=True)
    tiny_model_and_tokenizer(model_dir, tiny_tokenizer)
    (model_dir.parent / "run_status.json").write_text(
        json.dumps({"kind": "full", "selection_complete": True}), encoding="utf-8"
    )
    invalid_dir = runs / "invalid" / "best_model"
    invalid_dir.mkdir(parents=True)
    (invalid_dir.parent / "INVALID.md").write_text("fixture invalid", encoding="utf-8")
    monkeypatch.setenv("SENTIMENT_RUNS_DIR", str(runs))

    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.selectbox
    assert all("invalid" not in str(box.options).lower() for box in app.sidebar.selectbox)
    app.text_area[0].set_value("good movie")
    app.button[0].click().run()
    assert not app.exception
    assert any("confidence" in message.value for message in app.success)
