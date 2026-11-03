"""Small, testable helpers for the Streamlit demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentiment.constants import ID2LABEL
from sentiment.io_utils import read_json, read_jsonl, sha256_file


def discover_runs(runs_dir: str | Path = "runs") -> list[dict[str, Any]]:
    """Return model runs with invalid runs marked unavailable for selection."""
    root = Path(runs_dir)
    found: list[dict[str, Any]] = []
    if not root.is_dir():
        return found
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint = run_dir / "best_model"
        status_path = run_dir / "run_status.json"
        if not checkpoint.is_dir() and not status_path.is_file():
            continue
        status = read_json(status_path) if status_path.is_file() else {}
        invalid_marker = next(
            (
                p
                for p in (
                    run_dir / "INVALID.md",
                    run_dir / "INVALID",
                    checkpoint / "INVALID.md",
                    checkpoint / "INVALID",
                )
                if p.exists()
            ),
            None,
        )
        reason = invalid_marker.read_text(encoding="utf-8").strip() if invalid_marker else None
        if invalid_marker and not reason:
            reason = f"Invalid marker present: {invalid_marker.name}"
        config_path = run_dir / "config.json"
        config = read_json(config_path) if config_path.is_file() else {}
        val_path = run_dir / "eval_validation" / "metrics.json"
        val_metrics = read_json(val_path) if val_path.is_file() else {}
        found.append(
            {
                "run_id": run_dir.name,
                "path": run_dir,
                "checkpoint": checkpoint if checkpoint.is_dir() else run_dir,
                "kind": status.get("kind", "unknown"),
                "backbone": config.get("model", {}).get("name", "unknown"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "has_test_evaluation": (run_dir / "eval_test" / "metrics.json").is_file(),
                "invalid": invalid_marker is not None,
                "invalid_reason": reason,
            }
        )
    return found


def load_comparison(path: str | Path = "reports/results/comparison.json") -> dict[str, Any] | None:
    """Load comparison JSON, keeping PENDING runs and filtering invalid artifacts."""
    target = Path(path)
    if not target.is_file():
        return None
    data = read_json(target)
    data["runs"] = [
        run
        for run in data.get("runs", [])
        if not (Path(run.get("run_dir", "")) / "INVALID.md").exists()
        and not (Path(run.get("run_dir", "")) / "INVALID").exists()
    ]
    return data


def load_reference(
    path: str | Path = "reports/reference/recorded_baseline.json",
) -> dict[str, Any] | None:
    target = Path(path)
    return read_json(target) if target.is_file() else None


def load_benchmark(data_dir: str | Path = "data/amazon_reviews_benchmark") -> dict[str, Any]:
    """Read benchmark manifest and produce simple language/split/label counts."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "reason": f"Benchmark manifest not found: {manifest_path}"}
    manifest = read_json(manifest_path)
    counts: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        records = read_jsonl(path)
        tally: dict[tuple[str, int], int] = {}
        for record in records:
            language, label = str(record["language"]), int(record["label"])
            tally[(language, label)] = tally.get((language, label), 0) + 1
        for (language, label), count in sorted(tally.items()):
            counts.append(
                {"split": split, "language": language, "label": ID2LABEL[label], "count": count}
            )
        if split == "train":
            samples.extend(records)
    return {
        "available": True,
        "manifest": manifest,
        "manifest_hash": sha256_file(manifest_path),
        "counts": counts,
        "samples": samples,
    }


def neutral_recall_statement(metrics: dict[str, Any], probe_metrics: dict[str, Any] | None) -> str:
    if (
        not probe_metrics
        or metrics.get("neutral_recall") is None
        or probe_metrics.get("neutral_recall") is None
    ):
        return "Neutral recall comparison: PENDING."
    improved = float(metrics["neutral_recall"]) > float(probe_metrics["neutral_recall"])
    return (
        "Neutral recall improved vs probe."
        if improved
        else "Neutral recall did not improve vs probe."
    )


def records_from_csv(text: str) -> list[dict[str, str]]:
    """Parse batch CSV text; accept a text column and optional language column."""
    import csv
    from io import StringIO

    rows = list(csv.DictReader(StringIO(text)))
    if "text" not in (rows[0].keys() if rows else []):
        raise ValueError("CSV must have a 'text' column; 'language' is optional.")
    return [
        {"text": row.get("text", ""), "language": row.get("language", "") or "und"} for row in rows
    ]
