"""Compare final test evaluations without mixing benchmark manifests."""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sentiment.io_utils import read_json, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


def _manifest_hash(run_dir: Path, metrics: dict[str, Any] | None) -> str | None:
    if metrics is not None:
        metadata = metrics.get("metadata", {})
        value = metadata.get("dataset_manifest_hash")
        if value:
            return str(value)
    reference = run_dir / "dataset_manifest_ref.json"
    if reference.is_file():
        value = read_json(reference).get("sha256")
        return str(value) if value else None
    return None


def compare_runs(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path = "reports/results",
    *,
    benchmark_dir: str | Path = "data/amazon_reviews_benchmark",
) -> dict[str, Any]:
    """Write comparison JSON/CSV/Markdown and list unevaluated runs as PENDING."""
    if not run_dirs:
        raise ValueError("At least one run directory is required.")
    runs: list[dict[str, Any]] = []
    hashes: set[str] = set()
    probe_metrics: dict[str, Any] | None = None
    probe_id: str | None = None
    majority_metrics: dict[str, Any] | None = None
    any_test_evaluated = False
    for supplied in run_dirs:
        run_dir = Path(supplied)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        status = (
            read_json(run_dir / "run_status.json")
            if (run_dir / "run_status.json").is_file()
            else {}
        )
        metrics_path = run_dir / "eval_test" / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.is_file() else None
        manifest_hash = _manifest_hash(run_dir, metrics)
        if manifest_hash:
            hashes.add(manifest_hash)
        kind = status.get("kind", "unknown")
        entry = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "kind": kind,
            "status": "DONE" if metrics is not None else "PENDING",
            "dataset_manifest_hash": manifest_hash,
            "metrics": metrics,
        }
        runs.append(entry)
        if metrics is not None:
            any_test_evaluated = True
            if kind == "probe":
                probe_metrics = metrics
                probe_id = run_dir.name
            if kind == "majority":
                majority_metrics = metrics

    if len(hashes) > 1:
        raise ValueError(
            f"Cannot compare runs with different dataset manifest hashes: {sorted(hashes)}"
        )
    if any(entry["dataset_manifest_hash"] is None for entry in runs) and hashes:
        raise ValueError(
            "Cannot compare a run with an unknown dataset manifest hash to hashed runs."
        )
    manifest_hash = next(iter(hashes), None)

    if majority_metrics is not None:
        majority_baseline = {
            "accuracy": majority_metrics.get("accuracy"),
            "source": "majority run test evaluation",
        }
    elif any_test_evaluated:
        test_path = Path(benchmark_dir) / "test.jsonl"
        if not test_path.is_file():
            majority_baseline = {
                "accuracy": None,
                "source": "PENDING: benchmark test split missing",
            }
        else:
            labels = [int(record["label"]) for record in read_jsonl(test_path)]
            if not labels:
                raise ValueError(f"Benchmark test split is empty: {test_path}")
            counts = {label: labels.count(label) for label in (0, 1, 2)}
            majority_baseline = {
                "accuracy": max(counts.values()) / len(labels),
                "source": "computed from test labels for comparison; no majority run was supplied",
            }
    else:
        majority_baseline = {"accuracy": None, "source": "PENDING: no test evaluations exist"}

    for entry in runs:
        metrics = entry["metrics"]
        if metrics is None:
            entry["neutral_recall_statement"] = (
                "Neutral recall comparison: PENDING (test evaluation missing)."
            )
            continue
        neutral_recall = metrics.get("neutral_recall")
        if probe_metrics is None or entry["kind"] == "probe":
            entry["neutral_recall_statement"] = (
                "Neutral recall comparison: PENDING (probe test metrics unavailable for "
                f"{probe_id or 'comparison'})."
            )
        else:
            probe_recall = probe_metrics.get("neutral_recall")
            improved = neutral_recall > probe_recall
            verb = "improved" if improved else "did not improve"
            entry["neutral_recall_statement"] = f"Neutral recall {verb} vs {probe_id}."
        if "accuracy_above_majority" not in metrics and majority_baseline["accuracy"] is not None:
            metrics["accuracy_above_majority"] = metrics["accuracy"] - majority_baseline["accuracy"]

    result = {
        "dataset_manifest_hash": manifest_hash,
        "majority_baseline": majority_baseline,
        "runs": runs,
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "comparison.json", result)
    columns = [
        "run_id",
        "kind",
        "status",
        "accuracy",
        "macro_f1",
        "neutral_recall",
        "accuracy_above_majority",
        "neutral_recall_statement",
    ]
    with (target / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in runs:
            metrics = entry["metrics"] or {}
            writer.writerow(
                {
                    "run_id": entry["run_id"],
                    "kind": entry["kind"],
                    "status": entry["status"],
                    "accuracy": metrics.get("accuracy", "PENDING"),
                    "macro_f1": metrics.get("macro_f1", "PENDING"),
                    "neutral_recall": metrics.get("neutral_recall", "PENDING"),
                    "accuracy_above_majority": metrics.get("accuracy_above_majority", "PENDING"),
                    "neutral_recall_statement": entry["neutral_recall_statement"],
                }
            )
    baseline_accuracy = (
        majority_baseline["accuracy"] if majority_baseline["accuracy"] is not None else "PENDING"
    )
    baseline_summary = f"{baseline_accuracy} ({majority_baseline['source']})"
    lines = [
        "# Test-set run comparison",
        "",
        f"Dataset manifest hash: `{manifest_hash or 'PENDING'}`",
        f"Majority baseline: {baseline_summary}",
        "",
        "| Run | Kind | Status | Accuracy | Macro-F1 | Neutral recall | Accuracy above majority |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in runs:
        metrics = entry["metrics"] or {}
        lines.append(
            "| {run} | {kind} | {status} | {accuracy} | {macro} | {neutral} | {above} |".format(
                run=entry["run_id"],
                kind=entry["kind"],
                status=entry["status"],
                accuracy=metrics.get("accuracy", "PENDING"),
                macro=metrics.get("macro_f1", "PENDING"),
                neutral=metrics.get("neutral_recall", "PENDING"),
                above=metrics.get("accuracy_above_majority", "PENDING"),
            )
        )
        lines.append("")
        lines.append(entry["neutral_recall_statement"])
    (target / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote comparison files to %s", target)
    return result
