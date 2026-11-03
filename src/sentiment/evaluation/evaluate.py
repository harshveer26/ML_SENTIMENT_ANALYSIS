"""Finite-safe evaluation with checkpoint-selection and test-set locking."""

from __future__ import annotations

import csv
import json
import logging
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sentiment.env_info import capture_env
from sentiment.evaluation.metrics import compute_metrics
from sentiment.io_utils import read_json, sha256_file, write_json
from sentiment.models.factory import mixed_precision_policy
from sentiment.safeguards import (
    NonFiniteError,
    TestSetLockedError,
    assert_finite_array,
    assert_finite_tensor,
)

LOGGER = logging.getLogger(__name__)
_ARTIFACTS = (
    "metrics.json",
    "classification_report.txt",
    "per_language.json",
    "confusion_matrix.csv",
    "predictions.csv",
    "predictions_and_logits.npz",
    "evaluation_invalid.json",
)


def _test_count(run_dir: Path) -> tuple[int, bool]:
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        raise TestSetLockedError(f"Test evaluation requires run status file: {status_path}")
    status = read_json(status_path)
    if not status.get("selection_complete", False):
        raise TestSetLockedError(
            "Test evaluation is locked until checkpoint selection is complete "
            "(run_status.json selection_complete=true)."
        )
    lock = run_dir / "test_evaluated.lock"
    if lock.exists():
        lock_data = read_json(lock)
        return int(lock_data.get("test_evaluation_count", 1)), True
    return 0, False


def _metadata(
    run_dir: Path | None,
    split: str,
    metadata: dict[str, Any] | None,
    test_count: int | None,
) -> dict[str, Any]:
    result = dict(metadata or {})
    manifest_ref = result.get("dataset_manifest_hash")
    if manifest_ref is None and run_dir is not None:
        ref_path = run_dir / "dataset_manifest_ref.json"
        if ref_path.is_file():
            ref = read_json(ref_path)
            manifest_ref = ref.get("sha256") or ref.get("dataset_manifest_hash")
    result.update(
        {
            "dataset_manifest_hash": manifest_ref or "unknown",
            "split": split,
            "seed": result.get("seed"),
            "model_identifier": result.get("model_identifier", "unknown"),
            "config": result.get("config", {}),
            "env": result.get("env", capture_env()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if test_count is not None:
        result["test_evaluation_count"] = test_count
    return result


def _write_csv(
    path: Path,
    metadata: dict[str, Any],
    header: list[str],
    rows: list[list[Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# metadata={json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_artifacts(
    output_dir: Path,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    languages: list[str],
    sample_ids: list[str],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {**metrics, "metadata": metadata}
    write_json(output_dir / "metrics.json", metrics_payload)
    (output_dir / "classification_report.txt").write_text(
        "Metadata: "
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        + "\n\n"
        + metrics["classification_report"],
        encoding="utf-8",
    )
    write_json(
        output_dir / "per_language.json",
        {"metadata": metadata, "per_language": metrics["per_language"]},
    )

    labels = ["Negative", "Neutral", "Positive"]
    confusion = metrics["confusion_matrix"]
    _write_csv(
        output_dir / "confusion_matrix.csv",
        metadata,
        ["true_label", *labels],
        [[labels[index], *row] for index, row in enumerate(confusion)],
    )
    prediction_rows: list[list[Any]] = []
    for index, probs in enumerate(probabilities):
        prediction_rows.append(
            [
                sample_ids[index],
                languages[index],
                int(y_true[index]),
                int(y_pred[index]),
                float(np.max(probs)),
                *[float(value) for value in probs],
            ]
        )
    _write_csv(
        output_dir / "predictions.csv",
        metadata,
        ["sample_id", "language", "true", "pred", "confidence", "p_neg", "p_neu", "p_pos"],
        prediction_rows,
    )
    np.savez_compressed(
        output_dir / "predictions_and_logits.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        logits=logits,
        languages=np.asarray(languages),
        sample_ids=np.asarray(sample_ids),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    return sha256_file(output_dir / "metrics.json")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: str | torch.device,
    amp: str = "none",
    *,
    output_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    split: str = "validation",
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run finite-safe evaluation, optionally writing artifacts and enforcing test locks."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if split == "test" and run_dir is None:
        raise TestSetLockedError(
            "Test evaluation requires a run_dir so selection status can be checked."
        )
    run_root = Path(run_dir) if run_dir is not None else None
    output = (
        Path(output_dir)
        if output_dir is not None
        else (run_root / f"eval_{split}" if run_root is not None else None)
    )

    previous_test_count = 0
    if split == "test":
        previous_test_count, already_evaluated = _test_count(run_root)
        if already_evaluated and not force:
            raise TestSetLockedError(
                "Test evaluation has already been run for this checkpoint; pass force=True "
                "to record an additional test evaluation."
            )
        if already_evaluated:
            warnings.warn(
                "FORCE: running test evaluation again on a previously evaluated checkpoint.",
                stacklevel=2,
            )
            LOGGER.warning("FORCE: test evaluation repeated for %s", run_root)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for filename in _ARTIFACTS:
            artifact = output / filename
            if artifact.exists():
                artifact.unlink()

    policy = mixed_precision_policy(device, amp)
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_languages: list[str] = []
    all_sample_ids: list[str] = []
    loss_total = 0.0
    example_count = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"].to(device_obj)
                model_inputs = {
                    key: batch[key].to(device_obj)
                    for key in ("input_ids", "attention_mask")
                    if key in batch
                }
                model_inputs["labels"] = labels
                amp_context = (
                    torch.autocast(device_type=device_obj.type, dtype=policy.autocast_dtype)
                    if policy.autocast_dtype is not None
                    else nullcontext()
                )
                with amp_context:
                    outputs = model(**model_inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                loss = (
                    outputs.loss
                    if getattr(outputs, "loss", None) is not None
                    else torch.nn.functional.cross_entropy(logits, labels)
                )
                assert_finite_tensor(loss, "evaluation loss")
                assert_finite_tensor(logits, "evaluation logits")
                float_logits = logits.float()
                probabilities = torch.softmax(float_logits, dim=-1)
                assert_finite_tensor(probabilities, "evaluation probabilities")
                predictions = probabilities.argmax(dim=-1)
                logits_array = float_logits.detach().cpu().numpy()
                probabilities_array = probabilities.detach().cpu().numpy()
                truth_array = labels.detach().cpu().numpy().astype(np.int64, copy=False)
                prediction_array = predictions.detach().cpu().numpy().astype(np.int64, copy=False)
                assert_finite_array(logits_array, "evaluation logits")
                assert_finite_array(probabilities_array, "evaluation probabilities")
                all_logits.append(logits_array)
                all_probabilities.append(probabilities_array)
                all_truth.append(truth_array)
                all_predictions.append(prediction_array)
                batch_size = len(truth_array)
                loss_total += float(loss.detach().float().item()) * batch_size
                example_count += batch_size
                all_languages.extend(batch.get("languages", ["und"] * batch_size))
                fallback_ids = [
                    f"sample-{example_count - batch_size + index}" for index in range(batch_size)
                ]
                all_sample_ids.extend(batch.get("sample_ids", fallback_ids))
        if not all_truth:
            raise ValueError("Evaluation dataloader produced no examples.")
    except NonFiniteError as exc:
        if output is not None:
            for filename in _ARTIFACTS:
                artifact = output / filename
                if artifact.exists():
                    artifact.unlink()
            write_json(output / "evaluation_invalid.json", {"reason": str(exc)})
        raise

    y_true = np.concatenate(all_truth)
    y_pred = np.concatenate(all_predictions)
    probabilities_array = np.concatenate(all_probabilities)
    logits_array = np.concatenate(all_logits)
    assert_finite_array(logits_array, "evaluation logits")
    assert_finite_array(probabilities_array, "evaluation probabilities")
    metrics = compute_metrics(y_true, y_pred, all_languages)
    metrics["loss"] = loss_total / example_count
    count_for_metadata = previous_test_count + 1 if split == "test" else None
    result_metadata = _metadata(run_root, split, metadata, count_for_metadata)
    metrics_hash = None
    if output is not None:
        metrics_hash = _write_artifacts(
            output,
            result_metadata,
            metrics,
            y_true,
            y_pred,
            probabilities_array,
            logits_array,
            all_languages,
            all_sample_ids,
        )
    if split == "test":
        write_json(
            run_root / "test_evaluated.lock",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "metrics_sha256": metrics_hash,
                "test_evaluation_count": count_for_metadata,
            },
        )
    return {
        "metrics": metrics,
        "metadata": result_metadata,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities_array,
        "logits": logits_array,
        "languages": all_languages,
        "sample_ids": all_sample_ids,
        "output_dir": str(output) if output is not None else None,
    }
