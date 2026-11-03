"""Classification metrics with explicit three-class and multilingual breakdowns."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sentiment.constants import ID2LABEL, LABELS


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], languages: Sequence[str]
) -> dict[str, Any]:
    """Compute global, per-class, per-language, and majority-baseline metrics."""
    raw_truth = np.asarray(y_true)
    raw_predictions = np.asarray(y_pred)
    if not np.issubdtype(raw_truth.dtype, np.integer) or np.issubdtype(raw_truth.dtype, np.bool_):
        raise ValueError("y_true must contain integer class ids.")
    if not np.issubdtype(raw_predictions.dtype, np.integer) or np.issubdtype(
        raw_predictions.dtype, np.bool_
    ):
        raise ValueError("y_pred must contain integer class ids.")
    truth = raw_truth.astype(np.int64, copy=False)
    predictions = raw_predictions.astype(np.int64, copy=False)
    language_values = list(languages)
    if truth.ndim != 1 or predictions.ndim != 1 or len(truth) == 0:
        raise ValueError("y_true and y_pred must be non-empty one-dimensional sequences.")
    if len(truth) != len(predictions) or len(truth) != len(language_values):
        raise ValueError("y_true, y_pred, and languages must have the same length.")
    if not np.isin(truth, [0, 1, 2]).all() or not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError("Class ids must be integers in {0, 1, 2}.")
    if any(not isinstance(language, str) or not language for language in language_values):
        raise ValueError("Every language value must be a non-empty string.")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1, 2], zero_division=0
    )
    true_counts = Counter(truth.tolist())
    majority_label = min(true_counts, key=lambda label: (-true_counts[label], label))
    accuracy = float(accuracy_score(truth, predictions))
    majority_accuracy = float(np.mean(truth == majority_label))
    per_class = {
        ID2LABEL[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(3)
    }
    per_language: dict[str, dict[str, float | int]] = {}
    for language in sorted(set(language_values)):
        mask = np.asarray([value == language for value in language_values])
        per_language[language] = {
            "accuracy": float(accuracy_score(truth[mask], predictions[mask])),
            "macro_f1": float(
                f1_score(
                    truth[mask],
                    predictions[mask],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "support": int(mask.sum()),
        }
    report_text = classification_report(
        truth,
        predictions,
        labels=[0, 1, 2],
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    confusion = confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist()
    neutral_predicted = int(np.count_nonzero(predictions == 1))
    return {
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(truth, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "per_language": per_language,
        "confusion_matrix": confusion,
        "majority_class": {"label": majority_label, "name": ID2LABEL[majority_label]},
        "majority_class_accuracy": majority_accuracy,
        "accuracy_above_majority": accuracy - majority_accuracy,
        "neutral_recall": per_class["Neutral"]["recall"],
        "neutral_predicted_count": neutral_predicted,
        "neutral_never_predicted": neutral_predicted == 0,
        "classification_report": report_text,
    }
