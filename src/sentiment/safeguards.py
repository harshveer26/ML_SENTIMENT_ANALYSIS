"""Fail-fast checks for finite numeric values and benchmark records."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sentiment.constants import LANGUAGES


class NonFiniteError(ValueError):
    """Raised when a tensor, array, or model parameter contains NaN or infinity."""


class DataValidationError(ValueError):
    """Raised when a benchmark record violates the canonical schema."""


class SplitOverlapError(ValueError):
    """Raised when review text appears in more than one data split."""


class TestSetLockedError(RuntimeError):
    """Raised when test evaluation is requested before model selection is complete."""

    __test__ = False


def assert_finite_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Raise NonFiniteError if any element of a PyTorch tensor is NaN or infinite."""
    if not torch.isfinite(tensor).all().item():
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise NonFiniteError(f"{name} contains {bad_count} non-finite value(s).")


def assert_finite_array(array: np.ndarray, name: str = "array") -> None:
    """Raise NonFiniteError if a numeric NumPy array contains NaN or infinity."""
    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {values.dtype}.")
    finite = np.isfinite(values)
    if not finite.all():
        raise NonFiniteError(f"{name} contains {int((~finite).sum())} non-finite value(s).")


def assert_finite_model(model: torch.nn.Module) -> None:
    """Scan all named model parameters and report offending names and total count."""
    bad_parameters: list[tuple[str, int]] = []
    total_bad = 0
    for name, parameter in model.named_parameters():
        invalid = ~torch.isfinite(parameter.detach())
        count = int(invalid.sum().item())
        if count:
            total_bad += count
            if len(bad_parameters) < 10:
                bad_parameters.append((name, count))
    if bad_parameters:
        details = ", ".join(f"{name} ({count})" for name, count in bad_parameters)
        raise NonFiniteError(
            f"Model has {total_bad} non-finite parameter value(s); "
            f"first offending parameters: {details}."
        )


def validate_label(value: Any) -> int:
    """Validate and return an integer class id in the range 0–2."""
    invalid_type = isinstance(value, bool) or not isinstance(value, (int, np.integer))
    if invalid_type or int(value) not in range(3):
        raise DataValidationError(f"label must be an integer in {{0, 1, 2}}, got {value!r}.")
    return int(value)


def validate_record(record: Any) -> dict[str, Any]:
    """Validate the required canonical fields in a benchmark record."""
    if not isinstance(record, dict):
        raise DataValidationError(f"record must be a mapping, got {type(record).__name__}.")
    required = {"sample_id", "language", "text", "label"}
    missing = sorted(required - set(record))
    if missing:
        raise DataValidationError(f"record is missing required field(s): {', '.join(missing)}.")
    sample_id = record["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise DataValidationError("sample_id must be a non-empty string.")
    language = record["language"]
    if not isinstance(language, str) or language.lower() not in LANGUAGES:
        raise DataValidationError(f"language must be one of {LANGUAGES}, got {language!r}.")
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError("text must be a non-empty string.")
    validate_label(record["label"])
    return record
