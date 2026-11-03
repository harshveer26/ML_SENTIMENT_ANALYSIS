"""Tests for numerical and record safeguards."""

import numpy as np
import pytest
import torch

from sentiment.safeguards import (
    DataValidationError,
    NonFiniteError,
    assert_finite_array,
    assert_finite_model,
    assert_finite_tensor,
    validate_record,
)


def test_finite_checks_accept_clean_values() -> None:
    assert_finite_tensor(torch.tensor([1.0, 2.0]), "clean")
    assert_finite_array(np.array([1.0, 2.0]), "clean")
    assert_finite_model(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_finite_tensor_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(NonFiniteError, match="non-finite"):
        assert_finite_tensor(torch.tensor([value]), "broken")


def test_model_check_names_offending_parameter() -> None:
    model = torch.nn.Linear(2, 1)
    model.weight.data[0, 0] = float("nan")
    with pytest.raises(NonFiniteError, match="weight"):
        assert_finite_model(model)


@pytest.mark.parametrize(
    "record",
    [
        {"sample_id": "a", "language": "en", "text": "good"},
        {"sample_id": "a", "language": "en", "text": "good", "label": 7},
        {"sample_id": "a", "language": "en", "text": "  ", "label": 1},
    ],
)
def test_invalid_records_raise(record: dict[str, object]) -> None:
    with pytest.raises(DataValidationError):
        validate_record(record)
