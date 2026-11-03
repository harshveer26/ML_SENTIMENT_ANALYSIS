"""Tests for model loading, freezing, probe head, and AMP selection."""

import pytest
import torch

from sentiment.models.factory import (
    count_trainable,
    freeze_encoder,
    load_classifier,
    mixed_precision_policy,
)
from sentiment.models.probe_head import ProbeHead
from sentiment.safeguards import NonFiniteError
from tests.fixtures import tiny_model_and_tokenizer


def test_load_local_tiny_classifier_and_reject_nan(tmp_path, tiny_tokenizer) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "clean", tiny_tokenizer)
    model = load_classifier(model_path, local_files_only=True)
    assert model.config.num_labels == 3
    assert model.config.id2label[1] == "Neutral"

    bad_path, _ = tiny_model_and_tokenizer(tmp_path / "bad", tiny_tokenizer)
    bad_model = load_classifier(bad_path, local_files_only=True)
    bad_model.classifier.weight.data[0, 0] = float("nan")
    bad_model.save_pretrained(bad_path, safe_serialization=True)
    with pytest.raises(NonFiniteError, match="non-finite"):
        load_classifier(bad_path, local_files_only=True)


def test_freeze_encoder_leaves_classifier_trainable(tmp_path, tiny_tokenizer) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "model", tiny_tokenizer)
    model = load_classifier(model_path, local_files_only=True)
    freeze_encoder(model)
    assert all(
        parameter.requires_grad == name.startswith(("pooler.", "classifier."))
        for name, parameter in model.named_parameters()
    )
    counts = count_trainable(model)
    assert counts["trainable"] > 0
    assert counts["frozen"] > counts["trainable"]


def test_probe_head_saves_and_reloads(tmp_path) -> None:
    head = ProbeHead(hidden_size=8, dropout=0.0).eval()
    features = torch.randn(2, 8)
    expected = head(features)
    head.save_pretrained(tmp_path, encoder_name="tiny", max_length=16)
    reloaded, metadata = ProbeHead.from_pretrained(tmp_path)
    assert metadata["encoder_name"] == "tiny"
    assert metadata["max_length"] == 16
    torch.testing.assert_close(reloaded.eval()(features), expected)


def test_amp_policy_resolves_safe_modes() -> None:
    cpu = mixed_precision_policy("cpu", amp="auto")
    assert cpu.mode == "none" and cpu.autocast_dtype is None and not cpu.use_grad_scaler
    bf16 = mixed_precision_policy("cuda", amp="auto", bf16_supported=True)
    assert bf16.mode == "bf16" and bf16.autocast_dtype == torch.bfloat16
    unsupported_auto = mixed_precision_policy("cuda", amp="auto", bf16_supported=False)
    assert unsupported_auto.mode == "none"
    fp16 = mixed_precision_policy("cuda", amp="fp16", bf16_supported=False)
    assert fp16.mode == "fp16" and fp16.autocast_dtype == torch.float16 and fp16.use_grad_scaler
    with pytest.raises(ValueError, match="does not support"):
        mixed_precision_policy("cuda", amp="bf16", bf16_supported=False)
