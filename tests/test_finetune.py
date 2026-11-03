"""Tiny local-model tests for the fine-tuning loop and optional LoRA."""

import json

import pytest
import torch

from sentiment.config import TrainConfig
from sentiment.data.collate import DynamicPaddingCollator
from sentiment.data.dataset import SentimentDataset, make_dataloader
from sentiment.io_utils import sha256_file
from sentiment.models.factory import load_classifier
from sentiment.safeguards import NonFiniteError
from sentiment.seed import set_seed
from sentiment.training.finetune import prepare_lora, train_model
from sentiment.training.optim import balanced_class_weights, build_adamw
from tests.fixtures import tiny_model_and_tokenizer


def _records(prefix: str) -> list[dict[str, object]]:
    return [
        {"sample_id": f"{prefix}-0", "language": "en", "text": "good movie", "label": 2},
        {"sample_id": f"{prefix}-1", "language": "fr", "text": "bad review", "label": 0},
    ]


def _loaders(tokenizer):
    train = SentimentDataset(_records("train"), tokenizer, 16, False)
    validation = SentimentDataset(_records("validation"), tokenizer, 16, False)
    collator = DynamicPaddingCollator(tokenizer)
    train_loader = make_dataloader(train, batch_size=2, shuffle=True, seed=8, collate_fn=collator)
    validation_loader = make_dataloader(
        validation, batch_size=2, shuffle=False, seed=9, collate_fn=collator
    )
    return train_loader, validation_loader


def _tiny_model(tmp_path, tokenizer, name="base"):
    path, _ = tiny_model_and_tokenizer(tmp_path / name, tokenizer)
    return path, load_classifier(path, local_files_only=True)


def _settings(seed: int = 12) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        learning_rate=1e-3,
        weight_decay=0.01,
        batch_size=2,
        epochs=2,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        early_stopping_patience=2,
        mixed_precision="none",
    )


def test_tiny_training_saves_reloadable_best_model(tmp_path, tiny_tokenizer) -> None:
    _, model = _tiny_model(tmp_path, tiny_tokenizer)
    train_loader, validation_loader = _loaders(tiny_tokenizer)
    run_dir = tmp_path / "run"
    status = train_model(
        model,
        train_loader,
        validation_loader,
        run_dir,
        config=_settings(),
        tokenizer=tiny_tokenizer,
        model_identifier="tiny",
    )
    assert status["selection_complete"] is True
    assert len((run_dir / "best_model" / "model.safetensors").read_bytes()) > 0
    assert len((run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    reloaded = load_classifier(run_dir / "best_model", local_files_only=True)
    fixed_input = torch.tensor([[tiny_tokenizer.convert_tokens_to_ids("good")]])
    mask = torch.ones_like(fixed_input)
    reloaded.eval()
    with torch.no_grad():
        reloaded_again = load_classifier(run_dir / "best_model", local_files_only=True).eval()
        expected = reloaded_again(input_ids=fixed_input, attention_mask=mask).logits
        actual = reloaded(input_ids=fixed_input, attention_mask=mask).logits
    torch.testing.assert_close(actual, expected)


def test_non_finite_loss_stops_without_checkpoint(tmp_path, tiny_tokenizer, monkeypatch) -> None:
    _, model = _tiny_model(tmp_path, tiny_tokenizer)
    train_loader, validation_loader = _loaders(tiny_tokenizer)
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda *args, **kwargs: torch.tensor(float("nan"), requires_grad=True),
    )
    run_dir = tmp_path / "failed"
    with pytest.raises(NonFiniteError, match="training loss"):
        train_model(model, train_loader, validation_loader, run_dir, config=_settings())
    failure = json.loads((run_dir / "training_failed.json").read_text(encoding="utf-8"))
    assert failure["step"] == 1
    assert failure["last_good_loss"] is None
    assert not (run_dir / "best_model").exists()


def test_validation_selects_best_epoch_without_test_loader(
    tmp_path, tiny_tokenizer, monkeypatch
) -> None:
    from sentiment.training import finetune as finetune_module

    _, model = _tiny_model(tmp_path, tiny_tokenizer)
    train_loader, validation_loader = _loaders(tiny_tokenizer)
    scores = iter([0.8, 0.2])
    calls = []

    def fake_evaluate(model, dataloader, device, amp, **kwargs):
        calls.append(dataloader)
        score = next(scores)
        return {
            "metrics": {
                "macro_f1": score,
                "loss": 1.0,
                "accuracy": 0.5,
                "per_class": {
                    "Negative": {"recall": 0.0},
                    "Neutral": {"recall": 0.0},
                    "Positive": {"recall": 1.0},
                },
                "neutral_recall": 0.0,
            }
        }

    monkeypatch.setattr(finetune_module, "evaluate_model", fake_evaluate)

    class PoisonTestLoader:
        def __iter__(self):
            raise AssertionError("test loader must not be touched")

    poison_test_loader = PoisonTestLoader()
    status = train_model(
        model,
        train_loader,
        validation_loader,
        tmp_path / "selection",
        config=_settings(),
    )
    assert poison_test_loader is not None
    assert status["best_epoch"] == 1
    assert calls == [validation_loader, validation_loader]


def test_dry_run_does_not_write_checkpoint(tmp_path, tiny_tokenizer) -> None:
    _, model = _tiny_model(tmp_path, tiny_tokenizer)
    train_loader, validation_loader = _loaders(tiny_tokenizer)
    run_dir = tmp_path / "dry-run"
    result = train_model(
        model,
        train_loader,
        validation_loader,
        run_dir,
        config=_settings(),
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["examples_forwarded"] == 2
    assert result["checkpoint_written"] is False
    assert not run_dir.exists()


def test_balanced_class_weights_use_only_supplied_training_labels() -> None:
    weights = balanced_class_weights([0, 0, 1, 2])
    assert weights.shape == (3,)
    assert float(weights.sum()) == pytest.approx(10 / 3)
    assert weights.tolist() == pytest.approx([2 / 3, 4 / 3, 4 / 3])
    with pytest.raises(ValueError, match="missing"):
        balanced_class_weights([0, 0, 1])


def test_lora_wrapper_is_parameter_efficient_and_reloads(tmp_path, tiny_tokenizer) -> None:
    peft = pytest.importorskip("peft")
    base_path, base_model = _tiny_model(tmp_path, tiny_tokenizer)
    lora_model = prepare_lora(base_model, "mdeberta", rank=4)
    trainable = sum(
        parameter.numel() for parameter in lora_model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in lora_model.parameters())
    assert trainable < total / 2
    adapter_dir = tmp_path / "adapter"
    lora_model.save_pretrained(adapter_dir, safe_serialization=True)
    fresh_base = load_classifier(base_path, local_files_only=True)
    reloaded = peft.PeftModel.from_pretrained(fresh_base, adapter_dir)
    input_ids = torch.tensor([[tiny_tokenizer.convert_tokens_to_ids("good")]])
    mask = torch.ones_like(input_ids)
    lora_model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            lora_model(input_ids=input_ids, attention_mask=mask).logits,
            reloaded(input_ids=input_ids, attention_mask=mask).logits,
        )


def test_same_seed_reproduces_tiny_best_weights(tmp_path, tiny_tokenizer) -> None:
    set_seed(55)
    _, first_model = _tiny_model(tmp_path, tiny_tokenizer, name="first-base")
    first_train, first_validation = _loaders(tiny_tokenizer)
    first_run = tmp_path / "first-run"
    train_model(
        first_model,
        first_train,
        first_validation,
        first_run,
        config=_settings(seed=31),
        tokenizer=tiny_tokenizer,
        model_identifier="tiny",
    )

    set_seed(55)
    _, second_model = _tiny_model(tmp_path, tiny_tokenizer, name="second-base")
    second_train, second_validation = _loaders(tiny_tokenizer)
    second_run = tmp_path / "second-run"
    train_model(
        second_model,
        second_train,
        second_validation,
        second_run,
        config=_settings(seed=31),
        tokenizer=tiny_tokenizer,
        model_identifier="tiny",
    )
    assert sha256_file(first_run / "best_model" / "model.safetensors") == sha256_file(
        second_run / "best_model" / "model.safetensors"
    )


def test_adamw_no_decay_group_includes_bias_and_norm() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_adamw(model, 1e-3, 0.1)
    assert sorted(group["weight_decay"] for group in optimizer.param_groups) == [0.0, 0.1]
