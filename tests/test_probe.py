"""Tests for feature caching and validation-selected frozen-feature probes."""

import json

import numpy as np
import torch

from sentiment.data.collate import DynamicPaddingCollator
from sentiment.data.dataset import SentimentDataset, make_dataloader
from sentiment.io_utils import read_jsonl, sha256_file
from sentiment.models.factory import load_classifier
from sentiment.training.probe import (
    FeatureData,
    MajorityClassifier,
    ProbeTrainConfig,
    extract_features,
    feature_cache_key,
    train_probe,
)
from tests.fixtures import tiny_model_and_tokenizer


def _records(prefix: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{prefix}-{index}",
            "language": "en",
            "text": "good movie" if index % 2 else "bad review",
            "label": index % 3,
        }
        for index in range(count)
    ]


def test_feature_cache_reuse_and_invalidation(tmp_path, tiny_tokenizer) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "model", tiny_tokenizer)
    model = load_classifier(model_path, local_files_only=True)
    dataset = SentimentDataset(_records("train", 4), tiny_tokenizer, 16, True)
    loader = make_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=1,
        collate_fn=DynamicPaddingCollator(tiny_tokenizer),
    )
    cache_path = tmp_path / "features" / "train.npz"
    first = extract_features(
        model,
        loader,
        model_name="mdeberta",
        max_length=16,
        use_lang_tag=True,
        split_file_hash="split-hash",
        cache_path=cache_path,
    )
    second = extract_features(
        model,
        loader,
        model_name="mdeberta",
        max_length=16,
        use_lang_tag=True,
        split_file_hash="split-hash",
        cache_path=cache_path,
    )
    assert first.cache_key == second.cache_key
    assert np.array_equal(first.features, second.features)
    updated = extract_features(
        model,
        loader,
        model_name="mdeberta",
        max_length=32,
        use_lang_tag=True,
        split_file_hash="split-hash",
        cache_path=cache_path,
    )
    assert updated.cache_key != first.cache_key
    assert feature_cache_key("mdeberta", 16, True, "a") != feature_cache_key(
        "mdeberta", 16, True, "b"
    )


def _feature_data(seed: int, count: int, width: int = 8) -> FeatureData:
    rng = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 3
    features = rng.normal(size=(count, width)).astype(np.float32)
    features[:, 0] += labels * 1.5
    return FeatureData(
        features=features,
        labels=labels,
        sample_ids=np.asarray([f"sample-{seed}-{index}" for index in range(count)]),
        languages=np.asarray(["en"] * count),
        cache_key=f"key-{seed}",
    )


def test_probe_early_stops_on_validation_plateau(tmp_path, monkeypatch) -> None:
    from sentiment.training import probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "compute_metrics",
        lambda *_args, **_kwargs: {
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "neutral_recall": 0.25,
        },
    )
    status = train_probe(
        {"train": _feature_data(1, 12), "validation": _feature_data(2, 6)},
        tmp_path / "run",
        encoder_name="tiny",
        config=ProbeTrainConfig(batch_size=128, epochs=5, patience=1, seed=9),
    )
    log = read_jsonl(tmp_path / "run" / "train_log.jsonl")
    assert len(log) == 2
    assert status["best_epoch"] == 1
    assert json.loads((tmp_path / "run" / "run_status.json").read_text())["selection_complete"]


def test_best_checkpoint_selected_by_validation_macro_f1(tmp_path, monkeypatch) -> None:
    from sentiment.training import probe as probe_module

    scores = iter([0.9, 0.2])
    monkeypatch.setattr(
        probe_module,
        "compute_metrics",
        lambda *_args, **_kwargs: {
            "accuracy": 0.5,
            "macro_f1": next(scores),
            "neutral_recall": 0.0,
        },
    )
    status = train_probe(
        {"train": _feature_data(3, 15), "validation": _feature_data(4, 6)},
        tmp_path / "best-run",
        encoder_name="tiny",
        config=ProbeTrainConfig(batch_size=128, epochs=2, patience=3, seed=4),
    )
    assert status["best_epoch"] == 1
    assert status["best_val_macro_f1"] == 0.9


def test_poisoned_test_features_are_never_read(tmp_path) -> None:
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError(f"test feature accessed: {name}")

    common = {"train": _feature_data(5, 12), "validation": _feature_data(6, 6)}
    first = train_probe(
        {**common, "test": Poison()},
        tmp_path / "with-poison",
        encoder_name="tiny",
        config=ProbeTrainConfig(batch_size=128, epochs=2, patience=2, seed=7),
    )
    second = train_probe(
        common,
        tmp_path / "without-test",
        encoder_name="tiny",
        config=ProbeTrainConfig(batch_size=128, epochs=2, patience=2, seed=7),
    )
    first_hash = sha256_file(tmp_path / "with-poison" / "best_model" / "probe_head.safetensors")
    second_hash = sha256_file(tmp_path / "without-test" / "best_model" / "probe_head.safetensors")
    assert first_hash == second_hash
    assert first["best_val_macro_f1"] == second["best_val_macro_f1"]


def test_majority_classifier_uses_train_majority() -> None:
    model = MajorityClassifier.from_training_labels([0, 1, 1, 2, 1])
    predictions = model.predict(torch.tensor([[1], [2], [3]]))
    assert model.label == 1
    assert predictions.tolist() == [1, 1, 1]
