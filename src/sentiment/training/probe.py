"""Frozen-encoder feature caching, probe-head selection, and majority baseline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sentiment.constants import LABELS
from sentiment.evaluation.metrics import compute_metrics
from sentiment.io_utils import write_json, write_jsonl
from sentiment.models.factory import resolve_backbone
from sentiment.models.probe_head import ProbeHead
from sentiment.safeguards import assert_finite_array, assert_finite_tensor
from sentiment.seed import make_generator, set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureData:
    """Cached first-token encoder features and aligned evaluation metadata."""

    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    languages: np.ndarray
    cache_key: str


@dataclass(frozen=True)
class ProbeTrainConfig:
    """Optimizer, batch, epoch, and early-stopping settings for a probe."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 128
    epochs: int = 15
    patience: int = 3
    seed: int = 42

    def validate(self) -> None:
        """Reject invalid or out-of-protocol training settings."""
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.batch_size < 1 or not 1 <= self.epochs <= 15:
            raise ValueError("batch_size must be positive and epochs must be between 1 and 15.")
        if self.patience < 0 or self.seed < 0:
            raise ValueError("patience and seed must be non-negative.")


def feature_cache_key(
    model_name: str, max_length: int, use_lang_tag: bool, split_file_hash: str
) -> str:
    """Hash the four cache-defining inputs into a stable feature-cache key."""
    payload = {
        "model_name": resolve_backbone(model_name),
        "max_length": max_length,
        "use_lang_tag": use_lang_tag,
        "split_file_hash": split_file_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cache(path: Path, expected_key: str) -> FeatureData | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            cache_key = str(cache["cache_key"].item())
            if cache_key != expected_key:
                return None
            features = cache["features"].astype(np.float32, copy=False)
            labels = cache["labels"].astype(np.int64, copy=False)
            sample_ids = cache["sample_ids"].astype(str, copy=False)
            languages = cache["languages"].astype(str, copy=False)
    except (OSError, ValueError, KeyError):
        return None
    assert_finite_array(features, "cached features")
    if not (len(features) == len(labels) == len(sample_ids) == len(languages)):
        raise ValueError(f"Corrupt feature cache has misaligned arrays: {path}")
    return FeatureData(features, labels, sample_ids, languages, cache_key)


def _save_cache(path: Path, data: FeatureData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez_compressed(
            temp_path,
            features=data.features.astype(np.float32, copy=False),
            labels=data.labels.astype(np.int64, copy=False),
            sample_ids=data.sample_ids.astype(str, copy=False),
            languages=data.languages.astype(str, copy=False),
            cache_key=np.asarray(data.cache_key),
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def extract_features(
    model: torch.nn.Module,
    dataloader: Any,
    *,
    model_name: str,
    max_length: int,
    use_lang_tag: bool,
    split_file_hash: str,
    cache_path: str | Path,
    device: str | torch.device = "cpu",
    force: bool = False,
) -> FeatureData:
    """Extract CLS features once or reuse an NPZ cache whose metadata key matches."""
    expected_key = feature_cache_key(model_name, max_length, use_lang_tag, split_file_hash)
    target = Path(cache_path)
    if not force:
        cached = _load_cache(target, expected_key)
        if cached is not None:
            LOGGER.info("Reusing feature cache %s", target)
            return cached

    model = model.to(torch.device(device))
    model.eval()
    encoder = getattr(model, "base_model", None)
    if encoder is None:
        raise ValueError(f"Model {type(model).__name__} has no base_model encoder.")
    encoder.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    all_languages: list[str] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            cls_features = hidden[:, 0, :].float()
            assert_finite_tensor(cls_features, "extracted features")
            all_features.append(cls_features.cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64, copy=False))
            all_ids.extend(batch["sample_ids"])
            all_languages.extend(batch["languages"])
    if not all_features:
        raise ValueError("Feature extraction dataloader produced no examples.")
    data = FeatureData(
        features=np.concatenate(all_features),
        labels=np.concatenate(all_labels),
        sample_ids=np.asarray(all_ids, dtype=str),
        languages=np.asarray(all_languages, dtype=str),
        cache_key=expected_key,
    )
    assert_finite_array(data.features, "extracted features")
    if not (len(data.features) == len(data.labels) == len(data.sample_ids) == len(data.languages)):
        raise ValueError("Extracted features, labels, ids, and languages are not aligned.")
    _save_cache(target, data)
    return data


def _evaluation(
    head: ProbeHead,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    head.eval()
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False))
    loader = DataLoader(TensorDataset(feature_tensor, label_tensor), batch_size=batch_size)
    losses: list[float] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            logits = head(batch_features)
            assert_finite_tensor(logits, "probe validation logits")
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            assert_finite_tensor(loss, "probe validation loss")
            losses.append(float(loss.item()) * len(batch_labels))
            predictions.extend(logits.argmax(dim=-1).tolist())
    metrics = compute_metrics(labels.tolist(), predictions, ["und"] * len(labels))
    return sum(losses) / len(labels), metrics


def train_probe(
    features_by_split: Mapping[str, Any],
    output_dir: str | Path,
    *,
    encoder_name: str,
    max_length: int = 128,
    use_lang_tag: bool = True,
    config: ProbeTrainConfig | None = None,
) -> dict[str, Any]:
    """Train on train features, select only on validation macro-F1, and save the best head."""
    train_data = features_by_split["train"]
    validation_data = features_by_split["validation"]
    if not isinstance(train_data, FeatureData) or not isinstance(validation_data, FeatureData):
        raise TypeError("train and validation entries must be FeatureData instances.")
    settings = config or ProbeTrainConfig()
    settings.validate()
    if train_data.features.ndim != 2 or validation_data.features.ndim != 2:
        raise ValueError("train and validation features must be two-dimensional arrays.")
    if train_data.features.shape[1] != validation_data.features.shape[1]:
        raise ValueError("train and validation feature widths differ.")
    for split_name, data in (("train", train_data), ("validation", validation_data)):
        assert_finite_array(data.features, f"{split_name} features")
        if len(data.features) != len(data.labels) or len(data.features) == 0:
            raise ValueError(f"{split_name} features and labels must be non-empty and aligned.")

    set_seed(settings.seed)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    hidden_size = int(train_data.features.shape[1])
    head = ProbeHead(hidden_size).cpu()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_data.features.astype(np.float32, copy=False)),
            torch.from_numpy(train_data.labels.astype(np.int64, copy=False)),
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=make_generator(settings.seed),
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, settings.epochs + 1):
        head.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            assert_finite_tensor(loss, "probe training loss")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().item()) * len(batch_labels)
            train_count += len(batch_labels)
        validation_loss, validation_metrics = _evaluation(
            head,
            validation_data.features,
            validation_data.labels,
            settings.batch_size,
        )
        score = float(validation_metrics["macro_f1"])
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "val_loss": validation_loss,
                "val_accuracy": validation_metrics["accuracy"],
                "val_macro_f1": score,
                "val_neutral_recall": validation_metrics["neutral_recall"],
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= max(1, settings.patience):
            break

    if best_state is None:
        raise RuntimeError(
            "Probe training completed without selecting a finite validation checkpoint."
        )
    head.load_state_dict(best_state)
    head.save_pretrained(
        target / "best_model",
        encoder_name=encoder_name,
        max_length=max_length,
        use_lang_tag=use_lang_tag,
    )
    write_jsonl(target / "train_log.jsonl", log_rows)
    run_status = {
        "selection_complete": True,
        "kind": "probe",
        "encoder_name": encoder_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "test_evaluated": False,
        "config": asdict(settings),
    }
    write_json(target / "run_status.json", run_status)
    return run_status


class MajorityClassifier(torch.nn.Module):
    """A constant-logit classifier for the train-majority label baseline."""

    kind = "majority"

    def __init__(self, label: int) -> None:
        super().__init__()
        if label not in range(3):
            raise ValueError("majority label must be an integer in {0, 1, 2}.")
        self.label = int(label)
        self.num_labels = len(LABELS)

    @classmethod
    def from_training_labels(cls, labels: Any) -> MajorityClassifier:
        """Create the baseline using the most frequent training label (smallest id breaks ties)."""
        values = np.asarray(labels)
        if values.ndim != 1 or len(values) == 0 or not np.isin(values, [0, 1, 2]).all():
            raise ValueError("Training labels must be a non-empty sequence of class ids 0–2.")
        counts = Counter(values.astype(int).tolist())
        label = min(counts, key=lambda value: (-counts[value], value))
        return cls(label)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        """Return logits that predict the fixed majority class for every input row."""
        labels = kwargs.get("labels")
        logits = torch.zeros((input_ids.shape[0], self.num_labels), device=input_ids.device)
        logits[:, self.label] = 1.0
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return type("ClassifierOutput", (), {"logits": logits, "loss": loss})()

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the same majority class for each input row."""
        return torch.full(
            (input_ids.shape[0],),
            self.label,
            dtype=torch.long,
            device=input_ids.device,
        )
