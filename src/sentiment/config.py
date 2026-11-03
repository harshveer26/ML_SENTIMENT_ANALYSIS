"""Typed YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from sentiment.constants import DEFAULT_SEED_BENCHMARK, DEFAULT_SEED_TRAIN, LANGUAGES, SPLIT_SIZES


@dataclass
class DataConfig:
    """Dataset and benchmark sampling options."""

    benchmark_dir: str = "data/amazon_reviews_benchmark"
    source_dir: str = "data/amazon_reviews_multi"
    languages: list[str] | None = None
    seed: int = DEFAULT_SEED_BENCHMARK
    split_sizes: dict[str, int] | None = None


@dataclass
class ModelConfig:
    """Backbone and tokenization options."""

    name: str = "microsoft/mdeberta-v3-base"
    num_labels: int = 3
    max_length: int = 128
    use_lang_tag: bool = True
    local_files_only: bool = False
    peft: str | None = None
    lora_target_modules: list[str] | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training defaults and optimization options."""

    seed: int = DEFAULT_SEED_TRAIN
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    mixed_precision: str = "auto"
    class_weights: str = "off"


@dataclass
class RunConfig:
    """Complete configuration for a run."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output_dir: str = "runs"
    run_name: str | None = None


T = TypeVar("T")


def _construct_section(cls: type[T], values: Any, section: str) -> T:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping.")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {', '.join(unknown)}")
    return cls(**values)


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if "." in key:
            head, tail = key.split(".", 1)
            child = target.setdefault(head, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot apply nested override '{key}' to a non-mapping value.")
            _deep_merge(child, {tail: value})
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _validate(config: RunConfig) -> None:
    if not config.data.languages:
        config.data.languages = list(LANGUAGES)
    if not set(config.data.languages).issubset(LANGUAGES) or len(set(config.data.languages)) != len(
        config.data.languages
    ):
        raise ValueError(f"data.languages must be unique values from {LANGUAGES}.")
    if config.data.split_sizes is None:
        config.data.split_sizes = dict(SPLIT_SIZES)
    if set(config.data.split_sizes) != set(SPLIT_SIZES):
        raise ValueError(f"data.split_sizes must contain exactly {sorted(SPLIT_SIZES)}.")
    if any(not isinstance(size, int) or size < 1 for size in config.data.split_sizes.values()):
        raise ValueError("Every data.split_sizes value must be a positive integer.")
    if config.model.num_labels != 3:
        raise ValueError("model.num_labels must be 3 for Negative/Neutral/Positive classification.")
    if not 1 <= config.model.max_length <= 512:
        raise ValueError("model.max_length must be between 1 and 512.")
    if not config.model.name.strip():
        raise ValueError("model.name must not be empty.")
    if config.model.peft not in {None, "lora"}:
        raise ValueError("model.peft must be null or 'lora'.")
    if config.model.lora_rank < 1 or config.model.lora_alpha < 1:
        raise ValueError("model.lora_rank and model.lora_alpha must be positive.")
    if not 0 <= config.model.lora_dropout < 1:
        raise ValueError("model.lora_dropout must be in [0, 1).")
    if config.model.lora_target_modules is not None and not config.model.lora_target_modules:
        raise ValueError("model.lora_target_modules must not be an empty list.")
    if config.train.seed < 0:
        raise ValueError("train.seed must be non-negative.")
    if config.train.learning_rate <= 0 or config.train.weight_decay < 0:
        raise ValueError("train.learning_rate must be positive and weight_decay non-negative.")
    if config.train.batch_size < 1 or config.train.epochs < 1:
        raise ValueError("train.batch_size and train.epochs must be positive integers.")
    if not 0 <= config.train.warmup_ratio <= 1:
        raise ValueError("train.warmup_ratio must be between 0 and 1.")
    if config.train.gradient_accumulation_steps < 1 or config.train.max_grad_norm <= 0:
        raise ValueError("Gradient accumulation must be positive and max_grad_norm must be > 0.")
    if config.train.early_stopping_patience < 0:
        raise ValueError("train.early_stopping_patience must be non-negative.")
    if config.train.mixed_precision not in {"auto", "none", "bf16", "fp16"}:
        raise ValueError("train.mixed_precision must be one of: auto, none, bf16, fp16.")
    if config.train.class_weights not in {"off", "balanced"}:
        raise ValueError("train.class_weights must be 'off' or 'balanced'.")
    if not config.output_dir.strip():
        raise ValueError("output_dir must not be empty.")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load and validate a strict YAML run configuration with optional overrides."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read config file '{config_path}': {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("The config file must contain a YAML mapping.")

    raw = dict(loaded)
    if overrides:
        _deep_merge(raw, overrides)
    valid_top = {"data", "model", "train", "output_dir", "run_name"}
    unknown = sorted(set(raw) - valid_top)
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {', '.join(unknown)}")

    config = RunConfig(
        data=_construct_section(DataConfig, raw.get("data"), "data"),
        model=_construct_section(ModelConfig, raw.get("model"), "model"),
        train=_construct_section(TrainConfig, raw.get("train"), "train"),
        output_dir=raw.get("output_dir", "runs"),
        run_name=raw.get("run_name"),
    )
    _validate(config)
    return config


def config_dict(config: RunConfig) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of a run configuration."""
    return asdict(config)
