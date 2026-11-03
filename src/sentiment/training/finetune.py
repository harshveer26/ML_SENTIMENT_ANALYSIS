"""Deterministic full fine-tuning, optional LoRA, resume, and dry-run execution."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sentiment.config import TrainConfig
from sentiment.evaluation.evaluate import evaluate_model
from sentiment.io_utils import read_jsonl, write_json, write_jsonl
from sentiment.models.factory import (
    count_trainable,
    mixed_precision_policy,
    resolve_backbone,
)
from sentiment.safeguards import NonFiniteError, assert_finite_model, assert_finite_tensor
from sentiment.training.optim import (
    balanced_class_weights,
    build_adamw,
    build_linear_scheduler,
    make_grad_scaler,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_LORA_TARGETS = {
    "mdeberta": ["query_proj", "value_proj"],
    "xlmr": ["query", "value"],
}


def prepare_lora(
    model: torch.nn.Module,
    backbone: str,
    *,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """Wrap a classifier in sequence-classification LoRA after validating target names."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA requires peft; install requirements-qlora.txt or requirements.txt."
        ) from exc
    key = backbone.lower()
    targets = target_modules or _DEFAULT_LORA_TARGETS.get(key)
    if not targets:
        raise ValueError(f"No default LoRA targets are configured for backbone '{backbone}'.")
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("LoRA rank and alpha must be positive; dropout must be in [0, 1).")
    module_names = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(targets) - module_names)
    if missing:
        raise ValueError(
            f"LoRA target module(s) {missing} were not found in backbone '{backbone}'. "
            f"Available leaf names include: {sorted(module_names)[:30]}"
        )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        modules_to_save=["classifier", "pooler"],
    )
    wrapped = get_peft_model(model, lora_config)
    return wrapped


def _batch_for_model(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "labels")
        if key in batch
    }


def _fp32_parameters(model: torch.nn.Module) -> bool:
    return all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for parameter in model.parameters()
    )


def _dry_run(model: torch.nn.Module, train_loader: Any, device: torch.device) -> dict[str, Any]:
    try:
        batch = next(iter(train_loader))
    except StopIteration as exc:
        raise ValueError("Dry-run training dataloader has no examples.") from exc
    limited: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            limited[key] = value[:2]
        elif isinstance(value, list):
            limited[key] = value[:2]
    inputs = _batch_for_model(limited, device)
    model.eval()
    with torch.no_grad():
        outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    assert_finite_tensor(logits, "dry-run logits")
    counts = count_trainable(model)
    return {
        "dry_run": True,
        "examples_forwarded": int(logits.shape[0]),
        "parameter_counts": counts,
        "estimated_parameter_memory_bytes_fp32": counts["total"] * 4,
        "checkpoint_written": False,
    }


def _rng_state(train_loader: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        state["dataloader_generator"] = generator.get_state()
    return state


def _restore_rng(state: dict[str, Any], train_loader: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"])


def _save_last_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
    train_loader: Any,
    last_good_loss: float | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "last_good_loss": last_good_loss,
            "rng": _rng_state(train_loader),
        },
        path,
    )


def _training_labels(train_loader: Any) -> list[int]:
    dataset = getattr(train_loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "records"):
        return [int(record["label"]) for record in dataset.records]
    values: list[int] = []
    for batch in train_loader:
        values.extend(int(value) for value in batch["labels"].tolist())
    if not values:
        raise ValueError("Training dataloader has no labels.")
    return values


def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    validation_loader: Any,
    run_dir: str | Path,
    *,
    config: TrainConfig | None = None,
    tokenizer: Any = None,
    model_identifier: str = "unknown",
    device: str | torch.device = "cpu",
    class_weights: str | torch.Tensor | None = None,
    peft_mode: str | None = None,
    lora_target_modules: list[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Fine-tune on train data, choose checkpoints on validation, and never inspect test data."""
    settings = config or TrainConfig()
    if settings.batch_size < 1 or settings.epochs < 1 or settings.gradient_accumulation_steps < 1:
        raise ValueError("batch_size, epochs, and gradient_accumulation_steps must be positive.")
    if settings.learning_rate <= 0 or settings.weight_decay < 0 or settings.max_grad_norm <= 0:
        raise ValueError(
            "learning_rate/max_grad_norm must be positive and weight_decay non-negative."
        )
    if not 0 <= settings.warmup_ratio <= 1 or settings.early_stopping_patience < 0:
        raise ValueError("warmup_ratio must be in [0, 1] and early_stopping_patience non-negative.")
    if settings.class_weights not in {"off", "balanced"}:
        raise ValueError("config.class_weights must be 'off' or 'balanced'.")
    device_obj = torch.device(device)
    model = model.to(device_obj)
    if peft_mode is not None:
        if peft_mode != "lora":
            raise ValueError("peft_mode must be None or 'lora'.")
        model = prepare_lora(
            model,
            model_identifier,
            target_modules=lora_target_modules,
        )
        model = model.to(device_obj)
    if not _fp32_parameters(model):
        raise ValueError(
            "Model parameters must remain FP32 master weights before optimizer creation."
        )
    policy = mixed_precision_policy(device_obj, settings.mixed_precision)
    counts = count_trainable(model)
    if dry_run:
        return {
            **_dry_run(model, train_loader, device_obj),
            "amp_mode": policy.mode,
            "model_identifier": model_identifier,
            "config": asdict(settings),
            "estimated_memory_note": (
                "FP32 parameter storage only; activation and optimizer memory are excluded."
            ),
        }

    target = Path(run_dir)
    best_dir = target / "best_model"
    checkpoint_path = target / "last_checkpoint.pt"
    if best_dir.exists() and not resume:
        raise FileExistsError(
            f"{best_dir} already exists; use a new run directory or pass resume=True."
        )
    target.mkdir(parents=True, exist_ok=True)
    if isinstance(class_weights, str) and class_weights != "balanced":
        raise ValueError("String class_weights value must be 'balanced'.")
    if class_weights is not None and not isinstance(class_weights, (str, torch.Tensor)):
        raise ValueError("class_weights must be None, 'balanced', or a torch tensor.")
    if class_weights is None and settings.class_weights == "balanced":
        class_weights = "balanced"
    if isinstance(class_weights, str) and class_weights == "balanced":
        weights = balanced_class_weights(_training_labels(train_loader)).to(device_obj)
        class_weight_mode = "balanced"
    elif isinstance(class_weights, torch.Tensor):
        if class_weights.shape != (3,):
            raise ValueError("class_weights tensor must have shape (3,).")
        assert_finite_tensor(class_weights, "class weights")
        weights = class_weights.to(device_obj, dtype=torch.float32)
        class_weight_mode = "custom"
    else:
        weights = None
        class_weight_mode = "off"

    if not hasattr(train_loader, "__len__") or len(train_loader) < 1:
        raise ValueError("Training dataloader must be sized and non-empty.")
    total_steps = (
        math.ceil(len(train_loader) / settings.gradient_accumulation_steps) * settings.epochs
    )
    optimizer = build_adamw(model, settings.learning_rate, settings.weight_decay)
    scheduler = build_linear_scheduler(optimizer, total_steps, settings.warmup_ratio)
    scaler = make_grad_scaler(policy)
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_score = -1.0
    last_good_loss: float | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        last_good_loss = checkpoint["last_good_loss"]
        _restore_rng(checkpoint["rng"], train_loader)
        if (target / "train_log.jsonl").is_file():
            history = read_jsonl(target / "train_log.jsonl")

    patience = max(1, settings.early_stopping_patience)
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, settings.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            inputs = _batch_for_model(batch, device_obj)
            labels = inputs.pop("labels")
            amp_context = (
                torch.autocast(device_type=device_obj.type, dtype=policy.autocast_dtype)
                if policy.autocast_dtype is not None
                else torch.autocast(device_type=device_obj.type, enabled=False)
            )
            with amp_context:
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            try:
                assert_finite_tensor(loss, "training loss")
            except NonFiniteError:
                write_json(
                    target / "training_failed.json",
                    {
                        "step": global_step + 1,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "amp_mode": policy.mode,
                        "last_good_loss": last_good_loss,
                    },
                )
                write_json(
                    target / "run_status.json",
                    {"selection_complete": False, "failed": True},
                )
                raise
            last_good_loss = float(loss.detach().float().item())
            loss_sum += last_good_loss * len(labels)
            seen += len(labels)
            scaled_loss = loss / settings.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            update_now = (
                batch_index + 1
            ) % settings.gradient_accumulation_steps == 0 or batch_index + 1 == batches
            if update_now:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.max_grad_norm
                )
                try:
                    assert_finite_tensor(gradient_norm, "gradient norm")
                except NonFiniteError as exc:
                    write_json(
                        target / "training_failed.json",
                        {
                            "step": global_step + 1,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "amp_mode": policy.mode,
                            "last_good_loss": last_good_loss,
                            "reason": str(exc),
                        },
                    )
                    write_json(
                        target / "run_status.json",
                        {"selection_complete": False, "failed": True},
                    )
                    raise
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        validation = evaluate_model(
            model,
            validation_loader,
            device_obj,
            policy.mode,
            split="validation",
        )
        val_metrics = validation["metrics"]
        score = float(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": score,
            "val_per_class_recall": {
                label: values["recall"] for label, values in val_metrics["per_class"].items()
            },
            "val_neutral_recall": val_metrics["neutral_recall"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "amp_mode": policy.mode,
        }
        history.append(row)
        write_jsonl(target / "train_log.jsonl", history)
        if score > best_score:
            assert_finite_model(model)
            best_score = score
            best_epoch = epoch
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(best_dir)
            if peft_mode == "lora":
                write_json(
                    best_dir / "adapter_metadata.json",
                    {
                        "kind": "lora",
                        "base_model": resolve_backbone(model_identifier),
                        "target_modules": lora_target_modules
                        or _DEFAULT_LORA_TARGETS.get(model_identifier.lower()),
                        "rank": 8,
                        "alpha": 16,
                        "dropout": 0.1,
                        "task_type": "SEQ_CLS",
                        "modules_to_save": ["classifier", "pooler"],
                        "merge_supported": True,
                    },
                )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        completed_epoch = epoch
        _save_last_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_epoch=best_epoch,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
            train_loader=train_loader,
            last_good_loss=last_good_loss,
        )
        if epochs_without_improvement >= patience:
            break

    if not best_dir.is_dir():
        raise RuntimeError("Training ended without a validation-selected best_model checkpoint.")
    status = {
        "selection_complete": True,
        "kind": "lora" if peft_mode == "lora" else "full",
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "best_val_macro_f1": best_score,
        "class_weights": class_weight_mode,
        "peft_mode": peft_mode or "off",
        "model_identifier": model_identifier,
        "amp_mode": policy.mode,
        "parameter_counts": counts,
        "test_evaluated": False,
        "config": asdict(settings),
    }
    write_json(target / "run_status.json", status)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return status
