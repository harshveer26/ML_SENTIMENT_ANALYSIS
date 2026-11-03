"""Project command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("sentiment")


def _add_example(parser: argparse.ArgumentParser, example: str) -> None:
    parser.epilog = f"Example: {example}"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter


def _parser() -> argparse.ArgumentParser:
    from sentiment.constants import LANGUAGES

    parser = argparse.ArgumentParser(
        prog="sentiment", description="Multilingual review sentiment tools"
    )
    parser.add_argument("--version", action="version", version="sentiment 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and normalize source data")
    download.add_argument("--languages", nargs="+", default=LANGUAGES)
    download.add_argument("--out", default="data/amazon_reviews_multi")
    download.add_argument("--force", action="store_true")
    _add_example(download, "python -m sentiment download --languages en fr zh")

    prepare = commands.add_parser("prepare", help="Sample the reproducible benchmark")
    prepare.add_argument("--config", default="configs/benchmark.yaml")
    _add_example(prepare, "python -m sentiment prepare --config configs/benchmark.yaml")

    validate = commands.add_parser(
        "validate-data", help="Validate benchmark files and split isolation"
    )
    validate.add_argument("--data-dir", default="data/amazon_reviews_benchmark")
    _add_example(validate, "python -m sentiment validate-data")

    probe = commands.add_parser("probe", help="Train the frozen-encoder feature probe")
    probe.add_argument("--config", default="configs/probe_mdeberta.yaml")
    probe.add_argument("--dry-run", action="store_true")
    _add_example(probe, "python -m sentiment probe --config configs/probe_mdeberta.yaml --dry-run")

    train = commands.add_parser("train", help="Fine-tune a multilingual classifier")
    train.add_argument("--config", default="configs/finetune_mdeberta.yaml")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--class-weights", choices=["balanced"])
    train.add_argument("--peft", choices=["lora"])
    _add_example(
        train, "python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run"
    )

    majority = commands.add_parser("majority", help="Create the train-majority baseline run")
    majority.add_argument("--data-dir", default="data/amazon_reviews_benchmark")
    majority.add_argument("--out", default="runs")
    _add_example(majority, "python -m sentiment majority")

    evaluate = commands.add_parser("evaluate", help="Evaluate a selected run on validation or test")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], required=True)
    evaluate.add_argument("--force", action="store_true")
    _add_example(evaluate, "python -m sentiment evaluate --run runs/example --split validation")

    predict = commands.add_parser("predict", help="Predict sentiment for one review")
    predict.add_argument("--run", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--lang", default=None)
    _add_example(
        predict, 'python -m sentiment predict --run runs/example --text "Great product" --lang en'
    )

    predict_batch = commands.add_parser("predict-batch", help="Predict reviews from a JSONL file")
    predict_batch.add_argument("--run", required=True)
    predict_batch.add_argument("--file", required=True)
    predict_batch.add_argument("--batch-size", type=int, default=32)
    _add_example(
        predict_batch, "python -m sentiment predict-batch --run runs/example --file texts.jsonl"
    )

    compare = commands.add_parser("compare", help="Compare available final test evaluations")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--out", default="reports/results")
    compare.add_argument("--data-dir", default="data/amazon_reviews_benchmark")
    _add_example(compare, "python -m sentiment compare --runs runs/a runs/b --out reports/results")

    audit = commands.add_parser("audit", help="Audit checkpoints for non-finite weights")
    audit.add_argument("--root", required=True)
    _add_example(audit, "python -m sentiment audit --root checkpoints")

    env = commands.add_parser("env", help="Print detected runtime environment information")
    _add_example(env, "python -m sentiment env")
    return parser


def _load_resources(config: Any, *, dry_run: bool = False) -> tuple[Any, Any]:
    from transformers import AutoTokenizer

    from sentiment.models.factory import load_classifier, resolve_backbone

    model_name = resolve_backbone(config.model.name)
    local_only = bool(config.model.local_files_only)
    if dry_run and os.environ.get("RUN_HEAVY_TESTS") != "1":
        # A dry-run may reuse a local real checkpoint, but must not download full weights.
        local_only = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=config.model.local_files_only,
    )
    model = load_classifier(
        model_name,
        num_labels=config.model.num_labels,
        local_files_only=local_only,
    )
    return model, tokenizer


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _verify_benchmark(data_dir: str | Path) -> tuple[dict[str, Any], str]:
    from sentiment.data.validate import validate_benchmark
    from sentiment.io_utils import read_json, sha256_file

    root = Path(data_dir)
    validate_benchmark(root)
    manifest_path = root / "manifest.json"
    return read_json(manifest_path), sha256_file(manifest_path)


def _verify_run_manifest(run_dir: Path, manifest_hash: str) -> None:
    from sentiment.io_utils import read_json

    reference_path = run_dir / "dataset_manifest_ref.json"
    if not reference_path.is_file():
        raise ValueError(f"Run is missing its dataset manifest reference: {reference_path}")
    reference = read_json(reference_path)
    expected = reference.get("sha256")
    if expected != manifest_hash:
        raise ValueError(
            f"Run dataset manifest hash {expected!r} does not match current benchmark "
            f"{manifest_hash!r}."
        )


def _slug(value: str) -> str:
    candidate = Path(value).name if ":" in value or "\\" in value else value.split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate).strip("-.")
    return slug or "model"


def _create_run(
    output_root: str | Path,
    *,
    backbone: str,
    mode: str,
    seed: int,
    config: dict[str, Any],
    manifest_hash: str,
) -> Path:
    from sentiment.env_info import capture_env
    from sentiment.io_utils import write_json

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{_slug(backbone)}-{mode}-s{seed}-{stamp}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "env.json", capture_env())
    write_json(run_dir / "dataset_manifest_ref.json", {"sha256": manifest_hash})
    write_json(run_dir / "run_status.json", {"kind": mode, "selection_complete": False})
    return run_dir


def _make_loader(
    records: list[dict[str, Any]], tokenizer: Any, config: Any, *, shuffle: bool, seed: int
):
    from sentiment.data.collate import DynamicPaddingCollator
    from sentiment.data.dataset import SentimentDataset, make_dataloader

    dataset = SentimentDataset(
        records,
        tokenizer,
        max_length=config.model.max_length,
        use_lang_tag=config.model.use_lang_tag,
    )
    return make_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        seed=seed,
        collate_fn=DynamicPaddingCollator(tokenizer),
    )


def _do_download(args: argparse.Namespace) -> None:
    from sentiment.data.download import download_dataset

    download_dataset(args.languages, args.out, force=args.force)


def _do_prepare(args: argparse.Namespace) -> None:
    from sentiment.config import load_config
    from sentiment.data.prepare import prepare_benchmark

    config = load_config(args.config)
    prepare_benchmark(
        config.data.source_dir,
        config.data.benchmark_dir,
        languages=config.data.languages,
        seed=config.data.seed,
        split_sizes=config.data.split_sizes,
    )


def _do_validate(args: argparse.Namespace) -> None:
    _verify_benchmark(args.data_dir)


def _do_train(args: argparse.Namespace) -> None:
    from sentiment.config import config_dict, load_config
    from sentiment.io_utils import read_jsonl
    from sentiment.training.finetune import train_model

    config = load_config(args.config)
    if args.class_weights:
        config.train.class_weights = args.class_weights
    if args.peft:
        config.model.peft = args.peft
    _, manifest_hash = _verify_benchmark(config.data.benchmark_dir)
    data_root = Path(config.data.benchmark_dir)
    model, tokenizer = _load_resources(config, dry_run=args.dry_run)
    train_loader = _make_loader(
        read_jsonl(data_root / "train.jsonl"),
        tokenizer,
        config,
        shuffle=True,
        seed=config.train.seed,
    )
    validation_loader = _make_loader(
        read_jsonl(data_root / "validation.jsonl"),
        tokenizer,
        config,
        shuffle=False,
        seed=config.train.seed,
    )
    if args.dry_run:
        result = train_model(
            model,
            train_loader,
            validation_loader,
            Path(config.output_dir) / "dry-run-unused",
            config=config.train,
            tokenizer=tokenizer,
            model_identifier=config.model.name,
            device=_device(),
            class_weights="balanced" if args.class_weights else None,
            peft_mode=config.model.peft,
            lora_target_modules=config.model.lora_target_modules,
            dry_run=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    run_config = config_dict(config)
    mode = "lora" if config.model.peft == "lora" else "full"
    if args.resume:
        if not config.run_name:
            raise ValueError(
                "--resume requires config.run_name to identify the existing run folder."
            )
        run_dir = Path(config.output_dir) / config.run_name
        if not run_dir.is_dir() or not (run_dir / "last_checkpoint.pt").is_file():
            raise FileNotFoundError(f"No resumable checkpoint was found under {run_dir}.")
        _verify_run_manifest(run_dir, manifest_hash)
    else:
        run_dir = _create_run(
            config.output_dir,
            backbone=config.model.name,
            mode=mode,
            seed=config.train.seed,
            config=run_config,
            manifest_hash=manifest_hash,
        )
    status = train_model(
        model,
        train_loader,
        validation_loader,
        run_dir,
        config=config.train,
        tokenizer=tokenizer,
        model_identifier=config.model.name,
        device=_device(),
        class_weights="balanced" if args.class_weights else None,
        peft_mode=config.model.peft,
        lora_target_modules=config.model.lora_target_modules,
        resume=args.resume,
    )
    LOGGER.info("Training complete: %s (best epoch %s)", run_dir, status["best_epoch"])


def _do_probe(args: argparse.Namespace) -> None:
    import torch

    from sentiment.config import config_dict, load_config
    from sentiment.io_utils import read_jsonl, sha256_file
    from sentiment.safeguards import assert_finite_tensor
    from sentiment.training.probe import (
        ProbeTrainConfig,
        extract_features,
        train_probe,
    )

    config = load_config(args.config)
    _, manifest_hash = _verify_benchmark(config.data.benchmark_dir)
    data_root = Path(config.data.benchmark_dir)
    model, tokenizer = _load_resources(config, dry_run=args.dry_run)
    if args.dry_run:
        train_records = read_jsonl(data_root / "train.jsonl")
        train_loader = _make_loader(
            train_records, tokenizer, config, shuffle=False, seed=config.train.seed
        )
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"][:2].to(_device())
        attention_mask = batch["attention_mask"][:2].to(_device())
        model.to(_device()).eval()
        with torch.no_grad():
            features = model.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        assert_finite_tensor(features, "probe dry-run features")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "examples_forwarded": int(features.shape[0]),
                    "hidden_size": int(features.shape[-1]),
                    "model_identifier": config.model.name,
                    "checkpoint_written": False,
                },
                indent=2,
            )
        )
        return

    datasets = {
        split: read_jsonl(data_root / f"{split}.jsonl") for split in ("train", "validation", "test")
    }
    loaders = {
        split: _make_loader(
            datasets[split],
            tokenizer,
            config,
            shuffle=(split == "train"),
            seed=config.train.seed,
        )
        for split in ("train", "validation", "test")
    }
    run_dir = _create_run(
        config.output_dir,
        backbone=config.model.name,
        mode="probe",
        seed=config.train.seed,
        config=config_dict(config),
        manifest_hash=manifest_hash,
    )
    (run_dir / "features").mkdir()
    cached = {}
    for split, loader in loaders.items():
        cached[split] = extract_features(
            model,
            loader,
            model_name=config.model.name,
            max_length=config.model.max_length,
            use_lang_tag=config.model.use_lang_tag,
            split_file_hash=sha256_file(data_root / f"{split}.jsonl"),
            cache_path=run_dir / "features" / f"{split}.npz",
            device=_device(),
        )
    tokenizer.save_pretrained(run_dir / "tokenizer")
    settings = ProbeTrainConfig(
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        batch_size=config.train.batch_size,
        epochs=min(config.train.epochs, 15),
        patience=config.train.early_stopping_patience,
        seed=config.train.seed,
    )
    status = train_probe(
        cached,
        run_dir,
        encoder_name=config.model.name,
        max_length=config.model.max_length,
        use_lang_tag=config.model.use_lang_tag,
        config=settings,
    )
    LOGGER.info("Probe complete: %s (best epoch %s)", run_dir, status["best_epoch"])


def _do_majority(args: argparse.Namespace) -> None:
    from collections import Counter

    from sentiment.config import DataConfig, ModelConfig, RunConfig, TrainConfig, config_dict
    from sentiment.io_utils import read_jsonl, write_json

    _, manifest_hash = _verify_benchmark(args.data_dir)
    train_records = read_jsonl(Path(args.data_dir) / "train.jsonl")
    counts = Counter(int(record["label"]) for record in train_records)
    label = min(counts, key=lambda item: (-counts[item], item))
    run_config = config_dict(
        RunConfig(
            data=DataConfig(benchmark_dir=args.data_dir),
            model=ModelConfig(name="majority"),
            train=TrainConfig(),
            output_dir=args.out,
        )
    )
    run_dir = _create_run(
        args.out,
        backbone="majority",
        mode="majority",
        seed=run_config["train"]["seed"],
        config=run_config,
        manifest_hash=manifest_hash,
    )
    write_json(run_dir / "majority.json", {"label": label, "training_counts": dict(counts)})
    write_json(
        run_dir / "run_status.json",
        {"kind": "majority", "selection_complete": True, "majority_label": label},
    )
    LOGGER.info("Majority baseline run created: %s (label %s)", run_dir, label)


def _test_lock_precheck(run_dir: Path, force: bool) -> None:
    from sentiment.io_utils import read_json
    from sentiment.safeguards import TestSetLockedError

    status_path = run_dir / "run_status.json"
    if not status_path.is_file() or not read_json(status_path).get("selection_complete", False):
        raise TestSetLockedError(
            "Test evaluation is locked until checkpoint selection is complete."
        )
    if (run_dir / "test_evaluated.lock").exists() and not force:
        raise TestSetLockedError(
            "Test evaluation has already been run; pass --force to record another evaluation."
        )


def _do_evaluate(args: argparse.Namespace) -> None:
    from sentiment.data.dataset import make_dataloader
    from sentiment.evaluation.evaluate import evaluate_model
    from sentiment.inference.predictor import Predictor
    from sentiment.io_utils import read_json, read_jsonl

    run_dir = Path(args.run)
    if args.split == "test":
        _test_lock_precheck(run_dir, args.force)
    run_config_path = run_dir / "config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Run config is missing: {run_config_path}")
    raw_config = read_json(run_config_path)
    # Parse the run's JSON snapshot through the same strict dataclasses.
    from sentiment.config import DataConfig, ModelConfig, RunConfig, TrainConfig

    config = RunConfig(
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        train=TrainConfig(**raw_config["train"]),
        output_dir=raw_config.get("output_dir", "runs"),
        run_name=raw_config.get("run_name"),
    )
    _, manifest_hash = _verify_benchmark(config.data.benchmark_dir)
    _verify_run_manifest(run_dir, manifest_hash)
    predictor = Predictor.from_run(run_dir, device=_device())
    records = read_jsonl(Path(config.data.benchmark_dir) / f"{args.split}.jsonl")
    from sentiment.data.collate import DynamicPaddingCollator
    from sentiment.data.dataset import SentimentDataset

    dataset = SentimentDataset(
        records,
        predictor.tokenizer,
        max_length=predictor.max_length,
        use_lang_tag=predictor.use_lang_tag,
    )
    loader = make_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        seed=config.train.seed,
        collate_fn=DynamicPaddingCollator(predictor.tokenizer),
    )
    result = evaluate_model(
        predictor.evaluation_model(),
        loader,
        _device(),
        config.train.mixed_precision,
        run_dir=run_dir,
        split=args.split,
        metadata={
            "dataset_manifest_hash": manifest_hash,
            "seed": config.train.seed,
            "model_identifier": config.model.name,
            "config": raw_config,
        },
        force=args.force,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


def _do_predict(args: argparse.Namespace) -> None:
    from sentiment.inference.predictor import Predictor

    predictor = Predictor.from_run(args.run, device=_device())
    print(json.dumps(predictor.predict(args.text, args.lang), ensure_ascii=False, indent=2))


def _do_predict_batch(args: argparse.Namespace) -> None:
    from sentiment.inference.predictor import Predictor
    from sentiment.io_utils import read_jsonl

    records = read_jsonl(args.file)
    texts = [record.get("text") for record in records]
    languages = [record.get("language", record.get("lang")) for record in records]
    predictor = Predictor.from_run(args.run, device=_device())
    predictions = predictor.predict_batch(texts, languages=languages, batch_size=args.batch_size)
    output = [
        {"sample_id": record.get("sample_id"), **prediction}
        for record, prediction in zip(records, predictions, strict=True)
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _do_compare(args: argparse.Namespace) -> None:
    from sentiment.evaluation.compare import compare_runs

    result = compare_runs(args.runs, args.out, benchmark_dir=args.data_dir)
    LOGGER.info("Compared %d runs", len(result["runs"]))


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "download":
        _do_download(args)
    elif args.command == "prepare":
        _do_prepare(args)
    elif args.command == "validate-data":
        _do_validate(args)
    elif args.command == "probe":
        _do_probe(args)
    elif args.command == "train":
        _do_train(args)
    elif args.command == "majority":
        _do_majority(args)
    elif args.command == "evaluate":
        _do_evaluate(args)
    elif args.command == "predict":
        _do_predict(args)
    elif args.command == "predict-batch":
        _do_predict_batch(args)
    elif args.command == "compare":
        _do_compare(args)
    elif args.command == "audit":
        from sentiment.audit import audit_all

        raise SystemExit(audit_all(args.root))
    elif args.command == "env":
        from sentiment.env_info import capture_env

        print(json.dumps(capture_env(), ensure_ascii=False, indent=2))


def main() -> None:
    """Parse arguments and run a sentiment pipeline command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _parser()
    args = parser.parse_args()
    try:
        _dispatch(args)
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
