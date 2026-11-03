"""Isolated QLoRA causal-LM experiment checks."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from transformers import GPT2Config, GPT2LMHeadModel


def _experiment_script() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "experiments" / "qlora_causal_lm" / "train_qlora.py"
    )


def _load_module():
    import importlib.util

    script = _experiment_script()
    spec = importlib.util.spec_from_file_location("qlora_experiment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_validation_matches_gpt2_modules() -> None:
    module = _load_module()
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2))
    module.validate_targets(model, ["c_attn"])
    with pytest.raises(ValueError, match="q_proj.*do not exist"):
        module.validate_targets(model, ["q_proj"])


def test_qlora_dry_run_and_one_cpu_training_step(tmp_path) -> None:
    script = _experiment_script()
    experiment_dir = script.parent
    base_config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    base_config["output_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
    dry_run = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "dry_run" in dry_run.stdout and "Experimental." in dry_run.stdout

    train = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--tiny-fixture",
            "--max-steps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stderr
    assert '"training_steps": 1' in train.stdout
    assert (tmp_path / "outputs" / "adapter_config.json").is_file()


def test_classifier_package_does_not_import_experiments() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "sentiment"
    for source in source_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(
            not (isinstance(node, ast.ImportFrom) and node.module and "experiment" in node.module)
            for node in imports
        ), source
