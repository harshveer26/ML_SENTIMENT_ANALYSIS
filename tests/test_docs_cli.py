"""Documentation command references and report provenance checks."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documented_sentiment_commands_exist_in_cli_help() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "sentiment", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    docs = "\n".join(
        (ROOT / filename).read_text(encoding="utf-8")
        for filename in ("README.md", "docs/GPU_RUNBOOK.md")
    )
    commands = set(re.findall(r"python -m sentiment ([a-z][a-z-]+)", docs))
    assert commands
    for command in commands:
        subcommand_help = subprocess.run(
            [sys.executable, "-m", "sentiment", command, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert subcommand_help.returncode == 0, command
    assert all(command in help_result.stdout for command in commands)
    assert "check_gpu_env.py" in (ROOT / "docs/GPU_RUNBOOK.md").read_text(encoding="utf-8")


def test_report_decimals_are_historical_reference_values() -> None:
    reference = json.loads(
        (ROOT / "reports/reference/recorded_baseline.json").read_text(encoding="utf-8")
    )
    assert reference["source"] == "recorded from earlier project; not reproduced in this workspace"
    allowed = {f"{reference['accuracy']:.4f}", f"{reference['macro_f1']:.4f}"}
    allowed.update(f"{reference['majority_class_accuracy']:.4f}" for _ in [0])
    for values in reference["per_class"].values():
        allowed.update(f"{values[key]:.3f}" for key in ("precision", "recall", "f1"))
    report = (ROOT / "reports/ieee/main.tex").read_text(encoding="utf-8")
    assert set(re.findall(r"(?<!\d)0\.\d+", report)) <= allowed


def test_report_table_generator_marks_absent_results_pending(tmp_path, monkeypatch) -> None:
    import importlib.util

    module_path = ROOT / "reports/make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "COMPARISON", tmp_path / "missing.json")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "generated_tables.tex")
    module.main()
    generated = (tmp_path / "generated_tables.tex").read_text(encoding="utf-8")
    assert "PENDING" in generated
