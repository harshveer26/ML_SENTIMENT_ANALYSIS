"""Audit saved checkpoint tensors and mark invalid artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from sentiment.io_utils import write_json
from sentiment.safeguards import NonFiniteError, assert_finite_tensor

LOGGER = logging.getLogger(__name__)
_WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth"}


def _tensors_from_file(path: Path) -> list[tuple[str, torch.Tensor]]:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("Install safetensors to audit this checkpoint.") from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return [(key, handle.get_tensor(key)) for key in handle.keys()]

    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch versions do not expose weights_only
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a tensor mapping in {path}.")
    return [
        (str(name), value) for name, value in payload.items() if isinstance(value, torch.Tensor)
    ]


def audit_checkpoint(directory: str | Path) -> int:
    """Audit supported weights under a directory; write invalid markers and return 1 on failure."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
    )
    problems: list[str] = []
    if not files:
        problems.append("No .safetensors, .pt, or .pth weight files were found.")
    for path in files:
        try:
            tensors = _tensors_from_file(path)
            if not tensors:
                problems.append(f"{path.relative_to(root)} contains no tensors.")
            for name, tensor in tensors:
                try:
                    assert_finite_tensor(tensor, f"{path.name}:{name}")
                except NonFiniteError as exc:
                    problems.append(str(exc))
        except Exception as exc:  # malformed or unreadable weights are not safe to use
            problems.append(f"Could not audit {path.relative_to(root)}: {exc}")

    if problems:
        marker = "# INVALID checkpoint\n\nThis checkpoint failed the finite-weight audit.\n\n"
        marker += "Problems:\n" + "\n".join(f"- {problem}" for problem in problems) + "\n"
        (root / "INVALID.md").write_text(marker, encoding="utf-8")
        write_json(root / "invalid_metrics.json", {"valid": False, "problems": problems})
        LOGGER.error("Checkpoint audit failed for %s", root)
        return 1
    LOGGER.info("Checkpoint audit passed for %s (%d weight files)", root, len(files))
    return 0


def audit_all(root: str | Path) -> int:
    """Audit every directory containing supported checkpoint files under root."""
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Audit root does not exist: {directory}")
    checkpoints = sorted(
        {
            path.parent
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
        }
    )
    if not checkpoints:
        LOGGER.warning("No checkpoint weight files found under %s", directory)
        return 0
    statuses = [audit_checkpoint(checkpoint) for checkpoint in checkpoints]
    return int(any(status != 0 for status in statuses))
