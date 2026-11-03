"""Runtime environment capture without hard-coded machine claims."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def capture_env() -> dict[str, Any]:
    """Capture detected Python, OS, library, CUDA, GPU, and Git information."""
    try:
        import torch
    except ImportError:
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_bytes": properties.total_memory,
                }
            )
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__ if torch else None,
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "peft": _version("peft"),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if torch and cuda_available else None,
        "gpus": devices,
        "git_commit": _git_commit(),
    }
