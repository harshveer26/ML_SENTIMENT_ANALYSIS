"""Report detected accelerator state and clearly marked memory heuristics."""

from __future__ import annotations

import json

from sentiment.env_info import capture_env


def suggestions(vram_gib: float) -> list[str]:
    """Return rough starting points, explicitly not measured guarantees."""
    if vram_gib < 8:
        return ["Heuristic: batch size 1-2, max_length 128, consider LoRA."]
    if vram_gib < 16:
        return ["Heuristic: batch size 2-4, max_length 128, consider LoRA."]
    if vram_gib < 24:
        return ["Heuristic: batch size 4-8, max_length 128; monitor memory."]
    return ["Heuristic: begin with batch size 8 and monitor memory before increasing."]


def main() -> None:
    info = capture_env()
    try:
        import torch
    except ImportError:
        torch = None
    info["bf16_supported"] = bool(
        torch and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    if not info["cuda_available"]:
        info["warning"] = "CUDA is unavailable; GPU training cannot run in this environment."
        info["suggestions"] = ["Heuristic only: use a CUDA-capable host for full fine-tuning."]
    else:
        info["suggestions"] = [
            suggestion
            for device in info["gpus"]
            for suggestion in suggestions(device["memory_bytes"] / (1024**3))
        ]
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
