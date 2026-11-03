"""Pooler-like classifier used for cached frozen-encoder features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from sentiment.io_utils import read_json, write_json


class ProbeHead(nn.Module):
    """A tanh pooler, dropout, and three-class linear classifier."""

    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or num_labels != 3 or not 0 <= dropout < 1:
            raise ValueError(
                "ProbeHead requires hidden_size > 0, num_labels=3, and dropout in [0, 1)."
            )
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map final-layer first-token features to class logits."""
        if features.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected feature width {self.hidden_size}, got {features.shape[-1]}."
            )
        pooled = self.activation(self.pooler(features))
        return self.classifier(self.dropout(pooled))

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        encoder_name: str,
        max_length: int,
        use_lang_tag: bool = True,
    ) -> None:
        """Save safe tensor weights and metadata needed to reconstruct this probe head."""
        if max_length < 1 or not encoder_name:
            raise ValueError("encoder_name must be non-empty and max_length must be positive.")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        weights = {
            key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()
        }
        save_file(weights, target / "probe_head.safetensors")
        write_json(
            target / "probe_head.json",
            {
                "encoder_name": encoder_name,
                "hidden_size": self.hidden_size,
                "num_labels": self.num_labels,
                "max_length": max_length,
                "use_lang_tag": use_lang_tag,
                "dropout": self.dropout.p,
            },
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> tuple[ProbeHead, dict[str, Any]]:
        """Load a saved probe head and return it with its encoder metadata."""
        source = Path(directory)
        metadata = read_json(source / "probe_head.json")
        required = {"encoder_name", "hidden_size", "num_labels", "max_length", "dropout"}
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"Probe metadata is missing fields: {', '.join(sorted(missing))}.")
        head = cls(
            hidden_size=int(metadata["hidden_size"]),
            num_labels=int(metadata["num_labels"]),
            dropout=float(metadata["dropout"]),
        )
        state = load_file(source / "probe_head.safetensors", device="cpu")
        head.load_state_dict(state, strict=True)
        return head, metadata
