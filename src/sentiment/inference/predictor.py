"""Loadable prediction wrapper for full, LoRA, probe, and majority checkpoints."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from sentiment.constants import ID2LABEL, LABELS
from sentiment.data.dataset import format_text
from sentiment.models.factory import load_classifier
from sentiment.models.probe_head import ProbeHead
from sentiment.safeguards import assert_finite_array, assert_finite_model
from sentiment.training.probe import MajorityClassifier


class _ProbeEvaluationModel(nn.Module):
    """Expose a probe encoder/head pair through the sequence-classifier interface."""

    def __init__(self, encoder: nn.Module, head: ProbeHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids, attention_mask, labels=None):
        from types import SimpleNamespace

        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden[:, 0, :].float())
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


LOGGER = logging.getLogger(__name__)


class Predictor:
    """Predict three-class sentiment from a saved run or checkpoint directory."""

    def __init__(
        self,
        *,
        kind: str,
        device: str | torch.device,
        tokenizer: Any = None,
        model: torch.nn.Module | None = None,
        encoder: torch.nn.Module | None = None,
        probe_head: ProbeHead | None = None,
        majority_label: int | None = None,
        max_length: int = 128,
        use_lang_tag: bool = False,
    ) -> None:
        if kind not in {"full", "lora", "probe", "majority"}:
            raise ValueError(f"Unsupported saved model kind: {kind}.")
        self.kind = kind
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval() if model is not None else None
        self.encoder = encoder.to(self.device).eval() if encoder is not None else None
        self.probe_head = probe_head.to(self.device).eval() if probe_head is not None else None
        self.majority_label = majority_label
        self.max_length = max_length
        self.use_lang_tag = use_lang_tag

    @classmethod
    def from_run(
        cls,
        run_dir_or_checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        tokenizer: Any = None,
    ) -> Predictor:
        """Detect checkpoint kind and load all needed files without downloading model weights."""
        path = Path(run_dir_or_checkpoint_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Run/checkpoint directory does not exist: {path}")
        run_root = path
        best_dir = path / "best_model"
        checkpoint_dir = best_dir if best_dir.is_dir() else path
        if best_dir.is_dir() or (path / "run_status.json").is_file():
            run_root = path
        else:
            run_root = path.parent
        invalid_markers = [
            candidate
            for candidate in (
                path / "INVALID.md",
                path / "INVALID",
                run_root / "INVALID.md",
                run_root / "INVALID",
                checkpoint_dir / "INVALID.md",
                checkpoint_dir / "INVALID",
            )
            if candidate.exists()
        ]
        if invalid_markers:
            raise ValueError(
                f"Refusing invalid checkpoint {path}: marker {invalid_markers[0].name} is present."
            )

        status_path = run_root / "run_status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        )
        run_config_path = run_root / "config.json"
        run_config = (
            json.loads(run_config_path.read_text(encoding="utf-8"))
            if run_config_path.is_file()
            else {}
        )
        model_config = run_config.get("model", {}) if isinstance(run_config, dict) else {}
        max_length = int(model_config.get("max_length", 128))
        use_lang_tag = bool(model_config.get("use_lang_tag", False))

        if (checkpoint_dir / "probe_head.json").is_file():
            kind = "probe"
        elif (checkpoint_dir / "adapter_config.json").is_file():
            kind = "lora"
        elif (checkpoint_dir / "majority.json").is_file() or status.get("kind") == "majority":
            kind = "majority"
        else:
            kind = status.get("kind", "full")

        if kind == "majority":
            majority_path = checkpoint_dir / "majority.json"
            if not majority_path.is_file():
                majority_path = run_root / "majority.json"
            if majority_path.is_file():
                majority_label = int(json.loads(majority_path.read_text(encoding="utf-8"))["label"])
            elif "majority_label" in status:
                majority_label = int(status["majority_label"])
            else:
                raise ValueError(f"Majority run metadata is missing a class id: {run_root}")
            if majority_label not in range(3):
                raise ValueError(f"Invalid majority class id: {majority_label}")
            return cls(
                kind=kind,
                device=device,
                majority_label=majority_label,
                max_length=max_length,
                use_lang_tag=use_lang_tag,
            )

        if kind == "probe":
            probe_head, probe_metadata = ProbeHead.from_pretrained(checkpoint_dir)
            encoder_name = str(probe_metadata["encoder_name"])
            use_lang_tag = bool(probe_metadata.get("use_lang_tag", use_lang_tag))
            max_length = int(probe_metadata["max_length"])
            if tokenizer is None:
                tokenizer = cls._load_tokenizer(checkpoint_dir, run_root, encoder_name)
            try:
                encoder = AutoModel.from_pretrained(encoder_name, local_files_only=True)
            except OSError as exc:
                raise OSError(
                    f"Probe encoder weights are not available locally for {encoder_name!r}; "
                    "copy the trained run and its base model cache or provide local weights."
                ) from exc
            assert_finite_model(encoder)
            assert_finite_model(probe_head)
            return cls(
                kind=kind,
                device=device,
                tokenizer=tokenizer,
                encoder=encoder,
                probe_head=probe_head,
                max_length=max_length,
                use_lang_tag=use_lang_tag,
            )

        if kind == "lora":
            adapter_meta_path = checkpoint_dir / "adapter_metadata.json"
            adapter_meta = (
                json.loads(adapter_meta_path.read_text(encoding="utf-8"))
                if adapter_meta_path.is_file()
                else {}
            )
            adapter_config = json.loads(
                (checkpoint_dir / "adapter_config.json").read_text(encoding="utf-8")
            )
            base_name = adapter_meta.get("base_model") or adapter_config.get(
                "base_model_name_or_path"
            )
            if not base_name:
                raise ValueError("LoRA adapter metadata is missing base_model_name_or_path.")
            if tokenizer is None:
                tokenizer = cls._load_tokenizer(checkpoint_dir, run_root, str(base_name))
            try:
                base_model = load_classifier(base_name, local_files_only=True)
                from peft import PeftModel

                model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)
            except (OSError, ValueError) as exc:
                raise OSError(
                    f"Could not load the local base model for this LoRA adapter: {base_name}. "
                    "Full model weights are not downloaded by inference."
                ) from exc
            assert_finite_model(model)
            return cls(
                kind=kind,
                device=device,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                use_lang_tag=use_lang_tag,
            )

        if kind != "full":
            raise ValueError(f"Unsupported checkpoint metadata kind: {kind!r}.")
        if tokenizer is None:
            tokenizer = cls._load_tokenizer(checkpoint_dir, run_root, str(checkpoint_dir))
        model = load_classifier(checkpoint_dir, local_files_only=True)
        assert_finite_model(model)
        return cls(
            kind="full",
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            use_lang_tag=use_lang_tag,
        )

    @staticmethod
    def _load_tokenizer(checkpoint_dir: Path, run_root: Path, fallback_name: str) -> Any:
        for directory in (checkpoint_dir, run_root / "tokenizer"):
            if directory.is_dir() and any(
                (directory / filename).is_file()
                for filename in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "spm.model",
                    "sentencepiece.bpe.model",
                )
            ):
                return AutoTokenizer.from_pretrained(directory, local_files_only=True)
        return AutoTokenizer.from_pretrained(fallback_name, local_files_only=True)

    def _format(self, text: str, language: str | None) -> str:
        selected_language = "und" if language is None else language
        return format_text(
            text,
            selected_language,
            self.use_lang_tag,
            allow_und=True,
        )

    def evaluation_model(self) -> torch.nn.Module:
        """Return a classifier-shaped module for the shared evaluator."""
        if self.kind in {"full", "lora"}:
            return self.model
        if self.kind == "probe":
            return _ProbeEvaluationModel(self.encoder, self.probe_head)
        return MajorityClassifier(self.majority_label)

    def predict(self, text: str, language: str | None = None) -> dict[str, Any]:
        """Predict one review and return label, confidence, probabilities, and logits."""
        return self.predict_batch([text], languages=[language], batch_size=1)[0]

    def predict_batch(
        self,
        texts: Sequence[str],
        languages: Sequence[str | None] | None = None,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Predict a batch of reviews with dynamic padding and stable input ordering."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings.")
        selected_languages = list(languages) if languages is not None else [None] * len(texts)
        if len(selected_languages) != len(texts):
            raise ValueError("languages must have the same length as texts.")
        if not texts:
            return []

        outputs: list[dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_languages = selected_languages[start : start + batch_size]
            if self.kind == "majority":
                logits = torch.zeros((len(chunk_texts), len(LABELS)), dtype=torch.float32)
                logits[:, self.majority_label] = 1.0
            else:
                formatted = [
                    self._format(text, language)
                    for text, language in zip(chunk_texts, chunk_languages, strict=True)
                ]
                encoded = self.tokenizer(
                    formatted,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    if self.kind == "probe":
                        hidden = self.encoder(**encoded).last_hidden_state
                        features = hidden[:, 0, :].float()
                        logits = self.probe_head(features)
                    else:
                        logits = self.model(**encoded).logits
            logits = logits.float()
            assert_finite_array(logits.detach().cpu().numpy(), "prediction logits")
            probabilities = torch.softmax(logits, dim=-1)
            assert_finite_array(probabilities.detach().cpu().numpy(), "prediction probabilities")
            probs = probabilities.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            for row_index, probability in enumerate(probs):
                class_id = int(np.argmax(probability))
                outputs.append(
                    {
                        "label": ID2LABEL[class_id],
                        "class_id": class_id,
                        "confidence": float(probability[class_id]),
                        "probabilities": {
                            LABELS[index]: float(probability[index]) for index in range(len(LABELS))
                        },
                        "logits": [float(value) for value in logits_np[row_index]],
                    }
                )
        return outputs
