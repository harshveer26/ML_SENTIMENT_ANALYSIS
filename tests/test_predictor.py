"""Tests for full-model and majority prediction plus attention diagnostics."""

import json

import pytest

from sentiment.inference.attention import DISCLAIMER, explain_attention
from sentiment.inference.predictor import Predictor
from sentiment.models.factory import load_classifier
from sentiment.models.probe_head import ProbeHead
from sentiment.training.finetune import prepare_lora
from tests.fixtures import tiny_model_and_tokenizer


def test_single_and_batch_prediction_outputs(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "checkpoint", tiny_tokenizer)
    predictor = Predictor.from_run(model_path, tokenizer=tiny_tokenizer)
    text = "good movie"
    single = predictor.predict(text, "en")
    batch_one = predictor.predict_batch([text], languages=["en"], batch_size=1)[0]
    assert set(single) == {"label", "class_id", "confidence", "probabilities", "logits"}
    assert sum(single["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert single["confidence"] == max(single["probabilities"].values())
    assert single == batch_one


def test_predictions_are_batch_and_order_invariant(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "checkpoint", tiny_tokenizer)
    predictor = Predictor.from_run(model_path, tokenizer=tiny_tokenizer)
    texts = ["good movie", "bad review", "very nice film"]
    languages = ["en", "fr", "zh"]
    by_one = predictor.predict_batch(texts, languages, batch_size=1)
    by_all = predictor.predict_batch(texts, languages, batch_size=3)
    reversed_batch = predictor.predict_batch(
        list(reversed(texts)), list(reversed(languages)), batch_size=2
    )
    for first, second in zip(by_one, by_all, strict=True):
        assert first["class_id"] == second["class_id"]
        assert first["probabilities"] == pytest.approx(second["probabilities"], abs=1e-5)
    for first, reversed_item in zip(by_one, reversed(reversed_batch), strict=True):
        assert first["probabilities"] == pytest.approx(reversed_item["probabilities"], abs=1e-5)


def test_unlisted_multilingual_texts_are_tokenized(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "checkpoint", tiny_tokenizer)
    predictor = Predictor.from_run(model_path, tokenizer=tiny_tokenizer)
    result = predictor.predict_batch(
        ["A pleasant product", "Très bon produit", "这个产品很好", "منتج جيد", "bel prodotto"],
        ["en", "fr", "zh", "ar", "it"],
        batch_size=3,
    )
    assert len(result) == 5
    assert all(len(item["logits"]) == 3 for item in result)


def test_invalid_checkpoint_is_refused(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "invalid", tiny_tokenizer)
    (model_path / "INVALID.md").write_text("bad checkpoint", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing invalid checkpoint"):
        Predictor.from_run(model_path, tokenizer=tiny_tokenizer)


def test_majority_run_predicts_constant_label(tmp_path) -> None:
    run_dir = tmp_path / "majority"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"kind": "majority", "majority_label": 2}), encoding="utf-8"
    )
    predictor = Predictor.from_run(run_dir)
    results = predictor.predict_batch(["one", "two", "three"])
    assert [item["class_id"] for item in results] == [2, 2, 2]


def test_probe_predictor_loads_local_encoder_and_head(tiny_tokenizer, tmp_path) -> None:
    encoder_path, _ = tiny_model_and_tokenizer(tmp_path / "encoder", tiny_tokenizer)
    run_dir = tmp_path / "probe-run"
    best_dir = run_dir / "best_model"
    head = ProbeHead(hidden_size=32, dropout=0.0)
    head.save_pretrained(
        best_dir,
        encoder_name=str(encoder_path),
        max_length=16,
        use_lang_tag=True,
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"kind": "probe", "selection_complete": True}), encoding="utf-8"
    )
    predictor = Predictor.from_run(run_dir, tokenizer=tiny_tokenizer)
    result = predictor.predict("good movie", "en")
    assert len(result["probabilities"]) == 3


def test_lora_predictor_loads_local_adapter(tiny_tokenizer, tmp_path) -> None:
    pytest.importorskip("peft")
    base_path, _ = tiny_model_and_tokenizer(tmp_path / "base", tiny_tokenizer)
    base_model = load_classifier(base_path, local_files_only=True)
    wrapped = prepare_lora(base_model, "mdeberta", rank=2)
    run_dir = tmp_path / "lora-run"
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True)
    wrapped.save_pretrained(best_dir, safe_serialization=True)
    tiny_tokenizer.save_pretrained(best_dir)
    (best_dir / "adapter_metadata.json").write_text(
        json.dumps({"kind": "lora", "base_model": str(base_path)}), encoding="utf-8"
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"kind": "lora", "selection_complete": True}), encoding="utf-8"
    )
    predictor = Predictor.from_run(run_dir)
    result = predictor.predict("good movie", "en")
    assert result["class_id"] in range(3)


def test_attention_has_one_normalized_value_per_token(tiny_tokenizer, tmp_path) -> None:
    model_path, _ = tiny_model_and_tokenizer(tmp_path / "checkpoint", tiny_tokenizer)
    predictor = Predictor.from_run(model_path, tokenizer=tiny_tokenizer)
    predictor.model.config.output_attentions = True
    result = explain_attention(predictor, "good movie", language="en")
    assert len(result["tokens"]) == len(result["attention"])
    assert sum(result["attention"]) == pytest.approx(1.0, abs=1e-5)
    assert result["disclaimer"] == DISCLAIMER
