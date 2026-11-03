"""Streamlit demonstration for trained multilingual sentiment runs."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.ui_helpers import (
    discover_runs,
    load_benchmark,
    load_comparison,
    load_reference,
    neutral_recall_statement,
    records_from_csv,
)
from sentiment.constants import LABELS
from sentiment.inference.attention import DISCLAIMER, explain_attention
from sentiment.inference.predictor import Predictor

st.set_page_config(page_title="Multilingual Sentiment Demo", page_icon="💬", layout="wide")
st.warning(
    "Research prototype. The earlier valid baseline was weak "
    "(accuracy 0.4167 vs 0.4060 majority; never predicted Neutral). Not production-ready."
)


@st.cache_resource(show_spinner="Loading selected checkpoint…")
def _load_predictor(path: str) -> Predictor:
    import torch

    return Predictor.from_run(path, device="cuda" if torch.cuda.is_available() else "cpu")


def _metric_cards(metrics: dict) -> None:
    columns = st.columns(3)
    for column, label, key in zip(
        columns,
        ("Accuracy", "Macro-F1", "Neutral recall"),
        ("accuracy", "macro_f1", "neutral_recall"),
        strict=True,
    ):
        value = metrics.get(key)
        column.metric(label, "PENDING" if value is None else f"{float(value):.4f}")


runs = discover_runs(os.environ.get("SENTIMENT_RUNS_DIR", "runs"))
selectable = [run for run in runs if not run["invalid"]]
with st.sidebar:
    st.header("Checkpoint")
    for run in runs:
        if run["invalid"]:
            st.markdown(
                f"<span style='color:gray'>{run['run_id']} — INVALID</span>",
                unsafe_allow_html=True,
            )
            st.caption(run["invalid_reason"])
    selected_id = None
    if selectable:
        selected_id = st.selectbox(
            "Select a run",
            [run["run_id"] for run in selectable],
            format_func=lambda run_id: next(
                f"{item['run_id']} · {item['kind']} · {item['backbone']}"
                for item in selectable
                if item["run_id"] == run_id
            ),
        )
        selected = next(run for run in selectable if run["run_id"] == selected_id)
        st.caption(f"Kind: {selected['kind']} · Backbone: {selected['backbone']}")
        st.caption(
            "Validation macro-F1: "
            + ("PENDING" if selected["val_macro_f1"] is None else f"{selected['val_macro_f1']:.4f}")
        )
        test_state = "available" if selected["has_test_evaluation"] else "PENDING"
        st.caption(f"Test evaluation: {test_state}")

tabs = st.tabs(["Try it", "Results", "Data", "About & limitations"])

with tabs[0]:
    st.subheader("Try a review")
    if not selectable:
        st.info(
            "No trained model found — train on the GPU laptop and copy `runs/` here "
            "(see `docs/GPU_RUNBOOK.md`)."
        )
    else:
        predictor = None
        try:
            predictor = _load_predictor(str(selected["path"]))
        except Exception as exc:
            st.error(f"Could not load this run: {exc}")
        text = st.text_area("Review text", placeholder="Write or paste a review…")
        language = st.selectbox("Language", ["en", "fr", "zh", "other"])
        show_attention = st.checkbox("Show attention tokens")
        if st.button("Predict", type="primary") and predictor is not None:
            try:
                result = predictor.predict(text, None if language == "other" else language)
                st.success(f"{result['label']} · confidence {result['confidence']:.3f}")
                st.bar_chart(pd.Series(result["probabilities"], index=LABELS, name="Probability"))
                with st.expander("Raw logits"):
                    st.json(result["logits"])
                if show_attention:
                    st.caption(DISCLAIMER)
                    st.json(
                        explain_attention(
                            predictor, text, None if language == "other" else language
                        )
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        st.divider()
        st.subheader("Batch prediction")
        pasted = st.text_area("One review per line", key="batch_texts")
        upload = st.file_uploader("Or upload CSV with a text column", type=["csv"])
        if st.button("Predict batch") and predictor is not None:
            try:
                if upload:
                    records = records_from_csv(upload.getvalue().decode("utf-8-sig"))
                    texts = [row["text"] for row in records]
                    languages = [row["language"] for row in records]
                else:
                    texts = [line.strip() for line in pasted.splitlines() if line.strip()]
                    languages = [None] * len(texts)
                predictions = predictor.predict_batch(texts, languages)
                frame = pd.DataFrame(
                    [
                        {"text": text, **prediction}
                        for text, prediction in zip(texts, predictions, strict=True)
                    ]
                )
                st.dataframe(frame, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    frame.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

with tabs[1]:
    st.subheader("Benchmark results")
    comparison = load_comparison()
    reference = load_reference()
    if reference:
        st.info("Recorded from earlier project, not reproduced here.")
        _metric_cards(reference.get("metrics", reference))
    if comparison is None:
        st.info("No comparison report yet. Runs without test evaluation will appear as PENDING.")
    else:
        valid_runs = comparison.get("runs", [])
        probe_metrics = next(
            (
                run.get("metrics")
                for run in valid_runs
                if run.get("kind") == "probe" and run.get("metrics")
            ),
            None,
        )
        for run in valid_runs:
            st.markdown(f"**{run.get('run_id')}** · {run.get('status', 'PENDING')}")
            metrics = run.get("metrics") or {}
            _metric_cards(metrics)
            if metrics:
                st.caption(neutral_recall_statement(metrics, probe_metrics))
                per_class = metrics.get("per_class")
                if per_class:
                    st.dataframe(pd.DataFrame(per_class).T, use_container_width=True)
                matrix = metrics.get("confusion_matrix")
                if matrix:
                    st.subheader("Confusion matrix")
                    heatmap = pd.DataFrame(
                        matrix, index=LABELS, columns=LABELS
                    ).style.background_gradient(cmap="Blues")
                    st.dataframe(heatmap)
                languages = metrics.get("per_language", {})
                if languages:
                    st.subheader("Per-language metrics")
                    st.bar_chart(pd.DataFrame(languages).T)
            else:
                st.info("PENDING: test evaluation has not been run.")

with tabs[2]:
    st.subheader("Benchmark data")
    benchmark = load_benchmark()
    if not benchmark["available"]:
        st.info(benchmark["reason"])
    else:
        manifest = benchmark["manifest"]
        st.write(
            {key: manifest.get(key) for key in ("dataset", "seed", "label_mapping", "languages")}
        )
        st.caption(f"Manifest SHA-256: `{benchmark['manifest_hash']}`")
        count_frame = pd.DataFrame(benchmark["counts"])
        if not count_frame.empty:
            st.bar_chart(count_frame, x="language", y="count", color="label", stack=False)
            langs = sorted(count_frame["language"].unique())
            labels = sorted(count_frame["label"].unique())
            lang_filter = st.selectbox("Sample language", ["all", *langs])
            label_filter = st.selectbox("Sample label", ["all", *labels])
            samples = benchmark["samples"]
            if lang_filter != "all":
                samples = [row for row in samples if row["language"] == lang_filter]
            if label_filter != "all":
                samples = [row for row in samples if LABELS[int(row["label"])] == label_filter]
            if samples:
                st.json(samples[0])

with tabs[3]:
    st.subheader("About and limitations")
    st.markdown(
        """This research prototype separates four categories:

1. **Pipeline capabilities:** tested code paths and tiny random-model fixtures.
2. **EN/FR/ZH benchmark:** the sampled Amazon Reviews Multi benchmark and its held-out test split.
3. **Experimental QLoRA:** an isolated five-example causal-LM experiment; not a classifier result.
4. **Invalid checkpoints:** excluded from inference and reported metrics.

Full mDeBERTa fine-tuning needs a separate GPU laptop. Real probe, fine-tuning, and final test
evaluation remain pending until those commands are run there. Reproduction steps are in
`docs/GPU_RUNBOOK.md`. Attention values are diagnostic and do not establish causality.
"""
    )
