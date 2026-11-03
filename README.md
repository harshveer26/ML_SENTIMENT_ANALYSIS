# Multilingual Amazon Review Sentiment

An engineering and research prototype for three-class sentiment classification
(Negative / Neutral / Positive) on English, French, and Chinese Amazon Reviews Multi data.
The primary backbone is mDeBERTa-v3-base; XLM-RoBERTa and LoRA configurations are provided.

> **Status:** source data and a reproducible benchmark are prepared. Code paths are covered by
> tests using tiny randomly initialized models. Real probe/fine-tuning experiments and final
> checkpoint test evaluations are **PENDING** on the separate GPU laptop. This is not a
> production model.

## Quick start

From the project root, create an environment and install dependencies:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e . -r requirements.txt
```

On Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.
On the GPU laptop, install the CUDA-enabled PyTorch build that matches the driver first.

Download and prepare the benchmark (an existing benchmark is validated and reused):

```console
python -m sentiment download --languages en fr zh
python -m sentiment prepare --config configs/benchmark.yaml
python -m sentiment validate-data
```

Start the demo UI:

```console
streamlit run app/streamlit_app.py
```

The UI is usable without a trained checkpoint. It shows a clear message and keeps Results,
Data, and About tabs available. Real training and evaluation steps are described in
[`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md).

## Commands

Run `python -m sentiment --help` or add `--help` to any command. Main commands:

```console
python -m sentiment majority
python -m sentiment probe --config configs/probe_mdeberta.yaml --dry-run
python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run
python -m sentiment evaluate --run runs/<run_id> --split validation
python -m sentiment evaluate --run runs/<run_id> --split test
python -m sentiment predict --run runs/<run_id> --text "Very good product" --lang en
python -m sentiment compare --runs runs/<run_a> runs/<run_b> --out reports/results
python -m sentiment audit --root checkpoints
python -m sentiment env
```

Test evaluation is locked until checkpoint selection is complete and can be recorded only once
unless `--force` is supplied. Run comparison rejects different benchmark manifest hashes.
Missing test evaluations are reported as `PENDING`.

## Evidence categories

1. **Pipeline capabilities:** implementation and tiny random-model test coverage.
2. **EN/FR/ZH benchmark:** sampled benchmark data with a manifest and split-overlap validation.
3. **Experimental QLoRA:** a separate five-example causal-LM smoke experiment, not a classifier result.
4. **Invalid checkpoints:** marked invalid and excluded from prediction and reported comparisons.

The earlier project recorded test accuracy 0.4167 and majority accuracy 0.4060, with Neutral
recall 0. Those are historical numbers, labelled as not reproduced here; see
[`reports/reference/recorded_baseline.json`](reports/reference/recorded_baseline.json).

## Project layout

- `src/sentiment/`: data pipeline, model factory, probe/fine-tuning, inference, metrics, CLI.
- `configs/`: benchmark, mDeBERTa, XLM-R, and LoRA configurations.
- `data/amazon_reviews_multi/`: downloaded source JSONL; `data/amazon_reviews_benchmark/`:
  reproducible sampled train/validation/test files and manifest.
- `app/`: Streamlit demo and testable UI helpers.
- `experiments/qlora_causal_lm/`: isolated experimental causal-LM code and outputs.
- `reports/`: historical reference, generated comparison tables, and IEEE-style report.
- `docs/`: provenance, assumptions, limitations, and GPU runbook.

For the complete implementation/test status and pending GPU commands, see [`PROGRESS.md`](PROGRESS.md).
