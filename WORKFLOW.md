# WORKFLOW.md — Multilingual Sentiment Analysis: Build Plan

> This file is the single source of truth for building the project.
> It is written to be read **in chunks**. Do not load it all into working memory at once.

## How to read this file

1. Read **§0 Ground rules** and **§1 Specification** once, fully.
2. Then work through chunks **C01 → C15 in order**. Find a chunk by searching for its marker, e.g. `CHUNK:C05`.
   Read only the text between `<!-- CHUNK:Cxx START -->` and `<!-- CHUNK:Cxx END -->`.
3. After each chunk: run its **Checks**, tick it off in `PROGRESS.md`, write a 3-line summary (what was built, what was tested, anything pending), then move on.
4. If you are unsure about something not covered here, make the most conservative choice, record it in `docs/ASSUMPTIONS.md`, and continue. Only stop and ask if you are truly blocked.

---

## §0 Ground rules (apply to every chunk)

**R1. No real training.** Training will be done later on a separate GPU laptop. In this workspace you must NOT run real training, fine-tuning, or probe training on `microsoft/mdeberta-v3-base` or `xlm-roberta-base`.
- ALLOWED: downloading the dataset; preparing the benchmark; unit tests; smoke tests that train a **tiny randomly-initialised model** (hidden size ≤ 64, 2 layers, tiny vocab) for 1–3 optimiser steps on synthetic/mini data; `--dry-run` mode (builds config, data, model, runs ONE forward pass on ≤ 2 examples, then exits); downloading real *tokenizers* (small).
- ALLOWED ONLY IF the env var `RUN_HEAVY_TESTS=1` is set (default off): downloading real model weights for a single forward pass.
- FORBIDDEN: any multi-epoch run, any run whose purpose is to produce a metric.

**R2. Never fabricate.** No invented metrics, hardware specs, run results, or "completed" experiments. Anything not actually run is labelled `PENDING`. Numbers copied from the earlier project are labelled "recorded from earlier project, not reproduced here".

**R3. Test-set discipline.** The test split is used only by the final `evaluate --split test` command, only after checkpoint selection finished. Never use test metrics for tuning, early stopping, or model selection. The code must enforce this (see C07).

**R4. Fail loudly.** Missing fields, bad labels, split overlap, non-finite values (NaN/inf) → raise a clear exception with an actionable message. Never silently continue, clip, or skip.

**R5. Reproducibility.** Deterministic seeds everywhere (python, numpy, torch, DataLoader workers). Every result file records: dataset manifest hash, split, seed, model name, full config, library versions, device. Environment info is captured automatically at runtime, never typed by hand.

**R6. Code quality.** Python ≥ 3.10, type hints, docstrings on public functions, `pathlib`, no duplicated logic, clear error messages, `logging` (not bare prints) in library code. Must work on **Windows and Linux** from the project root. Format with `ruff`; test with `pytest`.

**R7. Keep four things separate everywhere (code, docs, UI, report):**
  (a) the general multilingual pipeline's capabilities,
  (b) the sampled EN/FR/ZH benchmark experiment,
  (c) the experimental QLoRA script (not validated, not comparable),
  (d) invalid historical checkpoints (never quoted as results),
  plus (e) new results produced by this rebuild (only what actually exists in `runs/`).

**R8. Do not silently change** the dataset, label mapping, sample sizes, seed, or test set.

---

## §1 Specification (condensed from the original project)

**Goal.** Reproducible multilingual sentiment classifier, 3 labels: `Negative=0`, `Neutral=1`, `Positive=2`. Main experiment: English, French, Chinese Amazon reviews. Pipeline is designed to extend to other languages (e.g. Arabic, Italian) via multilingual transformer representations, **not** machine translation. This is a research/engineering prototype; the earlier valid result is weak and must never be presented as production-ready.

**Dataset.** Hugging Face `mteb/amazon_reviews_multi` (https://huggingface.co/datasets/mteb/amazon_reviews_multi). Languages: `en`, `fr`, `zh`. Star ratings are integers 0–4 (verify the real column names/schema at download time and fail loudly if they differ from expectations):
- 0 or 1 → Negative (0)
- 2 → Neutral (1)
- 3 or 4 → Positive (2)

**Benchmark sampling (per language).** 600 train / 200 validation / 500 test → 1,800 / 600 / 1,500 total. Seed `20261021`, split-specific seed salts, reservoir sampling, exact-duplicate review text removed across splits. JSONL records contain at least `sample_id`, `language`, `text`, `label` (extra fields such as `stars` are fine). A manifest records provenance (dataset name, revision/commit if obtainable), sampling parameters, label mapping, language codes, split counts, class counts, and SHA-256 of each JSONL file.

> If a previous `data/amazon_reviews_benchmark/` folder already exists in the workspace, **validate it against its manifest and reuse it** instead of regenerating. If you regenerate, sample IDs may differ from the earlier project; record this in `docs/ASSUMPTIONS.md` and make sure all comparisons are re-run on the *same new* splits.

**Primary model.** `microsoft/mdeberta-v3-base` + 3-class sequence-classification head. Alternative: `xlm-roberta-base`. Pipeline steps: load tokenizer + model (`num_labels=3`) → optional language-tag prefix like `[LANG=fr]` → tokenize with truncation (up to 512 supported; baseline used 128) → dynamic padding per batch → cross-entropy + AdamW → mixed precision only when safe, **FP32 master weights** for optimiser updates → checkpoint selection by **validation macro-F1** → test evaluation only after selection → save reloadable model + tokenizer → single and batch inference returning label, class id, confidence, probabilities, logits.

**Recorded reference results from the earlier project** (frozen-encoder probe; NOT reproduced in this workspace until re-run):
- Setup: mdeberta-v3-base frozen; only pooler + classification head trainable; features = final hidden state at first token; max_len 128; seed 42; AdamW lr 1e-3, weight decay 0.01; batch 128; ≤ 15 epochs; early stop after 3 epochs without val macro-F1 gain; best val macro-F1 selected.
- Test (1,500 ex.): accuracy 0.4167; macro-F1 0.3060; majority-class accuracy 0.4060 (gain ≈ 0.0107).
- Negative: support 609, P 0.415, R 0.629, F1 0.500. Neutral: support 310, P 0.000, R 0.000, F1 0.000. Positive: support 581, P 0.419, R 0.417, F1 0.418.
- Per-language accuracy: en 0.406, fr 0.410, zh 0.434.
- **Major failure: the model never predicted Neutral.** Must be reported prominently everywhere.

**Invalid historical artifacts.** A checkpoint `checkpoints/mdeberta_v3_sentiment/` had non-finite weights/logits; its accuracy and attention outputs are invalid and never quoted. If it exists, an `INVALID` marker file must sit in it (C02 provides an audit tool to create markers).

**Experimental QLoRA script.** Causal-LM + 4-bit quantisation + LoRA on 5 hand-written prompt/response examples. Not evaluated on the Amazon benchmark; artifacts not comparable to the classifier. Must be isolated, configurable (base model + LoRA target modules consistent with that model), and documented (C13).

**Recommended next experiment (to be run later on the GPU laptop).** Full encoder fine-tuning on the same train split, select by val macro-F1, one final test evaluation. Compare against frozen probe and majority baseline on identical splits. Report accuracy, macro-F1, per-class P/R/F1, per-language metrics, and explicitly state whether Neutral recall improved. Parameter-efficient alternative (LoRA) if GPU memory is insufficient.

**Attention explanations** are an optional diagnostic. Never claim attention weights are faithful causal explanations.

---

## §2 Target project layout

```
multilingual-sentiment/
├── README.md
├── WORKFLOW.md                # this file
├── PROGRESS.md                # checklist you maintain
├── pyproject.toml             # package "sentiment", ruff + pytest config
├── requirements.txt           # runtime deps (pinned ranges)
├── requirements-dev.txt       # pytest, ruff
├── requirements-qlora.txt     # bitsandbytes, peft, etc. (optional)
├── configs/
│   ├── benchmark.yaml
│   ├── probe_mdeberta.yaml
│   ├── finetune_mdeberta.yaml
│   ├── finetune_xlmr.yaml
│   └── lora_mdeberta.yaml
├── src/sentiment/
│   ├── __init__.py
│   ├── constants.py           # labels, mapping, languages, seeds
│   ├── config.py              # dataclasses + YAML loading + validation
│   ├── seed.py
│   ├── io_utils.py            # json/jsonl/hash/atomic write
│   ├── env_info.py            # runtime environment capture
│   ├── safeguards.py          # finite checks, label validation
│   ├── data/{download,prepare,validate,dataset,collate}.py
│   ├── models/{factory,probe_head}.py
│   ├── training/{probe,finetune,optim,callbacks}.py
│   ├── evaluation/{metrics,evaluate,compare}.py
│   ├── inference/{predictor,attention}.py
│   ├── audit.py               # checkpoint auditing / INVALID markers
│   └── cli.py                 # `python -m sentiment <subcommand>`
├── app/streamlit_app.py       # demo UI (+ app/ui_helpers.py)
├── experiments/qlora_causal_lm/   # isolated, experimental
├── scripts/                   # check_gpu_env.py, pack_results.py, run_gpu_suite.{sh,ps1}
├── data/                      # amazon_reviews_multi/, amazon_reviews_benchmark/
├── runs/                      # one folder per run (created at run time)
├── reports/
│   ├── reference/recorded_baseline.json
│   ├── make_tables.py
│   └── ieee/main.tex
├── docs/{ASSUMPTIONS,LIMITATIONS,GPU_RUNBOOK,DATA_PROVENANCE}.md
└── tests/
```

## §3 Chunk index

| ID | Chunk | Depends on |
|----|-------|-----------|
| C01 | Scaffold, environment, PROGRESS.md | — |
| C02 | Core utilities: constants, config, seeds, IO, env capture, safeguards, audit | C01 |
| C03 | Data download | C02 |
| C04 | Benchmark preparation, validation, manifest | C03 |
| C05 | Dataset, tokenisation, dynamic-padding collator | C04 |
| C06 | Model factory + load-time checks | C02, C05 |
| C07 | Metrics, evaluation, test-set lock | C05, C06 |
| C08 | Frozen-encoder feature probe | C07 |
| C09 | Full fine-tuning (+ LoRA, AMP, class weights) | C07 |
| C10 | Inference + attention export | C06 |
| C11 | CLI, run folders, comparison | C08, C09, C10 |
| C12 | Streamlit demo UI | C10, C11 |
| C13 | Isolated QLoRA experiment | C02 |
| C14 | Docs, GPU runbook, IEEE report | C11–C13 |
| C15 | Final verification + handoff bundle | all |

---

<!-- CHUNK:C01 START -->
## C01 — Scaffold, environment, PROGRESS.md

**Do**
- Create the folder layout from §2 (empty `__init__.py` files where needed).
- `pyproject.toml`: package `sentiment` under `src/`, `pip install -e .` works; ruff + pytest config; `python -m sentiment` entry via `src/sentiment/__main__.py`.
- `requirements.txt` with compatible ranges for: `torch`, `transformers`, `datasets`, `tokenizers`, `sentencepiece`, `tiktoken`/`protobuf` (needed by DeBERTa-v3 tokenizers), `accelerate`, `peft`, `scikit-learn`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `streamlit`, `plotly` (or `altair`), `huggingface_hub`. Add a comment that the user must install the CUDA build of torch on the GPU laptop.
- `requirements-dev.txt`: `pytest`, `ruff`. `requirements-qlora.txt`: `bitsandbytes`, `peft`, `accelerate` (note: 4-bit needs CUDA GPU; bitsandbytes support on Windows is limited).
- Create `PROGRESS.md`: a checklist of C01–C15 with status (`TODO/DONE/BLOCKED`), a "Decisions" section, and a "Pending (needs GPU)" section.
- `.gitignore` (runs/, checkpoints/, *.pt, *.safetensors except small fixtures, `.venv`, `__pycache__`, `hf_cache`).
- Create a virtual environment instructions block in README (do not write the full README yet).

**Checks**
- `pip install -e .` succeeds; `python -c "import sentiment"` works; `pytest -q` runs (zero or trivial tests pass); `ruff check .` passes.
<!-- CHUNK:C01 END -->

---

<!-- CHUNK:C02 START -->
## C02 — Core utilities

**Create**
- `constants.py`: `LABELS = ["Negative","Neutral","Positive"]`, `LABEL2ID`, `ID2LABEL`, `STAR_TO_LABEL = {0:0,1:0,2:1,3:2,4:2}`, `LANGUAGES = ["en","fr","zh"]`, `DEFAULT_SEED_BENCHMARK = 20261021`, `DEFAULT_SEED_TRAIN = 42`, split sizes, dataset name.
- `config.py`: typed dataclasses (`DataConfig`, `ModelConfig`, `TrainConfig`, `RunConfig`), `load_config(path, overrides)` from YAML with CLI overrides, strict validation (unknown keys → error; invalid ranges → error).
- `seed.py`: `set_seed(seed, deterministic=True)` (python, numpy, torch, cuda), `seed_worker`, `make_generator`.
- `io_utils.py`: `read_jsonl`, `write_jsonl`, `read_json`, `write_json` (atomic: write temp then rename), `sha256_file`, `sha256_text`.
- `env_info.py`: `capture_env()` → python, OS, torch, transformers, datasets, peft versions, CUDA availability, GPU names/memory (only what is really detected), git commit if available.
- `safeguards.py`:
  - `assert_finite_tensor(t, name)`, `assert_finite_array(a, name)`, `assert_finite_model(model)` (scan all parameters, report the first offending names + counts).
  - `validate_label(x)`, `validate_record(rec)` (required fields `sample_id, language, text, label`; non-empty text; label ∈ {0,1,2}; language ∈ allowed).
  - Custom exceptions: `NonFiniteError`, `DataValidationError`, `SplitOverlapError`, `TestSetLockedError`.
- `audit.py` + CLI hook: `audit_checkpoint(dir)` loads weights (safetensors/pt) and checks finiteness; if invalid, writes `INVALID.md` marker + `invalid_metrics.json` in that folder and returns non-zero. `audit_all(root)` scans a directory tree.

**Tests** (`tests/test_safeguards.py`, `test_config.py`, `test_audit.py`)
- NaN/inf tensors and model params are caught; clean ones pass.
- Bad records (missing field, label 7, empty text) raise `DataValidationError`.
- Atomic JSON write + hash stable across runs.
- Audit creates an `INVALID` marker for a deliberately corrupted tiny checkpoint and leaves a clean one alone.
- Config rejects unknown keys.

**Done when** all tests pass and `ruff` is clean.
<!-- CHUNK:C02 END -->

---

<!-- CHUNK:C03 START -->
## C03 — Data download

**Create** `data/download.py` + CLI `python -m sentiment download --languages en fr zh --out data/amazon_reviews_multi`

- Use `datasets.load_dataset("mteb/amazon_reviews_multi", <lang>)` (inspect the real configs/splits first; if the loading API differs, adapt and note it in `docs/ASSUMPTIONS.md`).
- Detect and print the actual schema; **assert** the presence of a text column and an integer 0–4 rating column (do not assume names — find them, then map to canonical `text` / `stars`).
- Save each language/split as JSONL (`data/amazon_reviews_multi/<lang>/<split>.jsonl`) with `sample_id` (`<lang>-<split>-<row index>`), `language`, `text`, `stars`.
- Write `data/amazon_reviews_multi/download_manifest.json`: dataset name, revision/commit if available, languages, split sizes, schema found, sha256 per file, download timestamp (UTC), library versions.
- Idempotent: skip existing files whose hash matches the manifest; `--force` re-downloads.
- Support `HF_HOME`/offline mode (`HF_HUB_OFFLINE=1`) with a clear error if data is missing.

**Tests**
- Unit test the schema detection and record conversion using a tiny fake in-memory dataset (no network).
- Optional `@pytest.mark.network` test that downloads 1 language and checks counts (skipped by default; run it manually once for real during this chunk and report the outcome).

**Run for real (allowed):** download en, fr, zh. Report split sizes you actually got.
<!-- CHUNK:C03 END -->

---

<!-- CHUNK:C04 START -->
## C04 — Benchmark preparation, validation, manifest

**Create** `data/prepare.py`, `data/validate.py` + CLI `python -m sentiment prepare` and `python -m sentiment validate-data`

**Sampling algorithm**
- For each language and each split (train/validation/test) draw reservoir samples of the configured size (600/200/500) from the source split of the same name, using `numpy`/`random` RNG seeded with `seed ^ hash_salt(language, split)` where the salt is a **stable** function (e.g. CRC32 of `f"{language}:{split}"`) — never Python's built-in `hash()` (it is randomised per process).
- Map stars → label with `STAR_TO_LABEL`.
- Remove exact duplicate texts across splits (normalise only by `strip()` for duplicate detection; keep the original text). Order of precedence on conflicts: test is protected first, then validation, then train; replace removed items by drawing more candidates so final counts are exactly 600/200/500 per language. Fail loudly if impossible.
- Output `data/amazon_reviews_benchmark/{train,validation,test}.jsonl` with `sample_id, language, text, label, stars`, shuffled with the seed (deterministic).
- Write `manifest.json`: dataset name/revision (from C03 manifest), seed, salts, sizes, label mapping, language codes, split counts, per-language × per-label counts per split, sha256 per output file, the sampling algorithm description, UTC timestamp, library versions. Also compute and store the **majority-class accuracy on the test split** (computed, not hard-coded).

**Validation (`validate-data`)**
- Every record passes `validate_record`.
- Exact text overlap between train/val/test = 0, else raise `SplitOverlapError` listing counts and examples (truncated).
- Counts match the manifest; hashes match the manifest; `sample_id`s unique.
- Print a table of class balance per language/split. Warn (not fail) if Neutral share < 10% in any split.

**Tests**
- Same seed → byte-identical output (run twice on a small synthetic source and compare hashes).
- Different seed → different sample.
- Injected overlap is detected and raises.
- Counts exact; label mapping correct for every star value.
- Salt function is stable across processes (compare in a subprocess).

**Run for real (allowed):** prepare the real benchmark and validate it. Report the real counts and class distribution, and whether they match the recorded test support (Neg 609 / Neu 310 / Pos 581) — if they differ, just note it in `docs/ASSUMPTIONS.md` (do not force a match).
<!-- CHUNK:C04 END -->

---

<!-- CHUNK:C05 START -->
## C05 — Dataset, tokenisation, collator

**Create** `data/dataset.py`, `data/collate.py`

- `SentimentDataset(records, tokenizer, max_length, use_lang_tag)`: returns `input_ids`, `attention_mask`, `labels`, plus `sample_id` and `language` kept for evaluation (not fed to the model).
- Language tag helper: `format_text(text, lang, use_lang_tag)` → `"[LANG=fr] <text>"`; normalise language codes (`fr-FR`, `FR` → `fr`); unknown language → clear error (or `und` only if explicitly enabled).
- Truncation to `max_length` (1–512 validated). **Tokenise lazily or pre-tokenise without padding**; padding happens in the collator (dynamic per batch, pad to the longest in batch, optionally to a multiple of 8).
- Collator returns tensors and passes through metadata lists.
- `make_dataloader(...)` with deterministic generator + `seed_worker`.

**Tests** (use a **tiny local tokenizer** built in a fixture with the `tokenizers` library, wrapped in `PreTrainedTokenizerFast`; no network)
- Tag prefix present/absent as configured; lang normalisation.
- Truncation never exceeds `max_length`; padding length equals the batch max.
- Same seed → same batch order; different seed → different.
- Metadata alignment: `sample_id` order matches labels after shuffling.
- Optional `@pytest.mark.network` test with the real mDeBERTa tokenizer on a French and a Chinese sentence (no crash, sensible tokens, special tokens present). Run it once for real.
<!-- CHUNK:C05 END -->

---

<!-- CHUNK:C06 START -->
## C06 — Model factory + load-time checks

**Create** `models/factory.py`, `models/probe_head.py`, test fixture `tests/fixtures.py::tiny_model_and_tokenizer()`

- `load_classifier(model_name_or_path, num_labels=3, dtype=torch.float32, ...)`: loads `AutoModelForSequenceClassification` with `id2label/label2id`, then calls `assert_finite_model`. Supports `microsoft/mdeberta-v3-base`, `xlm-roberta-base`, or a local saved dir.
- `SUPPORTED_BACKBONES` registry with short names (`mdeberta`, `xlmr`) → HF ids, so configs can say `backbone: mdeberta`.
- `freeze_encoder(model)`; `count_trainable(model)` helper that logs trainable vs total parameters.
- **Mixed-precision policy** (`training/optim.py` can import from here or live there): master weights always FP32; use `torch.autocast` with `bfloat16` if the GPU supports it, else `float16` + `GradScaler` only when `amp: fp16` is explicitly enabled; default `amp: auto` = bf16 if supported else off. Document the mDeBERTa fp16 instability risk in a docstring. Never call `model.half()` before optimiser creation.
- `probe_head.py`: a small module mirroring "pooler + classifier" for the frozen-feature probe (Linear(h,h) + activation + dropout + Linear(h,3)), with `save/load` that records the encoder name, hidden size, and max_length.
- Tiny-model fixture: randomly-initialised small `DebertaV2Config`/`XLMRobertaConfig` (hidden ≤ 64, 2 layers) + the tiny tokenizer from C05, saved to a tmp dir so it can be reloaded via the same code path as real checkpoints.

**Tests**
- Loading the tiny model through `load_classifier` works from a local path; a model with injected NaN raises `NonFiniteError`.
- `freeze_encoder` leaves only head params trainable.
- AMP policy function returns the expected setting for mocked capabilities (CPU → off; CUDA bf16 supported → bf16; fp16 only if explicit).
<!-- CHUNK:C06 END -->

---

<!-- CHUNK:C07 START -->
## C07 — Metrics, evaluation, test-set lock

**Create** `evaluation/metrics.py`, `evaluation/evaluate.py`

- `compute_metrics(y_true, y_pred, languages)` → accuracy, macro-F1, weighted-F1, per-class precision/recall/F1/support (`zero_division=0`), confusion matrix, per-language accuracy + macro-F1, majority-class baseline accuracy (computed from `y_true`), `accuracy_above_majority`, **`neutral_recall`** and **`neutral_predicted_count`** as top-level fields, and a `neutral_never_predicted` boolean flag.
- `evaluate_model(model, dataloader, device, amp)` → logits (float32 numpy), probabilities (softmax), predictions, with `assert_finite_*` on loss, logits, probabilities **before** any metric is computed. On non-finite values: raise `NonFiniteError`, write `evaluation_invalid.json` (reason only, no metrics), and stop.
- Writes artifacts to a given folder: `metrics.json`, `classification_report.txt`, `per_language.json`, `confusion_matrix.csv`, `predictions.csv` (sample_id, language, true, pred, confidence, p_neg, p_neu, p_pos), `predictions_and_logits.npz`.
- Every metrics file embeds metadata: dataset manifest hash, split name, seed, model identifier, config snapshot, env info, timestamp.

**Test-set lock (enforces R3)**
- A run folder has `run_status.json` with `selection_complete: bool`. `evaluate --split test` raises `TestSetLockedError` unless `selection_complete` is true. After a test evaluation it writes `test_evaluated.lock` (with timestamp + metrics hash). A second test evaluation of the same run requires `--force` and records `test_evaluation_count` in the metrics metadata plus a visible warning in the output.
- Validation evaluation is always allowed.

**Tests**
- Metrics on a hand-computed tiny example match expected values exactly (including a case where Neutral is never predicted → neutral recall 0, flag true).
- Logits with NaN → `NonFiniteError`, no `metrics.json` written.
- Test evaluation refused before selection; allowed after; refused second time without `--force`.
- Per-language breakdown correct on a 2-language toy case.
<!-- CHUNK:C07 END -->

---

<!-- CHUNK:C08 START -->
## C08 — Frozen-encoder feature probe

**Create** `training/probe.py`, config `configs/probe_mdeberta.yaml`

Reproduces the earlier baseline protocol:
- Encoder frozen (eval mode, no dropout, `torch.no_grad`); features = last hidden state at position 0 (`CLS`); max_len 128; seed 42.
- **Feature cache**: extract features once per split → `runs/<run_id>/features/{train,validation,test}.npz` (features float32, labels, sample_ids, languages). Cache key = hash of (model name, max_length, lang-tag flag, split file hash). Reuse when the key matches. Assert finite features.
- Train `ProbeHead` (pooler-like layer + classifier) on cached features: AdamW lr 1e-3, weight decay 0.01, batch 128, ≤ 15 epochs, early stopping patience 3 on **validation macro-F1**, keep best state.
- After each epoch log train loss, val loss, val accuracy, val macro-F1, val Neutral recall to `train_log.jsonl`.
- When finished: save best head + metadata, set `selection_complete: true` in `run_status.json`. **Do not evaluate test automatically**; test evaluation is a separate command (C11). Extracting test features in the cache step is fine, but they must not be touched by training/selection code (assert this in a test by passing a poisoned test array).
- Also implement the **majority-class baseline** "model" (`kind: majority`) that predicts the most frequent training label; it follows the same evaluation/run-folder format so it can appear in comparisons.

**Tests (tiny random encoder, tiny synthetic data)**
- Cache created, reused on second call, invalidated when max_length changes.
- Training runs 2 epochs on synthetic features and early stopping triggers correctly with a patience of 1 on a rigged plateau.
- Best epoch selected by val macro-F1 (rig a case where last epoch ≠ best).
- Poisoned test features don't change training outcome.
- Majority model predicts the train-majority label for every input.

**Pending (GPU laptop):** the real probe run on mDeBERTa. Mark as PENDING in `PROGRESS.md`.
<!-- CHUNK:C08 END -->

---

<!-- CHUNK:C09 START -->
## C09 — Full fine-tuning (+ LoRA, AMP, class weights)

**Create** `training/finetune.py`, `training/optim.py`, configs `finetune_mdeberta.yaml`, `finetune_xlmr.yaml`, `lora_mdeberta.yaml`

Custom PyTorch loop (primary) — keep it simple, readable and deterministic:
- Model FP32 master weights; AdamW with decoupled weight decay (no decay on bias/LayerNorm); linear warmup + linear decay scheduler; gradient clipping (default 1.0); gradient accumulation; optional AMP per the C06 policy (bf16 preferred; fp16 with GradScaler only when explicitly enabled).
- Defaults for full fine-tuning (documented as *starting points, not tuned*): lr 2e-5, batch 16, epochs 5, warmup 10%, weight decay 0.01, max_len 128, seed 42, lang tag on. All overridable via YAML/CLI.
- **Every step:** `assert_finite_tensor(loss)`; on non-finite loss, stop immediately, save `training_failed.json` (step, lr, amp mode, last good loss) — never keep training or save a checkpoint after that.
- **Every epoch:** evaluate on **validation only**; log loss/accuracy/macro-F1/per-class recall/Neutral recall; save `best_model/` (model + tokenizer via `save_pretrained`, safetensors) only when val macro-F1 improves, after `assert_finite_model` on the weights about to be saved. Early stopping patience configurable (default 2).
- On completion write `run_status.json` with `selection_complete: true`, best epoch, best val macro-F1.
- Optional `--class-weights balanced` (inverse-frequency weights computed from the **train** split only) — default OFF; when ON it is recorded in the config and labelled as an ablation in comparisons. Document: it targets the Neutral-never-predicted failure, and it must be reported as a separate run, never as a silent replacement.
- Optional `--peft lora` using `peft` (`task_type=SEQ_CLS`, r=8, alpha=16, dropout 0.1; target modules configured per backbone, e.g. `query_proj,value_proj` for DeBERTa-v2/mDeBERTa and `query,value` for XLM-R; the classifier head must remain trainable via `modules_to_save`). Save adapter + merged-capable metadata so inference can reload it.
- `--resume` from the last epoch checkpoint (optimizer + scheduler + RNG state) — useful for laptops that sleep.
- `--dry-run`: build everything, run ONE forward pass on ≤ 2 real examples, print config + parameter counts + estimated memory note, exit without writing a model.
- Hugging Face `Trainer` mode: **not required**; if you keep it for parity, it must share the same data/metrics code and pass the same tests. Prefer deleting it over keeping dead code.

**Tests (tiny random model + synthetic data; seconds on CPU)**
- 2 epochs of training run end-to-end and produce `best_model/` that reloads via `load_classifier` and gives identical logits on a fixed input.
- Injected NaN loss (monkeypatched) stops training and writes `training_failed.json`, no checkpoint saved after it.
- Best-by-val-macro-F1 selection works (rigged case); test data never read (assert via a dataloader that raises if the test split is touched).
- Same seed → same final weights hash on CPU for the tiny model (determinism test).
- LoRA wrapper: trainable parameter count is far below total; adapter saves and reloads.
- Class weights computed from train only and sum/shape correct.
- `--dry-run` exits 0 on the tiny model and writes no checkpoint.
<!-- CHUNK:C09 END -->

---

<!-- CHUNK:C10 START -->
## C10 — Inference + attention export

**Create** `inference/predictor.py`, `inference/attention.py`

- `Predictor.from_run(run_dir | checkpoint_dir)` auto-detects `kind` (`full`, `lora`, `probe`, `majority`) from saved metadata and loads accordingly; `assert_finite_model` on load.
- `predict(text, language=None)` and `predict_batch(texts, languages=None, batch_size=32)` → list of dicts: `label`, `class_id`, `confidence`, `probabilities` `{Negative,Neutral,Positive}`, `logits`. Finite checks on logits/probs. Language `None` → no tag (or tag `und` if the model was trained with tags — follow the metadata, and document the behaviour).
- Optional light language guess helper is NOT required; do not add a heavy dependency. UI will let the user pick the language.
- `attention.py`: `explain_attention(text, ...)` returns tokens + normalised last-layer, head-averaged attention from CLS (requires `output_attentions=True`; some fast attention kernels don't return attentions — handle that with a clear message). Every output dict carries `"disclaimer": "Attention weights are a diagnostic, not a faithful causal explanation."`. Assert finite.

**Tests (tiny model fixture)**
- Single and batch outputs have the expected keys; probabilities sum to 1 (±1e-5); confidence = max probability; batch of 1 equals single prediction.
- Batch predictions are invariant to batch size and ordering.
- Multilingual strings (English, French, Chinese, Arabic, Italian) don't crash and are tokenised.
- Predictor refuses a directory containing an `INVALID` marker with a clear error.
- Attention output length equals token count and contains the disclaimer.
<!-- CHUNK:C10 END -->

---

<!-- CHUNK:C11 START -->
## C11 — CLI, run folders, comparison

**Create** `cli.py` (argparse subcommands), `evaluation/compare.py`

Subcommands (all work from the project root; every one supports `--help` with an example):
```
python -m sentiment download      --languages en fr zh
python -m sentiment prepare       --config configs/benchmark.yaml
python -m sentiment validate-data
python -m sentiment probe         --config configs/probe_mdeberta.yaml      [--dry-run]
python -m sentiment train         --config configs/finetune_mdeberta.yaml   [--dry-run] [--resume] [--class-weights balanced] [--peft lora]
python -m sentiment majority      # builds the majority baseline run
python -m sentiment evaluate      --run runs/<run_id> --split validation|test [--force]
python -m sentiment predict       --run runs/<run_id> --text "..." [--lang fr]
python -m sentiment predict-batch --run runs/<run_id> --file texts.jsonl
python -m sentiment compare       --runs runs/<a> runs/<b> ... --out reports/results
python -m sentiment audit         --root checkpoints
python -m sentiment env           # prints captured environment info
```
- Run folder: `runs/<run_id>/` with `config.json`, `env.json`, `train_log.jsonl`, `run_status.json`, `best_model/`, `eval_validation/`, `eval_test/` (created later), `dataset_manifest_ref.json` (hash of the manifest used). `run_id = <backbone>-<mode>-s<seed>-<YYYYmmdd-HHMMSS>`.
- Every command verifies `validate-data` (fast hash check) before training/evaluating; refuses to run on mismatched manifest hashes.
- `compare`: reads runs' `eval_test/metrics.json` (only for runs where test evaluation exists; others are listed as `PENDING`), produces `comparison.json`, `comparison.csv`, `comparison.md`: accuracy, macro-F1, per-class P/R/F1, **Neutral recall (highlighted)**, per-language accuracy/F1, accuracy above majority baseline. It must refuse to compare runs whose dataset manifest hashes differ. If the majority baseline run is missing, compute it from the test labels and mark it as such.
- `compare` writes an explicit sentence per run: "Neutral recall improved / did not improve vs <probe run>" computed from data, never hand-written.

**Tests**
- CLI smoke tests via `subprocess` on the tiny fixtures (`probe`, `train`, `evaluate` on validation, `predict`).
- `compare` refuses mismatched manifests; marks missing test evals as PENDING.
- `evaluate --split test` obeys the lock (end-to-end through the CLI).
<!-- CHUNK:C11 END -->

---

<!-- CHUNK:C12 START -->
## C12 — Streamlit demo UI

**Create** `app/streamlit_app.py`, `app/ui_helpers.py`, `docs/sample_texts.json` (a few multilingual example reviews: en, fr, zh, plus ar and it to show the extensibility idea — clearly labelled "not evaluated").

Run with: `streamlit run app/streamlit_app.py`

**Principles:** simple, clean, honest. Keep logic in `ui_helpers.py` (importing from `sentiment`) so it can be unit-tested. The UI must **work with no trained model present**.

**Layout**
- Persistent top banner: *"Research prototype. The earlier valid baseline was weak (accuracy 0.4167 vs 0.4060 majority; never predicted Neutral). Not production-ready."*
- Sidebar: choose a run/checkpoint from auto-discovered `runs/*/best_model` (+ probe heads). Shows the run's kind, backbone, val macro-F1 and whether test evaluation exists. Runs containing an `INVALID` marker are listed greyed-out and unselectable with the reason.
- **Tab 1 – Try it:** text area + language dropdown (en/fr/zh/other) + Predict. Shows label, confidence, a probability bar chart for the 3 classes, and (expander) raw logits. Batch mode: one review per line or CSV upload → table + download CSV. Optional "Show attention tokens" toggle, always with the attention disclaimer. If no model is available: show a friendly "No trained model found — train on the GPU laptop and copy `runs/` here (see docs/GPU_RUNBOOK.md)", and keep the other tabs usable.
- **Tab 2 – Results:** reads `reports/results/comparison.json` if present and the recorded reference in `reports/reference/recorded_baseline.json` (labelled "recorded from earlier project, not reproduced here"). Shows metric cards (accuracy, macro-F1, Neutral recall), per-class P/R/F1 table, confusion matrix heatmap, per-language bars, and a "Neutral recall: improved/not improved" callout computed from data. Runs without test evaluation appear as `PENDING` cards. Never display invalid-checkpoint numbers.
- **Tab 3 – Data:** manifest summary (dataset, seed, label mapping), counts per language/split/label chart, random sample viewer (filter by language/label). Shows manifest hash.
- **Tab 4 – About & limitations:** the four-way separation from R7 (pipeline capabilities / EN-FR-ZH benchmark / experimental QLoRA / invalid checkpoints), compute requirements, how to reproduce.
- Load models with `st.cache_resource`; handle errors with `st.error` (never a raw traceback).

**Tests**
- `streamlit.testing.v1.AppTest` smoke test: app starts with no runs; banner and 4 tabs render; no exception.
- Test with the tiny-model run fixture: prediction displayed; invalid run is not selectable.
- `ui_helpers` unit tests: run discovery, INVALID filtering, comparison loading with PENDING handling.
<!-- CHUNK:C12 END -->

---

<!-- CHUNK:C13 START -->
## C13 — Isolated QLoRA experiment

**Create** `experiments/qlora_causal_lm/{train_qlora.py,config.yaml,prompts.jsonl,README.md}`

- Fully isolated: its own README, config, output dir (`experiments/qlora_causal_lm/outputs/`), no imports from the classifier training code except generic utilities; **its outputs are never read by `compare`, the Results tab, or the report tables.**
- Configurable `base_model` (default a small, ungated causal LM such as `distilgpt2`; also documented larger options) and `lora_target_modules` (**must be consistent with the model**: e.g. `c_attn` for GPT-2 family; `q_proj,v_proj` for Llama/Mistral-style). Add a validation step that inspects the loaded model's module names and fails clearly if a target module doesn't exist.
- `use_4bit: true/false` — 4-bit needs a CUDA GPU + `bitsandbytes`; if unavailable, fall back to a clear error or a non-quantised LoRA run (config choice). Document hardware/dependency requirements.
- Keep the 5 hand-written multilingual prompt/response examples in `prompts.jsonl`, read from file instead of hard-coded.
- Prominent banner in the script output and README: *"Experimental. Trained on 5 examples. Does not establish model quality. Not comparable to the 3-class benchmark."*
- `--dry-run` loads config, validates target modules against the (tiny) model, tokenises the examples, runs one forward pass, exits.

**Tests**
- Use a tiny randomly-initialised GPT-2 config fixture: module-name validation passes for `c_attn`, fails clearly for `q_proj`.
- Dry-run passes; one LoRA training step on the tiny model runs (CPU, no quantisation).
- Confirm nothing under `experiments/` is imported by `src/sentiment`.
<!-- CHUNK:C13 END -->

---

<!-- CHUNK:C14 START -->
## C14 — Docs, GPU runbook, IEEE report

**Create / finish**
- `README.md`: overview, honest status, quick start (install → download → prepare → validate → run UI), command reference, project structure, the R7 separation.
- `docs/GPU_RUNBOOK.md` — step-by-step for the **separate GPU laptop**, with both PowerShell and bash commands:
  1. Copy the project (or `git clone`), create venv, install CUDA torch + `requirements.txt`.
  2. `python scripts/check_gpu_env.py` (prints torch/CUDA/GPU name/VRAM/bf16 support; warns if CUDA is unavailable; suggests batch size / LoRA fallback by VRAM ranges — suggestions, flagged as heuristics).
  3. `python -m sentiment validate-data` (manifest hash must match).
  4. Order of runs: `majority` → `probe` → `train` (mdeberta full fine-tune) → optionally `train --class-weights balanced` → optionally `train` with xlmr → optionally `--peft lora`. For each: `--dry-run` first.
  5. After each run: `evaluate --split validation`, then **once** `evaluate --split test`.
  6. `compare`, then `python scripts/pack_results.py` to zip `runs/*/` (metrics, predictions, logs, configs; `--include-model` optional) for copying back.
  7. Troubleshooting: OOM (smaller batch, grad accumulation, max_len 128, LoRA), fp16 NaNs (switch to bf16/fp32), laptop sleeping (`--resume`), slow tokenizer download.
- `scripts/check_gpu_env.py`, `scripts/pack_results.py`, `scripts/run_gpu_suite.ps1` and `.sh` (sequential runs with `--dry-run` first, stop on first error; test-evaluation steps clearly separated and commented).
- `docs/LIMITATIONS.md`, `docs/DATA_PROVENANCE.md` (dataset, label mapping, sampling parameters, how to verify), `docs/ASSUMPTIONS.md` (everything you could not verify).
- `reports/reference/recorded_baseline.json`: the recorded numbers from §1, with `"source": "recorded from earlier project; not reproduced in this workspace"`.
- `reports/make_tables.py`: generates `reports/ieee/generated_tables.tex` from `reports/results/comparison.json` (and writes `PENDING` rows when no runs exist).
- `reports/ieee/main.tex`: concise IEEE-style report (IEEEtran) with sections: Abstract, Introduction, Data & Protocol, Methods (pipeline; frozen probe; full fine-tuning; LoRA), Results (**three clearly separated blocks**: recorded earlier baseline; new results from this rebuild — PENDING until runs exist; not-evaluated/excluded items), Neutral-class failure analysis, Limitations, Reproducibility, Conclusion. The invalid checkpoint and QLoRA script are discussed only in a "Excluded and unevaluated artifacts" subsection. No claims beyond evidence. It must compile conceptually (`pdflatex`) — try compiling if a TeX distribution exists, otherwise state it was not compiled.

**Checks**
- Every command in README / GPU_RUNBOOK actually exists in `--help` output (write a small test that greps commands from the docs and checks the subcommand names).
- Report contains no number that isn't either in the recorded reference or generated from `runs/`.
<!-- CHUNK:C14 END -->

---

<!-- CHUNK:C15 START -->
## C15 — Final verification + handoff bundle

1. `ruff check .` clean; `pytest -q` fully green (network + heavy tests skipped by default; run the network ones once manually and record the outcome in `PROGRESS.md`).
2. Fresh-environment check: create a new venv, `pip install -e . -r requirements.txt`, run `python -m sentiment --help`, `python -m sentiment validate-data`, `python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run` (tiny override if the real model isn't downloaded), and `streamlit run app/streamlit_app.py` (headless start check via `AppTest`).
3. Confirm **no** run folders with real metrics were produced by this workspace (only tiny-fixture temp dirs, which live under `tmp_path` and aren't committed).
4. Write the final summary in `PROGRESS.md`:
   - What is implemented and tested (with the real test counts).
   - What is **PENDING on the GPU laptop** (list the exact commands).
   - Known limitations and assumptions.
   - Which files to copy to the GPU laptop and which folder to bring back (`runs/` zip).
5. Print the final project tree.

**Done when** the summary is written and no placeholder TODOs remain in code (`grep -R "TODO" src app scripts` returns nothing unexplained).
<!-- CHUNK:C15 END -->
