# Build progress

Status values: `TODO`, `DONE`, `BLOCKED`.

| Chunk | Scope | Status |
|---|---|---|
| C01 | Scaffold, environment, PROGRESS.md | DONE |
| C02 | Core utilities | DONE |
| C03 | Data download | DONE |
| C04 | Benchmark preparation, validation, manifest | DONE |
| C05 | Dataset, tokenisation, collator | DONE |
| C06 | Model factory + load-time checks | DONE |
| C07 | Metrics, evaluation, test-set lock | DONE |
| C08 | Frozen-encoder feature probe | DONE |
| C09 | Full fine-tuning (+ LoRA, AMP, class weights) | DONE |
| C10 | Inference + attention export | DONE |
| C11 | CLI, run folders, comparison | DONE |
| C12 | Streamlit demo UI | DONE |
| C13 | Isolated QLoRA experiment | DONE |
| C14 | Docs, GPU runbook, IEEE report | DONE |
| C15 | Final verification + handoff bundle | DONE |

## Decisions

- C01 scaffold follows the target layout; implementation files will be added chunk by chunk.
- Runtime dependencies are listed in `requirements.txt`; editable package installation remains lightweight.
- No benchmark data or model weights have been downloaded or trained in C01.
- C01 checks passed on the available Python 3.13 environment: editable install, import, one pytest, and Ruff.
- C02 tests pass (15 tests); pytest uses `--basetemp .pytest_tmp` because the default Windows temp path is inaccessible in this sandbox.
- C03 downloaded all nine source files at revision `c379a6705fec24a2493fa68e011692605f44e119`; each language has 200,000 train and 5,000 validation/test rows, all manifest hashes verified, and a second download reused every matching file.
- C03 tested the installed `datasets` API; see `docs/ASSUMPTIONS.md` for the raw JSONL fallback and the observed `text`/`label` schema.
- C04 prepared and validated 1,800 train / 600 validation / 1,500 test records with seed 20261021 and source revision `c379a6705fec24a2493fa68e011692605f44e119`; test support is 610 Negative / 306 Neutral / 584 Positive, differing from the earlier recorded 609 / 310 / 581.
- C04 tests pass (24 total), including deterministic output, seed variation, overlap rejection, label mapping, and subprocess salt stability; `ruff check .` passes.
- C05 tests pass (28 passed, 1 opt-in network test skipped by default); the real mDeBERTa tokenizer smoke test was also run separately and passed on French and Chinese. Only tokenizer assets were downloaded.
- C05 full local checks: `pytest -q --basetemp .pytest_tmp` and `ruff check .` both pass.
- C06 uses only a randomly initialized two-layer, 32-hidden-unit DeBERTa fixture; model factory, freeze/count helpers, AMP policy, and reloadable probe head pass local tests.
- C06 checks: `pytest -q --basetemp .pytest_tmp` passed (32 passed, 1 network test skipped); `ruff check .` passed. No model weights were downloaded.
- C07 added fixed-label metrics, per-language reporting, finite-safe evaluation artifacts, metadata, and run-status/test-lock enforcement; synthetic tests cover lock refusal, repeat-force count, NaN handling, and hand-computed metrics.
- C07 checks: `pytest -q --basetemp .pytest_tmp` passed (37 passed, 1 opt-in network test skipped); `ruff check .` passed. No benchmark metrics were produced from model experiments.
- C08 added finite-checked per-split CLS feature caching, validation-only probe selection, best-head persistence, and a train-majority classifier. The test map includes a poison object proving probe selection never reads test features.
- C08 checks: `pytest -q --basetemp .pytest_tmp` passed (42 passed, 1 opt-in network test skipped); `ruff check .` passed. The real mDeBERTa probe remains PENDING on the GPU laptop.
- C09 added FP32-master AdamW grouping, warmup/decay schedule, gradient accumulation/clipping, AMP/GradScaler policy, finite-loss failure capture, validation-only best saves, resume checkpoints, balanced train-only weights, LoRA, and no-checkpoint dry-run.
- C09 checks: `pytest -q --basetemp .pytest_tmp` passed (50 passed, 1 opt-in network test skipped); `ruff check .` passed. Only tiny random models and synthetic mini-data were used; full fine-tuning and LoRA benchmark runs remain PENDING on the GPU laptop.
- C10 added run/checkpoint autodetection for full, LoRA, probe, and majority models; finite-checked single/batch inference; and normalized last-layer CLS attention with the required diagnostic disclaimer. All weight loads are local-only.
- C10 checks: `pytest -q --basetemp .pytest_tmp` passed (58 passed, 1 opt-in network test skipped); `ruff check .` passed. Tests use tiny random models and cover prediction invariance, multilingual tokenization, invalid markers, LoRA/probe loading, and attention.
- C11 added argparse commands with examples, run metadata and manifest references, test-evaluation lock prechecks, single/batch prediction, and JSON/CSV/Markdown comparison output with manifest mismatch refusal and PENDING runs. CLI subprocess smoke tests cover help, local tiny-checkpoint prediction, and test-lock refusal; comparison tests cover PENDING and hash mismatch.
- C11 checks: `ruff check .` passed; the first sandboxed pytest attempt hit Windows temporary-directory access denials, so the approved full run was executed outside the sandbox and passed (62 passed, 1 opt-in network test skipped). One upstream TorchScript deprecation warning remains. No real model weights or training runs were used.
- C12 added a Streamlit demo that remains usable with no trained run, plus run discovery, INVALID exclusion, comparison/reference loading, benchmark manifest/count/sample browsing, batch CSV/text predictions, attention disclaimer, and limitations views. Example reviews cover en/fr/zh and ar/it are explicitly marked not evaluated.
- C12 checks: `ruff check .` passed; focused Streamlit AppTest and helper checks passed (5 passed), and the full suite passed (67 passed, 1 opt-in network test skipped). The tiny random checkpoint was used only for UI prediction smoke coverage. Streamlit was installed from the declared requirements to enable AppTest.
- C13 added a stand-alone QLoRA causal-LM experiment with its own config, five example file, target-module validation, explicit 4-bit prerequisites, dry-run, and one-step smoke option. It does not import classifier training code and its outputs are isolated from benchmark reports.
- C13 checks: the GPT-2 fixture accepts `c_attn` and rejects `q_proj`; dry-run and one-step CPU LoRA smoke passed. `ruff check .` passed; full pytest passed (70 passed, 1 opt-in network test skipped). All experiment model weights were randomly initialized; no real base-model weights or multi-step model-quality runs were produced.
- C14 completed the README, GPU runbook, limitations/data provenance/assumptions docs, detected-environment checker, results packer, PowerShell/bash sequential GPU scripts, recorded reference JSON, IEEE report, and generated PENDING table. CLI docs checks passed; report decimal checks limit static metrics to the recorded reference. `pdflatex` is not installed, so the report was not compiled.
- C14 checks: `ruff check .` passed; full pytest passed (73 passed, 1 opt-in network test skipped). `python scripts/check_gpu_env.py` ran and reported the detected host state; its no-CUDA warning is environment output, not a GPU-laptop claim.

## Final handoff (C15)

- Implemented: benchmark download/reuse/validation; dataset and tokenizer pipeline; model factory;
  frozen probe; full fine-tuning and LoRA; metrics and test lock; inference/attention; CLI and
  comparisons; Streamlit demo; isolated QLoRA experiment; GPU utilities; provenance, limitations,
  and IEEE report artifacts.
- Verification: `ruff check .` passes. Final `pytest -q`: 73 passed, 1 opt-in network test
  skipped. The network tokenizer test was run separately once and passed (1 passed; tokenizer
  assets only). Fresh `.venv_c15` installation succeeded; CLI help, benchmark validation, tiny
  random-model train dry-run, and Streamlit AppTest (0 exceptions, 4 tabs) passed. The report
  generator’s cross-drive output handling was fixed after the exact `pytest -q` run exposed it.
- No `runs/` or `reports/results/` artifacts exist. This workspace produced no real-model run
  metrics. Tiny random models were used only in tests and dry-run smoke checks.
- PENDING on the GPU laptop: majority, mDeBERTa probe, full fine-tuning, optional balanced,
  XLM-R, and LoRA runs; validation-based checkpoint selection; one final test evaluation per
  selected checkpoint; comparison and result bundle. See `docs/GPU_RUNBOOK.md`. Core commands:

  ```console
  python -m sentiment validate-data
  python -m sentiment majority
  python -m sentiment probe --config configs/probe_mdeberta.yaml --dry-run
  python -m sentiment probe --config configs/probe_mdeberta.yaml
  python -m sentiment evaluate --run runs/<probe_run> --split validation
  python -m sentiment evaluate --run runs/<probe_run> --split test
  python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run
  python -m sentiment train --config configs/finetune_mdeberta.yaml
  python -m sentiment evaluate --run runs/<full_run> --split validation
  python -m sentiment evaluate --run runs/<full_run> --split test
  python -m sentiment compare --runs runs/<probe_run> runs/<full_run> --out reports/results
  python scripts/pack_results.py --runs runs --out reports/run_bundle.zip
  ```

- GPU transfer: copy the project code, `configs/`, `requirements.txt`, and
  `data/amazon_reviews_benchmark/` with its manifest. Raw source data is only needed if
  regenerating the benchmark. Do not transfer the development virtual environment. Return
  `reports/run_bundle.zip`; add `--include-model` when checkpoint weights need to be copied back.
- Assumptions and limits: the source integer `label` is treated as stars; the current raw schema
  required a JSONL fallback to the legacy dataset loader; the pre-existing benchmark was reused
  and has a different test class distribution from the historical project. Arabic/Italian
  examples are not evaluated. `pdflatex` is unavailable, so the IEEE report was not compiled.
- No unexplained TODO markers remain in `src/`, `app/`, or `scripts/`.

