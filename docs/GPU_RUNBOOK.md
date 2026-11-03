# GPU laptop runbook

Real model training and final test-set evaluation are pending. Keep this project and its
benchmark manifest identical on the training laptop; `validate-data` checks benchmark integrity.

## Setup

PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA-enabled torch wheel matching the host and driver first.
python -m pip install -r requirements.txt
python -m pip install -e .
```

Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled torch wheel matching the host and driver first.
python -m pip install -r requirements.txt
python -m pip install -e .
```

Copy the project directory including `data/amazon_reviews_benchmark/`, or clone the same
revision and copy that data directory. Verify the environment and data:

```console
python scripts/check_gpu_env.py
python -m sentiment validate-data
```

The memory-based batch and LoRA suggestions from `check_gpu_env.py` are heuristics, not
measured recommendations for your specific workload.

## Run order

Start with the majority baseline, then frozen probe, then full fine-tuning. Do each dry-run
before the corresponding real run. Keep validation evaluation separate from the one final test
evaluation after checkpoint selection.

```console
python -m sentiment majority
python -m sentiment probe --config configs/probe_mdeberta.yaml --dry-run
python -m sentiment probe --config configs/probe_mdeberta.yaml
python -m sentiment evaluate --run runs/<probe_run> --split validation
python -m sentiment evaluate --run runs/<probe_run> --split test

python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run
python -m sentiment train --config configs/finetune_mdeberta.yaml
python -m sentiment evaluate --run runs/<full_run> --split validation
python -m sentiment evaluate --run runs/<full_run> --split test
```

Optional runs, each with a dry-run first and their own validation selection and final test
evaluation:

```console
python -m sentiment train --config configs/finetune_mdeberta.yaml --class-weights balanced --dry-run
python -m sentiment train --config configs/finetune_mdeberta.yaml --class-weights balanced
python -m sentiment train --config configs/finetune_xlmr.yaml --dry-run
python -m sentiment train --config configs/finetune_xlmr.yaml
python -m sentiment train --config configs/lora_mdeberta.yaml --peft lora --dry-run
python -m sentiment train --config configs/lora_mdeberta.yaml --peft lora
```

The test split is intended for the final evaluation only. Never use it for checkpoint
selection or tuning. The CLI enforces selection completion before test evaluation.

Compare completed runs and pack results for transfer back to the development workspace:

```console
python -m sentiment compare --runs runs/<probe_run> runs/<full_run> --out reports/results
python scripts/pack_results.py --runs runs --out reports/run_bundle.zip
# Add --include-model only when the trained weights are required in the bundle.
```

The helper `scripts/run_gpu_suite.ps1` and `scripts/run_gpu_suite.sh` perform the dry-run and
training steps sequentially and stop after the first failure. Review each run ID and evaluate
validation then test explicitly; the scripts leave test evaluation as a clearly separated step.

## Troubleshooting

- **Out of memory:** reduce batch size, increase gradient accumulation, set max length to 128,
  or use the LoRA configuration. Treat these as troubleshooting options, not guaranteed fixes.
- **FP16 NaNs:** try bf16 if the device supports it, otherwise use FP32.
- **Laptop sleeps or process stops:** resume only with the same config/run identity, e.g. set
  `run_name` to the existing run folder and pass `--resume`.
- **Slow tokenizer download:** allow the initial tokenizer download to complete before training;
  copy the Hugging Face cache if transferring the environment.
- **Manifest mismatch:** stop and copy the exact benchmark directory used by the development
  workspace; do not compare results across manifests.

`runs/` is the return artifact. The default bundle excludes large model weights but includes
configs, logs, run metadata, predictions, and evaluation metrics. The optional
`--include-model` switch includes saved checkpoints.
