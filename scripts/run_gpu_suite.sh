#!/usr/bin/env bash
set -euo pipefail

python scripts/check_gpu_env.py
python -m sentiment validate-data
python -m sentiment majority

python -m sentiment probe --config configs/probe_mdeberta.yaml --dry-run
python -m sentiment probe --config configs/probe_mdeberta.yaml

python -m sentiment train --config configs/finetune_mdeberta.yaml --dry-run
python -m sentiment train --config configs/finetune_mdeberta.yaml

# Final test evaluation is deliberately separate; select each checkpoint on validation first.
# python -m sentiment evaluate --run runs/<probe_run> --split validation
# python -m sentiment evaluate --run runs/<probe_run> --split test
# python -m sentiment evaluate --run runs/<full_run> --split validation
# python -m sentiment evaluate --run runs/<full_run> --split test
# python -m sentiment compare --runs runs/<probe_run> runs/<full_run> --out reports/results
# python scripts/pack_results.py --runs runs --out reports/run_bundle.zip
