# Isolated QLoRA causal-LM experiment

> **Experimental. Trained on 5 examples. Does not establish model quality. Not comparable to the 3-class benchmark.**

This experiment is intentionally separate from `src/sentiment`: its output directory is
`experiments/qlora_causal_lm/outputs/`, and benchmark evaluation and reports do not read it.
The five multilingual prompt/response pairs are in `prompts.jsonl`.

Run a no-download smoke pass against a randomly initialized tiny GPT-2 fixture:

```powershell
python experiments/qlora_causal_lm/train_qlora.py --dry-run
```

For a one-step CPU LoRA smoke run:

```powershell
python experiments/qlora_causal_lm/train_qlora.py --tiny-fixture --max-steps 1
```

For an actual experiment, edit `config.yaml` to select a model and matching target modules,
then run without `--tiny-fixture`. GPT-2 family models use `c_attn`; Llama/Mistral-style
models generally use `q_proj` and `v_proj`. The script inspects module names and fails if a
configured target is absent. Model weights are downloaded by Transformers on a real run.

4-bit mode requires a CUDA GPU and a working `bitsandbytes` installation. Set `use_4bit: true`
only after validating those dependencies. With `use_4bit: false`, ordinary LoRA can run on CPU
for a tiny fixture or on a compatible accelerator for a real base model. Larger base models
require hardware appropriate to their weight size; this is not a benchmark runbook.
