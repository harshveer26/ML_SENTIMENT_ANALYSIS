"""Isolated five-example causal-LM LoRA experiment; never a benchmark classifier result."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

BANNER = (
    "Experimental. Trained on 5 examples. Does not establish model quality. "
    "Not comparable to the 3-class benchmark."
)
HERE = Path(__file__).resolve().parent


def validate_targets(model: torch.nn.Module, target_modules: list[str]) -> None:
    """Require every requested PEFT target name to occur in the loaded model."""
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(target_modules) - module_names)
    if missing:
        raise ValueError(
            f"LoRA target module(s) {missing} do not exist in {type(model).__name__}; "
            f"available leaf examples: {sorted(module_names)[:20]}"
        )


def load_examples(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 5 or any(not row.get("prompt") or not row.get("response") for row in rows):
        raise ValueError("prompts.jsonl must contain exactly five non-empty prompt/response pairs.")
    return rows


def _tiny_batch(rows: list[dict[str, str]], max_length: int) -> dict[str, torch.Tensor]:
    texts = [f"{row['prompt']} {row['response']}" for row in rows]
    sequences = []
    for text in texts:
        # Deterministic toy token IDs for a random-weight smoke fixture only.
        ids = [1 + (sum(map(ord, token)) % 126) for token in text.split()][:max_length]
        sequences.append(ids or [1])
    length = min(max(map(len, sequences)), max_length)
    input_ids = torch.zeros((len(sequences), length), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(sequences):
        ids = ids[:length]
        input_ids[index, : len(ids)] = torch.tensor(ids)
        attention_mask[index, : len(ids)] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _real_batch(
    tokenizer: Any, rows: list[dict[str, str]], max_length: int
) -> dict[str, torch.Tensor]:
    texts = [f"{row['prompt']} {row['response']}" for row in rows]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded["labels"] = encoded["input_ids"].clone()
    encoded["labels"][encoded["attention_mask"] == 0] = -100
    return encoded


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_examples(HERE / "prompts.jsonl")
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    target_modules = list(config["lora_target_modules"])

    if args.tiny_fixture or args.dry_run:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=128,
                n_positions=int(config["max_length"]),
                n_ctx=int(config["max_length"]),
                n_embd=32,
                n_layer=2,
                n_head=4,
            )
        )
        batch = _tiny_batch(rows, int(config["max_length"]))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = None
        if config.get("use_4bit"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit QLoRA requires CUDA; set use_4bit: false or use a GPU host."
                )
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError(
                    "4-bit QLoRA requires a compatible bitsandbytes installation."
                ) from exc
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
        )
        batch = _real_batch(tokenizer, rows, int(config["max_length"]))

    validate_targets(model, target_modules)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    print(BANNER)
    if args.dry_run:
        model.eval()
        with torch.no_grad():
            output = model(**batch)
        if not torch.isfinite(output.logits).all():
            raise ValueError("Tiny-model dry-run produced non-finite logits.")
        print(
            json.dumps({"dry_run": True, "examples": len(rows), "target_modules": target_modules})
        )
        return

    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    steps = 0
    for _ in range(1):
        for index in range(len(rows)):
            one = {key: value[index : index + 1] for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**one).loss
            if not torch.isfinite(loss):
                raise ValueError("QLoRA training loss became non-finite.")
            loss.backward()
            optimizer.step()
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    print(json.dumps({"training_steps": steps, "output_dir": str(output_dir)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny-fixture", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
