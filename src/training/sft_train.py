"""
sft_train.py — Phase 1: Supervised Fine-Tuning warmup.

Fine-tunes Qwen2.5-7B-Instruct on gold reasoning + citation examples
using QLoRA (4-bit quantisation + LoRA adapters).

The goal is to teach the model the structured output format *before*
the RL phase refines attribution quality.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.data.data_formats import (
    RSATExample,
    SYSTEM_PROMPT,
    format_model_input,
    format_sft_target,
)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════


def load_sft_data(data_path: str) -> list[dict]:
    """
    Load SFT examples and format as chat-style dicts ready for
    the tokenizer's ``apply_chat_template``.
    """
    records: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = RSATExample.from_json(line)
            if not ex.reasoning_trace:
                continue  # skip examples without gold traces
            prompt = format_model_input(ex)
            response = format_sft_target(ex)
            records.append({"prompt": prompt, "completion": response})
    return records


def format_as_chat(record: dict, tokenizer) -> dict:
    """Render a record through the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["prompt"]},
        {"role": "assistant", "content": record["completion"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="RSAT Phase 1: SFT warmup.")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model_name: str = model_cfg["name"]
    output_dir: str = train_cfg.get("output_dir", "checkpoints/sft")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ────────────────────────────────────────────────────────────────
    use_qlora = model_cfg.get("use_qlora", True)

    if use_qlora:
        print(f"Loading model {model_name} (4-bit QLoRA) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model {model_name} (bf16) ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if model_cfg.get("flash_attn") else None,
        )
        model.enable_input_require_grads()  # needed for LoRA on non-quantized models

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora = model_cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora.get("r", 64),
        lora_alpha=lora.get("alpha", 128),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Loading SFT data from {data_cfg['train_file']} ...")
    raw = load_sft_data(data_cfg["train_file"])
    formatted = [format_as_chat(r, tokenizer) for r in raw]
    print(f"  {len(formatted)} training examples")
    dataset = Dataset.from_list(formatted)

    # Validation set
    eval_dataset = None
    if data_cfg.get("val_file"):
        print(f"Loading validation data from {data_cfg['val_file']} ...")
        raw_val = load_sft_data(data_cfg["val_file"])
        formatted_val = [format_as_chat(r, tokenizer) for r in raw_val]
        print(f"  {len(formatted_val)} validation examples")
        eval_dataset = Dataset.from_list(formatted_val)

    # ── Training config ──────────────────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_length=train_cfg.get("max_seq_length", 2048),
        bf16=True,
        logging_steps=10,
        # Evaluation & checkpointing
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy="steps" if eval_dataset is not None else "epoch",
        save_steps=train_cfg.get("eval_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) if eval_dataset is not None else False,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,  # lower eval_loss is better
        report_to="wandb" if train_cfg.get("use_wandb", True) else "none",
        run_name=train_cfg.get("run_name", "rsat-sft"),
        dataset_text_field="text",
        packing=False,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training ...")
    trainer.train()

    # Save final checkpoint
    final_path = str(Path(output_dir) / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"SFT model saved → {final_path}")


if __name__ == "__main__":
    main()