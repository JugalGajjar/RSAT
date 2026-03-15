"""
evaluate.py — Run evaluation across all baselines and RSAT variants.

Loads a model checkpoint, runs inference, computes all metrics, and
saves per-example predictions plus aggregated results.
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.data.data_formats import (
    RSATExample,
    RSATOutput,
    SYSTEM_PROMPT,
    format_model_input,
)
from src.evaluation.metrics import RSATEvaluator


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def load_model_for_eval(
    model_path: str,
    base_model_name: str | None = None,
    use_qlora: bool = True,
    flash_attn: bool = False,
):
    """
    Load a model for inference.  Handles:
      - Full model checkpoints
      - LoRA adapter checkpoints (requires base_model_name)
      - bf16 (H100) or 4-bit QLoRA (A100) loading
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = Path(model_path) / "adapter_config.json"
    attn_impl = "flash_attention_2" if flash_attn else None

    if adapter_cfg.exists() and base_model_name:
        # LoRA adapter checkpoint
        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if use_qlora:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        # Full model (or base model for zero-shot baseline)
        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception:
            # Fallback to 4-bit if OOM
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════


def generate_single(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Generate a single response from chat messages."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    output_ids = generated[0][prompt_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def generate_outputs(
    model,
    tokenizer,
    examples: list[RSATExample],
    max_new_tokens: int = 1024,
) -> list[RSATOutput]:
    """Run RSAT-style inference on a list of examples."""
    outputs: list[RSATOutput] = []
    for ex in tqdm(examples, desc="Generating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_model_input(ex)},
        ]
        text = generate_single(model, tokenizer, messages, max_new_tokens)
        outputs.append(RSATOutput.from_model_text(text))
    return outputs


def generate_posthoc_outputs(
    model,
    tokenizer,
    examples: list[RSATExample],
    max_new_tokens: int = 1024,
) -> list[RSATOutput]:
    """
    Baseline 3: Post-hoc attribution.
    Step 1: Generate answer with chain-of-thought.
    Step 2: Ask the model to cite cells in a second pass.
    """
    outputs: list[RSATOutput] = []
    for ex in tqdm(examples, desc="Post-hoc baseline"):
        # Step 1 — CoT answer
        cot_prompt = (
            "Given the following table, answer the question with step-by-step reasoning.\n\n"
            f"Table:\n{ex.table.serialize_flat()}\n\n"
            f"Question: {ex.question}\n\n"
            "Think step by step, then give your final answer."
        )
        cot_messages = [{"role": "user", "content": cot_prompt}]
        cot_response = generate_single(model, tokenizer, cot_messages, max_new_tokens)

        # Step 2 — Ask for citations
        cite_prompt = (
            "Given the table below and your previous reasoning, cite the specific cells "
            "that support each step.\n\n"
            f"Table (with indices):\n{ex.table.serialize_indexed()}\n\n"
            f"Your reasoning:\n{cot_response}\n\n"
            "Now format this as JSON with cell citations:\n"
            "{\n"
            '  "reasoning_steps": [\n'
            '    {"step": "your reasoning", "cited_cells": [[row_i, col_j], ...]},\n'
            "    ...\n"
            "  ],\n"
            '  "answer": "your answer"\n'
            "}"
        )
        cite_messages = [{"role": "user", "content": cite_prompt}]
        cite_response = generate_single(model, tokenizer, cite_messages, max_new_tokens)
        outputs.append(RSATOutput.from_model_text(cite_response))

    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def load_eval_examples(path: str) -> list[RSATExample]:
    examples: list[RSATExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(RSATExample.from_json(line))
    return examples


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="RSAT evaluation runner.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--posthoc",
        action="store_true",
        help="Use post-hoc attribution baseline (Baseline 3).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    model_path = args.model_path or eval_cfg.get("model_path", "checkpoints/grpo/final")
    base_model = eval_cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    out_dir = Path(args.output_dir or eval_cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens: int = eval_cfg.get("max_new_tokens", 1024)
    max_eval: int | None = eval_cfg.get("max_eval_examples")

    # Build eval file list
    eval_files: list[str] = eval_cfg.get("eval_files", [])
    if args.eval_data:
        eval_files = [args.eval_data]

    # Load model
    use_qlora = eval_cfg.get("use_qlora", True)
    flash_attn = eval_cfg.get("flash_attn", False)
    print(f"Loading model from {model_path} ({'4-bit' if use_qlora else 'bf16'}) ...")
    model, tokenizer = load_model_for_eval(
        model_path,
        base_model_name=base_model,
        use_qlora=use_qlora,
        flash_attn=flash_attn,
    )

    # Evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = RSATEvaluator(
        faithfulness_backend=eval_cfg.get("faithfulness_backend", "nli"),
        device=device,
    )

    # Run on each eval file
    all_results: dict[str, dict] = {}

    for eval_file in eval_files:
        ds_name = Path(eval_file).stem
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name}  ({eval_file})")
        print(f"{'='*60}")

        examples = load_eval_examples(eval_file)
        if max_eval:
            examples = examples[:max_eval]
        print(f"  {len(examples)} examples")

        if args.posthoc:
            outputs = generate_posthoc_outputs(
                model, tokenizer, examples, max_new_tokens
            )
        else:
            outputs = generate_outputs(model, tokenizer, examples, max_new_tokens)

        metrics = evaluator.evaluate(outputs, examples)
        all_results[ds_name] = {"aggregate": metrics.to_dict()}
        print(f"\n  Aggregate:")
        print(metrics)

        # Per-dataset breakdown (WTQ / FeTaQA / TabFact)
        per_ds = evaluator.evaluate_by_dataset(outputs, examples)
        if len(per_ds) > 1:
            for sub_ds, sub_metrics in per_ds.items():
                all_results[ds_name][sub_ds] = sub_metrics.to_dict()
                print(f"\n  {sub_ds} ({sub_metrics.num_examples} examples):")
                print(sub_metrics)

        # Save per-example predictions
        preds_path = out_dir / f"{ds_name}_predictions.jsonl"
        with open(preds_path, "w", encoding="utf-8") as f:
            for ex, out in zip(examples, outputs):
                f.write(
                    json.dumps(
                        {
                            "id": ex.id,
                            "source_dataset": ex.source_dataset,
                            "question": ex.question,
                            "gold_answer": ex.answer,
                            "predicted_answer": out.answer,
                            "reasoning_steps": [
                                s.to_dict() for s in out.reasoning_steps
                            ],
                            "parse_success": out.parse_success,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  Predictions → {preds_path}")

    # Save aggregated results
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results → {results_path}")


if __name__ == "__main__":
    main()