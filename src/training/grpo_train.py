"""
grpo_train.py — Phase 2: Group Relative Policy Optimization (GRPO).

Fine-tunes the SFT-warmup checkpoint using RL with the composite reward:
  R = R_answer + λ₁·R_citation + λ₂·R_faithfulness + λ₃·R_parsimony

Architecture follows DeepSeek-R1: generate G candidate outputs per question,
rank by reward, update policy using group-relative advantages.

TRL version notes:
    This script targets TRL ≥ 0.29.0.  If your installed version differs,
    you may need to adjust the reward function signature — see the docstring
    on ``RSATRewardWrapper`` below.
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
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

from src.data.data_formats import (
    RSATExample,
    RSATOutput,
    SYSTEM_PROMPT,
    format_model_input,
)
from src.rewards.composite_reward import (
    CompositeRewardFunction,
    RewardBreakdown,
    create_reward_function,
)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════


def load_grpo_data(
    data_path: str,
    max_examples: int | None = None,
) -> list[RSATExample]:
    """Load examples for GRPO training."""
    examples: list[RSATExample] = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if line:
                examples.append(RSATExample.from_json(line))
    return examples


def build_prompt_messages(example: RSATExample) -> list[dict]:
    """Build the chat-format prompt for a single example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_model_input(example)},
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Reward wrapper
# ═════════════════════════════════════════════════════════════════════════════


class RSATRewardWrapper:
    """
    Wraps CompositeRewardFunction for use with TRL's GRPOTrainer.

    GRPOTrainer calls  ``reward_func(prompts, completions, **kw)``
    where both ``prompts`` and ``completions`` are lists of conversation
    messages (list[list[dict]]).  Each completion is a list of message
    dicts; the last message (role=assistant) contains the model output.

    We serialise the user-turn text as a lookup key into ``example_map``
    to recover the original RSATExample needed for reward computation.
    """

    def __init__(
        self,
        reward_fn: CompositeRewardFunction,
        example_map: dict[str, RSATExample],
    ):
        self.reward_fn = reward_fn
        self.example_map = example_map
        self.log: list[dict] = []
        self.__name__ = "rsat_composite_reward"  # required by TRL 0.29.0

    def __call__(
        self,
        prompts: list[str] | list[list[dict]],
        completions: list[str] | list[list[dict]],
        **kwargs: Any,
    ) -> list[float]:
        """
        Compute rewards for a batch of (prompt, completion) pairs.

        Handles both str-based and message-based formats depending
        on the TRL version.
        """
        rewards: list[float] = []

        for prompt, completion in zip(prompts, completions):
            # Extract text from different possible formats
            prompt_text = self._extract_user_text(prompt)
            completion_text = self._extract_completion_text(completion)

            # Look up original example
            example = self.example_map.get(prompt_text)
            if example is None:
                rewards.append(-1.0)
                continue

            # Parse model output
            output = RSATOutput.from_model_text(completion_text)

            # Compute composite reward
            breakdown: RewardBreakdown = self.reward_fn(output, example)
            rewards.append(breakdown.total)

            # Log for reward curve analysis
            self.log.append({
                "total": breakdown.total,
                "r_answer": breakdown.r_answer,
                "r_citation": breakdown.r_citation,
                "r_faithfulness": breakdown.r_faithfulness,
                "r_parsimony": breakdown.r_parsimony,
                "parse_success": breakdown.parse_success,
            })

        return rewards

    @staticmethod
    def _extract_user_text(prompt: str | list[dict]) -> str:
        """Extract the user-turn text from a prompt (str or messages)."""
        if isinstance(prompt, str):
            return prompt
        # It's a list of message dicts — find the user turn
        for msg in prompt:
            if msg.get("role") == "user":
                return msg["content"]
        return str(prompt)

    @staticmethod
    def _extract_completion_text(completion: str | list[dict]) -> str:
        """Extract the assistant text from a completion."""
        if isinstance(completion, str):
            return completion
        # List of message dicts — last assistant message
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                return msg["content"]
        return str(completion)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="RSAT Phase 2: GRPO training.")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    base_model_name: str = model_cfg["name"]
    sft_path: str = model_cfg.get("sft_checkpoint", "checkpoints/sft/final")
    output_dir: str = train_cfg.get("output_dir", "checkpoints/grpo")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {sft_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding for generation

    # ── Base model + SFT adapter ─────────────────────────────────────────────
    use_qlora = model_cfg.get("use_qlora", True)
    merge_dir = Path(output_dir) / "merged_sft"

    # Step 1: Merge SFT LoRA into base (always done in bf16 for safety)
    if not merge_dir.exists():
        print(f"Loading base model {base_model_name} (bf16) for SFT merge ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        print(f"Merging SFT adapter from {sft_path} ...")
        model = PeftModel.from_pretrained(base_model, sft_path)
        model = model.merge_and_unload()

        print(f"Saving merged model → {merge_dir} ...")
        model.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)
        del model, base_model
        torch.cuda.empty_cache()
    else:
        print(f"Using cached merged model from {merge_dir}")

    # Step 2: Reload merged model for GRPO training
    if use_qlora:
        print(f"Reloading merged model (4-bit QLoRA) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            merge_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    else:
        print(f"Reloading merged model (bf16) ...")
        model = AutoModelForCausalLM.from_pretrained(
            merge_dir,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if model_cfg.get("flash_attn") else None,
        )

    # ── Fresh LoRA for RL phase ──────────────────────────────────────────────
    lora = model_cfg.get("lora_rl", model_cfg.get("lora", {}))
    peft_config = LoraConfig(
        r=lora.get("r", 32),
        lora_alpha=lora.get("alpha", 64),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training data ────────────────────────────────────────────────────────
    print(f"Loading training data from {data_cfg['train_file']} ...")
    examples = load_grpo_data(
        data_cfg["train_file"],
        max_examples=data_cfg.get("max_examples"),
    )
    print(f"  {len(examples)} training examples")

    # Build prompt → example lookup  (key = user-turn text)
    example_map: dict[str, RSATExample] = {}
    prompt_records: list[dict] = []
    for ex in examples:
        user_text = format_model_input(ex)
        example_map[user_text] = ex
        prompt_records.append({
            "prompt": build_prompt_messages(ex),
        })

    dataset = Dataset.from_list(prompt_records)

    # Validation set
    eval_dataset = None
    if data_cfg.get("val_file"):
        max_eval = data_cfg.get("max_eval_examples", 200)
        print(f"Loading validation data from {data_cfg['val_file']} (cap={max_eval}) ...")
        val_examples = load_grpo_data(data_cfg["val_file"], max_examples=max_eval)
        print(f"  {len(val_examples)} validation examples")
        val_records: list[dict] = []
        for ex in val_examples:
            user_text = format_model_input(ex)
            example_map[user_text] = ex  # reward wrapper needs these too
            val_records.append({
                "prompt": build_prompt_messages(ex),
            })
        eval_dataset = Dataset.from_list(val_records)

    # ── Reward function ──────────────────────────────────────────────────────
    print("Initialising composite reward function ...")
    reward_fn = create_reward_function(cfg)
    reward_wrapper = RSATRewardWrapper(reward_fn, example_map)

    print(f"  λ_cit={reward_fn.lambda_citation}  "
          f"λ_faith={reward_fn.lambda_faithfulness}  "
          f"λ_pars={reward_fn.lambda_parsimony}")

    # ── GRPO config ──────────────────────────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 8),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        bf16=True,
        logging_steps=1,
        save_steps=train_cfg.get("save_steps", 100),
        # Evaluation
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        report_to="wandb" if train_cfg.get("use_wandb", True) else "none",
        run_name=train_cfg.get("run_name", "rsat-grpo"),
        # GRPO-specific
        num_generations=train_cfg.get("group_size", 8),
        max_completion_length=train_cfg.get("max_completion_length", 1024),
        # max_prompt_length=train_cfg.get("max_prompt_length", 1536),
        temperature=train_cfg.get("temperature", 0.7),
        loss_type="grpo",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_wrapper],
        peft_config=peft_config,
    )

    print(f"Starting GRPO training  (G={grpo_args.num_generations}) ...")
    trainer.train()

    # Save LoRA adapter checkpoint (for resuming / inspection)
    adapter_path = str(Path(output_dir) / "adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"GRPO adapter saved → {adapter_path}")

    # Merge GRPO LoRA into the merged SFT base → fully self-contained model
    final_path = str(Path(output_dir) / "final")
    print(f"Merging GRPO adapter into base for final checkpoint ...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    del merged_model
    torch.cuda.empty_cache()
    print(f"GRPO merged model saved → {final_path}")

    # Save reward logs for reward-curve analysis
    log_path = Path(output_dir) / "reward_logs.jsonl"
    with open(log_path, "w") as f:
        for entry in reward_wrapper.log:
            f.write(json.dumps(entry) + "\n")
    print(f"Reward logs ({len(reward_wrapper.log)} entries) → {log_path}")


if __name__ == "__main__":
    main()