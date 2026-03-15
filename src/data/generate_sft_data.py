"""
generate_sft_data.py — Generate supervised fine-tuning data with gold
reasoning traces and validated cell citations.

Uses Groq (Llama-3.3-70B) or OpenAI (GPT-4o) to generate chain-of-thought
reasoning with explicit cell citations for each example. Generated traces
are automatically validated: any hallucinated cell coordinates are removed,
and examples with zero valid citations are discarded.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from src.data.data_formats import RSATExample, ReasoningStep, Table

# ═════════════════════════════════════════════════════════════════════════════
# Prompt template
# ═════════════════════════════════════════════════════════════════════════════

TRACE_GENERATION_PROMPT = """\
You are generating training data for a table QA system that must cite its sources.

Given a table and question, produce a JSON response with step-by-step reasoning
where each step cites specific table cells.

Table (with [row_index, col_index] labels):
{table_indexed}

Question: {question}
Correct answer: {answer}

Generate 2-4 reasoning steps that lead to the answer.  Each step MUST cite the
exact cells it uses as [row_index, col_index] coordinates.

Rules:
- Only cite cells that ACTUALLY EXIST in the table.
- Only cite cells that DIRECTLY SUPPORT the step's claim.
- Be precise — cite the minimum cells needed.
- Each step should make one clear logical point.
- The final step should arrive at the answer.

Respond with ONLY this JSON (no markdown, no explanation):
{{"reasoning_steps": [{{"step": "...", "cited_cells": [[row_i, col_j], ...]}}, ...], "answer": "{answer}"}}\
"""

SYSTEM_MSG = "You are a precise data annotator. Respond ONLY with valid JSON."


# ═════════════════════════════════════════════════════════════════════════════
# Trace generation backends
# ═════════════════════════════════════════════════════════════════════════════


def _build_prompt(example: RSATExample) -> str:
    return TRACE_GENERATION_PROMPT.format(
        table_indexed=example.table.serialize_indexed(),
        question=example.question,
        answer=example.answer,
    )


def generate_trace_groq(
    example: RSATExample,
    groq_client,
    use_cache: bool = True,
) -> dict | None:
    """Generate a reasoning trace using Groq (Llama-3.3-70B)."""
    try:
        response = groq_client.chat(
            user_message=_build_prompt(example),
            system_message=SYSTEM_MSG,
            use_cache=use_cache,
        )
        return _parse_and_validate(response, example.table)
    except Exception as exc:
        print(f"  Groq error: {exc}")
        return None


def generate_trace_openai(
    example: RSATExample,
    client,
) -> dict | None:
    """Generate a reasoning trace using OpenAI GPT-4o."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": _build_prompt(example)},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        return _parse_and_validate(text, example.table)
    except Exception as exc:
        print(f"  OpenAI error: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Parsing + validation
# ═════════════════════════════════════════════════════════════════════════════


def _parse_and_validate(text: str, table: Table) -> dict | None:
    """Parse generated JSON and validate that all cited cells exist."""
    try:
        cleaned = text.strip()
        # Strip markdown fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]

        parsed = json.loads(cleaned)
        steps_raw = parsed.get("reasoning_steps", [])
        if not steps_raw:
            return None

        validated_steps: list[dict] = []
        for s in steps_raw:
            step_text = s.get("step", "").strip()
            if not step_text:
                continue
            valid_cells: list[list[int]] = []
            for cell in s.get("cited_cells", []):
                if (
                    isinstance(cell, (list, tuple))
                    and len(cell) >= 2
                    and table.cell_exists(int(cell[0]), int(cell[1]))
                ):
                    valid_cells.append([int(cell[0]), int(cell[1])])
            if valid_cells:
                validated_steps.append(
                    {"step": step_text, "cited_cells": valid_cells}
                )

        if not validated_steps:
            return None

        return {
            "reasoning_steps": validated_steps,
            "answer": str(parsed.get("answer", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Generation with retry
# ═════════════════════════════════════════════════════════════════════════════


def generate_with_retry(
    example: RSATExample,
    gen_fn,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict | None:
    """
    Attempt trace generation with retries on parse failure.

    On retry, cache is bypassed (for Groq) to get a fresh response,
    since the same prompt might produce a different (valid) output
    with a new API call.
    """
    for attempt in range(max_retries):
        # First attempt uses cache; retries bypass it
        if hasattr(gen_fn, '__self__'):
            # It's a bound method — we handle cache outside
            trace = gen_fn(example)
        else:
            trace = gen_fn(example)

        if trace is not None:
            return trace

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT training data with gold reasoning traces."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/wtq_train.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/synthetic/sft_train.jsonl",
    )
    parser.add_argument("--num_examples", type=int, default=3000)
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq"],
        default="groq",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retry attempts per example on parse failure.",
    )
    parser.add_argument(
        "--batch_pause_every",
        type=int,
        default=25,
        help="Pause after every N examples to soften rate limits.",
    )
    parser.add_argument(
        "--batch_pause_seconds",
        type=float,
        default=5.0,
        help="How many seconds to pause between batches.",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Print N processed examples for spot-checking (0=off).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load input examples
    examples: list[RSATExample] = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(RSATExample.from_json(line))
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    # Subsample
    if args.num_examples < len(examples):
        examples = random.sample(examples, args.num_examples)
    print(f"Generating traces for {len(examples)} examples via {args.provider} ...")
    print(f"  max_retries={args.max_retries}  pause_every={args.batch_pause_every}  "
          f"pause_sec={args.batch_pause_seconds}")

    # Initialize provider
    if args.provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        gen_fn = lambda ex: generate_trace_openai(ex, client)
    else:
        from src.utils.groq_client import GroqRotatingClient

        groq_client = GroqRotatingClient(model="llama-3.3-70b-versatile")

        def gen_fn_groq(ex: RSATExample, use_cache: bool = True) -> dict | None:
            return generate_trace_groq(ex, groq_client, use_cache=use_cache)

        gen_fn = gen_fn_groq

    # Generate
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    shown = 0
    processed = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(examples, desc="Generating SFT data")):

            # ── Rate limit pause ──────────────────────────────────────
            if i > 0 and i % args.batch_pause_every == 0:
                tqdm.write(
                    f"  ... pausing {args.batch_pause_seconds}s after {i} examples "
                    f"({success} ok, {failed} failed)"
                )
                time.sleep(args.batch_pause_seconds)

            # ── Generate with retry ───────────────────────────────────
            trace = None
            for attempt in range(args.max_retries):
                # First attempt uses cache; retries bypass it for a fresh response
                use_cache = (attempt == 0)
                if args.provider == "groq":
                    trace = generate_trace_groq(ex, groq_client, use_cache=use_cache)
                else:
                    trace = gen_fn(ex)

                if trace is not None:
                    break
                if attempt < args.max_retries - 1:
                    time.sleep(2.0)  # brief pause between retries

            processed += 1

            if trace is None:
                failed += 1
                continue

            # ── Save ──────────────────────────────────────────────────
            ex.reasoning_trace = [
                ReasoningStep.from_dict(s) for s in trace["reasoning_steps"]
            ]
            fout.write(ex.to_json() + "\n")
            success += 1

            # ── Show examples for spot-checking ───────────────────────
            if args.show_examples > 0 and shown < args.show_examples:
                shown += 1
                tqdm.write(f"\n  ── Example {shown}/{args.show_examples} ──")
                tqdm.write(f"  Q: {ex.question}")
                tqdm.write(f"  A: {ex.answer}")
                tqdm.write(f"  Table: {ex.table.num_rows} rows × {ex.table.num_cols} cols")
                for j, step in enumerate(ex.reasoning_trace):
                    cells_text = ex.table.get_cells_text(step.cited_cells)
                    tqdm.write(f"  Step {j+1}: {step.step}")
                    tqdm.write(f"    Cells: {step.cited_cells} → {cells_text}")
                tqdm.write("")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Success: {success}/{processed}  ({success/max(processed,1)*100:.1f}%)")
    print(f"  Failed:  {failed}/{processed}  ({failed/max(processed,1)*100:.1f}%)")
    print(f"  Saved → {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()