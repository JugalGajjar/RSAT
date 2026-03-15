# RSAT: Reward-Guided Structured Reasoning with Attribution Traces

## Project Structure

```
rsat/
├── configs/
│   ├── sft_config.yaml            # Phase 1: Supervised fine-tuning
│   ├── grpo_config.yaml           # Phase 2: GRPO RL training
│   └── eval_config.yaml           # Evaluation settings
├── data/
│   ├── raw/                       # Downloaded datasets (WTQ, FeTaQA, TabFact)
│   ├── processed/                 # Cleaned, unified JSONL format
│   └── synthetic/                 # LLM-generated SFT training examples
├── src/
│   ├── data/                      # Data loading, formatting, SFT generation
│   │   ├── data_formats.py        # Canonical schemas: Table, RSATExample, RSATOutput
│   │   ├── prepare_wtq.py         # WikiTableQuestions → RSAT format
│   │   ├── prepare_fetaqa.py      # FeTaQA → RSAT format
│   │   ├── prepare_tabfact.py     # TabFact → RSAT format
│   │   └── generate_sft_data.py.  # Generate gold reasoning traces with citations
│   ├── rewards/                   # All reward components
│   │   ├── answer_reward.py       # R_answer: exact match + F1
│   │   ├── citation_reward.py     # R_citation: structural validity
│   │   ├── faithfulness_reward.py # R_faithfulness: NLI + LLM judge
│   │   ├── parsimony_reward.py    # R_parsimony: over-citation penalty
│   │   └── composite_reward.py    # Combined reward with λ weights
│   ├── training/                  # SFT and GRPO training loops
│   │   ├── sft_train.py           # Phase 1: QLoRA SFT warmup
│   │   └── grpo_train.py          # Phase 2: GRPO RL fine-tuning
│   ├── evaluation/                # Metrics and evaluation pipeline
│   │   ├── metrics.py             # All eval metrics
│   │   └── evaluate.py            # Inference + evaluation runner
│   └── utils/                     # Shared utilities
│       ├── groq_client.py         # Round-robin Groq API client
│       └── table_utils.py         # Table manipulation helpers
├── prompts/
│   └── all_prompts.yaml           # All prompt templates (for paper appendix)
├── scripts/
│   ├── run_full_pipeline.sh       # End-to-end: data → train → eval
│   └── run_ablations.sh           # All ablation experiments
├── tests/
│   └── test_rewards.py            # Unit tests for reward functions
└── requirements.txt
```

## Setup

```bash
# 1. Create environment
conda create -n rsat python=3.10 -y
conda activate rsat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
export GROQ_API_KEYS="key1,key2,key3,key4"  # comma-separated for rotation
export OPENAI_API_KEY="sk-..."              # optional, for SFT data gen

# 4. Run tests locally first (MacBook)
python -m pytest tests/test_rewards.py -v

# 5. Full pipeline (A100)
bash scripts/run_full_pipeline.sh
```

## Data Format

Every dataset is converted into a unified JSONL format:

```json
{
  "id": "wtq_train_42",
  "question": "What country had the most medals?",
  "table": {
    "headers": ["Country", "Gold", "Silver", "Bronze", "Total"],
    "rows": [
      ["USA", "10", "8", "7", "25"],
      ["China", "8", "6", "5", "19"]
    ],
    "caption": null
  },
  "answer": "USA",
  "source_dataset": "wtq",
  "reasoning_trace": [
    {
      "step": "Look at the Total column to find the highest value.",
      "cited_cells": [[0, 4], [1, 4]]
    },
    {
      "step": "USA has 25 total medals, which is the highest.",
      "cited_cells": [[0, 0], [0, 4]]
    }
  ],
  "metadata": {"all_answers": ["USA"]}
}
```

## Model Output Format

The model produces structured JSON with cell-level citations:

```json
{
  "reasoning_steps": [
    {"step": "...", "cited_cells": [[row_i, col_j], ...]},
    ...
  ],
  "answer": "..."
}
```