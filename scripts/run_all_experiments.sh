#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# RSAT — Master Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════════
# Run all 6 model configs × 4 methods = 24 experiments
#
# Usage:
#   bash run_all_experiments.sh                    # run everything
#   bash run_all_experiments.sh qwen_7b            # run one model only
#   bash run_all_experiments.sh qwen_7b eval_only  # eval only (skip training)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

MODEL=${1:-"all"}
MODE=${2:-"full"}  # "full" | "eval_only"

MODELS="qwen_1.5b qwen_3b qwen_7b llama_1b llama_3b llama_8b"
if [ "$MODEL" != "all" ]; then
    MODELS="$MODEL"
fi

for M in $MODELS; do
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Model: $M"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    CFG_DIR="configs/$M"
    SFT_CFG="$CFG_DIR/sft_config.yaml"
    GRPO_CFG="$CFG_DIR/grpo_config.yaml"
    EVAL_CFG="$CFG_DIR/eval_config.yaml"

    # Read base model name from eval config
    BASE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$EVAL_CFG'))['evaluation']['base_model'])")
    SFT_PATH="checkpoints/$M/sft/final"
    GRPO_PATH="checkpoints/$M/grpo/final"

    # ── Phase 1: SFT ────────────────────────────────────────────────
    if [ "$MODE" = "full" ]; then
        echo ""
        echo "── Phase 1: SFT Training ($M) ──"
        python -m src.training.sft_train --config "$SFT_CFG"
    fi

    # ── Phase 2: GRPO ───────────────────────────────────────────────
    if [ "$MODE" = "full" ]; then
        echo ""
        echo "── Phase 2: GRPO Training ($M) ──"
        python -m src.training.grpo_train --config "$GRPO_CFG"
    fi

    # ── Phase 3: Evaluation (4 methods) ─────────────────────────────
    echo ""
    echo "── Phase 3: Evaluation ($M) ──"

    # Experiment 1: Zero-shot
    echo "  → Zero-shot baseline"
    python -m src.evaluation.evaluate \
        --config "$EVAL_CFG" \
        --model_path "$BASE_MODEL" \
        --output_dir "results/$M/baseline_zeroshot"

    # Experiment 2: SFT-only
    echo "  → SFT-only baseline"
    python -m src.evaluation.evaluate \
        --config "$EVAL_CFG" \
        --model_path "$SFT_PATH" \
        --output_dir "results/$M/baseline_sft"

    # Experiment 3: Post-hoc attribution
    echo "  → Post-hoc baseline"
    python -m src.evaluation.evaluate \
        --config "$EVAL_CFG" \
        --model_path "$BASE_MODEL" \
        --output_dir "results/$M/baseline_posthoc" \
        --posthoc

    # Experiment 4: RSAT (ours)
    echo "  → RSAT (full pipeline)"
    python -m src.evaluation.evaluate \
        --config "$EVAL_CFG" \
        --output_dir "results/$M/rsat_full"

    echo ""
    echo "$M complete"
done

# ── Compile Results ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Compiling results table"
echo "═══════════════════════════════════════════════════════════════"

python3 << 'PYEOF'
import json, os

models = ["qwen_1.5b", "qwen_3b", "qwen_7b", "llama_1b", "llama_3b", "llama_8b"]
methods = [
    ("Zero-Shot", "baseline_zeroshot"),
    ("SFT-Only", "baseline_sft"),
    ("Post-Hoc", "baseline_posthoc"),
    ("RSAT", "rsat_full"),
]

print()
header = f"{'Model':<14} {'Method':<12} {'EM':>6} {'F1':>6} {'Cite':>6} {'Faith':>6} {'Pars':>6} {'Fmt%':>6}"
print(header)
print("-" * len(header))

for model in models:
    for method_name, method_dir in methods:
        path = f"results/{model}/{method_dir}/eval_results.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        key = list(data.keys())[0]
        m = data[key].get("aggregate", data[key])
        print(f"{model:<14} {method_name:<12} {m.get('answer_em',0):>6.3f} {m.get('answer_f1',0):>6.3f} "
              f"{m.get('citation_validity',0):>6.3f} {m.get('faithfulness',0):>6.3f} "
              f"{m.get('parsimony',0):>6.3f} {m.get('format_success_rate',0):>6.3f}")
    print()
PYEOF

echo "Done. Results saved in results/<model>/<method>/eval_results.json"
