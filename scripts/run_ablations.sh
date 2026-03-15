#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RSAT — Ablation Experiments (Multi-Model)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash run_ablations.sh                    # all models
#   bash run_ablations.sh qwen_7b            # one model only
#   bash run_ablations.sh qwen_7b eval_only  # eval only (training done)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL=${1:-"all"}
MODE=${2:-"full"}  # "full" | "eval_only"

ALL_MODELS="qwen_1.5b qwen_3b qwen_7b llama_1b llama_3b llama_8b"
if [ "$MODEL" != "all" ]; then
    ALL_MODELS="$MODEL"
fi

# Ablation definitions: name → (λ_cit, λ_faith, λ_pars)
declare -A CIT FAITH PARS
CIT[no_faithfulness]=0.3;   FAITH[no_faithfulness]=0.0;  PARS[no_faithfulness]=0.2
CIT[no_parsimony]=0.3;      FAITH[no_parsimony]=0.5;     PARS[no_parsimony]=0.0
CIT[no_citation]=0.0;       FAITH[no_citation]=0.5;      PARS[no_citation]=0.2

for M in $ALL_MODELS; do
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Ablations for: $M"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    BASE_GRPO_CFG="configs/$M/grpo_config.yaml"
    EVAL_CFG="configs/$M/eval_config.yaml"

    if [ ! -f "$BASE_GRPO_CFG" ]; then
        echo "  ⚠ Config not found: $BASE_GRPO_CFG — skipping $M"
        continue
    fi

    for abl in no_faithfulness no_parsimony no_citation; do
        echo ""
        echo "═════════════════════════════════════════"
        echo "  $M / $abl"
        echo "  λ_cit=${CIT[$abl]}  λ_faith=${FAITH[$abl]}  λ_pars=${PARS[$abl]}"
        echo "═════════════════════════════════════════"

        ABL_CFG="configs/$M/grpo_${abl}.yaml"
        ABL_CKPT="checkpoints/$M/grpo_${abl}"
        ABL_RESULTS="results/$M/ablation_${abl}"

        # Generate ablation config from base
        cp "$BASE_GRPO_CFG" "$ABL_CFG"

        # Patch λ values
        sed -i "s/lambda_citation:.*/lambda_citation: ${CIT[$abl]}/"         "$ABL_CFG"
        sed -i "s/lambda_faithfulness:.*/lambda_faithfulness: ${FAITH[$abl]}/" "$ABL_CFG"
        sed -i "s/lambda_parsimony:.*/lambda_parsimony: ${PARS[$abl]}/"      "$ABL_CFG"

        # Patch output dir and run name (preserve indentation)
        sed -i "s|output_dir:.*checkpoints.*|output_dir: \"${ABL_CKPT}\"|" "$ABL_CFG"
        sed -i "s|run_name:.*|run_name: \"rsat-grpo-${M}-${abl}\"|" "$ABL_CFG"

        # ── Train ────────────────────────────────────────────────
        if [ "$MODE" = "full" ]; then
            echo "  Training..."
            python -m src.training.grpo_train --config "$ABL_CFG"
        fi

        # ── Evaluate ─────────────────────────────────────────────
        echo "  Evaluating..."
        python -m src.evaluation.evaluate \
            --config "$EVAL_CFG" \
            --model_path "${ABL_CKPT}/final" \
            --output_dir "$ABL_RESULTS"

        echo "  ✓ $M / $abl done → $ABL_RESULTS"
    done

    echo ""
    echo "All ablations for $M complete"
done

# ── Compile Results ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Ablation Results Summary"
echo "═══════════════════════════════════════════════════════════════"

python3 << 'PYEOF'
import json, os

models = ["qwen_1.5b", "qwen_3b", "qwen_7b", "llama_1b", "llama_3b", "llama_8b"]
ablations = [
    ("RSAT (full)", "rsat_full"),
    ("-Faithfulness", "ablation_no_faithfulness"),
    ("-Parsimony", "ablation_no_parsimony"),
    ("-Citation", "ablation_no_citation"),
]

print()
header = f"{'Model':<14} {'Variant':<18} {'EM':>6} {'F1':>6} {'Cite':>6} {'Faith':>6} {'Pars':>6} {'Fmt%':>6}"
print(header)
print("-" * len(header))

for model in models:
    found = False
    for abl_name, abl_dir in ablations:
        path = f"results/{model}/{abl_dir}/eval_results.json"
        if not os.path.exists(path):
            continue
        found = True
        with open(path) as f:
            data = json.load(f)
        key = list(data.keys())[0]
        m = data[key].get("aggregate", data[key])
        print(f"{model:<14} {abl_name:<18} {m.get('answer_em',0):>6.3f} {m.get('answer_f1',0):>6.3f} "
              f"{m.get('citation_validity',0):>6.3f} {m.get('faithfulness',0):>6.3f} "
              f"{m.get('parsimony',0):>6.3f} {m.get('format_success_rate',0):>6.3f}")
    if found:
        print()
PYEOF

echo "Done. Results → results/<model>/ablation_*/"
