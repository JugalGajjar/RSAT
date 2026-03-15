"""
answer_reward.py — R_answer: reward for answer correctness.

Two modes:
  - exact_match: binary 0/1 after normalization
  - f1:          token-level F1 between predicted and gold answer
  - combined:    average of both
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles / punctuation / extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, gold: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold."""
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_answer_reward(
    prediction: str,
    gold: str,
    gold_alternatives: list[str] | None = None,
    mode: str = "f1",
) -> float:
    """
    Compute R_answer.

    Takes the *best* score over all acceptable gold answers.

    Args:
        prediction: Model's predicted answer string.
        gold: Primary gold answer.
        gold_alternatives: Optional list of acceptable alternatives.
        mode: "exact_match" | "f1" | "combined".

    Returns:
        Score in [0, 1].
    """
    all_golds = [gold]
    if gold_alternatives:
        all_golds.extend(gold_alternatives)

    if mode == "exact_match":
        return max(exact_match(prediction, g) for g in all_golds)
    elif mode == "f1":
        return max(token_f1(prediction, g) for g in all_golds)
    elif mode == "combined":
        em = max(exact_match(prediction, g) for g in all_golds)
        f1 = max(token_f1(prediction, g) for g in all_golds)
        return 0.5 * em + 0.5 * f1
    else:
        raise ValueError(f"Unknown answer reward mode: {mode}")