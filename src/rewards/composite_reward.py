"""
composite_reward.py — Multi-component reward for GRPO training.

  R = R_answer + λ₁·R_citation + λ₂·R_faithfulness + λ₃·R_parsimony

Also applies a format penalty for outputs that fail to parse as JSON.
Returns a full RewardBreakdown for per-component logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.data_formats import RSATExample, RSATOutput
from src.rewards.answer_reward import compute_answer_reward
from src.rewards.citation_reward import compute_citation_reward
from src.rewards.faithfulness_reward import create_faithfulness_scorer
from src.rewards.parsimony_reward import compute_parsimony_reward


@dataclass
class RewardBreakdown:
    """Per-component reward scores (logged to W&B for reward curve analysis)."""

    total: float
    r_answer: float
    r_citation: float
    r_faithfulness: float
    r_parsimony: float
    r_format: float
    parse_success: bool


class CompositeRewardFunction:
    """
    The full RSAT reward function.

    The faithfulness scorer is initialised once and reused across all calls
    (loading the NLI model or establishing the Groq connection is expensive).
    """

    def __init__(
        self,
        lambda_citation: float = 0.3,
        lambda_faithfulness: float = 0.5,
        lambda_parsimony: float = 0.2,
        format_penalty: float = -1.0,
        faithfulness_backend: str = "nli",
        answer_mode: str = "f1",
        parsimony_soft_threshold: int = 3,
        parsimony_hard_threshold: int = 8,
        device: str = "cuda",
        groq_client=None,
    ):
        self.lambda_citation = lambda_citation
        self.lambda_faithfulness = lambda_faithfulness
        self.lambda_parsimony = lambda_parsimony
        self.format_penalty = format_penalty
        self.answer_mode = answer_mode
        self.parsimony_soft = parsimony_soft_threshold
        self.parsimony_hard = parsimony_hard_threshold

        self.faithfulness_scorer = create_faithfulness_scorer(
            backend=faithfulness_backend,
            device=device,
            groq_client=groq_client,
        )

    def __call__(
        self,
        output: RSATOutput,
        example: RSATExample,
    ) -> RewardBreakdown:
        """Compute composite reward with full breakdown."""

        if not output.parse_success:
            return RewardBreakdown(
                total=self.format_penalty,
                r_answer=0.0,
                r_citation=0.0,
                r_faithfulness=0.0,
                r_parsimony=0.0,
                r_format=self.format_penalty,
                parse_success=False,
            )

        r_ans = compute_answer_reward(
            prediction=output.answer,
            gold=example.answer,
            gold_alternatives=example.metadata.get("all_answers"),
            mode=self.answer_mode,
        )
        r_cit = compute_citation_reward(output, example.table)
        r_faith = self.faithfulness_scorer.score_output(output, example.table)
        r_pars = compute_parsimony_reward(
            output,
            example.table,
            soft_threshold=self.parsimony_soft,
            hard_threshold=self.parsimony_hard,
        )

        total = (
            r_ans
            + self.lambda_citation * r_cit
            + self.lambda_faithfulness * r_faith
            + self.lambda_parsimony * r_pars
        )

        return RewardBreakdown(
            total=total,
            r_answer=r_ans,
            r_citation=r_cit,
            r_faithfulness=r_faith,
            r_parsimony=r_pars,
            r_format=0.0,
            parse_success=True,
        )

    def compute_batch(
        self,
        outputs: list[RSATOutput],
        examples: list[RSATExample],
    ) -> list[RewardBreakdown]:
        """Compute rewards for a batch (1:1 correspondence)."""
        assert len(outputs) == len(examples)
        return [self(o, e) for o, e in zip(outputs, examples)]


# ═════════════════════════════════════════════════════════════════════════════
# Config helper
# ═════════════════════════════════════════════════════════════════════════════


def create_reward_function(config: dict) -> CompositeRewardFunction:
    """Instantiate from a config dict (typically loaded from YAML)."""
    rc = config.get("reward", {})
    return CompositeRewardFunction(
        lambda_citation=rc.get("lambda_citation", 0.3),
        lambda_faithfulness=rc.get("lambda_faithfulness", 0.5),
        lambda_parsimony=rc.get("lambda_parsimony", 0.2),
        format_penalty=rc.get("format_penalty", -1.0),
        faithfulness_backend=rc.get("faithfulness_backend", "nli"),
        answer_mode=rc.get("answer_mode", "f1"),
        parsimony_soft_threshold=rc.get("parsimony_soft_threshold", 3),
        parsimony_hard_threshold=rc.get("parsimony_hard_threshold", 8),
        device=rc.get("device", "cuda"),
    )