"""
metrics.py — All evaluation metrics for RSAT.

Aggregates:
  - Answer Exact Match / F1
  - Citation Validity (fraction of cited cells that exist)
  - Faithfulness (NLI or LLM)
  - Parsimony
  - Format Success Rate
  - Structural stats (avg steps, avg cells per step)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.data.data_formats import RSATExample, RSATOutput
from src.rewards.answer_reward import exact_match, token_f1
from src.rewards.citation_reward import compute_citation_reward
from src.rewards.faithfulness_reward import create_faithfulness_scorer
from src.rewards.parsimony_reward import compute_parsimony_reward


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across a dataset split."""

    answer_em: float = 0.0
    answer_f1: float = 0.0
    citation_validity: float = 0.0
    faithfulness: float = 0.0
    parsimony: float = 0.0
    format_success_rate: float = 0.0
    avg_steps_per_output: float = 0.0
    avg_cells_per_step: float = 0.0
    num_examples: int = 0

    def to_dict(self) -> dict:
        return {
            "answer_em": round(self.answer_em, 4),
            "answer_f1": round(self.answer_f1, 4),
            "citation_validity": round(self.citation_validity, 4),
            "faithfulness": round(self.faithfulness, 4),
            "parsimony": round(self.parsimony, 4),
            "format_success_rate": round(self.format_success_rate, 4),
            "avg_steps_per_output": round(self.avg_steps_per_output, 2),
            "avg_cells_per_step": round(self.avg_cells_per_step, 2),
            "num_examples": self.num_examples,
        }

    def __repr__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.to_dict().items()]
        return "EvalMetrics(\n" + "\n".join(lines) + "\n)"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


class RSATEvaluator:
    """Evaluate model outputs against gold examples."""

    def __init__(
        self,
        faithfulness_backend: str = "nli",
        device: str = "cuda",
        groq_client=None,
    ):
        self.faithfulness_scorer = create_faithfulness_scorer(
            backend=faithfulness_backend,
            device=device,
            groq_client=groq_client,
        )

    def evaluate(
        self,
        outputs: list[RSATOutput],
        examples: list[RSATExample],
    ) -> EvalMetrics:
        """Evaluate outputs 1:1 against gold examples."""
        assert len(outputs) == len(examples)

        em_s: list[float] = []
        f1_s: list[float] = []
        cit_s: list[float] = []
        faith_s: list[float] = []
        pars_s: list[float] = []
        fmt_s: list[float] = []
        n_steps: list[float] = []
        cells_per_step: list[float] = []

        for out, ex in zip(outputs, examples):
            fmt_s.append(1.0 if out.parse_success else 0.0)

            if not out.parse_success:
                em_s.append(0.0)
                f1_s.append(0.0)
                cit_s.append(0.0)
                faith_s.append(0.0)
                pars_s.append(0.0)
                n_steps.append(0.0)
                cells_per_step.append(0.0)
                continue

            # Answer
            golds = [ex.answer]
            if "all_answers" in ex.metadata:
                golds.extend(ex.metadata["all_answers"])
            em_s.append(max(exact_match(out.answer, g) for g in golds))
            f1_s.append(max(token_f1(out.answer, g) for g in golds))

            # Citations
            cit_s.append(compute_citation_reward(out, ex.table))
            faith_s.append(self.faithfulness_scorer.score_output(out, ex.table))
            pars_s.append(compute_parsimony_reward(out, ex.table))

            # Structural
            ns = len(out.reasoning_steps)
            n_steps.append(float(ns))
            if ns > 0:
                avg_c = sum(len(s.cited_cells) for s in out.reasoning_steps) / ns
                cells_per_step.append(avg_c)
            else:
                cells_per_step.append(0.0)

        return EvalMetrics(
            answer_em=_mean(em_s),
            answer_f1=_mean(f1_s),
            citation_validity=_mean(cit_s),
            faithfulness=_mean(faith_s),
            parsimony=_mean(pars_s),
            format_success_rate=_mean(fmt_s),
            avg_steps_per_output=_mean(n_steps),
            avg_cells_per_step=_mean(cells_per_step),
            num_examples=len(outputs),
        )

    def evaluate_by_dataset(
        self,
        outputs: list[RSATOutput],
        examples: list[RSATExample],
    ) -> dict[str, EvalMetrics]:
        """Group results by ``source_dataset``."""
        grouped_o: dict[str, list[RSATOutput]] = defaultdict(list)
        grouped_e: dict[str, list[RSATExample]] = defaultdict(list)
        for o, e in zip(outputs, examples):
            grouped_o[e.source_dataset].append(o)
            grouped_e[e.source_dataset].append(e)
        return {
            ds: self.evaluate(grouped_o[ds], grouped_e[ds]) for ds in grouped_o
        }