"""
faithfulness_reward.py — R_faithfulness: semantic faithfulness of citations.

For each reasoning step, checks whether the cited cells *actually support*
the claim made in that step.

Two backends:

1. **NLI-based (DeBERTa)** — reproducible, no API dependency.
   Uses a cross-encoder NLI model: premise = serialised cell contents,
   hypothesis = reasoning step text.  Entailment probability = score.

2. **LLM judge (Llama-3.3-70B via Groq)** — more accurate, especially for
   numerical reasoning.  Requires API access.

Step-level scores are averaged to produce the output-level R_faithfulness.
"""

from __future__ import annotations

from typing import Optional

from src.data.data_formats import RSATOutput, Table
from src.utils.table_utils import serialize_cells_for_nli


# ═════════════════════════════════════════════════════════════════════════════
# NLI-Based Faithfulness  (Primary / Reproducible)
# ═════════════════════════════════════════════════════════════════════════════


class NLIFaithfulnessScorer:
    """
    Uses a cross-encoder NLI model to score entailment between
    cited cell contents (premise) and reasoning step text (hypothesis).

    Model: `cross-encoder/nli-deberta-v3-base`
      → outputs logits for [contradiction, entailment, neutral]
    We take softmax and use the entailment probability as the score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "cuda",
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.model.eval()

        # Label mapping for cross-encoder/nli-deberta-v3-base:
        #   0 = contradiction, 1 = entailment, 2 = neutral
        self.entailment_idx = 1

    def score_step(self, reasoning_step: str, cited_cells_text: str) -> float:
        """
        Score a single reasoning step.

        Args:
            reasoning_step: The claim made in the step (hypothesis).
            cited_cells_text: Serialised cell contents (premise).

        Returns:
            Entailment probability in [0, 1].
        """
        import torch

        if not cited_cells_text:
            return 0.0

        inputs = self.tokenizer(
            cited_cells_text,
            reasoning_step,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: (1, 3)
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, self.entailment_idx].item()

        return float(score)

    def score_output(self, output: RSATOutput, table: Table) -> float:
        """Average faithfulness over all reasoning steps."""
        if not output.parse_success or not output.reasoning_steps:
            return 0.0

        scores: list[float] = []
        for step in output.reasoning_steps:
            cells_text = serialize_cells_for_nli(table, step.cited_cells)
            scores.append(self.score_step(step.step, cells_text))

        return sum(scores) / len(scores) if scores else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# LLM Judge Faithfulness  (Enhanced Variant — Groq)
# ═════════════════════════════════════════════════════════════════════════════


class LLMFaithfulnessScorer:
    """
    Uses Llama-3.3-70B via Groq to judge faithfulness.
    More accurate than NLI, especially for numerical reasoning and comparisons.
    """

    def __init__(self, groq_client=None):
        if groq_client is None:
            from src.utils.groq_client import GroqRotatingClient

            self.groq_client = GroqRotatingClient(model="llama-3.3-70b-versatile")
        else:
            self.groq_client = groq_client

    def score_step(self, reasoning_step: str, cited_cells_text: str) -> float:
        if not cited_cells_text:
            return 0.0
        result = self.groq_client.judge_faithfulness(
            reasoning_step=reasoning_step,
            cited_cells_text=cited_cells_text,
        )
        return result["score"]

    def score_output(self, output: RSATOutput, table: Table) -> float:
        if not output.parse_success or not output.reasoning_steps:
            return 0.0
        scores = [
            self.score_step(
                step.step,
                serialize_cells_for_nli(table, step.cited_cells),
            )
            for step in output.reasoning_steps
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════


def create_faithfulness_scorer(
    backend: str = "nli",
    device: str = "cuda",
    groq_client=None,
) -> NLIFaithfulnessScorer | LLMFaithfulnessScorer:
    """
    Create the appropriate faithfulness scorer.

    Args:
        backend: "nli" for DeBERTa cross-encoder, "llm" for Groq judge.
        device: Torch device for NLI model.
        groq_client: Optional pre-initialised GroqRotatingClient.
    """
    if backend == "nli":
        return NLIFaithfulnessScorer(device=device)
    elif backend == "llm":
        return LLMFaithfulnessScorer(groq_client=groq_client)
    else:
        raise ValueError(f"Unknown faithfulness backend: {backend}")


def compute_faithfulness_reward(
    output: RSATOutput,
    table: Table,
    scorer=None,
    backend: str = "nli",
) -> float:
    """Convenience wrapper — prefer reusing a scorer across calls."""
    if scorer is None:
        scorer = create_faithfulness_scorer(backend=backend)
    return scorer.score_output(output, table)