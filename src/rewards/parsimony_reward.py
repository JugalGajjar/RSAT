"""
parsimony_reward.py — R_parsimony: penalises over-citation.

Encourages the model to cite the minimum cells needed to support
its reasoning, rather than citing half the table per step.

Scoring (per step):
  cells ≤ soft_threshold  →  1.0  (no penalty)
  soft_threshold < cells < hard_threshold  →  linear decay
  cells ≥ hard_threshold  →  0.0  (full penalty)
"""

from __future__ import annotations

from src.data.data_formats import RSATOutput, Table
from src.utils.table_utils import deduplicate_cells


def compute_parsimony_reward(
    output: RSATOutput,
    table: Table,
    soft_threshold: int = 3,
    hard_threshold: int = 8,
    per_step: bool = True,
) -> float:
    """
    Compute R_parsimony.

    Args:
        output: Parsed model output.
        table: The table being reasoned over.
        soft_threshold: ≤ this many valid cells per unit → no penalty.
        hard_threshold: ≥ this many valid cells per unit → full penalty.
        per_step: If True, score each step independently and average.
                  If False, score total unique valid cells across all steps.

    Returns:
        Score in [0, 1].  Higher = more parsimonious.
    """
    if not output.parse_success or not output.reasoning_steps:
        return 0.0

    if per_step:
        step_scores: list[float] = []
        for step in output.reasoning_steps:
            unique = deduplicate_cells(step.cited_cells)
            valid_count = sum(
                1 for c in unique if table.cell_exists(c[0], c[1])
            )
            step_scores.append(
                _linear_decay(valid_count, soft_threshold, hard_threshold)
            )
        return sum(step_scores) / len(step_scores) if step_scores else 0.0
    else:
        all_cells: list[list[int]] = []
        for step in output.reasoning_steps:
            all_cells.extend(step.cited_cells)
        unique = deduplicate_cells(all_cells)
        valid_count = sum(
            1 for c in unique if table.cell_exists(c[0], c[1])
        )
        return _linear_decay(valid_count, soft_threshold, hard_threshold)


def _linear_decay(count: int, soft: int, hard: int) -> float:
    """Linear decay between soft and hard thresholds."""
    if count <= soft:
        return 1.0
    if count >= hard:
        return 0.0
    return 1.0 - (count - soft) / (hard - soft)