"""
citation_reward.py — R_citation: structural validity of cited cells.

Checks whether cited cell coordinates actually exist in the table.
A model hallucinating [row_12, col_5] in a 5-row, 4-column table is penalised.
"""

from __future__ import annotations

from src.data.data_formats import RSATOutput, Table


def compute_citation_reward(output: RSATOutput, table: Table) -> float:
    """
    R_citation: fraction of cited cells that exist in the table.

    Returns:
        Score in [0, 1].
        - 1.0: every cited cell is valid.
        - 0.0: no cells cited, or all cited cells are hallucinated, or parse
          failure.
    """
    if not output.parse_success:
        return 0.0

    total = 0
    valid = 0
    for step in output.reasoning_steps:
        for cell in step.cited_cells:
            total += 1
            if table.cell_exists(cell[0], cell[1]):
                valid += 1

    if total == 0:
        return 0.0  # model cited nothing — penalise

    return valid / total


def compute_citation_precision_recall(
    output: RSATOutput,
    gold_cells: list[list[int]],
    table: Table,
) -> dict[str, float]:
    """
    Citation precision / recall / F1 against gold cell annotations.
    Used for **evaluation only**, not as a training reward.
    """
    if not output.parse_success:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set: set[tuple[int, int]] = set()
    for step in output.reasoning_steps:
        for cell in step.cited_cells:
            if table.cell_exists(cell[0], cell[1]):
                pred_set.add((cell[0], cell[1]))

    gold_set: set[tuple[int, int]] = set()
    for c in gold_cells:
        if len(c) >= 2:
            gold_set.add((int(c[0]), int(c[1])))

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = pred_set & gold_set
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}