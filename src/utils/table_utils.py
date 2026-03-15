"""
table_utils.py — Helpers for table manipulation, cell extraction, and serialization.
"""

from __future__ import annotations

import re
from typing import Optional

from src.data.data_formats import Table


def normalize_cell_value(value: str) -> str:
    """Normalize a cell value for comparison."""
    v = value.strip().lower()
    v = re.sub(r",(\d{3})", r"\1", v)  # remove thousand separators: 1,000 -> 1000
    v = re.sub(r"\s+", " ", v)
    return v


def extract_cell_contents(
    table: Table,
    cited_cells: list[list[int]],
) -> list[dict]:
    """
    Extract cell contents from a table given [[row, col], ...] coordinates.
    Returns a list of dicts: {row, col, header, value}.
    """
    results: list[dict] = []
    for coord in cited_cells:
        if len(coord) < 2:
            continue
        row_idx, col_idx = int(coord[0]), int(coord[1])
        value = table.get_cell(row_idx, col_idx)
        if value is not None:
            header = (
                table.headers[col_idx]
                if col_idx < len(table.headers)
                else f"col_{col_idx}"
            )
            results.append(
                {"row": row_idx, "col": col_idx, "header": header, "value": value}
            )
    return results


def serialize_cells_for_nli(
    table: Table,
    cited_cells: list[list[int]],
) -> str:
    """
    Serialize cited cells into a natural language premise for NLI-based
    faithfulness checking.  Fed to DeBERTa or the LLM judge.
    """
    cells = extract_cell_contents(table, cited_cells)
    if not cells:
        return ""
    parts = [
        f"The value of '{c['header']}' in row {c['row']} is '{c['value']}'"
        for c in cells
    ]
    return ". ".join(parts) + "."


def count_valid_cells(
    table: Table, cited_cells: list[list[int]]
) -> tuple[int, int]:
    """Returns (valid_count, total_count)."""
    valid = 0
    for c in cited_cells:
        if len(c) >= 2 and table.cell_exists(int(c[0]), int(c[1])):
            valid += 1
    return valid, len(cited_cells)


def deduplicate_cells(cited_cells: list[list[int]]) -> list[list[int]]:
    """Remove duplicate cell coordinates, preserving order."""
    seen: set[tuple[int, int]] = set()
    unique: list[list[int]] = []
    for c in cited_cells:
        if len(c) < 2:
            continue
        key = (int(c[0]), int(c[1]))
        if key not in seen:
            seen.add(key)
            unique.append([key[0], key[1]])
    return unique