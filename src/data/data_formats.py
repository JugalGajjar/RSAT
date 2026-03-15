"""
data_formats.py — Canonical data formats for RSAT.

Every dataset (WTQ, FeTaQA, TabFact) gets converted into RSATExample.
The model output is parsed into RSATOutput.
These two structures are what the reward functions consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Table:
    """Structured table representation."""

    headers: list[str]
    rows: list[list[str]]
    caption: Optional[str] = None

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.headers)

    def get_cell(self, row: int, col: int) -> Optional[str]:
        """Safely retrieve a cell value. Returns None if out of bounds."""
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            return self.rows[row][col]
        return None

    def cell_exists(self, row: int, col: int) -> bool:
        """Check whether a (row, col) coordinate is within the table."""
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols

    def serialize_flat(self) -> str:
        """Serialize table to a Markdown-style flat text format."""
        lines: list[str] = []
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join(["---"] * self.num_cols) + " |")
        for row in self.rows:
            # Pad row to match header count if needed
            padded = row + [""] * (self.num_cols - len(row))
            lines.append("| " + " | ".join(padded[: self.num_cols]) + " |")
        return "\n".join(lines)

    def serialize_indexed(self) -> str:
        """Serialize with explicit [row_i, col_j] indices for citation clarity."""
        lines: list[str] = []
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        header_parts = [f"[col_{j}] {h}" for j, h in enumerate(self.headers)]
        lines.append("Headers: " + " | ".join(header_parts))
        for i, row in enumerate(self.rows):
            cell_parts = []
            for j in range(self.num_cols):
                val = row[j] if j < len(row) else ""
                cell_parts.append(f"[row_{i}, col_{j}] {val}")
            lines.append(f"Row {i}: " + " | ".join(cell_parts))
        return "\n".join(lines)

    def get_cells_text(self, cell_coords: list[list[int]]) -> str:
        """Given [[row, col], ...], return a readable string of cell contents."""
        parts: list[str] = []
        for coord in cell_coords:
            if len(coord) < 2:
                continue
            row, col = int(coord[0]), int(coord[1])
            val = self.get_cell(row, col)
            if val is not None:
                header = (
                    self.headers[col] if col < len(self.headers) else f"col_{col}"
                )
                parts.append(f"{header}: {val}")
        return "; ".join(parts) if parts else ""

    def to_dict(self) -> dict:
        d: dict = {"headers": self.headers, "rows": self.rows}
        if self.caption is not None:
            d["caption"] = self.caption
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Table:
        return cls(
            headers=d["headers"],
            rows=d["rows"],
            caption=d.get("caption"),
        )


@dataclass
class ReasoningStep:
    """A single step in the model's reasoning trace."""

    step: str  # natural language reasoning
    cited_cells: list[list[int]]  # list of [row_i, col_j] coordinates

    def to_dict(self) -> dict:
        return {"step": self.step, "cited_cells": self.cited_cells}

    @classmethod
    def from_dict(cls, d: dict) -> ReasoningStep | None:
        if not isinstance(d, dict):
            return None  # skip malformed entries (e.g. lists)
        cells: list[list[int]] = []
        for c in d.get("cited_cells", []):
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                try:
                    cells.append([int(c[0]), int(c[1])])
                except (ValueError, TypeError):
                    continue
        return cls(step=d.get("step", ""), cited_cells=cells)


@dataclass
class RSATExample:
    """
    Unified training / evaluation example.
    Every dataset gets converted into this format before use.
    """

    id: str
    question: str
    table: Table
    answer: str
    source_dataset: str  # "wtq", "fetaqa", "tabfact"
    reasoning_trace: Optional[list[ReasoningStep]] = None  # gold trace (SFT only)
    metadata: dict = field(default_factory=dict)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "question": self.question,
            "table": self.table.to_dict(),
            "answer": self.answer,
            "source_dataset": self.source_dataset,
            "metadata": self.metadata,
        }
        if self.reasoning_trace is not None:
            d["reasoning_trace"] = [s.to_dict() for s in self.reasoning_trace]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RSATExample:
        trace = None
        if d.get("reasoning_trace"):
            trace = [step for step in (ReasoningStep.from_dict(raw) for raw in d["reasoning_trace"]) if step is not None]
        return cls(
            id=d["id"],
            question=d["question"],
            table=Table.from_dict(d["table"]),
            answer=d["answer"],
            source_dataset=d["source_dataset"],
            reasoning_trace=trace,
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> RSATExample:
        return cls.from_dict(json.loads(s))


@dataclass
class RSATOutput:
    """
    Parsed model output: reasoning steps + final answer.
    This is what the reward functions evaluate.
    """

    reasoning_steps: list[ReasoningStep]
    answer: str
    raw_text: str = ""
    parse_success: bool = True

    def to_dict(self) -> dict:
        return {
            "reasoning_steps": [s.to_dict() for s in self.reasoning_steps],
            "answer": self.answer,
        }

    @classmethod
    def from_model_text(cls, text: str) -> RSATOutput:
        """
        Parse model output text into structured RSATOutput.
        Handles common failure modes: markdown fences, extra text, etc.
        """
        try:
            cleaned = text.strip()

            # Strip markdown code fences
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1]
                if "```" in cleaned:
                    cleaned = cleaned.split("```", 1)[0]
                cleaned = cleaned.strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1]
                if "```" in cleaned:
                    cleaned = cleaned.split("```", 1)[0]
                cleaned = cleaned.strip()

            # Try to find JSON object boundaries if there's extra text
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

            parsed = json.loads(cleaned)

            steps: list[ReasoningStep] = []
            for s in parsed.get("reasoning_steps", []):
                step = ReasoningStep.from_dict(s)
                if step is not None:
                    steps.append(step)

            answer = str(parsed.get("answer", ""))

            return cls(
                reasoning_steps=steps,
                answer=answer,
                raw_text=text,
                parse_success=True,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
            return cls(
                reasoning_steps=[],
                answer="",
                raw_text=text,
                parse_success=False,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Prompt formatting helpers
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a precise table question answering assistant. "
    "Always respond with valid JSON containing reasoning steps with cell citations."
)


def format_model_input(example: RSATExample, use_indexed: bool = True) -> str:
    """
    Format an RSATExample into the user-turn prompt the model receives.
    This is the canonical input format used in both SFT and RL.
    """
    table_str = (
        example.table.serialize_indexed()
        if use_indexed
        else example.table.serialize_flat()
    )
    return (
        "Given the following table and question, provide your reasoning and answer.\n"
        "\n"
        "For each reasoning step, cite the specific table cells [row_index, col_index] that support it.\n"
        "\n"
        f"Table:\n{table_str}\n"
        "\n"
        f"Question: {example.question}\n"
        "\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        '  "reasoning_steps": [\n'
        '    {"step": "your reasoning here", "cited_cells": [[row_i, col_j], ...]},\n'
        "    ...\n"
        "  ],\n"
        '  "answer": "your final answer"\n'
        "}"
    )


def format_sft_target(example: RSATExample) -> str:
    """Format the gold reasoning trace + answer as the target for SFT training."""
    if not example.reasoning_trace:
        return json.dumps({"reasoning_steps": [], "answer": example.answer})
    output = {
        "reasoning_steps": [s.to_dict() for s in example.reasoning_trace],
        "answer": example.answer,
    }
    return json.dumps(output, ensure_ascii=False)


def build_chat_messages(
    user_content: str,
    assistant_content: Optional[str] = None,
) -> list[dict]:
    """Build a chat messages list with system + user (+ optional assistant)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages