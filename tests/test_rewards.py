"""
test_rewards.py — Unit tests for all reward components + data formats.

Run locally (no GPU required):
    cd rsat
    python -m pytest tests/test_rewards.py -v
"""

from __future__ import annotations

import json

import pytest

from src.data.data_formats import (
    RSATExample,
    RSATOutput,
    ReasoningStep,
    Table,
    format_model_input,
    format_sft_target,
)
from src.rewards.answer_reward import (
    compute_answer_reward,
    exact_match,
    token_f1,
)
from src.rewards.citation_reward import (
    compute_citation_precision_recall,
    compute_citation_reward,
)
from src.rewards.parsimony_reward import compute_parsimony_reward
from src.utils.table_utils import (
    deduplicate_cells,
    extract_cell_contents,
    serialize_cells_for_nli,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def medals_table() -> Table:
    return Table(
        headers=["Country", "Gold", "Silver", "Bronze", "Total"],
        rows=[
            ["USA", "10", "8", "7", "25"],
            ["China", "8", "6", "5", "19"],
            ["UK", "5", "7", "3", "15"],
        ],
    )


@pytest.fixture
def medals_example(medals_table) -> RSATExample:
    return RSATExample(
        id="test_1",
        question="Which country had the most total medals?",
        table=medals_table,
        answer="USA",
        source_dataset="test",
        metadata={"all_answers": ["USA", "United States"]},
    )


@pytest.fixture
def good_output() -> RSATOutput:
    return RSATOutput(
        reasoning_steps=[
            ReasoningStep(
                step="Look at the Total column for each country.",
                cited_cells=[[0, 4], [1, 4], [2, 4]],
            ),
            ReasoningStep(
                step="USA has 25, which is the highest total.",
                cited_cells=[[0, 0], [0, 4]],
            ),
        ],
        answer="USA",
        parse_success=True,
    )


@pytest.fixture
def hallucinated_output() -> RSATOutput:
    """Cites cells that don't exist."""
    return RSATOutput(
        reasoning_steps=[
            ReasoningStep(
                step="Row 10 shows the answer.",
                cited_cells=[[10, 0], [10, 4]],
            ),
        ],
        answer="USA",
        parse_success=True,
    )


@pytest.fixture
def overciting_output() -> RSATOutput:
    """Cites way too many cells."""
    return RSATOutput(
        reasoning_steps=[
            ReasoningStep(
                step="Looking at everything.",
                cited_cells=[[i, j] for i in range(3) for j in range(5)],
            ),
        ],
        answer="USA",
        parse_success=True,
    )


@pytest.fixture
def empty_citation_output() -> RSATOutput:
    return RSATOutput(
        reasoning_steps=[
            ReasoningStep(step="The answer is USA.", cited_cells=[]),
        ],
        answer="USA",
        parse_success=True,
    )


@pytest.fixture
def unparsed_output() -> RSATOutput:
    return RSATOutput(
        reasoning_steps=[],
        answer="",
        raw_text="this is not json!!!",
        parse_success=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Table tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTable:
    def test_dimensions(self, medals_table: Table):
        assert medals_table.num_rows == 3
        assert medals_table.num_cols == 5

    def test_cell_exists_valid(self, medals_table: Table):
        assert medals_table.cell_exists(0, 0) is True
        assert medals_table.cell_exists(2, 4) is True

    def test_cell_exists_invalid(self, medals_table: Table):
        assert medals_table.cell_exists(3, 0) is False
        assert medals_table.cell_exists(0, 5) is False
        assert medals_table.cell_exists(-1, 0) is False

    def test_get_cell_valid(self, medals_table: Table):
        assert medals_table.get_cell(0, 0) == "USA"
        assert medals_table.get_cell(1, 4) == "19"

    def test_get_cell_invalid(self, medals_table: Table):
        assert medals_table.get_cell(10, 0) is None

    def test_serialize_indexed(self, medals_table: Table):
        text = medals_table.serialize_indexed()
        assert "[row_0, col_0] USA" in text
        assert "[col_0] Country" in text
        assert "Row 2:" in text

    def test_serialize_flat(self, medals_table: Table):
        text = medals_table.serialize_flat()
        assert "| USA |" in text or "| USA " in text
        assert "Country" in text

    def test_get_cells_text(self, medals_table: Table):
        text = medals_table.get_cells_text([[0, 0], [0, 4]])
        assert "USA" in text
        assert "25" in text

    def test_roundtrip(self, medals_table: Table):
        d = medals_table.to_dict()
        restored = Table.from_dict(d)
        assert restored.headers == medals_table.headers
        assert restored.rows == medals_table.rows


# ═════════════════════════════════════════════════════════════════════════════
# RSATExample serialization
# ═════════════════════════════════════════════════════════════════════════════


class TestRSATExample:
    def test_json_roundtrip(self, medals_example: RSATExample):
        j = medals_example.to_json()
        restored = RSATExample.from_json(j)
        assert restored.id == medals_example.id
        assert restored.question == medals_example.question
        assert restored.answer == medals_example.answer
        assert restored.table.num_rows == 3

    def test_with_trace(self, medals_example: RSATExample):
        medals_example.reasoning_trace = [
            ReasoningStep(step="test step", cited_cells=[[0, 0]]),
        ]
        j = medals_example.to_json()
        restored = RSATExample.from_json(j)
        assert restored.reasoning_trace is not None
        assert len(restored.reasoning_trace) == 1
        assert restored.reasoning_trace[0].cited_cells == [[0, 0]]


# ═════════════════════════════════════════════════════════════════════════════
# RSATOutput parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestRSATOutputParsing:
    def test_valid_json(self):
        text = json.dumps({
            "reasoning_steps": [
                {"step": "test", "cited_cells": [[0, 0]]},
            ],
            "answer": "yes",
        })
        out = RSATOutput.from_model_text(text)
        assert out.parse_success is True
        assert out.answer == "yes"
        assert len(out.reasoning_steps) == 1
        assert out.reasoning_steps[0].cited_cells == [[0, 0]]

    def test_json_in_code_block(self):
        text = '```json\n{"reasoning_steps": [], "answer": "no"}\n```'
        out = RSATOutput.from_model_text(text)
        assert out.parse_success is True
        assert out.answer == "no"

    def test_json_with_preamble(self):
        text = 'Here is the answer:\n{"reasoning_steps": [], "answer": "42"}'
        out = RSATOutput.from_model_text(text)
        assert out.parse_success is True
        assert out.answer == "42"

    def test_invalid_json(self):
        out = RSATOutput.from_model_text("I think the answer is USA")
        assert out.parse_success is False
        assert out.answer == ""

    def test_malformed_cells_are_skipped(self):
        text = json.dumps({
            "reasoning_steps": [
                {"step": "test", "cited_cells": ["bad", [0, 1], [2]]},
            ],
            "answer": "x",
        })
        out = RSATOutput.from_model_text(text)
        assert out.parse_success is True
        # Only [0, 1] should survive (len ≥ 2)
        assert out.reasoning_steps[0].cited_cells == [[0, 1]]

    def test_empty_output(self):
        out = RSATOutput.from_model_text("")
        assert out.parse_success is False


# ═════════════════════════════════════════════════════════════════════════════
# Answer reward
# ═════════════════════════════════════════════════════════════════════════════


class TestAnswerReward:
    def test_em_identical(self):
        assert exact_match("USA", "USA") == 1.0

    def test_em_case_insensitive(self):
        assert exact_match("usa", "USA") == 1.0

    def test_em_different(self):
        assert exact_match("China", "USA") == 0.0

    def test_em_strips_articles(self):
        assert exact_match("the USA", "USA") == 1.0

    def test_f1_identical(self):
        assert token_f1("USA", "USA") == 1.0

    def test_f1_partial_overlap(self):
        score = token_f1("United States of America", "United States")
        assert 0.0 < score < 1.0

    def test_f1_no_overlap(self):
        assert token_f1("China", "USA") == 0.0

    def test_f1_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_f1_one_empty(self):
        assert token_f1("USA", "") == 0.0
        assert token_f1("", "USA") == 0.0

    def test_alternatives_hit(self):
        score = compute_answer_reward(
            "US",
            "USA",
            gold_alternatives=["US", "United States"],
            mode="exact_match",
        )
        assert score == 1.0

    def test_alternatives_miss(self):
        score = compute_answer_reward(
            "France",
            "USA",
            gold_alternatives=["US"],
            mode="exact_match",
        )
        assert score == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Citation reward
# ═════════════════════════════════════════════════════════════════════════════


class TestCitationReward:
    def test_all_valid(self, good_output, medals_table):
        score = compute_citation_reward(good_output, medals_table)
        assert score == 1.0

    def test_all_hallucinated(self, hallucinated_output, medals_table):
        score = compute_citation_reward(hallucinated_output, medals_table)
        assert score == 0.0

    def test_no_citations(self, empty_citation_output, medals_table):
        score = compute_citation_reward(empty_citation_output, medals_table)
        assert score == 0.0

    def test_unparsed(self, unparsed_output, medals_table):
        score = compute_citation_reward(unparsed_output, medals_table)
        assert score == 0.0

    def test_mixed_valid_invalid(self, medals_table):
        out = RSATOutput(
            reasoning_steps=[
                ReasoningStep(
                    step="Mixed",
                    cited_cells=[[0, 0], [99, 99]],  # one valid, one invalid
                ),
            ],
            answer="USA",
            parse_success=True,
        )
        score = compute_citation_reward(out, medals_table)
        assert score == pytest.approx(0.5)

    def test_precision_recall(self, good_output, medals_table):
        gold_cells = [[0, 4], [1, 4], [2, 4], [0, 0]]
        pr = compute_citation_precision_recall(good_output, gold_cells, medals_table)
        assert 0 < pr["precision"] <= 1
        assert 0 < pr["recall"] <= 1
        assert 0 < pr["f1"] <= 1


# ═════════════════════════════════════════════════════════════════════════════
# Parsimony reward
# ═════════════════════════════════════════════════════════════════════════════


class TestParsimonyReward:
    def test_few_citations_high_score(self, good_output, medals_table):
        score = compute_parsimony_reward(good_output, medals_table)
        assert score > 0.5

    def test_overciting_low_score(self, overciting_output, medals_table):
        score = compute_parsimony_reward(overciting_output, medals_table)
        assert score < 0.5

    def test_unparsed(self, unparsed_output, medals_table):
        score = compute_parsimony_reward(unparsed_output, medals_table)
        assert score == 0.0

    def test_single_cell_max_score(self, medals_table):
        out = RSATOutput(
            reasoning_steps=[
                ReasoningStep(step="One cell.", cited_cells=[[0, 0]]),
            ],
            answer="USA",
            parse_success=True,
        )
        score = compute_parsimony_reward(out, medals_table)
        assert score == 1.0

    def test_threshold_boundary(self, medals_table):
        # Exactly at soft threshold (3) → 1.0
        out = RSATOutput(
            reasoning_steps=[
                ReasoningStep(
                    step="Three cells.",
                    cited_cells=[[0, 0], [0, 1], [0, 2]],
                ),
            ],
            answer="x",
            parse_success=True,
        )
        assert compute_parsimony_reward(out, medals_table, soft_threshold=3) == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Table utilities
# ═════════════════════════════════════════════════════════════════════════════


class TestTableUtils:
    def test_extract_cell_contents(self, medals_table):
        cells = extract_cell_contents(medals_table, [[0, 0], [1, 4]])
        assert len(cells) == 2
        assert cells[0]["value"] == "USA"
        assert cells[1]["value"] == "19"

    def test_extract_invalid_coords(self, medals_table):
        cells = extract_cell_contents(medals_table, [[99, 99]])
        assert cells == []

    def test_serialize_for_nli(self, medals_table):
        text = serialize_cells_for_nli(medals_table, [[0, 0], [0, 4]])
        assert "Country" in text
        assert "USA" in text
        assert "Total" in text
        assert "25" in text

    def test_serialize_empty(self, medals_table):
        assert serialize_cells_for_nli(medals_table, []) == ""

    def test_deduplicate(self):
        cells = [[0, 0], [1, 1], [0, 0], [1, 1], [2, 2]]
        deduped = deduplicate_cells(cells)
        assert len(deduped) == 3
        assert [0, 0] in deduped
        assert [1, 1] in deduped
        assert [2, 2] in deduped


# ═════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═════════════════════════════════════════════════════════════════════════════


class TestFormatting:
    def test_format_model_input(self, medals_example):
        prompt = format_model_input(medals_example)
        assert "Question:" in prompt
        assert "most total medals" in prompt
        assert "[row_0, col_0]" in prompt

    def test_format_sft_target_with_trace(self, medals_example):
        medals_example.reasoning_trace = [
            ReasoningStep(step="test", cited_cells=[[0, 0]]),
        ]
        target = format_sft_target(medals_example)
        parsed = json.loads(target)
        assert "reasoning_steps" in parsed
        assert parsed["answer"] == "USA"

    def test_format_sft_target_without_trace(self, medals_example):
        target = format_sft_target(medals_example)
        parsed = json.loads(target)
        assert parsed["reasoning_steps"] == []
        assert parsed["answer"] == "USA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])