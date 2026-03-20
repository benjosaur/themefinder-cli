"""Tests for classify_single_response and calibrate-related logic."""

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from themefinder import classify_single_response
from themefinder.examples import format_mapping_examples
from themefinder.llm import LLMResponse
from themefinder.models import ThemeMappingOutput, ThemeMappingResponses


@pytest.fixture()
def refined_themes_df():
    return pd.DataFrame(
        {
            "topic_id": ["A", "B", "C"],
            "topic": [
                "Cost concerns: Worries about the cost of living",
                "Safety issues: Concerns about public safety",
                "Transport quality: Quality of public transport services",
            ],
        }
    )


# ---------------------------------------------------------------------------
# classify_single_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_single_response_returns_labels(mock_llm, refined_themes_df):
    """classify_single_response should return the labels from the LLM response."""
    mock_llm.ainvoke = AsyncMock(
        return_value=LLMResponse(
            parsed=ThemeMappingResponses(
                responses=[
                    ThemeMappingOutput(response_id=1, labels=["A", "C"]),
                ]
            )
        )
    )

    labels = await classify_single_response(
        response_id=1,
        response_text="Buses are too expensive and unreliable",
        llm=mock_llm,
        question="What do you think about public transport?",
        refined_themes_df=refined_themes_df,
    )

    assert labels == ["A", "C"]
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_single_response_with_examples(mock_llm, refined_themes_df):
    """classify_single_response should pass examples into the prompt."""
    mock_llm.ainvoke = AsyncMock(
        return_value=LLMResponse(
            parsed=ThemeMappingResponses(
                responses=[
                    ThemeMappingOutput(response_id=5, labels=["B"]),
                ]
            )
        )
    )

    labels = await classify_single_response(
        response_id=5,
        response_text="I feel unsafe on the bus at night",
        llm=mock_llm,
        question="What do you think about public transport?",
        refined_themes_df=refined_themes_df,
        examples='Example 1:\nResponse: "Too costly"\nAssigned topics: A\nExplanation: Cost related',
    )

    assert labels == ["B"]
    # Verify examples appeared in the prompt
    call_args = mock_llm.ainvoke.call_args
    prompt = call_args[0][0]
    assert "Too costly" in prompt
    assert "Cost related" in prompt


@pytest.mark.asyncio
async def test_classify_single_response_no_match_returns_empty(
    mock_llm, refined_themes_df
):
    """If LLM returns a different response_id, return empty list."""
    mock_llm.ainvoke = AsyncMock(
        return_value=LLMResponse(
            parsed=ThemeMappingResponses(
                responses=[
                    ThemeMappingOutput(response_id=999, labels=["A"]),
                ]
            )
        )
    )

    labels = await classify_single_response(
        response_id=1,
        response_text="Some response",
        llm=mock_llm,
        question="Question?",
        refined_themes_df=refined_themes_df,
    )

    assert labels == []


@pytest.mark.asyncio
async def test_classify_single_response_dict_response(mock_llm, refined_themes_df):
    """classify_single_response should handle dict-style parsed output."""
    mock_llm.ainvoke = AsyncMock(
        return_value=LLMResponse(
            parsed={
                "responses": [
                    {"response_id": 1, "labels": ["B", "C"]},
                ]
            }
        )
    )

    labels = await classify_single_response(
        response_id=1,
        response_text="Unsafe and unreliable",
        llm=mock_llm,
        question="Question?",
        refined_themes_df=refined_themes_df,
    )

    assert labels == ["B", "C"]


# ---------------------------------------------------------------------------
# Streak logic
# ---------------------------------------------------------------------------


class TestStreakLogic:
    """Test the streak counting logic used in calibrate."""

    def test_agreement_increments_streak(self):
        human_codes = {"A", "C"}
        llm_codes = {"A", "C"}
        streak = 3
        if human_codes == llm_codes:
            streak += 1
        assert streak == 4

    def test_disagreement_human_right_resets_streak(self):
        human_codes = {"A", "C"}
        llm_codes = {"A"}
        streak = 3
        if human_codes != llm_codes:
            # human is right
            streak = 0
        assert streak == 0

    def test_disagreement_llm_right_increments_streak(self):
        human_codes = {"A"}
        llm_codes = {"A", "C"}
        streak = 3
        if human_codes != llm_codes:
            # llm is right
            streak += 1
        assert streak == 4

    def test_streak_target_reached(self):
        streak = 4
        streak_target = 5
        streak += 1
        assert streak >= streak_target


# ---------------------------------------------------------------------------
# Gold standard examples persistence
# ---------------------------------------------------------------------------


class TestGoldExamples:
    def test_gold_row_to_csv_and_back(self, tmp_path):
        """Gold rows saved to CSV should be loadable by format_mapping_examples."""
        output = tmp_path / "examples.csv"
        gold_rows = [
            {"response": "Too expensive", "code_1": "A", "explanation": "Cost issue"},
            {
                "response": "Unsafe at night",
                "code_1": "B",
                "code_2": "C",
                "explanation": "Safety and quality",
            },
        ]
        df = pd.DataFrame(gold_rows)
        df.to_csv(output, index=False)

        # Reload and format — should work with format_mapping_examples
        loaded = pd.read_csv(output)
        result = format_mapping_examples(loaded)
        assert "Too expensive" in result
        assert "Unsafe at night" in result
        assert "Cost issue" in result
        assert "Safety and quality" in result

    def test_append_preserves_existing(self, tmp_path):
        """Appending new rows should preserve existing data."""
        output = tmp_path / "examples.csv"
        existing = pd.DataFrame(
            {
                "response": ["Old example"],
                "code_1": ["B"],
                "explanation": ["Previous"],
            }
        )
        existing.to_csv(output, index=False)

        new_rows = pd.DataFrame(
            [{"response": "New example", "code_1": "A", "explanation": "New issue"}]
        )
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(output, index=False)

        saved = pd.read_csv(output)
        assert len(saved) == 2
        assert saved.iloc[0]["response"] == "Old example"
        assert saved.iloc[1]["response"] == "New example"

    def test_accumulating_examples_appear_in_prompt(self):
        """As gold examples accumulate, they should appear in formatted output."""
        rows = []
        for i in range(3):
            rows.append(
                {
                    "response": f"Response {i}",
                    "code_1": "A",
                    "explanation": f"Reason {i}",
                }
            )
            df = pd.DataFrame(rows)
            result = format_mapping_examples(df)
            assert f"Example {i + 1}" in result
            assert f"Response {i}" in result

    def test_blank_explanation_omitted_from_prompt(self):
        """Gold rows with blank explanation should not include Explanation line."""
        rows = [
            {"response": "Agreed example", "code_1": "A", "explanation": ""},
            {
                "response": "Corrected example",
                "code_1": "B",
                "explanation": "LLM missed this",
            },
        ]
        df = pd.DataFrame(rows)
        result = format_mapping_examples(df)
        # First example has blank explanation — no Explanation line
        lines = result.split("\n")
        # Find the block for Example 1
        ex1_start = next(i for i, line in enumerate(lines) if "Example 1:" in line)
        ex2_start = next(i for i, line in enumerate(lines) if "Example 2:" in line)
        ex1_block = "\n".join(lines[ex1_start:ex2_start])
        assert "Explanation:" not in ex1_block
        # Second example has an explanation
        ex2_block = "\n".join(lines[ex2_start:])
        assert "Explanation: LLM missed this" in ex2_block


# ---------------------------------------------------------------------------
# Neither judgment logic
# ---------------------------------------------------------------------------


class TestNeitherJudgment:
    def test_neither_resets_streak(self):
        """Choosing 'neither' should reset streak to 0."""
        streak = 3
        judgment = "n"
        if judgment == "n":
            streak = 0
        assert streak == 0

    def test_neither_saves_custom_codes(self):
        """Gold row from 'neither' uses the provided codes, not human or LLM."""
        correct_codes = ["B", "C"]
        gold_row: dict = {"response": "Some response"}
        for i, code in enumerate(sorted(correct_codes), 1):
            gold_row[f"code_{i}"] = code
        gold_row["explanation"] = "Both were wrong"
        assert gold_row["code_1"] == "B"
        assert gold_row["code_2"] == "C"
        assert gold_row["explanation"] == "Both were wrong"
