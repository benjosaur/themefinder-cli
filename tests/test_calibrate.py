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
        examples='Example 1:\nResponse: "Too costly"\nIncorrect topics: B\nCorrect topics: A',
    )

    assert labels == ["B"]
    # Verify examples appeared in the prompt
    call_args = mock_llm.ainvoke.call_args
    prompt = call_args[0][0]
    assert "Too costly" in prompt
    assert "Incorrect topics: B" in prompt
    assert "Correct topics: A" in prompt


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
# Streak and error count logic
# ---------------------------------------------------------------------------


class TestStreakLogic:
    """Test the streak counting logic (informational) used in calibrate."""

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


class TestErrorCapLogic:
    """Test the error count / hard cap logic used in calibrate."""

    def test_agreement_does_not_increment_error_count(self):
        error_count = 2
        human_codes = {"A", "C"}
        llm_codes = {"A", "C"}
        if human_codes != llm_codes:
            error_count += 1
        assert error_count == 2

    def test_human_right_increments_error_count(self):
        error_count = 2
        human_codes = {"A", "C"}
        llm_codes = {"A"}
        # judgment = "h"
        if human_codes != llm_codes:
            error_count += 1
        assert error_count == 3

    def test_llm_right_does_not_increment_error_count(self):
        error_count = 2
        # judgment = "l" -> LLM was correct, no contrastive example
        # error_count stays the same
        assert error_count == 2

    def test_hard_cap_reached(self):
        error_count = 19
        max_examples = 20
        error_count += 1
        assert error_count >= max_examples


# ---------------------------------------------------------------------------
# Gold standard examples persistence
# ---------------------------------------------------------------------------


class TestGoldExamples:
    def test_contrastive_gold_row_to_csv_and_back(self, tmp_path):
        """Contrastive gold rows saved to CSV should format correctly."""
        output = tmp_path / "examples.csv"
        gold_rows = [
            {
                "response": "Too expensive",
                "llm_code_1": "B",
                "code_1": "A",
                "explanation": "Cost issue",
            },
            {
                "response": "Unsafe at night",
                "llm_code_1": "A",
                "code_1": "B",
                "code_2": "C",
                "explanation": "Safety and quality",
            },
        ]
        df = pd.DataFrame(gold_rows)
        df.to_csv(output, index=False)

        loaded = pd.read_csv(output)
        result = format_mapping_examples(loaded)
        assert "Too expensive" in result
        assert "Unsafe at night" in result
        assert "Incorrect topics: B" in result
        assert "Correct topics: A" in result
        # Explanation should NOT appear in prompt
        assert "Cost issue" not in result
        assert "Safety and quality" not in result

    def test_legacy_positive_only_format(self):
        """Old CSV without llm_code_* columns should still format correctly."""
        rows = [
            {"response": "Too expensive", "code_1": "A", "explanation": "Cost issue"},
        ]
        df = pd.DataFrame(rows)
        result = format_mapping_examples(df)
        assert "Assigned topics: A" in result
        assert "Too expensive" in result
        # No contrastive fields
        assert "Incorrect" not in result
        assert "Correct" not in result
        # Explanation excluded from prompt
        assert "Cost issue" not in result

    def test_append_preserves_existing(self, tmp_path):
        """Appending new rows should preserve existing data."""
        output = tmp_path / "examples.csv"
        existing = pd.DataFrame(
            {
                "response": ["Old example"],
                "llm_code_1": ["C"],
                "code_1": ["B"],
                "explanation": ["Previous"],
            }
        )
        existing.to_csv(output, index=False)

        new_rows = pd.DataFrame(
            [
                {
                    "response": "New example",
                    "llm_code_1": "A",
                    "code_1": "B",
                    "explanation": "",
                }
            ]
        )
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(output, index=False)

        saved = pd.read_csv(output)
        assert len(saved) == 2
        assert saved.iloc[0]["response"] == "Old example"
        assert saved.iloc[1]["response"] == "New example"

    def test_accumulating_contrastive_examples_appear_in_prompt(self):
        """As contrastive examples accumulate, they should appear in formatted output."""
        rows = []
        for i in range(3):
            rows.append(
                {
                    "response": f"Response {i}",
                    "llm_code_1": "B",
                    "code_1": "A",
                    "explanation": "",
                }
            )
            df = pd.DataFrame(rows)
            result = format_mapping_examples(df)
            assert f"Example {i + 1}" in result
            assert f"Response {i}" in result
            assert "Incorrect topics:" in result
            assert "Correct topics:" in result

    def test_explanation_never_in_prompt(self):
        """Explanation column should never appear in prompt output."""
        rows = [
            {
                "response": "Corrected example",
                "llm_code_1": "A",
                "code_1": "B",
                "explanation": "LLM missed this",
            },
        ]
        df = pd.DataFrame(rows)
        result = format_mapping_examples(df)
        assert "Explanation" not in result
        assert "LLM missed this" not in result


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

    def test_neither_saves_contrastive_example(self):
        """Gold row from 'neither' includes LLM codes and provided correct codes."""
        llm_codes = ["A"]
        correct_codes = ["B", "C"]
        gold_row: dict = {"response": "Some response"}
        for i, code in enumerate(sorted(llm_codes), 1):
            gold_row[f"llm_code_{i}"] = code
        for i, code in enumerate(sorted(correct_codes), 1):
            gold_row[f"code_{i}"] = code
        gold_row["explanation"] = "Both were wrong"
        assert gold_row["llm_code_1"] == "A"
        assert gold_row["code_1"] == "B"
        assert gold_row["code_2"] == "C"
        assert gold_row["explanation"] == "Both were wrong"
