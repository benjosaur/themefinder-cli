import pandas as pd

from themefinder.examples import format_mapping_examples


# ---------------------------------------------------------------------------
# format_mapping_examples
# ---------------------------------------------------------------------------


class TestFormatMappingExamples:
    def test_basic_legacy_positive_only(self):
        """Legacy format (no llm_code_*) uses 'Assigned topics' and omits explanation."""
        df = pd.DataFrame(
            {
                "response": [
                    "I think this will hurt small businesses",
                    "Great idea, fully support it",
                ],
                "code_1": ["A", "B"],
                "code_2": ["C", None],
                "explanation": [
                    "Discusses economic impact (A) and business concerns (C).",
                    "Expresses general support (B).",
                ],
            }
        )
        result = format_mapping_examples(df)
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert 'Response: "I think this will hurt small businesses"' in result
        assert "Assigned topics: A, C" in result
        assert "Assigned topics: B" in result
        # Explanations are excluded from prompt (human reference only)
        assert "Explanation" not in result

    def test_contrastive_format(self):
        """When llm_code_* columns are present, use contrastive format."""
        df = pd.DataFrame(
            {
                "response": ["Too expensive and unsafe"],
                "llm_code_1": ["A"],
                "code_1": ["B"],
                "code_2": ["C"],
                "explanation": ["LLM missed safety"],
            }
        )
        result = format_mapping_examples(df)
        assert "Incorrect topics: A" in result
        assert "Correct topics: B, C" in result
        assert "Explanation" not in result

    def test_empty_returns_empty_string(self):
        assert format_mapping_examples(None) == ""
        assert format_mapping_examples(pd.DataFrame()) == ""

    def test_no_code_columns_returns_empty(self):
        df = pd.DataFrame({"response": ["hello"], "explanation": ["world"]})
        assert format_mapping_examples(df) == ""

    def test_missing_explanation(self):
        df = pd.DataFrame(
            {
                "response": ["Some response"],
                "code_1": ["A"],
                "explanation": [None],
            }
        )
        result = format_mapping_examples(df)
        assert "Example 1:" in result
        assert "Assigned topics: A" in result
        assert "Explanation:" not in result
