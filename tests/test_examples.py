import pandas as pd

from themefinder.examples import format_mapping_examples


# ---------------------------------------------------------------------------
# format_mapping_examples
# ---------------------------------------------------------------------------


class TestFormatMappingExamples:
    def test_contrastive_format(self):
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
        assert "Example 1:" in result
        assert 'Response: "Too expensive and unsafe"' in result
        assert "Incorrect topics: A" in result
        assert "Correct topics: B, C" in result
        assert "Explanation" not in result

    def test_multiple_examples(self):
        df = pd.DataFrame(
            {
                "response": ["First response", "Second response"],
                "llm_code_1": ["A", "B"],
                "code_1": ["B", "A"],
                "explanation": ["", ""],
            }
        )
        result = format_mapping_examples(df)
        assert "Example 1:" in result
        assert "Example 2:" in result

    def test_empty_returns_empty_string(self):
        assert format_mapping_examples(None) == ""
        assert format_mapping_examples(pd.DataFrame()) == ""

    def test_no_code_columns_returns_empty(self):
        df = pd.DataFrame({"response": ["hello"], "explanation": ["world"]})
        assert format_mapping_examples(df) == ""

    def test_no_llm_code_columns_returns_empty(self):
        df = pd.DataFrame({"response": ["hello"], "code_1": ["A"]})
        assert format_mapping_examples(df) == ""
