import pandas as pd
import pytest

from themefinder.examples import format_discovery_examples, format_mapping_examples


# ---------------------------------------------------------------------------
# format_mapping_examples
# ---------------------------------------------------------------------------


class TestFormatMappingExamples:
    def test_basic(self):
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
        assert result.startswith("## EXAMPLES")
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert 'Response: "I think this will hurt small businesses"' in result
        assert "Assigned topics: A, C" in result
        assert "Assigned topics: B" in result
        assert "Explanation: Discusses economic impact" in result
        assert "Explanation: Expresses general support" in result

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


# ---------------------------------------------------------------------------
# format_discovery_examples
# ---------------------------------------------------------------------------


class TestFormatDiscoveryExamples:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "responses": [
                    "I think this is a terrible idea",
                    "The policy will help everyone",
                ],
                "topics": [
                    "Job losses: The policy change would lead to unemployment",
                    "Economic growth: The policy boosts the economy",
                ],
            }
        )
        result = format_discovery_examples(df)
        assert result.startswith("## EXAMPLES")
        assert "kinds of responses you might encounter" in result
        assert '- "I think this is a terrible idea"' in result
        assert "kinds of topics that should be extracted" in result
        assert "- Job losses: The policy change would lead to unemployment" in result

    def test_empty_returns_empty_string(self):
        assert format_discovery_examples(None) == ""
        assert format_discovery_examples(pd.DataFrame()) == ""

    def test_partial_columns(self):
        """One column has more rows than the other (NaN padding)."""
        df = pd.DataFrame(
            {
                "responses": ["Response A", "Response B", "Response C"],
                "topics": ["Topic 1", None, None],
            }
        )
        result = format_discovery_examples(df)
        assert '- "Response A"' in result
        assert '- "Response B"' in result
        assert '- "Response C"' in result
        assert "- Topic 1" in result
        # NaN topics should not appear
        assert result.count("- Topic") == 1

    def test_only_responses(self):
        df = pd.DataFrame({"responses": ["one", "two"]})
        result = format_discovery_examples(df)
        assert "kinds of responses" in result
        assert "kinds of topics" not in result

    def test_only_topics(self):
        df = pd.DataFrame({"topics": ["Topic A", "Topic B"]})
        result = format_discovery_examples(df)
        assert "kinds of topics" in result
        assert "kinds of responses" not in result
