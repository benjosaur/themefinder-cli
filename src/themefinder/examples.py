"""Helpers for formatting few-shot examples for LLM prompts."""

from __future__ import annotations

import pandas as pd


def format_mapping_examples(examples_df: pd.DataFrame | None) -> str:
    """Format a mapping-examples DataFrame into a prompt string.

    Expected columns: ``response``, one or more ``code_*`` columns, and
    optionally ``explanation``.

    Returns ``""`` when *examples_df* is ``None`` or empty.
    """
    if examples_df is None or examples_df.empty:
        return ""

    code_cols = sorted(c for c in examples_df.columns if c.startswith("code_"))
    if not code_cols:
        return ""

    lines: list[str] = []

    for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
        codes = [
            str(row[c]).strip()
            for c in code_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        response = str(row.get("response", ""))
        explanation = row.get("explanation", None)

        lines.append("")
        lines.append(f"Example {idx}:")
        lines.append(f'Response: "{response}"')
        lines.append(f"Assigned topics: {', '.join(codes)}")
        if pd.notna(explanation) and str(explanation).strip():
            lines.append(f"Explanation: {explanation}")

    return "\n".join(lines)
