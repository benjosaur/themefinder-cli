"""Helpers for formatting few-shot examples for LLM prompts."""

from __future__ import annotations

import pandas as pd


def format_mapping_examples(examples_df: pd.DataFrame | None) -> str:
    """Format a contrastive-examples DataFrame into a prompt string.

    Expected columns: ``response``, ``llm_code_*`` (what the LLM predicted
    incorrectly), and ``code_*`` (the correct labels).

    The ``explanation`` column is intentionally excluded from the prompt — it
    exists in the CSV for human reference only.

    Returns ``""`` when *examples_df* is ``None`` or empty.
    """
    if examples_df is None or examples_df.empty:
        return ""

    code_cols = sorted(c for c in examples_df.columns if c.startswith("code_"))
    llm_code_cols = sorted(
        c for c in examples_df.columns if c.startswith("llm_code_")
    )
    if not code_cols or not llm_code_cols:
        return ""

    lines: list[str] = []

    for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
        codes = [
            str(row[c]).strip()
            for c in code_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        llm_codes = [
            str(row[c]).strip()
            for c in llm_code_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        response = str(row.get("response", ""))

        lines.append("")
        lines.append(f"Example {idx}:")
        lines.append(f'Response: "{response}"')
        lines.append(f"Incorrect topics: {', '.join(llm_codes)}")
        lines.append(f"Correct topics: {', '.join(codes)}")

    return "\n".join(lines)
