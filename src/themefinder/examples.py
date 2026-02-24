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

    lines: list[str] = [
        "## EXAMPLES",
        "",
        "Below are examples of how responses should be classified against the TOPIC LIST.",
    ]

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


def format_discovery_examples(examples_df: pd.DataFrame | None) -> str:
    """Format a discovery-examples DataFrame into a prompt string.

    Expected columns: ``responses`` and/or ``topics``.  Each column is an
    independent list — values are collected independently (not paired
    row-by-row).

    Returns ``""`` when *examples_df* is ``None`` or empty.
    """
    if examples_df is None or examples_df.empty:
        return ""

    responses: list[str] = []
    topics: list[str] = []

    if "responses" in examples_df.columns:
        responses = [
            str(v).strip()
            for v in examples_df["responses"]
            if pd.notna(v) and str(v).strip()
        ]

    if "topics" in examples_df.columns:
        topics = [
            str(v).strip()
            for v in examples_df["topics"]
            if pd.notna(v) and str(v).strip()
        ]

    if not responses and not topics:
        return ""

    lines: list[str] = ["## EXAMPLES"]

    if responses:
        lines.append("")
        lines.append("Here are examples of the kinds of responses you might encounter:")
        for r in responses:
            lines.append(f'- "{r}"')

    if topics:
        lines.append("")
        lines.append(
            "Here are examples of the kinds of topics that should be extracted:"
        )
        for t in topics:
            lines.append(f"- {t}")

    return "\n".join(lines)
