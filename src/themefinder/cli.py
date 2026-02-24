"""Minimal CLI wrapper for themefinder — Bedrock only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import typer
from langchain_aws import ChatBedrockConverse
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from themefinder import (
    detail_detection,
    sentiment_analysis,
    theme_clustering,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
)
from themefinder.examples import format_discovery_examples, format_mapping_examples

app = typer.Typer(help="themefinder CLI — discover, classify, and evaluate themes.")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_llm(
    model: str,
    region: str,
    profile: str | None = None,
    temperature: float = 0.0,
):
    """Create a ChatBedrockConverse LLM wrapped with fallbacks (for type compat)."""
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("bedrock-runtime", region_name=region)
    llm = ChatBedrockConverse(client=client, model=model, temperature=temperature)
    return llm.with_fallbacks([])  # tasks.py type-hints RunnableWithFallbacks


def read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_responses(
    path: Path,
    column: str | None = None,
    id_col: str = "response_id",
    text_col: str = "response",
) -> pd.DataFrame:
    """Load a responses CSV/XLSX and normalise column names to response_id / response."""
    df = read_tabular(path)
    actual_text_col = column or text_col
    rename = {}
    if id_col != "response_id":
        rename[id_col] = "response_id"
    if actual_text_col != "response":
        rename[actual_text_col] = "response"
    if rename:
        df = df.rename(columns=rename)
    if "response_id" not in df.columns:
        raise typer.BadParameter(f"Column '{id_col}' not found in {path}")
    if "response" not in df.columns:
        raise typer.BadParameter(f"No response text column found in {path}")
    return df


def load_themes(path: Path) -> pd.DataFrame:
    """Load a themes CSV/XLSX. Accepts both refined and condensed formats."""
    df = read_tabular(path)
    if "topic" not in df.columns:
        if "topic_label" in df.columns and "topic_description" in df.columns:
            df["topic"] = df["topic_label"] + ": " + df["topic_description"]
        else:
            raise typer.BadParameter(
                "Themes CSV must have a 'topic' column or 'topic_label'+'topic_description' columns"
            )
    if "topic_id" not in df.columns:
        raise typer.BadParameter("Themes CSV must have a 'topic_id' column")
    return df


def to_wide_csv(mapping_df: pd.DataFrame, output: Path) -> None:
    """Convert mapping DataFrame (with 'labels' list column) to wide code_N CSV."""
    rows = []
    for _, row in mapping_df.iterrows():
        labels = row.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        entry: dict = {
            "response_id": row["response_id"],
            "response": row.get("response", ""),
        }
        for i, label in enumerate(labels, 1):
            entry[f"code_{i}"] = label
        rows.append(entry)
    wide = pd.DataFrame(rows)
    wide.to_csv(output, index=False)


def read_coded_csv(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX with code_* columns. Returns DataFrame with response_id + labels set."""
    df = read_tabular(path)
    code_cols = sorted([c for c in df.columns if c.startswith("code_")])
    if not code_cols:
        raise typer.BadParameter(f"No code_* columns found in {path}")

    def extract_labels(row):
        labels = set()
        for c in code_cols:
            val = row[c]
            if pd.notna(val) and str(val).strip():
                labels.add(str(val).strip())
        return frozenset(labels)

    df["labels"] = df.apply(extract_labels, axis=1)
    return df[["response_id", "labels"]]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def discover(
    input_csv: Path = typer.Argument(..., help="Path to responses CSV"),
    column: Optional[str] = typer.Option(
        None, "--column", "-c", help="Response text column name"
    ),
    question: str = typer.Option(
        ..., "--question", "-q", help="The survey/consultation question"
    ),
    output: Path = typer.Option(
        "themes.csv", "--output", "-o", help="Output themes CSV path"
    ),
    model: str = typer.Option(
        "anthropic.claude-3-7-sonnet-20250219-v1:0", "--model", help="Bedrock model ID"
    ),
    region: str = typer.Option("eu-west-2", "--region", help="AWS region"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Max concurrent LLM calls"
    ),
    id_col: str = typer.Option(
        "response_id", "--id-col", help="Column name for response IDs"
    ),
    text_col: str = typer.Option(
        "response", "--text-col", help="Column name for response text"
    ),
    cluster: bool = typer.Option(
        False,
        "--cluster/--no-cluster",
        help="Use hierarchical clustering after condensation",
    ),
    target_themes: int = typer.Option(
        10, "--target-themes", help="Target number of themes for clustering"
    ),
    significance_pct: float = typer.Option(
        10.0, "--significance-pct", help="Significance threshold for clustering (%)"
    ),
    examples: Optional[Path] = typer.Option(
        None,
        "--examples",
        "-e",
        help="Path to discovery examples CSV/XLSX (columns: responses, topics)",
    ),
):
    """Discover themes from survey responses (sentiment -> generation -> condensation -> refinement)."""
    df = load_responses(input_csv, column=column, id_col=id_col, text_col=text_col)
    llm = make_llm(model, region, profile)
    examples_str = (
        format_discovery_examples(read_tabular(examples)) if examples else ""
    )

    async def _run():
        console.print(
            f"[bold]Running sentiment analysis on {len(df)} responses...[/bold]"
        )
        sentiment_df, _ = await sentiment_analysis(
            df, llm, question=question, concurrency=concurrency
        )

        console.print("[bold]Generating themes...[/bold]")
        theme_df, _ = await theme_generation(
            sentiment_df, llm, question=question, concurrency=concurrency,
            examples=examples_str,
        )

        console.print("[bold]Condensing themes...[/bold]")
        condensed_df, _ = await theme_condensation(
            theme_df, llm, question=question, concurrency=concurrency
        )

        if cluster:
            console.print(f"[bold]Clustering to ~{target_themes} themes...[/bold]")
            condensed_df["topic_id"] = [str(i) for i in range(1, len(condensed_df) + 1)]
            clustered_df, _ = theme_clustering(
                condensed_df,
                llm,
                target_themes=target_themes,
                significance_percentage=significance_pct,
            )
            refine_input = clustered_df[
                ["topic_label", "topic_description", "source_topic_count"]
            ]
        else:
            refine_input = condensed_df

        console.print("[bold]Refining themes...[/bold]")
        refined_df, _ = await theme_refinement(
            refine_input, llm, question=question, concurrency=concurrency
        )

        return refined_df

    refined = asyncio.run(_run())
    refined.to_csv(output, index=False)
    console.print(f"[green]Wrote {len(refined)} themes to {output}[/green]")


@app.command()
def classify(
    input_csv: Path = typer.Argument(..., help="Path to responses CSV"),
    themes: Path = typer.Option(..., "--themes", "-t", help="Path to themes CSV"),
    column: Optional[str] = typer.Option(
        None, "--column", "-c", help="Response text column name"
    ),
    question: str = typer.Option(
        ..., "--question", "-q", help="The survey/consultation question"
    ),
    output: Path = typer.Option(
        "coded.csv", "--output", "-o", help="Output coded CSV path"
    ),
    detail: bool = typer.Option(
        False, "--detail/--no-detail", help="Run detail detection"
    ),
    model: str = typer.Option(
        "anthropic.claude-3-7-sonnet-20250219-v1:0", "--model", help="Bedrock model ID"
    ),
    region: str = typer.Option("eu-west-2", "--region", help="AWS region"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Max concurrent LLM calls"
    ),
    id_col: str = typer.Option(
        "response_id", "--id-col", help="Column name for response IDs"
    ),
    text_col: str = typer.Option(
        "response", "--text-col", help="Column name for response text"
    ),
    examples: Optional[Path] = typer.Option(
        None,
        "--examples",
        "-e",
        help="Path to mapping examples CSV/XLSX (columns: response, code_*, explanation)",
    ),
):
    """Classify responses against a theme set (theme mapping + optional detail detection)."""
    df = load_responses(input_csv, column=column, id_col=id_col, text_col=text_col)
    themes_df = load_themes(themes)
    llm = make_llm(model, region, profile)
    examples_str = (
        format_mapping_examples(read_tabular(examples)) if examples else ""
    )

    async def _run():
        console.print(
            f"[bold]Mapping {len(df)} responses to {len(themes_df)} themes...[/bold]"
        )
        mapping_df, _ = await theme_mapping(
            df[["response_id", "response"]],
            llm,
            question=question,
            refined_themes_df=themes_df,
            concurrency=concurrency,
            examples=examples_str,
        )

        if detail:
            console.print("[bold]Running detail detection...[/bold]")
            detail_df, _ = await detail_detection(
                df[["response_id", "response"]],
                llm,
                question=question,
                concurrency=concurrency,
            )
            mapping_df = mapping_df.merge(
                detail_df[["response_id", "evidence_rich"]],
                on="response_id",
                how="left",
            )

        return mapping_df

    result = asyncio.run(_run())
    to_wide_csv(result, output)
    console.print(f"[green]Wrote coded responses to {output}[/green]")


@app.command()
def evaluate(
    predicted_csv: Path = typer.Argument(..., help="Path to predicted (coded) CSV"),
    reference_csv: Path = typer.Argument(
        ..., help="Path to reference (ground truth) CSV"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional JSON output path"
    ),
):
    """Evaluate coded responses against a reference set (F1, precision, recall, exact match, overlap)."""
    pred = read_coded_csv(predicted_csv)
    ref = read_coded_csv(reference_csv)

    merged = pred.merge(ref, on="response_id", suffixes=("_pred", "_ref"))
    if merged.empty:
        console.print(
            "[red]No matching response_ids found between predicted and reference.[/red]"
        )
        raise typer.Exit(1)

    pred_labels = merged["labels_pred"].tolist()
    ref_labels = merged["labels_ref"].tolist()

    # Binarize for sklearn
    all_labels = set()
    for s in pred_labels + ref_labels:
        all_labels |= s
    all_labels = sorted(all_labels)

    mlb = MultiLabelBinarizer(classes=all_labels)
    y_pred = mlb.fit_transform(pred_labels)
    y_ref = mlb.transform(ref_labels)

    f1 = f1_score(y_ref, y_pred, average="samples", zero_division=0)
    precision = precision_score(y_ref, y_pred, average="samples", zero_division=0)
    recall = recall_score(y_ref, y_pred, average="samples", zero_division=0)

    exact_match = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r) / len(
        merged
    )
    overlap = sum(1 for p, r in zip(pred_labels, ref_labels) if p & r) / len(merged)

    metrics = {
        "n_responses": len(merged),
        "n_labels": len(all_labels),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "exact_match": round(exact_match, 4),
        "overlap_rate": round(overlap, 4),
    }

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)

    if output:
        output.write_text(json.dumps(metrics, indent=2))
        console.print(f"[green]Wrote metrics to {output}[/green]")


if __name__ == "__main__":
    app()
