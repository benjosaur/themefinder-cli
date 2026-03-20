"""Minimal CLI wrapper for themefinder — Bedrock only."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import typer
from langchain_aws import ChatBedrockConverse
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from themefinder import (
    classify_single_response,
    detail_detection,
    sentiment_analysis,
    theme_clustering,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
)
from themefinder.examples import format_mapping_examples

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


def _compute_theme_frequencies(
    pred_labels: list[frozenset[str]],
    ref_labels: list[frozenset[str]],
    all_labels: list[str],
    n_responses: int,
    label_map: dict[str, str] | None = None,
) -> tuple[list[dict], float, float]:
    """Per-theme frequency comparison and Kendall's tau rank correlation."""
    pred_counter = Counter(label for s in pred_labels for label in s)
    ref_counter = Counter(label for s in ref_labels for label in s)

    theme_freq = []
    for label in all_labels:
        pc = pred_counter.get(label, 0)
        rc = ref_counter.get(label, 0)
        theme_freq.append(
            {
                "label": label,
                "theme_label": label_map.get(label, label) if label_map else label,
                "pred_count": pc,
                "pred_pct": round(pc / n_responses * 100, 1),
                "ref_count": rc,
                "ref_pct": round(rc / n_responses * 100, 1),
            }
        )

    pred_freq_vec = [pred_counter.get(label, 0) for label in all_labels]
    ref_freq_vec = [ref_counter.get(label, 0) for label in all_labels]
    if len(all_labels) >= 2:
        tau, tau_p = kendalltau(pred_freq_vec, ref_freq_vec)
    else:
        tau, tau_p = float("nan"), float("nan")

    return theme_freq, float(tau), float(tau_p)


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
):
    """Discover themes from survey responses (sentiment -> generation -> condensation -> refinement)."""
    df = load_responses(input_csv, column=column, id_col=id_col, text_col=text_col)
    llm = make_llm(model, region, profile)

    async def _run():
        console.print(
            f"[bold]Running sentiment analysis on {len(df)} responses...[/bold]"
        )
        sentiment_df, _ = await sentiment_analysis(
            df, llm, question=question, concurrency=concurrency
        )

        console.print("[bold]Generating themes...[/bold]")
        theme_df, _ = await theme_generation(
            sentiment_df,
            llm,
            question=question,
            concurrency=concurrency,
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
    examples_str = format_mapping_examples(read_tabular(examples)) if examples else ""

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
    reference_csv: Path = typer.Argument(
        ..., help="Path to reference (ground truth) CSV"
    ),
    predicted_csv: Path = typer.Argument(..., help="Path to predicted (coded) CSV"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional JSON output path"
    ),
    themes: Optional[Path] = typer.Option(
        None, "--themes", "-t", help="Path to themes CSV for label lookup"
    ),
):
    """Evaluate coded responses against a reference set (F1, precision, recall, exact match, overlap)."""
    pred = read_coded_csv(predicted_csv)
    ref = read_coded_csv(reference_csv)

    label_map: dict[str, str] | None = None
    if themes:
        themes_df = pd.read_csv(themes)
        label_map = dict(
            zip(themes_df["topic_id"].astype(str), themes_df["topic_label"])
        )

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
    prec = precision_score(y_ref, y_pred, average="samples", zero_division=0)
    rec = recall_score(y_ref, y_pred, average="samples", zero_division=0)

    # Bootstrap confidence intervals (95%, 1000 resamples)
    n_bootstrap = 1000
    n_instances = y_ref.shape[0]
    rng = np.random.default_rng(seed=42)
    boot_f1, boot_prec, boot_rec = [], [], []
    boot_exact, boot_overlap, boot_tau = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_instances, size=n_instances, replace=True)
        boot_f1.append(
            f1_score(y_ref[idx], y_pred[idx], average="samples", zero_division=0)
        )
        boot_prec.append(
            precision_score(y_ref[idx], y_pred[idx], average="samples", zero_division=0)
        )
        boot_rec.append(
            recall_score(y_ref[idx], y_pred[idx], average="samples", zero_division=0)
        )
        sampled_pred = [pred_labels[i] for i in idx]
        sampled_ref = [ref_labels[i] for i in idx]
        boot_exact.append(
            sum(1 for p, r in zip(sampled_pred, sampled_ref) if p == r) / len(idx)
        )
        boot_overlap.append(
            sum(1 for p, r in zip(sampled_pred, sampled_ref) if p & r) / len(idx)
        )
        if len(all_labels) >= 2:
            pred_ctr = Counter(lbl for s in sampled_pred for lbl in s)
            ref_ctr = Counter(lbl for s in sampled_ref for lbl in s)
            pv = [pred_ctr.get(lbl, 0) for lbl in all_labels]
            rv = [ref_ctr.get(lbl, 0) for lbl in all_labels]
            t, _ = kendalltau(pv, rv)
            if not np.isnan(t):
                boot_tau.append(t)

    def ci(samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    exact_match = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r) / len(
        merged
    )
    overlap = sum(1 for p, r in zip(pred_labels, ref_labels) if p & r) / len(merged)

    f1_ci, prec_ci, rec_ci = ci(boot_f1), ci(boot_prec), ci(boot_rec)
    exact_ci, overlap_ci = ci(boot_exact), ci(boot_overlap)
    tau_ci = ci(boot_tau) if boot_tau else None

    theme_freq, tau, tau_p = _compute_theme_frequencies(
        pred_labels, ref_labels, all_labels, len(merged), label_map
    )

    metrics = {
        "n_responses": len(merged),
        "n_labels": len(all_labels),
        "f1": round(f1, 4),
        "f1_ci": f1_ci,
        "precision": round(prec, 4),
        "precision_ci": prec_ci,
        "recall": round(rec, 4),
        "recall_ci": rec_ci,
        "exact_match": round(exact_match, 4),
        "exact_match_ci": exact_ci,
        "overlap_rate": round(overlap, 4),
        "overlap_rate_ci": overlap_ci,
        "kendall_tau": round(tau, 4) if not np.isnan(tau) else None,
        "kendall_tau_ci": tau_ci,
        "kendall_tau_pvalue": round(tau_p, 4) if not np.isnan(tau_p) else None,
        "theme_frequencies": theme_freq,
    }

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        if k in ("theme_frequencies",):
            continue
        if k.endswith("_ci"):
            table.add_row(k, f"[{v[0]}, {v[1]}]")
        else:
            table.add_row(k, str(v))
    console.print(table)

    pred_name = predicted_csv.stem
    ref_name = reference_csv.stem
    n_pred = len(pred)
    n_ref = len(ref)

    # Build rank maps for green highlighting
    by_ref = sorted(theme_freq, key=lambda x: x["ref_count"], reverse=True)
    by_pred = sorted(theme_freq, key=lambda x: x["pred_count"], reverse=True)
    ref_rank = {tf["label"]: i for i, tf in enumerate(by_ref)}
    pred_rank = {tf["label"]: i for i, tf in enumerate(by_pred)}

    # Table 1: sorted by reference (ground truth) count
    ref_table = Table(title=f"Theme Frequencies (by {ref_name})")
    ref_table.add_column("Theme", style="bold")
    ref_table.add_column(f"{ref_name}\nCount (n={n_ref})", justify="right")
    ref_table.add_column(f"{ref_name}\n%", justify="right")
    ref_table.add_column(f"{pred_name}\nCount (n={n_pred})", justify="right")
    ref_table.add_column(f"{pred_name}\n%", justify="right")
    for tf in by_ref:
        row_style = "green" if ref_rank[tf["label"]] == pred_rank[tf["label"]] else None
        ref_table.add_row(
            tf["theme_label"],
            str(tf["ref_count"]),
            f"{tf['ref_pct']:.1f}",
            str(tf["pred_count"]),
            f"{tf['pred_pct']:.1f}",
            style=row_style,
        )
    console.print(ref_table)

    # Table 2: sorted by predicted count
    pred_table = Table(title=f"Theme Frequencies (by {pred_name})")
    pred_table.add_column("Theme", style="bold")
    pred_table.add_column(f"{pred_name}\nCount (n={n_pred})", justify="right")
    pred_table.add_column(f"{pred_name}\n%", justify="right")
    pred_table.add_column(f"{ref_name}\nCount (n={n_ref})", justify="right")
    pred_table.add_column(f"{ref_name}\n%", justify="right")
    for tf in by_pred:
        row_style = "green" if ref_rank[tf["label"]] == pred_rank[tf["label"]] else None
        pred_table.add_row(
            tf["theme_label"],
            str(tf["pred_count"]),
            f"{tf['pred_pct']:.1f}",
            str(tf["ref_count"]),
            f"{tf['ref_pct']:.1f}",
            style=row_style,
        )
    console.print(pred_table)

    if output:
        output.write_text(json.dumps(metrics, indent=2))
        console.print(f"[green]Wrote metrics to {output}[/green]")


# ---------------------------------------------------------------------------
# Calibrate helpers
# ---------------------------------------------------------------------------


def _print_themes_table(themes_df: pd.DataFrame) -> None:
    """Display themes in a Rich table."""
    table = Table(title="Theme List")
    table.add_column("ID", style="bold cyan")
    table.add_column("Label", style="bold")
    table.add_column("Description")
    for _, row in themes_df.iterrows():
        topic = str(row["topic"])
        if ":" in topic:
            label, desc = topic.split(":", 1)
        else:
            label, desc = topic, ""
        table.add_row(str(row["topic_id"]), label.strip(), desc.strip())
    console.print(table)


def _prompt_human_codes(
    valid_codes: set[str], themes_df: pd.DataFrame
) -> list[str] | None:
    """Read topic codes from stdin. Returns None on quit.

    Handles 't' to re-display themes inline, so the caller doesn't need
    sentinel values.
    """
    while True:
        raw = console.input(
            "[bold]Your codes[/bold] (comma-separated, or [cyan]t[/cyan]=themes, [cyan]q[/cyan]=quit): "
        )
        raw = raw.strip()
        if raw.lower() == "q":
            return None
        if raw.lower() == "t":
            _print_themes_table(themes_df)
            continue
        codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
        if not codes:
            console.print("[red]Please enter at least one code.[/red]")
            continue
        invalid = [c for c in codes if c not in valid_codes]
        if invalid:
            console.print(f"[red]Invalid codes: {', '.join(invalid)}. Try again.[/red]")
            continue
        return codes


def _display_comparison(
    human_codes: list[str], llm_codes: list[str], themes_df: pd.DataFrame
) -> None:
    """Show human vs LLM codes side-by-side."""
    topic_lookup = {}
    for _, row in themes_df.iterrows():
        topic = str(row["topic"])
        label = topic.split(":", 1)[0].strip() if ":" in topic else topic
        topic_lookup[str(row["topic_id"])] = label

    table = Table(title="Comparison", show_header=True)
    table.add_column("Human", style="green")
    table.add_column("LLM", style="blue")

    human_str = ", ".join(
        f"{c} ({topic_lookup.get(c, '?')})" for c in sorted(human_codes)
    )
    llm_str = ", ".join(f"{c} ({topic_lookup.get(c, '?')})" for c in sorted(llm_codes))
    table.add_row(human_str, llm_str)
    console.print(table)


def _prompt_judgment() -> str:
    """When codes differ, ask human to judge. Returns 'h', 'l', 'n', or 's'."""
    while True:
        choice = (
            console.input(
                "[bold]Who is right?[/bold] [green]\\[h][/green]uman / [blue]\\[l][/blue]lm / [magenta]\\[n][/magenta]either / [yellow]\\[s][/yellow]kip: "
            )
            .strip()
            .lower()
        )
        if choice in ("h", "l", "n", "s"):
            return choice
        console.print("[red]Please enter h, l, n, or s.[/red]")


def _prompt_explanation() -> str:
    """Get a brief explanation from the human."""
    return console.input("[bold]Brief explanation:[/bold] ").strip()


def _save_gold_examples(
    gold_rows: list[dict], output_path: Path, existing_df: pd.DataFrame | None
) -> None:
    """Write accumulated gold standard examples to CSV, appending to existing."""
    if not gold_rows:
        return
    new_df = pd.DataFrame(gold_rows)
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Calibrate command
# ---------------------------------------------------------------------------


@app.command()
def calibrate(
    input_csv: Path = typer.Argument(..., help="Path to responses CSV"),
    themes: Path = typer.Option(..., "--themes", "-t", help="Path to themes CSV"),
    question: str = typer.Option(
        ..., "--question", "-q", help="The survey/consultation question"
    ),
    output: Path = typer.Option(
        "examples.csv", "--output", "-o", help="Gold standard examples CSV path"
    ),
    model: str = typer.Option(
        "anthropic.claude-3-7-sonnet-20250219-v1:0", "--model", help="Bedrock model ID"
    ),
    region: str = typer.Option("eu-west-2", "--region", help="AWS region"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    streak_target: int = typer.Option(
        5, "--streak", help="Consecutive correct LLM responses before stop is offered"
    ),
    id_col: str = typer.Option(
        "response_id", "--id-col", help="Column name for response IDs"
    ),
    text_col: str = typer.Option(
        "response", "--text-col", help="Column name for response text"
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Randomize response order"
    ),
):
    """Interactively calibrate LLM mapping by comparing human and LLM labels row by row.

    Every reviewed row is saved as a gold standard example. Agreements are saved
    with a blank explanation; disagreements are saved with the correct codes and
    an optional explanation. If neither human nor LLM got it right, choose
    [n]either to provide the correct codes.

    You must label at least STREAK consecutive correct LLM responses before you
    can stop (default 5).
    """
    df = load_responses(input_csv, id_col=id_col, text_col=text_col)
    themes_df = load_themes(themes)
    llm = make_llm(model, region, profile)

    # Load existing examples if output file exists
    existing_df = None
    if output.exists():
        existing_df = read_tabular(output)
        console.print(
            f"[dim]Loaded {len(existing_df)} existing examples from {output}[/dim]"
        )

    # Build initial examples string from existing file
    examples_df = existing_df.copy() if existing_df is not None else pd.DataFrame()

    valid_codes = set(str(tid) for tid in themes_df["topic_id"])

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Display themes
    _print_themes_table(themes_df)
    console.print(
        f"\n[bold]{len(df)} responses to label. Enter [cyan]t[/cyan] to re-show themes, [cyan]q[/cyan] to quit.[/bold]\n"
    )

    gold_rows: list[dict] = []
    consecutive_correct = 0
    total_reviewed = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        response_id = int(row["response_id"])
        response_text = str(row["response"])

        console.rule(
            f"[bold]Response {idx + 1}/{len(df)}[/bold] (streak: {consecutive_correct}/{streak_target})"
        )
        console.print(f"\n[italic]{response_text}[/italic]\n")

        # Get human codes (handles 't' for themes internally)
        human_codes = _prompt_human_codes(valid_codes, themes_df)
        if human_codes is None:
            break

        # Get LLM codes
        examples_str = format_mapping_examples(examples_df)
        console.print("[dim]Asking LLM...[/dim]")
        try:
            llm_codes = asyncio.run(
                classify_single_response(
                    response_id=response_id,
                    response_text=response_text,
                    llm=llm,
                    question=question,
                    refined_themes_df=themes_df,
                    examples=examples_str,
                )
            )
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            console.print("[yellow]Skipping this response.[/yellow]")
            continue

        # Compare
        _display_comparison(human_codes, llm_codes, themes_df)
        total_reviewed += 1

        human_set = set(human_codes)
        llm_set = set(llm_codes)

        correct_codes: list[str] | None = None
        explanation = ""

        if human_set == llm_set:
            consecutive_correct += 1
            console.print(
                f"[green]Agreement! Streak: {consecutive_correct}/{streak_target}[/green]"
            )
            correct_codes = human_codes
        else:
            judgment = _prompt_judgment()
            if judgment == "h":
                explanation = _prompt_explanation()
                correct_codes = human_codes
                consecutive_correct = 0
                console.print(
                    "[yellow]Saved with human codes. Streak reset to 0.[/yellow]"
                )
            elif judgment == "l":
                correct_codes = llm_codes
                consecutive_correct += 1
                console.print(
                    f"[blue]LLM was right. Streak: {consecutive_correct}/{streak_target}[/blue]"
                )
            elif judgment == "n":
                correct_codes = _prompt_human_codes(valid_codes, themes_df)
                if correct_codes is None:
                    break
                explanation = _prompt_explanation()
                consecutive_correct = 0
                console.print(
                    "[yellow]Saved with corrected codes. Streak reset to 0.[/yellow]"
                )
            else:
                console.print("[dim]Skipped.[/dim]")

        # Save gold standard example (unless skipped)
        if correct_codes is not None:
            gold_row: dict = {"response": response_text}
            for i, code in enumerate(sorted(correct_codes), 1):
                gold_row[f"code_{i}"] = code
            gold_row["explanation"] = explanation
            gold_rows.append(gold_row)
            examples_df = pd.concat(
                [examples_df, pd.DataFrame([gold_row])], ignore_index=True
            )

        # Check if streak target reached
        if consecutive_correct >= streak_target:
            console.print(
                f"\n[bold green]Streak target reached ({streak_target})![/bold green]"
            )
            choice = (
                console.input(
                    "[bold]Continue?[/bold] [green]\\[y][/green]es / [red]\\[n][/red]o: "
                )
                .strip()
                .lower()
            )
            if choice != "y":
                break
            consecutive_correct = 0

    # Save results
    _save_gold_examples(gold_rows, output, existing_df)

    # Summary
    console.print()
    console.rule("[bold]Calibration Summary[/bold]")
    summary = Table()
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Responses reviewed", str(total_reviewed))
    summary.add_row("New gold examples", str(len(gold_rows)))
    total_examples = (len(existing_df) if existing_df is not None else 0) + len(
        gold_rows
    )
    summary.add_row("Total examples in file", str(total_examples))
    summary.add_row("Final streak", str(consecutive_correct))
    console.print(summary)

    if gold_rows:
        console.print(f"[green]Wrote {len(gold_rows)} new examples to {output}[/green]")
    else:
        console.print("[dim]No new examples to save.[/dim]")


if __name__ == "__main__":
    app()
