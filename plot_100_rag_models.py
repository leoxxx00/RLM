#!/usr/bin/env python3
"""Plot first-50-question Natural Questions RAG comparisons.

This script reads the detailed trace CSVs, takes the first N questions from
each model, normalizes metric names, and writes clean comparison CSVs and plots.

Default N is 50. If one model has fewer than N rows, the script keeps the
available rows and records that in the summary, so you can see when a comparison
is not fully fair yet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM")
DEFAULT_OUT_DIR = PROJECT_DIR / "nq_clean_monitor_exports" / "first50_model_comparison"


TRACE_FILES = {
    "Llama 3 8B + RAG": {
        "path": PROJECT_DIR / "nq_langgraph_rag_outputs" / "nq_rag_llama_assessment_traces.csv",
        "prefix": "rag",
    },
    "GRLM + RAG": {
        "path": PROJECT_DIR / "nq_langgraph_rag_outputs" / "nq_genetic_rag_llama_assessment_traces.csv",
        "prefix": "rag",
    },
    "RLM REPL + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl_rag_outputs" / "nq_rlm_repl_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "RLM REPL 8-step + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl8_rag_outputs" / "nq_rlm_repl8_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "RLM REPL 12-step + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl12_rag_outputs" / "nq_rlm_repl12_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "Focused RLM REPL + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl_focused_outputs" / "nq_rlm_repl_focused_assessment_traces.csv",
        "prefix": "rlm",
    },
}


MODEL_ORDER = list(TRACE_FILES)

MODEL_COLORS = {
    "Llama 3 8B + RAG": "#2ca02c",
    "GRLM + RAG": "#ff7f0e",
    "RLM REPL + RAG": "#9467bd",
    "RLM REPL 8-step + RAG": "#8c564b",
    "RLM REPL 12-step + RAG": "#d62728",
    "Focused RLM REPL + RAG": "#1f77b4",
}


QUALITY_METRICS = [
    "short_f1",
    "long_f1",
    "short_bleu",
    "long_bleu",
    "short_rouge1",
    "long_rouge1",
    "llm_judge_score_norm",
    "balanced_fused_score",
]


LABELS = {
    "short_f1": "Short F1",
    "long_f1": "Long F1",
    "short_bleu": "Short BLEU",
    "long_bleu": "Long BLEU",
    "short_rouge1": "Short ROUGE-1",
    "long_rouge1": "Long ROUGE-1",
    "llm_judge_score_norm": "LLM Judge\n0-1",
    "balanced_fused_score": "Balanced\nFused",
    "elapsed_seconds": "Seconds",
    "estimated_lm_calls": "LM Calls",
}


def coerce_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def row_value(row: pd.Series, name: str, default: Any = "") -> Any:
    if name in row.index and pd.notna(row[name]):
        return row[name]
    return default


def metric_value(row: pd.Series, prefix: str, metric: str) -> Any:
    return row_value(row, f"{prefix}_{metric}", row_value(row, metric, np.nan))


def normalize_trace(model: str, path: Path, prefix: str, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        print(f"[missing] {model}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "row_number" in df.columns:
        df = df.sort_values("row_number")
    df = df.head(max_rows)
    print(f"[read] {model}: rows_used={len(df)} path={path}")

    rows = []
    for _, row in df.iterrows():
        judge_score = coerce_number(row_value(row, "llm_judge_score_0_to_5", np.nan))
        item = {
            "model": model,
            "source_file": str(path),
            "row_number": row_value(row, "row_number", ""),
            "sample_id": row_value(row, "sample_id", ""),
            "question": row_value(row, "question", ""),
            "short_f1": metric_value(row, prefix, "short_token_f1"),
            "long_f1": metric_value(row, prefix, "long_token_f1"),
            "short_bleu": metric_value(row, prefix, "short_bleu"),
            "long_bleu": metric_value(row, prefix, "long_bleu"),
            "short_rouge1": metric_value(row, prefix, "short_rouge1"),
            "long_rouge1": metric_value(row, prefix, "long_rouge1"),
            "short_rouge2": metric_value(row, prefix, "short_rouge2"),
            "long_rouge2": metric_value(row, prefix, "long_rouge2"),
            "short_rougeL": metric_value(row, prefix, "short_rougeL"),
            "long_rougeL": metric_value(row, prefix, "long_rougeL"),
            "llm_judge_score_0_to_5": judge_score,
            "llm_judge_score_norm": judge_score / 5.0 if pd.notna(judge_score) else np.nan,
            "retrieved_self_hit": row_value(row, "retrieved_self_hit", np.nan),
            "estimated_lm_calls": row_value(row, "estimated_lm_calls", np.nan),
            "elapsed_seconds": row_value(row, "elapsed_seconds", np.nan),
            "answer_summary": row_value(row, "answer_summary", ""),
            "trace_summary": row_value(row, "trace_summary", ""),
            "judge_summary": row_value(row, "judge_summary", ""),
        }
        rows.append(item)

    out = pd.DataFrame(rows)
    numeric_cols = [
        "short_f1",
        "long_f1",
        "short_bleu",
        "long_bleu",
        "short_rouge1",
        "long_rouge1",
        "short_rouge2",
        "long_rouge2",
        "short_rougeL",
        "long_rougeL",
        "llm_judge_score_0_to_5",
        "llm_judge_score_norm",
        "retrieved_self_hit",
        "estimated_lm_calls",
        "elapsed_seconds",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_balanced_fused_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["balanced_fused_score_raw"] = (
        0.15 * out["short_f1"].fillna(0)
        + 0.20 * out["long_f1"].fillna(0)
        + 0.10 * out["short_bleu"].fillna(0)
        + 0.10 * out["long_bleu"].fillna(0)
        + 0.10 * out["short_rouge1"].fillna(0)
        + 0.10 * out["long_rouge1"].fillna(0)
        + 0.20 * out["llm_judge_score_norm"].fillna(0)
        + 0.05 * out["retrieved_self_hit"].fillna(0)
    )

    runtime = out["elapsed_seconds"].fillna(out["elapsed_seconds"].median())
    calls = out["estimated_lm_calls"].fillna(out["estimated_lm_calls"].median())
    runtime_range = runtime.max() - runtime.min()
    calls_range = calls.max() - calls.min()
    out["runtime_penalty"] = 0.0 if runtime_range == 0 else 0.05 * (runtime - runtime.min()) / runtime_range
    out["call_penalty"] = 0.0 if calls_range == 0 else 0.05 * (calls - calls.min()) / calls_range
    out["balanced_fused_score"] = (
        out["balanced_fused_score_raw"] - out["runtime_penalty"] - out["call_penalty"]
    )
    return out


def read_all(max_rows: int) -> pd.DataFrame:
    frames = [
        normalize_trace(model, meta["path"], meta["prefix"], max_rows)
        for model, meta in TRACE_FILES.items()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No trace CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = add_balanced_fused_score(df)
    return df.sort_values(["model", "row_number"]).reset_index(drop=True)


def write_csvs(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / f"first{max_rows}_trace_rows_long.csv"
    df.to_csv(long_path, index=False)
    print(f"[write] {long_path}")

    summary_cols = [
        "short_f1",
        "long_f1",
        "short_bleu",
        "long_bleu",
        "short_rouge1",
        "long_rouge1",
        "llm_judge_score_0_to_5",
        "llm_judge_score_norm",
        "retrieved_self_hit",
        "estimated_lm_calls",
        "elapsed_seconds",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    summary = df.groupby("model", observed=False)[summary_cols].mean(numeric_only=True).reset_index()
    counts = df.groupby("model", observed=False).size().reset_index(name="rows_used")
    summary = counts.merge(summary, on="model", how="left")
    summary["target_rows"] = max_rows
    summary["is_full_sample"] = summary["rows_used"] >= max_rows
    summary["balanced_fused_rank"] = summary["balanced_fused_score"].rank(
        ascending=False,
        method="min",
    )
    summary_path = out_dir / f"first{max_rows}_model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")

    chart_ready = summary.melt(
        id_vars=["model", "rows_used", "target_rows", "is_full_sample"],
        value_vars=[col for col in QUALITY_METRICS + ["elapsed_seconds", "estimated_lm_calls"] if col in summary.columns],
        var_name="metric",
        value_name="value",
    )
    chart_ready["metric_label"] = chart_ready["metric"].map(LABELS).fillna(chart_ready["metric"])
    chart_path = out_dir / f"first{max_rows}_chart_ready_long.csv"
    chart_ready.to_csv(chart_path, index=False)
    print(f"[write] {chart_path}")


def model_means(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("model", observed=False)[
        QUALITY_METRICS
        + [
            "balanced_fused_score_raw",
            "elapsed_seconds",
            "estimated_lm_calls",
            "runtime_penalty",
            "call_penalty",
        ]
    ].mean(numeric_only=True)
    return means.reindex(MODEL_ORDER).dropna(how="all")


def plot_quality(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    means = model_means(df)
    metrics = QUALITY_METRICS
    x = np.arange(len(metrics))
    width = min(0.16, 0.82 / max(1, len(means)))
    fig, ax = plt.subplots(figsize=(15, 7))
    for offset, model in enumerate(means.index):
        positions = x + (offset - (len(means) - 1) / 2) * width
        ax.bar(
            positions,
            means.loc[model, metrics],
            width=width,
            label=str(model),
            color=MODEL_COLORS.get(str(model)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in metrics], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Natural Questions first-{max_rows} comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"first{max_rows}_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_runtime(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    means = model_means(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        means.index.astype(str),
        means["elapsed_seconds"],
        color=[MODEL_COLORS.get(str(model)) for model in means.index],
    )
    ax.set_ylabel("Mean seconds per question")
    ax.set_title(f"Runtime, first {max_rows} rows")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / f"first{max_rows}_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_long(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    means = model_means(df)
    fig, ax = plt.subplots(figsize=(8, 7))
    for model, row in means.iterrows():
        ax.scatter(
            row["short_f1"],
            row["long_f1"],
            s=220,
            color=MODEL_COLORS.get(str(model)),
            label=str(model),
        )
        ax.annotate(str(model), (row["short_f1"], row["long_f1"]), xytext=(8, 5), textcoords="offset points")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Short-answer F1")
    ax.set_ylabel("Long-answer F1")
    ax.set_title(f"Short vs long F1, first {max_rows} rows")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"first{max_rows}_short_vs_long_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused_components(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    means = model_means(df)
    cols = ["balanced_fused_score_raw", "runtime_penalty", "call_penalty", "balanced_fused_score"]
    labels = ["Raw fused", "Runtime penalty", "Call penalty", "Final fused"]
    x = np.arange(len(cols))
    width = min(0.16, 0.82 / max(1, len(means)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, model in enumerate(means.index):
        positions = x + (offset - (len(means) - 1) / 2) * width
        ax.bar(
            positions,
            means.loc[model, cols],
            width=width,
            label=str(model),
            color=MODEL_COLORS.get(str(model)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean value")
    ax.set_title(f"Balanced fused score components, first {max_rows} rows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"first{max_rows}_balanced_fused_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_judge(df: pd.DataFrame, out_dir: Path, max_rows: int) -> None:
    means = model_means(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        means.index.astype(str),
        means["llm_judge_score_norm"],
        color=[MODEL_COLORS.get(str(model)) for model in means.index],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean judge score, normalized 0-1")
    ax.set_title(f"LLM-as-judge, first {max_rows} rows")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / f"first{max_rows}_llm_judge_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_all(args.max_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(df, args.out_dir, args.max_rows)
    plot_quality(df, args.out_dir, args.max_rows)
    plot_judge(df, args.out_dir, args.max_rows)
    plot_runtime(df, args.out_dir, args.max_rows)
    plot_short_long(df, args.out_dir, args.max_rows)
    plot_fused_components(df, args.out_dir, args.max_rows)

    means = model_means(df)
    print("\n=== First-row comparison means ===")
    print(
        means[
            [
                "short_f1",
                "long_f1",
                "short_bleu",
                "long_bleu",
                "short_rouge1",
                "long_rouge1",
                "llm_judge_score_norm",
                "balanced_fused_score",
                "elapsed_seconds",
            ]
        ].round(4)
    )


if __name__ == "__main__":
    main()
