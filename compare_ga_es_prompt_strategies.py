#!/usr/bin/env python3
"""Compare GA prompt search vs evolutionary strategy prompt search on NQ.

This reads the detailed trace CSV from:
- GA + RAG: crossover + mutation + elitism
- ES + RAG: tournament selection + mutation + elitism, no crossover

It uses the first 100 questions by default and writes clean CSVs plus plots.
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
DEFAULT_OUT_DIR = PROJECT_DIR / "nq_clean_monitor_exports" / "ga_vs_es_first100"

TRACE_FILES = {
    "GA + RAG": {
        "path": PROJECT_DIR / "nq_langgraph_rag_outputs" / "nq_genetic_rag_llama_assessment_traces.csv",
        "strategy": "GA: crossover + mutation + elitism",
    },
    "ES + RAG": {
        "path": PROJECT_DIR
        / "nq_evolution_strategy_rag_outputs"
        / "nq_evolution_strategy_rag_llama_assessment_traces.csv",
        "strategy": "ES: tournament selection + mutation + elitism, no crossover",
    },
}

MODEL_ORDER = list(TRACE_FILES)
COLORS = {"GA + RAG": "#ff7f0e", "ES + RAG": "#d62728"}

METRICS = [
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
}


def value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if name in row.index and pd.notna(row[name]):
        return row[name]
    return default


def normalize(path: Path, model: str, strategy: str, max_rows: int) -> pd.DataFrame:
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
        judge = pd.to_numeric(value(row, "llm_judge_score_0_to_5"), errors="coerce")
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "source_file": str(path),
                "row_number": value(row, "row_number", ""),
                "sample_id": value(row, "sample_id", ""),
                "question": value(row, "question", ""),
                "short_f1": value(row, "rag_short_token_f1", value(row, "short_token_f1")),
                "long_f1": value(row, "rag_long_token_f1", value(row, "long_token_f1")),
                "short_bleu": value(row, "rag_short_bleu"),
                "long_bleu": value(row, "rag_long_bleu"),
                "short_rouge1": value(row, "rag_short_rouge1"),
                "long_rouge1": value(row, "rag_long_rouge1"),
                "llm_judge_score_0_to_5": judge,
                "llm_judge_score_norm": judge / 5.0 if pd.notna(judge) else np.nan,
                "retrieved_self_hit": value(row, "retrieved_self_hit"),
                "estimated_lm_calls": value(row, "estimated_lm_calls"),
                "elapsed_seconds": value(row, "elapsed_seconds"),
                "evolved_prompt_score": value(row, "evolved_prompt_score"),
                "evolved_prompt_history_json": value(row, "evolved_prompt_history_json", ""),
                "prompt": value(row, "prompt", ""),
            }
        )
    out = pd.DataFrame(rows)
    numeric_cols = [
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
        "evolved_prompt_score",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_fused(df: pd.DataFrame) -> pd.DataFrame:
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
        normalize(meta["path"], model, meta["strategy"], max_rows)
        for model, meta in TRACE_FILES.items()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No GA or ES trace CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return add_fused(df).sort_values(["model", "row_number"]).reset_index(drop=True)


def write_csvs(df: pd.DataFrame, out_dir: Path, max_rows: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "ga_vs_es_first100_trace_rows_long.csv"
    df.to_csv(long_path, index=False)
    print(f"[write] {long_path}")

    cols = [
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
        "evolved_prompt_score",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    means = df.groupby("model", observed=True)[cols].mean(numeric_only=True).reset_index()
    counts = df.groupby("model", observed=True).size().reset_index(name="rows_used")
    strategies = df.groupby("model", observed=True)["strategy"].first().reset_index()
    summary = counts.merge(strategies, on="model", how="left").merge(means, on="model", how="left")
    summary["model"] = summary["model"].astype(str)
    summary = summary.drop_duplicates(subset=["model"], keep="first")
    summary["target_rows"] = max_rows
    summary["is_full_100"] = summary["rows_used"] >= max_rows
    summary["balanced_fused_rank"] = summary["balanced_fused_score"].rank(
        ascending=False,
        method="min",
    )
    summary_path = out_dir / "ga_vs_es_first100_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")

    chart = summary.melt(
        id_vars=["model", "strategy", "rows_used", "target_rows", "is_full_100"],
        value_vars=[m for m in METRICS + ["elapsed_seconds", "estimated_lm_calls"] if m in summary.columns],
        var_name="metric",
        value_name="value",
    )
    chart["metric_label"] = chart["metric"].map(LABELS).fillna(chart["metric"])
    chart_path = out_dir / "ga_vs_es_first100_chart_ready_long.csv"
    chart.to_csv(chart_path, index=False)
    print(f"[write] {chart_path}")
    return summary


def plot_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    data = (
        summary.drop_duplicates(subset=["model"], keep="first")
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(how="all")
    )
    x = np.arange(len(METRICS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, model in enumerate(data.index):
        ax.bar(
            x + (offset - 0.5) * width,
            data.loc[model, METRICS],
            width=width,
            label=str(model),
            color=COLORS.get(str(model)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METRICS], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("GA vs Evolution Strategy prompt search, first 100 questions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "ga_vs_es_first100_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_runtime(summary: pd.DataFrame, out_dir: Path) -> None:
    data = (
        summary.drop_duplicates(subset=["model"], keep="first")
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(how="all")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data.index.astype(str), data["elapsed_seconds"], color=[COLORS.get(str(m)) for m in data.index])
    ax.set_ylabel("Mean seconds per question")
    ax.set_title("GA vs ES runtime, first 100 questions")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = out_dir / "ga_vs_es_first100_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused(summary: pd.DataFrame, out_dir: Path) -> None:
    data = (
        summary.drop_duplicates(subset=["model"], keep="first")
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(how="all")
    )
    cols = ["balanced_fused_score_raw", "runtime_penalty", "call_penalty", "balanced_fused_score"]
    labels = ["Raw fused", "Runtime penalty", "Call penalty", "Final fused"]
    x = np.arange(len(cols))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, model in enumerate(data.index):
        ax.bar(
            x + (offset - 0.5) * width,
            data.loc[model, cols],
            width=width,
            label=str(model),
            color=COLORS.get(str(model)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("GA vs ES balanced fused score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "ga_vs_es_first100_fused_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_all(args.max_rows)
    summary = write_csvs(df, args.out_dir, args.max_rows)
    plot_metrics(summary, args.out_dir)
    plot_runtime(summary, args.out_dir)
    plot_fused(summary, args.out_dir)
    print("\n=== GA vs ES summary ===")
    print(
        summary[
            [
                "model",
                "rows_used",
                "is_full_100",
                "short_f1",
                "long_f1",
                "llm_judge_score_norm",
                "balanced_fused_score",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
