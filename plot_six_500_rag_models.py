#!/usr/bin/env python3
"""Plot the six full 500-row Natural Questions RAG/RLM experiments.

This matches the style of plot_six_rag_models.py, but uses the full
500-row trace files and the checkpointed 8-step RLM output.

Run:
    cd /Users/htet/Desktop/Projects/X-RLM
    source venv/bin/activate
    python plot_six_500_rag_models.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_six_500_metrics import MODEL_ORDER, read_all, summarize


CLEAN_DIR = Path(
    os.environ.get(
        "NQ_CLEAN_MONITOR_EXPORTS",
        "/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports",
    )
)
OUT_DIR = CLEAN_DIR / "six_models_500_plots_like_original"

COLORS = {
    "Llama 3 8B + RAG": "#2ca02c",
    "GA + RAG": "#ff7f0e",
    "ES + RAG": "#d62728",
    "RLM REPL + RAG": "#9467bd",
    "RLM REPL 8-step checkpoint + RAG": "#8c564b",
    "Focused RLM REPL + RAG": "#1f77b4",
}

MODEL_EXPLANATIONS = {
    "Llama 3 8B + RAG": "Plain LangGraph RAG baseline.",
    "GA + RAG": "Genetic prompt search plus RAG.",
    "ES + RAG": "Evolution-strategy prompt optimization plus RAG.",
    "RLM REPL + RAG": "REPL-style recursive language model with RAG snippets.",
    "RLM REPL 8-step checkpoint + RAG": "Checkpointed 8-step REPL-style RLM with RAG snippets.",
    "Focused RLM REPL + RAG": "Focused three-job REPL pipeline.",
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

QUALITY_LABELS = {
    "short_f1": "Short F1",
    "long_f1": "Long F1",
    "short_bleu": "Short BLEU",
    "long_bleu": "Long BLEU",
    "short_rouge1": "Short ROUGE-1",
    "long_rouge1": "Long ROUGE-1",
    "llm_judge_score_norm": "LLM Judge\n(0-1)",
    "balanced_fused_score": "Balanced\nFused Score",
}

SHORT_LONG_PAIRS = [
    ("short_f1", "long_f1", "F1"),
    ("short_bleu", "long_bleu", "BLEU"),
    ("short_rouge1", "long_rouge1", "ROUGE-1"),
]


def row_count_title(df: pd.DataFrame) -> str:
    if "rows_used" not in df.columns:
        return "unknown rows"
    counts = sorted(set(int(v) for v in df["rows_used"].dropna()))
    return str(counts[0]) if len(counts) == 1 else "mixed sample sizes"


def add_context_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["model_explanation"] = out["model"].astype(str).map(MODEL_EXPLANATIONS)
    out["source"] = "Full 500-row trace aggregation"
    return out


def save_combined_summary(df: pd.DataFrame) -> None:
    path = OUT_DIR / "six_model_500_comparison_summary.csv"
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def save_chart_ready_long_csv(df: pd.DataFrame) -> None:
    metrics = [
        *QUALITY_METRICS,
        "llm_judge_score_0_to_5",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_rank",
        "estimated_lm_calls",
        "elapsed_seconds",
    ]
    value_vars = [metric for metric in metrics if metric in df.columns]
    long_df = df.melt(
        id_vars=[
            "model",
            "rows_used",
            "target_rows",
            "is_full_target",
            "model_explanation",
            "source",
        ],
        value_vars=value_vars,
        var_name="metric",
        value_name="value",
    )
    long_df["metric_label"] = long_df["metric"].map(
        {
            **QUALITY_LABELS,
            "llm_judge_score_0_to_5": "LLM Judge",
            "balanced_fused_score_raw": "Raw fused score",
            "runtime_penalty": "Runtime penalty",
            "call_penalty": "Call penalty",
            "balanced_fused_rank": "Balanced fused rank",
            "estimated_lm_calls": "Estimated LM calls",
            "elapsed_seconds": "Elapsed seconds",
        }
    )
    path = OUT_DIR / "six_model_500_chart_ready_long.csv"
    long_df.to_csv(path, index=False)
    print(f"[write] {path}")


def save_short_long_comparison_csv(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        for short_metric, long_metric, family in SHORT_LONG_PAIRS:
            short_value = row.get(short_metric, np.nan)
            long_value = row.get(long_metric, np.nan)
            rows.append(
                {
                    "model": row["model"],
                    "rows_used": row["rows_used"],
                    "metric_family": family,
                    "short_metric": short_metric,
                    "long_metric": long_metric,
                    "short_value": short_value,
                    "long_value": long_value,
                    "long_minus_short": long_value - short_value,
                }
            )
    path = OUT_DIR / "six_model_500_short_long_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[write] {path}")


def save_correlations(df: pd.DataFrame) -> None:
    metrics = [
        *QUALITY_METRICS,
        "llm_judge_score_0_to_5",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_rank",
        "estimated_lm_calls",
        "elapsed_seconds",
    ]
    corr = df[[m for m in metrics if m in df.columns]].corr(numeric_only=True)
    path = OUT_DIR / "six_model_500_metric_correlations.csv"
    corr.to_csv(path)
    print(f"[write] {path}")


def plot_quality_metrics(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(QUALITY_METRICS))
    width = min(0.8 / max(len(df), 1), 0.18)
    for i, (_, row) in enumerate(df.iterrows()):
        model = str(row["model"])
        values = [row.get(metric, np.nan) for metric in QUALITY_METRICS]
        offset = (i - (len(df) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([QUALITY_LABELS[m] for m in QUALITY_METRICS], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Natural Questions six-model comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_long_metric_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(SHORT_LONG_PAIRS), figsize=(14, 5), sharey=True)
    labels = [str(model) for model in df["model"]]
    x = np.arange(len(labels))
    width = 0.35
    for ax, (short_metric, long_metric, family) in zip(axes, SHORT_LONG_PAIRS):
        ax.bar(x - width / 2, df[short_metric], width=width, label="Short", color="#4c78a8")
        ax.bar(x + width / 2, df[long_metric], width=width, label="Long", color="#f58518")
        ax.set_title(family)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Short vs long answer metrics, n={row_count_title(df)}")
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_short_long_metric_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_vs_long_f1_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in df.iterrows():
        model = str(row["model"])
        x = row.get("short_f1", np.nan)
        y = row.get("long_f1", np.nan)
        ax.scatter(x, y, s=160, color=COLORS[model])
        ax.annotate(model, (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Short-answer F1")
    ax.set_ylabel("Long-answer F1")
    ax.set_title(f"Short vs long F1 by model, n={row_count_title(df)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_short_vs_long_f1_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_runtime(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = df["model"].astype(str).tolist()
    ax.bar(labels, df["elapsed_seconds"], color=[COLORS[label] for label in labels])
    ax.set_ylabel("Mean seconds per question")
    ax.set_title(f"Runtime comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_llm_judge(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = df["model"].astype(str).tolist()
    ax.bar(labels, df["llm_judge_score_0_to_5"], color=[COLORS[label] for label in labels])
    ax.set_ylim(0, 5.1)
    ax.set_ylabel("Mean LLM-as-judge score")
    ax.set_title(f"LLM judge comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_llm_judge_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_quality_vs_runtime(df: pd.DataFrame) -> None:
    out = df.copy()
    out["quality_mean"] = out[["short_f1", "long_f1", "short_rouge1", "long_rouge1"]].mean(axis=1)
    path_csv = OUT_DIR / "six_model_500_quality_runtime_scatter_data.csv"
    out[
        [
            "model",
            "rows_used",
            "quality_mean",
            "elapsed_seconds",
            "estimated_lm_calls",
            "balanced_fused_score_raw",
            "runtime_penalty",
            "call_penalty",
            "balanced_fused_score",
            "balanced_fused_rank",
        ]
    ].to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in out.iterrows():
        model = str(row["model"])
        ax.scatter(row["elapsed_seconds"], row["quality_mean"], s=150, color=COLORS[model])
        ax.annotate(model, (row["elapsed_seconds"], row["quality_mean"]), xytext=(7, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean seconds per question")
    ax.set_ylabel("Mean quality score")
    ax.set_title("Quality vs runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_quality_vs_runtime_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")
    print(f"[write] {path_csv}")


def plot_balanced_fused_score(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = df.sort_values("balanced_fused_score", ascending=False)
    labels = ordered["model"].astype(str).tolist()
    ax.bar(labels, ordered["balanced_fused_score"], color=[COLORS[label] for label in labels])
    ax.set_ylabel("Balanced fused score")
    ax.set_title("Balanced fused score with runtime and call penalties")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_balanced_fused_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused_score_components(df: pd.DataFrame) -> None:
    component_cols = [
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    labels = ["Raw fused", "Runtime penalty", "Call penalty", "Final fused"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(component_cols))
    width = min(0.8 / max(len(df), 1), 0.18)
    for i, (_, row) in enumerate(df.iterrows()):
        model = str(row["model"])
        values = [row.get(col, np.nan) for col in component_cols]
        offset = (i - (len(df) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score / penalty")
    ax.set_title("Balanced fused score components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "six_model_500_balanced_fused_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def print_suggestions(df: pd.DataFrame) -> None:
    print("\n=== Six full 500-row model comparison ===")
    best_fused = df.loc[df["balanced_fused_score"].idxmax()]
    best_short = df.loc[df["short_f1"].idxmax()]
    best_long = df.loc[df["long_f1"].idxmax()]
    fastest = df.loc[df["elapsed_seconds"].idxmin()]
    print(f"Best balanced fused score: {best_fused['model']} ({best_fused['balanced_fused_score']:.3f})")
    print(f"Best short F1: {best_short['model']} ({best_short['short_f1']:.3f})")
    print(f"Best long F1: {best_long['model']} ({best_long['long_f1']:.3f})")
    print(f"Fastest: {fastest['model']} ({fastest['elapsed_seconds']:.2f}s/question)")
    print("\nOpen six_model_500_comparison_summary.csv first, then the PNG plots.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = read_all(max_rows=500)
    summary = summarize(traces, max_rows=500)
    summary["model"] = pd.Categorical(summary["model"].astype(str), MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")
    summary = add_context_columns(summary)

    save_combined_summary(summary)
    save_chart_ready_long_csv(summary)
    save_short_long_comparison_csv(summary)
    save_correlations(summary)
    plot_quality_metrics(summary)
    plot_short_long_metric_comparison(summary)
    plot_short_vs_long_f1_scatter(summary)
    plot_runtime(summary)
    plot_llm_judge(summary)
    plot_quality_vs_runtime(summary)
    plot_balanced_fused_score(summary)
    plot_fused_score_components(summary)
    print_suggestions(summary)


if __name__ == "__main__":
    main()
