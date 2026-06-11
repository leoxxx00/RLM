#!/usr/bin/env python3
"""Plot the active Natural Questions RAG experiments.

This script handles:
1. Llama 3 8B + RAG
2. GRLM + RAG
3. RLM REPL + RAG
4. RLM REPL 8-step + RAG
5. Focused RLM REPL + RAG

It reads each model summary CSV, normalizes RLM metric names to the same
rag_* columns, and writes clean comparison CSVs/plots.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLEAN_DIR = Path(
    os.environ.get(
        "NQ_CLEAN_MONITOR_EXPORTS",
        "/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports",
    )
)
RAG_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs")
RLM_REPL_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_outputs")
RLM_REPL8_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_outputs")
FOCUSED_RLM_REPL_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_outputs")

OUT_DIR = CLEAN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


SUMMARY_FILES = {
    "Llama 3 8B + RAG": RAG_DIR / "nq_rag_llama_summary.csv",
    "GRLM + RAG": RAG_DIR / "nq_genetic_rag_llama_summary.csv",
    "RLM REPL + RAG": RLM_REPL_DIR / "nq_rlm_repl_rag_summary.csv",
    "RLM REPL 8-step + RAG": RLM_REPL8_DIR / "nq_rlm_repl8_rag_summary.csv",
    "Focused RLM REPL + RAG": FOCUSED_RLM_REPL_DIR / "nq_rlm_repl_focused_summary.csv",
}


MODEL_ORDER = [
    "Llama 3 8B + RAG",
    "GRLM + RAG",
    "RLM REPL + RAG",
    "RLM REPL 8-step + RAG",
    "Focused RLM REPL + RAG",
]


MODEL_EXPLANATIONS = {
    "Llama 3 8B + RAG": "Plain LangGraph RAG baseline.",
    "GRLM + RAG": "Genetic prompt search plus RAG.",
    "RLM REPL + RAG": (
        "Algorithm-1 REPL-style recursive language model with external P snippets."
    ),
    "RLM REPL 8-step + RAG": (
        "Algorithm-1 REPL-style recursive model with an expanded 8-step budget."
    ),
    "Focused RLM REPL + RAG": (
        "Focused three-job REPL: understand question, extract evidence, generate answer."
    ),
}


MODEL_METHODS = {
    "Llama 3 8B + RAG": "Baseline retrieve -> answer -> metrics -> judge.",
    "GRLM + RAG": "Genetic prompt population is scored and evolved before answer.",
    "RLM REPL + RAG": (
        "InitREPL(P), AddFunction(sub_RLM), LLM proposes safe REPL actions, "
        "stdout metadata is appended to history, then FINAL/root aggregation."
    ),
    "RLM REPL 8-step + RAG": (
        "Same REPL scaffold as RLM REPL + RAG, but the LLM can run up to 8 "
        "safe REPL actions before fallback."
    ),
    "Focused RLM REPL + RAG": (
        "InitREPL(P), then fixed jobs: understand_question, "
        "extract_answer_evidence, and generate_final_answer."
    ),
}


ESTIMATED_LM_CALLS = {
    "Llama 3 8B + RAG": 2,
    "GRLM + RAG": 26,
    "RLM REPL + RAG": 10,
    "RLM REPL 8-step + RAG": 14,
    "Focused RLM REPL + RAG": 8,
}


RLM_REPL_RENAME = {
    "rlm_short_token_f1": "rag_short_token_f1",
    "rlm_long_token_f1": "rag_long_token_f1",
    "rlm_short_bleu": "rag_short_bleu",
    "rlm_long_bleu": "rag_long_bleu",
    "rlm_short_rouge1": "rag_short_rouge1",
    "rlm_long_rouge1": "rag_long_rouge1",
    "rlm_short_rouge2": "rag_short_rouge2",
    "rlm_long_rouge2": "rag_long_rouge2",
    "rlm_short_rougeL": "rag_short_rougeL",
    "rlm_long_rougeL": "rag_long_rougeL",
}


QUALITY_METRICS = [
    "rag_short_token_f1",
    "rag_long_token_f1",
    "rag_short_bleu",
    "rag_long_bleu",
    "rag_short_rouge1",
    "rag_long_rouge1",
    "llm_judge_score_norm",
    "balanced_fused_score",
]


QUALITY_LABELS = {
    "rag_short_token_f1": "Short F1",
    "rag_long_token_f1": "Long F1",
    "rag_short_bleu": "Short BLEU",
    "rag_long_bleu": "Long BLEU",
    "rag_short_rouge1": "Short ROUGE-1",
    "rag_long_rouge1": "Long ROUGE-1",
    "llm_judge_score_norm": "LLM Judge\n(0-1)",
    "balanced_fused_score": "Balanced\nFused Score",
}


SHORT_LONG_PAIRS = [
    ("rag_short_token_f1", "rag_long_token_f1", "F1"),
    ("rag_short_bleu", "rag_long_bleu", "BLEU"),
    ("rag_short_rouge1", "rag_long_rouge1", "ROUGE-1"),
]


COLORS = {
    "Llama 3 8B + RAG": "#2ca02c",
    "GRLM + RAG": "#ff7f0e",
    "RLM REPL + RAG": "#9467bd",
    "RLM REPL 8-step + RAG": "#8c564b",
    "Focused RLM REPL + RAG": "#1f77b4",
}


def number_or_nan(value):
    try:
        if value == "" or value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fused_faithfulness(row):
    judge = number_or_nan(row.get("llm_judge_score_0_to_5"))
    values = [
        number_or_nan(row.get("rag_short_token_f1")),
        number_or_nan(row.get("rag_long_token_f1")),
        number_or_nan(row.get("rag_short_rouge1")),
        number_or_nan(row.get("rag_long_rouge1")),
        judge / 5.0 if not np.isnan(judge) else np.nan,
    ]
    values = [value for value in values if not np.isnan(value)]
    return round(float(np.mean(values)), 6) if values else np.nan


def minmax_penalty(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def add_balanced_fused_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["runtime_penalty"] = minmax_penalty(out.get("elapsed_seconds", pd.Series(dtype=float))) * 0.05
    out["call_penalty"] = minmax_penalty(out.get("estimated_lm_calls", pd.Series(dtype=float))) * 0.05
    out["balanced_fused_score_raw"] = (
        0.15 * out["rag_short_token_f1"].fillna(0)
        + 0.20 * out["rag_long_token_f1"].fillna(0)
        + 0.10 * out["rag_short_bleu"].fillna(0)
        + 0.10 * out["rag_long_bleu"].fillna(0)
        + 0.10 * out["rag_short_rouge1"].fillna(0)
        + 0.10 * out["rag_long_rouge1"].fillna(0)
        + 0.20 * out["llm_judge_score_norm"].fillna(0)
        + 0.05 * out["retrieved_self_hit"].fillna(0)
    )
    out["balanced_fused_score"] = (
        out["balanced_fused_score_raw"]
        - out["runtime_penalty"].fillna(0)
        - out["call_penalty"].fillna(0)
    )
    out["balanced_fused_rank"] = out["balanced_fused_score"].rank(
        ascending=False,
        method="min",
    )
    out["balanced_fused_formula"] = (
        "0.15*short_f1 + 0.20*long_f1 + 0.10*short_bleu + "
        "0.10*long_bleu + 0.10*short_rouge1 + 0.10*long_rouge1 + "
        "0.20*llm_judge_score_norm + 0.05*retrieved_self_hit - "
        "runtime_penalty - call_penalty"
    )
    return out


def normalize_summary_row(row, model_name):
    row = dict(row)
    for old, new in RLM_REPL_RENAME.items():
        if old in row and (
            new not in row
            or pd.isna(row.get(new))
            or row.get(new) == ""
        ):
            row[new] = row[old]
    if "elapsed_seconds" in row and "mean_elapsed_display" not in row:
        elapsed = number_or_nan(row.get("elapsed_seconds"))
        row["mean_elapsed_display"] = (
            f"elapsed={elapsed:.3f}s" if not np.isnan(elapsed) else ""
        )
    row["model"] = model_name
    row["model_explanation"] = MODEL_EXPLANATIONS.get(model_name, "")
    judge_score = number_or_nan(row.get("llm_judge_score_0_to_5"))
    row["llm_judge_score_norm"] = (
        judge_score / 5.0 if not np.isnan(judge_score) else np.nan
    )
    row["paper_method_mapping"] = row.get(
        "paper_method_mapping",
        MODEL_METHODS.get(model_name, ""),
    )
    if "estimated_lm_calls" not in row or pd.isna(row.get("estimated_lm_calls")):
        row["estimated_lm_calls"] = ESTIMATED_LM_CALLS.get(model_name, np.nan)
    if "faithfulness_fused_score" not in row or pd.isna(
        row.get("faithfulness_fused_score")
    ):
        row["faithfulness_fused_score"] = fused_faithfulness(row)
    row["compound_call_warning"] = row.get(
        "compound_call_warning",
        (
            "More LM calls can improve or hurt answers; compare quality against runtime."
            if number_or_nan(row.get("estimated_lm_calls")) > 2
            else "Low compound-call budget."
        ),
    )
    return row


def read_summaries():
    rows = []
    for model_name, path in SUMMARY_FILES.items():
        if not path.exists():
            print(f"[missing] {model_name}: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"[empty] {model_name}: {path}")
            continue
        row = normalize_summary_row(df.iloc[0].to_dict(), model_name)
        row["source_file"] = str(path)
        rows.append(row)
        print(f"[read] {model_name}: {path}")

    if not rows:
        raise RuntimeError("No summary CSVs found for the active model comparison.")

    combined = pd.DataFrame(rows)
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    combined = combined.sort_values("model")
    numeric_cols = [
        col
        for col in [
            *QUALITY_METRICS,
            "llm_judge_score_0_to_5",
            "faithfulness_fused_score",
            "balanced_fused_score_raw",
            "runtime_penalty",
            "call_penalty",
            "balanced_fused_score",
            "balanced_fused_rank",
            "recursive_call_count",
            "repl_step_count",
            "repl_code_call_count",
            "question_understanding_call_count",
            "prompt_improvement_call_count",
            "prompt_improvement_temperature",
            "prompt_improvement_max_tokens",
            "final_generation_call_count",
            "estimated_lm_calls",
            "elapsed_seconds",
        ]
        if col in combined.columns
    ]
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return combined


def row_count_title(df):
    if "rows_evaluated" not in df.columns:
        return "unknown rows"
    counts = sorted(set(int(v) for v in df["rows_evaluated"].dropna()))
    return str(counts[0]) if len(counts) == 1 else "mixed sample sizes"


def save_combined_summary(df):
    path = OUT_DIR / "five_model_comparison_summary.csv"
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def save_chart_ready_long_csv(df):
    metrics = [
        *QUALITY_METRICS,
        "llm_judge_score_0_to_5",
        "faithfulness_fused_score",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
        "balanced_fused_rank",
        "recursive_call_count",
        "repl_step_count",
        "repl_code_call_count",
        "question_understanding_call_count",
        "prompt_improvement_call_count",
        "prompt_improvement_temperature",
        "prompt_improvement_max_tokens",
        "final_generation_call_count",
        "estimated_lm_calls",
        "elapsed_seconds",
    ]
    label_map = {
        **QUALITY_LABELS,
        "llm_judge_score_0_to_5": "LLM Judge",
        "llm_judge_score_norm": "LLM Judge (0-1)",
        "faithfulness_fused_score": "Fused faithfulness",
        "balanced_fused_score_raw": "Balanced fused score before penalties",
        "runtime_penalty": "Runtime penalty",
        "call_penalty": "Call penalty",
        "balanced_fused_score": "Balanced fused score",
        "balanced_fused_rank": "Balanced fused rank",
        "recursive_call_count": "Recursive sub_RLM calls",
        "repl_step_count": "REPL steps",
        "repl_code_call_count": "LLM REPL-code calls",
        "question_understanding_call_count": "Question-understanding calls",
        "prompt_improvement_call_count": "Prompt-improvement calls",
        "prompt_improvement_temperature": "Improved prompt temperature",
        "prompt_improvement_max_tokens": "Improved prompt max tokens",
        "final_generation_call_count": "Final-generation calls",
        "estimated_lm_calls": "Estimated LM calls",
        "elapsed_seconds": "Elapsed seconds",
    }
    long_df = df.melt(
        id_vars=[
            col
            for col in [
                "model",
                "rows_evaluated",
                "source_file",
                "model_explanation",
                "paper_method_mapping",
                "compound_call_warning",
                "mean_elapsed_display",
            ]
            if col in df.columns
        ],
        value_vars=[metric for metric in metrics if metric in df.columns],
        var_name="metric",
        value_name="value",
    )
    long_df["metric_label"] = long_df["metric"].map(label_map)
    path = OUT_DIR / "five_model_chart_ready_long.csv"
    long_df.to_csv(path, index=False)
    print(f"[write] {path}")


def save_short_long_comparison_csv(df):
    rows = []
    for _, row in df.iterrows():
        for short_metric, long_metric, family in SHORT_LONG_PAIRS:
            short_value = row.get(short_metric, np.nan)
            long_value = row.get(long_metric, np.nan)
            rows.append(
                {
                    "model": row["model"],
                    "rows_evaluated": row.get("rows_evaluated", ""),
                    "metric_family": family,
                    "short_metric": short_metric,
                    "long_metric": long_metric,
                    "short_value": short_value,
                    "long_value": long_value,
                    "long_minus_short": long_value - short_value,
                    "source_file": row.get("source_file", ""),
                }
            )
    out = pd.DataFrame(rows)
    path = OUT_DIR / "five_model_short_long_comparison.csv"
    out.to_csv(path, index=False)
    print(f"[write] {path}")


def plot_quality_metrics(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(QUALITY_METRICS))
    width = min(0.8 / max(len(df), 1), 0.22)
    for i, (_, row) in enumerate(df.iterrows()):
        model = str(row["model"])
        values = [row.get(metric, np.nan) for metric in QUALITY_METRICS]
        offset = (i - (len(df) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([QUALITY_LABELS[m] for m in QUALITY_METRICS], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Natural Questions RAG comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUT_DIR / "five_model_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_long_metric_comparison(df):
    pair_df = []
    for _, row in df.iterrows():
        for short_metric, long_metric, family in SHORT_LONG_PAIRS:
            pair_df.append(
                {
                    "model": str(row["model"]),
                    "metric_family": family,
                    "answer_type": "Short",
                    "value": row.get(short_metric, np.nan),
                }
            )
            pair_df.append(
                {
                    "model": str(row["model"]),
                    "metric_family": family,
                    "answer_type": "Long",
                    "value": row.get(long_metric, np.nan),
                }
            )
    plot_df = pd.DataFrame(pair_df)
    fig, axes = plt.subplots(1, len(SHORT_LONG_PAIRS), figsize=(14, 5), sharey=True)
    for ax, (_, _, family) in zip(axes, SHORT_LONG_PAIRS):
        subset = plot_df[plot_df["metric_family"] == family]
        x = np.arange(len(df))
        width = 0.35
        short_values = subset[subset["answer_type"] == "Short"]["value"].to_numpy()
        long_values = subset[subset["answer_type"] == "Long"]["value"].to_numpy()
        labels = [str(model) for model in df["model"]]
        ax.bar(x - width / 2, short_values, width=width, label="Short", color="#4c78a8")
        ax.bar(x + width / 2, long_values, width=width, label="Long", color="#f58518")
        ax.set_title(family)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Short vs long answer metrics, n={row_count_title(df)}")
    fig.tight_layout()
    path = OUT_DIR / "five_model_short_long_metric_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_vs_long_f1_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in df.iterrows():
        model = str(row["model"])
        x = row.get("rag_short_token_f1", np.nan)
        y = row.get("rag_long_token_f1", np.nan)
        ax.scatter(x, y, s=160, color=COLORS[model])
        ax.annotate(model, (x, y), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Short-answer F1")
    ax.set_ylabel("Long-answer F1")
    ax.set_title(f"Short vs long F1 by model, n={row_count_title(df)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "five_model_short_vs_long_f1_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_runtime(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"].astype(str), df["elapsed_seconds"], color=[COLORS[str(m)] for m in df["model"]])
    ax.set_ylabel("Mean seconds per question")
    ax.set_title(f"Runtime comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "five_model_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_llm_judge(df):
    if "llm_judge_score_0_to_5" not in df.columns:
        print("[skip] No llm_judge_score_0_to_5 column")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        df["model"].astype(str),
        df["llm_judge_score_0_to_5"],
        color=[COLORS[str(m)] for m in df["model"]],
    )
    ax.set_ylim(0, 5.1)
    ax.set_ylabel("Mean LLM-as-judge score")
    ax.set_title(f"LLM judge comparison, n={row_count_title(df)}")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "five_model_llm_judge_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_quality_vs_runtime(df):
    out = df.copy()
    out["quality_mean"] = out[
        [
            "rag_short_token_f1",
            "rag_long_token_f1",
            "rag_short_rouge1",
            "rag_long_rouge1",
        ]
    ].mean(axis=1)
    path_csv = OUT_DIR / "five_model_quality_runtime_scatter_data.csv"
    out[
        [
            col
            for col in [
                "model",
                "rows_evaluated",
                "quality_mean",
                "elapsed_seconds",
                "estimated_lm_calls",
                "recursive_call_count",
                "repl_step_count",
                "repl_code_call_count",
                "question_understanding_call_count",
                "prompt_improvement_call_count",
                "final_generation_call_count",
                "faithfulness_fused_score",
                "balanced_fused_score_raw",
                "runtime_penalty",
                "call_penalty",
                "balanced_fused_score",
                "balanced_fused_rank",
            ]
            if col in out.columns
        ]
    ].to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in out.iterrows():
        ax.scatter(row["elapsed_seconds"], row["quality_mean"], s=150, color=COLORS[str(row["model"])])
        ax.annotate(
            str(row["model"]),
            (row["elapsed_seconds"], row["quality_mean"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Mean seconds per question")
    ax.set_ylabel("Mean quality score")
    ax.set_title("Quality vs runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "five_model_quality_vs_runtime_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")
    print(f"[write] {path_csv}")


def plot_balanced_fused_score(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = df.sort_values("balanced_fused_score", ascending=False)
    ax.bar(
        ordered["model"].astype(str),
        ordered["balanced_fused_score"],
        color=[COLORS[str(model)] for model in ordered["model"]],
    )
    ax.set_ylabel("Balanced fused score")
    ax.set_title("Balanced fused score with runtime and call penalties")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = OUT_DIR / "five_model_balanced_fused_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused_score_components(df):
    component_cols = [
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    labels = [
        "Raw fused",
        "Runtime penalty",
        "Call penalty",
        "Final fused",
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(component_cols))
    width = min(0.8 / max(len(df), 1), 0.22)
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
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUT_DIR / "five_model_balanced_fused_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def save_correlations(df):
    metrics = [
        metric
        for metric in [
            *QUALITY_METRICS,
            "llm_judge_score_0_to_5",
            "faithfulness_fused_score",
            "balanced_fused_score_raw",
            "runtime_penalty",
            "call_penalty",
            "balanced_fused_score",
            "balanced_fused_rank",
            "recursive_call_count",
            "repl_step_count",
            "repl_code_call_count",
            "question_understanding_call_count",
            "prompt_improvement_call_count",
            "final_generation_call_count",
            "estimated_lm_calls",
            "elapsed_seconds",
        ]
        if metric in df.columns
    ]
    corr = df[metrics].corr(numeric_only=True)
    path = OUT_DIR / "five_model_metric_correlations.csv"
    corr.to_csv(path)
    print(f"[write] {path}")


def print_suggestions(df):
    print("\n=== Active model comparison ===")
    if "balanced_fused_score" in df.columns:
        best_fused = df.loc[df["balanced_fused_score"].idxmax()]
        print(f"Best balanced fused score: {best_fused['model']} ({best_fused['balanced_fused_score']:.3f})")
    if "rag_short_token_f1" in df.columns:
        best_short = df.loc[df["rag_short_token_f1"].idxmax()]
        print(f"Best short F1: {best_short['model']} ({best_short['rag_short_token_f1']:.3f})")
    if "rag_long_token_f1" in df.columns:
        best_long = df.loc[df["rag_long_token_f1"].idxmax()]
        print(f"Best long F1: {best_long['model']} ({best_long['rag_long_token_f1']:.3f})")
    if "elapsed_seconds" in df.columns:
        fastest = df.loc[df["elapsed_seconds"].idxmin()]
        print(f"Fastest: {fastest['model']} ({fastest['elapsed_seconds']:.2f}s/question)")
    print("\nOpen five_model_comparison_summary.csv first, then the PNG plots.")


def main():
    df = read_summaries()
    df = add_balanced_fused_scores(df)
    save_combined_summary(df)
    save_chart_ready_long_csv(df)
    save_short_long_comparison_csv(df)
    save_correlations(df)
    plot_quality_metrics(df)
    plot_short_long_metric_comparison(df)
    plot_short_vs_long_f1_scatter(df)
    plot_runtime(df)
    plot_llm_judge(df)
    plot_quality_vs_runtime(df)
    plot_balanced_fused_score(df)
    plot_fused_score_components(df)
    print_suggestions(df)


if __name__ == "__main__":
    main()
