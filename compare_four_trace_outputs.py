#!/usr/bin/env python3
"""Clean and compare five Natural Questions trace CSV outputs.

This script reads detailed per-question trace CSVs for:
- Llama 3 8B + RAG
- Genetic prompt RAG
- RLM REPL + RAG
- RLM REPL 8-step + RAG
- Focused RLM REPL + RAG

It normalizes the different column names, writes clean comparison CSVs, and
creates plots for quality, LLM-as-judge, runtime, best-model counts, and
metric correlations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM")
DEFAULT_OUT_DIR = PROJECT_DIR / "nq_clean_monitor_exports" / "trace_compare_five_models"

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
    "Focused RLM REPL + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl_focused_outputs" / "nq_rlm_repl_focused_assessment_traces.csv",
        "prefix": "rlm",
    },
}

MODEL_ORDER = [
    "Llama 3 8B + RAG",
    "GRLM + RAG",
    "RLM REPL + RAG",
    "RLM REPL 8-step + RAG",
    "Focused RLM REPL + RAG",
]

QUALITY_METRICS = [
    "short_f1",
    "long_f1",
    "short_bleu",
    "long_bleu",
    "short_rouge1",
    "long_rouge1",
    "llm_judge_score_norm",
    "quality_mean",
]

NUMERIC_COLUMNS = [
    "short_exact_match",
    "short_contains_any_gt",
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
    "quality_without_judge",
    "quality_mean",
    "balanced_fused_score_raw",
    "runtime_penalty",
    "call_penalty",
    "balanced_fused_score",
    "judge_minus_quality",
    "faithfulness_fused_score",
    "estimated_lm_calls",
    "elapsed_seconds",
    "retrieved_self_hit",
    "evolved_prompt_score",
    "recursive_call_count",
    "repl_step_count",
    "repl_code_call_count",
    "question_understanding_call_count",
    "prompt_improvement_call_count",
    "prompt_improvement_temperature",
    "prompt_improvement_max_tokens",
    "final_generation_call_count",
]

COLORS = {
    "Llama 3 8B + RAG": "#1f77b4",
    "GRLM + RAG": "#ff7f0e",
    "RLM REPL + RAG": "#2ca02c",
    "RLM REPL 8-step + RAG": "#8c564b",
    "Focused RLM REPL + RAG": "#9467bd",
}


def coerce_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def value(row: pd.Series, name: str, default: Any = "") -> Any:
    return row[name] if name in row.index and pd.notna(row[name]) else default


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
        0.15 * out["short_f1"].fillna(0)
        + 0.20 * out["long_f1"].fillna(0)
        + 0.10 * out["short_bleu"].fillna(0)
        + 0.10 * out["long_bleu"].fillna(0)
        + 0.10 * out["short_rouge1"].fillna(0)
        + 0.10 * out["long_rouge1"].fillna(0)
        + 0.20 * out["llm_judge_score_norm"].fillna(0)
        + 0.05 * out["retrieved_self_hit"].fillna(0)
    )
    out["balanced_fused_score"] = (
        out["balanced_fused_score_raw"]
        - out["runtime_penalty"].fillna(0)
        - out["call_penalty"].fillna(0)
    )
    out["balanced_fused_formula"] = (
        "0.15*short_f1 + 0.20*long_f1 + 0.10*short_bleu + "
        "0.10*long_bleu + 0.10*short_rouge1 + 0.10*long_rouge1 + "
        "0.20*llm_judge_score_norm + 0.05*retrieved_self_hit - "
        "runtime_penalty - call_penalty"
    )
    return out


def normalize_trace(model_name: str, path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[missing] {path}")
        return pd.DataFrame()

    print(f"[read] {model_name}: {path}")
    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        item = {
            "model": model_name,
            "source_file": str(path),
            "row_number": value(row, "row_number"),
            "sample_id": value(row, "sample_id"),
            "question": value(row, "question"),
            "ground_truth_short_answers": value(row, "ground_truth_short_answers"),
            "ground_truth_long_answers": value(row, "ground_truth_long_answers"),
            "model_short_answer": value(row, f"{prefix}_short_answer"),
            "model_long_answer": value(row, f"{prefix}_long_answer"),
            "answer_summary": value(row, "answer_summary"),
            "retrieved_sample_ids": value(row, "retrieved_sample_ids"),
            "retrieval_summary": value(row, "retrieval_summary"),
            "short_exact_match": value(row, "short_exact_match", np.nan),
            "short_contains_any_gt": value(row, "short_contains_any_gt", np.nan),
            "short_f1": value(row, f"{prefix}_short_token_f1", value(row, "short_token_f1", np.nan)),
            "long_f1": value(row, f"{prefix}_long_token_f1", value(row, "long_token_f1", np.nan)),
            "short_bleu": value(row, f"{prefix}_short_bleu", np.nan),
            "long_bleu": value(row, f"{prefix}_long_bleu", np.nan),
            "short_rouge1": value(row, f"{prefix}_short_rouge1", np.nan),
            "long_rouge1": value(row, f"{prefix}_long_rouge1", np.nan),
            "short_rouge2": value(row, f"{prefix}_short_rouge2", np.nan),
            "long_rouge2": value(row, f"{prefix}_long_rouge2", np.nan),
            "short_rougeL": value(row, f"{prefix}_short_rougeL", np.nan),
            "long_rougeL": value(row, f"{prefix}_long_rougeL", np.nan),
            "llm_judge_score_0_to_5": value(row, "llm_judge_score_0_to_5", np.nan),
            "llm_judge_verdict": value(row, "llm_judge_verdict"),
            "llm_judge_token_level_notes": value(row, "llm_judge_token_level_notes"),
            "llm_judge_json": value(row, "llm_judge_json"),
            "faithfulness_fused_score": value(row, "faithfulness_fused_score", np.nan),
            "estimated_lm_calls": value(row, "estimated_lm_calls", np.nan),
            "elapsed_seconds": value(row, "elapsed_seconds", np.nan),
            "elapsed_display": value(row, "elapsed_display"),
            "retrieved_self_hit": value(row, "retrieved_self_hit", np.nan),
            "trace_summary": value(row, "trace_summary"),
            "trace_json": value(row, "trace_json", value(row, "rlm_trace_json", "")),
            "p_summary": value(row, "p_summary"),
            "p_environment_json": value(row, "p_environment_json", value(row, "rlm_environment_json", "")),
            "evolved_prompt_score": value(row, "evolved_prompt_score", np.nan),
            "evolved_prompt_history_json": value(row, "evolved_prompt_history_json"),
            "recursive_call_count": value(row, "recursive_call_count", np.nan),
            "repl_step_count": value(row, "repl_step_count", np.nan),
            "repl_code_call_count": value(row, "repl_code_call_count", np.nan),
            "focused_repl_job_sequence": value(row, "focused_repl_job_sequence"),
            "question_understanding_call_count": value(row, "question_understanding_call_count", np.nan),
            "prompt_improvement_enabled": value(row, "prompt_improvement_enabled"),
            "prompt_improvement_call_count": value(row, "prompt_improvement_call_count", np.nan),
            "prompt_improvement_original_prompt": value(row, "prompt_improvement_original_prompt"),
            "prompt_improvement_updated_prompt": value(row, "prompt_improvement_updated_prompt"),
            "prompt_improvement_temperature": value(row, "prompt_improvement_temperature", np.nan),
            "prompt_improvement_max_tokens": value(row, "prompt_improvement_max_tokens", np.nan),
            "prompt_improvement_raw_output": value(row, "prompt_improvement_raw_output"),
            "prompt_improvement_messages_json": value(row, "prompt_improvement_messages_json"),
            "final_generation_call_count": value(row, "final_generation_call_count", np.nan),
            "question_understanding_json": value(row, "question_understanding_json"),
            "question_understanding_prompt": value(row, "question_understanding_prompt"),
            "question_understanding_raw": value(row, "question_understanding_raw"),
            "selected_evidence_handles": value(row, "selected_evidence_handles"),
            "snippet_classifications_json": value(row, "snippet_classifications_json"),
            "repl_history_json": value(row, "repl_history_json"),
            "paper_method_mapping": value(row, "paper_method_mapping"),
            "paper_principles_applied": value(row, "paper_principles_applied"),
        }
        rows.append(item)

    normalized = pd.DataFrame(rows)
    for col in NUMERIC_COLUMNS:
        if col in normalized.columns:
            normalized[col] = coerce_number(normalized[col])

    normalized["llm_judge_score_norm"] = normalized["llm_judge_score_0_to_5"] / 5.0
    no_judge_cols = ["short_f1", "long_f1", "short_rouge1", "long_rouge1"]
    with_judge_cols = no_judge_cols + ["llm_judge_score_norm"]
    normalized["quality_without_judge"] = normalized[no_judge_cols].mean(axis=1, skipna=True)
    normalized["quality_mean"] = normalized[with_judge_cols].mean(axis=1, skipna=True)
    normalized["judge_minus_quality"] = (
        normalized["llm_judge_score_norm"] - normalized["quality_without_judge"]
    )
    normalized["long_minus_short_f1"] = normalized["long_f1"] - normalized["short_f1"]
    return normalized


def read_all_traces() -> pd.DataFrame:
    frames = [
        normalize_trace(name, meta["path"], meta["prefix"])
        for name, meta in TRACE_FILES.items()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No trace CSVs were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    return combined.sort_values(["row_number", "model"]).reset_index(drop=True)


def add_best_model_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for metric in ["short_f1", "long_f1", "quality_mean", "balanced_fused_score", "llm_judge_score_norm"]:
        rank_col = f"{metric}_rank"
        best_col = f"is_best_{metric}"
        result[rank_col] = result.groupby("sample_id", observed=False)[metric].rank(
            method="min",
            ascending=False,
        )
        result[best_col] = (result[rank_col] == 1).astype(int)
    return result


def write_csvs(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    long_path = out_dir / "trace_model_question_comparison_long.csv"
    df.to_csv(long_path, index=False)
    paths["long"] = long_path

    summary = df.groupby("model", observed=False)[
        [col for col in NUMERIC_COLUMNS if col in df.columns]
    ].agg(["count", "mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    summary_path = out_dir / "trace_metric_summary_by_model.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    best_rows = []
    for sample_id, group in df.groupby("sample_id", observed=False):
        question = group["question"].iloc[0]
        best_rows.append(
            {
                "sample_id": sample_id,
                "question": question,
                "best_short_f1_model": group.loc[group["short_f1"].idxmax(), "model"],
                "best_short_f1": group["short_f1"].max(),
                "best_long_f1_model": group.loc[group["long_f1"].idxmax(), "model"],
                "best_long_f1": group["long_f1"].max(),
                "best_quality_model": group.loc[group["quality_mean"].idxmax(), "model"],
                "best_quality_mean": group["quality_mean"].max(),
                "best_balanced_fused_model": group.loc[group["balanced_fused_score"].idxmax(), "model"],
                "best_balanced_fused_score": group["balanced_fused_score"].max(),
                "mean_quality": group["quality_mean"].mean(),
                "mean_balanced_fused_score": group["balanced_fused_score"].mean(),
                "mean_elapsed_seconds": group["elapsed_seconds"].mean(),
            }
        )
    best_df = pd.DataFrame(best_rows)
    best_path = out_dir / "trace_best_model_by_question.csv"
    best_df.to_csv(best_path, index=False)
    paths["best"] = best_path

    difficulty = best_df.sort_values("mean_quality", ascending=True)
    difficulty_path = out_dir / "trace_question_difficulty.csv"
    difficulty.to_csv(difficulty_path, index=False)
    paths["difficulty"] = difficulty_path

    chart_ready = df.melt(
        id_vars=["model", "sample_id", "question"],
        value_vars=[col for col in [*QUALITY_METRICS, "balanced_fused_score_raw", "runtime_penalty", "call_penalty"] if col in df.columns],
        var_name="metric",
        value_name="value",
    )
    chart_ready_path = out_dir / "trace_chart_ready_long.csv"
    chart_ready.to_csv(chart_ready_path, index=False)
    paths["chart_ready"] = chart_ready_path

    return paths


def plot_grouped_quality(df: pd.DataFrame, out_dir: Path) -> Path:
    means = df.groupby("model", observed=False)[QUALITY_METRICS].mean(numeric_only=True)
    means = means.reindex(MODEL_ORDER)
    labels = [
        "Short F1",
        "Long F1",
        "Short BLEU",
        "Long BLEU",
        "Short ROUGE-1",
        "Long ROUGE-1",
        "LLM judge",
        "Quality mean",
    ]
    x = np.arange(len(QUALITY_METRICS))
    width = min(0.8 / max(len(means.index), 1), 0.22)

    fig, ax = plt.subplots(figsize=(14, 7))
    for offset, model in enumerate(means.index):
        position = x + (offset - (len(means.index) - 1) / 2) * width
        ax.bar(position, means.loc[model], width, label=model, color=COLORS.get(str(model)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Five-model trace comparison: F1, BLEU, ROUGE, and LLM-as-judge")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "trace_quality_metrics_by_model.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_judge(df: pd.DataFrame, out_dir: Path) -> Path:
    means = df.groupby("model", observed=False)["llm_judge_score_0_to_5"].mean()
    means = means.reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(means.index.astype(str), means.values, color=[COLORS.get(str(model)) for model in means.index])
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean LLM judge score (0-5)")
    ax.set_title("LLM-as-judge score by model")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = out_dir / "trace_llm_judge_score_by_model.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_runtime(df: pd.DataFrame, out_dir: Path) -> Path:
    means = df.groupby("model", observed=False)["elapsed_seconds"].mean()
    means = means.reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(means.index.astype(str), means.values, color=[COLORS.get(str(model)) for model in means.index])
    ax.set_ylabel("Mean seconds per question")
    ax.set_title("Runtime by model")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = out_dir / "trace_runtime_by_model.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_short_long_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    for model, group in df.groupby("model", observed=False):
        ax.scatter(
            group["short_f1"],
            group["long_f1"],
            s=42,
            alpha=0.72,
            label=model,
            color=COLORS.get(str(model)),
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Short-answer F1")
    ax.set_ylabel("Long-answer F1")
    ax.set_title("Short vs long answer F1 per question")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "trace_short_vs_long_f1_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_quality_runtime(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    means = df.groupby("model", observed=False)[["quality_mean", "elapsed_seconds"]].mean()
    means = means.reindex(MODEL_ORDER)
    for model, row in means.iterrows():
        ax.scatter(row["elapsed_seconds"], row["quality_mean"], s=220, color=COLORS.get(str(model)), label=model)
        ax.annotate(str(model), (row["elapsed_seconds"], row["quality_mean"]), xytext=(8, 5), textcoords="offset points")
    ax.set_xlabel("Mean seconds per question")
    ax.set_ylabel("Mean quality score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Quality vs runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "trace_quality_vs_runtime_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_balanced_fused_score(df: pd.DataFrame, out_dir: Path) -> Path:
    means = df.groupby("model", observed=False)["balanced_fused_score"].mean()
    means = means.reindex(MODEL_ORDER).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(means.index.astype(str), means.values, color=[COLORS.get(str(model)) for model in means.index])
    ax.set_ylabel("Mean balanced fused score")
    ax.set_title("Trace-level balanced fused score")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = out_dir / "trace_balanced_fused_score_by_model.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_fused_components(df: pd.DataFrame, out_dir: Path) -> Path:
    component_cols = [
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    labels = ["Raw fused", "Runtime penalty", "Call penalty", "Final fused"]
    means = df.groupby("model", observed=False)[component_cols].mean(numeric_only=True).reindex(MODEL_ORDER)
    x = np.arange(len(component_cols))
    width = min(0.8 / max(len(means.index), 1), 0.22)
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, model in enumerate(means.index):
        position = x + (offset - (len(means.index) - 1) / 2) * width
        ax.bar(position, means.loc[model], width, label=model, color=COLORS.get(str(model)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean score / penalty")
    ax.set_title("Trace-level balanced fused score components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "trace_balanced_fused_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    count_cols = ["is_best_short_f1", "is_best_long_f1", "is_best_quality_mean", "is_best_balanced_fused_score", "is_best_llm_judge_score_norm"]
    labels = ["Best short F1", "Best long F1", "Best quality", "Best fused", "Best judge"]
    counts = df.groupby("model", observed=False)[count_cols].sum().reindex(MODEL_ORDER)
    x = np.arange(len(labels))
    width = min(0.8 / max(len(counts.index), 1), 0.22)
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, model in enumerate(counts.index):
        position = x + (offset - (len(counts.index) - 1) / 2) * width
        ax.bar(position, counts.loc[model], width, label=model, color=COLORS.get(str(model)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Question count, ties included")
    ax.set_title("How often each model is best per question")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "trace_best_model_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "short_f1",
        "long_f1",
        "short_bleu",
        "long_bleu",
        "short_rouge1",
        "long_rouge1",
        "llm_judge_score_norm",
        "quality_mean",
        "balanced_fused_score",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "elapsed_seconds",
        "estimated_lm_calls",
    ]
    corr = df[[col for col in cols if col in df.columns]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Trace metric correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "trace_metric_correlation_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hardest_questions(df: pd.DataFrame, out_dir: Path) -> Path:
    hardest = (
        df.groupby(["sample_id", "question"], observed=False)["quality_mean"]
        .mean()
        .sort_values()
        .head(20)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = hardest["sample_id"].astype(str)
    ax.barh(labels, hardest["quality_mean"], color="#9467bd")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean quality across models")
    ax.set_title("Hardest questions across the five models")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "trace_hardest_questions_top20.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return [
        plot_grouped_quality(df, out_dir),
        plot_judge(df, out_dir),
        plot_runtime(df, out_dir),
        plot_short_long_scatter(df, out_dir),
        plot_quality_runtime(df, out_dir),
        plot_balanced_fused_score(df, out_dir),
        plot_fused_components(df, out_dir),
        plot_best_counts(df, out_dir),
        plot_correlation(df, out_dir),
        plot_hardest_questions(df, out_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Folder for cleaned CSVs and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_all_traces()
    df = add_balanced_fused_scores(df)
    df = add_best_model_flags(df)
    csv_paths = write_csvs(df, out_dir)
    plot_paths = write_plots(df, out_dir)

    print("\n=== CSV outputs ===")
    for path in csv_paths.values():
        print(path)

    print("\n=== Plot outputs ===")
    for path in plot_paths:
        print(path)

    print("\n=== Mean comparison ===")
    cols = [
        "short_f1",
        "long_f1",
        "short_bleu",
        "long_bleu",
        "short_rouge1",
        "long_rouge1",
        "llm_judge_score_0_to_5",
        "quality_mean",
        "balanced_fused_score",
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "elapsed_seconds",
    ]
    print(df.groupby("model", observed=False)[cols].mean(numeric_only=True).round(4))


if __name__ == "__main__":
    main()
