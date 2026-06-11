#!/usr/bin/env python3
"""Compare all available Natural Questions RAG/RLM model traces on 500 rows.

The fused score is quality-only:
0.15 short F1 + 0.20 long F1 + 0.10 short BLEU + 0.10 long BLEU
+ 0.10 short ROUGE-1 + 0.10 long ROUGE-1 + 0.20 normalized LLM judge.
Runtime is plotted separately and is not part of the quality score.
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
DEFAULT_OUT_DIR = PROJECT_DIR / "nq_clean_monitor_exports" / "all_models_500_checkpoint"

TRACE_FILES = {
    "Llama 3 8B + RAG": {
        "path": PROJECT_DIR / "nq_langgraph_rag_outputs" / "nq_rag_llama_assessment_traces.csv",
        "prefix": "rag",
    },
    "GA + RAG": {
        "path": PROJECT_DIR / "nq_langgraph_rag_outputs" / "nq_genetic_rag_llama_assessment_traces.csv",
        "prefix": "rag",
    },
    "ES + RAG": {
        "path": PROJECT_DIR
        / "nq_evolution_strategy_rag_outputs"
        / "nq_evolution_strategy_rag_llama_assessment_traces.csv",
        "prefix": "rag",
    },
    "RLM REPL + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl_rag_outputs" / "nq_rlm_repl_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "RLM REPL 8-step checkpoint + RAG": {
        "path": PROJECT_DIR
        / "nq_rlm_repl8_checkpoint_rag_outputs"
        / "nq_rlm_repl8_checkpoint_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "RLM REPL 12-step + RAG": {
        "path": PROJECT_DIR
        / "nq_rlm_repl12_500_rag_outputs"
        / "nq_rlm_repl12_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "Focused RLM REPL + RAG": {
        "path": PROJECT_DIR / "nq_rlm_repl_focused_outputs" / "nq_rlm_repl_focused_assessment_traces.csv",
        "prefix": "rlm",
    },
    "GA + short REPL + RAG": {
        "path": PROJECT_DIR / "nq_ga_short_repl_rag_outputs" / "nq_ga_short_repl_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "Adaptive-P short REPL + RAG": {
        "path": PROJECT_DIR
        / "nq_adaptive_p_short_repl_rag_outputs"
        / "nq_adaptive_p_short_repl_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
}

MODEL_ORDER = list(TRACE_FILES)
COLORS = {
    "Llama 3 8B + RAG": "#2ca02c",
    "GA + RAG": "#ff7f0e",
    "ES + RAG": "#d62728",
    "RLM REPL + RAG": "#9467bd",
    "RLM REPL 8-step checkpoint + RAG": "#8c564b",
    "RLM REPL 12-step + RAG": "#e377c2",
    "Focused RLM REPL + RAG": "#1f77b4",
    "GA + short REPL + RAG": "#17becf",
    "Adaptive-P short REPL + RAG": "#bcbd22",
}

METRICS = [
    "short_f1",
    "long_f1",
    "short_bleu",
    "long_bleu",
    "short_rouge1",
    "long_rouge1",
    "llm_judge_score_norm",
    "quality_fused_score",
]

LABELS = {
    "short_f1": "Short F1",
    "long_f1": "Long F1",
    "short_bleu": "Short BLEU",
    "long_bleu": "Long BLEU",
    "short_rouge1": "Short ROUGE-1",
    "long_rouge1": "Long ROUGE-1",
    "llm_judge_score_norm": "LLM Judge\n0-1",
    "quality_fused_score": "Quality\nFused",
    "elapsed_seconds": "Seconds",
    "estimated_lm_calls": "LM Calls",
}


def row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if name in row.index and pd.notna(row[name]):
        return row[name]
    return default


def metric(row: pd.Series, prefix: str, suffix: str) -> Any:
    return row_value(row, f"{prefix}_{suffix}", row_value(row, suffix))


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
        judge = pd.to_numeric(row_value(row, "llm_judge_score_0_to_5"), errors="coerce")
        rows.append(
            {
                "model": model,
                "source_file": str(path),
                "row_number": row_value(row, "row_number", ""),
                "sample_id": row_value(row, "sample_id", ""),
                "question": row_value(row, "question", ""),
                "short_f1": metric(row, prefix, "short_token_f1"),
                "long_f1": metric(row, prefix, "long_token_f1"),
                "short_bleu": metric(row, prefix, "short_bleu"),
                "long_bleu": metric(row, prefix, "long_bleu"),
                "short_rouge1": metric(row, prefix, "short_rouge1"),
                "long_rouge1": metric(row, prefix, "long_rouge1"),
                "short_rouge2": metric(row, prefix, "short_rouge2"),
                "long_rouge2": metric(row, prefix, "long_rouge2"),
                "short_rougeL": metric(row, prefix, "short_rougeL"),
                "long_rougeL": metric(row, prefix, "long_rougeL"),
                "llm_judge_score_0_to_5": judge,
                "llm_judge_score_norm": judge / 5.0 if pd.notna(judge) else np.nan,
                "retrieved_self_hit": row_value(row, "retrieved_self_hit"),
                "estimated_lm_calls": row_value(row, "estimated_lm_calls"),
                "recursive_call_count": row_value(row, "recursive_call_count"),
                "repl_step_count": row_value(row, "repl_step_count"),
                "elapsed_seconds": row_value(row, "elapsed_seconds"),
                "answer_summary": row_value(row, "answer_summary", ""),
                "judge_summary": row_value(row, "judge_summary", ""),
                "trace_summary": row_value(row, "trace_summary", ""),
            }
        )
    out = pd.DataFrame(rows)
    for col in [
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
        "recursive_call_count",
        "repl_step_count",
        "elapsed_seconds",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_fused(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quality_fused_score"] = (
        0.15 * out["short_f1"].fillna(0)
        + 0.20 * out["long_f1"].fillna(0)
        + 0.10 * out["short_bleu"].fillna(0)
        + 0.10 * out["long_bleu"].fillna(0)
        + 0.10 * out["short_rouge1"].fillna(0)
        + 0.10 * out["long_rouge1"].fillna(0)
        + 0.20 * out["llm_judge_score_norm"].fillna(0)
    )
    out["balanced_fused_score"] = out["quality_fused_score"]
    return out


def read_all(max_rows: int) -> pd.DataFrame:
    frames = [
        normalize_trace(model, meta["path"], meta["prefix"], max_rows)
        for model, meta in TRACE_FILES.items()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No trace files found.")
    df = pd.concat(frames, ignore_index=True)
    return add_fused(df)


def write_csvs(df: pd.DataFrame, out_dir: Path, max_rows: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "all_models_500_checkpoint_trace_rows_long.csv"
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
        "recursive_call_count",
        "repl_step_count",
        "elapsed_seconds",
        "quality_fused_score",
        "balanced_fused_score",
    ]
    means = df.groupby("model", observed=True)[cols].mean(numeric_only=True).reset_index()
    counts = df.groupby("model", observed=True).size().reset_index(name="rows_used")
    summary = counts.merge(means, on="model", how="left")
    summary["model"] = summary["model"].astype(str)
    summary["target_rows"] = max_rows
    summary["is_full_target"] = summary["rows_used"] >= max_rows
    summary["quality_fused_rank"] = summary["quality_fused_score"].rank(
        ascending=False,
        method="min",
    )
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")
    summary_path = out_dir / "all_models_500_checkpoint_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")
    chart = summary.melt(
        id_vars=["model", "rows_used", "target_rows", "is_full_target"],
        value_vars=[m for m in METRICS + ["elapsed_seconds", "estimated_lm_calls"] if m in summary.columns],
        var_name="metric",
        value_name="value",
    )
    chart["metric_label"] = chart["metric"].map(LABELS).fillna(chart["metric"])
    chart_path = out_dir / "all_models_500_checkpoint_chart_ready_long.csv"
    chart.to_csv(chart_path, index=False)
    print(f"[write] {chart_path}")
    return summary


def ordered(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["model"] = data["model"].astype(str)
    return data.set_index("model").reindex(MODEL_ORDER).dropna(how="all")


def plot_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    x = np.arange(len(METRICS))
    width = min(0.12, 0.82 / max(1, len(data)))
    fig, ax = plt.subplots(figsize=(16, 7))
    for offset, model in enumerate(data.index):
        ax.bar(
            x + (offset - (len(data) - 1) / 2) * width,
            data.loc[model, METRICS],
            width=width,
            label=str(model),
            color=COLORS.get(str(model)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METRICS], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("All available models, 500-question comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_runtime(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index.astype(str), data["elapsed_seconds"], color=[COLORS.get(str(m)) for m in data.index])
    ax.set_ylabel("Mean seconds per question")
    ax.set_title("Runtime, 500-question comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary).sort_values("quality_fused_score", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(data.index.astype(str), data["quality_fused_score"], color=[COLORS.get(str(m)) for m in data.index])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Quality fused score")
    ax.set_title("Quality fused ranking, 500-question comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_quality_fused_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_short_long(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    fig, ax = plt.subplots(figsize=(8, 7))
    for model, row in data.iterrows():
        ax.scatter(row["short_f1"], row["long_f1"], s=190, color=COLORS.get(str(model)), label=str(model))
        ax.annotate(str(model), (row["short_f1"], row["long_f1"]), xytext=(7, 4), textcoords="offset points", fontsize=8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Short-answer F1")
    ax.set_ylabel("Long-answer F1")
    ax.set_title("Short vs long F1, 500-question comparison")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_short_vs_long_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_metric_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    heat = data[METRICS].astype(float)
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([LABELS[m].replace("\n", " ") for m in METRICS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(str))
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = heat.iloc[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontsize=8,
            )
    ax.set_title("Metric heatmap")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_metric_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    data = [df.loc[df["model"] == model, "quality_fused_score"].dropna().values for model in MODEL_ORDER]
    labels = [model for model, values in zip(MODEL_ORDER, data) if len(values)]
    values = [values for values in data if len(values)]
    fig, ax = plt.subplots(figsize=(14, 6))
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS.get(label, "#888888"))
        patch.set_alpha(0.55)
    ax.set_ylabel("Per-question quality fused score")
    ax.set_title("Per-question fused score distribution")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_fused_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_metric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    for metric_name in ["short_f1", "long_f1", "llm_judge_score_norm"]:
        data = [df.loc[df["model"] == model, metric_name].dropna().values for model in MODEL_ORDER]
        labels = [model for model, values in zip(MODEL_ORDER, data) if len(values)]
        values = [values for values in data if len(values)]
        fig, ax = plt.subplots(figsize=(14, 6))
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS.get(label, "#888888"))
            patch.set_alpha(0.55)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(LABELS.get(metric_name, metric_name).replace("\n", " "))
        ax.set_title(f"Per-question distribution: {LABELS.get(metric_name, metric_name).replace(chr(10), ' ')}")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        path = out_dir / f"all_models_500_checkpoint_{metric_name}_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[write] {path}")


def plot_runtime_vs_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    fig, ax = plt.subplots(figsize=(10, 7))
    for model, row in data.iterrows():
        ax.scatter(
            row["elapsed_seconds"],
            row["quality_fused_score"],
            s=220,
            color=COLORS.get(str(model)),
            alpha=0.85,
        )
        ax.annotate(str(model), (row["elapsed_seconds"], row["quality_fused_score"]), xytext=(8, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean seconds per question")
    ax.set_ylabel("Quality fused score")
    ax.set_title("Quality fused score vs runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_fused_vs_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_quality_per_second(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary).copy()
    data["quality_per_second"] = data["quality_fused_score"] / data["elapsed_seconds"].replace(0, np.nan)
    data = data.sort_values("quality_per_second", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index.astype(str), data["quality_per_second"], color=[COLORS.get(str(m)) for m in data.index])
    ax.set_ylabel("Quality fused score / second")
    ax.set_title("Quality efficiency, higher is better")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_quality_per_second.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_fused_components(summary: pd.DataFrame, out_dir: Path) -> None:
    component_weights = {
        "short_f1": 0.15,
        "long_f1": 0.20,
        "short_bleu": 0.10,
        "long_bleu": 0.10,
        "short_rouge1": 0.10,
        "long_rouge1": 0.10,
        "llm_judge_score_norm": 0.20,
    }
    data = ordered(summary).sort_values("quality_fused_score", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    for metric_name, weight in component_weights.items():
        values = data[metric_name].fillna(0).values * weight
        ax.bar(x, values, bottom=bottom, label=f"{LABELS[metric_name].replace(chr(10), ' ')} x {weight:.2f}")
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(data.index.astype(str), rotation=25, ha="right")
    ax.set_ylabel("Weighted contribution")
    ax.set_title("Quality fused score composition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_fused_components_stacked.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_delta_vs_baseline(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary).copy()
    baseline = "Llama 3 8B + RAG"
    if baseline not in data.index:
        return
    base_score = float(data.loc[baseline, "quality_fused_score"])
    data["delta_vs_llama_rag"] = data["quality_fused_score"] - base_score
    data = data.sort_values("delta_vs_llama_rag", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in data["delta_vs_llama_rag"]]
    ax.bar(data.index.astype(str), data["delta_vs_llama_rag"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Quality fused delta")
    ax.set_title("Delta vs Llama 3 8B + RAG baseline")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_delta_vs_llama_rag.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_per_question_win_count(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(
        index="sample_id",
        columns="model",
        values="quality_fused_score",
        aggfunc="mean",
    )
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return
    winners = pivot.idxmax(axis=1)
    counts = winners.value_counts().reindex(MODEL_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index.astype(str), counts.values, color=[COLORS.get(str(m)) for m in counts.index])
    ax.set_ylabel("Questions won")
    ax.set_title("Per-question fused score win count")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_per_question_win_count.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_rank_by_question(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(
        index="sample_id",
        columns="model",
        values="quality_fused_score",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean().reindex(MODEL_ORDER).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(avg_rank.index.astype(str), avg_rank.values, color=[COLORS.get(str(m)) for m in avg_rank.index])
    ax.invert_yaxis()
    ax.set_ylabel("Average per-question rank, lower is better")
    ax.set_title("Average rank across questions")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_average_question_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_self_hit(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary).dropna(subset=["retrieved_self_hit"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index.astype(str), data["retrieved_self_hit"], color=[COLORS.get(str(m)) for m in data.index])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean retrieved_self_hit")
    ax.set_title("Retrieval self-hit rate")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "all_models_500_checkpoint_retrieval_self_hit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_all(args.max_rows)
    summary = write_csvs(df, args.out_dir, args.max_rows)
    plot_quality(summary, args.out_dir)
    plot_runtime(summary, args.out_dir)
    plot_fused(summary, args.out_dir)
    plot_short_long(summary, args.out_dir)
    plot_metric_heatmap(summary, args.out_dir)
    plot_fused_distribution(df, args.out_dir)
    plot_metric_distributions(df, args.out_dir)
    plot_runtime_vs_quality(summary, args.out_dir)
    plot_quality_per_second(summary, args.out_dir)
    plot_fused_components(summary, args.out_dir)
    plot_delta_vs_baseline(summary, args.out_dir)
    plot_per_question_win_count(df, args.out_dir)
    plot_rank_by_question(df, args.out_dir)
    plot_self_hit(summary, args.out_dir)
    print("\n=== All models 500 checkpoint summary ===")
    print(
        summary[
            [
                "model",
                "rows_used",
                "is_full_target",
                "short_f1",
                "long_f1",
                "llm_judge_score_norm",
                "quality_fused_score",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
