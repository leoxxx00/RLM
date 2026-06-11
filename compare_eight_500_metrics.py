#!/usr/bin/env python3
"""Compare the eight full 500-row NQ RAG/RLM outputs and make plots.

Run:
    cd /Users/htet/Desktop/Projects/X-RLM
    source venv/bin/activate
    python compare_eight_500_metrics.py

Outputs:
    nq_clean_monitor_exports/eight_models_500_metrics_formula7/
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
DEFAULT_OUT_DIR = PROJECT_DIR / "nq_clean_monitor_exports" / "eight_models_500_metrics_formula7"

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
    "GA + short REPL + RAG": {
        "path": PROJECT_DIR
        / "nq_ga_short_repl_rag_outputs"
        / "nq_ga_short_repl_rag_assessment_traces.csv",
        "prefix": "rlm",
    },
    "Focused RLM REPL + RAG": {
        "path": PROJECT_DIR
        / "nq_rlm_repl_focused_outputs"
        / "nq_rlm_repl_focused_assessment_traces.csv",
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
    "GA + short REPL + RAG": "#17becf",
    "Focused RLM REPL + RAG": "#1f77b4",
}

SUMMARY_METRICS = [
    "short_f1",
    "long_f1",
    "short_bleu",
    "long_bleu",
    "short_rouge1",
    "long_rouge1",
    "llm_judge_score_norm",
    "balanced_fused_score_raw",
    "runtime_penalty",
    "call_penalty",
    "balanced_fused_score",
    "elapsed_seconds",
]

PLOT_METRICS = [
    "short_f1",
    "long_f1",
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
    "llm_judge_score_norm": "LLM Judge",
    "balanced_fused_score": "Quality Fused",
    "elapsed_seconds": "Seconds / question",
}


def row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if name in row.index and pd.notna(row[name]):
        return row[name]
    return default


def metric(row: pd.Series, prefix: str, suffix: str) -> Any:
    return row_value(row, f"{prefix}_{suffix}", row_value(row, suffix))


def normalize_trace(model: str, path: Path, prefix: str, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trace file for {model}: {path}")

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
                "llm_judge_score_0_to_5": judge,
                "llm_judge_score_norm": judge / 5.0 if pd.notna(judge) else np.nan,
                "retrieved_self_hit": row_value(row, "retrieved_self_hit"),
                "estimated_lm_calls": row_value(row, "estimated_lm_calls"),
                "elapsed_seconds": row_value(row, "elapsed_seconds"),
                "answer_summary": row_value(row, "answer_summary", ""),
                "judge_summary": row_value(row, "judge_summary", ""),
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
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_fused_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["balanced_fused_score_raw"] = (
        0.15 * out["short_f1"].fillna(0)
        + 0.20 * out["long_f1"].fillna(0)
        + 0.10 * out["short_bleu"].fillna(0)
        + 0.10 * out["long_bleu"].fillna(0)
        + 0.10 * out["short_rouge1"].fillna(0)
        + 0.10 * out["long_rouge1"].fillna(0)
        + 0.20 * out["llm_judge_score_norm"].fillna(0)
    )

    runtime = out["elapsed_seconds"].fillna(out["elapsed_seconds"].median())
    calls = out["estimated_lm_calls"].fillna(out["estimated_lm_calls"].median())
    runtime_range = runtime.max() - runtime.min()
    calls_range = calls.max() - calls.min()
    out["runtime_penalty"] = 0.0 if runtime_range == 0 else 0.05 * (runtime - runtime.min()) / runtime_range
    out["call_penalty"] = 0.0 if calls_range == 0 else 0.05 * (calls - calls.min()) / calls_range
    # Quality score only. Runtime/call counts are reported separately, not subtracted.
    out["balanced_fused_score"] = out["balanced_fused_score_raw"]
    return out


def read_all(max_rows: int) -> pd.DataFrame:
    frames = [
        normalize_trace(model, meta["path"], meta["prefix"], max_rows)
        for model, meta in TRACE_FILES.items()
    ]
    return add_fused_score(pd.concat(frames, ignore_index=True))


def summarize(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
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
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    means = df.groupby("model", observed=True)[cols].mean(numeric_only=True).reset_index()
    counts = df.groupby("model", observed=True).size().reset_index(name="rows_used")
    summary = counts.merge(means, on="model", how="left")
    summary["target_rows"] = max_rows
    summary["is_full_target"] = summary["rows_used"] >= max_rows
    summary["balanced_fused_rank"] = summary["balanced_fused_score"].rank(ascending=False, method="min")
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def save_bar_chart(summary: pd.DataFrame, metrics: list[str], path: Path) -> None:
    labels = [str(model) for model in summary["model"]]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * width
    for offset, metric_name in zip(offsets, metrics):
        ax.bar(x + offset, summary[metric_name], width, label=LABELS[metric_name])

    ax.set_title("Eight 500-row model comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_fused_chart(summary: pd.DataFrame, path: Path) -> None:
    ranked = summary.sort_values("balanced_fused_score", ascending=True)
    colors = [COLORS[str(model)] for model in ranked["model"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(ranked["model"].astype(str), ranked["balanced_fused_score"], color=colors)
    ax.set_title("Quality fused score")
    ax.set_xlabel("Score")
    ax.set_xlim(0, max(0.65, ranked["balanced_fused_score"].max() + 0.05))
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(ranked["balanced_fused_score"]):
        ax.text(value + 0.006, idx, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_runtime_chart(summary: pd.DataFrame, path: Path) -> None:
    labels = [str(model) for model in summary["model"]]
    colors = [COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, summary["elapsed_seconds"], color=colors)
    ax.set_title("Average runtime per question")
    ax.set_ylabel("Seconds")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_short_vs_long_chart(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in summary.iterrows():
        model = str(row["model"])
        ax.scatter(row["short_f1"], row["long_f1"], s=120, color=COLORS[model], label=model)
        ax.annotate(model, (row["short_f1"], row["long_f1"]), xytext=(7, 4), textcoords="offset points", fontsize=8)

    ax.set_title("Short F1 vs Long F1")
    ax.set_xlabel("Short F1")
    ax.set_ylabel("Long F1")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_metric_heatmap(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        "short_f1",
        "long_f1",
        "short_bleu",
        "long_bleu",
        "short_rouge1",
        "long_rouge1",
        "llm_judge_score_norm",
        "balanced_fused_score",
    ]
    data = summary.set_index("model")[metrics].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(data.to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Metric heatmap")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([LABELS.get(metric, metric) for metric in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([str(model) for model in data.index])
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = data.iloc[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_rank_chart(summary: pd.DataFrame, path: Path) -> None:
    ranked = summary.sort_values("balanced_fused_score", ascending=False).reset_index(drop=True)
    labels = ranked["model"].astype(str).tolist()
    colors = [COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, ranked["balanced_fused_score"], color=colors)
    ax.set_title("Model ranking by quality fused score")
    ax.set_ylabel("Quality fused score")
    ax.set_ylim(0, max(0.65, ranked["balanced_fused_score"].max() + 0.06))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for rank, (bar, value) in enumerate(zip(bars, ranked["balanced_fused_score"]), start=1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"#{rank}\n{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_fused_vs_runtime_chart(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in summary.iterrows():
        model = str(row["model"])
        ax.scatter(
            row["elapsed_seconds"],
            row["balanced_fused_score"],
            s=180,
            color=COLORS[model],
            alpha=0.9,
        )
        ax.annotate(model, (row["elapsed_seconds"], row["balanced_fused_score"]), xytext=(8, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Quality fused score vs runtime")
    ax.set_xlabel("Average seconds per question")
    ax.set_ylabel("Quality fused score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_fused_components_chart(summary: pd.DataFrame, path: Path) -> None:
    component_cols = [
        "balanced_fused_score_raw",
        "runtime_penalty",
        "call_penalty",
        "balanced_fused_score",
    ]
    labels = ["Quality fused", "Runtime penalty", "Call penalty", "Final quality"]
    x = np.arange(len(component_cols))
    width = min(0.8 / max(len(summary), 1), 0.16)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(summary.iterrows()):
        model = str(row["model"])
        values = [row.get(col, np.nan) for col in component_cols]
        offset = (i - (len(summary) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model, color=COLORS[model])
    ax.set_title("Quality fused score components")
    ax.set_ylabel("Score / penalty")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_distribution_boxplot(df: pd.DataFrame, metric_name: str, path: Path) -> None:
    labels = MODEL_ORDER
    values = [df.loc[df["model"] == model, metric_name].dropna().to_numpy() for model in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.65)
    ax.set_title(f"Per-question distribution: {LABELS.get(metric_name, metric_name)}")
    ax.set_ylabel(LABELS.get(metric_name, metric_name))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_judge_distribution_chart(df: pd.DataFrame, path: Path) -> None:
    counts = (
        df.assign(judge_round=df["llm_judge_score_0_to_5"].round().clip(0, 5))
        .groupby(["model", "judge_round"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(MODEL_ORDER)
        .reindex(columns=[0, 1, 2, 3, 4, 5], fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    left = np.zeros(len(shares))
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    for score, color in zip(shares.columns, colors):
        values = shares[score].fillna(0).to_numpy()
        ax.barh(shares.index.astype(str), values, left=left, color=color, label=str(int(score)))
        left += values
    ax.set_title("LLM judge score distribution")
    ax.set_xlabel("Share of questions")
    ax.set_xlim(0, 1)
    ax.legend(title="Judge score", ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def save_win_count_chart(df: pd.DataFrame, path: Path) -> None:
    metric_cols = ["short_f1", "long_f1", "short_bleu", "long_bleu", "short_rouge1", "long_rouge1", "balanced_fused_score"]
    wins = {model: 0 for model in MODEL_ORDER}
    usable = df.dropna(subset=["sample_id"])
    for _, group in usable.groupby("sample_id", observed=True):
        for metric_name in metric_cols:
            if metric_name not in group.columns or group[metric_name].dropna().empty:
                continue
            best = group[metric_name].max()
            winners = group.loc[group[metric_name] == best, "model"].astype(str).tolist()
            for winner in winners:
                wins[winner] += 1 / max(len(winners), 1)

    labels = MODEL_ORDER
    values = [wins[label] for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, color=[COLORS[label] for label in labels])
    ax.set_title("Per-question metric win count")
    ax.set_ylabel("Weighted wins across metrics/questions")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "eight_models_500_trace_rows_long.csv"
    summary_path = out_dir / "eight_models_500_summary.csv"
    chart_ready_path = out_dir / "eight_models_500_chart_ready_long.csv"

    df.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.melt(
        id_vars=["model", "rows_used", "target_rows", "is_full_target"],
        value_vars=SUMMARY_METRICS,
        var_name="metric",
        value_name="value",
    ).to_csv(chart_ready_path, index=False)

    print(f"[write] {rows_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {chart_ready_path}")

    save_bar_chart(summary, PLOT_METRICS, out_dir / "eight_models_500_quality_metrics.png")
    save_fused_chart(summary, out_dir / "eight_models_500_balanced_fused_score.png")
    save_runtime_chart(summary, out_dir / "eight_models_500_runtime.png")
    save_short_vs_long_chart(summary, out_dir / "eight_models_500_short_vs_long_f1.png")
    save_metric_heatmap(summary, out_dir / "eight_models_500_metric_heatmap.png")
    save_rank_chart(summary, out_dir / "eight_models_500_ranked_fused_score.png")
    save_fused_vs_runtime_chart(summary, out_dir / "eight_models_500_fused_vs_runtime.png")
    save_fused_components_chart(summary, out_dir / "eight_models_500_fused_components.png")
    save_distribution_boxplot(df, "short_f1", out_dir / "eight_models_500_short_f1_distribution.png")
    save_distribution_boxplot(df, "long_f1", out_dir / "eight_models_500_long_f1_distribution.png")
    save_distribution_boxplot(df, "balanced_fused_score", out_dir / "eight_models_500_fused_distribution.png")
    save_judge_distribution_chart(df, out_dir / "eight_models_500_judge_distribution.png")
    save_win_count_chart(df, out_dir / "eight_models_500_per_question_win_count.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_all(args.max_rows)
    summary = summarize(df, args.max_rows)
    write_outputs(df, summary, args.out_dir)

    print("\n=== Eight full 500-row model summary ===")
    print(
        summary[
            [
                "model",
                "rows_used",
                "is_full_target",
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
