#!/usr/bin/env python3
"""Compare WikiLingua model runs and generate CSV/PNG outputs."""

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
OUT_ROOT = PROJECT_DIR / "wikilingua_outputs"
DEFAULT_OUT_DIR = PROJECT_DIR / "wikilingua_clean_monitor_exports" / "all_wikilingua_models"

TRACE_FILES = {
    "WikiLingua Llama 3 8B + RAG": OUT_ROOT / "wikilingua_rag" / "wikilingua_rag_assessment_traces.csv",
    "WikiLingua GA + RAG": OUT_ROOT / "wikilingua_ga_rag" / "wikilingua_ga_rag_assessment_traces.csv",
    "WikiLingua ES + RAG": OUT_ROOT / "wikilingua_es_rag" / "wikilingua_es_rag_assessment_traces.csv",
    "WikiLingua RLM REPL 4-step + RAG": OUT_ROOT / "wikilingua_rlm_repl4_rag" / "wikilingua_rlm_repl4_rag_assessment_traces.csv",
    "WikiLingua RLM REPL 8-step + RAG": OUT_ROOT / "wikilingua_rlm_repl8_rag" / "wikilingua_rlm_repl8_rag_assessment_traces.csv",
    "WikiLingua Adaptive-P short REPL + RAG": OUT_ROOT
    / "wikilingua_adaptive_p_short_repl_rag"
    / "wikilingua_adaptive_p_short_repl_rag_assessment_traces.csv",
}

MODEL_ORDER = list(TRACE_FILES)
COLORS = {
    "WikiLingua Llama 3 8B + RAG": "#2ca02c",
    "WikiLingua GA + RAG": "#ff7f0e",
    "WikiLingua ES + RAG": "#d62728",
    "WikiLingua RLM REPL 4-step + RAG": "#9467bd",
    "WikiLingua RLM REPL 8-step + RAG": "#8c564b",
    "WikiLingua Adaptive-P short REPL + RAG": "#bcbd22",
}

METRICS = [
    "summary_f1",
    "summary_bleu",
    "summary_rouge1",
    "summary_rouge2",
    "summary_rougeL",
    "llm_judge_score_norm",
    "quality_fused_score",
]

LABELS = {
    "summary_f1": "Summary F1",
    "summary_bleu": "BLEU",
    "summary_rouge1": "ROUGE-1",
    "summary_rouge2": "ROUGE-2",
    "summary_rougeL": "ROUGE-L",
    "llm_judge_score_norm": "LLM Judge",
    "quality_fused_score": "Quality Fused",
    "retrieved_self_hit": "Retrieval Self Hit",
    "elapsed_seconds": "Seconds",
}


def value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    return row[name] if name in row.index and pd.notna(row[name]) else default


def prefix_for(model: str) -> str:
    return "rlm" if "RLM" in model or "Adaptive-P" in model else "rag"


def normalize_trace(model: str, path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        print(f"[missing] {model}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "row_number" in df.columns:
        df = df.sort_values("row_number")
    df = df.head(max_rows)
    print(f"[read] {model}: rows_used={len(df)} path={path}")
    prefix = prefix_for(model)
    rows = []
    for _, row in df.iterrows():
        judge = pd.to_numeric(value(row, "llm_judge_score_0_to_5"), errors="coerce")
        rows.append(
            {
                "model": model,
                "row_number": value(row, "row_number", ""),
                "sample_id": value(row, "sample_id", ""),
                "target_lang": value(row, "target_lang", ""),
                "target_language": value(row, "target_language", ""),
                "summary_f1": value(row, f"{prefix}_summary_token_f1"),
                "summary_bleu": value(row, f"{prefix}_summary_bleu"),
                "summary_rouge1": value(row, f"{prefix}_summary_rouge1"),
                "summary_rouge2": value(row, f"{prefix}_summary_rouge2"),
                "summary_rougeL": value(row, f"{prefix}_summary_rougeL"),
                "llm_judge_score_0_to_5": judge,
                "llm_judge_score_norm": judge / 5.0 if pd.notna(judge) else np.nan,
                "quality_fused_score": value(row, "quality_fused_score"),
                "retrieved_self_hit": value(row, "retrieved_self_hit"),
                "estimated_lm_calls": value(row, "estimated_lm_calls"),
                "elapsed_seconds": value(row, "elapsed_seconds"),
            }
        )
    out = pd.DataFrame(rows)
    for col in METRICS + ["retrieved_self_hit", "estimated_lm_calls", "elapsed_seconds"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["quality_fused_score"] = (
        0.25 * out["summary_f1"].fillna(0)
        + 0.15 * out["summary_bleu"].fillna(0)
        + 0.20 * out["summary_rouge1"].fillna(0)
        + 0.15 * out["summary_rougeL"].fillna(0)
        + 0.25 * out["llm_judge_score_norm"].fillna(0)
    )
    return out


def read_all(max_rows: int) -> pd.DataFrame:
    frames = [normalize_trace(model, path, max_rows) for model, path in TRACE_FILES.items()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No WikiLingua trace files found.")
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame, max_rows: int, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "wikilingua_all_models_trace_rows_long.csv", index=False)
    means = df.groupby("model", observed=True)[
        METRICS + ["retrieved_self_hit", "estimated_lm_calls", "elapsed_seconds"]
    ].mean(numeric_only=True)
    counts = df.groupby("model", observed=True).size().rename("rows_used")
    summary = counts.to_frame().join(means).reset_index()
    summary["target_rows"] = max_rows
    summary["is_full_target"] = summary["rows_used"] >= max_rows
    summary["quality_fused_rank"] = summary["quality_fused_score"].rank(ascending=False, method="min")
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")
    summary.to_csv(out_dir / "wikilingua_all_models_summary.csv", index=False)
    return summary


def ordered(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["model"] = data["model"].astype(str)
    return data.set_index("model").reindex(MODEL_ORDER).dropna(how="all")


def bar_plot(summary: pd.DataFrame, metric: str, title: str, path: Path, ylim: tuple[float, float] | None = None) -> None:
    data = ordered(summary).sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index.astype(str), data[metric], color=[COLORS.get(str(m)) for m in data.index])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    x = np.arange(len(METRICS))
    width = min(0.13, 0.82 / max(1, len(data)))
    fig, ax = plt.subplots(figsize=(15, 6))
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
    ax.set_title("WikiLingua model quality metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "wikilingua_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    heat = data[METRICS].astype(float)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([LABELS[m] for m in METRICS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(str))
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    ax.set_title("WikiLingua metric heatmap")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = out_dir / "wikilingua_metric_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["quality_fused_score", "summary_f1", "summary_rouge1", "llm_judge_score_norm"]:
        values = [df.loc[df["model"] == model, metric].dropna().values for model in MODEL_ORDER]
        labels = [model for model, vals in zip(MODEL_ORDER, values) if len(vals)]
        values = [vals for vals in values if len(vals)]
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(13, 5))
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS.get(label, "#888888"))
            patch.set_alpha(0.55)
        ax.set_title(f"WikiLingua distribution: {LABELS.get(metric, metric)}")
        ax.set_ylabel(LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        path = out_dir / f"wikilingua_{metric}_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[write] {path}")


def plot_scatter(summary: pd.DataFrame, out_dir: Path) -> None:
    data = ordered(summary)
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, row in data.iterrows():
        ax.scatter(row["elapsed_seconds"], row["quality_fused_score"], s=180, color=COLORS.get(str(model)))
        ax.annotate(str(model), (row["elapsed_seconds"], row["quality_fused_score"]), xytext=(7, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean seconds per document")
    ax.set_ylabel("Quality fused score")
    ax.set_title("WikiLingua quality vs runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "wikilingua_fused_vs_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def plot_per_question_wins(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="sample_id", columns="model", values="quality_fused_score", aggfunc="mean")
    if pivot.empty:
        return
    counts = pivot.idxmax(axis=1).value_counts().reindex(MODEL_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index.astype(str), counts.values, color=[COLORS.get(str(m)) for m in counts.index])
    ax.set_title("WikiLingua per-document fused-score win count")
    ax.set_ylabel("Document wins")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "wikilingua_per_document_win_count.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-lang", choices=["all", "nl", "fr", "de", "es"], default="all")
    args = parser.parse_args()

    df = read_all(args.max_rows)
    summary = summary_table(df, args.max_rows, args.out_dir)
    print("\n=== All WikiLingua models summary ===")
    print(summary.to_string(index=False))

    plot_quality(summary, args.out_dir)
    plot_heatmap(summary, args.out_dir)
    plot_distributions(df, args.out_dir)
    plot_scatter(summary, args.out_dir)
    plot_per_question_wins(df, args.out_dir)
    bar_plot(summary, "quality_fused_score", "WikiLingua quality fused score", args.out_dir / "wikilingua_quality_fused_score.png", (0, 1))
    bar_plot(summary, "retrieved_self_hit", "WikiLingua retrieval self-hit rate", args.out_dir / "wikilingua_retrieval_self_hit.png", (0, 1))
    bar_plot(summary, "elapsed_seconds", "WikiLingua average runtime per document", args.out_dir / "wikilingua_runtime.png")


if __name__ == "__main__":
    main()
