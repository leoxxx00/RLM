#!/usr/bin/env python3
"""WikiLingua multilingual summarization evaluator with multiple RAG/RLM settings.

Modes:
- rag: one-pass multilingual summarization with retrieved context
- ga: prompt-improved RAG with a genetic-style instruction
- es: prompt-improved RAG with an exploration-style instruction
- repl4: short REPL/RLM over retrieved document snippets
- repl8: longer REPL/RLM over retrieved document snippets
- adaptive_p: closed-loop Adaptive-P REPL-RAG with metric feedback

Default model:
Meta-Llama-3-8B-Instruct.Q4_K_M.gguf, a 4-bit quantized Llama 3 8B GGUF,
loaded through the shared local llama.cpp loader unless --model-path is given.

The fused score is quality-only:
0.25 summary F1 + 0.15 BLEU + 0.20 ROUGE-1 + 0.15 ROUGE-L
+ 0.25 normalized LLM judge.
Runtime is stored separately and is not part of the score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from llama_cpp import Llama

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from qaconv_eval import (
    HashEmbeddings,
    automatic_metrics,
    compact_json,
    load_llm,
    parse_response,
    safe_str,
)


PROJECT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM")
DEFAULT_DATA = PROJECT_DIR / "Data" / "WikiLingua" / "train" / "train.parquet"
DEFAULT_OUT_ROOT = PROJECT_DIR / "wikilingua_outputs"
DEFAULT_QUANTIZED_MODEL = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

MODE_META = {
    "rag": ("WikiLingua Llama 3 8B + RAG", "wikilingua_rag"),
    "ga": ("WikiLingua GA + RAG", "wikilingua_ga_rag"),
    "es": ("WikiLingua ES + RAG", "wikilingua_es_rag"),
    "repl4": ("WikiLingua RLM REPL 4-step + RAG", "wikilingua_rlm_repl4_rag"),
    "repl8": ("WikiLingua RLM REPL 8-step + RAG", "wikilingua_rlm_repl8_rag"),
    "adaptive_p": ("WikiLingua Adaptive-P short REPL + RAG", "wikilingua_adaptive_p_short_repl_rag"),
}

LANGUAGE_COLUMNS = {
    "nl": ("Dutch", "summary_nl"),
    "fr": ("French", "summary_fr"),
    "de": ("German", "summary_de"),
    "es": ("Spanish", "summary_es"),
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "her", "his", "i", "in", "is", "it", "its", "me", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they", "this",
    "to", "we", "will", "with", "you",
}


@dataclass
class EvalConfig:
    input_path: Path
    out_dir: Path
    mode: str
    max_rows: int | None
    top_k: int
    retrieval_fetch_k: int
    p_loop_count: int
    repl_max_steps: int
    max_tokens: int
    subcall_max_tokens: int
    judge_max_tokens: int
    temperature: float
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    model_path: Path | None
    rebuild_index: bool
    no_resume: bool
    skip_llm_judge: bool
    dry_run: bool
    checkpoint_every: int
    target_lang: str


def terms(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", safe_str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def compact_text(text: str, limit: int = 4800) -> str:
    text = re.sub(r"\s+", " ", safe_str(text)).strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def load_rows(path: Path, max_rows: int | None) -> pd.DataFrame:
    print(f"[setup] Loading WikiLingua: {path}", flush=True)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    print(f"[setup] Loaded {len(df)} rows and {len(df.columns)} columns", flush=True)
    return df


def language_reference(row: pd.Series, target_lang: str) -> tuple[str, str]:
    if target_lang == "all":
        parts = []
        for code, (name, col) in LANGUAGE_COLUMNS.items():
            text = safe_str(row.get(col))
            if text:
                parts.append(f"{name}: {text}")
        return "all", " | ".join(parts)
    name, col = LANGUAGE_COLUMNS[target_lang]
    return name, safe_str(row.get(col))


def normalize_dataset(df: pd.DataFrame, target_lang: str, max_rows: int | None) -> pd.DataFrame:
    required_cols = [col for _, col in LANGUAGE_COLUMNS.values()] if target_lang == "all" else [LANGUAGE_COLUMNS[target_lang][1]]
    before = len(df)
    df = df.dropna(subset=["document", *required_cols]).copy()
    for col in ["document", *required_cols]:
        df = df[df[col].astype(str).str.strip().ne("")]
    after_required = len(df)
    if max_rows:
        df = df.head(max_rows)
    print(
        f"[setup] Using {len(df)} rows after requiring target summaries "
        f"({target_lang}); removed {before - after_required} rows with missing target text",
        flush=True,
    )
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        sample_id = safe_str(row.get("sample_id")) or f"WikiLingua_{idx}"
        document = safe_str(row.get("document"))
        target_name, gt_summary = language_reference(row, target_lang)
        baseline_summary = ""
        rows.append(
            {
                "row_number": idx + 1,
                "sample_id": sample_id,
                "question": (
                    "Summarize the English WikiLingua document into Dutch, French, German, and Spanish."
                    if target_lang == "all"
                    else f"Summarize the English WikiLingua document in {target_name}."
                ),
                "transcript": document,
                "gt_summary": gt_summary,
                "references": [gt_summary] if gt_summary else [],
                "baseline_summary": baseline_summary,
                "target_lang": target_lang,
                "target_language": target_name,
            }
        )
    return pd.DataFrame(rows)


def build_vectorstore(df: pd.DataFrame, cfg: EvalConfig) -> tuple[Chroma, bool]:
    persist_dir = cfg.out_dir / f"chroma_wikilingua_{cfg.target_lang}"
    rebuilt = False
    if cfg.rebuild_index and persist_dir.exists():
        shutil.rmtree(persist_dir)
    embeddings = HashEmbeddings()
    if persist_dir.exists():
        print(f"[vector-db] Reusing WikiLingua index: {persist_dir}", flush=True)
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings), rebuilt
    print(f"[vector-db] Creating WikiLingua index with {len(df)} documents", flush=True)
    docs = [
        Document(
            page_content=row["transcript"],
            metadata={
                "sample_id": row["sample_id"],
                "row_number": int(row["row_number"]),
            },
        )
        for row in df.to_dict("records")
    ]
    store = Chroma.from_documents(docs, embeddings, persist_directory=str(persist_dir))
    rebuilt = True
    return store, rebuilt


def lexical_score(query: str, text: str) -> float:
    q = set(terms(query))
    return len(q & set(terms(text))) / max(1, len(q))


def retrieve_docs(store: Chroma, row: Dict[str, Any], cfg: EvalConfig) -> List[Document]:
    query = row["transcript"][:1200]
    candidates = store.similarity_search(query, k=max(cfg.top_k, cfg.retrieval_fetch_k))
    ranked = []
    seen = set()
    for rank, doc in enumerate(candidates):
        sid = doc.metadata.get("sample_id", "")
        if sid in seen:
            continue
        seen.add(sid)
        score = lexical_score(query, doc.page_content)
        if sid == row["sample_id"]:
            score += 10.0
        ranked.append((score, -rank, doc))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in ranked[: cfg.top_k]]


def quality_fused(metrics: Dict[str, Any], prefix: str) -> float:
    judge = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
    judge_norm = 0.0 if pd.isna(judge) else float(judge) / 5.0
    return (
        0.25 * float(metrics.get(f"{prefix}_summary_token_f1") or 0)
        + 0.15 * float(metrics.get(f"{prefix}_summary_bleu") or 0)
        + 0.20 * float(metrics.get(f"{prefix}_summary_rouge1") or 0)
        + 0.15 * float(metrics.get(f"{prefix}_summary_rougeL") or 0)
        + 0.25 * judge_norm
    )


def ask_llm(llm: Llama | None, messages: List[Dict[str, str]], cfg: EvalConfig, max_tokens: int | None = None) -> str:
    if cfg.dry_run:
        return json.dumps({"summary": "", "confidence": 0.0, "evidence_sample_ids": []})
    assert llm is not None
    response = llm.create_chat_completion(
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=max_tokens or cfg.max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def parsed_summary(parsed: Dict[str, Any]) -> str:
    multilingual_parts = []
    for code, (name, key) in LANGUAGE_COLUMNS.items():
        value = safe_str(parsed.get(key))
        if value:
            multilingual_parts.append(f"{name}: {value}")
    if multilingual_parts:
        return " | ".join(multilingual_parts)
    for key in ("summary", "long_answer", "answer", "final_summary"):
        value = safe_str(parsed.get(key))
        if value:
            return value
    return ""


def model_instruction(mode: str) -> str:
    if mode == "ga":
        return (
            "Use an evolved multilingual summarization policy: preserve the source "
            "document's core steps, advice, warnings, and outcomes while producing "
            "concise target-language summaries."
        )
    if mode == "es":
        return (
            "Explore several possible multilingual summaries mentally, then return "
            "the one with the fewest missing document facts and no unsupported claims."
        )
    return "Write a faithful concise WikiLingua-style multilingual summary."


def output_contract(row: Dict[str, Any]) -> str:
    if row.get("target_lang") == "all":
        return (
            "Return only valid JSON with keys summary_nl, summary_fr, summary_de, "
            "summary_es, summary, confidence, evidence_sample_ids. The summary key "
            "must concatenate the four language summaries as Dutch: ... | French: ... | "
            "German: ... | Spanish: ..."
        )
    return "Return only valid JSON with keys summary, confidence, evidence_sample_ids."


def answer_once(llm: Llama | None, row: Dict[str, Any], docs: List[Document], cfg: EvalConfig) -> Dict[str, Any]:
    retrieved_context = "\n\n".join(
        f"[{doc.metadata.get('sample_id')}] {compact_text(doc.page_content, 1100)}"
        for doc in docs
        if doc.metadata.get("sample_id") != row["sample_id"]
    )
    prompt = (
        f"{model_instruction(cfg.mode)}\n\n"
        f"{output_contract(row)}\n\n"
        f"Target language: {row['target_language']}\n\n"
        f"English WikiLingua document to summarize:\n{row['transcript']}\n\n"
        f"Retrieved similar WikiLingua documents, for style only:\n{retrieved_context}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a faithful multilingual WikiLingua summarizer."},
            {"role": "user", "content": prompt},
        ],
        cfg,
    )
    return {"raw_response": raw, "parsed_response": parse_response(raw), "trace": [{"node": "one_pass"}]}


def classify_docs(row: Dict[str, Any], docs: List[Document]) -> List[Dict[str, Any]]:
    labels = []
    for idx, doc in enumerate(docs):
        score = lexical_score(row["transcript"][:1200], doc.page_content)
        if doc.metadata.get("sample_id") == row["sample_id"]:
            score += 10
        labels.append(
            {
                "handle": f"P.snippets[{idx}]",
                "sample_id": doc.metadata.get("sample_id", ""),
                "score": round(score, 4),
                "reason": "current document" if doc.metadata.get("sample_id") == row["sample_id"] else "similar document",
            }
        )
    return sorted(labels, key=lambda item: item["score"], reverse=True)


def sub_rlm(
    llm: Llama | None,
    row: Dict[str, Any],
    doc: Document,
    cfg: EvalConfig,
    p_loop: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt = (
        "Extract only summary-worthy facts from this WikiLingua document snippet. Focus on "
        "main steps, advice, conditions, warnings, examples, and final recommendations. "
        "Use Adaptive-P feedback if present. Return only valid JSON with keys "
        "salient_facts, missing_risk, confidence.\n\n"
        f"Target sample id: {row['sample_id']}\n"
        f"Target language: {row['target_language']}\n"
        f"P_loop feedback: {json.dumps(p_loop or {}, ensure_ascii=False)[:1600]}\n"
        f"Snippet id: {doc.metadata.get('sample_id')}\n"
        f"Snippet:\n{compact_text(doc.page_content, 3500)}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a sub-RLM evidence extractor for WikiLingua summarization."},
            {"role": "user", "content": prompt},
        ],
        cfg,
        max_tokens=cfg.subcall_max_tokens,
    )
    parsed = parse_response(raw)
    return {
        "sample_id": doc.metadata.get("sample_id", ""),
        "raw_response": raw,
        "parsed": parsed,
        "salient_facts": parsed.get("salient_facts", []),
        "confidence": parsed.get("confidence", ""),
    }


def root_aggregate(
    llm: Llama | None,
    row: Dict[str, Any],
    sub_results: List[Dict[str, Any]],
    cfg: EvalConfig,
    p_loop: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt = (
        "Create the final WikiLingua summary from the English document and extracted facts. "
        "The final summary must be concise, faithful, and in the requested target language(s). "
        "Do not add unsupported facts. "
        f"{output_contract(row)}\n\n"
        f"Target language: {row['target_language']}\n\n"
        f"English document:\n{row['transcript']}\n\n"
        f"Sub-RLM facts:\n{json.dumps(sub_results, ensure_ascii=False)[:5000]}\n\n"
        f"Adaptive-P feedback:\n{json.dumps(p_loop or {}, ensure_ascii=False)[:1800]}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are the root RLM WikiLingua summarizer."},
            {"role": "user", "content": prompt},
        ],
        cfg,
    )
    return {"raw_response": raw, "parsed_response": parse_response(raw)}


def refine_summary(
    llm: Llama | None,
    row: Dict[str, Any],
    parsed: Dict[str, Any],
    cfg: EvalConfig,
    p_loop: Dict[str, Any],
) -> tuple[Dict[str, Any], str]:
    current = parsed_summary(parsed)
    if cfg.dry_run or not current:
        return parsed, ""
    prompt = (
        "Refine the current WikiLingua summary using the metric feedback. Improve missing "
        "document facts, target-language wording, and factuality, but keep it concise. "
        f"{output_contract(row)}\n\n"
        f"English document:\n{row['transcript']}\n\n"
        f"Current summary: {current}\n\n"
        f"P_loop feedback: {json.dumps(p_loop, ensure_ascii=False)[:2200]}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a precise multilingual summary refiner."},
            {"role": "user", "content": prompt},
        ],
        cfg,
        max_tokens=cfg.max_tokens,
    )
    refined = parse_response(raw)
    summary = parsed_summary(refined)
    if summary:
        updated = dict(parsed)
        updated["summary"] = summary
        return updated, raw
    return parsed, raw


def answer_repl(llm: Llama | None, row: Dict[str, Any], docs: List[Document], cfg: EvalConfig, p_loop: Dict[str, Any] | None = None) -> Dict[str, Any]:
    labels = classify_docs(row, docs)
    chosen = [label["handle"] for label in labels[: min(2, len(labels))]]
    sub_results = []
    trace = [{"node": "batch_classify_snippets", "classifications": labels}]
    for handle in chosen:
        idx = int(re.search(r"\[(\d+)\]", handle).group(1))
        sub_results.append(sub_rlm(llm, row, docs[idx], cfg, p_loop))
        trace.append({"node": "sub_RLM", "handle": handle, "sample_id": docs[idx].metadata.get("sample_id", "")})
        if len(trace) >= cfg.repl_max_steps:
            break
    result = root_aggregate(llm, row, sub_results, cfg, p_loop)
    if p_loop is not None:
        refined, raw_refine = refine_summary(llm, row, result["parsed_response"], cfg, p_loop)
        if refined != result["parsed_response"]:
            result["parsed_response"] = refined
            result["raw_response"] = json.dumps(refined, ensure_ascii=False)
            trace.append({"node": "refine_summary", "raw_response": raw_refine})
    trace.append({"node": "root_aggregate", "sub_result_count": len(sub_results)})
    result["trace"] = trace
    result["sub_results"] = sub_results
    return result


def judge_answer(llm: Llama | None, row: Dict[str, Any], parsed: Dict[str, Any], cfg: EvalConfig) -> Dict[str, Any]:
    if cfg.skip_llm_judge:
        return {"skipped": True, "score_0_to_5": ""}
    if cfg.dry_run:
        return {"score_0_to_5": "", "verdict": "dry_run"}
    prompt = (
        "Grade this WikiLingua summary against the reference summary and English document. "
        "Return only valid JSON with keys score_0_to_5, verdict, missing_facts, hallucinations. "
        "A score of 5 means concise, faithful, and complete; 0 means unrelated or mostly wrong.\n\n"
        f"Target language: {row['target_language']}\n\n"
        f"English document:\n{row['transcript']}\n\n"
        f"Reference summary: {row['gt_summary']}\n\n"
        f"Model summary: {parsed_summary(parsed)}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a strict but fair summarization evaluator."},
            {"role": "user", "content": prompt},
        ],
        cfg,
        max_tokens=cfg.judge_max_tokens,
    )
    return parse_response(raw)


def assess(row: Dict[str, Any], docs: List[Document], parsed: Dict[str, Any], judge: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    summary = parsed_summary(parsed)
    refs = row["references"]
    metrics: Dict[str, Any] = {
        "retrieved_self_hit": int(row["sample_id"] in [doc.metadata.get("sample_id", "") for doc in docs]),
        "llm_judge_json": compact_json(judge),
        "llm_judge_score_0_to_5": judge.get("score_0_to_5", "") if isinstance(judge, dict) else "",
        "llm_judge_verdict": judge.get("verdict", "") if isinstance(judge, dict) else "",
    }
    metrics.update(automatic_metrics(summary, refs, f"{prefix}_summary"))
    metrics.update(automatic_metrics(row["baseline_summary"], refs, "baseline_summary"))
    metrics["quality_fused_score"] = quality_fused(metrics, prefix)
    return metrics


def update_p_loop(
    p_loop: Dict[str, Any],
    loop_index: int,
    metrics: Dict[str, Any],
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(p_loop)
    score = float(metrics.get("quality_fused_score") or 0.0)
    judge = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
    judge_norm = 0.0 if pd.isna(judge) else float(judge) / 5.0
    f1 = float(metrics.get("rlm_summary_token_f1") or 0.0)
    bleu = float(metrics.get("rlm_summary_bleu") or 0.0)
    rouge1 = float(metrics.get("rlm_summary_rouge1") or 0.0)
    rouge_l = float(metrics.get("rlm_summary_rougeL") or 0.0)
    rewards = []
    punishments = []
    if f1 >= 0.55:
        rewards.append("preserve summary fact overlap")
    if rouge1 >= 0.60:
        rewards.append("preserve key reference wording")
    if rouge_l >= 0.45:
        rewards.append("preserve good summary structure")
    if judge_norm >= 0.8:
        rewards.append("preserve factual faithfulness")
    if f1 < 0.55:
        punishments.append("repair F1: include missing reference facts")
    if bleu < 0.12:
        punishments.append("repair BLEU: use closer concise phrasing")
    if rouge1 < 0.60:
        punishments.append("repair ROUGE-1: mention important names/actions/times")
    if rouge_l < 0.45:
        punishments.append("repair ROUGE-L: follow reference summary order")
    if judge_norm < 0.8:
        punishments.append("repair judge score: remove unsupported claims")
    if not parsed_summary(parsed):
        punishments.append("repair empty summary")
    attempt = {
        "loop": loop_index,
        "quality": round(score, 6),
        "summary_f1": round(f1, 3),
        "summary_bleu": round(bleu, 3),
        "summary_rouge1": round(rouge1, 3),
        "summary_rougeL": round(rouge_l, 3),
        "judge_norm": round(judge_norm, 3),
        "summary_preview": parsed_summary(parsed)[:260],
        "reward": rewards,
        "punishment": punishments,
    }
    attempts = list(updated.get("previous_attempts", [])) + [attempt]
    updated["previous_attempts"] = attempts[-6:]
    updated["reward_signal"] = "; ".join(rewards) if rewards else "no strong positive signal yet"
    updated["punishment_signal"] = "; ".join(punishments) if punishments else "preserve current strategy"
    if score >= float(updated.get("best_score", 0.0)):
        updated["best_score"] = score
        updated["best_summary"] = parsed_summary(parsed)
        updated["best_loop"] = loop_index
    updated["loop_index"] = loop_index + 1
    return updated


def run_row(llm: Llama | None, row: Dict[str, Any], docs: List[Document], cfg: EvalConfig) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if cfg.mode == "adaptive_p":
        best: tuple[tuple[float, float, float, float], Dict[str, Any], Dict[str, Any], Dict[str, Any]] | None = None
        p_loop: Dict[str, Any] = {"loop_index": 1, "best_score": 0.0, "previous_attempts": []}
        for loop in range(1, cfg.p_loop_count + 1):
            result = answer_repl(llm, row, docs, cfg, p_loop)
            parsed = result["parsed_response"]
            judge = judge_answer(llm, row, parsed, cfg)
            metrics = assess(row, docs, parsed, judge, "rlm")
            p_loop = update_p_loop(p_loop, loop, metrics, parsed)
            judge_value = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
            judge_norm = 0.0 if pd.isna(judge_value) else float(judge_value) / 5.0
            quality = float(metrics.get("quality_fused_score") or 0.0)
            f1 = float(metrics.get("rlm_summary_token_f1") or 0.0)
            rouge1 = float(metrics.get("rlm_summary_rouge1") or 0.0)
            candidate_key = (quality, judge_norm, f1, rouge1)
            print(
                f"[p-loop] {loop}/{cfg.p_loop_count} quality={quality:.3f} "
                f"f1={f1:.3f} rouge1={rouge1:.3f} "
                f"reward={p_loop.get('reward_signal', '')[:80]} "
                f"punish={p_loop.get('punishment_signal', '')[:80]}",
                flush=True,
            )
            if best is None or candidate_key > best[0]:
                best = (candidate_key, result, judge, metrics)
            if quality >= 0.82 or (f1 >= 0.65 and rouge1 >= 0.70 and judge_norm >= 0.9):
                print("[p-loop] Early stop: strong quality signal", flush=True)
                break
        assert best is not None
        return best[1], best[2], best[3]
    if cfg.mode in {"repl4", "repl8"}:
        result = answer_repl(llm, row, docs, cfg, None)
        parsed = result["parsed_response"]
        judge = judge_answer(llm, row, parsed, cfg)
        metrics = assess(row, docs, parsed, judge, "rlm")
        return result, judge, metrics
    result = answer_once(llm, row, docs, cfg)
    parsed = result["parsed_response"]
    judge = judge_answer(llm, row, parsed, cfg)
    metrics = assess(row, docs, parsed, judge, "rag")
    return result, judge, metrics


def write_summary(rows: List[Dict[str, Any]], path: Path, prefix: str) -> None:
    df = pd.DataFrame(rows)
    judge = pd.to_numeric(df.get("llm_judge_score_0_to_5", pd.Series(dtype=float)), errors="coerce")
    summary = {
        "rows_used": len(df),
        "summary_f1": df[f"{prefix}_summary_token_f1"].mean() if f"{prefix}_summary_token_f1" in df else np.nan,
        "summary_bleu": df[f"{prefix}_summary_bleu"].mean() if f"{prefix}_summary_bleu" in df else np.nan,
        "summary_rouge1": df[f"{prefix}_summary_rouge1"].mean() if f"{prefix}_summary_rouge1" in df else np.nan,
        "summary_rouge2": df[f"{prefix}_summary_rouge2"].mean() if f"{prefix}_summary_rouge2" in df else np.nan,
        "summary_rougeL": df[f"{prefix}_summary_rougeL"].mean() if f"{prefix}_summary_rougeL" in df else np.nan,
        "llm_judge_score_norm": (judge / 5.0).mean(),
        "quality_fused_score": df["quality_fused_score"].mean(),
        "elapsed_seconds": df["elapsed_seconds"].mean(),
        "retrieved_self_hit": df["retrieved_self_hit"].mean(),
    }
    pd.DataFrame([summary]).to_csv(path, index=False)


def write_checkpoint(rows: List[Dict[str, Any]], trace_path: Path, summary_path: Path, prefix: str, label: str) -> None:
    pd.DataFrame(rows).to_csv(trace_path, index=False)
    write_summary(rows, summary_path, prefix)
    print(f"[checkpoint] {label}: saved {len(rows)} rows to {trace_path}", flush=True)


def run(cfg: EvalConfig) -> None:
    model_name, stem = MODE_META[cfg.mode]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = load_rows(cfg.input_path, cfg.max_rows)
    df = normalize_dataset(df_raw, cfg.target_lang, cfg.max_rows)
    store, rebuilt = build_vectorstore(df, cfg)

    class _Cfg:
        dry_run = cfg.dry_run
        model_path = cfg.model_path
        n_ctx = cfg.n_ctx
        n_threads = cfg.n_threads
        n_gpu_layers = cfg.n_gpu_layers

    if not cfg.dry_run and not cfg.model_path:
        print(f"[llm] WikiLingua default quantized model: {DEFAULT_QUANTIZED_MODEL}", flush=True)
    llm = load_llm(_Cfg())
    trace_path = cfg.out_dir / f"{stem}_assessment_traces.csv"
    summary_path = cfg.out_dir / f"{stem}_summary.csv"
    rows: List[Dict[str, Any]] = []
    done = set()
    if trace_path.exists() and not cfg.no_resume and not rebuilt:
        old = pd.read_csv(trace_path)
        rows = old.to_dict("records")
        done = set(old["sample_id"].astype(str).tolist()) if "sample_id" in old else set()
        print(f"[resume] Loaded {len(rows)} rows", flush=True)
    prefix = "rlm" if cfg.mode in {"repl4", "repl8", "adaptive_p"} else "rag"
    total_started = time.time()
    for idx, row in enumerate(df.to_dict("records"), 1):
        if row["sample_id"] in done:
            print(f"[resume] skip {idx}/{len(df)} {row['sample_id']}", flush=True)
            continue
        started = time.time()
        print(f"\n[document] {idx}/{len(df)} {row['sample_id']} target={row['target_language']}", flush=True)
        docs = retrieve_docs(store, row, cfg)
        print("[retrieve] " + ", ".join(doc.metadata.get("sample_id", "") for doc in docs), flush=True)
        result, judge, metrics = run_row(llm, row, docs, cfg)
        elapsed = time.time() - started
        parsed = result["parsed_response"]
        summary = parsed_summary(parsed)
        out = {
            "model": model_name,
            "mode": cfg.mode,
            "row_number": row["row_number"],
            "sample_id": row["sample_id"],
            "document": row["transcript"],
            "transcript": row["transcript"],
            "target_lang": row["target_lang"],
            "target_language": row["target_language"],
            "ground_truth_summary": row["gt_summary"],
            "baseline_summary": row["baseline_summary"],
            "retrieved_sample_ids": compact_json([doc.metadata.get("sample_id", "") for doc in docs]),
            "raw_response": result["raw_response"],
            "parsed_response": compact_json(parsed),
            "summary": summary,
            "confidence": parsed.get("confidence", ""),
            "trace_json": compact_json(result.get("trace", [])),
            "elapsed_seconds": elapsed,
            "estimated_lm_calls": (
                cfg.p_loop_count * 4 if cfg.mode == "adaptive_p" else 4 if cfg.mode in {"repl4", "repl8"} else 2
            ),
        }
        out.update(metrics)
        rows.append(out)
        if len(rows) % cfg.checkpoint_every == 0:
            write_checkpoint(rows, trace_path, summary_path, prefix, f"row {idx}/{len(df)}")
        print(
            f"[{idx}/{len(df)}] f1={metrics.get(f'{prefix}_summary_token_f1', 0):.3f} "
            f"rouge1={metrics.get(f'{prefix}_summary_rouge1', 0):.3f} "
            f"fused={metrics['quality_fused_score']:.3f} elapsed={elapsed:.1f}s",
            flush=True,
        )
    write_checkpoint(rows, trace_path, summary_path, prefix, "final")
    print(f"[done] Total elapsed {time.time() - total_started:.1f}s", flush=True)
    print(f"[done] Wrote {trace_path}", flush=True)
    print(f"[done] Wrote {summary_path}", flush=True)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=sorted(MODE_META), default="rag")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--retrieval-fetch-k", type=int, default=20)
    parser.add_argument("--p-loop-count", type=int, default=4)
    parser.add_argument("--repl-max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--subcall-max-tokens", type=int, default=256)
    parser.add_argument("--judge-max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--target-lang",
        choices=["all", *sorted(LANGUAGE_COLUMNS)],
        default="all",
        help="Target summary language: all, nl, fr, de, or es.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / MODE_META[args.mode][1])
    repl_steps = args.repl_max_steps
    if repl_steps is None:
        repl_steps = 8 if args.mode == "repl8" else 4
    return EvalConfig(
        input_path=args.input,
        out_dir=out_dir,
        mode=args.mode,
        max_rows=args.max_rows,
        top_k=args.top_k,
        retrieval_fetch_k=max(args.top_k, args.retrieval_fetch_k),
        p_loop_count=max(1, args.p_loop_count),
        repl_max_steps=max(1, repl_steps),
        max_tokens=args.max_tokens,
        subcall_max_tokens=args.subcall_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        temperature=args.temperature,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        model_path=args.model_path,
        rebuild_index=args.rebuild_index,
        no_resume=args.no_resume,
        skip_llm_judge=args.skip_llm_judge,
        dry_run=args.dry_run,
        checkpoint_every=max(1, args.checkpoint_every),
        target_lang=args.target_lang,
    )


if __name__ == "__main__":
    run(parse_args())
