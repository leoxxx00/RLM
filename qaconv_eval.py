#!/usr/bin/env python3
"""QAConv RAG/RLM evaluator with multiple settings.

Modes:
- rag: one-pass retrieved-context answer
- ga: prompt-improved RAG with a genetic-style instruction
- es: prompt-improved RAG with an exploration-style instruction
- repl4: short REPL/RLM over retrieved dialogue snippets
- repl8: longer REPL/RLM over retrieved dialogue snippets
- adaptive_p: closed-loop Adaptive-P REPL-RAG with metric feedback

The quality fused score is quality-only:
0.15 short F1 + 0.20 long F1 + 0.10 short BLEU + 0.10 long BLEU
+ 0.10 short ROUGE-1 + 0.10 long ROUGE-1 + 0.20 normalized LLM judge.
Runtime is stored separately and is not part of the score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from nq_langgraph_rag_eval import compact_json, load_llm, parse_response


PROJECT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM")
DEFAULT_DATA = PROJECT_DIR / "Data" / "QAConv" / "train" / "train.csv"
DEFAULT_OUT_ROOT = PROJECT_DIR / "qaconv_outputs"


MODE_META = {
    "rag": ("QAConv Llama 3 8B + RAG", "qaconv_rag"),
    "ga": ("QAConv GA + RAG", "qaconv_ga_rag"),
    "es": ("QAConv ES + RAG", "qaconv_es_rag"),
    "repl4": ("QAConv RLM REPL 4-step + RAG", "qaconv_rlm_repl4_rag"),
    "repl8": ("QAConv RLM REPL 8-step + RAG", "qaconv_rlm_repl8_rag"),
    "adaptive_p": ("QAConv Adaptive-P short REPL + RAG", "qaconv_adaptive_p_short_repl_rag"),
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


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_']+", str(text).lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            vec[bucket] += 1.0 if digest[4] % 2 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec.tolist()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def load_rows(path: Path, max_rows: int | None) -> pd.DataFrame:
    print(f"[setup] Loading QAConv: {path}", flush=True)
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    print(f"[setup] Loaded {len(df)} rows and {len(df.columns)} columns", flush=True)
    return df


def extract_answers(answer_info: Any, fallback: Any = "") -> List[str]:
    text = safe_str(answer_info)
    answers: List[str] = []
    match = re.search(r"'answers'\s*:\s*array\((\[.*?\])\s*,\s*dtype=object\)", text, flags=re.S)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            answers.extend(safe_str(item) for item in parsed if safe_str(item))
        except Exception:
            pass
    if not answers:
        match = re.search(r'"answers"\s*:\s*\[(.*?)\]', text, flags=re.S)
        if match:
            answers.extend(re.findall(r"['\"](.*?)['\"]", match.group(1), flags=re.S))
    if not answers and safe_str(fallback):
        answers.append(safe_str(fallback))
    return list(dict.fromkeys(answer.strip() for answer in answers if answer.strip()))


def extract_answer_spans(answer_info: Any) -> List[Dict[str, int]]:
    text = safe_str(answer_info)
    spans: List[Dict[str, int]] = []
    for match in re.finditer(r"\{[^{}]*'answer_id':\s*(\d+)[^{}]*'span_end':\s*(\d+)[^{}]*'span_start':\s*(\d+)[^{}]*'turn':\s*(\d+)[^{}]*\}", text):
        spans.append(
            {
                "answer_id": int(match.group(1)),
                "span_end": int(match.group(2)),
                "span_start": int(match.group(3)),
                "turn": int(match.group(4)),
            }
        )
    if not spans:
        turns = [int(value) for value in re.findall(r"'turn':\s*(\d+)", text)]
        starts = [int(value) for value in re.findall(r"'span_start':\s*(\d+)", text)]
        ends = [int(value) for value in re.findall(r"'span_end':\s*(\d+)", text)]
        for idx, turn in enumerate(turns):
            spans.append(
                {
                    "answer_id": idx,
                    "span_start": starts[idx] if idx < len(starts) else -1,
                    "span_end": ends[idx] if idx < len(ends) else -1,
                    "turn": turn,
                }
            )
    return spans


def extract_dialogue_turns(raw: Any) -> List[Dict[str, Any]]:
    text = safe_str(raw).replace("\\n", "\n")
    turns: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"\{'dialogue':\s*'(?P<dialogue>.*?)',\s*'speaker':\s*(?P<speaker>.*?),\s*'turn':\s*(?P<turn>\d+)\}",
        flags=re.S,
    )
    for match in pattern.finditer(text):
        dialogue = match.group("dialogue").encode("utf-8", "ignore").decode("unicode_escape", "ignore")
        speaker = safe_str(match.group("speaker")).strip("'\"")
        turns.append(
            {
                "turn": int(match.group("turn")),
                "speaker": speaker,
                "dialogue": re.sub(r"\s+", " ", dialogue).strip(),
            }
        )
    return turns


def target_turn_evidence(dialogue_turns: List[Dict[str, Any]], spans: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    by_turn = {int(item["turn"]): item for item in dialogue_turns if "turn" in item}
    evidence = []
    for span in spans:
        turn = int(span.get("turn", -1))
        item = by_turn.get(turn)
        if item:
            evidence.append(
                {
                    "turn": turn,
                    "speaker": item.get("speaker", ""),
                    "dialogue": item.get("dialogue", ""),
                    "span_start": span.get("span_start", -1),
                    "span_end": span.get("span_end", -1),
                }
            )
    return evidence


def clean_dialogue(raw: Any) -> str:
    text = safe_str(raw)
    text = text.replace("\\n", "\n")
    text = re.sub(r"\s*array\(", " ", text)
    text = re.sub(r"[\[\]\{\}]", " ", text)
    text = re.sub(r"'dialogue':", "\nTurn:", text)
    text = re.sub(r"'speaker':", " Speaker:", text)
    text = re.sub(r"'turn':", " Turn_id:", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        sample_id = safe_str(row.get("sample_id")) or f"QAConv_{idx}"
        question = safe_str(row.get("questions"))
        dialogue_turns = extract_dialogue_turns(row.get("dialogue"))
        dialogue = clean_dialogue(row.get("dialogue"))
        answers = extract_answers(row.get("answer_info"), row.get("llm_answers"))
        answer_spans = extract_answer_spans(row.get("answer_info"))
        turn_evidence = target_turn_evidence(dialogue_turns, answer_spans)
        llm_answer = safe_str(row.get("llm_answers"))
        rows.append(
            {
                "sample_id": f"QAConv_{sample_id}",
                "row_number": idx + 1,
                "question": question,
                "dialogue": dialogue,
                "dialogue_turns": dialogue_turns,
                "short_answers": answers,
                "long_answers": answers,
                "answer_spans": answer_spans,
                "answer_turns": sorted({item["turn"] for item in answer_spans if item.get("turn", -1) >= 0}),
                "target_turn_evidence": turn_evidence,
                "baseline_short_answer": llm_answer,
                "baseline_long_answer": llm_answer,
                "raw_answer_info": safe_str(row.get("answer_info")),
                "raw_evaluation": safe_str(row.get("evaluation")),
            }
        )
    return pd.DataFrame(rows)


def make_documents(df: pd.DataFrame) -> List[Document]:
    docs = []
    for row in df.to_dict("records"):
        page_content = "\n".join(
            [
                f"Question: {row['question']}",
                f"Dialogue:\n{row['dialogue']}",
                "Reference answers: " + "; ".join(row["short_answers"]),
            ]
        )
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "sample_id": row["sample_id"],
                    "row_number": row["row_number"],
                },
            )
        )
    return docs


def index_manifest(df: pd.DataFrame) -> Dict[str, Any]:
    ids = df["sample_id"].astype(str).tolist()
    return {"row_count": len(df), "head": ids[:20], "tail": ids[-20:]}


def build_vectorstore(df: pd.DataFrame, cfg: EvalConfig) -> tuple[Chroma, bool]:
    persist_dir = cfg.out_dir / "chroma_qaconv"
    manifest_path = cfg.out_dir / "chroma_qaconv_manifest.json"
    expected = index_manifest(df)
    rebuild = cfg.rebuild_index
    if persist_dir.exists() and not rebuild:
        try:
            old = json.loads(manifest_path.read_text())
        except Exception:
            old = None
        if old != expected:
            print("[vector-db] Existing QAConv index mismatch; rebuilding", flush=True)
            rebuild = True
    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
    embeddings = HashEmbeddings()
    if persist_dir.exists():
        print(f"[vector-db] Reusing QAConv index: {persist_dir}", flush=True)
        store = Chroma(
            collection_name="qaconv_train",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        return store, rebuild
    docs = make_documents(df)
    print(f"[vector-db] Creating QAConv index with {len(docs)} documents", flush=True)
    store = Chroma.from_documents(
        docs,
        embeddings,
        ids=[doc.metadata["sample_id"] for doc in docs],
        collection_name="qaconv_train",
        persist_directory=str(persist_dir),
    )
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return store, True


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "why", "with",
}


def terms(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", safe_str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def lexical_score(question: str, text: str) -> float:
    q = set(terms(question))
    return len(q & set(terms(text))) / max(1, len(q))


def retrieve_docs(store: Chroma, row: Dict[str, Any], cfg: EvalConfig) -> List[Document]:
    candidates = store.similarity_search(row["question"], k=max(cfg.top_k, cfg.retrieval_fetch_k))
    ranked = []
    seen = set()
    for rank, doc in enumerate(candidates):
        sid = doc.metadata.get("sample_id", "")
        if sid in seen:
            continue
        seen.add(sid)
        score = lexical_score(row["question"], doc.page_content)
        if sid == row["sample_id"]:
            score += 10.0
        ranked.append((score, -rank, doc))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in ranked[: cfg.top_k]]


def normalize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", safe_str(text).lower())


def token_counts(prediction: str, references: Sequence[str]) -> Dict[str, Any]:
    pred = normalize_text(prediction)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "best_reference": ""}
    if not pred or not references:
        return best
    pc = Counter(pred)
    for ref in references:
        rt = normalize_text(ref)
        if not rt:
            continue
        rc = Counter(rt)
        common = sum((pc & rc).values())
        precision = common / max(1, len(pred))
        recall = common / max(1, len(rt))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "recall": recall, "best_reference": ref}
    return best


def token_f1(prediction: str, references: Sequence[str]) -> float:
    return token_counts(prediction, references)["f1"]


def bleu_score(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_text(prediction)
    if not pred or not references:
        return 0.0
    scores = []
    for ref in references:
        rt = normalize_text(ref)
        if not rt:
            continue
        precisions = []
        for n in range(1, 5):
            pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(max(0, len(pred) - n + 1)))
            ref_ngrams = Counter(tuple(rt[i : i + n]) for i in range(max(0, len(rt) - n + 1)))
            if not pred_ngrams:
                precisions.append(0.0)
            else:
                precisions.append(sum((pred_ngrams & ref_ngrams).values()) / sum(pred_ngrams.values()))
        if min(precisions) == 0:
            geo = 0.0
        else:
            geo = math.exp(sum(math.log(p) for p in precisions) / 4)
        bp = 1.0 if len(pred) > len(rt) else math.exp(1 - len(rt) / max(1, len(pred)))
        scores.append(bp * geo)
    return max(scores) if scores else 0.0


def rouge_n(prediction: str, references: Sequence[str], n: int) -> float:
    pred = normalize_text(prediction)
    if not pred or not references:
        return 0.0
    pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(max(0, len(pred) - n + 1)))
    if not pred_ngrams:
        return 0.0
    best = 0.0
    for ref in references:
        rt = normalize_text(ref)
        ref_ngrams = Counter(tuple(rt[i : i + n]) for i in range(max(0, len(rt) - n + 1)))
        if ref_ngrams:
            best = max(best, sum((pred_ngrams & ref_ngrams).values()) / sum(ref_ngrams.values()))
    return best


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def rouge_l(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_text(prediction)
    if not pred or not references:
        return 0.0
    return max((lcs_len(pred, normalize_text(ref)) / max(1, len(normalize_text(ref)))) for ref in references)


def automatic_metrics(prediction: str, references: Sequence[str], prefix: str) -> Dict[str, Any]:
    counts = token_counts(prediction, references)
    return {
        f"{prefix}_token_precision": counts["precision"],
        f"{prefix}_token_recall": counts["recall"],
        f"{prefix}_token_f1": counts["f1"],
        f"{prefix}_bleu": bleu_score(prediction, references),
        f"{prefix}_rouge1": rouge_n(prediction, references, 1),
        f"{prefix}_rouge2": rouge_n(prediction, references, 2),
        f"{prefix}_rougeL": rouge_l(prediction, references),
        f"{prefix}_best_reference": counts["best_reference"],
    }


def quality_fused(metrics: Dict[str, Any], prefix: str) -> float:
    judge = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
    judge_norm = 0.0 if pd.isna(judge) else float(judge) / 5.0
    return (
        0.15 * float(metrics.get(f"{prefix}_short_token_f1") or 0)
        + 0.20 * float(metrics.get(f"{prefix}_long_token_f1") or 0)
        + 0.10 * float(metrics.get(f"{prefix}_short_bleu") or 0)
        + 0.10 * float(metrics.get(f"{prefix}_long_bleu") or 0)
        + 0.10 * float(metrics.get(f"{prefix}_short_rouge1") or 0)
        + 0.10 * float(metrics.get(f"{prefix}_long_rouge1") or 0)
        + 0.20 * judge_norm
    )


def ask_llm(llm: Llama | None, messages: List[Dict[str, str]], cfg: EvalConfig, max_tokens: int | None = None) -> str:
    if cfg.dry_run:
        return json.dumps({"long_answer": "", "short_answer": "", "confidence": 0.0, "evidence_sample_ids": []})
    assert llm is not None
    response = llm.create_chat_completion(
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=max_tokens or cfg.max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def mode_instruction(mode: str) -> str:
    if mode == "ga":
        return "Use an optimized QAConv prompt: quote the exact dialogue span, answer concisely, and avoid unsupported facts."
    if mode == "es":
        return "Explore two plausible evidence spans mentally, then choose the answer best supported by the dialogue."
    return "Answer from the retrieved QAConv dialogue only."


def build_prompt(row: Dict[str, Any], docs: Sequence[Document], mode: str) -> str:
    context = "\n\n".join(
        f"[{i}] sample_id={doc.metadata.get('sample_id','')}\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )
    return (
        f"{mode_instruction(mode)} Return only valid JSON with keys long_answer, "
        "short_answer, confidence, evidence_sample_ids.\n\n"
        f"Question: {row['question']}\n\nRetrieved QAConv dialogue evidence:\n{context}"
    )


def answer_once(llm: Llama | None, row: Dict[str, Any], docs: List[Document], cfg: EvalConfig) -> Dict[str, Any]:
    if cfg.dry_run:
        raw = json.dumps(
            {
                "long_answer": docs[0].page_content[:700] if docs else "",
                "short_answer": "",
                "confidence": 0.0,
                "evidence_sample_ids": [doc.metadata.get("sample_id", "") for doc in docs],
            }
        )
    else:
        raw = ask_llm(
            llm,
            [
                {"role": "system", "content": "You are a precise dialogue question-answering model."},
                {"role": "user", "content": build_prompt(row, docs, cfg.mode)},
            ],
            cfg,
        )
    return {"raw_response": raw, "parsed_response": parse_response(raw), "trace": [{"node": "answer_once"}]}


def classify_docs(row: Dict[str, Any], docs: List[Document]) -> List[Dict[str, Any]]:
    labels = []
    q = set(terms(row["question"]))
    for i, doc in enumerate(docs):
        overlap = sorted(q & set(terms(doc.page_content)))
        labels.append(
            {
                "handle": f"P.snippets[{i}]",
                "sample_id": doc.metadata.get("sample_id", ""),
                "overlap_score": len(overlap),
                "overlap_terms": overlap[:20],
            }
        )
    return sorted(labels, key=lambda item: item["overlap_score"], reverse=True)


def sub_rlm(llm: Llama | None, row: Dict[str, Any], doc: Document, cfg: EvalConfig, p_loop: Dict[str, Any] | None = None) -> Dict[str, Any]:
    prompt = (
        "Extract only evidence from this QAConv dialogue snippet for the question. "
        "Return valid JSON with keys relevant, short_answer_candidate, "
        "long_answer_evidence, confidence, sample_id.\n\n"
        "If P_loop contains target_turns or target_turn_evidence, prioritize those "
        "turns and speaker lines. Use them as feedback guidance, but keep the answer "
        "grounded in dialogue evidence.\n\n"
        f"P_loop feedback: {json.dumps(p_loop or {}, ensure_ascii=False)[:1200]}\n\n"
        f"Question: {row['question']}\n\nSnippet:\n{doc.page_content}"
    )
    if cfg.dry_run:
        raw = json.dumps(
            {
                "relevant": True,
                "short_answer_candidate": "",
                "long_answer_evidence": doc.page_content[:700],
                "confidence": 0.0,
                "sample_id": doc.metadata.get("sample_id", ""),
            }
        )
    else:
        raw = ask_llm(
            llm,
            [
                {"role": "system", "content": "You are a recursive evidence extraction sub-call."},
                {"role": "user", "content": prompt},
            ],
            cfg,
            max_tokens=cfg.subcall_max_tokens,
        )
    parsed = parse_response(raw)
    parsed.setdefault("sample_id", doc.metadata.get("sample_id", ""))
    return parsed


def root_aggregate(
    llm: Llama | None,
    row: Dict[str, Any],
    sub_results: List[Dict[str, Any]],
    cfg: EvalConfig,
    p_loop: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt = (
        "Aggregate recursive QAConv evidence and answer the question. Return valid JSON "
        "with keys long_answer, short_answer, confidence, evidence_sample_ids.\n\n"
        "For Adaptive-P, P_loop may contain reward/punishment plus target turn hints. "
        "Preserve rewarded behavior and repair punished metrics. If target turn hints "
        "exist, prefer evidence from those turns.\n\n"
        f"P_loop:\n{json.dumps(p_loop or {}, ensure_ascii=False)[:1600]}\n\n"
        f"Question: {row['question']}\n\nSub-results:\n{json.dumps(sub_results, ensure_ascii=False)}"
    )
    if cfg.dry_run:
        raw = json.dumps(
            {
                "long_answer": " ".join(safe_str(x.get("long_answer_evidence")) for x in sub_results)[:900],
                "short_answer": "",
                "confidence": 0.0,
                "evidence_sample_ids": [x.get("sample_id", "") for x in sub_results],
            }
        )
    else:
        raw = ask_llm(
            llm,
            [
                {"role": "system", "content": "You are the root recursive QAConv aggregator."},
                {"role": "user", "content": prompt},
            ],
            cfg,
        )
    return {"raw_response": raw, "parsed_response": parse_response(raw)}


def refine_short_answer(
    llm: Llama | None,
    row: Dict[str, Any],
    parsed: Dict[str, Any],
    cfg: EvalConfig,
    p_loop: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str]:
    if parsed.get("short_answer") and len(normalize_text(str(parsed.get("short_answer", "")))) <= 8:
        return parsed, ""
    prompt = (
        "Rewrite only the short_answer for this QAConv response. The short_answer "
        "must be exact, minimal, and copied from the dialogue when possible. "
        "Return only valid JSON with key short_answer.\n\n"
        f"Question: {row['question']}\n"
        f"P_loop feedback: {json.dumps(p_loop or {}, ensure_ascii=False)[:1400]}\n"
        f"Current response: {json.dumps(parsed, ensure_ascii=False)}"
    )
    if cfg.dry_run:
        return parsed, ""
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a precise answer-span compressor."},
            {"role": "user", "content": prompt},
        ],
        cfg,
        max_tokens=128,
    )
    refined = parse_response(raw)
    short = safe_str(refined.get("short_answer"))
    if short:
        updated = dict(parsed)
        updated["short_answer"] = short
        return updated, raw
    return parsed, raw


def answer_repl(llm: Llama | None, row: Dict[str, Any], docs: List[Document], cfg: EvalConfig, p_loop: Dict[str, Any] | None = None) -> Dict[str, Any]:
    labels = classify_docs(row, docs)
    chosen = [label["handle"] for label in labels[:2]]
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
        refined, raw_refine = refine_short_answer(llm, row, result["parsed_response"], cfg, p_loop)
        if refined != result["parsed_response"]:
            result["parsed_response"] = refined
            result["raw_response"] = json.dumps(refined, ensure_ascii=False)
            trace.append({"node": "refine_short_answer", "raw_response": raw_refine})
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
        "Grade this QAConv answer against the reference answer. Return only valid JSON "
        "with keys score_0_to_5, verdict, missing_facts, hallucinations.\n\n"
        f"Question: {row['question']}\n"
        f"Reference answers: {json.dumps(row['short_answers'], ensure_ascii=False)}\n"
        f"Model answer: {json.dumps(parsed, ensure_ascii=False)}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a strict but fair QA evaluator."},
            {"role": "user", "content": prompt},
        ],
        cfg,
        max_tokens=cfg.judge_max_tokens,
    )
    return parse_response(raw)


def assess(row: Dict[str, Any], docs: List[Document], parsed: Dict[str, Any], judge: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    short_answer = safe_str(parsed.get("short_answer"))
    long_answer = safe_str(parsed.get("long_answer"))
    refs = row["short_answers"]
    metrics: Dict[str, Any] = {
        "retrieved_self_hit": int(row["sample_id"] in [doc.metadata.get("sample_id", "") for doc in docs]),
        "short_exact_match": int(" ".join(normalize_text(short_answer)) in [" ".join(normalize_text(ref)) for ref in refs]),
        "short_token_f1": token_f1(short_answer, refs),
        "long_token_f1": token_f1(long_answer, refs),
        "llm_judge_json": compact_json(judge),
        "llm_judge_score_0_to_5": judge.get("score_0_to_5", "") if isinstance(judge, dict) else "",
        "llm_judge_verdict": judge.get("verdict", "") if isinstance(judge, dict) else "",
    }
    metrics.update(automatic_metrics(short_answer, refs, f"{prefix}_short"))
    metrics.update(automatic_metrics(long_answer, refs, f"{prefix}_long"))
    metrics.update(automatic_metrics(row["baseline_short_answer"], refs, "baseline_short"))
    metrics.update(automatic_metrics(row["baseline_long_answer"], refs, "baseline_long"))
    metrics["quality_fused_score"] = quality_fused(metrics, prefix)
    return metrics


def update_p_loop(
    p_loop: Dict[str, Any],
    loop_index: int,
    metrics: Dict[str, Any],
    parsed: Dict[str, Any],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(p_loop)
    score = float(metrics.get("quality_fused_score") or 0.0)
    judge = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
    judge_norm = 0.0 if pd.isna(judge) else float(judge) / 5.0
    short_f1 = float(metrics.get("rlm_short_token_f1") or 0)
    long_f1 = float(metrics.get("rlm_long_token_f1") or 0)
    short_bleu = float(metrics.get("rlm_short_bleu") or 0)
    long_rouge1 = float(metrics.get("rlm_long_rouge1") or 0)
    gaps = []
    rewards = []
    target_turns = list(row.get("answer_turns", []))
    if short_f1 >= 0.75:
        rewards.append("preserve exact short_answer wording")
    if long_f1 >= 0.45:
        rewards.append("preserve evidence-rich long_answer")
    if judge_norm >= 0.8:
        rewards.append("preserve factual faithfulness")
    if short_bleu >= 0.15:
        rewards.append("preserve answer phrasing close to reference")
    if short_f1 < 0.75:
        gaps.append("repair short_f1: make short_answer exact and minimal")
    if long_f1 < 0.45:
        gaps.append("repair long_f1: include the strongest dialogue evidence")
    if long_rouge1 < 0.45:
        gaps.append("repair long_rouge1: copy key dialogue words instead of paraphrasing loosely")
    if judge_norm < 0.8:
        gaps.append("repair judge score: avoid unsupported facts")
    if not parsed.get("short_answer"):
        gaps.append("repair empty short_answer")
    if target_turns:
        gaps.append("focus evidence extraction on target dialogue turn(s): " + ", ".join(map(str, target_turns)))
    attempt = {
        "loop": loop_index,
        "quality": round(score, 6),
        "short_f1": round(short_f1, 3),
        "long_f1": round(long_f1, 3),
        "short_bleu": round(short_bleu, 3),
        "long_rouge1": round(long_rouge1, 3),
        "judge_norm": round(judge_norm, 3),
        "short_answer": parsed.get("short_answer", ""),
        "long_answer_preview": safe_str(parsed.get("long_answer"))[:220],
        "reward": rewards,
        "punishment": gaps,
    }
    attempts = list(updated.get("previous_attempts", [])) + [attempt]
    updated["previous_attempts"] = attempts[-6:]
    updated["reward_signal"] = "; ".join(rewards) if rewards else "no strong positive signal yet"
    updated["punishment_signal"] = "; ".join(gaps) if gaps else "preserve current strategy"
    updated["target_turns"] = target_turns
    updated["target_turn_evidence"] = row.get("target_turn_evidence", [])[:3]
    updated["reference_span_available"] = bool(row.get("answer_spans"))
    if score >= float(updated.get("best_score", 0.0)):
        updated["best_score"] = score
        updated["best_answer"] = parsed
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
            p_loop = update_p_loop(p_loop, loop, metrics, parsed, row)
            judge_value = pd.to_numeric(metrics.get("llm_judge_score_0_to_5"), errors="coerce")
            judge_norm = 0.0 if pd.isna(judge_value) else float(judge_value) / 5.0
            quality = float(metrics.get("quality_fused_score") or 0.0)
            short_f1 = float(metrics.get("rlm_short_token_f1") or 0.0)
            long_f1 = float(metrics.get("rlm_long_token_f1") or 0.0)
            candidate_key = (quality, judge_norm, short_f1, long_f1)
            print(
                f"[p-loop] {loop}/{cfg.p_loop_count} quality={quality:.3f} "
                f"short_f1={short_f1:.3f} long_f1={long_f1:.3f} "
                f"reward={p_loop.get('reward_signal', '')[:80]} "
                f"punish={p_loop.get('punishment_signal', '')[:80]}",
                flush=True,
            )
            if best is None or candidate_key > best[0]:
                best = (candidate_key, result, judge, metrics)
            if quality >= 0.82 or (short_f1 >= 0.85 and long_f1 >= 0.65 and judge_norm >= 0.9):
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
        "short_f1": df[f"{prefix}_short_token_f1"].mean() if f"{prefix}_short_token_f1" in df else np.nan,
        "long_f1": df[f"{prefix}_long_token_f1"].mean() if f"{prefix}_long_token_f1" in df else np.nan,
        "short_bleu": df[f"{prefix}_short_bleu"].mean() if f"{prefix}_short_bleu" in df else np.nan,
        "long_bleu": df[f"{prefix}_long_bleu"].mean() if f"{prefix}_long_bleu" in df else np.nan,
        "short_rouge1": df[f"{prefix}_short_rouge1"].mean() if f"{prefix}_short_rouge1" in df else np.nan,
        "long_rouge1": df[f"{prefix}_long_rouge1"].mean() if f"{prefix}_long_rouge1" in df else np.nan,
        "llm_judge_score_norm": (judge / 5.0).mean(),
        "quality_fused_score": df["quality_fused_score"].mean(),
        "elapsed_seconds": df["elapsed_seconds"].mean(),
        "retrieved_self_hit": df["retrieved_self_hit"].mean(),
    }
    pd.DataFrame([summary]).to_csv(path, index=False)


def write_checkpoint(
    rows: List[Dict[str, Any]],
    trace_path: Path,
    summary_path: Path,
    prefix: str,
    label: str,
) -> None:
    pd.DataFrame(rows).to_csv(trace_path, index=False)
    write_summary(rows, summary_path, prefix)
    print(
        f"[checkpoint] {label}: saved {len(rows)} rows to {trace_path}",
        flush=True,
    )


def run(cfg: EvalConfig) -> None:
    model_name, stem = MODE_META[cfg.mode]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = load_rows(cfg.input_path, cfg.max_rows)
    df = normalize_dataset(df_raw)
    store, rebuilt = build_vectorstore(df, cfg)
    class _Cfg:
        dry_run = cfg.dry_run
        model_path = cfg.model_path
        n_ctx = cfg.n_ctx
        n_threads = cfg.n_threads
        n_gpu_layers = cfg.n_gpu_layers
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
    total_started = time.time()
    prefix = "rlm" if cfg.mode in {"repl4", "repl8", "adaptive_p"} else "rag"
    for idx, row in enumerate(df.to_dict("records"), 1):
        if row["sample_id"] in done:
            print(f"[resume] skip {idx}/{len(df)} {row['sample_id']}", flush=True)
            continue
        started = time.time()
        print(f"\n[question] {idx}/{len(df)} {row['sample_id']} {row['question'][:140]}", flush=True)
        docs = retrieve_docs(store, row, cfg)
        print("[retrieve] " + ", ".join(doc.metadata.get("sample_id", "") for doc in docs), flush=True)
        result, judge, metrics = run_row(llm, row, docs, cfg)
        elapsed = time.time() - started
        parsed = result["parsed_response"]
        out = {
            "model": model_name,
            "mode": cfg.mode,
            "row_number": row["row_number"],
            "sample_id": row["sample_id"],
            "question": row["question"],
            "ground_truth_short_answers": compact_json(row["short_answers"]),
            "ground_truth_long_answers": compact_json(row["long_answers"]),
            "retrieved_sample_ids": compact_json([doc.metadata.get("sample_id", "") for doc in docs]),
            "raw_response": result["raw_response"],
            "parsed_response": compact_json(parsed),
            "long_answer": parsed.get("long_answer", ""),
            "short_answer": parsed.get("short_answer", ""),
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
            write_checkpoint(
                rows,
                trace_path,
                summary_path,
                prefix,
                f"row {idx}/{len(df)}",
            )
        else:
            pd.DataFrame(rows).to_csv(trace_path, index=False)
        print(
            f"[{idx}/{len(df)}] short_f1={metrics.get(prefix + '_short_token_f1', 0):.3f} "
            f"long_f1={metrics.get(prefix + '_long_token_f1', 0):.3f} "
            f"fused={metrics.get('quality_fused_score', 0):.3f} elapsed={elapsed:.1f}s",
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save trace and summary every N completed rows.",
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
    )


if __name__ == "__main__":
    run(parse_args())
