#!/usr/bin/env python3
"""Run Natural Questions through a local LangGraph + Chroma RAG evaluator.

Outputs are written to /Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs
by default. If llama_cpp fails with a Metal command queue error inside Codex,
run this script from a normal macOS terminal or pass --model-path to a GGUF that
your local llama_cpp runtime can open.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypedDict

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, StateGraph
from llama_cpp import Llama

try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover - depends on local LangChain packaging
    from langchain_community.vectorstores import Chroma


DEFAULT_DATA = Path(
    "/Users/htet/Desktop/Projects/X-RLM/Data/NaturalQuestions/train/train.parquet"
)
DEFAULT_OUT_DIR = Path("outputs/nq_langgraph_rag")
PROJECT_OUT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs")
P_COUNTER_ENV = "P_NUMERIC_VALUE"
P_TEXT_ENV = "P_TEXT_VALUE"




PAPER_SOURCES = (
    "Are More LM Calls All You Need?; Faithfulness metric fusion; "
    "Agentic Reasoning for LLMs; Recursive Introspection; "
    "Beyond neural scaling laws/data pruning; Recursive Language Models"
)


def _paper_float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def add_paper_columns(
    out_row: Dict[str, Any],
    metric_prefix: str,
    method_mapping: str,
    principles: str,
    estimated_lm_calls: int,
) -> None:
    judge = _paper_float(out_row.get("llm_judge_score_0_to_5"))
    judge_norm = judge / 5.0 if judge is not None else None
    candidates = [
        _paper_float(out_row.get(f"{metric_prefix}_short_token_f1")),
        _paper_float(out_row.get(f"{metric_prefix}_long_token_f1")),
        _paper_float(out_row.get(f"{metric_prefix}_short_rouge1")),
        _paper_float(out_row.get(f"{metric_prefix}_long_rouge1")),
        judge_norm,
    ]
    values = [value for value in candidates if value is not None]
    fused = sum(values) / len(values) if values else None
    out_row["paper_sources"] = PAPER_SOURCES
    out_row["paper_method_mapping"] = method_mapping
    out_row["paper_principles_applied"] = principles
    out_row["estimated_lm_calls"] = estimated_lm_calls
    out_row["compound_call_warning"] = (
        "More LM calls can help easy questions but hurt hard questions; compare call budget against quality."
        if estimated_lm_calls > 2
        else "Single answer call plus optional judge; low compound-call risk."
    )
    out_row["faithfulness_fused_score"] = round(fused, 6) if fused is not None else ""
    out_row["faithfulness_fused_explanation"] = (
        "Mean of available short/long F1, short/long ROUGE-1, and normalized LLM judge score."
    )


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings, so the vector DB works without downloads."""

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
            sign = 1.0 if digest[4] % 2 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec.tolist()


class RagState(TypedDict, total=False):
    row: Dict[str, Any]
    question: str
    retrieved: List[Document]
    prompt: str
    raw_response: str
    parsed_response: Dict[str, Any]
    metrics: Dict[str, Any]
    trace: List[Dict[str, Any]]


@dataclass
class EvalConfig:
    input_path: Path
    out_dir: Path
    results_csv: str
    summary_csv: str
    profile_csv: str
    max_rows: int | None
    top_k: int
    temperature: float
    max_tokens: int
    judge_max_tokens: int
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    model_path: Path | None
    use_genetic_prompt: bool
    population_size: int
    generations: int
    keep_top: int
    random_seed: int
    skip_llm_judge: bool
    rebuild_index: bool
    dry_run: bool


def safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [safe_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, np.ndarray)) else False:
        return ""
    return value


def compact_json(value: Any) -> str:
    return json.dumps(safe_value(value), ensure_ascii=False)


def first_text(value: Any) -> str:
    value = safe_value(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                return first_text(parsed)
            except Exception:
                return stripped
        return stripped
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def list_text(value: Any) -> List[str]:
    value = safe_value(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return list_text(ast.literal_eval(stripped))
            except Exception:
                pass
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def load_rows(input_path: Path, max_rows: int | None) -> pd.DataFrame:
    print(f"[setup] Loading dataset: {input_path}", flush=True)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".json":
        df = pd.read_json(input_path, lines=True)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input file: {input_path}")
    if max_rows:
        df = df.head(max_rows)
    print(f"[setup] Loaded {len(df)} rows and {len(df.columns)} columns", flush=True)
    return df


def ground_truth(row: Dict[str, Any]) -> Dict[str, List[str]]:
    gt = safe_value(row.get("ground_truth", {}))
    if isinstance(gt, str):
        try:
            gt = ast.literal_eval(gt)
        except Exception:
            gt = {}
    if not isinstance(gt, dict):
        gt = {}
    return {
        "long_answers": list_text(gt.get("long_answers", [])),
        "short_answers": list_text(gt.get("short_answers", [])),
    }


def make_documents(df: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []
    for row in df.to_dict("records"):
        gt = ground_truth(row)
        question = first_text(row.get("question", ""))
        page_content = "\n".join(
            [
                f"Question: {question}",
                "Short answers: " + "; ".join(gt["short_answers"]),
                "Long answers: " + "\n".join(gt["long_answers"]),
            ]
        ).strip()
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "sample_id": first_text(row.get("sample_id", "")),
                    "source": "; ".join(list_text(row.get("source", []))),
                    "topic": first_text(row.get("topic", "")),
                    "domain": first_text(row.get("domain", "")),
                },
            )
        )
    return docs


def build_vectorstore(df: pd.DataFrame, out_dir: Path, rebuild: bool) -> Chroma:
    persist_dir = out_dir / "chroma_nq"
    print(f"[vector-db] Preparing Chroma index: {persist_dir}", flush=True)
    if rebuild and persist_dir.exists():
        import shutil

        print("[vector-db] Rebuilding existing index", flush=True)
        shutil.rmtree(persist_dir)
    embeddings = HashEmbeddings()
    docs = make_documents(df)
    ids = [doc.metadata["sample_id"] or f"row_{i}" for i, doc in enumerate(docs)]
    if persist_dir.exists():
        print("[vector-db] Reusing existing index", flush=True)
        return Chroma(
            collection_name="natural_questions_train",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
    print(f"[vector-db] Creating index with {len(docs)} documents", flush=True)
    return Chroma.from_documents(
        docs,
        embeddings,
        ids=ids,
        collection_name="natural_questions_train",
        persist_directory=str(persist_dir),
    )


def normalize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_counts(prediction: str, references: Sequence[str]) -> Dict[str, Any]:
    pred_tokens = normalize(prediction)
    if not pred_tokens or not references:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "matched_tokens": [],
            "missing_reference_tokens": [],
            "extra_prediction_tokens": pred_tokens,
            "best_reference": "",
        }
    best: Dict[str, Any] = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "matched_tokens": [],
        "missing_reference_tokens": [],
        "extra_prediction_tokens": pred_tokens,
        "best_reference": "",
    }
    for ref in references:
        ref_tokens = normalize(ref)
        if not ref_tokens:
            continue
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        overlap = pred_counter & ref_counter
        common = sum(overlap.values())
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0
        f1 = 0.0 if not common else 2 * precision * recall / (precision + recall)
        if f1 >= best["f1"]:
            matched = list(overlap.elements())
            missing = list((ref_counter - pred_counter).elements())
            extra = list((pred_counter - ref_counter).elements())
            best = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "matched_tokens": matched,
                "missing_reference_tokens": missing,
                "extra_prediction_tokens": extra,
                "best_reference": ref,
            }
    return best


def token_f1(prediction: str, references: Sequence[str]) -> float:
    return float(token_counts(prediction, references)["f1"])


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu_score(prediction: str, references: Sequence[str], max_n: int = 4) -> float:
    pred_tokens = normalize(prediction)
    ref_token_lists = [normalize(ref) for ref in references if normalize(ref)]
    if not pred_tokens or not ref_token_lists:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = ngrams(pred_tokens, n)
        total = sum(pred_ngrams.values())
        if not total:
            precisions.append(1e-9)
            continue
        max_ref_counts: Counter = Counter()
        for ref_tokens in ref_token_lists:
            max_ref_counts |= ngrams(ref_tokens, n)
        overlap = pred_ngrams & max_ref_counts
        precisions.append(max(sum(overlap.values()) / total, 1e-9))
    pred_len = len(pred_tokens)
    closest_ref_len = min(
        (len(ref_tokens) for ref_tokens in ref_token_lists),
        key=lambda ref_len: (abs(ref_len - pred_len), ref_len),
    )
    bp = 1.0 if pred_len > closest_ref_len else math.exp(1 - closest_ref_len / pred_len)
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def rouge_n(prediction: str, references: Sequence[str], n: int) -> float:
    pred_tokens = normalize(prediction)
    if not pred_tokens:
        return 0.0
    pred_ngrams = ngrams(pred_tokens, n)
    if not pred_ngrams:
        return 0.0
    best = 0.0
    for ref in references:
        ref_ngrams = ngrams(normalize(ref), n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        denom = sum(ref_ngrams.values())
        if denom:
            best = max(best, overlap / denom)
    return float(best)


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, references: Sequence[str]) -> float:
    pred_tokens = normalize(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize(ref)
        if ref_tokens:
            best = max(best, lcs_len(pred_tokens, ref_tokens) / len(ref_tokens))
    return float(best)


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
        f"{prefix}_matched_tokens": compact_json(counts["matched_tokens"]),
        f"{prefix}_missing_reference_tokens": compact_json(counts["missing_reference_tokens"]),
        f"{prefix}_extra_prediction_tokens": compact_json(counts["extra_prediction_tokens"]),
        f"{prefix}_best_reference": counts["best_reference"],
    }


def parse_response(raw: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"long_answer": raw.strip(), "short_answer": "", "confidence": None}


def build_prompt(question: str, docs: Sequence[Document]) -> str:
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"[{i}] sample_id={doc.metadata.get('sample_id', '')} "
            f"source={doc.metadata.get('source', '')}\n{doc.page_content}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "Use the retrieved Natural Questions evidence to answer the question. "
        "Return only valid JSON with keys long_answer, short_answer, confidence, and evidence_sample_ids.\n\n"
        f"Question: {question}\n\nRetrieved evidence:\n{context}\n"
    )


def ask_llm(llm: Llama, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def store_p_in_environment(text: str) -> int:
    numeric_value = int(os.environ.get(P_COUNTER_ENV, "0")) + 1
    os.environ[P_COUNTER_ENV] = str(numeric_value)
    os.environ[P_TEXT_ENV] = text
    return numeric_value


def get_prompt_environment_block() -> str:
    return (
        "External prompt environment:\n"
        f"- P_NUMERIC_VALUE={os.environ.get(P_COUNTER_ENV, '0')}\n"
        f"- P_TEXT_VALUE={os.environ.get(P_TEXT_ENV, '')}"
    )


def seed_prompts(user_request: str, base_prompt: str) -> List[str]:
    env_block = get_prompt_environment_block()
    return [
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Do not mention P_TEXT_VALUE or the environment. Answer clearly and directly.\n\n{base_prompt}",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Give the best concise answer using the retrieved evidence.\n\n{base_prompt}",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Act as a careful expert assistant and answer with useful detail.\n\n{base_prompt}",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Rewrite it internally into a precise task, then answer only the user.\n\n{base_prompt}",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Give a practical high-quality response. Include evidence IDs when useful.\n\n{base_prompt}",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Find the user's intent and answer naturally.\n\n{base_prompt}",
    ]


def mutate_prompt(prompt: str) -> str:
    mutations = [
        "Be specific and avoid vague language.",
        "Use a clean structure with short paragraphs.",
        "Check assumptions before giving the answer.",
        "Prefer practical evidence-backed answers over theory.",
        "Return valid JSON only.",
        "Make the short answer brief and exact.",
        "Mention risks, limits, or missing evidence only if useful.",
        "Do not mention P_TEXT_VALUE, P_NUMERIC_VALUE, or the external environment.",
        "Answer naturally as if replying directly to the user.",
    ]
    action = random.choice(["append", "prepend", "replace"])
    if action == "append":
        return f"{prompt}\n\nExtra instruction: {random.choice(mutations)}"
    if action == "prepend":
        return f"{random.choice(mutations)}\n\n{prompt}"
    lines = prompt.splitlines()
    if not lines:
        return prompt
    lines[random.randrange(len(lines))] = random.choice(mutations)
    return "\n".join(lines)


def crossover_prompt(parent_a: str, parent_b: str) -> str:
    a_lines = parent_a.splitlines()
    b_lines = parent_b.splitlines()
    a_cut = max(1, len(a_lines) // 2)
    b_cut = max(1, len(b_lines) // 2)
    return "\n".join(a_lines[:a_cut] + b_lines[b_cut:])


def score_prompt(llm: Llama, user_request: str, candidate_prompt: str, max_tokens: int) -> float:
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict prompt quality judge. Score how well the candidate "
                "prompt will answer the user's request using retrieved evidence. Prefer "
                "prompts that answer naturally, return valid JSON, and do not expose "
                "internal variables. Return only a number from 0 to 100."
            ),
        },
        {
            "role": "user",
            "content": f"User request:\n{user_request}\n\nCandidate prompt:\n{candidate_prompt}",
        },
    ]
    raw_score = ask_llm(llm, judge_messages, temperature=0.1, max_tokens=max_tokens)
    match = re.search(r"\d+(?:\.\d+)?", raw_score)
    if not match:
        return 0.0
    return max(0.0, min(100.0, float(match.group())))


def evolve_prompt(
    llm: Llama,
    user_request: str,
    base_prompt: str,
    cfg: EvalConfig,
) -> tuple[float, str, List[Dict[str, Any]]]:
    store_p_in_environment(user_request)
    population = seed_prompts(user_request, base_prompt)[: cfg.population_size]
    while len(population) < cfg.population_size:
        population.append(mutate_prompt(random.choice(population)))
    history: List[Dict[str, Any]] = []
    for generation in range(1, cfg.generations + 1):
        print(f"[genetic] Generation {generation}/{cfg.generations}", flush=True)
        scored = []
        for index, prompt in enumerate(population, start=1):
            score = score_prompt(llm, user_request, prompt, max_tokens=16)
            scored.append((score, prompt))
            history.append(
                {
                    "generation": generation,
                    "candidate": index,
                    "score": score,
                    "prompt_preview": prompt.replace("\n", " ")[:240],
                }
            )
            print(f"[genetic] G{generation} P{index} score={score:.1f}", flush=True)
        scored.sort(reverse=True, key=lambda item: item[0])
        survivors = [prompt for _, prompt in scored[: cfg.keep_top]]
        next_population = survivors[:]
        while len(next_population) < cfg.population_size:
            parent_a, parent_b = random.sample(survivors, 2)
            next_population.append(mutate_prompt(crossover_prompt(parent_a, parent_b)))
        population = next_population
    final_scored = [(score_prompt(llm, user_request, prompt, max_tokens=16), prompt) for prompt in population]
    final_scored.sort(reverse=True, key=lambda item: item[0])
    return final_scored[0][0], final_scored[0][1], history


def make_graph(vectorstore: Chroma, llm: Llama | None, cfg: EvalConfig):
    def retrieve_node(state: RagState) -> RagState:
        docs = vectorstore.similarity_search(state["question"], k=cfg.top_k)
        trace = state.get("trace", []) + [
            {
                "node": "retrieve",
                "top_k": cfg.top_k,
                "sample_ids": [doc.metadata.get("sample_id", "") for doc in docs],
            }
        ]
        return {**state, "retrieved": docs, "trace": trace}

    def prompt_node(state: RagState) -> RagState:
        prompt = build_prompt(state["question"], state["retrieved"])
        evolved_prompt = prompt
        evolved_score: float | str = ""
        evolved_history: List[Dict[str, Any]] = []
        if cfg.use_genetic_prompt and not cfg.dry_run:
            assert llm is not None
            evolved_score, evolved_prompt, evolved_history = evolve_prompt(
                llm,
                state["question"],
                prompt,
                cfg,
            )
        elif cfg.use_genetic_prompt:
            evolved_history = [{"dry_run": True, "note": "Genetic prompt evolution skipped in dry run."}]
        p_environment = {
            P_COUNTER_ENV: os.environ.get(P_COUNTER_ENV, ""),
            P_TEXT_ENV: os.environ.get(P_TEXT_ENV, ""),
            "environment_block": get_prompt_environment_block(),
        }
        trace = state.get("trace", []) + [
            {
                "node": "prompt",
                "chars": len(evolved_prompt),
                "prompt": evolved_prompt,
                "base_prompt": prompt,
                "evolved_prompt_score": evolved_score,
                "evolved_prompt_history": evolved_history,
                "p_environment": p_environment,
            }
        ]
        return {
            **state,
            "prompt": evolved_prompt,
            "evolved_prompt": evolved_prompt,
            "evolved_prompt_score": evolved_score,
            "evolved_prompt_history": evolved_history,
            "p_environment": p_environment,
            "trace": trace,
        }

    def llm_node(state: RagState) -> RagState:
        if cfg.dry_run:
            first_doc = state["retrieved"][0].page_content if state["retrieved"] else ""
            raw = json.dumps(
                {
                    "long_answer": first_doc[:700],
                    "short_answer": "",
                    "confidence": 0.0,
                    "evidence_sample_ids": [
                        doc.metadata.get("sample_id", "") for doc in state["retrieved"]
                    ],
                }
            )
        else:
            assert llm is not None
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful local AI assistant.",
                    },
                    {"role": "user", "content": state["prompt"]},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            raw = response["choices"][0]["message"]["content"]
        parsed = parse_response(raw)
        trace = state.get("trace", []) + [
            {"node": "llm", "raw_response": raw, "parsed_response": parsed}
        ]
        return {**state, "raw_response": raw, "parsed_response": parsed, "trace": trace}

    def assess_node(state: RagState) -> RagState:
        gt = ground_truth(state["row"])
        parsed = state.get("parsed_response", {})
        long_answer = str(parsed.get("long_answer", ""))
        short_answer = str(parsed.get("short_answer", ""))
        baseline = safe_value(state["row"].get("llm_response", {}))
        if isinstance(baseline, str):
            try:
                baseline = ast.literal_eval(baseline)
            except Exception:
                baseline = {"long_answer": baseline, "short_answer": ""}
        if not isinstance(baseline, dict):
            baseline = {"long_answer": compact_json(baseline), "short_answer": ""}
        baseline_long = str(baseline.get("long_answer", ""))
        baseline_short = str(baseline.get("short_answer", ""))
        short_refs = gt["short_answers"]
        long_refs = gt["long_answers"]
        short_norm = " ".join(normalize(short_answer))
        ref_norms = [" ".join(normalize(ref)) for ref in short_refs]
        metrics = {
            "short_exact_match": int(bool(short_norm) and short_norm in ref_norms),
            "short_contains_any_gt": int(
                any(ref and ref in short_answer.lower() for ref in [r.lower() for r in short_refs])
            ),
            "short_token_f1": token_f1(short_answer, short_refs),
            "long_token_f1": token_f1(long_answer, long_refs),
            "retrieved_self_hit": int(
                first_text(state["row"].get("sample_id", ""))
                in [doc.metadata.get("sample_id", "") for doc in state["retrieved"]]
            ),
        }
        metrics.update(automatic_metrics(short_answer, short_refs, "rag_short"))
        metrics.update(automatic_metrics(long_answer, long_refs, "rag_long"))
        metrics.update(automatic_metrics(baseline_short, short_refs, "baseline_short"))
        metrics.update(automatic_metrics(baseline_long, long_refs, "baseline_long"))
        trace = state.get("trace", []) + [{"node": "assess", "metrics": metrics}]
        return {**state, "metrics": metrics, "trace": trace}

    def judge_node(state: RagState) -> RagState:
        if cfg.skip_llm_judge:
            judge = {"skipped": True}
        else:
            gt = ground_truth(state["row"])
            parsed = state.get("parsed_response", {})
            judge_prompt = (
                "You are grading a question-answering response. Compare the response "
                "to the reference answers. Return only valid JSON with keys "
                "score_0_to_5, verdict, token_level_notes, missing_facts, hallucinations.\n\n"
                f"Question: {state['question']}\n"
                f"Reference short answers: {json.dumps(gt['short_answers'], ensure_ascii=False)}\n"
                f"Reference long answers: {json.dumps(gt['long_answers'], ensure_ascii=False)}\n"
                f"Model response: {json.dumps(parsed, ensure_ascii=False)}\n"
            )
            if cfg.dry_run:
                judge = {
                    "score_0_to_5": None,
                    "verdict": "dry_run",
                    "token_level_notes": "",
                    "missing_facts": [],
                    "hallucinations": [],
                }
            else:
                assert llm is not None
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict but fair LLM-as-judge evaluator.",
                        },
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=cfg.judge_max_tokens,
                )
                judge = parse_response(response["choices"][0]["message"]["content"])
        trace = state.get("trace", []) + [{"node": "llm_judge", "judge": judge}]
        metrics = {**state.get("metrics", {}), "llm_judge_json": compact_json(judge)}
        if isinstance(judge, dict):
            metrics.update(
                {
                    "llm_judge_score_0_to_5": judge.get("score_0_to_5", ""),
                    "llm_judge_verdict": judge.get("verdict", ""),
                    "llm_judge_token_level_notes": judge.get("token_level_notes", ""),
                    "llm_judge_missing_facts": compact_json(judge.get("missing_facts", [])),
                    "llm_judge_hallucinations": compact_json(judge.get("hallucinations", [])),
                }
            )
        return {**state, "metrics": metrics, "trace": trace}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("prompt", prompt_node)
    graph.add_node("llm", llm_node)
    graph.add_node("assess", assess_node)
    graph.add_node("llm_judge", judge_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "prompt")
    graph.add_edge("prompt", "llm")
    graph.add_edge("llm", "assess")
    graph.add_edge("assess", "llm_judge")
    graph.add_edge("llm_judge", END)
    return graph.compile()


def load_llm(cfg: EvalConfig) -> Llama | None:
    if cfg.dry_run:
        print("[llm] Dry run enabled; skipping Llama model load", flush=True)
        return None
    if cfg.model_path:
        print(f"[llm] Loading local GGUF: {cfg.model_path}", flush=True)
        return Llama(
            model_path=str(cfg.model_path),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
    print(
        "[llm] Loading QuantFactory/Meta-Llama-3-8B-Instruct-GGUF "
        "via Llama.from_pretrained",
        flush=True,
    )
    return Llama.from_pretrained(
        repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        n_ctx=cfg.n_ctx,
        n_threads=cfg.n_threads,
        n_gpu_layers=cfg.n_gpu_layers,
        verbose=False,
    )


def write_profile(df: pd.DataFrame, out_dir: Path, profile_csv: str) -> None:
    print("[output] Writing dataset profile", flush=True)
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": ", ".join(df.columns),
        "dataset_names": ", ".join(sorted(set(map(str, df.get("dataset_name", []))))),
        "domains": ", ".join(sorted(set(map(str, df.get("domain", []))))),
        "languages": ", ".join(sorted(set(map(str, df.get("lang", []))))),
        "topics": ", ".join(sorted(set(map(str, df.get("topic", []))))),
    }
    pd.DataFrame([profile]).to_csv(out_dir / profile_csv, index=False)


def run(cfg: EvalConfig) -> None:
    print("[start] Natural Questions LangGraph RAG assessment", flush=True)
    print(f"[start] Output directory: {cfg.out_dir}", flush=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.random_seed)
    df = load_rows(cfg.input_path, cfg.max_rows)
    write_profile(df, cfg.out_dir, cfg.profile_csv)
    vectorstore = build_vectorstore(df, cfg.out_dir, cfg.rebuild_index)
    llm = load_llm(cfg)
    print("[graph] Compiling LangGraph workflow", flush=True)
    app = make_graph(vectorstore, llm, cfg)

    rows: List[Dict[str, Any]] = []
    total_started = time.time()
    print(f"[run] Evaluating {len(df)} questions", flush=True)
    for index, row in enumerate(df.to_dict("records"), start=1):
        question = first_text(row.get("question", ""))
        started = time.time()
        result = app.invoke({"row": row, "question": question, "trace": []})
        gt = ground_truth(row)
        parsed = result.get("parsed_response", {})
        baseline = safe_value(row.get("llm_response", {}))
        retrieved_ids = [doc.metadata.get("sample_id", "") for doc in result.get("retrieved", [])]
        out_row = {
            "row_number": index,
            "experiment_name": "GRLM + RAG",
            "row_explanation": (
                "One Natural Questions item evaluated with genetic prompt search, "
                "RAG evidence, automatic metrics, and LLM-as-judge."
            ),
            "sample_id": first_text(row.get("sample_id", "")),
            "question": question,
            "answer_summary": (
                f"short={parsed.get('short_answer', '')} | "
                f"long={str(parsed.get('long_answer', ''))[:240]}"
            ),
            "ground_truth_short_answers": compact_json(gt["short_answers"]),
            "ground_truth_long_answers": compact_json(gt["long_answers"]),
            "baseline_llm_response": compact_json(baseline),
            "rag_long_answer": parsed.get("long_answer", ""),
            "rag_short_answer": parsed.get("short_answer", ""),
            "rag_confidence": parsed.get("confidence", ""),
            "raw_llm_response": result.get("raw_response", ""),
            "prompt": result.get("prompt", ""),
            "use_genetic_prompt": cfg.use_genetic_prompt,
            "evolved_prompt_score": result.get("evolved_prompt_score", ""),
            "evolved_prompt_history_json": compact_json(result.get("evolved_prompt_history", [])),
            "p_environment_json": compact_json(result.get("p_environment", {})),
            "p_summary": (
                f"P_NUMERIC_VALUE={result.get('p_environment', {}).get(P_COUNTER_ENV, '')}; "
                f"P_TEXT_VALUE={str(result.get('p_environment', {}).get(P_TEXT_ENV, ''))[:160]}"
            ),
            "retrieval_summary": (
                f"top_k={cfg.top_k}; retrieved_sample_ids={', '.join(retrieved_ids)}"
            ),
            "retrieved_sample_ids": compact_json(retrieved_ids),
            "retrieved_context": "\n\n".join(
                doc.page_content for doc in result.get("retrieved", [])
            ),
            "trace_summary": (
                "retrieve -> genetic prompt evolution candidates -> prompt -> llm -> "
                "assess -> llm_judge"
            ),
            "trace_json": json.dumps(result.get("trace", []), ensure_ascii=False),
            "elapsed_seconds": round(time.time() - started, 3),
            **result.get("metrics", {}),
        }
        out_row["elapsed_display"] = f"elapsed={out_row['elapsed_seconds']:.3f}s"
        out_row["judge_summary"] = (
            f"score={out_row.get('llm_judge_score_0_to_5', '')}; "
            f"verdict={out_row.get('llm_judge_verdict', '')}; "
            f"notes={str(out_row.get('llm_judge_token_level_notes', ''))[:240]}"
        )
        genetic_calls = (
            (cfg.population_size * cfg.generations) + cfg.population_size
            if cfg.use_genetic_prompt and not cfg.dry_run
            else 0
        )
        add_paper_columns(
            out_row,
            metric_prefix="rag",
            method_mapping="Compound LM calls for prompt search, then RAG answer and judge.",
            principles=(
                "Applies compound inference scaling ideas from more-LM-calls paper; "
                "faithfulness metric fusion combines metrics; agentic search explores prompt candidates."
            ),
            estimated_lm_calls=genetic_calls + 1 + (0 if cfg.skip_llm_judge else 1),
        )
        rows.append(out_row)
        print(
            f"[{index}/{len(df)}] {out_row['sample_id']} "
            f"short_f1={out_row['short_token_f1']:.3f} "
            f"long_f1={out_row['long_token_f1']:.3f} "
            f"elapsed={out_row['elapsed_seconds']:.3f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results_path = cfg.out_dir / cfg.results_csv
    print(f"[output] Writing detailed traces CSV: {results_path}", flush=True)
    results.to_csv(results_path, index=False)
    metric_cols = [
        "short_exact_match",
        "short_contains_any_gt",
        "rag_short_token_f1",
        "rag_long_token_f1",
        "rag_short_bleu",
        "rag_long_bleu",
        "rag_short_rouge1",
        "rag_long_rouge1",
        "baseline_short_token_f1",
        "baseline_long_token_f1",
        "llm_judge_score_0_to_5",
        "retrieved_self_hit",
        "estimated_lm_calls",
        "faithfulness_fused_score",
        "elapsed_seconds",
    ]
    metric_cols = [col for col in metric_cols if col in results.columns]
    results[metric_cols] = results[metric_cols].apply(pd.to_numeric, errors="coerce")
    summary = results[metric_cols].mean(numeric_only=True).to_frame("mean").T
    summary.insert(0, "rows_evaluated", len(results))
    summary.insert(1, "method", "GENETIC_PROMPT_RAG")
    summary.insert(2, "summary_explanation", "Mean metrics for genetic prompt RAG with P environment traces.")
    if "elapsed_seconds" in summary.columns:
        summary["mean_elapsed_display"] = summary["elapsed_seconds"].apply(
            lambda value: f"elapsed={value:.3f}s" if pd.notna(value) else ""
        )
    summary_path = cfg.out_dir / cfg.summary_csv
    print(f"[output] Writing summary CSV: {summary_path}", flush=True)
    summary.to_csv(summary_path, index=False)
    print(f"[done] Total elapsed: {time.time() - total_started:.2f}s", flush=True)
    print(f"[done] Wrote: {results_path}", flush=True)
    print(f"[done] Wrote: {summary_path}", flush=True)
    print(f"[done] Wrote: {cfg.out_dir / cfg.profile_csv}", flush=True)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_OUT_DIR)
    parser.add_argument("--results-csv", default="nq_genetic_rag_llama_assessment_traces.csv")
    parser.add_argument("--summary-csv", default="nq_genetic_rag_llama_summary.csv")
    parser.add_argument("--profile-csv", default="nq_genetic_dataset_profile.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=320)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local .gguf path. If omitted, Llama.from_pretrained is used.",
    )
    parser.add_argument("--use-genetic-prompt", action="store_true")
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--keep-top", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the vector DB and CSV traces without loading the Llama model.",
    )
    parser.add_argument("--skip-llm-judge", action="store_true")
    args = parser.parse_args()
    if args.population_size < 2:
        parser.error("--population-size must be at least 2")
    if args.keep_top < 2:
        parser.error("--keep-top must be at least 2")
    if args.keep_top > args.population_size:
        parser.error("--keep-top cannot be larger than --population-size")
    return EvalConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        results_csv=args.results_csv,
        summary_csv=args.summary_csv,
        profile_csv=args.profile_csv,
        max_rows=args.max_rows,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        model_path=args.model_path,
        use_genetic_prompt=args.use_genetic_prompt,
        population_size=args.population_size,
        generations=args.generations,
        keep_top=args.keep_top,
        random_seed=args.random_seed,
        skip_llm_judge=args.skip_llm_judge,
        rebuild_index=args.rebuild_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    run(parse_args())
