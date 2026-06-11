#!/usr/bin/env python3
"""Paper-applied Recursive Language Model REPL scaffold for Natural Questions.

This follows the REPL-style RLM loop from the paper:

state <- InitREPL(prompt=P)
state <- AddFunction(state, sub_RLM)
hist <- [Metadata(state)]
while True:
    code <- LLM(hist)
    state, stdout <- REPL(state, code)
    hist <- hist || code || Metadata(stdout)
    if state[Final] is set:
        return state[Final]

The REPL is intentionally safe and narrow. It accepts JSON actions or simple
function-like calls for: find_snippets, batch_classify_snippets, sub_RLM,
root_aggregate, and FINAL_VAR. All code/stdout/history is saved to CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from llama_cpp import Llama

from nq_langgraph_rag_eval import (
    DEFAULT_DATA,
    automatic_metrics,
    build_vectorstore,
    compact_json,
    first_text,
    ground_truth,
    load_rows,
    load_llm,
    parse_response,
    safe_value,
    token_f1,
)




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


PROJECT_OUT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_outputs")


@dataclass
class RlmConfig:
    input_path: Path
    out_dir: Path
    max_rows: int | None
    top_k: int
    subcall_max_tokens: int
    repl_max_steps: int
    repl_code_max_tokens: int
    final_max_tokens: int
    judge_max_tokens: int
    temperature: float
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    model_path: Path | None
    rebuild_index: bool
    skip_llm_judge: bool
    dry_run: bool


def ask_llm(
    llm: Llama,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def load_rlm_llm(cfg: RlmConfig) -> Llama | None:
    class _Adapter:
        dry_run = cfg.dry_run
        model_path = cfg.model_path
        n_ctx = cfg.n_ctx
        n_threads = cfg.n_threads
        n_gpu_layers = cfg.n_gpu_layers

    return load_llm(_Adapter())


def make_environment(question: str, docs: list[Any]) -> Dict[str, Any]:
    snippets = []
    for index, doc in enumerate(docs, start=1):
        snippets.append(
            {
                "handle": f"P.snippets[{index - 1}]",
                "snippet_id": index,
                "sample_id": doc.metadata.get("sample_id", ""),
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
                "chars": len(doc.page_content),
            }
        )
    prompt = {
        "question": question,
        "snippets": snippets,
        "instruction": (
            "Answer using recursive sub-results over snippets. Return JSON with "
            "long_answer, short_answer, confidence, and evidence_sample_ids."
        ),
    }
    return {
        "P": prompt,
        "metadata": {
            "P_handle": "P",
            "question_chars": len(question),
            "snippet_count": len(snippets),
            "total_snippet_chars": sum(item["chars"] for item in snippets),
            "available_handles": [item["handle"] for item in snippets],
        },
    }


def subcall_prompt(question: str, snippet: Dict[str, Any]) -> str:
    return (
        "You are a recursive sub-call in an RLM scaffold. Inspect only this snippet "
        "and extract evidence relevant to the question. Return only valid JSON with "
        "keys snippet_id, sample_id, relevant, short_answer_candidate, "
        "long_answer_evidence, confidence.\n\n"
        f"Question: {question}\n\n"
        f"Snippet handle: {snippet['handle']}\n"
        f"Sample id: {snippet['sample_id']}\n"
        f"Source: {snippet['source']}\n\n"
        f"Snippet text:\n{snippet['text']}"
    )


def final_prompt(question: str, environment_metadata: Dict[str, Any], sub_results: List[Dict[str, Any]]) -> str:
    return (
        "You are the root RLM call. The full prompt/evidence is stored outside your "
        "context as variable P. You have only metadata plus recursive sub-call "
        "results. Aggregate the sub-results and answer the original question. "
        "Return only valid JSON with keys long_answer, short_answer, confidence, "
        "and evidence_sample_ids.\n\n"
        f"Question: {question}\n\n"
        f"External environment metadata:\n{json.dumps(environment_metadata, ensure_ascii=False)}\n\n"
        f"Recursive sub-call results:\n{json.dumps(sub_results, ensure_ascii=False)}"
    )


def metadata_for_repl(state: Dict[str, Any]) -> Dict[str, Any]:
    env = state["environment"]
    return {
        "P_handle": env["metadata"].get("P_handle", "P"),
        "available_handles": env["metadata"].get("available_handles", []),
        "snippet_count": env["metadata"].get("snippet_count", 0),
        "sub_result_count": len(state.get("sub_results", [])),
        "last_hits_count": len(state.get("last_hits", [])),
        "final_is_set": state.get("final") is not None,
        "allowed_functions": [
            "find_snippets(keyword, window=400, max_hits=5)",
            "batch_classify_snippets()",
            "sub_RLM(handles)",
            "root_aggregate()",
            "FINAL_VAR(value)",
        ],
    }


def build_repl_code_prompt(question: str, state: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    compact_history = history[-8:]
    return (
        "You are controlling a safe REPL for a Recursive Language Model. "
        "The full prompt and retrieved evidence are stored in external variable P. "
        "Choose exactly one next action. Return only valid JSON, no markdown.\n\n"
        "Allowed actions:\n"
        '{"action":"find_snippets","keyword":"...","window":400,"max_hits":5}\n'
        '{"action":"batch_classify_snippets"}\n'
        '{"action":"sub_RLM","handles":["P.snippets[0]"]}\n'
        '{"action":"root_aggregate"}\n'
        '{"action":"FINAL_VAR","value":{"long_answer":"...","short_answer":"...",'
        '"confidence":0.0,"evidence_sample_ids":[]}}\n\n'
        "Rules:\n"
        "- First search/classify if you are unsure which snippets matter.\n"
        "- Use sub_RLM before root_aggregate unless useful sub-results already exist.\n"
        "- Use root_aggregate to create the final answer from sub-results.\n"
        "- Use FINAL_VAR only when the answer JSON is already available.\n\n"
        f"Question:\n{question}\n\n"
        f"Current metadata:\n{json.dumps(metadata_for_repl(state), ensure_ascii=False)}\n\n"
        f"Recent REPL history:\n{json.dumps(compact_history, ensure_ascii=False)}"
    )


def parse_repl_code(code: str) -> Dict[str, Any]:
    text = str(code).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    find_match = re.search(r"find_snippets\(\s*['\"](.+?)['\"]", text, flags=re.S)
    if find_match:
        return {"action": "find_snippets", "keyword": find_match.group(1)}

    if "batch_classify_snippets" in text:
        return {"action": "batch_classify_snippets"}

    if "root_aggregate" in text:
        return {"action": "root_aggregate"}

    if "FINAL_VAR" in text:
        return {"action": "FINAL_VAR", "value": parse_response(text)}

    return {"action": "unknown", "raw_code": text}


def find_snippets_action(state: Dict[str, Any], keyword: str, window: int = 400, max_hits: int = 5) -> Dict[str, Any]:
    hits = []
    for snippet in state["environment"]["P"]["snippets"]:
        text = snippet.get("text", "")
        start = 0
        while True:
            idx = text.lower().find(str(keyword).lower(), start)
            if idx == -1:
                break
            left = max(0, idx - window)
            right = min(len(text), idx + len(str(keyword)) + window)
            hits.append(
                {
                    "handle": snippet["handle"],
                    "sample_id": snippet.get("sample_id", ""),
                    "keyword": keyword,
                    "snippet": text[left:right],
                }
            )
            if len(hits) >= max_hits:
                return {"hits": hits, "hit_count": len(hits)}
            start = idx + 1
    return {"hits": hits, "hit_count": len(hits)}


def batch_classify_snippets_action(question: str, state: Dict[str, Any]) -> Dict[str, Any]:
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    labels = []
    for snippet in state["environment"]["P"]["snippets"]:
        text_tokens = set(re.findall(r"[a-z0-9]+", snippet.get("text", "").lower()))
        overlap = sorted(question_tokens & text_tokens)
        score = len(overlap)
        labels.append(
            {
                "handle": snippet["handle"],
                "sample_id": snippet.get("sample_id", ""),
                "category": "likely_relevant" if score else "unknown",
                "overlap_score": score,
                "overlap_terms": overlap[:20],
            }
        )
    labels.sort(key=lambda item: item["overlap_score"], reverse=True)
    return {"classifications": labels}


def handles_from_action(action: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    handles = action.get("handles") or action.get("handle") or []
    if isinstance(handles, str):
        handles = [handles]
    if not handles:
        hits = state.get("last_hits", [])
        handles = [hit.get("handle", "") for hit in hits if hit.get("handle")]
    if not handles:
        labels = state.get("snippet_classifications", [])
        handles = [item["handle"] for item in labels[: state["top_k"]] if item.get("handle")]
    if not handles:
        handles = state["environment"]["metadata"].get("available_handles", [])
    return list(dict.fromkeys(handles))


def sub_rlm_one(
    llm: Llama | None,
    question: str,
    snippet: Dict[str, Any],
    cfg: RlmConfig,
) -> tuple[Dict[str, Any], str, str]:
    prompt = subcall_prompt(question, snippet)
    if cfg.dry_run:
        raw = json.dumps(
            {
                "snippet_id": snippet["snippet_id"],
                "sample_id": snippet["sample_id"],
                "relevant": True,
                "short_answer_candidate": "",
                "long_answer_evidence": snippet["text"][:700],
                "confidence": 0.0,
            }
        )
    else:
        assert llm is not None
        raw = ask_llm(
            llm,
            [
                {
                    "role": "system",
                    "content": "You are a precise recursive evidence extraction sub-call.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=cfg.subcall_max_tokens,
        )
    parsed = parse_response(raw)
    parsed.setdefault("snippet_id", snippet["snippet_id"])
    parsed.setdefault("sample_id", snippet["sample_id"])
    parsed.setdefault("handle", snippet["handle"])
    return parsed, raw, prompt


def root_aggregate_action(
    llm: Llama | None,
    question: str,
    state: Dict[str, Any],
    cfg: RlmConfig,
) -> tuple[Dict[str, Any], str, str]:
    root_prompt = final_prompt(
        question,
        state["environment"]["metadata"],
        state.get("sub_results", []),
    )
    if cfg.dry_run:
        raw = json.dumps(
            {
                "long_answer": " ".join(
                    str(item.get("long_answer_evidence", ""))
                    for item in state.get("sub_results", [])
                )[:900],
                "short_answer": "",
                "confidence": 0.0,
                "evidence_sample_ids": [
                    item.get("sample_id", "") for item in state.get("sub_results", [])
                ],
            }
        )
    else:
        assert llm is not None
        raw = ask_llm(
            llm,
            [
                {
                    "role": "system",
                    "content": "You are the root recursive language model aggregator.",
                },
                {"role": "user", "content": root_prompt},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.final_max_tokens,
        )
    return parse_response(raw), raw, root_prompt


def execute_repl_action(
    llm: Llama | None,
    question: str,
    state: Dict[str, Any],
    code: str,
    cfg: RlmConfig,
) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    action = parse_repl_code(code)
    name = str(action.get("action", "")).strip()
    stdout: Dict[str, Any] = {"action": name}
    trace_events: List[Dict[str, Any]] = []

    if name == "find_snippets":
        result = find_snippets_action(
            state,
            keyword=str(action.get("keyword", question.split()[0] if question.split() else "")),
            window=int(action.get("window", 400) or 400),
            max_hits=int(action.get("max_hits", 5) or 5),
        )
        state["last_hits"] = result["hits"]
        stdout.update(result)
        print(
            f"[repl-action] find_snippets keyword={action.get('keyword', '')} "
            f"hits={result['hit_count']}",
            flush=True,
        )

    elif name == "batch_classify_snippets":
        result = batch_classify_snippets_action(question, state)
        state["snippet_classifications"] = result["classifications"]
        stdout.update(result)
        top_label = result["classifications"][0] if result["classifications"] else {}
        print(
            "[repl-action] batch_classify_snippets "
            f"count={len(result['classifications'])} "
            f"top_handle={top_label.get('handle', '')} "
            f"top_score={top_label.get('overlap_score', '')}",
            flush=True,
        )

    elif name == "sub_RLM":
        handles = handles_from_action(action, state)
        print(
            f"[repl-action] sub_RLM handles={', '.join(handles)}",
            flush=True,
        )
        snippets = {
            snippet["handle"]: snippet for snippet in state["environment"]["P"]["snippets"]
        }
        new_results = []
        for handle in handles:
            snippet = snippets.get(handle)
            if not snippet:
                continue
            print(
                f"[sub_RLM] Calling evidence extractor handle={handle} "
                f"sample_id={snippet.get('sample_id', '')}",
                flush=True,
            )
            parsed, raw, prompt = sub_rlm_one(llm, question, snippet, cfg)
            state["sub_results"].append(parsed)
            new_results.append(parsed)
            print(
                f"[sub_RLM] Done handle={handle} "
                f"relevant={parsed.get('relevant', '')} "
                f"confidence={parsed.get('confidence', '')}",
                flush=True,
            )
            trace_events.append(
                {
                    "event": "sub_RLM",
                    "handle": handle,
                    "prompt_chars": len(prompt),
                    "raw_response": raw,
                    "parsed_response": parsed,
                }
            )
        stdout.update({"handles": handles, "new_sub_result_count": len(new_results)})

    elif name == "root_aggregate":
        if not state.get("sub_results"):
            print("[repl-action] root_aggregate requested with no sub_results; auto-running sub_RLM", flush=True)
            for snippet in state["environment"]["P"]["snippets"]:
                print(
                    f"[sub_RLM] Auto before root handle={snippet['handle']} "
                    f"sample_id={snippet.get('sample_id', '')}",
                    flush=True,
                )
                parsed, raw, prompt = sub_rlm_one(llm, question, snippet, cfg)
                state["sub_results"].append(parsed)
                trace_events.append(
                    {
                        "event": "sub_RLM_auto_before_root",
                        "handle": snippet["handle"],
                        "prompt_chars": len(prompt),
                        "raw_response": raw,
                        "parsed_response": parsed,
                    }
                )
        print(
            f"[root] Aggregating sub_results={len(state.get('sub_results', []))}",
            flush=True,
        )
        parsed_final, raw_final, root_prompt = root_aggregate_action(llm, question, state, cfg)
        state["root_prompt"] = root_prompt
        state["raw_response"] = raw_final
        state["parsed_response"] = parsed_final
        state["final"] = parsed_final
        stdout.update({"final_set": True, "parsed_response": parsed_final})
        trace_events.append(
            {
                "event": "root_aggregate",
                "prompt_chars": len(root_prompt),
                "raw_response": raw_final,
                "parsed_response": parsed_final,
            }
        )
        print(
            f"[root] Done short_answer={str(parsed_final.get('short_answer', ''))[:120]} "
            f"confidence={parsed_final.get('confidence', '')}",
            flush=True,
        )

    elif name == "FINAL_VAR":
        value = action.get("value", {})
        if not isinstance(value, dict):
            value = {"long_answer": str(value), "short_answer": "", "confidence": None}
        state["final"] = value
        state["parsed_response"] = value
        state["raw_response"] = json.dumps(value, ensure_ascii=False)
        stdout.update({"final_set": True, "parsed_response": value})
        print("[repl-action] FINAL_VAR set directly", flush=True)

    else:
        stdout.update(
            {
                "error": "Unknown action; expected find_snippets, batch_classify_snippets, sub_RLM, root_aggregate, or FINAL_VAR.",
                "raw_code": code,
            }
        )
        print(f"[repl-action] Unknown action raw={str(code)[:160]}", flush=True)

    return state, stdout, trace_events


def run_recursive_answer(
    llm: Llama | None,
    question: str,
    docs: list[Any],
    cfg: RlmConfig,
) -> Dict[str, Any]:
    env = make_environment(question, docs)
    state: Dict[str, Any] = {
        "environment": env,
        "sub_results": [],
        "last_hits": [],
        "snippet_classifications": [],
        "final": None,
        "top_k": cfg.top_k,
    }
    history: List[Dict[str, Any]] = [
        {
            "type": "metadata",
            "value": metadata_for_repl(state),
        }
    ]
    trace: List[Dict[str, Any]] = [
        {
            "node": "init_repl_environment",
            "paper_principle": "P is kept as an external variable; root sees metadata/handles.",
            "metadata": env["metadata"],
        }
    ]
    print(
        "[repl] InitREPL(P): "
        f"snippets={env['metadata']['snippet_count']} "
        f"chars={env['metadata']['total_snippet_chars']} "
        f"handles={', '.join(env['metadata']['available_handles'])}",
        flush=True,
    )

    fallback_actions = [
        {"action": "batch_classify_snippets"},
        {"action": "sub_RLM", "handles": env["metadata"]["available_handles"]},
        {"action": "root_aggregate"},
    ]
    code_call_count = 0
    for step in range(1, cfg.repl_max_steps + 1):
        if cfg.dry_run:
            action = fallback_actions[min(step - 1, len(fallback_actions) - 1)]
            code = json.dumps(action, ensure_ascii=False)
        else:
            assert llm is not None
            code_prompt = build_repl_code_prompt(question, state, history)
            print(
                f"[repl] Step {step}/{cfg.repl_max_steps}: asking LLM for next REPL action",
                flush=True,
            )
            code = ask_llm(
                llm,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the LLM in Algorithm 1 of a Recursive "
                            "Language Model. Return only the next safe REPL "
                            "action as JSON."
                        ),
                    },
                    {"role": "user", "content": code_prompt},
                ],
                temperature=0.0,
                max_tokens=cfg.repl_code_max_tokens,
            )
            code_call_count += 1

        action_preview = parse_repl_code(code)
        print(
            f"[repl] Step {step}/{cfg.repl_max_steps}: action={action_preview.get('action')} "
            f"sub_results_before={len(state.get('sub_results', []))}",
            flush=True,
        )
        state, stdout, extra_events = execute_repl_action(llm, question, state, code, cfg)
        print(
            f"[repl] Step {step}/{cfg.repl_max_steps}: stdout_keys={list(stdout.keys())} "
            f"sub_results_after={len(state.get('sub_results', []))} "
            f"final_set={state.get('final') is not None}",
            flush=True,
        )
        history.append({"type": "code", "step": step, "value": code})
        history.append(
            {
                "type": "stdout_metadata",
                "step": step,
                "value": {
                    "stdout": stdout,
                    "metadata": metadata_for_repl(state),
                },
            }
        )
        trace.append(
            {
                "node": "repl_step",
                "step": step,
                "code": code,
                "parsed_action": parse_repl_code(code),
                "stdout": stdout,
                "metadata_after": metadata_for_repl(state),
            }
        )
        trace.extend(extra_events)
        if state.get("final") is not None:
            print(f"[repl] Final set at step {step}", flush=True)
            break

    if state.get("final") is None:
        print(
            "[repl] No FINAL from REPL loop; running fallback sub_RLM calls and root aggregation",
            flush=True,
        )
        trace.append(
            {
                "node": "repl_fallback",
                "reason": "REPL loop ended without FINAL; running all sub_RLM calls then root_aggregate.",
            }
        )
        for snippet in env["P"]["snippets"]:
            print(
                f"[repl] Fallback sub_RLM: handle={snippet['handle']} "
                f"sample_id={snippet.get('sample_id', '')}",
                flush=True,
            )
            parsed, raw, prompt = sub_rlm_one(llm, question, snippet, cfg)
            state["sub_results"].append(parsed)
            trace.append(
                {
                    "node": "sub_RLM_fallback",
                    "snippet_handle": snippet["handle"],
                    "prompt_chars": len(prompt),
                    "raw_response": raw,
                    "parsed_response": parsed,
                }
            )
        parsed_final, raw_final, root_prompt = root_aggregate_action(llm, question, state, cfg)
        state["root_prompt"] = root_prompt
        state["raw_response"] = raw_final
        state["parsed_response"] = parsed_final
        state["final"] = parsed_final
        trace.append(
            {
                "node": "root_rlm_aggregate_fallback",
                "prompt_chars": len(root_prompt),
                "raw_response": raw_final,
                "parsed_response": parsed_final,
            }
        )

    print(
        "[repl] Complete: "
        f"steps={sum(1 for item in trace if item.get('node') == 'repl_step')} "
        f"code_calls={code_call_count} "
        f"sub_results={len(state.get('sub_results', []))}",
        flush=True,
    )
    return {
        "environment": env,
        "sub_results": state.get("sub_results", []),
        "root_prompt": state.get("root_prompt", ""),
        "raw_response": state.get("raw_response", compact_json(state.get("final", {}))),
        "parsed_response": state.get("parsed_response", state.get("final", {})),
        "trace": trace,
        "repl_history": history,
        "repl_final_state": {
            "metadata": metadata_for_repl(state),
            "last_hits": state.get("last_hits", []),
            "snippet_classifications": state.get("snippet_classifications", []),
            "final": state.get("final"),
        },
        "repl_code_call_count": code_call_count,
        "repl_step_count": sum(1 for item in trace if item.get("node") == "repl_step"),
    }

def judge_answer(
    llm: Llama | None,
    question: str,
    gt: Dict[str, List[str]],
    parsed: Dict[str, Any],
    cfg: RlmConfig,
) -> Dict[str, Any]:
    if cfg.skip_llm_judge:
        print("[judge] Skipped by --skip-llm-judge", flush=True)
        return {"skipped": True}
    if cfg.dry_run:
        print("[judge] Dry run judge placeholder", flush=True)
        return {"score_0_to_5": None, "verdict": "dry_run", "token_level_notes": ""}
    assert llm is not None
    print("[judge] Running LLM-as-judge", flush=True)
    prompt = (
        "Grade this RLM/RAG answer against the references. Return only valid JSON "
        "with keys score_0_to_5, verdict, token_level_notes, missing_facts, "
        "hallucinations.\n\n"
        f"Question: {question}\n"
        f"Reference short answers: {json.dumps(gt['short_answers'], ensure_ascii=False)}\n"
        f"Reference long answers: {json.dumps(gt['long_answers'], ensure_ascii=False)}\n"
        f"Model response: {json.dumps(parsed, ensure_ascii=False)}"
    )
    raw = ask_llm(
        llm,
        [
            {"role": "system", "content": "You are a strict but fair evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=cfg.judge_max_tokens,
    )
    parsed = parse_response(raw)
    print(
        f"[judge] Done score={parsed.get('score_0_to_5', '')} "
        f"verdict={parsed.get('verdict', '')}",
        flush=True,
    )
    return parsed


def assess(row: Dict[str, Any], parsed: Dict[str, Any], docs: list[Any], judge: Dict[str, Any]) -> Dict[str, Any]:
    gt = ground_truth(row)
    long_answer = str(parsed.get("long_answer", ""))
    short_answer = str(parsed.get("short_answer", ""))
    baseline = safe_value(row.get("llm_response", {}))
    if not isinstance(baseline, dict):
        baseline = {"long_answer": compact_json(baseline), "short_answer": ""}
    metrics = {
        "short_token_f1": token_f1(short_answer, gt["short_answers"]),
        "long_token_f1": token_f1(long_answer, gt["long_answers"]),
        "retrieved_self_hit": int(
            first_text(row.get("sample_id", ""))
            in [doc.metadata.get("sample_id", "") for doc in docs]
        ),
        "llm_judge_json": compact_json(judge),
        "llm_judge_score_0_to_5": judge.get("score_0_to_5", "") if isinstance(judge, dict) else "",
        "llm_judge_verdict": judge.get("verdict", "") if isinstance(judge, dict) else "",
        "llm_judge_token_level_notes": judge.get("token_level_notes", "") if isinstance(judge, dict) else "",
    }
    metrics.update(automatic_metrics(short_answer, gt["short_answers"], "rlm_short"))
    metrics.update(automatic_metrics(long_answer, gt["long_answers"], "rlm_long"))
    metrics.update(automatic_metrics(str(baseline.get("short_answer", "")), gt["short_answers"], "baseline_short"))
    metrics.update(automatic_metrics(str(baseline.get("long_answer", "")), gt["long_answers"], "baseline_long"))
    return metrics


def write_profile(df: pd.DataFrame, out_dir: Path) -> None:
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "paper_applied": "Recursive Language Models",
        "method": (
            "Algorithm-1 REPL-style RLM: InitREPL(P), AddFunction(sub_RLM), "
            "LLM proposes safe REPL actions, stdout metadata is appended to history, "
            "FINAL_VAR/root aggregation ends the loop."
        ),
    }
    pd.DataFrame([profile]).to_csv(out_dir / "dataset_profile.csv", index=False)


def run(cfg: RlmConfig) -> None:
    print("[start] Paper-applied RLM REPL RAG evaluator", flush=True)
    print(f"[start] Output directory: {cfg.out_dir}", flush=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(cfg.input_path, cfg.max_rows)
    write_profile(df, cfg.out_dir)
    vectorstore = build_vectorstore(df, cfg.out_dir, cfg.rebuild_index)
    llm = load_rlm_llm(cfg)
    rows = []
    total_started = time.time()
    print(
        f"[run] Evaluating {len(df)} questions "
        f"top_k={cfg.top_k} repl_max_steps={cfg.repl_max_steps} "
        f"judge={'off' if cfg.skip_llm_judge else 'on'}",
        flush=True,
    )
    for index, row in enumerate(df.to_dict("records"), start=1):
        started = time.time()
        question = first_text(row.get("question", ""))
        sample_id = first_text(row.get("sample_id", ""))
        print(
            f"\n[question] {index}/{len(df)} sample_id={sample_id} "
            f"question={question[:160]}",
            flush=True,
        )
        print("[retrieve] Searching vector DB", flush=True)
        docs = vectorstore.similarity_search(question, k=cfg.top_k)
        print(
            "[retrieve] Retrieved sample_ids="
            + ", ".join(doc.metadata.get("sample_id", "") for doc in docs),
            flush=True,
        )
        result = run_recursive_answer(llm, question, docs, cfg)
        gt = ground_truth(row)
        parsed = result["parsed_response"]
        judge = judge_answer(llm, question, gt, parsed, cfg)
        print("[metrics] Computing F1/BLEU/ROUGE and fused score", flush=True)
        metrics = assess(row, parsed, docs, judge)
        p_value = result["environment"].get("P", {})
        p_metadata = result["environment"].get("metadata", {})
        retrieved_ids = [doc.metadata.get("sample_id", "") for doc in docs]
        out_row = {
            "row_number": index,
            "experiment_name": "RLM REPL 8-step + RAG",
            "row_explanation": (
                "One Natural Questions item evaluated with external P, recursive "
                "Algorithm-1 REPL actions, up to 8 REPL steps, sub_RLM calls, "
                "root aggregation, automatic metrics, and LLM-as-judge."
            ),
            "sample_id": first_text(row.get("sample_id", "")),
            "question": question,
            "answer_summary": (
                f"short={parsed.get('short_answer', '')} | "
                f"long={str(parsed.get('long_answer', ''))[:240]}"
            ),
            "ground_truth_short_answers": compact_json(gt["short_answers"]),
            "ground_truth_long_answers": compact_json(gt["long_answers"]),
            "rlm_short_answer": parsed.get("short_answer", ""),
            "rlm_long_answer": parsed.get("long_answer", ""),
            "rlm_confidence": parsed.get("confidence", ""),
            "raw_llm_response": result["raw_response"],
            "retrieval_summary": (
                f"top_k={cfg.top_k}; retrieved_sample_ids={', '.join(retrieved_ids)}"
            ),
            "retrieved_sample_ids": compact_json(retrieved_ids),
            "p_question": p_value.get("question", ""),
            "p_instruction": p_value.get("instruction", ""),
            "p_handle": p_metadata.get("P_handle", "P"),
            "p_snippet_count": p_metadata.get("snippet_count", ""),
            "p_total_snippet_chars": p_metadata.get("total_snippet_chars", ""),
            "p_available_handles": compact_json(p_metadata.get("available_handles", [])),
            "p_summary": (
                f"P has {p_metadata.get('snippet_count', 0)} snippets and "
                f"{p_metadata.get('total_snippet_chars', 0)} chars; root sees "
                "metadata plus recursive sub-call results."
            ),
            "recursive_call_count": len(result["sub_results"]),
            "repl_step_count": result.get("repl_step_count", ""),
            "repl_code_call_count": result.get("repl_code_call_count", ""),
            "repl_history_json": compact_json(result.get("repl_history", [])),
            "repl_final_state_json": compact_json(result.get("repl_final_state", {})),
            "repl_allowed_functions": compact_json(
                [
                    "find_snippets(keyword, window=400, max_hits=5)",
                    "batch_classify_snippets()",
                    "sub_RLM(handles)",
                    "root_aggregate()",
                    "FINAL_VAR(value)",
                ]
            ),
            "repl_algorithm_mapping": (
                "state=InitREPL(P); AddFunction(sub_RLM); hist=[Metadata(state)]; "
                "while code=LLM(hist): state,stdout=REPL(state,code); "
                "hist+=code+Metadata(stdout); stop when Final is set."
            ),
            "trace_summary": (
                "init_repl_environment -> LLM proposes REPL code/action -> "
                "safe REPL executes action -> stdout metadata appended to history -> "
                "FINAL/root_aggregate -> LLM judge -> metrics"
            ),
            "rlm_environment_json": compact_json(result["environment"]),
            "rlm_root_prompt": result["root_prompt"],
            "rlm_sub_results_json": compact_json(result["sub_results"]),
            "rlm_trace_json": compact_json(result["trace"]),
            "elapsed_seconds": round(time.time() - started, 3),
            **metrics,
        }
        out_row["elapsed_display"] = f"elapsed={out_row['elapsed_seconds']:.3f}s"
        out_row["judge_summary"] = (
            f"score={out_row.get('llm_judge_score_0_to_5', '')}; "
            f"verdict={out_row.get('llm_judge_verdict', '')}; "
            f"notes={str(out_row.get('llm_judge_token_level_notes', ''))[:240]}"
        )
        add_paper_columns(
            out_row,
            metric_prefix="rlm",
            method_mapping="8-step REPL-style Recursive Language Model over external P snippets.",
            principles=(
                "Applies Recursive Language Models Algorithm 1 with external P, "
                "LLM-generated REPL actions, safe tool/function execution, stdout "
                "metadata feedback, sub_RLM recursion, and an expanded 8-step "
                "REPL budget."
            ),
            estimated_lm_calls=result.get("repl_code_call_count", 0)
            + len(result["sub_results"])
            + 1
            + (0 if cfg.skip_llm_judge else 1),
        )
        rows.append(out_row)
        print(
            f"[{index}/{len(df)}] {out_row['sample_id']} "
            f"short_f1={metrics['rlm_short_token_f1']:.3f} "
            f"long_f1={metrics['rlm_long_token_f1']:.3f} "
            f"elapsed={out_row['elapsed_seconds']:.3f}s",
            flush=True,
        )
    results = pd.DataFrame(rows)
    results_path = cfg.out_dir / "nq_rlm_repl8_rag_assessment_traces.csv"
    print(f"[output] Writing detailed traces CSV: {results_path}", flush=True)
    results.to_csv(results_path, index=False)
    metric_cols = [
        "rlm_short_token_f1",
        "rlm_long_token_f1",
        "rlm_short_bleu",
        "rlm_long_bleu",
        "rlm_short_rouge1",
        "rlm_long_rouge1",
        "baseline_short_token_f1",
        "baseline_long_token_f1",
        "llm_judge_score_0_to_5",
        "retrieved_self_hit",
        "recursive_call_count",
        "repl_step_count",
        "repl_code_call_count",
        "estimated_lm_calls",
        "faithfulness_fused_score",
        "elapsed_seconds",
    ]
    metric_cols = [col for col in metric_cols if col in results.columns]
    results[metric_cols] = results[metric_cols].apply(pd.to_numeric, errors="coerce")
    summary = results[metric_cols].mean(numeric_only=True).to_frame("mean").T
    summary.insert(0, "rows_evaluated", len(results))
    summary.insert(1, "method", "RLM_REPL8_RAG")
    summary.insert(
        2,
        "summary_explanation",
        (
            "Mean metrics for 8-step Algorithm-1 REPL-style RLM: LLM-generated "
            "safe REPL actions over external P, sub_RLM calls, stdout metadata "
            "history, and final root aggregation."
        ),
    )
    if "elapsed_seconds" in summary.columns:
        summary["mean_elapsed_display"] = summary["elapsed_seconds"].apply(
            lambda value: f"elapsed={value:.3f}s" if pd.notna(value) else ""
        )
    summary_path = cfg.out_dir / "nq_rlm_repl8_rag_summary.csv"
    print(f"[output] Writing summary CSV: {summary_path}", flush=True)
    summary.to_csv(summary_path, index=False)
    print(f"[done] Total elapsed: {time.time() - total_started:.2f}s", flush=True)
    print(f"[done] Wrote: {results_path}", flush=True)
    print(f"[done] Wrote: {summary_path}", flush=True)


def parse_args() -> RlmConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--subcall-max-tokens", type=int, default=256)
    parser.add_argument(
        "--repl-max-steps",
        type=int,
        default=8,
        help="Maximum Algorithm-1 REPL iterations before fallback root aggregation.",
    )
    parser.add_argument(
        "--repl-code-max-tokens",
        type=int,
        default=192,
        help="Max tokens for the LLM-generated safe REPL action/code.",
    )
    parser.add_argument("--final-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return RlmConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        top_k=args.top_k,
        subcall_max_tokens=args.subcall_max_tokens,
        repl_max_steps=max(1, args.repl_max_steps),
        repl_code_max_tokens=args.repl_code_max_tokens,
        final_max_tokens=args.final_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        temperature=args.temperature,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        model_path=args.model_path,
        rebuild_index=args.rebuild_index,
        skip_llm_judge=args.skip_llm_judge,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    run(parse_args())
