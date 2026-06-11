#!/usr/bin/env python3
"""Local Llama chat with prompt improvement, generation settings, and traces."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

from llama_cpp import Llama


P_COUNTER_ENV = "P_NUMERIC_VALUE"
P_TEXT_ENV = "P_TEXT_VALUE"
P_ORIGINAL_TEXT_ENV = "P_ORIGINAL_TEXT_VALUE"
P_TEMPERATURE_ENV = "P_TEMPERATURE_VALUE"
P_MAX_TOKENS_ENV = "P_MAX_TOKENS_VALUE"
P_SYNTAX_ENV = "P_EXTERNAL_ENVIRONMENT_SYNTAX"

DEFAULT_REPO_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
DEFAULT_FILENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 1.2

MIN_MAX_TOKENS = 64
MAX_MAX_TOKENS = 1024


def load_llm(args: argparse.Namespace) -> Llama:
    if args.model_path:
        print(f"[llm] Loading local model: {args.model_path}", flush=True)
        return Llama(
            model_path=str(args.model_path),
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
        )

    print(f"[llm] Loading from Hugging Face: {args.repo_id}", flush=True)
    return Llama.from_pretrained(
        repo_id=args.repo_id,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
    )


def ask_llm(
    llm: Llama,
    messages: list[dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def print_trace(title: str) -> None:
    print(f"\n--- TRACE: {title} ---")


def print_messages(label: str, chat_messages: list[dict[str, str]], trace_prompts: bool) -> None:
    if not trace_prompts:
        return

    print_trace(label)
    for index, message in enumerate(chat_messages, start=1):
        role = message["role"].upper()
        content = message["content"]
        print(f"[{index}] {role}:")
        print(content)
        print()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_save_dir(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "turn_id",
        "original_prompt",
        "updated_prompt",
        "temperature",
        "max_tokens",
        "raw_prompt_improvement_output",
        "final_answer",
        "p_environment_before",
        "p_environment_after",
        "prompt_improvement_messages_json",
        "final_answer_messages_json",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_p_environment_syntax() -> str:
    p_id = os.environ.get(P_COUNTER_ENV, "0")
    original_text = os.environ.get(P_ORIGINAL_TEXT_ENV, "")
    evolved_text = os.environ.get(P_TEXT_ENV, "")
    temperature = os.environ.get(P_TEMPERATURE_ENV, str(DEFAULT_TEMPERATURE))
    max_tokens = os.environ.get(P_MAX_TOKENS_ENV, str(DEFAULT_MAX_TOKENS))

    return (
        "External P Environment Syntax:\n"
        f"P[{p_id}].id = {p_id}\n"
        f'P[{p_id}].original_text = """{original_text}"""\n'
        f'P[{p_id}].text = """{evolved_text}"""\n'
        f"P[{p_id}].temperature = {temperature}\n"
        f"P[{p_id}].max_tokens = {max_tokens}"
    )


def refresh_p_environment_syntax() -> None:
    os.environ[P_SYNTAX_ENV] = build_p_environment_syntax()


def print_p_environment_syntax() -> None:
    refresh_p_environment_syntax()
    print(os.environ[P_SYNTAX_ENV])


def get_p_environment_snapshot() -> dict[str, str]:
    refresh_p_environment_syntax()
    return {
        P_COUNTER_ENV: os.environ.get(P_COUNTER_ENV, "0"),
        P_ORIGINAL_TEXT_ENV: os.environ.get(P_ORIGINAL_TEXT_ENV, ""),
        P_TEXT_ENV: os.environ.get(P_TEXT_ENV, ""),
        P_TEMPERATURE_ENV: os.environ.get(P_TEMPERATURE_ENV, ""),
        P_MAX_TOKENS_ENV: os.environ.get(P_MAX_TOKENS_ENV, ""),
        P_SYNTAX_ENV: os.environ.get(P_SYNTAX_ENV, ""),
    }


def store_p_in_environment(prompt: str) -> int:
    p_numeric_value = int(os.environ.get(P_COUNTER_ENV, "0")) + 1

    os.environ[P_COUNTER_ENV] = str(p_numeric_value)
    os.environ[P_ORIGINAL_TEXT_ENV] = prompt
    os.environ[P_TEXT_ENV] = prompt
    os.environ[P_TEMPERATURE_ENV] = str(DEFAULT_TEMPERATURE)
    os.environ[P_MAX_TOKENS_ENV] = str(DEFAULT_MAX_TOKENS)

    refresh_p_environment_syntax()
    return p_numeric_value


def evolve_p_values_in_environment(
    updated_prompt_value: str,
    temperature: float,
    max_tokens: int,
) -> None:
    os.environ[P_TEXT_ENV] = updated_prompt_value
    os.environ[P_TEMPERATURE_ENV] = str(temperature)
    os.environ[P_MAX_TOKENS_ENV] = str(max_tokens)
    refresh_p_environment_syntax()


def clamp_number(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, number))


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    return int(clamp_number(value, minimum, maximum, fallback))


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def extract_labeled_settings(text: str) -> dict[str, str]:
    settings: dict[str, str] = {}

    prompt_match = re.search(
        r"(?:improved prompt|improved_prompt)\s*:\s*(.+?)(?=\n\s*(?:temperature|max tokens|max_tokens)\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    temperature_match = re.search(
        r"temperature\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )

    max_tokens_match = re.search(
        r"(?:max tokens|max_tokens)\s*:\s*([0-9]+)",
        text,
        re.IGNORECASE,
    )

    if prompt_match:
        settings["improved_prompt"] = prompt_match.group(1).strip().strip('"')

    if temperature_match:
        settings["temperature"] = temperature_match.group(1)

    if max_tokens_match:
        settings["max_tokens"] = max_tokens_match.group(1)

    return settings


def build_improve_prompt_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You improve user prompts for a local AI assistant and choose "
                "good generation settings. Keep the user's original intent. "
                "Do not answer the prompt. Return only valid JSON. Do not include "
                "markdown, labels, notes, or explanations. Use exactly this shape: "
                '{"improved_prompt":"...","temperature":0.7,"max_tokens":512}. '
                "temperature must be between 0.1 and 1.2. max_tokens must be between "
                "64 and 1024."
            ),
        },
        {
            "role": "user",
            "content": f"Original prompt:\n{prompt}",
        },
    ]


def improve_prompt(
    llm: Llama,
    prompt: str,
    trace_prompts: bool,
) -> tuple[str, float, int, str, list[dict[str, str]]]:
    rewrite_messages = build_improve_prompt_messages(prompt)

    print_messages("Prompt Improvement Messages Sent To LLM", rewrite_messages, trace_prompts)

    raw_output = ask_llm(
        llm,
        rewrite_messages,
        temperature=0.2,
        max_tokens=256,
    )

    settings = extract_json_object(raw_output) or extract_labeled_settings(raw_output)

    improved_prompt = str(settings.get("improved_prompt") or prompt)

    temperature = clamp_number(
        settings.get("temperature"),
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
        DEFAULT_TEMPERATURE,
    )

    max_tokens = clamp_int(
        settings.get("max_tokens"),
        MIN_MAX_TOKENS,
        MAX_MAX_TOKENS,
        DEFAULT_MAX_TOKENS,
    )

    if trace_prompts:
        print_trace("Prompt Improvement Output From LLM")
        print(raw_output)

        print_trace("Parsed Prompt And Generation Settings")
        print(f"improved_prompt={improved_prompt}")
        print(f"temperature={temperature}")
        print(f"max_tokens={max_tokens}")

    return improved_prompt, temperature, max_tokens, raw_output, rewrite_messages


def run_chat(args: argparse.Namespace) -> None:
    llm = load_llm(args)
    ensure_save_dir(args.save_dir)
    jsonl_path = args.save_dir / args.jsonl_name
    csv_path = args.save_dir / args.csv_name
    print(f"[save] JSONL trace: {jsonl_path}", flush=True)
    print(f"[save] CSV summary: {csv_path}", flush=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful local Llama 3 assistant. Answer naturally, clearly, "
                "and directly. Do not mention internal prompt rewriting unless asked."
            ),
        }
    ]

    print("Llama 3 8B Q4 prompt-improving chat started. Type 'exit' to quit.\n")
    print(f"Trace prompts={args.trace_prompts}\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        p_numeric_value = store_p_in_environment(prompt)

        print_trace("New Turn")
        print(f"Raw user input: {prompt}")

        print("\n=== External Environment Values Before Prompt Evolution ===")
        print(f"{P_COUNTER_ENV}={p_numeric_value}")
        print(f"{P_ORIGINAL_TEXT_ENV}={os.environ[P_ORIGINAL_TEXT_ENV]}")
        print(f"{P_TEXT_ENV}={os.environ[P_TEXT_ENV]}")
        print(f"{P_TEMPERATURE_ENV}={os.environ[P_TEMPERATURE_ENV]}")
        print(f"{P_MAX_TOKENS_ENV}={os.environ[P_MAX_TOKENS_ENV]}")
        print_p_environment_syntax()

        env_before = get_p_environment_snapshot()

        (
            updated_prompt_value,
            temperature_value,
            max_tokens_value,
            raw_prompt_improvement_output,
            prompt_improvement_messages,
        ) = improve_prompt(
            llm,
            prompt,
            args.trace_prompts,
        )

        evolve_p_values_in_environment(
            updated_prompt_value,
            temperature_value,
            max_tokens_value,
        )

        print("\n=== External Environment Values After Prompt Evolution ===")
        print(f"{P_COUNTER_ENV}={os.environ[P_COUNTER_ENV]}")
        print(f"{P_ORIGINAL_TEXT_ENV}={os.environ[P_ORIGINAL_TEXT_ENV]}")
        print(f"{P_TEXT_ENV}={os.environ[P_TEXT_ENV]}")
        print(f"{P_TEMPERATURE_ENV}={os.environ[P_TEMPERATURE_ENV]}")
        print(f"{P_MAX_TOKENS_ENV}={os.environ[P_MAX_TOKENS_ENV]}")
        print_p_environment_syntax()

        print("\n=== Original Prompt Value ===")
        print(prompt)

        print("\n=== Updated Prompt Value ===")
        print(updated_prompt_value)

        print("\n=== Updated Generation Settings ===")
        print(f"temperature={temperature_value}")
        print(f"max_tokens={max_tokens_value}")

        env_after = get_p_environment_snapshot()

        answer_messages = messages + [
            {
                "role": "user",
                "content": os.environ[P_TEXT_ENV],
            }
        ]

        print_messages("Final Answer Messages Sent To LLM", answer_messages, args.trace_prompts)

        answer = ask_llm(
            llm,
            answer_messages,
            temperature=temperature_value,
            max_tokens=max_tokens_value,
        )

        print("\n=== Final Answer ===")
        print(answer, "\n")

        saved_row = {
            "timestamp": timestamp(),
            "turn_id": os.environ.get(P_COUNTER_ENV, ""),
            "original_prompt": prompt,
            "updated_prompt": updated_prompt_value,
            "temperature": temperature_value,
            "max_tokens": max_tokens_value,
            "raw_prompt_improvement_output": raw_prompt_improvement_output,
            "final_answer": answer,
            "p_environment_before": json.dumps(env_before, ensure_ascii=False),
            "p_environment_after": json.dumps(env_after, ensure_ascii=False),
            "prompt_improvement_messages_json": json.dumps(
                prompt_improvement_messages,
                ensure_ascii=False,
            ),
            "final_answer_messages_json": json.dumps(answer_messages, ensure_ascii=False),
        }
        append_jsonl(jsonl_path, saved_row)
        append_csv(csv_path, saved_row)
        print(f"[save] Saved turn {saved_row['turn_id']} to {jsonl_path}")
        print(f"[save] Updated CSV: {csv_path}")

        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": answer})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-improving local Llama chat with P environment tracing."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-trace-prompts", dest="trace_prompts", action="store_false")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/Users/htet/Desktop/Projects/X-RLM/prompt_improving_outputs"),
    )
    parser.add_argument("--jsonl-name", default="prompt_improving_chat_traces.jsonl")
    parser.add_argument("--csv-name", default="prompt_improving_chat_outputs.csv")
    parser.set_defaults(trace_prompts=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_chat(parse_args())
