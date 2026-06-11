#!/usr/bin/env python3
"""Gradio app for trying all six SAMSum summarizer settings through Ollama.

This is an interactive demo version of the experiment settings. It uses the
same high-level prompting styles as the SAMSum experiments:

- Llama 3 8B + RAG style
- GA + RAG style
- ES + RAG style
- RLM REPL 4-step + RAG style
- RLM REPL 8-step + RAG style
- Adaptive-P short REPL + RAG style

The app calls an Ollama-compatible local endpoint.
"""

from __future__ import annotations

import json
import re
from typing import Dict

import gradio as gr
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q4_0"


SUMMARIZER_MODES = {
    "Llama 3 8B + RAG": {
        "steps": 1,
        "instruction": (
            "Write a concise, faithful dialogue summary. Preserve the main "
            "participants, decisions, requests, times, places, and outcomes."
        ),
    },
    "GA + RAG": {
        "steps": 1,
        "instruction": (
            "Use an evolved summarization policy. First identify the most "
            "important facts, then produce the most compact faithful summary. "
            "Avoid unsupported details."
        ),
    },
    "ES + RAG": {
        "steps": 1,
        "instruction": (
            "Explore multiple possible summaries mentally. Select the summary "
            "with the fewest missing facts, least redundancy, and strongest "
            "faithfulness to the dialogue."
        ),
    },
    "RLM REPL 4-step + RAG": {
        "steps": 4,
        "instruction": (
            "Act as a short REPL/RLM summarizer. For each internal step, refine "
            "the evidence focus: people, actions, decisions, times, places, and "
            "commitments. Then return the final concise summary."
        ),
    },
    "RLM REPL 8-step + RAG": {
        "steps": 8,
        "instruction": (
            "Act as a longer REPL/RLM summarizer. Repeatedly inspect the dialogue "
            "for missing facts, contradictions, and unsupported claims. Then "
            "return the final concise summary."
        ),
    },
    "Adaptive-P short REPL + RAG": {
        "steps": 4,
        "instruction": (
            "Use Adaptive-P closed-loop summarization. Maintain P as feedback "
            "about what to preserve and what to repair. Reward faithful facts, "
            "important names/actions/times, and concise wording. Punish missing "
            "facts, unsupported claims, vague wording, and excessive length. "
            "Return the best final summary after the loop."
        ),
    },
}


def call_ollama(model: str, prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model.strip() or DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Error calling Ollama: {exc}"
    return response.json().get("response", "No summary generated.").strip()


def clean_summary(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.strip()
    if not text:
        return "No summary generated."
    return text


def build_prompt(transcript: str, mode: str, previous_feedback: str = "") -> str:
    spec = SUMMARIZER_MODES[mode]
    feedback = f"\n\nAdaptive feedback / P:\n{previous_feedback}" if previous_feedback else ""
    return (
        f"{spec['instruction']}\n\n"
        "Summarize the following dialogue in 3 concise bullet points. "
        "Only include facts supported by the dialogue. Do not invent details."
        f"{feedback}\n\n"
        f"Dialogue:\n{transcript}\n\n"
        "Final summary:"
    )


def summarize_one(transcript: str, mode: str, ollama_model: str, temperature: float) -> str:
    if not transcript.strip():
        return "Please enter a dialogue or long text to summarize."
    prompt = build_prompt(transcript, mode)
    return clean_summary(call_ollama(ollama_model, prompt, temperature))


def summarize_adaptive_loop(transcript: str, ollama_model: str, temperature: float) -> str:
    if not transcript.strip():
        return "Please enter a dialogue or long text to summarize."
    best_summary = ""
    p_feedback = "Initial P: produce a concise faithful summary."
    loop_notes = []
    for loop in range(1, 5):
        prompt = build_prompt(transcript, "Adaptive-P short REPL + RAG", p_feedback)
        summary = clean_summary(call_ollama(ollama_model, prompt, temperature))
        best_summary = summary
        p_feedback = (
            f"Loop {loop} reward: preserve concrete names, actions, decisions, dates, and places.\n"
            f"Loop {loop} punishment: repair missing facts, hallucinations, vagueness, and overly long text.\n"
            f"Previous summary:\n{summary}"
        )
        loop_notes.append(f"### Loop {loop}\n{summary}")
    return "\n\n".join(loop_notes) + f"\n\n### Best final summary\n{best_summary}"


def summarize_all(transcript: str, ollama_model: str, temperature: float) -> str:
    if not transcript.strip():
        return "Please enter a dialogue or long text to summarize."
    outputs: Dict[str, str] = {}
    for mode in SUMMARIZER_MODES:
        if mode == "Adaptive-P short REPL + RAG":
            outputs[mode] = summarize_adaptive_loop(transcript, ollama_model, temperature)
        else:
            outputs[mode] = summarize_one(transcript, mode, ollama_model, temperature)
    return "\n\n---\n\n".join(f"## {mode}\n{summary}" for mode, summary in outputs.items())


def route_summarizer(
    transcript: str,
    mode: str,
    ollama_model: str,
    temperature: float,
) -> str:
    if mode == "Run all six models":
        return summarize_all(transcript, ollama_model, temperature)
    if mode == "Adaptive-P short REPL + RAG":
        return summarize_adaptive_loop(transcript, ollama_model, temperature)
    return summarize_one(transcript, mode, ollama_model, temperature)


with gr.Blocks(title="SAMSum Six-Model Summarizer") as demo:
    gr.Markdown("# SAMSum Six-Model Summarizer")
    gr.Markdown(
        "Paste a dialogue and compare the six summarizer settings through a local "
        "Llama 3 8B 4-bit quantized Ollama model."
    )
    with gr.Row():
        mode = gr.Dropdown(
            choices=["Run all six models"] + list(SUMMARIZER_MODES.keys()),
            value="Run all six models",
            label="Summarizer setting",
        )
        ollama_model = gr.Textbox(value=DEFAULT_OLLAMA_MODEL, label="Ollama model")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
    transcript = gr.Textbox(lines=14, placeholder="Paste SAMSum dialogue here...", label="Dialogue / text")
    summarize_button = gr.Button("Summarize")
    output = gr.Markdown(label="Summaries")
    summarize_button.click(
        fn=route_summarizer,
        inputs=[transcript, mode, ollama_model, temperature],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
