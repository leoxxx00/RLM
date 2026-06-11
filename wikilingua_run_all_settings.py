#!/usr/bin/env python3
"""Run all six WikiLingua settings sequentially, then generate comparison plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/Users/htet/Desktop/Projects/X-RLM")
MODES = ["rag", "ga", "es", "repl4", "repl8", "adaptive_p"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--target-lang", choices=["all", "nl", "fr", "de", "es"], default="all")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    for mode in MODES:
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "wikilingua_eval.py"),
            "--mode",
            mode,
            "--max-rows",
            str(args.max_rows),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--n-ctx",
            str(args.n_ctx),
            "--target-lang",
            args.target_lang,
        ]
        if args.input:
            cmd.extend(["--input", str(args.input)])
        if args.rebuild_index:
            cmd.append("--rebuild-index")
        if args.skip_llm_judge:
            cmd.append("--skip-llm-judge")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.no_resume:
            cmd.append("--no-resume")
        print("\n[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "compare_wikilingua_models.py"),
            "--max-rows",
            str(args.max_rows),
            "--target-lang",
            args.target_lang,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
