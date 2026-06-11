#!/usr/bin/env python3
import sys

from wikilingua_eval import parse_args, run

if "--mode" not in sys.argv:
    sys.argv.extend(["--mode", "es"])

if __name__ == "__main__":
    run(parse_args())
