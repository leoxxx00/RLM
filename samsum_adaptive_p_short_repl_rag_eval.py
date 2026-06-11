#!/usr/bin/env python3
import sys

from samsum_eval import parse_args, run

if "--mode" not in sys.argv:
    sys.argv.extend(["--mode", "adaptive_p"])

if __name__ == "__main__":
    run(parse_args())
