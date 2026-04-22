#!/usr/bin/env python
"""Top-level convenience wrapper:

    python replay.py --model-file examples/model_submission.py \\
                     --class ModelSubmission \\
                     --data tests/fixtures/recorded_event.parquet

Equivalent to ``polybench replay ...``. Kept at the repo root so candidates
can iterate without remembering the subcommand.
"""

from __future__ import annotations

import sys

from polybench.cli import main

if __name__ == "__main__":
    # Rewrite argv: insert "replay" subcommand.
    argv = ["replay", *sys.argv[1:]]
    sys.exit(main(argv))
