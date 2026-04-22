#!/usr/bin/env python
"""Static-analysis security screen for a candidate submission.

Usage:
    python scripts/scan_submission.py --file path/to/model_submission.py
    python scripts/scan_submission.py --file ... --json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polybench.submission_scan import scan_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = scan_file(Path(args.file))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_text())
    return 0 if report.verdict == "accept" else 2


if __name__ == "__main__":
    sys.exit(main())
