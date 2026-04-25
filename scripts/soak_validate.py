#!/usr/bin/env python
"""Run an operational live soak and validate feed health.

Evaluators should run this shortly before an official scoring window. It uses
the reference MomentumBaseline only; no candidate code is loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from polybench.baselines import MomentumBaseline
from polybench.harness import Harness, HarnessConfig, format_summary
from scripts.validate_live_run import validate_live_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--price-source", choices=["polymarket", "binance"], default="polymarket")
    parser.add_argument("--postmortem-timeout", type=float, default=120.0)
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--allow-unknown", action="store_true")
    parser.add_argument("--max-clob-stale-count", type=int, default=0)
    parser.add_argument("--max-missing-book-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)

    output = args.output or Path(f"runs/soak_{args.price_source}_{int(time.time())}")
    cfg = HarnessConfig(
        duration_s=args.duration,
        output_dir=output,
        price_source=args.price_source,
        postmortem_resolution_s=args.postmortem_timeout,
        command=tuple(sys.argv if argv is None else [sys.argv[0], *argv]),
    )
    result = asyncio.run(Harness(model=MomentumBaseline(), config=cfg).run())
    print(format_summary(result))
    summary = validate_live_run(
        report_path=output / "report.json",
        ticks_path=output / "ticks.parquet",
        min_duration_s=args.duration,
        min_events=args.min_events,
        require_binance=args.price_source == "binance",
        allow_unknown=args.allow_unknown,
        expect_price_source=args.price_source,
        max_clob_stale_count=args.max_clob_stale_count,
        max_missing_book_fraction=args.max_missing_book_fraction,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
