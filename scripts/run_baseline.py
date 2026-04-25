#!/usr/bin/env python
"""Run the reference MomentumBaseline against live Polymarket.

The harness always runs the baseline in parallel with itself — the report
prints two columns (Model vs Baseline) that will be identical on this run
since the primary model IS the baseline. Use this to sanity-check the
harness, confirm the market is live, and see what "the bar" actually
produces under current conditions.

Example:
    python scripts/run_baseline.py --duration 300
    python scripts/run_baseline.py --duration 3600
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from polybench.baselines import MomentumBaseline
from polybench.cli import _load_config
from polybench.harness import Harness, HarnessConfig, format_summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["momentum"], default="momentum",
                   help="Only 'momentum' is shipped; kept for argparse compatibility.")
    p.add_argument("--duration", type=float, default=300.0, help="seconds")
    p.add_argument("--starting-capital", type=float, default=1000.0)
    p.add_argument("--slippage-bps", type=float, default=50.0)
    p.add_argument("--fee-rate", type=float, default=0.072,
                   help="Polymarket-style fee coefficient")
    p.add_argument("--price-source", default="polymarket",
                   choices=["polymarket", "binance"],
                   help="Default 'polymarket' disables external BTC WebSockets.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--resolution-timeout",
        type=float,
        default=45.0,
        help="Deprecated; event resolution now continues asynchronously after rollover.",
    )
    p.add_argument(
        "--postmortem-timeout",
        type=float,
        default=0.0,
        help="Extra seconds after the run to wait for still-pending resolutions (default 0).",
    )
    args = p.parse_args(argv)

    model = MomentumBaseline(config=_load_config(args.config))
    out_dir = Path(args.output_dir or f"runs/baseline_momentum_{int(time.time())}")
    cfg = HarnessConfig(
        duration_s=args.duration,
        starting_capital=args.starting_capital,
        slippage_bps=args.slippage_bps,
        fee_rate=args.fee_rate,
        price_source=args.price_source,
        output_dir=out_dir,
        resolution_poll_timeout_s=args.resolution_timeout,
        postmortem_resolution_s=args.postmortem_timeout,
    )
    result = asyncio.run(Harness(model=model, config=cfg).run())
    print(format_summary(result))
    print(f"report: {out_dir / 'report.json'}")
    print(f"ticks:  {out_dir / 'ticks.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
