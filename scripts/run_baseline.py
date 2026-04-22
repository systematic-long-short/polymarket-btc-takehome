#!/usr/bin/env python
"""Run one of the reference baselines against live Polymarket.

Examples:
    python scripts/run_baseline.py --model momentum --duration 300
    python scripts/run_baseline.py --model random --duration 60
    python scripts/run_baseline.py --model alwaysup --duration 300
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from polybench.cli import _load_config
from polybench.harness import Harness, HarnessConfig, format_summary

BASELINE_SPECS = {
    "random": "polybench.baselines:RandomModel",
    "alwaysup": "polybench.baselines:AlwaysUpModel",
    "momentum": "polybench.baselines:MomentumBaseline",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(BASELINE_SPECS), default="momentum")
    p.add_argument("--duration", type=float, default=300.0, help="seconds")
    p.add_argument("--starting-capital", type=float, default=1000.0)
    p.add_argument("--slippage-bps", type=float, default=200.0)
    p.add_argument("--price-source", default="binance",
                   choices=["binance", "binance-us", "coinbase"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    import importlib
    module_name, class_name = BASELINE_SPECS[args.model].split(":")
    klass = getattr(importlib.import_module(module_name), class_name)
    model = klass(config=_load_config(args.config))

    out_dir = Path(args.output_dir or f"runs/baseline_{args.model}_{int(time.time())}")
    cfg = HarnessConfig(
        duration_s=args.duration,
        starting_capital=args.starting_capital,
        slippage_bps=args.slippage_bps,
        price_source=args.price_source,
        output_dir=out_dir,
    )
    result = asyncio.run(Harness(model=model, config=cfg).run())
    print(format_summary(result))
    print(f"report: {out_dir / 'report.json'}")
    print(f"ticks:  {out_dir / 'ticks.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
