#!/usr/bin/env python
"""Run a candidate submission against live Polymarket.

Runs the security scanner FIRST, refuses to load the submission if any
critical finding is raised (override with --allow-unsafe, intended only
for a local developer sanity-check).

Example:
    python scripts/run_candidate.py \\
        --submission examples/model_submission.py \\
        --duration 300
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import time
from pathlib import Path

from polybench.cli import _load_config
from polybench.harness import Harness, HarnessConfig, format_summary
from polybench.submission_scan import scan_file


def _load_module(path: Path, class_name: str, config):
    module_name = f"_polybench_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    klass = getattr(module, class_name, None)
    if klass is None:
        raise SystemExit(f"class {class_name!r} not found in {path}")
    return klass(config=config)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, help="path to model_submission.py")
    p.add_argument("--class", dest="class_name", default="ModelSubmission")
    p.add_argument("--duration", type=float, default=7200.0)
    p.add_argument("--starting-capital", type=float, default=1000.0)
    p.add_argument("--slippage-bps", type=float, default=200.0)
    p.add_argument("--price-source", default="binance",
                   choices=["binance", "binance-us", "coinbase"])
    p.add_argument("--output", dest="output_dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Load the submission even if the scanner rejects it (debug-only).",
    )
    args = p.parse_args(argv)

    path = Path(args.submission).resolve()
    if not path.is_file():
        print(f"ERROR: submission file not found: {path}", file=sys.stderr)
        return 2

    print(f"scanning {path}...")
    report = scan_file(path)
    print(report.format_text())
    if report.verdict != "accept":
        if not args.allow_unsafe:
            print("\nREJECTED: submission failed security scan.", file=sys.stderr)
            return 2
        print("\nWARNING: --allow-unsafe set; loading despite scanner rejection.",
              file=sys.stderr)

    model = _load_module(path, args.class_name, _load_config(args.config))

    out_dir = Path(
        args.output_dir or f"runs/candidate_{path.stem}_{int(time.time())}"
    )
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
