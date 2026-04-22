#!/usr/bin/env python
"""Record a live Polymarket BTC 5m event to a parquet fixture.

Runs the harness with a no-op model for the given duration, capturing all
ticks to a parquet so offline replay works without hitting the network.

Default targets: 10 minutes = 2 events, output = tests/fixtures/recorded_event.parquet.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from polybench.harness import Harness, HarnessConfig, format_summary
from polybench.model import FLAT, Model, Signal, Tick


class _NoopModel(Model):
    """Always emits FLAT — records ticks without trading."""

    def on_tick(self, tick: Tick) -> Signal | None:
        return FLAT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=600.0, help="seconds (default 600 = 10 min)")
    p.add_argument(
        "--output",
        default="tests/fixtures/recorded_event.parquet",
        help="destination parquet path",
    )
    p.add_argument("--price-source", default="binance",
                   choices=["binance", "binance-us", "coinbase"])
    args = p.parse_args(argv)

    run_dir = Path(f"runs/record_fixture_tmp")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    cfg = HarnessConfig(
        duration_s=args.duration,
        price_source=args.price_source,
        output_dir=run_dir,
    )
    result = asyncio.run(Harness(model=_NoopModel(), config=cfg).run())
    print(format_summary(result))

    src = run_dir / "ticks.parquet"
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"fixture written to: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
