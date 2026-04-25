#!/usr/bin/env python
"""DEVELOPER-ITERATION TOOL — NOT USED FOR SCORING.

Synthesizes a replay-compatible parquet from real Binance BTC prices plus
heuristic Polymarket token prices. Intended for offline iteration when you
don't have internet access to Polymarket.

    Scoring is performed EXCLUSIVELY against the live Polymarket market by
    `scripts/run_baseline.py` / `scripts/run_candidate.py`. The output of
    this script is never accepted as an official score. See EVALUATION.md.

Captures real BTC prices from the Binance WebSocket over the requested
duration, then synthesizes Polymarket UP/DOWN token prices whose mid
responds to BTC momentum.

Usage:
    python scripts/synthesize_fixture.py --duration 600 \\
        --output tests/fixtures/recorded_event.parquet

The output parquet is schema-compatible with live recordings, so replay
treats it identically.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from polybench.pricefeed import PriceFeed
from polybench.recorder import TICK_COLUMNS


async def _capture_btc(duration_s: float, source: str) -> list[tuple[float, float, float, float]]:
    """Capture (ts, last, bid, ask) tuples from the live price feed at 1 Hz."""
    feed = PriceFeed(source=source, window_size=int(duration_s) + 30)
    await feed.start()
    ok = await feed.wait_ready(timeout=15.0)
    if not ok:
        raise RuntimeError(f"price feed {source} never became ready")
    samples: list[tuple[float, float, float, float]] = []
    end = time.time() + duration_s
    try:
        while time.time() < end:
            snap = feed.snapshot()
            if snap.last > 0.0:
                samples.append((time.time(), snap.last, snap.bid, snap.ask))
            await asyncio.sleep(1.0)
    finally:
        await feed.stop()
    return samples


def _synthesize_polymarket(
    btc_samples: list[tuple[float, float, float, float]],
    *,
    event_length_s: int = 300,
    winner_bias_bps: float = 15.0,
) -> pd.DataFrame:
    """Build a tick DataFrame from BTC samples.

    Splits the capture into consecutive 5-minute events. For each event:
      - Computes BTC return from event start to end.
      - Sets the "true" winner to UP if return > 0 else DOWN.
      - Polymarket UP price mid = 0.5 + response(BTC momentum) + noise,
        drifts toward 1.0 (winner) or 0.0 (loser) over the event window.
    """
    if not btc_samples:
        raise ValueError("no BTC samples captured")
    rng = np.random.default_rng(seed=17)

    rows: list[dict] = []
    n = len(btc_samples)
    total_events = max(1, n // event_length_s)

    for ev_idx in range(total_events):
        start_i = ev_idx * event_length_s
        end_i = min((ev_idx + 1) * event_length_s, n)
        if end_i - start_i < 10:
            break
        event_id = f"SYN-{ev_idx}"
        slug = f"btc-updown-5m-synthetic-{ev_idx}"
        btc_start = btc_samples[start_i][1]
        btc_end = btc_samples[end_i - 1][1]
        winner_up = btc_end >= btc_start      # UP wins if BTC ended higher
        duration = end_i - start_i

        for step, (ts, last, bid, ask) in enumerate(btc_samples[start_i:end_i]):
            # Progress 0.0 at start → 1.0 at end.
            progress = step / max(duration - 1, 1)
            # Momentum component: BTC return so far, scaled.
            if step == 0 or last <= 0.0:
                momentum = 0.0
            else:
                anchor = btc_samples[start_i][1]
                momentum = ((last - anchor) / anchor) * (10_000.0 / winner_bias_bps)
                momentum = float(np.clip(momentum, -0.4, 0.4))
            # Drift toward winner proportional to progress.
            drift = (0.4 if winner_up else -0.4) * progress
            noise = rng.normal(0.0, 0.01)
            up_mid = 0.5 + 0.5 * (momentum * (1 - progress) + drift) + noise
            up_mid = float(np.clip(up_mid, 0.01, 0.99))
            down_mid = 1.0 - up_mid
            spread = 0.005
            rows.append({
                "ts": ts,
                "event_id": event_id,
                "slug": slug,
                "time_to_resolve": float(duration - step),
                "btc_last": last,
                "btc_bid": bid if bid > 0 else last - 0.5,
                "btc_ask": ask if ask > 0 else last + 0.5,
                "up_bid": max(0.01, up_mid - spread),
                "up_ask": min(0.99, up_mid + spread),
                "up_mid": up_mid,
                "down_bid": max(0.01, down_mid - spread),
                "down_ask": min(0.99, down_mid + spread),
                "down_mid": down_mid,
                "signal_side": "NONE",
                "signal_size": 0.0,
                "signal_confidence": 0.0,
                "position_up": 0.0,
                "position_down": 0.0,
                "cash": 1000.0,
                "equity": 1000.0,
                "fills_this_tick": 0,
                "timeout": False,
                "resolution_up": math.nan,
                "resolution_down": math.nan,
                "resolved_outcome": "",
            })
        # Settlement row
        winner = 1.0 if winner_up else 0.0
        final_row = dict(rows[-1])
        final_row["ts"] += 1.0
        final_row["time_to_resolve"] = 0.0
        final_row["resolution_up"] = winner
        final_row["resolution_down"] = 1.0 - winner
        final_row["resolved_outcome"] = "UP" if winner_up else "DOWN"
        rows.append(final_row)

    df = pd.DataFrame(rows, columns=list(TICK_COLUMNS))
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=600.0,
                   help="seconds of BTC data to capture (default 600 = 10 min = 2 events)")
    p.add_argument("--output", default="tests/fixtures/recorded_event.parquet")
    p.add_argument("--price-source", default="binance",
                   choices=["binance"])
    p.add_argument("--event-length", type=int, default=300,
                   help="seconds per synthetic event (default 300 = 5 min)")
    args = p.parse_args(argv)

    print(f"capturing {args.duration}s of BTC from {args.price_source}...")
    samples = asyncio.run(_capture_btc(args.duration, args.price_source))
    print(f"captured {len(samples)} samples")

    df = _synthesize_polymarket(samples, event_length_s=args.event_length)
    n_events = len(df["event_id"].unique())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"wrote {len(df)} rows across {n_events} events to {out}")
    print(f"  BTC range: {df['btc_last'].min():.2f} – {df['btc_last'].max():.2f}")
    print(f"  outcomes: {df[df['resolved_outcome'] != '']['resolved_outcome'].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
