#!/usr/bin/env python
"""Validate a live scoring run.

This checks that the report score is internally correct and that the tick
parquet contains both Polymarket token quotes and live Binance BTC data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _expected_score(metrics: dict[str, Any]) -> float:
    pnl = float(metrics["pnl_total"])
    sharpe = max(float(metrics["sharpe"]), 0.0)
    drawdown = float(metrics["max_drawdown"])
    if sharpe == 0.0:
        return 0.0
    return pnl * sharpe * max(0.0, 1.0 - drawdown)


def _positive_rows(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    frame = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    return frame.gt(0.0).all(axis=1)


def _validate_track(payload: dict[str, Any], track: str, *, min_events: int, min_ticks: int) -> dict[str, Any]:
    section = payload[track]
    metrics = section["metrics"]
    for key in ("pnl_total", "sharpe", "max_drawdown", "primary_score"):
        _require(_finite(metrics.get(key)), f"{track}.{key} is not finite")
    _require(int(metrics["n_events"]) >= min_events, f"{track} has too few events")
    _require(int(metrics["n_ticks"]) >= min_ticks, f"{track} has too few ticks")
    _require(len(section.get("events", [])) == int(metrics["n_events"]), f"{track} event count mismatch")

    top_pnl = float(section["pnl_total"])
    metric_pnl = float(metrics["pnl_total"])
    _require(math.isclose(top_pnl, metric_pnl, rel_tol=1e-9, abs_tol=1e-6), f"{track} pnl mismatch")

    expected = _expected_score(metrics)
    actual = float(metrics["primary_score"])
    _require(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6),
        f"{track} primary score mismatch: expected {expected}, got {actual}",
    )
    unknown = [e for e in section.get("events", []) if e.get("resolved_outcome") == "UNKNOWN"]
    return {
        "pnl_total": top_pnl,
        "primary_score": actual,
        "events": int(metrics["n_events"]),
        "ticks": int(metrics["n_ticks"]),
        "trades": int(metrics["n_trades"]),
        "unknown_events": len(unknown),
    }


def validate_live_run(
    *,
    report_path: Path,
    ticks_path: Path,
    min_duration_s: float = 3600.0,
    min_events: int = 10,
    require_binance: bool = True,
    allow_unknown: bool = False,
) -> dict[str, Any]:
    payload = json.loads(report_path.read_text())
    df = pd.read_parquet(ticks_path, engine="pyarrow")

    _require(not df.empty, "ticks parquet is empty")
    report_duration = float(payload["ended_ts"]) - float(payload["started_ts"])
    tick_duration = float(df["ts"].max()) - float(df["ts"].min())
    _require(report_duration >= min_duration_s, "report duration is shorter than requested")
    # The harness runs for wall-clock duration, but it intentionally skips
    # model ticks during discovery and resolution gaps between 5-minute events.
    # Gate active coverage by event/tick counts below instead of requiring the
    # first-to-last tick timestamp to span the full wall-clock run.

    min_ticks = max(1, int(min_duration_s * 0.70))
    model = _validate_track(payload, "model", min_events=min_events, min_ticks=min_ticks)
    baseline = _validate_track(payload, "baseline", min_events=min_events, min_ticks=min_ticks)

    poly_mid_cols = ("up_mid", "down_mid")
    poly_quote_cols = ("up_bid", "up_ask", "down_bid", "down_ask")
    poly_cols = poly_quote_cols + poly_mid_cols
    btc_cols = ("btc_last", "btc_bid", "btc_ask")
    for col in poly_cols:
        _require(col in df.columns, f"missing Polymarket column {col}")
    for col in btc_cols:
        _require(col in df.columns, f"missing BTC column {col}")

    if "resolved_outcome" in df.columns:
        active_mask = df["resolved_outcome"].fillna("").astype(str).eq("")
    else:
        active_mask = pd.Series(True, index=df.index)
    active_df = df.loc[active_mask]

    poly_valid_mask = _positive_rows(active_df, poly_mid_cols)
    poly_two_sided_mask = poly_valid_mask & _positive_rows(active_df, poly_quote_cols)
    poly_valid_rows = int(poly_valid_mask.sum())
    poly_two_sided_rows = int(poly_two_sided_mask.sum())
    poly_one_sided_rows = poly_valid_rows - poly_two_sided_rows
    btc_rows = int(_positive_rows(active_df, btc_cols).sum())
    _require(poly_valid_rows >= min_ticks, "too few rows have valid Polymarket mids")
    _require(btc_rows >= min_ticks, "too few rows have complete BTC quotes")

    sources: list[str] = []
    if "btc_source" in df.columns:
        sources = sorted(str(s) for s in df["btc_source"].dropna().unique())
    if require_binance:
        _require("btc_source" in df.columns, "ticks parquet lacks btc_source")
        binance_rows = int(df.loc[active_mask, "btc_source"].astype(str).str.startswith("binance").sum())
        _require(binance_rows >= min_ticks, "too few rows identify Binance as BTC source")

    if not allow_unknown:
        _require(model["unknown_events"] == 0, "model has UNKNOWN events")
        _require(baseline["unknown_events"] == 0, "baseline has UNKNOWN events")

    return {
        "report": str(report_path),
        "ticks": str(ticks_path),
        "report_duration_s": report_duration,
        "tick_duration_s": tick_duration,
        "tick_rows": int(len(df)),
        "active_tick_rows": int(len(active_df)),
        "settlement_rows": int(len(df) - len(active_df)),
        "polymarket_valid_rows": poly_valid_rows,
        "polymarket_two_sided_rows": poly_two_sided_rows,
        "polymarket_one_sided_rows": poly_one_sided_rows,
        "btc_complete_rows": btc_rows,
        "btc_sources": sources,
        "model": model,
        "baseline": baseline,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ticks", required=True, type=Path)
    parser.add_argument("--min-duration", type=float, default=3600.0)
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--require-binance", action="store_true")
    parser.add_argument("--allow-unknown", action="store_true")
    args = parser.parse_args(argv)

    summary = validate_live_run(
        report_path=args.report,
        ticks_path=args.ticks,
        min_duration_s=args.min_duration,
        min_events=args.min_events,
        require_binance=args.require_binance,
        allow_unknown=args.allow_unknown,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
