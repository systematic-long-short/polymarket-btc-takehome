"""Offline replay engine.

Reads a parquet tick log recorded by the live harness (see ``recorder.py``),
reconstructs ``Tick`` objects, and feeds them into a fresh ``Model`` instance.
Signals execute on the next recorded tick, matching the live harness.
Deterministic, no network. Intended for rapid iteration on candidate code.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from polybench.metrics import summarize
from polybench.model import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick
from polybench.pnl import BookTop, PaperSimulator

log = logging.getLogger("polybench.replay")


@dataclass
class ReplayConfig:
    starting_capital: float = 1000.0
    slippage_bps: float = 50.0
    fee_rate: float = 0.072
    model_budget_s: float = 0.5
    output_dir: Path = Path("runs/replay")
    scratch_dir: Path = Path("runs/replay/scratch")
    price_window_size: int = 300
    up_mid_window_size: int = 300


def replay(
    model: Model,
    parquet_path: Path | str,
    config: ReplayConfig | None = None,
    on_summary: Callable[[str], None] | None = None,
    baseline_model: Model | None = None,
) -> RunResult:
    cfg = config or ReplayConfig()
    cfg.output_dir = Path(cfg.output_dir)
    cfg.scratch_dir = Path(cfg.scratch_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)

    if baseline_model is None:
        from polybench.baselines import MomentumBaseline
        baseline_model = MomentumBaseline()

    df = pd.read_parquet(Path(parquet_path), engine="pyarrow")
    if df.empty:
        raise ValueError(f"replay parquet is empty: {parquet_path}")

    simulator = PaperSimulator(
        starting_capital=cfg.starting_capital,
        slippage_bps=cfg.slippage_bps,
        fee_rate=cfg.fee_rate,
    )
    baseline_simulator = PaperSimulator(
        starting_capital=cfg.starting_capital,
        slippage_bps=cfg.slippage_bps,
        fee_rate=cfg.fee_rate,
    )
    btc_window: deque[float] = deque(maxlen=cfg.price_window_size)
    up_mid_window: deque[float] = deque(maxlen=cfg.up_mid_window_size)
    equity_curve: list[float] = []
    baseline_equity_curve: list[float] = []
    current_event: str | None = None
    pending_signal: Signal | None = None
    baseline_pending_signal: Signal | None = None

    started_ts = float(df["ts"].iloc[0])

    def _safe_on_tick(m: Model, t: Tick) -> Signal | None:
        try:
            sig = m.on_tick(t)
        except Exception:  # noqa: BLE001
            log.exception("%s.on_tick raised during replay", type(m).__name__)
            return None
        if sig is None:
            return None
        if not isinstance(sig, Signal):
            return None
        try:
            side = sig.side if isinstance(sig.side, Side) else Side(str(sig.side))
            size_raw = float(sig.size)
            conf_raw = float(sig.confidence)
        except (TypeError, ValueError):
            log.warning("%s.on_tick returned an invalid Signal during replay: %r", type(m).__name__, sig)
            return None
        size = max(0.0, min(1.0, size_raw)) if math.isfinite(size_raw) else 0.0
        conf = max(0.0, min(1.0, conf_raw)) if math.isfinite(conf_raw) else 0.0
        return Signal(side=side, size=size, confidence=conf)

    for row in df.itertuples(index=False):
        event_id = str(getattr(row, "event_id", "") or "")
        slug = str(getattr(row, "slug", "") or "")
        is_settlement_row = not math.isnan(
            float(getattr(row, "resolution_up", float("nan")))
        )

        # Event rollover detection.
        if event_id and event_id != current_event:
            if current_event is not None:
                simulator.finish_event(float(row.ts), None, None)
                baseline_simulator.finish_event(float(row.ts), None, None)
            current_event = event_id
            pending_signal = None
            baseline_pending_signal = None
            up_mid_window.clear()
            simulator.start_event(
                event_id=event_id,
                slug=slug,
                ts=float(row.ts),
                up_mid=float(row.up_mid) or 0.5,
                down_mid=float(row.down_mid) or 0.5,
            )
            baseline_simulator.start_event(
                event_id=event_id,
                slug=slug,
                ts=float(row.ts),
                up_mid=float(row.up_mid) or 0.5,
                down_mid=float(row.down_mid) or 0.5,
            )
            market_info = MarketInfo(
                event_id=event_id,
                slug=slug,
                question="",
                end_date_ts=float(row.ts) + float(getattr(row, "time_to_resolve", 0.0) or 0.0),
                up_token_id="",
                down_token_id="",
                scratch_dir=cfg.scratch_dir,
            )
            try:
                model.on_start(market_info)
            except Exception:  # noqa: BLE001
                log.exception("model.on_start raised during replay")
            try:
                baseline_model.on_start(market_info)
            except Exception:  # noqa: BLE001
                log.exception("baseline_model.on_start raised during replay")

        if is_settlement_row and current_event is not None:
            up_price = float(row.resolution_up)
            down_price = float(row.resolution_down)
            simulator.finish_event(float(row.ts), up_price, down_price)
            baseline_simulator.finish_event(float(row.ts), up_price, down_price)
            current_event = None
            pending_signal = None
            baseline_pending_signal = None
            continue

        # Update rolling windows from the recorded row.
        btc_last = float(row.btc_last)
        btc_source = str(getattr(row, "btc_source", "recorded") or "recorded")
        up_mid = float(row.up_mid)
        if btc_last > 0.0 and not math.isnan(btc_last):
            btc_window.append(btc_last)
        if up_mid > 0.0 and not math.isnan(up_mid):
            up_mid_window.append(up_mid)

        tick = Tick(
            ts=float(row.ts),
            time_to_resolve=float(getattr(row, "time_to_resolve", 0.0) or 0.0),
            btc_last=btc_last,
            btc_bid=float(row.btc_bid),
            btc_ask=float(row.btc_ask),
            up_bid=float(row.up_bid),
            up_ask=float(row.up_ask),
            up_mid=up_mid,
            down_bid=float(row.down_bid),
            down_ask=float(row.down_ask),
            down_mid=float(row.down_mid),
            btc_recent=tuple(btc_window),
            up_mid_recent=tuple(up_mid_window),
            event_id=event_id,
            btc_source=btc_source,
        )

        up_top = BookTop(best_bid=tick.up_bid, best_ask=tick.up_ask, mid=tick.up_mid)
        down_top = BookTop(best_bid=tick.down_bid, best_ask=tick.down_ask, mid=tick.down_mid)
        simulator.apply_signal(pending_signal, up_top, down_top)
        equity = simulator.mark_to_market(up_top, down_top, tick.ts)
        equity_curve.append(equity)
        baseline_simulator.apply_signal(baseline_pending_signal, up_top, down_top)
        baseline_equity = baseline_simulator.mark_to_market(up_top, down_top, tick.ts)
        baseline_equity_curve.append(baseline_equity)

        pending_signal = _safe_on_tick(model, tick)
        baseline_pending_signal = _safe_on_tick(baseline_model, tick)

    # Close dangling event if any (unresolved at EOF).
    if current_event is not None:
        simulator.finish_event(float(df["ts"].iloc[-1]), None, None)
        baseline_simulator.finish_event(float(df["ts"].iloc[-1]), None, None)

    def _final_equity(sim, curve):
        if abs(sim.position.cash) > 1e-9:
            return sim.position.cash
        return curve[-1] if curve else sim.starting_capital

    ended_ts = float(df["ts"].iloc[-1])
    starting = simulator.starting_capital
    final_equity = _final_equity(simulator, equity_curve)
    baseline_final_equity = _final_equity(baseline_simulator, baseline_equity_curve)

    metrics = summarize(
        starting_capital=starting,
        final_equity=final_equity,
        equity_curve=[starting, *equity_curve],
        events=simulator.completed_events,
    )
    baseline_metrics = summarize(
        starting_capital=starting,
        final_equity=baseline_final_equity,
        equity_curve=[starting, *baseline_equity_curve],
        events=baseline_simulator.completed_events,
    )
    result = RunResult(
        started_ts=started_ts,
        ended_ts=ended_ts,
        starting_capital=starting,
        final_equity=final_equity,
        pnl_total=final_equity - starting,
        pnl_pct=(final_equity - starting) / starting if starting else 0.0,
        events=simulator.completed_events,
        metrics=metrics,
        baseline_events=baseline_simulator.completed_events,
        baseline_metrics=baseline_metrics,
        baseline_final_equity=baseline_final_equity,
        baseline_pnl_total=baseline_final_equity - starting,
        baseline_pnl_pct=(
            (baseline_final_equity - starting) / starting if starting else 0.0
        ),
    )
    try:
        model.on_finish(result)
    except Exception:  # noqa: BLE001
        log.exception("model.on_finish raised during replay")

    # Write the two-column report next to the input.
    report_path = cfg.output_dir / "report.json"
    payload: dict[str, Any] = {
        "started_ts": result.started_ts,
        "ended_ts": result.ended_ts,
        "starting_capital": result.starting_capital,
        "model": {
            "final_equity": result.final_equity,
            "pnl_total": result.pnl_total,
            "pnl_pct": result.pnl_pct,
            "metrics": result.metrics,
            "events": [asdict(e) if is_dataclass(e) else dict(e) for e in result.events],
        },
        "baseline": {
            "final_equity": result.baseline_final_equity,
            "pnl_total": result.baseline_pnl_total,
            "pnl_pct": result.baseline_pnl_pct,
            "metrics": result.baseline_metrics,
            "events": [
                asdict(e) if is_dataclass(e) else dict(e) for e in result.baseline_events
            ],
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    if on_summary:
        from polybench.harness import format_summary
        on_summary(format_summary(result))

    return result
