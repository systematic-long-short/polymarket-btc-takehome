"""Metrics — Sharpe, Sortino, max drawdown, hit rate, timeout rate.

All metrics take equity / return series or ``EventResult`` lists. Results are
pure numbers; the harness formats them into the terminal table and JSON report.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from polybench.model import EventResult

# 1 Hz ticks → 31,536,000 seconds per year.
TICKS_PER_YEAR_AT_1HZ = 365.0 * 24.0 * 3600.0


def tick_returns(equity_curve: Sequence[float]) -> list[float]:
    """Simple percentage returns from an equity curve (first element is 0)."""
    out: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        cur = equity_curve[i]
        if prev == 0.0 or not math.isfinite(prev):
            out.append(0.0)
            continue
        out.append((cur - prev) / prev)
    return out


def sharpe_ratio(
    returns: Sequence[float],
    ticks_per_year: float = TICKS_PER_YEAR_AT_1HZ,
) -> float:
    """Annualized Sharpe of a return series (risk-free rate = 0)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(ticks_per_year)


def sortino_ratio(
    returns: Sequence[float],
    ticks_per_year: float = TICKS_PER_YEAR_AT_1HZ,
) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    downside = [min(r, 0.0) for r in returns]
    dsum = sum(d * d for d in downside)
    dstd = math.sqrt(dsum / (n - 1)) if n > 1 else 0.0
    if dstd == 0.0:
        return 0.0
    return (mean / dstd) * math.sqrt(ticks_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Worst peak-to-trough drawdown as a fraction of the peak (0..1)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0.0:
            dd = (peak - equity) / peak
            if dd > worst:
                worst = dd
    return worst


def hit_rate(events: Iterable[EventResult]) -> float:
    """Fraction of events closed with positive PnL (intra-event + resolution)."""
    events_list = list(events)
    if not events_list:
        return 0.0
    wins = sum(1 for e in events_list if e.pnl_total > 0.0)
    return wins / len(events_list)


def outcome_accuracy(events: Iterable[EventResult]) -> float:
    """Secondary metric: fraction of resolved events where the candidate's
    resolution-PnL was positive. This approximates "positioned on the winning
    side at settlement" without separately tracking direction.
    """
    events_list = [e for e in events if e.resolved_outcome != "UNKNOWN"]
    if not events_list:
        return 0.0
    correct = sum(1 for e in events_list if e.pnl_resolution > 0.0)
    return correct / len(events_list)


def timeout_rate(events: Iterable[EventResult]) -> float:
    events_list = list(events)
    total_ticks = sum(e.n_ticks for e in events_list)
    total_timeouts = sum(e.n_timeouts for e in events_list)
    if total_ticks == 0:
        return 0.0
    return total_timeouts / total_ticks


def summarize(
    *,
    starting_capital: float,
    final_equity: float,
    equity_curve: Sequence[float],
    events: Sequence[EventResult],
    ticks_per_year: float = TICKS_PER_YEAR_AT_1HZ,
) -> dict[str, Any]:
    returns = tick_returns(equity_curve)
    pnl = final_equity - starting_capital
    pnl_pct = pnl / starting_capital if starting_capital else 0.0
    sharpe = sharpe_ratio(returns, ticks_per_year=ticks_per_year)
    # |Sharpe| × sign(PnL): magnitude of consistency, signed by P&L direction.
    # A losing model with small consistent losses has low-magnitude negative
    # score; a winning model with steady gains has high-magnitude positive.
    primary_score = abs(sharpe) * (1.0 if pnl >= 0.0 else -1.0)
    intra = sum(e.pnl_intra_event for e in events)
    reso = sum(e.pnl_resolution for e in events)
    denom = abs(intra) + abs(reso)
    intra_fraction = (intra / denom) if denom > 0 else 0.0
    return {
        "starting_capital": starting_capital,
        "final_equity": final_equity,
        "pnl_total": pnl,
        "pnl_pct": pnl_pct,
        "sharpe": sharpe,
        "sortino": sortino_ratio(returns, ticks_per_year=ticks_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "hit_rate": hit_rate(events),
        "outcome_accuracy": outcome_accuracy(events),
        "timeout_rate": timeout_rate(events),
        "n_events": len(events),
        "n_trades": sum(e.n_trades for e in events),
        "n_ticks": sum(e.n_ticks for e in events),
        "pnl_intra_event": intra,
        "pnl_resolution": reso,
        "intra_vs_resolution_fraction": intra_fraction,  # 1 = all intra, -1 = all resolution
        "primary_score": primary_score,                  # Sharpe × sign(PnL)
    }
