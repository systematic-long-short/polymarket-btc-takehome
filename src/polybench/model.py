"""Core dataclasses + Model ABC — the surface a candidate submission touches.

A candidate submits ``model_submission.py`` containing ``class ModelSubmission(Model)``.
The harness constructs it with an optional config dict, calls ``on_start`` once when
it locks onto a live event, calls ``on_tick`` once per second during the event,
and calls ``on_finish`` when the event resolves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class Side(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class MarketInfo:
    """Static per-event context, passed to ``on_start``."""

    event_id: str
    slug: str
    question: str
    end_date_ts: float             # unix seconds when the event resolves
    up_token_id: str
    down_token_id: str
    scratch_dir: Path              # writable scratch area (per run)


@dataclass(frozen=True, slots=True)
class Tick:
    """Snapshot delivered to ``on_tick`` every second.

    All prices in USD except polymarket token prices which are in [0, 1].
    Rolling windows are newest-last (``prices[-1]`` is the most recent sample).
    """

    ts: float                      # unix seconds, tick dispatch time
    time_to_resolve: float         # seconds remaining until event end_date_ts

    btc_last: float                # last BTC trade (from price feed)
    btc_bid: float
    btc_ask: float

    up_bid: float                  # Polymarket UP token best bid ($)
    up_ask: float
    up_mid: float
    down_bid: float
    down_ask: float
    down_mid: float

    btc_recent: Sequence[float] = field(default_factory=tuple)     # rolling window of btc_last
    up_mid_recent: Sequence[float] = field(default_factory=tuple)  # rolling window of up_mid

    event_id: str = ""


@dataclass(frozen=True, slots=True)
class Signal:
    """Desired position emitted by ``on_tick``.

    ``side``     — UP, DOWN, or FLAT (no exposure).
    ``size``     — fraction of starting capital to allocate, clipped to [0, 1].
    ``confidence`` — optional [0, 1]; recorded for analytics, does not affect fills.
    """

    side: Side
    size: float = 0.0
    confidence: float = 0.0


FLAT = Signal(side=Side.FLAT, size=0.0, confidence=0.0)


@dataclass(frozen=True, slots=True)
class EventResult:
    """Per-event outcome delivered inside ``RunResult.events``."""

    event_id: str
    slug: str
    start_ts: float
    end_ts: float
    resolved_outcome: str          # "UP", "DOWN", or "UNKNOWN"
    pnl_total: float               # $ realized + mtm through settlement
    pnl_intra_event: float         # $ captured before last tick of event
    pnl_resolution: float          # $ captured at settlement ($1/$0 snap)
    n_trades: int
    n_ticks: int
    n_timeouts: int


@dataclass(frozen=True, slots=True)
class RunResult:
    """Delivered to ``on_finish`` when the harness stops.

    ``metrics`` carries the primary model's summary. ``baseline_metrics`` and
    ``baseline_events`` mirror the same harness run against the reference
    MomentumBaseline on the same tape, so the report can show a two-column
    Model-vs-Baseline comparison without re-running the market.
    """

    started_ts: float
    ended_ts: float
    starting_capital: float
    final_equity: float
    pnl_total: float
    pnl_pct: float
    events: Sequence[EventResult] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_events: Sequence[EventResult] = field(default_factory=tuple)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_final_equity: float = 0.0
    baseline_pnl_total: float = 0.0
    baseline_pnl_pct: float = 0.0


class Model(ABC):
    """Abstract base class candidates subclass.

    Candidates override ``on_tick`` (required) and optionally ``on_start`` and
    ``on_finish``. State is owned by ``self`` — no shared globals.

    The harness enforces a 500 ms wall-clock budget on ``on_tick``. Overrun
    causes the signal to be dropped for that tick and counted toward
    ``timeout_rate``; it does NOT raise to the candidate.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    def on_start(self, market_info: MarketInfo) -> None:  # noqa: B027 — optional override
        """Called once per event, immediately after the harness locks onto it."""

    @abstractmethod
    def on_tick(self, tick: Tick) -> Signal | None:
        """Return the desired position for the current tick.

        Return ``None`` or ``FLAT`` to request no exposure.
        """

    def on_finish(self, result: RunResult) -> None:  # noqa: B027 — optional override
        """Called once when the harness stops. Use for teardown/logging."""
