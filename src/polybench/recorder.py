"""Parquet tick log. One file per run.

Schema (one row per tick):
    ts                   float64   unix seconds
    event_id             string
    slug                 string
    time_to_resolve      float64   seconds until endDate
    btc_last             float64
    btc_bid              float64
    btc_ask              float64
    btc_source           string    source for BTC fields; e.g. binance
    up_bid               float64   Polymarket UP token best bid
    up_ask               float64
    up_mid               float64
    down_bid             float64
    down_ask             float64
    down_mid             float64
    signal_side          string    "UP" | "DOWN" | "FLAT" | "NONE" queued for next tick
    signal_size          float64
    signal_confidence    float64
    position_up          float64   shares held after applying prior tick's signal
    position_down        float64
    cash                 float64
    equity               float64
    fills_this_tick      int64
    timeout              bool      True if on_tick overran 500 ms
    baseline_*           (same shape as above for the MomentumBaseline
                          track run in parallel on the same tape)
    resolution_up        float64   NaN except on the settlement row
    resolution_down      float64   NaN except on the settlement row
    resolved_outcome     string    "UP" | "DOWN" | "UNKNOWN" | ""
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("polybench.recorder")


@dataclass
class TickRow:
    ts: float = 0.0
    event_id: str = ""
    slug: str = ""
    time_to_resolve: float = 0.0
    btc_last: float = 0.0
    btc_bid: float = 0.0
    btc_ask: float = 0.0
    btc_source: str = "polymarket"
    up_bid: float = 0.0
    up_ask: float = 0.0
    up_mid: float = 0.0
    down_bid: float = 0.0
    down_ask: float = 0.0
    down_mid: float = 0.0
    signal_side: str = "NONE"
    signal_size: float = 0.0
    signal_confidence: float = 0.0
    position_up: float = 0.0
    position_down: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    fills_this_tick: int = 0
    timeout: bool = False
    # Parallel baseline track — same columns for the MomentumBaseline run on
    # the same tape. Identical to the model columns when the caller's model
    # IS MomentumBaseline.
    baseline_signal_side: str = "NONE"
    baseline_signal_size: float = 0.0
    baseline_signal_confidence: float = 0.0
    baseline_position_up: float = 0.0
    baseline_position_down: float = 0.0
    baseline_cash: float = 0.0
    baseline_equity: float = 0.0
    baseline_fills_this_tick: int = 0
    baseline_timeout: bool = False
    resolution_up: float = float("nan")
    resolution_down: float = float("nan")
    resolved_outcome: str = ""


TICK_COLUMNS: tuple[str, ...] = tuple(f.name for f in TickRow.__dataclass_fields__.values())


class Recorder:
    def __init__(self, output_path: Path | str) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    @property
    def output_path(self) -> Path:
        return self._path

    def record(self, row: TickRow) -> None:
        self._rows.append(asdict(row))

    def record_dict(self, row: dict[str, Any]) -> None:
        filled = {col: row.get(col, _default_for(col)) for col in TICK_COLUMNS}
        self._rows.append(filled)

    def __len__(self) -> int:
        return len(self._rows)

    def flush(self) -> Path:
        if not self._rows:
            # Still produce an empty file with the right schema.
            df = pd.DataFrame(columns=list(TICK_COLUMNS))
        else:
            df = pd.DataFrame(self._rows, columns=list(TICK_COLUMNS))
        df.to_parquet(self._path, engine="pyarrow", index=False)
        log.info("recorder: wrote %d rows to %s", len(self._rows), self._path)
        return self._path

    @staticmethod
    def load(path: Path | str) -> pd.DataFrame:
        return pd.read_parquet(Path(path), engine="pyarrow")


def _default_for(column: str) -> Any:
    if column in {
        "event_id",
        "slug",
        "btc_source",
        "signal_side",
        "baseline_signal_side",
        "resolved_outcome",
    }:
        return "NONE" if column in {"signal_side", "baseline_signal_side"} else ""
    if column in {"timeout", "baseline_timeout"}:
        return False
    if column in {"fills_this_tick", "baseline_fills_this_tick"}:
        return 0
    if column in {"resolution_up", "resolution_down"}:
        return math.nan
    return 0.0
