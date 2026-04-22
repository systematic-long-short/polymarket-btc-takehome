"""Shared pytest fixtures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polybench.recorder import TICK_COLUMNS


FIXTURES_DIR = Path(__file__).parent / "fixtures"
RECORDED_EVENT_PATH = FIXTURES_DIR / "recorded_event.parquet"
SYNTHETIC_EVENT_PATH = FIXTURES_DIR / "synthetic_event.parquet"


def _make_synthetic(path: Path, *, n_events: int = 2, ticks_per_event: int = 300) -> Path:
    """Generate a synthetic two-event tick parquet for offline tests.

    BTC drifts randomly; UP/DOWN token prices mean-revert around a 0.5
    mid. First event resolves UP (outcomePrices=[1,0]); second resolves DOWN.
    """
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    ts = 1_700_000_000.0  # arbitrary
    btc = 65_000.0
    for ev_idx in range(n_events):
        event_id = f"SYN-{ev_idx}"
        slug = f"synthetic-{ev_idx}"
        up_mid = 0.5
        for i in range(ticks_per_event):
            btc += rng.normal(0.0, 5.0)
            # UP price drifts toward the eventual winner.
            drift = 0.0003 if ev_idx == 0 else -0.0003
            up_mid = 0.5 + drift * (i + 1) + rng.normal(0.0, 0.005)
            up_mid = float(np.clip(up_mid, 0.01, 0.99))
            down_mid = 1.0 - up_mid
            rows.append({
                "ts": ts,
                "event_id": event_id,
                "slug": slug,
                "time_to_resolve": (ticks_per_event - i) * 1.0,
                "btc_last": btc,
                "btc_bid": btc - 1.0,
                "btc_ask": btc + 1.0,
                "up_bid": up_mid - 0.005,
                "up_ask": up_mid + 0.005,
                "up_mid": up_mid,
                "down_bid": down_mid - 0.005,
                "down_ask": down_mid + 0.005,
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
            ts += 1.0
        # Settlement row
        winner_up = 1.0 if ev_idx == 0 else 0.0
        rows.append({
            "ts": ts,
            "event_id": event_id,
            "slug": slug,
            "time_to_resolve": 0.0,
            "btc_last": btc,
            "btc_bid": btc - 1.0,
            "btc_ask": btc + 1.0,
            "up_bid": up_mid - 0.005,
            "up_ask": up_mid + 0.005,
            "up_mid": up_mid,
            "down_bid": down_mid - 0.005,
            "down_ask": down_mid + 0.005,
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
            "resolution_up": winner_up,
            "resolution_down": 1.0 - winner_up,
            "resolved_outcome": "UP" if winner_up > 0.5 else "DOWN",
        })
        ts += 2.0
    df = pd.DataFrame(rows, columns=list(TICK_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


@pytest.fixture(scope="session")
def synthetic_fixture(tmp_path_factory) -> Path:
    path = FIXTURES_DIR / "synthetic_event.parquet"
    if not path.exists():
        _make_synthetic(path)
    return path


@pytest.fixture(scope="session")
def any_event_fixture(synthetic_fixture: Path) -> Path:
    """Prefer the real recorded fixture if committed, else synthetic."""
    if RECORDED_EVENT_PATH.exists():
        return RECORDED_EVENT_PATH
    return synthetic_fixture
