from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from polybench.reconciliation import reconcile_run_files
from polybench.recorder import TICK_COLUMNS


def _event(track: str) -> dict:
    prefix = "baseline_" if track == "baseline" else ""
    return {
        "event_id": "E1",
        "slug": "btc-updown-test",
        "start_ts": 0.0,
        "end_ts": 300.0,
        "resolved_outcome": "UNKNOWN",
        "pnl_total": 0.0,
        "pnl_intra_event": 0.0,
        "pnl_resolution": 0.0,
        "n_trades": 1,
        "n_ticks": 2,
        "n_timeouts": 0,
        "pending_resolution_up_shares": 100.0 if not prefix else 50.0,
        "pending_resolution_down_shares": 0.0,
        "pending_resolution_up_mark": 0.50,
        "pending_resolution_down_mark": 0.50,
    }


def test_reconcile_run_files_updates_report_ticks_and_status(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    payload = {
        "started_ts": 0.0,
        "ended_ts": 301.0,
        "starting_capital": 1000.0,
        "model": {
            "final_equity": 1000.0,
            "pnl_total": 0.0,
            "pnl_pct": 0.0,
            "metrics": {},
            "events": [_event("model")],
        },
        "baseline": {
            "final_equity": 1000.0,
            "pnl_total": 0.0,
            "pnl_pct": 0.0,
            "metrics": {},
            "events": [_event("baseline")],
        },
        "scoring_status": {"state": "pending_resolution", "unresolved_event_count": 2},
        "metadata": {"feed_health": {"unknown_event_count": 2}},
    }
    report.write_text(json.dumps(payload))

    rows = []
    for ts, resolved in [(1.0, ""), (300.0, "UNKNOWN")]:
        row = {col: 0 for col in TICK_COLUMNS}
        row.update({
            "ts": ts,
            "event_id": "E1",
            "slug": "btc-updown-test",
            "up_bid": 0.49,
            "up_ask": 0.51,
            "up_mid": 0.50,
            "down_bid": 0.49,
            "down_ask": 0.51,
            "down_mid": 0.50,
            "cash": 1000.0,
            "equity": 1000.0,
            "baseline_cash": 1000.0,
            "baseline_equity": 1000.0,
            "resolution_up": float("nan"),
            "resolution_down": float("nan"),
            "resolved_outcome": resolved,
        })
        rows.append(row)
    pd.DataFrame(rows, columns=list(TICK_COLUMNS)).to_parquet(ticks, engine="pyarrow", index=False)

    summary = reconcile_run_files(
        report_path=report,
        ticks_path=ticks,
        resolutions={"btc-updown-test": (1.0, 0.0)},
    )

    updated = json.loads(report.read_text())
    updated_ticks = pd.read_parquet(ticks)
    assert summary["updated_event_count"] == 2
    assert updated["scoring_status"]["state"] == "final"
    assert updated["model"]["events"][0]["resolved_outcome"] == "UP"
    assert updated["model"]["events"][0]["pnl_resolution"] == pytest.approx(50.0)
    assert updated["model"]["final_equity"] == pytest.approx(1050.0)
    assert updated["baseline"]["final_equity"] == pytest.approx(1025.0)
    settlement = updated_ticks.iloc[-1]
    assert settlement["resolved_outcome"] == "UP"
    assert settlement["resolution_up"] == pytest.approx(1.0)
    assert settlement["equity"] == pytest.approx(1050.0)
    assert settlement["baseline_equity"] == pytest.approx(1025.0)
