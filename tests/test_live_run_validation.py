from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_live_run import validate_live_run


def _write_report(path: Path, *, duration: float = 600.0, score: float = 194.0) -> None:
    events = [
        {
            "event_id": "E1",
            "slug": "btc-updown-5m-test-1",
            "start_ts": 0.0,
            "end_ts": 300.0,
            "resolved_outcome": "UP",
            "pnl_total": 10.0,
            "pnl_intra_event": 10.0,
            "pnl_resolution": 0.0,
            "n_trades": 2,
            "n_ticks": 300,
            "n_timeouts": 0,
        },
        {
            "event_id": "E2",
            "slug": "btc-updown-5m-test-2",
            "start_ts": 302.0,
            "end_ts": 600.0,
            "resolved_outcome": "DOWN",
            "pnl_total": 10.0,
            "pnl_intra_event": 10.0,
            "pnl_resolution": 0.0,
            "n_trades": 2,
            "n_ticks": 300,
            "n_timeouts": 0,
        },
    ]
    metrics = {
        "starting_capital": 1000.0,
        "final_equity": 1020.0,
        "pnl_total": 20.0,
        "pnl_pct": 0.02,
        "sharpe": 10.0,
        "sortino": 12.0,
        "max_drawdown": 0.03,
        "hit_rate": 1.0,
        "outcome_accuracy": 1.0,
        "timeout_rate": 0.0,
        "n_events": 2,
        "n_trades": 4,
        "n_ticks": 600,
        "pnl_intra_event": 20.0,
        "pnl_resolution": 0.0,
        "intra_vs_resolution_fraction": 1.0,
        "primary_score": score,
    }
    payload = {
        "started_ts": 1_700_000_000.0,
        "ended_ts": 1_700_000_000.0 + duration,
        "starting_capital": 1000.0,
        "model": {
            "final_equity": 1020.0,
            "pnl_total": 20.0,
            "pnl_pct": 0.02,
            "metrics": metrics,
            "events": events,
        },
        "baseline": {
            "final_equity": 1020.0,
            "pnl_total": 20.0,
            "pnl_pct": 0.02,
            "metrics": dict(metrics),
            "events": list(events),
        },
    }
    path.write_text(json.dumps(payload))


def test_validate_live_run_accepts_dual_feed_tick_data(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report)

    df = pd.read_parquet(synthetic_fixture)
    df = df.iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["resolved_outcome"] = ""
    df.loc[:49, "up_ask"] = 0.0
    df.loc[:49, "down_bid"] = 0.0
    df["btc_source"] = "binance"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    summary = validate_live_run(
        report_path=report,
        ticks_path=ticks,
        min_duration_s=600.0,
        min_events=2,
        require_binance=True,
    )
    assert summary["btc_sources"] == ["binance"]
    assert summary["polymarket_valid_rows"] == 601
    assert summary["polymarket_two_sided_rows"] == 551
    assert summary["polymarket_one_sided_rows"] == 50
    assert summary["model"]["primary_score"] == 194.0


def test_validate_live_run_accepts_polymarket_only_tick_data(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report)

    df = pd.read_parquet(synthetic_fixture)
    df = df.iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["resolved_outcome"] = ""
    df["btc_last"] = 0.0
    df["btc_bid"] = 0.0
    df["btc_ask"] = 0.0
    df["btc_source"] = "polymarket"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    summary = validate_live_run(
        report_path=report,
        ticks_path=ticks,
        min_duration_s=600.0,
        min_events=2,
        require_binance=False,
    )

    assert summary["btc_complete_rows"] == 0
    assert summary["btc_sources"] == ["polymarket"]
    assert summary["polymarket_valid_rows"] == 601


def test_validate_live_run_rejects_wrong_score(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report, score=193.0)

    df = pd.read_parquet(synthetic_fixture).iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["btc_source"] = "binance"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    with pytest.raises(AssertionError, match="primary score mismatch"):
        validate_live_run(
            report_path=report,
            ticks_path=ticks,
            min_duration_s=600.0,
            min_events=2,
            require_binance=True,
        )


def test_validate_live_run_requires_binance_source(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report)

    df = pd.read_parquet(synthetic_fixture).iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["btc_source"] = "other"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    with pytest.raises(AssertionError, match="Binance"):
        validate_live_run(
            report_path=report,
            ticks_path=ticks,
            min_duration_s=600.0,
            min_events=2,
            require_binance=True,
        )


def test_validate_live_run_rejects_missing_polymarket_mids(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report)

    df = pd.read_parquet(synthetic_fixture).iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["resolved_outcome"] = ""
    df.loc[:250, "up_mid"] = 0.0
    df["btc_source"] = "binance"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    with pytest.raises(AssertionError, match="valid Polymarket mids"):
        validate_live_run(
            report_path=report,
            ticks_path=ticks,
            min_duration_s=600.0,
            min_events=2,
            require_binance=True,
        )


def test_validate_live_run_rejects_pending_scoring_status(
    synthetic_fixture: Path, tmp_path: Path
) -> None:
    report = tmp_path / "report.json"
    ticks = tmp_path / "ticks.parquet"
    _write_report(report)
    payload = json.loads(report.read_text())
    payload["scoring_status"] = {
        "state": "pending_resolution",
        "unresolved_event_count": 1,
        "unresolved_events": [{"track": "model", "slug": "s", "event_id": "e"}],
    }
    report.write_text(json.dumps(payload))

    df = pd.read_parquet(synthetic_fixture).iloc[:601].copy()
    df["ts"] = [1_700_000_000.0 + i for i in range(len(df))]
    df["btc_source"] = "binance"
    df.to_parquet(ticks, engine="pyarrow", index=False)

    with pytest.raises(AssertionError, match="pending_resolution"):
        validate_live_run(
            report_path=report,
            ticks_path=ticks,
            min_duration_s=600.0,
            min_events=2,
            require_binance=True,
        )
