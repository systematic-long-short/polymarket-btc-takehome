"""Late-resolution reconciliation for completed live runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from polybench.metrics import summarize
from polybench.model import EventResult
from polybench.reporting import TRACKS, scoring_status


ResolutionMap = Mapping[str, tuple[float, float]]


def reconcile_run_files(
    *,
    report_path: Path,
    ticks_path: Path,
    resolutions: ResolutionMap,
) -> dict[str, Any]:
    payload = json.loads(report_path.read_text())
    ticks = pd.read_parquet(ticks_path, engine="pyarrow")
    summary = reconcile_payload(payload, ticks, resolutions)
    report_path.write_text(json.dumps(payload, indent=2, default=str))
    ticks.to_parquet(ticks_path, engine="pyarrow", index=False)
    return summary


def reconcile_payload(
    payload: dict[str, Any],
    ticks: pd.DataFrame,
    resolutions: ResolutionMap,
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    starting = float(payload.get("starting_capital", 0.0))
    for track in TRACKS:
        section = payload.get(track, {})
        events = section.get("events", [])
        cash_delta_total = 0.0
        for event in events:
            if event.get("resolved_outcome") != "UNKNOWN":
                continue
            slug = str(event.get("slug", ""))
            outcome = resolutions.get(slug)
            if outcome is None:
                continue
            cash_delta = _resolve_event_dict(event, outcome)
            cash_delta_total += cash_delta
            _update_ticks_for_resolution(
                ticks=ticks,
                slug=slug,
                outcome=outcome,
                track=track,
                cash_delta=cash_delta,
            )
            updates.append({
                "track": track,
                "slug": slug,
                "cash_delta": cash_delta,
                "outcome": event["resolved_outcome"],
            })

        if cash_delta_total:
            final_equity = float(section.get("final_equity", starting)) + cash_delta_total
            section["final_equity"] = final_equity
            section["pnl_total"] = final_equity - starting
            section["pnl_pct"] = (final_equity - starting) / starting if starting else 0.0
        section["metrics"] = _recompute_metrics(
            ticks=ticks,
            events=events,
            track=track,
            starting_capital=starting,
            final_equity=float(section.get("final_equity", starting)),
        )

    payload["scoring_status"] = scoring_status(payload)
    metadata = payload.setdefault("metadata", {})
    metadata["reconciliation"] = {
        "updated_event_count": len(updates),
        "updates": updates,
    }
    feed_health = metadata.setdefault("feed_health", {})
    feed_health["unknown_event_count"] = payload["scoring_status"]["unresolved_event_count"]
    return metadata["reconciliation"]


def _resolve_event_dict(event: dict[str, Any], outcome: tuple[float, float]) -> float:
    up_outcome, down_outcome = outcome
    up_shares = float(event.get("pending_resolution_up_shares") or 0.0)
    down_shares = float(event.get("pending_resolution_down_shares") or 0.0)
    up_mark = float(event.get("pending_resolution_up_mark") or 0.0)
    down_mark = float(event.get("pending_resolution_down_mark") or 0.0)
    provisional_payout = up_shares * up_mark + down_shares * down_mark
    final_payout = up_shares * up_outcome + down_shares * down_outcome
    resolution_pnl = final_payout - provisional_payout
    event["resolved_outcome"] = _outcome_label(outcome)
    event["pnl_resolution"] = resolution_pnl
    event["pnl_total"] = float(event.get("pnl_intra_event") or 0.0) + resolution_pnl
    event["pending_resolution_up_shares"] = 0.0
    event["pending_resolution_down_shares"] = 0.0
    event["pending_resolution_up_mark"] = 0.0
    event["pending_resolution_down_mark"] = 0.0
    return final_payout - provisional_payout


def _update_ticks_for_resolution(
    *,
    ticks: pd.DataFrame,
    slug: str,
    outcome: tuple[float, float],
    track: str,
    cash_delta: float,
) -> None:
    if ticks.empty or "slug" not in ticks.columns:
        return
    slug_mask = ticks["slug"].fillna("").astype(str).eq(slug)
    if not slug_mask.any():
        return
    settlement_mask = slug_mask
    if "resolved_outcome" in ticks.columns:
        settlement_mask = slug_mask & ticks["resolved_outcome"].fillna("").astype(str).ne("")
    if not settlement_mask.any():
        settlement_mask = slug_mask
    settlement_index = ticks.index[settlement_mask][-1]
    settlement_ts = float(ticks.loc[settlement_index, "ts"])
    ticks.loc[settlement_index, "resolution_up"] = outcome[0]
    ticks.loc[settlement_index, "resolution_down"] = outcome[1]
    ticks.loc[settlement_index, "resolved_outcome"] = _outcome_label(outcome)

    cash_col = "cash" if track == "model" else "baseline_cash"
    equity_col = "equity" if track == "model" else "baseline_equity"
    later_mask = pd.to_numeric(ticks["ts"], errors="coerce") >= settlement_ts
    for column in (cash_col, equity_col):
        if column in ticks.columns and cash_delta:
            ticks.loc[later_mask, column] = (
                pd.to_numeric(ticks.loc[later_mask, column], errors="coerce")
                .fillna(0.0)
                + cash_delta
            )


def _recompute_metrics(
    *,
    ticks: pd.DataFrame,
    events: list[dict[str, Any]],
    track: str,
    starting_capital: float,
    final_equity: float,
) -> dict[str, Any]:
    equity_col = "equity" if track == "model" else "baseline_equity"
    equity_curve: list[float] = []
    if equity_col in ticks.columns:
        if "resolved_outcome" in ticks.columns:
            active = ticks["resolved_outcome"].fillna("").astype(str).eq("")
            series = ticks.loc[active, equity_col]
        else:
            series = ticks[equity_col]
        equity_curve = [
            float(value)
            for value in pd.to_numeric(series, errors="coerce").dropna().tolist()
            if math.isfinite(float(value))
        ]
    event_results = [_event_result_from_dict(event) for event in events]
    return summarize(
        starting_capital=starting_capital,
        final_equity=final_equity,
        equity_curve=[starting_capital, *equity_curve],
        events=event_results,
    )


def _event_result_from_dict(event: dict[str, Any]) -> EventResult:
    return EventResult(
        event_id=str(event.get("event_id", "")),
        slug=str(event.get("slug", "")),
        start_ts=float(event.get("start_ts") or 0.0),
        end_ts=float(event.get("end_ts") or 0.0),
        resolved_outcome=str(event.get("resolved_outcome", "UNKNOWN")),
        pnl_total=float(event.get("pnl_total") or 0.0),
        pnl_intra_event=float(event.get("pnl_intra_event") or 0.0),
        pnl_resolution=float(event.get("pnl_resolution") or 0.0),
        n_trades=int(event.get("n_trades") or 0),
        n_ticks=int(event.get("n_ticks") or 0),
        n_timeouts=int(event.get("n_timeouts") or 0),
        pending_resolution_up_shares=float(event.get("pending_resolution_up_shares") or 0.0),
        pending_resolution_down_shares=float(event.get("pending_resolution_down_shares") or 0.0),
        pending_resolution_up_mark=float(event.get("pending_resolution_up_mark") or 0.0),
        pending_resolution_down_mark=float(event.get("pending_resolution_down_mark") or 0.0),
    )


def _outcome_label(outcome: tuple[float, float]) -> str:
    up, down = outcome
    if up >= 0.99 and down <= 0.01:
        return "UP"
    if down >= 0.99 and up <= 0.01:
        return "DOWN"
    return "UNKNOWN"
