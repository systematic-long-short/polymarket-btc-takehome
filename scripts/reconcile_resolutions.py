#!/usr/bin/env python
"""Refresh UNKNOWN event outcomes and update ``report.json`` / ``ticks.parquet``.

Use this after an official run exits with pending resolutions. The command
only uses Gamma for final outcome metadata; executable fills remain from the
recorded Polymarket CLOB tick stream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from polybench.market import EventDescriptor, PolymarketClient
from polybench.reconciliation import reconcile_run_files


RESOLVED_EPSILON = 0.02


async def _fetch_resolutions(slugs: set[str]) -> dict[str, tuple[float, float]]:
    resolutions: dict[str, tuple[float, float]] = {}
    async with PolymarketClient() as client:
        for slug in sorted(slugs):
            event = await client.refresh_event(slug)
            outcome = _resolution_from_event(event)
            if outcome is not None:
                resolutions[slug] = outcome
    return resolutions


def _unknown_slugs(report_path: Path) -> set[str]:
    payload = json.loads(report_path.read_text())
    slugs: set[str] = set()
    for track in ("model", "baseline"):
        for event in payload.get(track, {}).get("events", []):
            if event.get("resolved_outcome") == "UNKNOWN" and event.get("slug"):
                slugs.add(str(event["slug"]))
    return slugs


def _resolution_from_event(event: EventDescriptor | None) -> tuple[float, float] | None:
    if event is None:
        return None
    up, down = event.outcome_prices
    if up > (1.0 - RESOLVED_EPSILON) and down < RESOLVED_EPSILON:
        return (1.0, 0.0)
    if down > (1.0 - RESOLVED_EPSILON) and up < RESOLVED_EPSILON:
        return (0.0, 1.0)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ticks", required=True, type=Path)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Exit 0 even if some UNKNOWN events have not resolved yet.",
    )
    args = parser.parse_args(argv)

    slugs = _unknown_slugs(args.report)
    if not slugs:
        print(json.dumps({"updated_event_count": 0, "remaining_unknown_slugs": []}, indent=2))
        return 0
    resolutions = asyncio.run(_fetch_resolutions(slugs))
    summary: dict[str, Any] = reconcile_run_files(
        report_path=args.report,
        ticks_path=args.ticks,
        resolutions=resolutions,
    )
    remaining = sorted(slugs - set(resolutions))
    summary["remaining_unknown_slugs"] = remaining
    print(json.dumps(summary, indent=2))
    if remaining and not args.allow_pending:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
