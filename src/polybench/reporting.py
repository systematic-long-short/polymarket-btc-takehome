"""Report metadata helpers.

The live harness and replay path both write ``report.json``. Keep the
reproducibility block in one place so evaluator-only metadata does not leak
into the candidate API.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

from polybench.model import EventResult


TRACKS = ("model", "baseline")
KEY_PACKAGES = (
    "polybench",
    "numpy",
    "pandas",
    "pyarrow",
    "httpx",
    "websockets",
)


def sha256_file(path: Path | str | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(repo_root: Path | str | None = None) -> str | None:
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in KEY_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def scoring_status(
    report_payload: dict[str, Any],
    *,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    unresolved: list[dict[str, str]] = []
    unique_events: set[tuple[str, str]] = set()
    for track in TRACKS:
        events = report_payload.get(track, {}).get("events", [])
        for event in events:
            if event.get("resolved_outcome") == "UNKNOWN":
                slug = str(event.get("slug", ""))
                event_id = str(event.get("event_id", ""))
                unique_events.add((slug, event_id))
                unresolved.append({
                    "track": track,
                    "slug": slug,
                    "event_id": event_id,
                })
    if not unresolved:
        state = "final"
    elif allow_unresolved:
        state = "final_unresolved_allowed"
    else:
        state = "pending_resolution"
    return {
        "state": state,
        "unresolved_event_count": len(unique_events),
        "unresolved_track_count": len(unresolved),
        "unresolved_events": unresolved,
    }


def build_reproducibility_metadata(
    *,
    candidate_path: Path | str | None,
    command: Sequence[str] | None,
    config: Any,
    feed_health: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate = Path(candidate_path).resolve() if candidate_path else None
    config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else {}
    if "output_dir" in config_dict:
        config_dict["output_dir"] = str(config_dict["output_dir"])
    if "candidate_path" in config_dict and config_dict["candidate_path"] is not None:
        config_dict["candidate_path"] = str(config_dict["candidate_path"])
    return {
        "candidate": {
            "path": str(candidate) if candidate is not None else None,
            "sha256": sha256_file(candidate),
        },
        "git": {
            "commit_sha": git_commit_sha(Path(__file__).resolve().parents[2]),
        },
        "runtime": {
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "command": list(command or sys.argv),
        "config": config_dict,
        "container": {
            "image": os.environ.get("POLYBENCH_CONTAINER_IMAGE") or None,
            "digest": os.environ.get("POLYBENCH_CONTAINER_DIGEST") or None,
            "id": _container_id(),
        },
        "execution": {
            "price_source": str(config_dict.get("price_source", "")),
            "slippage_bps": config_dict.get("slippage_bps"),
            "fee_rate": config_dict.get("fee_rate"),
            "starting_capital": config_dict.get("starting_capital"),
            "duration_s": config_dict.get("duration_s"),
        },
        "feed_health": feed_health or {},
    }


def event_dicts(events: Sequence[EventResult]) -> list[dict[str, Any]]:
    return [
        dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(e)
        for e in events
    ]


def _container_id() -> str | None:
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        value = line.rsplit("/", 1)[-1]
        if len(value) >= 12 and all(c in "0123456789abcdef" for c in value[:12].lower()):
            return value
    return None
