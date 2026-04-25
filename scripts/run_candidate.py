#!/usr/bin/env python
"""Run a candidate submission against live Polymarket.

Runs the security scanner FIRST, refuses to load the submission if any
critical finding is raised (override with --allow-unsafe, intended only
for a local developer sanity-check).

Example:
    python scripts/run_candidate.py \\
        --submission examples/model_submission.py \\
        --duration 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import importlib.util
import os
import sys
import time
from pathlib import Path

from polybench.cli import _load_config
from polybench.harness import Harness, HarnessConfig, format_summary
from polybench.model import Model
from polybench.submission_scan import scan_file


SAFE_ENV_KEYS = frozenset({
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "POLYBENCH_CONTAINER_DIGEST",
    "POLYBENCH_CONTAINER_IMAGE",
    "POLYBENCH_OFFICIAL_EVALUATOR",
    "PYTHONHASHSEED",
    "TZ",
})

SENSITIVE_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "OPENAI_",
    "ANTHROPIC_",
    "GITHUB_",
    "DOCKER_",
)
SENSITIVE_ENV_KEYS = {
    "API_KEY",
    "AUTH_TOKEN",
    "DOCKER_HOST",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "KUBECONFIG",
    "SECRET",
    "TOKEN",
}


def _running_in_container() -> bool:
    """Best-effort check used by evaluator automation before importing code."""
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    markers = ("docker", "containerd", "kubepods", "podman")
    return any(marker in cgroup for marker in markers)


def _scrub_environment() -> None:
    """Remove evaluator secrets before importing untrusted candidate code."""
    preserved = {k: v for k, v in os.environ.items() if k in SAFE_ENV_KEYS}
    os.environ.clear()
    os.environ.update(preserved)


def _apply_resource_limits(*, memory_mb: int, file_size_mb: int) -> None:
    """Best-effort Unix process limits for the untrusted model process."""
    try:
        import resource
    except ImportError:
        return

    def _set_limit(resource_name: int, soft_limit: int) -> None:
        old_soft, old_hard = resource.getrlimit(resource_name)
        hard_limit = soft_limit if old_hard == resource.RLIM_INFINITY else min(old_hard, soft_limit)
        requested_soft = min(old_soft, soft_limit) if old_soft != resource.RLIM_INFINITY else soft_limit
        new_soft = min(requested_soft, hard_limit)
        resource.setrlimit(resource_name, (new_soft, hard_limit))

    try:
        _set_limit(resource.RLIMIT_AS, int(memory_mb) * 1024 * 1024)
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not apply memory limit: {exc}", file=sys.stderr)
    try:
        _set_limit(resource.RLIMIT_FSIZE, int(file_size_mb) * 1024 * 1024)
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not apply file-size limit: {exc}", file=sys.stderr)
    try:
        _set_limit(resource.RLIMIT_NOFILE, 256)
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not apply fd limit: {exc}", file=sys.stderr)


def _path_is_read_only(path: Path) -> bool:
    probe = path / f".polybench-write-probe-{os.getpid()}"
    try:
        probe.write_text("probe")
    except OSError:
        return True
    try:
        probe.unlink()
    except OSError:
        pass
    return False


def _official_isolation_failures(*, submission: Path, output_dir: Path) -> list[str]:
    failures: list[str] = []
    if not _running_in_container():
        failures.append("not running inside a container")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        failures.append("container user is root")
    if Path("/var/run/docker.sock").exists():
        failures.append("Docker socket is mounted inside the evaluator")

    out = output_dir.resolve()
    if not (str(out) == "/output" or str(out).startswith("/output/")):
        failures.append("official output directory must be under /output")
    else:
        try:
            out.mkdir(parents=True, exist_ok=True)
            probe = out / f".polybench-output-probe-{os.getpid()}"
            probe.write_text("probe")
            probe.unlink()
        except OSError as exc:
            failures.append(f"official output directory is not writable: {exc}")

    if submission.exists() and not _path_is_read_only(submission.parent):
        failures.append("submission mount is writable")

    workdir = Path("/polybench")
    if workdir.exists() and not _path_is_read_only(workdir):
        failures.append("evaluator filesystem is writable; expected --read-only")

    exposed = []
    for key in os.environ:
        if key in SAFE_ENV_KEYS:
            continue
        if key in SENSITIVE_ENV_KEYS or any(key.startswith(prefix) for prefix in SENSITIVE_ENV_PREFIXES):
            exposed.append(key)
    if exposed:
        failures.append("sensitive environment variables are present: " + ", ".join(sorted(exposed)))
    if os.environ.get("POLYBENCH_OFFICIAL_EVALUATOR") != "1":
        failures.append("POLYBENCH_OFFICIAL_EVALUATOR=1 was not set by the official runner")
    return failures


def _unknown_event_count(report_path: Path) -> int:
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    status = payload.get("scoring_status", {})
    try:
        return int(status.get("unresolved_event_count", 0))
    except (TypeError, ValueError):
        return 0


def _load_module(path: Path, class_name: str, config):
    module_name = f"_polybench_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    klass = getattr(module, class_name, None)
    if klass is None:
        raise SystemExit(f"class {class_name!r} not found in {path}")
    model = klass(config=config)
    if not isinstance(model, Model):
        raise SystemExit(f"{class_name!r} must subclass polybench.Model")
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, help="path to model_submission.py")
    p.add_argument("--class", dest="class_name", default="ModelSubmission")
    p.add_argument("--duration", type=float, default=7200.0)
    p.add_argument("--starting-capital", type=float, default=1000.0)
    p.add_argument("--slippage-bps", type=float, default=50.0)
    p.add_argument("--fee-rate", type=float, default=0.072,
                   help="Polymarket-style fee coefficient")
    p.add_argument("--price-source", default="polymarket",
                   choices=["polymarket", "binance"],
                   help="Default 'polymarket' disables external BTC WebSockets.")
    p.add_argument("--output", dest="output_dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--resolution-timeout",
        type=float,
        default=45.0,
        help="Deprecated; event resolution now continues asynchronously after rollover.",
    )
    p.add_argument(
        "--postmortem-timeout",
        type=float,
        default=0.0,
        help="Extra seconds after the run to wait for still-pending resolutions (default 0).",
    )
    p.add_argument("--memory-mb", type=int, default=4096,
                   help="Best-effort process virtual memory cap for the candidate run.")
    p.add_argument("--file-size-mb", type=int, default=1024,
                   help="Best-effort max file size cap for the candidate run.")
    p.add_argument("--no-limits", action="store_true",
                   help="Do not apply process resource limits (debug-only).")
    p.add_argument("--keep-env", action="store_true",
                   help="Do not scrub environment variables before loading candidate code (debug-only).")
    p.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Load the submission even if the scanner rejects it (debug-only).",
    )
    p.add_argument(
        "--require-container",
        action="store_true",
        help=(
            "Fail unless the runner appears to be inside Docker/Podman/Kubernetes. "
            "Use this for evaluator automation with untrusted submissions."
        ),
    )
    p.add_argument(
        "--official",
        action="store_true",
        help="Fail closed unless official container isolation checks pass.",
    )
    p.add_argument(
        "--allow-unresolved-final",
        action="store_true",
        help="Official-only override: accept a final report even if events are still UNKNOWN.",
    )
    args = p.parse_args(argv)

    path = Path(args.submission).resolve()
    if not path.is_file():
        print(f"ERROR: submission file not found: {path}", file=sys.stderr)
        return 2
    out_dir = Path(
        args.output_dir or f"runs/candidate_{path.stem}_{int(time.time())}"
    )

    if args.require_container and not _running_in_container():
        print(
            "ERROR: refusing to import an untrusted submission outside a container. "
            "Run the evaluator command from EVALUATION.md, or omit "
            "--require-container only for trusted local debugging.",
            file=sys.stderr,
        )
        return 2
    if args.official:
        if args.allow_unsafe or args.keep_env or args.no_limits:
            print(
                "ERROR: --official cannot be combined with debug-only bypass flags.",
                file=sys.stderr,
            )
            return 2
        failures = _official_isolation_failures(submission=path, output_dir=out_dir)
        if failures:
            print("ERROR: official evaluator isolation checks failed:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 2

    print(f"scanning {path}...")
    report = scan_file(path)
    print(report.format_text())
    if report.verdict != "accept":
        if not args.allow_unsafe:
            print("\nREJECTED: submission failed security scan.", file=sys.stderr)
            return 2
        print("\nWARNING: --allow-unsafe set; loading despite scanner rejection.",
              file=sys.stderr)

    if not args.keep_env:
        _scrub_environment()
    if not args.no_limits:
        _apply_resource_limits(
            memory_mb=args.memory_mb,
            file_size_mb=args.file_size_mb,
        )

    model = _load_module(path, args.class_name, _load_config(args.config))

    cfg = HarnessConfig(
        duration_s=args.duration,
        starting_capital=args.starting_capital,
        slippage_bps=args.slippage_bps,
        fee_rate=args.fee_rate,
        price_source=args.price_source,
        output_dir=out_dir,
        resolution_poll_timeout_s=args.resolution_timeout,
        postmortem_resolution_s=args.postmortem_timeout,
        candidate_path=path,
        command=tuple(sys.argv if argv is None else [sys.argv[0], *argv]),
        official_scoring=args.official,
        allow_unresolved_final=args.allow_unresolved_final,
    )
    result = asyncio.run(Harness(model=model, config=cfg).run())
    print(format_summary(result))
    report_path = out_dir / "report.json"
    print(f"report: {report_path}")
    print(f"ticks:  {out_dir / 'ticks.parquet'}")
    unknown_events = _unknown_event_count(report_path)
    if args.official and unknown_events and not args.allow_unresolved_final:
        print(
            "PENDING RESOLUTION: official score is not final because "
            f"{unknown_events} completed event(s) are still UNKNOWN. Run "
            "scripts/reconcile_resolutions.py after Gamma publishes final outcomes.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
