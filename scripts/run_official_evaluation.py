#!/usr/bin/env python
"""Host-side official evaluator runner.

This is the supported path for untrusted submissions. It builds (or reuses)
the evaluator image and starts a locked-down container with only the candidate
file mounted read-only and a single writable output directory mounted at
``/output``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_IMAGE = "polybench-eval:latest"


def docker_build_cmd(*, image: str, repo_root: Path) -> list[str]:
    return ["docker", "build", "-t", image, str(repo_root)]


def docker_run_cmd(
    *,
    image: str,
    submission: Path,
    output_dir: Path,
    duration: float,
    starting_capital: float,
    slippage_bps: float,
    fee_rate: float,
    price_source: str,
    postmortem_timeout: float,
    memory: str,
    cpus: str,
    pids_limit: int,
    file_size_mb: int,
    allow_unresolved_final: bool,
) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    inner = [
        "scripts/run_candidate.py",
        "--submission",
        "/submission/model_submission.py",
        "--duration",
        str(duration),
        "--starting-capital",
        str(starting_capital),
        "--slippage-bps",
        str(slippage_bps),
        "--fee-rate",
        str(fee_rate),
        "--price-source",
        price_source,
        "--output",
        "/output",
        "--require-container",
        "--official",
        "--postmortem-timeout",
        str(postmortem_timeout),
        "--file-size-mb",
        str(file_size_mb),
    ]
    if allow_unresolved_final:
        inner.append("--allow-unresolved-final")

    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "bridge",
        "--user",
        f"{uid}:{gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--ulimit",
        "nofile=256:256",
        "--ulimit",
        f"fsize={file_size_mb * 1024 * 1024}:{file_size_mb * 1024 * 1024}",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
        "--tmpfs",
        "/run:rw,noexec,nosuid,nodev,size=16m",
        "-e",
        "PYTHONHASHSEED=0",
        "-e",
        "TZ=UTC",
        "-e",
        "POLYBENCH_OFFICIAL_EVALUATOR=1",
        "-e",
        f"POLYBENCH_CONTAINER_IMAGE={image}",
        "--mount",
        f"type=bind,source={submission},target=/submission/model_submission.py,readonly",
        "--mount",
        f"type=bind,source={output_dir},target=/output",
        "--entrypoint",
        "python",
        image,
        *inner,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--duration", type=float, default=7200.0)
    parser.add_argument("--starting-capital", type=float, default=1000.0)
    parser.add_argument("--slippage-bps", type=float, default=50.0)
    parser.add_argument("--fee-rate", type=float, default=0.072)
    parser.add_argument("--price-source", choices=["polymarket", "binance"], default="polymarket")
    parser.add_argument("--postmortem-timeout", type=float, default=600.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--pids-limit", type=int, default=256)
    parser.add_argument("--file-size-mb", type=int, default=1024)
    parser.add_argument("--allow-unresolved-final", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    submission = args.submission.resolve()
    output_dir = args.output.resolve()
    if not submission.is_file():
        print(f"ERROR: submission file not found: {submission}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    if not args.skip_build:
        commands.append(docker_build_cmd(image=args.image, repo_root=repo_root))
    commands.append(
        docker_run_cmd(
            image=args.image,
            submission=submission,
            output_dir=output_dir,
            duration=args.duration,
            starting_capital=args.starting_capital,
            slippage_bps=args.slippage_bps,
            fee_rate=args.fee_rate,
            price_source=args.price_source,
            postmortem_timeout=args.postmortem_timeout,
            memory=args.memory,
            cpus=args.cpus,
            pids_limit=args.pids_limit,
            file_size_mb=args.file_size_mb,
            allow_unresolved_final=args.allow_unresolved_final,
        )
    )
    if args.print_only:
        for command in commands:
            print(" ".join(shlex.quote(part) for part in command))
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
