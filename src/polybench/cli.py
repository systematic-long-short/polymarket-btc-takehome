"""``polybench`` CLI.

Subcommands:
  polybench run       — run a live paper-trading session against Polymarket
  polybench replay    — replay a recorded parquet log against a model
  polybench scan      — security-scan a candidate submission

Model selection for run/replay:
  --model MODULE:CLASS           e.g. polybench.baselines:MomentumBaseline
  --model-file PATH --class NAME (for file-based candidate submissions)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from polybench.harness import HarnessConfig, Harness, format_summary
from polybench.model import Model


def _load_model_from_spec(spec: str, config: dict[str, Any] | None) -> Model:
    """Load ``module.path:ClassName`` and instantiate with config."""
    if ":" not in spec:
        raise SystemExit(f"--model must be MODULE:CLASS form, got {spec!r}")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name, None)
    if klass is None:
        raise SystemExit(f"class {class_name!r} not found in {module_name!r}")
    model = klass(config=config)
    if not isinstance(model, Model):
        raise SystemExit(f"{class_name!r} must subclass polybench.Model")
    return model


def _load_model_from_file(
    file_path: str, class_name: str, config: dict[str, Any] | None
) -> Model:
    path = Path(file_path).resolve()
    if not path.is_file():
        raise SystemExit(f"--model-file not found: {path}")
    module_name = f"_polybench_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path} as a module")
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


def _load_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    text = Path(path).read_text()
    return json.loads(text)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--model",
        help="MODULE:CLASS (e.g. polybench.baselines:MomentumBaseline)",
    )
    grp.add_argument(
        "--model-file",
        help="Path to a candidate's model_submission.py",
    )
    p.add_argument(
        "--class",
        dest="class_name",
        default="ModelSubmission",
        help="Class name inside --model-file (default: ModelSubmission)",
    )
    p.add_argument("--config", help="Path to a JSON config file passed to the Model")


def _load_model(args: argparse.Namespace) -> Model:
    cfg = _load_config(args.config)
    if getattr(args, "model_file", None):
        return _load_model_from_file(args.model_file, args.class_name, cfg)
    return _load_model_from_spec(args.model, cfg)


# ---- subcommand handlers ----

def _cmd_run(args: argparse.Namespace) -> int:
    model = _load_model(args)
    cfg = HarnessConfig(
        duration_s=args.duration,
        tick_interval_s=args.tick_interval,
        model_budget_s=args.model_budget,
        resolution_poll_timeout_s=args.resolution_timeout,
        postmortem_resolution_s=args.postmortem_timeout,
        starting_capital=args.starting_capital,
        slippage_bps=args.slippage_bps,
        fee_rate=args.fee_rate,
        price_source=args.price_source,
        output_dir=Path(args.output_dir),
    )
    harness = Harness(model=model, config=cfg)
    result = asyncio.run(harness.run())
    print(format_summary(result))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from polybench.replay import ReplayConfig, replay

    model = _load_model(args)
    cfg = ReplayConfig(
        starting_capital=args.starting_capital,
        slippage_bps=args.slippage_bps,
        fee_rate=args.fee_rate,
        output_dir=Path(args.output_dir),
    )
    result = replay(model, Path(args.data), config=cfg)
    print(format_summary(result))
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    from polybench.submission_scan import scan_file

    report = scan_file(Path(args.file))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_text())
    return 0 if report.verdict == "accept" else 2


# ---- main ----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polybench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    r = sub.add_parser("run", help="Run a live session against Polymarket")
    _add_model_args(r)
    r.add_argument("--duration", type=float, default=3600.0, help="Run duration in seconds")
    r.add_argument("--tick-interval", type=float, default=1.0)
    r.add_argument("--model-budget", type=float, default=0.5, help="on_tick wall-clock budget (s)")
    r.add_argument(
        "--resolution-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for a just-ended event to resolve before marking UNKNOWN.",
    )
    r.add_argument(
        "--postmortem-timeout",
        type=float,
        default=0.0,
        help="Extra seconds after the run to refresh UNKNOWN resolutions (default 0).",
    )
    r.add_argument("--starting-capital", type=float, default=1000.0)
    r.add_argument("--slippage-bps", type=float, default=50.0, help="Default 50 = 0.5%% per order")
    r.add_argument("--fee-rate", type=float, default=0.072,
                   help="Polymarket-style fee coefficient (0.072 = ~1.8%% at p=0.5)")
    r.add_argument(
        "--price-source",
        choices=["polymarket", "binance", "binance-us", "coinbase"],
        default="polymarket",
        help="Default 'polymarket' disables external BTC WebSockets.",
    )
    r.add_argument("--output-dir", default="runs/latest")
    r.set_defaults(func=_cmd_run)

    # replay
    p_r = sub.add_parser("replay", help="Replay a recorded parquet against a model")
    _add_model_args(p_r)
    p_r.add_argument("--data", required=True, help="Path to recorded ticks parquet")
    p_r.add_argument("--starting-capital", type=float, default=1000.0)
    p_r.add_argument("--slippage-bps", type=float, default=50.0)
    p_r.add_argument("--fee-rate", type=float, default=0.072,
                     help="Polymarket-style fee coefficient")
    p_r.add_argument("--output-dir", default="runs/replay")
    p_r.set_defaults(func=_cmd_replay)

    # scan
    p_s = sub.add_parser("scan", help="Security-scan a candidate submission")
    p_s.add_argument("--file", required=True, help="Path to model_submission.py")
    p_s.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p_s.set_defaults(func=_cmd_scan)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
