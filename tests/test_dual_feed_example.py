from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from polybench.model import Model
from polybench.replay import ReplayConfig, replay
from polybench.submission_scan import scan_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DUAL_FEED_SUBMISSION = REPO_ROOT / "model_submissions" / "dual_feed_momentum" / "model_submission.py"


def _load_model(config: dict | None = None) -> Model:
    spec = importlib.util.spec_from_file_location("dual_feed_example", DUAL_FEED_SUBMISSION)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ModelSubmission(config=config)
    assert isinstance(model, Model)
    return model


def test_dual_feed_example_passes_submission_scan() -> None:
    report = scan_file(DUAL_FEED_SUBMISSION)
    assert report.verdict == "accept", report.format_text()


def test_dual_feed_example_replays_with_real_tick_shape(
    any_event_fixture: Path, tmp_path: Path
) -> None:
    model = _load_model()
    result = replay(
        model,
        any_event_fixture,
        ReplayConfig(output_dir=tmp_path / "replay", scratch_dir=tmp_path / "scratch"),
    )

    assert math.isfinite(result.metrics["primary_score"])
    assert result.metrics["n_events"] >= 1
    assert result.metrics["n_ticks"] > 0
    assert result.metrics["n_trades"] > 0
    assert result.baseline_metrics["n_ticks"] == result.metrics["n_ticks"]
