from __future__ import annotations

import hashlib
from pathlib import Path

from polybench.harness import HarnessConfig
from polybench.reporting import build_reproducibility_metadata


def test_reproducibility_metadata_includes_candidate_sha_and_config(tmp_path: Path) -> None:
    candidate = tmp_path / "model_submission.py"
    candidate.write_text("class ModelSubmission: pass\n")
    cfg = HarnessConfig(
        duration_s=123.0,
        starting_capital=1500.0,
        slippage_bps=25.0,
        fee_rate=0.071,
        price_source="polymarket",
        output_dir=tmp_path / "out",
        candidate_path=candidate,
        command=("python", "scripts/run_candidate.py"),
    )

    metadata = build_reproducibility_metadata(
        candidate_path=candidate,
        command=cfg.command,
        config=cfg,
        feed_health={"active_tick_rows": 10, "price_source": "polymarket"},
    )

    assert metadata["candidate"]["path"] == str(candidate.resolve())
    assert metadata["candidate"]["sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert metadata["git"]["commit_sha"]
    assert metadata["runtime"]["packages"]["python"]
    assert metadata["command"] == ["python", "scripts/run_candidate.py"]
    assert metadata["execution"]["duration_s"] == 123.0
    assert metadata["execution"]["price_source"] == "polymarket"
    assert metadata["feed_health"]["active_tick_rows"] == 10
