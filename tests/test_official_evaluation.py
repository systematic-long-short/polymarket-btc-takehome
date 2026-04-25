from __future__ import annotations

from pathlib import Path

from scripts.run_official_evaluation import docker_run_cmd


def test_official_docker_command_is_locked_down(tmp_path: Path) -> None:
    submission = tmp_path / "model_submission.py"
    submission.write_text("x = 1\n")
    output = tmp_path / "out"
    output.mkdir()

    command = docker_run_cmd(
        image="polybench-eval:test",
        submission=submission,
        output_dir=output,
        duration=7200.0,
        starting_capital=1000.0,
        slippage_bps=50.0,
        fee_rate=0.072,
        price_source="polymarket",
        postmortem_timeout=600.0,
        memory="4g",
        cpus="2",
        pids_limit=256,
        file_size_mb=1024,
        allow_unresolved_final=False,
    )

    joined = " ".join(command)
    assert "--user" in command
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "no-new-privileges" in joined
    assert "--pids-limit" in command
    assert "--memory" in command
    assert "--cpus" in command
    assert "type=bind" in joined
    assert "target=/submission/model_submission.py,readonly" in joined
    assert "target=/output" in joined
    assert "--official" in command
    assert "--require-container" in command
    assert "/var/run/docker.sock" not in joined
