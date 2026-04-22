"""Submission-scanner acceptance and rejection tests."""

from __future__ import annotations

from pathlib import Path

from polybench.submission_scan import scan_file, scan_source

FIXTURES = Path(__file__).parent / "fixtures"


def test_safe_submission_accepted() -> None:
    report = scan_file(FIXTURES / "safe_submission.py")
    assert report.verdict == "accept", report.format_text()
    assert all(f.severity != "critical" for f in report.findings)


def test_unsafe_submission_rejected_with_findings() -> None:
    report = scan_file(FIXTURES / "unsafe_submission.py")
    assert report.verdict == "reject"
    rule_set = {f.rule for f in report.findings if f.severity == "critical"}
    assert "blocked_import" in rule_set         # subprocess
    assert "unknown_import" in rule_set         # mysterious_unknown_lib
    assert "blocked_attr" in rule_set           # os.system
    assert "blocked_call" in rule_set           # exec


def test_example_candidate_template_is_accepted() -> None:
    repo_root = Path(__file__).parent.parent
    report = scan_file(repo_root / "examples" / "model_submission.py")
    assert report.verdict == "accept", report.format_text()


def test_baselines_would_pass_scanner() -> None:
    repo_root = Path(__file__).parent.parent
    report = scan_file(repo_root / "models" / "baseline_models.py")
    assert report.verdict == "accept", report.format_text()


def test_eval_call_rejected() -> None:
    src = "x = eval('1+1')"
    report = scan_source(src)
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_call" for f in report.findings)


def test_pickle_import_rejected() -> None:
    report = scan_source("import pickle")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_import" for f in report.findings)


def test_ctypes_rejected() -> None:
    report = scan_source("from ctypes import CDLL")
    assert report.verdict == "reject"


def test_syntax_error_rejected() -> None:
    report = scan_source("def oops(:")
    assert report.verdict == "reject"
    assert any(f.rule == "syntax_error" for f in report.findings)


def test_open_for_write_is_warning_not_reject() -> None:
    src = "f = open('/tmp/x', 'w')"
    report = scan_source(src)
    # open for write is a warning, so verdict remains accept.
    assert report.verdict == "accept"
    assert any(f.rule == "open_for_write" for f in report.findings)
