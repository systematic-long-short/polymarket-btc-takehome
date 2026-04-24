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


def test_open_for_write_is_rejected() -> None:
    src = "f = open('/tmp/x', 'w')"
    report = scan_source(src)
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_call" for f in report.findings)


def test_import_alias_does_not_bypass_blocked_os_call() -> None:
    src = "import os as o\no.system('echo pwned')"
    report = scan_source(src)
    assert report.verdict == "reject"
    rules = {f.rule for f in report.findings if f.severity == "critical"}
    assert "blocked_import" in rules
    assert "blocked_attr" in rules


def test_from_import_alias_does_not_bypass_blocked_call() -> None:
    src = "from os import system as run\nrun('echo pwned')"
    report = scan_source(src)
    assert report.verdict == "reject"
    rules = {f.rule for f in report.findings if f.severity == "critical"}
    assert "blocked_import" in rules
    assert "blocked_call" in rules


def test_pathlib_file_io_rejected() -> None:
    src = "from pathlib import Path\nPath('/tmp/x').write_text('secret')"
    report = scan_source(src)
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_attr" for f in report.findings)


def test_io_fileio_rejected() -> None:
    report = scan_source("import io\nf = io.FileIO('/tmp/x', 'w')")
    assert report.verdict == "reject"
    assert any(f.rule in {"blocked_attr", "blocked_call"} for f in report.findings)


def test_codecs_open_rejected() -> None:
    report = scan_source("import codecs\nf = codecs.open('/tmp/x', 'w')")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_attr" for f in report.findings)


def test_common_string_replace_is_allowed() -> None:
    report = scan_source("x = 'BTC-USD'.replace('-', '')")
    assert report.verdict == "accept", report.format_text()


def test_environment_access_rejected() -> None:
    report = scan_source("import os\nTOKEN = os.environ.get('TOKEN')")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_attr" for f in report.findings)


def test_dynamic_getattr_rejected() -> None:
    report = scan_source("fn = getattr(object, '__subclasses__')")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_call" for f in report.findings)


def test_blocked_builtin_alias_rejected() -> None:
    report = scan_source("fn = open\nfn('/tmp/x')")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_name" for f in report.findings)


def test_operator_attrgetter_introspection_rejected() -> None:
    report = scan_source("from operator import attrgetter\nfn = attrgetter('__subclasses__')")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_name" for f in report.findings)


def test_operator_methodcaller_introspection_rejected() -> None:
    report = scan_source("import operator\nfn = operator.methodcaller('__subclasses__')")
    assert report.verdict == "reject"
    assert any(f.rule in {'blocked_attr', 'blocked_name'} for f in report.findings)


def test_dunder_introspection_rejected() -> None:
    report = scan_source("x = (1).__class__.__mro__")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_attr" for f in report.findings)


def test_polybench_internal_import_rejected() -> None:
    report = scan_source("import polybench.harness")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_import" for f in report.findings)


def test_polybench_from_import_internal_module_rejected() -> None:
    report = scan_source("from polybench import harness")
    assert report.verdict == "reject"
    assert any(f.rule == "blocked_import" for f in report.findings)


def test_polybench_from_import_public_api_allowed() -> None:
    report = scan_source("from polybench import FLAT, Model, Signal, Side, Tick")
    assert report.verdict == "accept", report.format_text()


def test_polybench_module_mutation_rejected() -> None:
    report = scan_source("import polybench\npolybench.Model = object")
    assert report.verdict == "reject"
    assert any(f.rule == "module_mutation" for f in report.findings)
