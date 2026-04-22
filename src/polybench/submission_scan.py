"""AST-based security screener for candidate submissions.

This is a static, best-effort check — NOT a sandbox. It catches accidental
filesystem damage, obviously-malicious code, and disallowed imports before
the harness loads the submission module. A motivated bad actor could defeat
it; for that reason the scoring run also applies OS-level resource limits
and runs the submission inside a dedicated working directory.

Policy:
  - Blocked imports (hard reject)                — see ``BLOCKED_MODULES``
  - Only allowlisted imports permitted          — see ``ALLOWED_MODULES``
  - Blocked call targets (hard reject)          — eval/exec/compile/__import__/breakpoint
  - Blocked attribute paths (hard reject)       — os.system, os.popen, os.execve, etc.
  - Suspicious open() for writing outside scratch — warning
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# ----- policy -----

# Harness + third-party (must stay in sync with requirements.txt).
THIRD_PARTY_ALLOWED: frozenset[str] = frozenset({
    # harness
    "polybench",
    "httpx",
    "websockets",
    "pyarrow",
    "pytest",
    # numerical / data
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "statsmodels",
    # classical ML
    "sklearn",          # scikit-learn's package name
    "lightgbm",
    "xgboost",
    "optuna",
    # deep learning
    "torch",
    "tensorflow",
    # indicators
    "ta",
})

# Stdlib safe subset — candidates commonly need these.
STDLIB_ALLOWED: frozenset[str] = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "collections", "heapq", "bisect", "itertools", "functools",
    "operator", "array",
    "random",
    "typing", "dataclasses", "enum", "abc", "types",
    "datetime", "time", "calendar", "zoneinfo",
    "json", "re", "string", "textwrap",
    "copy", "weakref", "contextlib", "contextvars",
    "pathlib", "io", "codecs",
    "logging",
    "hashlib", "hmac", "secrets", "uuid",
    "base64", "gzip", "zlib", "bz2", "lzma",
    "threading", "queue", "asyncio", "concurrent",
    "warnings", "traceback",
    "os",            # allowed but dangerous attrs blocked (see below)
    "sys",           # allowed but some attrs blocked
    "inspect",
    "struct",
    "unicodedata",
    "__future__",
})

ALLOWED_MODULES: frozenset[str] = STDLIB_ALLOWED | THIRD_PARTY_ALLOWED

# Hard-reject imports (takes precedence over "allowed" for anything that slips in).
BLOCKED_MODULES: frozenset[str] = frozenset({
    "subprocess",
    "ctypes",
    "pty",
    "telnetlib",
    "ftplib",
    "smtplib",
    "imaplib",
    "poplib",
    "socket",
    "ssl",
    "multiprocessing",
    "marshal",
    "pickle",
    "dill",
    "cloudpickle",
    "importlib",
    "resource",
    "fcntl",
    "signal",
    "shutil",
})

BLOCKED_CALL_NAMES: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "globals",
    "locals",
    "vars",
})

# attribute-path blocks: os.system(...), os.popen(...), etc.
BLOCKED_ATTR_PATHS: frozenset[str] = frozenset({
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.unlink",
    "os.remove",
    "os.rmdir",
    "os.removedirs",
    "sys.exit",          # not dangerous but disruptive to the harness loop
    "sys.modules",
})


# ----- scan result types -----

@dataclass(frozen=True, slots=True)
class Finding:
    severity: str       # "critical" | "warning"
    rule: str
    line: int
    col: int
    message: str


@dataclass(frozen=True, slots=True)
class ScanReport:
    file: str
    verdict: str        # "accept" | "reject"
    findings: tuple[Finding, ...]
    imports: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "verdict": self.verdict,
            "imports": list(self.imports),
            "findings": [asdict(f) for f in self.findings],
        }

    def format_text(self) -> str:
        lines = [
            f"polybench submission scan — {self.file}",
            "=" * 62,
            f"verdict: {self.verdict.upper()}",
            f"imports seen: {', '.join(self.imports) if self.imports else '(none)'}",
            "",
        ]
        if not self.findings:
            lines.append("no findings.")
        else:
            for f in self.findings:
                lines.append(f"  [{f.severity:>8}] {f.rule}  line {f.line}:{f.col}  — {f.message}")
        return "\n".join(lines)


# ----- scanner -----

def _attr_path(node: ast.AST) -> str | None:
    """Resolve ``a.b.c`` attribute chains into ``"a.b.c"``. Returns ``None``
    if the chain does not resolve to pure name+attributes."""
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.imports: list[str] = []

    # -- imports --
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod:
            self._check_module(mod, node)
        self.generic_visit(node)

    def _check_module(self, name: str, node: ast.AST) -> None:
        self.imports.append(name)
        root = name.split(".", 1)[0]
        if root in BLOCKED_MODULES:
            self._add(
                node,
                severity="critical",
                rule="blocked_import",
                message=f"import of blocked module {name!r}",
            )
            return
        if root not in ALLOWED_MODULES:
            self._add(
                node,
                severity="critical",
                rule="unknown_import",
                message=(
                    f"import of {name!r} — not in allowlist "
                    "(see requirements.txt and stdlib allowlist)"
                ),
            )

    # -- calls --
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BLOCKED_CALL_NAMES:
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_call",
                    message=f"call to blocked builtin {func.id!r}",
                )
            elif func.id == "open":
                self._check_open_call(node)
        # Attribute-path violations are handled in visit_Attribute (once).
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attr_path(node)
        if path is not None and path in BLOCKED_ATTR_PATHS:
            self._add(
                node,
                severity="critical",
                rule="blocked_attr",
                message=f"reference to blocked attribute {path}",
            )
        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call) -> None:
        mode: str | None = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value if isinstance(node.args[1].value, str) else None
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value if isinstance(kw.value.value, str) else mode
        if mode is None:
            return
        if any(m in mode for m in ("w", "a", "x", "+")):
            self._add(
                node,
                severity="warning",
                rule="open_for_write",
                message=(
                    f"open() with write mode {mode!r} — writes must stay inside "
                    "MarketInfo.scratch_dir or a tempfile.NamedTemporaryFile path"
                ),
            )

    # -- helpers --
    def _add(self, node: ast.AST, *, severity: str, rule: str, message: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        self.findings.append(
            Finding(severity=severity, rule=rule, line=line, col=col, message=message)
        )


def scan_source(source: str, *, path: str = "<string>") -> ScanReport:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        finding = Finding(
            severity="critical",
            rule="syntax_error",
            line=exc.lineno or 0,
            col=exc.offset or 0,
            message=f"syntax error: {exc.msg}",
        )
        return ScanReport(file=path, verdict="reject", findings=(finding,))

    visitor = _Visitor()
    visitor.visit(tree)

    critical = any(f.severity == "critical" for f in visitor.findings)
    verdict = "reject" if critical else "accept"
    return ScanReport(
        file=path,
        verdict=verdict,
        findings=tuple(visitor.findings),
        imports=tuple(visitor.imports),
    )


def scan_file(path: Path | str) -> ScanReport:
    p = Path(path)
    source = p.read_text()
    return scan_source(source, path=str(p))


def iter_critical(report: ScanReport) -> Iterable[Finding]:
    return (f for f in report.findings if f.severity == "critical")


def as_json(report: ScanReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
