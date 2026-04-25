"""AST-based security screener for candidate submissions.

This is a static, best-effort check — NOT a sandbox. It catches accidental
filesystem damage, obviously-malicious code, and disallowed imports before
the harness loads the submission module. A motivated bad actor could defeat
it; for that reason the scoring run also applies OS-level resource limits
and runs the submission inside a dedicated working directory.

Policy:
  - Blocked imports (hard reject)                — see ``BLOCKED_MODULES``
  - Only allowlisted imports permitted          — see ``ALLOWED_MODULES``
  - Blocked call targets (hard reject)          — eval/exec/compile/__import__/open/etc.
  - Blocked attribute paths (hard reject)       — os.system, Path.write_text, os.environ, etc.
  - Blocked module monkey-patching              — assignments to imported module attributes
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# ----- policy -----

# Candidate-facing third-party imports. Harness-only runtime dependencies
# can exist in requirements.txt without being valid submission imports.
THIRD_PARTY_ALLOWED: frozenset[str] = frozenset({
    # harness
    "polybench",
    "httpx",
    "websockets",
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
    "inspect",
    "os",
    "resource",
    "fcntl",
    "signal",
    "shutil",
    "sys",
})

BLOCKED_CALL_NAMES: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
})

# attribute-path blocks: os.system(...), os.popen(...), Path.write_text(...), etc.
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
    "os.rename",
    "os.replace",
    "os.renames",
    "os.mkdir",
    "os.makedirs",
    "os.chmod",
    "os.chown",
    "os.getenv",
    "os.putenv",
    "os.unsetenv",
    "os.environ",
    "sys.exit",          # not dangerous but disruptive to the harness loop
    "sys.modules",
    "pathlib.Path.open",
    "pathlib.Path.read_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
    "pathlib.Path.unlink",
    "pathlib.Path.rmdir",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "pathlib.Path.mkdir",
    "pathlib.Path.touch",
    "pathlib.Path.chmod",
    "pathlib.Path.owner",
    "pathlib.Path.group",
    "pathlib.Path.home",
    "pathlib.Path.cwd",
    "pathlib.Path.resolve",
    "pathlib.Path.absolute",
    "bz2.open",
    "codecs.open",
    "gzip.open",
    "io.FileIO",
    "io.open",
    "lzma.open",
    "numpy.load",
    "numpy.loadtxt",
    "numpy.genfromtxt",
    "numpy.fromfile",
    "numpy.fromregex",
    "numpy.memmap",
    "numpy.save",
    "numpy.savez",
    "numpy.savetxt",
    "numpy.DataSource",
    "numpy.lib.npyio.DataSource",
    "operator.attrgetter",
    "operator.methodcaller",
    "pandas.ExcelFile",
    "pandas.HDFStore",
    "pandas.read_csv",
    "pandas.read_clipboard",
    "pandas.read_excel",
    "pandas.read_feather",
    "pandas.read_fwf",
    "pandas.read_gbq",
    "pandas.read_hdf",
    "pandas.read_html",
    "pandas.read_json",
    "pandas.read_orc",
    "pandas.read_parquet",
    "pandas.read_pickle",
    "pandas.read_sas",
    "pandas.read_sql",
    "pandas.read_spss",
    "pandas.read_stata",
    "pandas.read_table",
    "pandas.read_xml",
    "polars.read_csv",
    "polars.read_database",
    "polars.read_delta",
    "polars.read_excel",
    "polars.read_ipc",
    "polars.read_ndjson",
    "polars.read_json",
    "polars.read_parquet",
    "polars.scan_csv",
    "polars.scan_delta",
    "polars.scan_iceberg",
    "polars.scan_ipc",
    "polars.scan_ndjson",
    "polars.scan_parquet",
    "polars.scan_pyarrow_dataset",
})

BLOCKED_ATTR_PREFIXES: tuple[str, ...] = (
    "pandas.read_",
    "polars.read_",
    "polars.scan_",
)

BLOCKED_ATTR_NAMES: frozenset[str] = frozenset({
    # Filesystem methods on pathlib objects and file-like helpers. The scanner
    # cannot prove a path stays under MarketInfo.scratch_dir, so direct file IO
    # is rejected for submitted models.
    "open",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "unlink",
    "rmdir",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "owner",
    "group",
    "home",
    "cwd",
    "resolve",
    "absolute",
    "expanduser",
    # Data-library file IO helpers.
    "to_csv",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_feather",
    "to_sql",
})

BLOCKED_DUNDER_ATTRS: frozenset[str] = frozenset({
    "__bases__",
    "__base__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__globals__",
    "__getattribute__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__setattr__",
    "__subclasses__",
})

BLOCKED_NAMES: frozenset[str] = frozenset({
    "__builtins__",
})

PROTECTED_MODULE_ROOTS: frozenset[str] = frozenset({
    "polybench",
    "httpx",
    "websockets",
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "statsmodels",
    "sklearn",
    "lightgbm",
    "xgboost",
    "optuna",
    "torch",
    "tensorflow",
    "ta",
})

POLYBENCH_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "polybench",
    "polybench.model",
    "polybench.baselines",
})

POLYBENCH_ROOT_ALLOWED_NAMES: frozenset[str] = frozenset({
    "FLAT",
    "EventResult",
    "MarketInfo",
    "Model",
    "RunResult",
    "Side",
    "Signal",
    "Tick",
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


def _is_blocked_attr_path(path: str) -> bool:
    return path in BLOCKED_ATTR_PATHS or any(
        path.startswith(prefix) for prefix in BLOCKED_ATTR_PREFIXES
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.imports: list[str] = []
        self.aliases: dict[str, str] = {}

    # -- imports --
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
            local_name = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod:
            self._check_module(mod, node)
        for alias in node.names:
            if alias.name == "*":
                self._add(
                    node,
                    severity="critical",
                    rule="star_import",
                    message="star imports are not allowed in submissions",
                )
                continue
            if mod == "polybench" and alias.name not in POLYBENCH_ROOT_ALLOWED_NAMES:
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_import",
                    message=(
                        f"from polybench import {alias.name!r} is not allowed — "
                        "import only the public model API names"
                    ),
                )
            local_name = alias.asname or alias.name
            self.aliases[local_name] = f"{mod}.{alias.name}" if mod else alias.name
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
        if root == "polybench" and name not in POLYBENCH_ALLOWED_IMPORTS:
            self._add(
                node,
                severity="critical",
                rule="blocked_import",
                message=(
                    f"import of {name!r} — submissions may import only "
                    "polybench's public model API"
                ),
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
            normalized_name = self._normalize_path(func.id)
            if func.id in BLOCKED_CALL_NAMES:
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_call",
                    message=f"call to blocked builtin {func.id!r}",
                )
            elif _is_blocked_attr_path(normalized_name):
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_call",
                    message=f"call to blocked target {normalized_name}",
                )
        # Attribute-path violations are handled in visit_Attribute (once).
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raw_path = _attr_path(node)
        path = self._normalize_path(raw_path) if raw_path is not None else None
        if path is not None and _is_blocked_attr_path(path):
            self._add(
                node,
                severity="critical",
                rule="blocked_attr",
                message=f"reference to blocked attribute {path}",
            )
        elif node.attr in BLOCKED_ATTR_NAMES:
            self._add(
                node,
                severity="critical",
                rule="blocked_attr",
                message=f"reference to blocked attribute {node.attr}",
            )
        elif node.attr in BLOCKED_DUNDER_ATTRS:
            self._add(
                node,
                severity="critical",
                rule="blocked_attr",
                message=f"reference to blocked introspection attribute {node.attr}",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BLOCKED_NAMES:
            self._add(
                node,
                severity="critical",
                rule="blocked_name",
                message=f"reference to blocked name {node.id}",
            )
        if isinstance(node.ctx, ast.Load):
            normalized_name = self._normalize_path(node.id)
            if node.id in BLOCKED_CALL_NAMES:
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_name",
                    message=f"reference to blocked builtin {node.id!r}",
                )
            elif _is_blocked_attr_path(normalized_name):
                self._add(
                    node,
                    severity="critical",
                    rule="blocked_name",
                    message=f"reference to blocked target {normalized_name}",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_assignment_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment_targets([node.target])
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment_targets([node.target])
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._check_assignment_targets(node.targets, action="delete")
        self.generic_visit(node)

    def _check_assignment_targets(
        self, targets: Iterable[ast.expr], *, action: str = "assignment"
    ) -> None:
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                self._check_assignment_targets(target.elts, action=action)
                continue
            if not isinstance(target, ast.Attribute):
                continue
            raw_path = _attr_path(target)
            if raw_path is None:
                continue
            path = self._normalize_path(raw_path)
            root = path.split(".", 1)[0]
            if root in PROTECTED_MODULE_ROOTS:
                self._add(
                    target,
                    severity="critical",
                    rule="module_mutation",
                    message=f"{action} to imported module attribute {path}",
                )

    def _normalize_path(self, path: str) -> str:
        head, sep, rest = path.partition(".")
        mapped = self.aliases.get(head)
        if mapped is None:
            return path
        return f"{mapped}{sep}{rest}" if sep else mapped

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
