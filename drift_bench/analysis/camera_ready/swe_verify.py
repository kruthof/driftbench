"""Executable-verification harness for the extended SWE pilot (plan items E6, C10).

The first SWE pilot judged *design proposals*; nothing was run. This module
is the "executable arm": for 4 briefs the subject must also emit a
``solution.py`` (see ``drift_bench/prompts_swe_code/``), and this harness
runs a brief-specific pytest file against it.

Pieces:

* ``run_tests(code_dir, brief_id, ...)``: copy ``solution.py`` and
  ``drift_bench/briefs_swe/tests/<brief_id>_test.py`` into a fresh temp
  directory, run pytest there in its own process group with a timeout,
  kill the whole group afterwards, and return pass/fail counts.
* ``static_checks(solution_path, brief_id)``: the programmatically
  checkable hard constraints (LOC cap, stdlib-only imports, forbidden
  modules and calls, public-function and public-method counts). Rules
  live in ``STATIC_RULES``; each rule names the constraint it checks.
* ``extract_blocks(text)`` / ``choose_block(blocks, entry_symbol)`` /
  ``extract_solution(text, entry_symbol)``: pull the python code block
  out of a final assistant message. The block that defines the brief's
  entry symbol wins; otherwise the longest.
* ``interface_spec(brief_id)``: the interface contract (the test file's
  module docstring). ``make_exec_briefs`` appends it to the brief
  objective so the subject sees the same contract the tests assume.
  The generated copies are checked in under ``drift_bench/briefs_swe_exec/``.
* ``extract_all(data_root)``: write every ``solution.py`` and run nothing,
  so the author can read the code first (``--extract-only``).
* ``run(data_root, out_dir)``: verify every transcript under
  ``<data_root>/camera_ready/e6_swe_verify/transcripts/`` and write
  ``swe_verify_results.csv`` with a provenance file.

Static-check limits: ``forbidden_calls`` resolves import aliases
(``import os as o; o.system(...)`` and ``from os import system; system(...)``
both read as ``os.system``) and ``builtins`` is a forbidden module for the
agentic briefs, but a call assembled from strings
(``getattr(__import__("o" + "s"), "sys" + "tem")``) is not caught; the
``__import__`` and ``getattr``-on-``builtins`` paths are, string building
is not. Reading the solution remains mandatory (``--extract-only``).

Isolation: NONE beyond a temp directory, a wall-clock timeout and a
process-group kill. The subject's code runs with this user's permissions
and network access. Read the generated solution before running it, or
run the harness inside a container. This is a deliberate limitation:
the arm has 16 runs and the author inspects each one.

Run: ``python -m drift_bench.analysis.camera_ready.swe_verify``
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from drift_bench.analysis.camera_ready.kbv_table import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    write_csv_with_provenance,
)

DRIFT_BENCH_DIR = Path(__file__).resolve().parents[2]
BRIEFS_SWE_DIR = DRIFT_BENCH_DIR / "briefs_swe"
BRIEFS_SWE_EXEC_DIR = DRIFT_BENCH_DIR / "briefs_swe_exec"
DEFAULT_TESTS_DIR = BRIEFS_SWE_DIR / "tests"
# Phase C config: config.yaml with subject max_tokens raised to 8192.
DEFAULT_CODE_CONFIG = DRIFT_BENCH_DIR / "config_swe_code.yaml"
SOLUTION_NAME = "solution.py"
# Never run from run_analysis item 6: this module executes model-written code.
RUN_IN_ITEM_6 = False
DEFAULT_TIMEOUT_S = 120.0

# Briefs in the executable arm, chosen for deterministic I/O and pure
# interfaces (clock injection makes the time-dependent ones deterministic).
EXEC_BRIEFS = ("swe_02", "swe_06", "swe_08", "swe_10")

# The symbol a solution must define, per contract. ``__main__`` marks a
# script. Used to pick the right block when the model emits several.
ENTRY_SYMBOLS = {
    "swe_02": "lru_ttl_cache",
    "swe_06": "__main__",
    "swe_08": "WorkQueue",
    "swe_10": "execute",
}

# Static rules per brief. Each entry maps to one hard constraint of the
# brief (quoted in "constraint"). Rules not listed here are left to the
# judge; for swe_06 that is the 10 GB / 2 min / 500 MB performance set.
# forbidden_calls: ``names`` are bare-name calls (``eval(...)``), never
# attribute calls, so ``re.compile`` does not trip ``compile``.
# ``attributes`` are ``<name>.<attr>(...)`` calls such as ``os.system``.
STATIC_RULES: dict[str, list[dict[str, Any]]] = {
    "swe_02": [
        {"rule": "stdlib_only", "constraint": "must use only the Python standard library"},
        {"rule": "forbidden_modules", "modules": ["sqlite3", "shelve", "pickle", "socket", "dbm"],
         "constraint": "must not persist cache state to disk or to an external cache server"},
    ],
    "swe_06": [
        {"rule": "stdlib_only", "constraint": "single-file Python script using only the standard library"},
        {"rule": "forbidden_modules", "modules": ["socket", "http", "urllib", "ssl", "ftplib", "smtplib",
                                                   "xmlrpc", "asyncio"],
         "constraint": "must not make any network calls"},
        {"rule": "forbidden_modules", "modules": ["pandas", "polars", "pyarrow"],
         "constraint": "no third-party parsing dependency (banned move)"},
    ],
    "swe_08": [
        {"rule": "stdlib_only", "constraint": "single process and use only the Python standard library"},
        {"rule": "forbidden_modules", "modules": ["multiprocessing", "subprocess", "sqlite3", "shelve", "pickle"],
         "constraint": "in-memory only, single process"},
        {"rule": "public_methods", "class": "WorkQueue", "count": 4,
         "constraint": "the public API must expose exactly four operations, enqueue, claim, complete, and fail"},
    ],
    "swe_10": [
        {"rule": "stdlib_only", "constraint": "must use only the Python standard library"},
        {"rule": "forbidden_calls", "names": ["eval", "exec", "compile", "__import__"],
         "attributes": ["importlib.import_module"],
         "constraint": "no eval, exec, or dynamic code loading"},
        {"rule": "forbidden_modules", "modules": ["importlib", "builtins", "subprocess", "multiprocessing", "asyncio"],
         "constraint": "no dynamic code loading; no subprocess isolation or async runtime (banned moves)"},
        {"rule": "max_loc", "limit": 250, "constraint": "under 250 lines of Python excluding tests and comments"},
        {"rule": "public_functions", "count": 1, "constraint": "exactly one public function, execute(call, registry)"},
    ],
    # Briefs outside the executable arm still get a static pass when a
    # solution exists (for anyone who widens the arm later).
    "swe_01": [
        {"rule": "stdlib_only", "constraint": "only the Python standard library"},
        {"rule": "max_loc", "limit": 200, "constraint": "under 200 lines of Python excluding tests and comments"},
        {"rule": "public_functions", "count": 2, "constraint": "exactly two functions, allow and reset"},
        {"rule": "forbidden_modules", "modules": ["threading", "multiprocessing", "asyncio", "sched"],
         "constraint": "must not spawn background threads, processes, or asyncio tasks"},
    ],
    "swe_09": [
        {"rule": "stdlib_only", "constraint": "only the Python standard library"},
        {"rule": "max_loc", "limit": 150, "constraint": "under 150 lines excluding tests and comments"},
        {"rule": "forbidden_modules", "modules": ["subprocess", "builtins", "pytest", "unittest"],
         "constraint": "no shell, no code execution, no test runner"},
        {"rule": "forbidden_calls", "names": ["exec", "eval"], "attributes": ["os.system", "os.popen"],
         "constraint": "no shell, no code execution"},
    ],
}

_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")
# A fenced python block. Opening fence: ``` then python / Python / python3 /
# py, then optional attributes (title=...), then a newline. Closing fence
# optional: an unclosed final block runs to the end of the text.
_PY_BLOCK_RE = re.compile(
    r"```[ \t]*(?:[Pp]ython3?|py)\b[^\n]*\n(.*?)(?:^[ \t]*```|\Z)",
    re.DOTALL | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Code extraction and interface contracts
# ---------------------------------------------------------------------------


def extract_blocks(text: str) -> list[dict[str, Any]]:
    """Every fenced python block in ``text`` as ``{code, unclosed}`` in order.

    CRLF line endings are normalised first. A block whose closing fence is
    missing runs to the end of the text and is flagged ``unclosed=True``.
    Each block is dedented; whitespace-only blocks are dropped, so a message
    holding only an empty fence counts as having no code.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    for m in _PY_BLOCK_RE.finditer(text):
        # The closing fence is part of the match only when it exists.
        unclosed = not m.group(0).endswith("```")
        # Dedent (models often indent a block under a list item) and drop
        # empty blocks, which otherwise read as a zero-line solution.
        code = textwrap.dedent(m.group(1)).strip("\n").rstrip()
        if not code.strip():
            continue
        blocks.append({"code": code + "\n", "unclosed": unclosed})
    return blocks


def _defines(code: str, symbol: str) -> bool:
    if symbol == "__main__":
        return "__main__" in code
    return re.search(rf"^\s*(?:def|class)\s+{re.escape(symbol)}\b", code, re.MULTILINE) is not None


def choose_block(blocks: list[dict[str, Any]], entry_symbol: str | None = None) -> dict[str, Any] | None:
    """Prefer a block that defines ``entry_symbol``; else the longest. Ties: the later block."""
    if not blocks:
        return None
    candidates = [b for b in blocks if entry_symbol and _defines(b["code"], entry_symbol)] or blocks
    best = None
    for b in candidates:
        if best is None or len(b["code"]) >= len(best["code"]):
            best = b
    return best


def extract_solution(text: str, entry_symbol: str | None = None) -> str | None:
    """The chosen block's code, or None when the text has no python block."""
    chosen = choose_block(extract_blocks(text), entry_symbol)
    return None if chosen is None else chosen["code"]


def test_file_for(brief_id: str, tests_dir: Path = DEFAULT_TESTS_DIR) -> Path:
    return Path(tests_dir) / f"{brief_id}_test.py"


def interface_spec(brief_id: str, tests_dir: Path = DEFAULT_TESTS_DIR) -> str:
    """The interface contract for a brief: the module docstring of its test file."""
    path = test_file_for(brief_id, tests_dir)
    doc = ast.get_docstring(ast.parse(path.read_text()))
    if not doc:
        raise ValueError(f"{path} has no module docstring (interface contract)")
    return doc.strip()


def make_exec_briefs(
    briefs_dir: Path, out_dir: Path, brief_ids: tuple[str, ...] = EXEC_BRIEFS,
    tests_dir: Path = DEFAULT_TESTS_DIR,
) -> list[Path]:
    """Write copies of ``brief_ids`` with the interface contract appended to the objective.

    The schema has no field for an interface, so the contract goes into the
    objective text. Ids and every other field stay unchanged, so the judge
    and the probe analysis treat the copies like the originals. Output is
    deterministic: regenerating gives byte-identical files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for brief_id in brief_ids:
        src = Path(briefs_dir) / f"{brief_id}.yaml"
        brief = yaml.safe_load(src.read_text())
        brief["objective"] = (
            brief["objective"].strip()
            + "\n\nInterface contract for the implementation (solution.py):\n"
            + interface_spec(brief_id, tests_dir)
        )
        dst = out_dir / src.name
        dst.write_text(yaml.dump(brief, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=100))
        written.append(dst)
    return written


class _LiteralDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line strings as literal blocks (``|``), for readable briefs."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_LiteralDumper.add_representer(str, _represent_str)


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def count_loc(source: str) -> int:
    """Non-blank lines that are not pure comment lines.

    Docstrings count as code. The briefs say "excluding tests and comments";
    they do not mention docstrings, so the count is conservative (higher).
    """
    n = 0
    for line in source.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def imported_modules(tree: ast.AST) -> set[str]:
    """Dotted names of every import in ``tree`` plus their top-level package."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
    return names


def name_calls(tree: ast.AST) -> set[str]:
    """Bare names that are called: ``eval(...)`` yes, ``re.compile(...)`` no."""
    return {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def import_aliases(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Alias maps from the imports in ``tree``.

    Returns ``(module_aliases, from_imports)``: ``import os as o`` gives
    ``{"o": "os"}``; ``from os import system as s`` gives
    ``{"s": "os.system"}``. Unaliased imports map a name to itself.
    """
    module_aliases: dict[str, str] = {}
    from_imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                from_imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return module_aliases, from_imports


def attribute_calls(tree: ast.AST) -> set[str]:
    """Dotted names of attribute-style calls, with import aliases resolved.

    ``os.system(...)`` -> ``os.system``; ``import os as o; o.system(...)``
    -> ``os.system``; ``from os import system; system(...)`` -> ``os.system``.
    Calls assembled through ``getattr`` or string building are not seen.
    """
    module_aliases, from_imports = import_aliases(tree)
    out: set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            base = module_aliases.get(f.value.id, f.value.id)
            out.add(f"{base}.{f.attr}")
        elif isinstance(f, ast.Name) and f.id in from_imports:
            out.add(from_imports[f.id])
    return out


def public_functions(tree: ast.Module) -> list[str]:
    """Module-level ``def`` names that do not start with an underscore.

    Only top-level ``def`` statements count. Classes, nested functions
    and assigned lambdas do not.
    """
    return [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]


def public_methods(tree: ast.Module, class_name: str) -> list[str] | None:
    """Public method names of the top-level class ``class_name``; None if the class is absent."""
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            return [m.name for m in n.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")]
    return None


def static_checks(solution_path: Path, brief_id: str, rules: dict | None = None) -> list[dict[str, Any]]:
    """Apply ``STATIC_RULES[brief_id]`` to a solution file.

    Returns one dict per rule: ``rule, constraint, passed, detail``. A file
    that does not parse fails every rule with the syntax error as detail.
    """
    rules = (rules or STATIC_RULES).get(brief_id, [])
    source = Path(solution_path).read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"rule": r["rule"], "constraint": r["constraint"], "passed": False,
                 "detail": f"syntax error: {e}"} for r in rules]
    imports = imported_modules(tree)
    results = []
    for r in rules:
        kind = r["rule"]
        if kind == "stdlib_only":
            bad = sorted(m for m in imports if "." not in m and m not in sys.stdlib_module_names)
            passed, detail = not bad, f"non-stdlib imports: {bad}" if bad else "all imports are stdlib"
        elif kind == "forbidden_modules":
            hit = sorted(m for m in r["modules"] if m in imports or any(i.startswith(m + ".") for i in imports))
            passed, detail = not hit, f"forbidden imports: {hit}" if hit else "none of the forbidden modules imported"
        elif kind == "forbidden_calls":
            hit = sorted(set(r.get("names", [])) & name_calls(tree))
            hit += sorted(set(r.get("attributes", [])) & attribute_calls(tree))
            passed, detail = not hit, f"forbidden calls: {hit}" if hit else "no forbidden call"
        elif kind == "max_loc":
            loc = count_loc(source)
            passed, detail = loc < r["limit"], f"{loc} lines (limit {r['limit']})"
        elif kind == "public_functions":
            fns = public_functions(tree)
            passed, detail = len(fns) == r["count"], f"public functions: {fns} (expected {r['count']})"
        elif kind == "public_methods":
            methods = public_methods(tree, r["class"])
            if methods is None:
                passed, detail = False, f"class {r['class']} not defined at module level"
            else:
                passed, detail = len(methods) == r["count"], f"{r['class']} public methods: {methods} (expected {r['count']})"
        else:
            raise ValueError(f"unknown static rule {kind!r}")
        results.append({"rule": kind, "constraint": r["constraint"], "passed": bool(passed), "detail": detail})
    return results


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Read ``N passed, M failed, K error`` from pytest's last summary line."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            for n, kind in _SUMMARY_RE.findall(line):
                key = "errors" if kind.startswith("error") else kind
                counts[key] = int(n)
            if any(counts.values()):
                break
    return counts


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the process group pytest started (``start_new_session=True`` makes pgid == pid)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_tests(
    code_dir: Path,
    brief_id: str,
    tests_dir: Path = DEFAULT_TESTS_DIR,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    python: str | None = None,
) -> dict[str, Any]:
    """Run ``<tests_dir>/<brief_id>_test.py`` against ``<code_dir>/solution.py``.

    Both files are copied into a fresh temp directory (deleted afterwards)
    and pytest runs there in its own session (process group) with
    ``timeout_s`` as a hard wall-clock cap. Output goes to a file, not a
    pipe, so a child the solution spawned cannot hold the harness open.
    The whole group is killed when pytest exits or times out, so such
    children do not survive. No other sandboxing: see the module docstring.

    Returns a dict with ``brief_id, passed, failed, errors, skipped,
    total, all_passed, returncode, timed_out, duration_s, missing,
    stdout_tail``. ``missing`` names a file that was not found; the
    counts are then zero and ``returncode`` is None.
    """
    code_dir = Path(code_dir)
    solution = code_dir / SOLUTION_NAME
    test_file = test_file_for(brief_id, tests_dir)
    base = {"brief_id": brief_id, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0,
            "all_passed": False, "returncode": None, "timed_out": False, "duration_s": 0.0,
            "missing": None, "stdout_tail": ""}
    if not solution.is_file():
        return {**base, "missing": str(solution)}
    if not test_file.is_file():
        return {**base, "missing": str(test_file)}

    tmp = Path(tempfile.mkdtemp(prefix=f"swe_verify_{brief_id}_"))
    try:
        shutil.copy(solution, tmp / SOLUTION_NAME)
        shutil.copy(test_file, tmp / test_file.name)
        # An empty ini pins pytest's rootdir and config to the temp dir, so
        # the project's pyproject.toml and conftest.py are never picked up.
        (tmp / "pytest.ini").write_text("[pytest]\n")
        cmd = [python or sys.executable, "-m", "pytest", test_file.name, "-q", "-p", "no:cacheprovider",
               "--rootdir", str(tmp), "-c", str(tmp / "pytest.ini")]
        log_path = tmp / "pytest_output.txt"
        t0 = time.monotonic()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd, cwd=tmp, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                     "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                start_new_session=True,
            )
            try:
                returncode, timed_out = proc.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
            finally:
                _kill_group(proc)
                proc.wait()
        duration = time.monotonic() - t0
        output = log_path.read_text(errors="replace")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    counts = parse_pytest_summary(output)
    total = counts["passed"] + counts["failed"] + counts["errors"]
    return {
        **base, **counts, "total": total,
        "all_passed": (not timed_out) and returncode == 0 and counts["passed"] > 0
                      and counts["failed"] == 0 and counts["errors"] == 0,
        "returncode": returncode, "timed_out": timed_out, "duration_s": round(duration, 3),
        "stdout_tail": output[-2000:],
    }


def verify_solution(code_dir: Path, brief_id: str, tests_dir: Path = DEFAULT_TESTS_DIR,
                    timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Static checks plus tests for one solution directory, flattened for a CSV row."""
    result = run_tests(code_dir, brief_id, tests_dir=tests_dir, timeout_s=timeout_s)
    solution = Path(code_dir) / SOLUTION_NAME
    checks = static_checks(solution, brief_id) if solution.is_file() else []
    result["static_checks"] = len(checks)
    result["static_failed"] = sum(1 for c in checks if not c["passed"])
    result["static_detail"] = json.dumps(checks)
    return result


# ---------------------------------------------------------------------------
# Transcript-level driver
# ---------------------------------------------------------------------------


def subject_max_tokens(config_path: Path, model_id: str) -> int | None:
    """``max_tokens`` of ``model_id`` in a config file, or None when the model is absent."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    for m in config["models"]["subjects"]:
        if m["id"] == model_id:
            return int(m["max_tokens"])
    return None


def final_assistant_message(record: dict) -> dict | None:
    for msg in reversed(record.get("messages", [])):
        if msg.get("role") == "assistant" and not msg.get("is_probe", False):
            return msg
    return None


def final_assistant_text(record: dict) -> str | None:
    msg = final_assistant_message(record)
    return None if msg is None else msg.get("content", "")


def verify_transcript(path: Path, tests_dir: Path = DEFAULT_TESTS_DIR,
                      timeout_s: float = DEFAULT_TIMEOUT_S, keep_code_dir: Path | None = None,
                      config_path: Path = DEFAULT_CODE_CONFIG,
                      refused_reason: str | None = None) -> dict[str, Any]:
    """Extract solution.py from a transcript's final answer, then verify it.

    ``keep_code_dir``, when set, receives ``<run_id>/solution.py`` so the
    author can read what ran. Rows with ``no_code=True`` had no python
    block. ``truncated`` is True when the final message's ``finish_reason``
    is ``"length"`` or its ``token_count`` reached the subject's
    ``max_tokens`` in ``config_path``; ``n_blocks`` and ``unclosed_block``
    describe the extraction. ``refused_reason``, when given, marks the row
    ``refused`` and nothing is extracted to disk or executed.
    """
    with open(path) as fh:
        record = json.loads(fh.readline())
    meta = record["metadata"]
    row = {"run_id": meta["run_id"], "brief_id": meta["brief_id"], "model_id": meta["model_id"],
           "condition": meta["condition"], "transcript": path.name, "no_code": False,
           "n_blocks": 0, "unclosed_block": False, "finish_reason": None, "truncated": False,
           "refused": False, "refused_reason": ""}
    msg = final_assistant_message(record) or {}
    cap = subject_max_tokens(config_path, meta["model_id"]) if Path(config_path).is_file() else None
    tokens = msg.get("token_count")
    row["finish_reason"] = msg.get("finish_reason")
    row["truncated"] = bool(row["finish_reason"] == "length"
                            or (cap is not None and tokens is not None and tokens >= cap))
    blocks = extract_blocks(msg.get("content", "") or "")
    chosen = choose_block(blocks, ENTRY_SYMBOLS.get(meta["brief_id"]))
    row["n_blocks"] = len(blocks)
    if refused_reason is not None:
        # The author read this solution and refused to execute it (see README).
        row["refused"], row["refused_reason"] = True, refused_reason
        row["no_code"] = chosen is None
        row["unclosed_block"] = bool(chosen and chosen["unclosed"])
        row.update({"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0, "all_passed": False,
                    "returncode": None, "timed_out": False, "duration_s": 0.0, "missing": None,
                    "stdout_tail": "", "static_checks": 0, "static_failed": 0, "static_detail": "[]"})
        return row
    if chosen is None:
        row["no_code"] = True
        with tempfile.TemporaryDirectory() as empty:
            row.update(verify_solution(Path(empty), meta["brief_id"], tests_dir, timeout_s))
        return row
    row["unclosed_block"] = bool(chosen["unclosed"])
    code_root = Path(keep_code_dir) if keep_code_dir else Path(tempfile.mkdtemp(prefix="swe_verify_code_"))
    code_dir = code_root / meta["run_id"]
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / SOLUTION_NAME).write_text(chosen["code"])
    try:
        row.update(verify_solution(code_dir, meta["brief_id"], tests_dir, timeout_s))
    finally:
        if keep_code_dir is None:
            shutil.rmtree(code_root, ignore_errors=True)
    return row


def extract_all(data_root: Path = DEFAULT_DATA_ROOT) -> list[Path]:
    """Write ``<run_id>/solution.py`` for every transcript of the arm. Runs nothing.

    Use this first, read every file, then call ``run``. Transcripts without
    a python block get no directory.
    """
    data_root = Path(data_root)
    tdir = data_root / "camera_ready" / "e6_swe_verify" / "transcripts"
    code_root = tdir.parent / "code"
    written = []
    for path in sorted(tdir.glob("*.jsonl")) if tdir.is_dir() else []:
        with open(path) as fh:
            record = json.loads(fh.readline())
        meta = record["metadata"]
        msg = final_assistant_message(record) or {}
        chosen = choose_block(extract_blocks(msg.get("content", "") or ""), ENTRY_SYMBOLS.get(meta["brief_id"]))
        if chosen is None:
            continue
        code_dir = code_root / meta["run_id"]
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / SOLUTION_NAME).write_text(chosen["code"])
        written.append(code_dir / SOLUTION_NAME)
    return written


def load_exclude_file(path: Path | None) -> dict[str, str]:
    """``run_id [reason...]`` per line, ``#`` comments allowed -> ``{run_id: reason}``."""
    if path is None:
        return {}
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            run_id, _, reason = line.partition(" ")
            out[run_id] = reason.strip() or "refused"
    return out


def _run_id_of(path: Path) -> str:
    with open(path) as fh:
        return json.loads(fh.readline())["metadata"]["run_id"]


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR,
        tests_dir: Path = DEFAULT_TESTS_DIR, timeout_s: float = DEFAULT_TIMEOUT_S,
        config_path: Path = DEFAULT_CODE_CONFIG,
        exclude_run_ids: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Verify every transcript of the executable arm and write ``swe_verify_results.csv``.

    Input: ``<data_root>/camera_ready/e6_swe_verify/transcripts/*.jsonl``.
    Extracted code is kept under ``<data_root>/camera_ready/e6_swe_verify/code/<run_id>/``.
    ``exclude_run_ids`` (``{run_id: reason}``) marks rows ``refused``: the
    author read the code and declined to execute it. They get a row with
    zero counts and are never executed.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    exclude_run_ids = exclude_run_ids or {}
    tdir = data_root / "camera_ready" / "e6_swe_verify" / "transcripts"
    paths = sorted(tdir.glob("*.jsonl")) if tdir.is_dir() else []
    rows = [verify_transcript(p, tests_dir, timeout_s, keep_code_dir=tdir.parent / "code",
                              config_path=config_path, refused_reason=exclude_run_ids.get(_run_id_of(p)))
            for p in paths]
    df = pd.DataFrame(rows)
    write_csv_with_provenance(df, out_dir / "swe_verify_results.csv", {
        "inputs": [str(tdir / "*.jsonl"), str(config_path)],
        "rows_in": len(paths),
        "filters": [
            "one row per transcript; the final non-probe assistant message is the source of solution.py",
            "solution.py = the fenced python block that defines the brief's entry symbol "
            "(ENTRY_SYMBOLS), else the longest block; no block -> no_code=True and zero counts",
            "n_blocks = non-empty fenced python blocks in that message (dedented); "
            "unclosed_block = chosen block had no closing fence",
            f"truncated = final message finish_reason == 'length' or token_count >= subject max_tokens in {Path(config_path).name}",
            f"refused rows ({len(exclude_run_ids)} run ids given): read by the author, not executed, zero counts",
            f"tests: drift_bench/briefs_swe/tests/<brief_id>_test.py, pytest subprocess, timeout {timeout_s}s",
            "static checks: STATIC_RULES[brief_id] in swe_verify.py",
            "isolation: temp dir, timeout and process-group kill only (no sandbox)",
        ],
        "date": dt.date.today().isoformat(),
    })
    return {"swe_verify_results": df}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CODE_CONFIG,
                        help="Config whose subject max_tokens defines the truncated column")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--make-exec-briefs", type=Path, default=None,
                        help="Write the 4 executable-arm briefs with interface contracts to this directory and exit")
    parser.add_argument("--extract-only", action="store_true",
                        help="Write code/<run_id>/solution.py for every transcript and exit without running anything")
    parser.add_argument("--exclude-run-ids-file", type=Path, default=None,
                        help="Lines 'run_id reason': rows marked refused and never executed")
    args = parser.parse_args()
    if args.make_exec_briefs:
        for p in make_exec_briefs(BRIEFS_SWE_DIR, args.make_exec_briefs, tests_dir=args.tests_dir):
            print(p)
        return
    if args.extract_only:
        for p in extract_all(args.data_root):
            print(p)
        return
    out = run(args.data_root, args.out_dir, args.tests_dir, args.timeout, args.config,
              exclude_run_ids=load_exclude_file(args.exclude_run_ids_file))
    df = out["swe_verify_results"]
    print(f"swe_verify_results.csv: {len(df)} rows")
    if len(df):
        print(df[["run_id", "brief_id", "model_id", "condition", "passed", "failed", "errors",
                  "timed_out", "truncated", "refused", "static_failed"]].to_string(index=False))


if __name__ == "__main__":
    main()
