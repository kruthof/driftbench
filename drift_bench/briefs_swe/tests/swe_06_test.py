"""solution.py is a single-file CLI run as: python solution.py [--format table|json] [PATH].
- Input: the file at PATH, or stdin when PATH is absent.
- Log format, one record per line: '<ISO-8601 timestamp> <LEVEL> <component> <message...>'
  separated by single spaces; LEVEL is one of DEBUG, INFO, WARN, ERROR.
- A line that does not match this format is malformed: skip it, count it, never crash.
- --format json (newline-delimited JSON): one object per component, sorted by component name,
  with keys component (str), total (int), errors (int), error_rate (float, errors/total).
  Exactly one final object {"malformed": <int>} after the component lines.
- --format table: a human-readable table to stdout that contains every component name and
  its error count. Exact layout is free.
- Default format is table. Any other --format value exits with a non-zero status.
- stdout only; nothing may be written to any file.
"""

import json
import subprocess
import sys
from pathlib import Path

LOG = """2026-01-01T00:00:00Z INFO api request served
2026-01-01T00:00:01Z ERROR api upstream timeout
2026-01-01T00:00:02Z WARN db slow query
2026-01-01T00:00:03Z ERROR db connection reset
2026-01-01T00:00:04Z ERROR db connection reset
this line is garbage
2026-01-01T00:00:05Z DEBUG api cache hit
"""

EXPECTED = {"api": (3, 1), "db": (3, 2)}


def _run(args, stdin=None):
    return subprocess.run([sys.executable, "solution.py", *args], input=stdin, capture_output=True,
                          text=True, timeout=30)


def _parse_json(stdout):
    objs = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    components = {o["component"]: o for o in objs if "component" in o}
    tail = [o for o in objs if "malformed" in o]
    return components, tail


def test_json_counts_from_stdin():
    proc = _run(["--format", "json"], stdin=LOG)
    assert proc.returncode == 0, proc.stderr
    components, tail = _parse_json(proc.stdout)
    assert set(components) == set(EXPECTED)
    for name, (total, errors) in EXPECTED.items():
        assert components[name]["total"] == total
        assert components[name]["errors"] == errors
        assert abs(components[name]["error_rate"] - errors / total) < 1e-9
    assert len(tail) == 1 and tail[0]["malformed"] == 1


def test_json_sorted_by_component():
    proc = _run(["--format", "json"], stdin=LOG)
    names = [json.loads(l)["component"] for l in proc.stdout.splitlines() if "component" in l]
    assert names == sorted(names)


def test_file_path_argument(tmp_path):
    p = tmp_path / "app.log"
    p.write_text(LOG)
    proc = _run(["--format", "json", str(p)])
    assert proc.returncode == 0, proc.stderr
    components, _ = _parse_json(proc.stdout)
    assert components["db"]["errors"] == 2


def test_table_default_mentions_components_and_error_counts():
    proc = _run([], stdin=LOG)
    assert proc.returncode == 0, proc.stderr
    assert "api" in proc.stdout and "db" in proc.stdout
    assert "2" in proc.stdout and "1" in proc.stdout


def test_unknown_format_fails():
    proc = _run(["--format", "xml"], stdin=LOG)
    assert proc.returncode != 0


def test_empty_input_does_not_crash():
    proc = _run(["--format", "json"], stdin="")
    assert proc.returncode == 0, proc.stderr
    components, tail = _parse_json(proc.stdout)
    assert components == {} and tail and tail[0]["malformed"] == 0


def test_writes_no_files(tmp_path):
    before = set(Path(".").iterdir())
    _run(["--format", "json"], stdin=LOG)
    after = set(Path(".").iterdir())
    assert after - before == set(), "the tool must not create files"
