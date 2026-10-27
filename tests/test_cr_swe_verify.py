"""E6 executable arm: the swe_verify harness on a toy brief in tmp_path.

Runs pytest in a subprocess (no network, no API keys). The toy brief and its
test file are written to tmp_path; the real brief test files are only
parsed, never executed here.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from drift_bench.analysis.camera_ready import swe_verify as sv
from drift_bench.schema.validate import load_all_briefs, validate_brief

TOY_TEST = '''"""solution.py must define add(a, b) -> a + b."""


def test_add():
    from solution import add
    assert add(1, 2) == 3


def test_add_wrong_on_purpose():
    from solution import add
    assert add(1, 1) == 3
'''


@pytest.fixture
def toy(tmp_path: Path) -> dict:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "toy_01_test.py").write_text(TOY_TEST)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    return {"tests_dir": tests_dir, "code_dir": code_dir}


# --- run_tests ---------------------------------------------------------------


def test_run_tests_counts_pass_and_fail(toy: dict) -> None:
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=60)
    assert (r["passed"], r["failed"], r["errors"]) == (1, 1, 0)
    assert r["total"] == 2 and r["returncode"] == 1
    assert r["all_passed"] is False and r["timed_out"] is False and r["missing"] is None
    assert "test_add_wrong_on_purpose" in r["stdout_tail"]


def test_run_tests_all_pass(toy: dict) -> None:
    (toy["tests_dir"] / "toy_01_test.py").write_text(TOY_TEST.split("def test_add_wrong")[0])
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=60)
    assert r["all_passed"] is True and r["passed"] == 1 and r["returncode"] == 0


def test_run_tests_missing_files(toy: dict, tmp_path: Path) -> None:
    r = sv.run_tests(tmp_path / "nowhere", "toy_01", tests_dir=toy["tests_dir"])
    assert r["missing"].endswith("solution.py") and r["total"] == 0 and r["returncode"] is None
    r = sv.run_tests(toy["code_dir"], "toy_99", tests_dir=toy["tests_dir"])
    assert r["missing"].endswith("toy_99_test.py")


def test_run_tests_timeout(toy: dict) -> None:
    (toy["code_dir"] / "solution.py").write_text("import time\ndef add(a, b):\n    time.sleep(30)\n    return a + b\n")
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=2)
    assert r["timed_out"] is True and r["returncode"] is None and r["all_passed"] is False
    assert r["duration_s"] < 15


def test_run_tests_broken_solution_counts_errors(toy: dict) -> None:
    (toy["code_dir"] / "solution.py").write_text("def add(a, b)\n    return a + b\n")  # syntax error
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=60)
    assert r["passed"] == 0 and r["all_passed"] is False and r["returncode"] not in (0, None)


def _process_gone(pid: int, wait_s: float = 5.0) -> bool:
    """True once ``pid`` no longer exists or is a zombie awaiting reaping."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        time.sleep(0.2)
    return False


def _spawning_solution(pid_file: Path) -> str:
    """A solution that spawns ``sleep 40`` at import and records the child's pid."""
    return (
        "import subprocess\n"
        f"_child = subprocess.Popen(['sleep', '40'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(_child.pid))\n"
        "def add(a, b):\n    return a + b\n"
    )


def test_orphan_spawned_at_import_is_killed_after_normal_exit(toy: dict, tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    (toy["code_dir"] / "solution.py").write_text(_spawning_solution(pid_file))
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=30)
    assert r["timed_out"] is False and r["passed"] == 1, r["stdout_tail"]
    assert r["duration_s"] < 20, "an inherited pipe must not hold the harness open"
    pid = int(pid_file.read_text())
    assert _process_gone(pid), f"sleep 40 (pid {pid}) survived the process-group kill"


def test_orphan_spawned_at_import_is_killed_after_timeout(toy: dict, tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    (toy["code_dir"] / "solution.py").write_text(
        _spawning_solution(pid_file).replace("    return a + b", "    import time; time.sleep(30); return a + b"))
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=2)
    assert r["timed_out"] is True
    pid = int(pid_file.read_text())
    assert _process_gone(pid), f"sleep 40 (pid {pid}) survived the process-group kill"


def test_parse_pytest_summary() -> None:
    assert sv.parse_pytest_summary("....\n4 passed in 0.1s\n") == {"passed": 4, "failed": 0, "errors": 0, "skipped": 0}
    assert sv.parse_pytest_summary("2 failed, 3 passed, 1 error in 1s") == {"passed": 3, "failed": 2, "errors": 1, "skipped": 0}
    assert sv.parse_pytest_summary("no summary here") == {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}


# --- extraction --------------------------------------------------------------


@pytest.mark.parametrize("fence", ["```python", "```Python", "```python3", "```py", "```python title=solution.py", "``` python"])
def test_extract_blocks_fence_variants(fence: str) -> None:
    blocks = sv.extract_blocks(f"intro\n{fence}\nx = 1\n```\nafter")
    assert len(blocks) == 1 and blocks[0] == {"code": "x = 1\n", "unclosed": False}


def test_extract_blocks_dedents_and_drops_empty_blocks() -> None:
    """L2: indented blocks (under a list item) are dedented. L3: empty fences are no code."""
    blocks = sv.extract_blocks("- Implementation:\n    ```python\n    def add(a, b):\n        return a + b\n    ```\n")
    assert blocks == [{"code": "def add(a, b):\n    return a + b\n", "unclosed": False}]
    assert sv.extract_blocks("```python\n```\n```python\n   \n\n```") == []
    assert sv.extract_solution("```python\n\n```") is None


def test_extract_blocks_ignores_other_languages_and_untagged() -> None:
    assert sv.extract_blocks("```json\n{}\n```\n```\nraw\n```\n```pythonic\nz\n```") == []
    assert sv.extract_blocks("no code here") == []


def test_extract_blocks_crlf() -> None:
    blocks = sv.extract_blocks("```python\r\nx = 1\r\ny = 2\r\n```\r\n")
    assert blocks == [{"code": "x = 1\ny = 2\n", "unclosed": False}]


def test_extract_blocks_unclosed_final_block() -> None:
    blocks = sv.extract_blocks("```python\na = 1\n```\ntext\n```python\nb = 1\nc = 2")
    assert [b["unclosed"] for b in blocks] == [False, True]
    assert blocks[1]["code"] == "b = 1\nc = 2\n"


def test_choose_block_prefers_entry_symbol_then_longest() -> None:
    long_block = {"code": "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\ng = 7\nh = 8\n", "unclosed": False}
    entry = {"code": "def execute(call, registry):\n    pass\n", "unclosed": False}
    cls = {"code": "class WorkQueue:\n    pass\n", "unclosed": False}
    script = {"code": "if __name__ == '__main__':\n    pass\n", "unclosed": False}
    assert sv.choose_block([long_block, entry], "execute") is entry
    assert sv.choose_block([entry, long_block], "execute") is entry
    assert sv.choose_block([entry, cls], "WorkQueue") is cls
    assert sv.choose_block([cls, script], "__main__") is script
    assert sv.choose_block([entry, long_block], "lru_ttl_cache") is long_block  # no match: longest
    assert sv.choose_block([entry, long_block], None) is long_block
    assert sv.choose_block([], "execute") is None
    # Symbol must be a definition, not a mention.
    mention = {"code": "# execute is defined elsewhere\nx = 1\n", "unclosed": False}
    assert sv.choose_block([mention, long_block], "execute") is long_block


def test_extract_solution_wrapper() -> None:
    text = "```python\nx = 1\n```\n```python\ndef execute(c, r):\n    return {}\n```"
    assert sv.extract_solution(text, "execute").startswith("def execute")
    assert sv.extract_solution(text) == "def execute(c, r):\n    return {}\n"  # longest
    assert sv.extract_solution("nothing") is None


# --- static checks -----------------------------------------------------------


def test_static_checks_rules(tmp_path: Path) -> None:
    good = tmp_path / "solution.py"
    good.write_text("import threading\n\n\ndef execute(call, registry):\n    return {}\n\n\ndef _helper():\n    pass\n")
    res = {r["rule"]: r for r in sv.static_checks(good, "swe_10")}
    assert set(res) == {"stdlib_only", "forbidden_calls", "forbidden_modules", "max_loc", "public_functions"}
    assert all(r["passed"] for r in res.values()), res

    bad = tmp_path / "bad.py"
    bad.write_text(
        "import requests\nimport subprocess\n\n\ndef execute(call, registry):\n    eval('1')\n    return {}\n\n\ndef other():\n    pass\n"
        + "x = 1\n" * 250
    )
    res = {r["rule"]: r for r in sv.static_checks(bad, "swe_10")}
    assert not res["stdlib_only"]["passed"] and "requests" in res["stdlib_only"]["detail"]
    assert not res["forbidden_modules"]["passed"] and "subprocess" in res["forbidden_modules"]["detail"]
    assert not res["forbidden_calls"]["passed"] and "eval" in res["forbidden_calls"]["detail"]
    assert not res["max_loc"]["passed"]
    assert not res["public_functions"]["passed"] and "other" in res["public_functions"]["detail"]

    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n")
    assert all(not r["passed"] for r in sv.static_checks(broken, "swe_10"))
    assert sv.static_checks(good, "no_such_brief") == []


def test_forbidden_calls_are_bare_names_or_listed_attributes(tmp_path: Path) -> None:
    p = tmp_path / "solution.py"

    def forbidden_calls_pass(src: str, brief: str = "swe_10") -> bool:
        p.write_text(src)
        return [r for r in sv.static_checks(p, brief) if r["rule"] == "forbidden_calls"][0]["passed"]

    assert forbidden_calls_pass("import re\ndef execute(c, r):\n    return re.compile('x')\n")
    assert not forbidden_calls_pass("def execute(c, r):\n    return compile('x', '<s>', 'exec')\n")
    assert not forbidden_calls_pass("def execute(c, r):\n    return __import__('os')\n")
    assert not forbidden_calls_pass("import importlib\ndef execute(c, r):\n    return importlib.import_module('os')\n")
    # swe_09: os.system as an attribute call is caught; a function merely named system is not.
    assert not forbidden_calls_pass("import os\ndef run():\n    os.system('ls')\n", "swe_09")
    assert forbidden_calls_pass("def system():\n    pass\ndef run():\n    system()\n", "swe_09")


def test_forbidden_calls_resolve_import_aliases(tmp_path: Path) -> None:
    """M1: aliases and from-imports resolve to the dotted name; builtins is a forbidden module."""
    p = tmp_path / "solution.py"

    def rule(src: str, brief: str, name: str) -> dict:
        p.write_text(src)
        return [r for r in sv.static_checks(p, brief) if r["rule"] == name][0]

    assert not rule("import os as o\ndef run():\n    o.system('ls')\n", "swe_09", "forbidden_calls")["passed"]
    assert not rule("from os import system\ndef run():\n    system('ls')\n", "swe_09", "forbidden_calls")["passed"]
    assert not rule("from os import system as sh\ndef run():\n    sh('ls')\n", "swe_09", "forbidden_calls")["passed"]
    assert not rule("import importlib as il\ndef execute(c, r):\n    return il.import_module('os')\n", "swe_10", "forbidden_calls")["passed"]
    assert not rule("from importlib import import_module\ndef execute(c, r):\n    return import_module('os')\n", "swe_10", "forbidden_calls")["passed"]
    assert rule("import os\ndef run():\n    return os.getcwd()\n", "swe_09", "forbidden_calls")["passed"]
    for brief in ("swe_09", "swe_10"):
        r = rule("import builtins\ndef execute(c, r):\n    return getattr(builtins, 'ev' + 'al')\n", brief, "forbidden_modules")
        assert not r["passed"] and "builtins" in r["detail"]
    assert sv.import_aliases(ast_parse("import os.path as osp\nfrom a.b import c as d\nimport json")) == (
        {"osp": "os.path", "json": "json"}, {"d": "a.b.c"})


def ast_parse(src: str):
    import ast
    return ast.parse(src)


def test_subprocess_env_disables_plugin_autoload(toy: dict) -> None:
    """L4: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 reaches the pytest subprocess."""
    (toy["tests_dir"] / "toy_01_test.py").write_text(
        '"""contract"""\nimport os\n\n\ndef test_env():\n    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"\n')
    r = sv.run_tests(toy["code_dir"], "toy_01", tests_dir=toy["tests_dir"], timeout_s=60)
    assert r["all_passed"], r["stdout_tail"]


def test_public_methods_rule_for_work_queue(tmp_path: Path) -> None:
    p = tmp_path / "solution.py"
    p.write_text("class QueueFull(Exception):\n    pass\n\nclass WorkQueue:\n    def __init__(self):\n        pass\n"
                 "    def enqueue(self, t): pass\n    def claim(self, w): pass\n    def complete(self, t, w): pass\n"
                 "    def fail(self, t, w): pass\n    def _expire(self): pass\n")
    r = [r for r in sv.static_checks(p, "swe_08") if r["rule"] == "public_methods"][0]
    assert r["passed"], r
    p.write_text(p.read_text() + "    def stats(self): pass\n")
    r = [r for r in sv.static_checks(p, "swe_08") if r["rule"] == "public_methods"][0]
    assert not r["passed"] and "stats" in r["detail"]
    p.write_text("x = 1\n")
    r = [r for r in sv.static_checks(p, "swe_08") if r["rule"] == "public_methods"][0]
    assert not r["passed"] and "not defined" in r["detail"]


def test_count_loc_skips_blank_and_comment_lines() -> None:
    assert sv.count_loc("# c\n\nx = 1  # trailing\n   \n# another\ny = 2\n") == 2


def test_every_exec_brief_has_rules_symbol_and_test_file() -> None:
    for bid in sv.EXEC_BRIEFS:
        assert bid in sv.STATIC_RULES and bid in sv.ENTRY_SYMBOLS
        assert sv.test_file_for(bid).is_file()
        assert sv.interface_spec(bid).startswith("solution.py")
        for rule in sv.STATIC_RULES[bid]:
            assert rule["constraint"]
    assert "os.system" not in str(sv.STATIC_RULES["swe_09"][2]["modules"])


# --- exec briefs and config --------------------------------------------------


def test_make_exec_briefs_keeps_schema_and_ids(tmp_path: Path) -> None:
    out = tmp_path / "exec_briefs"
    written = sv.make_exec_briefs(sv.BRIEFS_SWE_DIR, out)
    assert [p.name for p in written] == [f"{b}.yaml" for b in sv.EXEC_BRIEFS]
    briefs = load_all_briefs(out)
    assert [b["id"] for b in briefs] == list(sv.EXEC_BRIEFS)
    for b in briefs:
        assert validate_brief(b) == []
        assert "Interface contract for the implementation (solution.py):" in b["objective"]
        original = yaml.safe_load((sv.BRIEFS_SWE_DIR / f"{b['id']}.yaml").read_text())
        for key in ("hard_constraints", "success_criteria", "plausible_directions", "banned_moves", "domain"):
            assert b[key] == original[key], (b["id"], key)
        assert b["objective"].startswith(original["objective"].strip())


def test_checked_in_exec_briefs_match_regeneration(tmp_path: Path) -> None:
    """drift_bench/briefs_swe_exec/ is generated; regenerating must give identical bytes."""
    out = tmp_path / "regen"
    sv.make_exec_briefs(sv.BRIEFS_SWE_DIR, out)
    checked_in = sorted(p.name for p in sv.BRIEFS_SWE_EXEC_DIR.glob("*.yaml"))
    assert checked_in == [f"{b}.yaml" for b in sv.EXEC_BRIEFS]
    for name in checked_in:
        assert (out / name).read_bytes() == (sv.BRIEFS_SWE_EXEC_DIR / name).read_bytes(), name
    assert len(load_all_briefs(sv.BRIEFS_SWE_EXEC_DIR)) == 4


def test_code_config_raises_subject_max_tokens_only(config_path: Path) -> None:
    base = yaml.safe_load(config_path.read_text())
    code = yaml.safe_load(sv.DEFAULT_CODE_CONFIG.read_text())
    assert [m["id"] for m in code["models"]["subjects"]] == [m["id"] for m in base["models"]["subjects"]]
    for m_base, m_code in zip(base["models"]["subjects"], code["models"]["subjects"]):
        assert m_base["max_tokens"] == 4096 and m_code["max_tokens"] == 8192
        assert {k: v for k, v in m_base.items() if k != "max_tokens"} == {k: v for k, v in m_code.items() if k != "max_tokens"}
    for key in ("judges", "auditor"):
        assert code["models"][key] == base["models"][key]
    for key in ("experiment", "rate_limits", "pricing", "budget", "paths"):
        assert code[key] == base[key]
    assert sv.subject_max_tokens(sv.DEFAULT_CODE_CONFIG, "gemini/gemini-3.1-pro-preview") == 8192
    assert sv.subject_max_tokens(config_path, "gemini/gemini-3.1-pro-preview") == 4096
    assert sv.subject_max_tokens(config_path, "nobody/nothing") is None


# --- transcript driver -------------------------------------------------------


def _transcript(run_id: str, brief_id: str, final_text: str, token_count: int = 100,
                finish_reason: str | None = "stop") -> dict:
    return {
        "metadata": {"run_id": run_id, "brief_id": brief_id, "model_id": "openai/gpt-5.4-mini",
                     "condition": "single_shot", "repetition": 0, "start_time": 0, "end_time": 1,
                     "total_input_tokens": 1, "total_output_tokens": 1, "total_cost_usd": 0.0, "error": None},
        "messages": [
            {"role": "system", "content": "s", "timestamp": 0},
            {"role": "user", "content": "u", "timestamp": 0},
            {"role": "assistant", "content": final_text, "timestamp": 0, "is_probe": False,
             "token_count": token_count, "finish_reason": finish_reason},
            {"role": "user", "content": "probe", "timestamp": 0, "is_probe": True},
            {"role": "assistant", "content": "```python\nWRONG = 1\n```", "timestamp": 0, "is_probe": True},
        ],
    }


def test_run_on_synthetic_transcripts(toy: dict, tmp_path: Path) -> None:
    """run(): one row per transcript, code from the final non-probe message, provenance written."""
    data_root = tmp_path / "data"
    tdir = data_root / "camera_ready" / "e6_swe_verify" / "transcripts"
    tdir.mkdir(parents=True)
    two_blocks = "Design...\n```python\nx = 1\n```\nthen\n```python\ndef add(a, b):\n    return a + b\n```\n"
    with open(tdir / "aaa_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("aaa", "toy_01", two_blocks)) + "\n")
    with open(tdir / "bbb_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("bbb", "toy_01", "no code at all", token_count=8192)) + "\n")
    with open(tdir / "ccc_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("ccc", "toy_01", "```python\ndef add(a, b):\n    return a + b", token_count=8192)) + "\n")
    with open(tdir / "ddd_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("ddd", "toy_01", "```python\nx = 1\n```", token_count=50, finish_reason="length")) + "\n")
    with open(tdir / "eee_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("eee", "toy_01", "```python\nimport os; os.system('touch NOPE')\n```")) + "\n")
    out_dir = tmp_path / "out"
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("# refused after reading\neee shells out via os.system\n")

    frames = sv.run(data_root, out_dir, tests_dir=toy["tests_dir"], timeout_s=60,
                    exclude_run_ids=sv.load_exclude_file(exclude))
    df = frames["swe_verify_results"].set_index("run_id")
    assert list(df.index) == ["aaa", "bbb", "ccc", "ddd", "eee"]
    # L1: finish_reason column; truncated by finish_reason even under the cap.
    assert df.loc["ddd", "finish_reason"] == "length" and df.loc["ddd", "truncated"] and df.loc["aaa", "finish_reason"] == "stop"
    # Refused row: present, flagged, never extracted or executed.
    assert df.loc["eee", "refused"] and df.loc["eee", "refused_reason"] == "shells out via os.system"
    assert df.loc["eee", "total"] == 0 and df.loc["eee", "static_checks"] == 0
    assert not (data_root / "camera_ready" / "e6_swe_verify" / "code" / "eee").exists()
    assert not (tmp_path / "NOPE").exists() and not (data_root / "NOPE").exists()
    assert not df.loc["aaa", "refused"]
    assert sv.load_exclude_file(None) == {} and sv.load_exclude_file(exclude) == {"eee": "shells out via os.system"}
    assert df.loc["aaa", "passed"] == 1 and df.loc["aaa", "failed"] == 1 and not df.loc["aaa", "no_code"]
    assert df.loc["aaa", "n_blocks"] == 2 and not df.loc["aaa", "unclosed_block"] and not df.loc["aaa", "truncated"]
    assert df.loc["bbb", "no_code"] and df.loc["bbb", "total"] == 0 and df.loc["bbb", "missing"].endswith("solution.py")
    assert df.loc["bbb", "n_blocks"] == 0 and df.loc["bbb", "truncated"]
    assert df.loc["ccc", "unclosed_block"] and df.loc["ccc", "truncated"] and df.loc["ccc", "passed"] == 1
    # The longest block was chosen (toy_01 has no entry symbol) and the probe's block was not used.
    kept = data_root / "camera_ready" / "e6_swe_verify" / "code" / "aaa" / "solution.py"
    assert kept.read_text() == "def add(a, b):\n    return a + b\n"
    assert not (data_root / "camera_ready" / "e6_swe_verify" / "code" / "bbb").exists()

    csv = out_dir / "swe_verify_results.csv"
    prov = json.loads((out_dir / "swe_verify_results.csv.provenance.json").read_text())
    assert csv.is_file() and prov["rows_in"] == 5 and prov["rows_out"] == 5
    assert any("refused rows (1 run ids given)" in f for f in prov["filters"])
    assert any("entry symbol" in f for f in prov["filters"])
    assert any("truncated" in f for f in prov["filters"])
    assert any("no sandbox" in f for f in prov["filters"])
    for col in ("n_blocks", "unclosed_block", "finish_reason", "truncated", "refused", "refused_reason", "static_failed"):
        assert col in df.columns


def test_run_with_no_transcripts_writes_empty_csv(tmp_path: Path) -> None:
    frames = sv.run(tmp_path / "data", tmp_path / "out")
    assert frames["swe_verify_results"].empty
    prov = json.loads((tmp_path / "out" / "swe_verify_results.csv.provenance.json").read_text())
    assert prov["rows_in"] == 0 and prov["rows_out"] == 0


def test_extract_all_writes_code_without_running(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    tdir = data_root / "camera_ready" / "e6_swe_verify" / "transcripts"
    tdir.mkdir(parents=True)
    with open(tdir / "aaa_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("aaa", "toy_01", "```python\nimport os; os.system('touch SHOULD_NOT_RUN')\n```")) + "\n")
    with open(tdir / "bbb_toy_01_single_shot.jsonl", "w") as fh:
        fh.write(json.dumps(_transcript("bbb", "toy_01", "no code")) + "\n")
    written = sv.extract_all(data_root)
    assert [p.parent.name for p in written] == ["aaa"]
    assert written[0].read_text() == "import os; os.system('touch SHOULD_NOT_RUN')\n"
    assert not (tmp_path / "SHOULD_NOT_RUN").exists() and not (data_root / "SHOULD_NOT_RUN").exists()
    assert not (tmp_path / "out").exists()
    assert sv.extract_all(tmp_path / "nowhere") == []
