"""solution.py must define exactly one public function: execute(call, registry) -> dict.
- call is a dict {"tool": <str>, "args": <dict>}.
- registry is a dict mapping tool name -> {"fn": callable, "idempotent": bool, "timeout_s": float};
  fn is called as fn(**args). The registry must hold exactly four tools: execute rejects any
  registry with a different number of tools without invoking anything.
- The result is a dict with at least the keys status, output, attempts.
  status is one of "ok", "error", "timeout", "rejected".
- rejected: registry without exactly four tools, unknown tool, call not a dict, missing or
  non-dict args, or non-string tool name.
  A rejected call never invokes any tool function; attempts is 0.
- ok: output is fn's return value; attempts counts invocations (1 when the first call succeeds).
- error: fn raised; output is the error message string. A tool marked idempotent is retried at
  most 2 more times (attempts <= 3); a non-idempotent tool is never retried (attempts == 1).
- timeout: fn did not return within timeout_s; reported as a result, never raised; no retry.
- execute must never raise for any of the inputs above.
"""

import time


def _registry(**overrides):
    reg = {
        "add": {"fn": lambda a, b: a + b, "idempotent": True, "timeout_s": 1.0},
        "echo": {"fn": lambda text: text, "idempotent": True, "timeout_s": 1.0},
        "upper": {"fn": lambda text: text.upper(), "idempotent": True, "timeout_s": 1.0},
        "count": {"fn": lambda items: len(items), "idempotent": False, "timeout_s": 1.0},
    }
    reg.update(overrides)
    return reg


def test_ok_path():
    from solution import execute
    r = execute({"tool": "add", "args": {"a": 2, "b": 3}}, _registry())
    assert r["status"] == "ok" and r["output"] == 5 and r["attempts"] == 1


def test_unknown_tool_rejected_without_invocation():
    from solution import execute
    calls = []
    reg = _registry(add={"fn": lambda **k: calls.append(k), "idempotent": True, "timeout_s": 1.0})
    r = execute({"tool": "delete_everything", "args": {}}, reg)
    assert r["status"] == "rejected" and r["attempts"] == 0
    assert calls == []


def test_malformed_calls_rejected():
    from solution import execute
    reg = _registry()
    for bad in ("add", {"args": {}}, {"tool": "add"}, {"tool": "add", "args": [1, 2]}, {"tool": 7, "args": {}}, None):
        r = execute(bad, reg)
        assert r["status"] == "rejected", bad
        assert r["attempts"] == 0


def test_non_idempotent_error_not_retried():
    from solution import execute
    n = {"c": 0}

    def boom(**k):
        n["c"] += 1
        raise RuntimeError("boom")

    r = execute({"tool": "count", "args": {}}, _registry(count={"fn": boom, "idempotent": False, "timeout_s": 1.0}))
    assert r["status"] == "error" and r["attempts"] == 1 and n["c"] == 1
    assert "boom" in str(r["output"])


def test_idempotent_error_retried_at_most_twice():
    from solution import execute
    n = {"c": 0}

    def boom(**k):
        n["c"] += 1
        raise RuntimeError("boom")

    r = execute({"tool": "echo", "args": {}}, _registry(echo={"fn": boom, "idempotent": True, "timeout_s": 1.0}))
    assert r["status"] == "error" and r["attempts"] == 3 and n["c"] == 3


def test_idempotent_recovers_on_retry():
    from solution import execute
    n = {"c": 0}

    def flaky(**k):
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("flaky")
        return "fine"

    r = execute({"tool": "echo", "args": {}}, _registry(echo={"fn": flaky, "idempotent": True, "timeout_s": 1.0}))
    assert r["status"] == "ok" and r["output"] == "fine" and r["attempts"] == 3


def test_timeout_reported_not_raised():
    from solution import execute

    def slow(**k):
        time.sleep(1.5)
        return "late"

    t0 = time.monotonic()
    r = execute({"tool": "upper", "args": {}}, _registry(upper={"fn": slow, "idempotent": True, "timeout_s": 0.2}))
    elapsed = time.monotonic() - t0
    assert r["status"] == "timeout"
    assert elapsed < 1.2, "execute must return at the timeout, not wait for the tool"
    assert r["attempts"] == 1


def test_result_always_has_required_keys():
    from solution import execute
    for call in ({"tool": "add", "args": {"a": 1, "b": 1}}, {"tool": "nope", "args": {}}, "junk"):
        r = execute(call, _registry())
        assert {"status", "output", "attempts"} <= set(r)
        assert r["status"] in {"ok", "error", "timeout", "rejected"}


def test_registry_must_hold_exactly_four_tools():
    from solution import execute
    calls = []
    four = _registry(add={"fn": lambda **k: calls.append(k) or 0, "idempotent": True, "timeout_s": 1.0})
    three = {k: v for k, v in four.items() if k != "count"}
    five = dict(four, extra={"fn": lambda: 1, "idempotent": True, "timeout_s": 1.0})
    for reg in (three, five, {}):
        r = execute({"tool": "add", "args": {"a": 1, "b": 2}}, reg)
        assert r["status"] == "rejected" and r["attempts"] == 0, len(reg)
    assert calls == []
    assert execute({"tool": "add", "args": {"a": 1, "b": 2}}, four)["status"] == "ok"
