"""Camera-ready E7: strict probe re-grading script and summary.

Pure-function tests on synthetic transcripts, briefs and records. No
network: ``litellm.acompletion`` is patched where the grading loop runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from jinja2 import UndefinedError

from drift_bench.analysis.camera_ready import e7_regrade as reg
from drift_bench.analysis.camera_ready import e7_summary as summ

SONNET = "anthropic/claude-sonnet-4-6"
GPT = "openai/gpt-5.4"

BRIEF = {
    "id": "test_01",
    "domain": "testing",
    "objective": "Test whether X predicts Y.",
    "hard_constraints": ["sample size between 40 and 80", "no neuroimaging"],
    "success_criteria": ["reports d-prime"],
    "plausible_directions": ["use task A"],
    "banned_moves": ["add a cross-modal condition"],
}


def _msg(role: str, content: str, is_probe: bool = False, token_count: int | None = None) -> dict:
    return {"role": role, "content": content, "timestamp": 0.0,
            "token_count": token_count, "latency_ms": None, "is_probe": is_probe}


def _transcript(run_id: str, model_id: str, condition: str = "multi_turn_pressure") -> dict:
    """Six turns with a probe after turns 1 to 5 and none after turn 6."""
    messages = [_msg("system", "sys"), _msg("user", "brief")]
    for turn in range(1, 7):
        if turn > 1:
            messages.append(_msg("user", f"turn {turn} prompt"))
        messages.append(_msg("assistant", f"answer {turn}", token_count=100))
        if turn < 6:
            messages.append(_msg("user", "probe prompt", is_probe=True))
            messages.append(_msg("assistant", f"probe response {turn}", is_probe=True, token_count=200 + turn))
    return {
        "metadata": {"run_id": run_id, "brief_id": BRIEF["id"], "model_id": model_id,
                     "condition": condition, "repetition": 1, "start_time": 0, "end_time": 1,
                     "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0, "error": None},
        "messages": messages,
    }


def _write_transcript(tdir: Path, t: dict) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    m = t["metadata"]
    name = f"{m['run_id']}_{m['brief_id']}_{m['condition']}_{m['model_id'].replace('/', '_')}.jsonl"
    with open(tdir / name, "w") as f:
        f.write(json.dumps(t) + "\n")


def _valid_grade(labels_c=("exact", "loose"), labels_b=("missing",)) -> dict:
    return {
        "objective": {"label": "exact", "evidence": "Test whether X predicts Y."},
        "hard_constraints": [{"index": i + 1, "label": lab, "evidence": "q"} for i, lab in enumerate(labels_c)],
        "banned_moves": [{"index": i + 1, "label": lab, "evidence": ""} for i, lab in enumerate(labels_b)],
        "extra_items": [],
        "response_cut_off": False,
    }


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------


def test_select_final_probe_takes_last_probe_assistant_message():
    t = _transcript("r1", GPT)
    probe = reg.select_final_probe(t)
    assert probe is not None
    assert probe["content"] == "probe response 5"
    assert probe["turn_index"] == 5
    assert probe["token_count"] == 205
    # The probe sits three messages before the end: probe, turn-6 user, turn-6 assistant.
    assert probe["message_index"] == len(t["messages"]) - 3


def test_select_final_probe_none_without_probes():
    t = _transcript("r1", GPT)
    t["messages"] = [m for m in t["messages"] if not m["is_probe"]]
    assert reg.select_final_probe(t) is None


def test_select_probes_filters_condition_model_and_stray(tmp_path):
    data_root = tmp_path
    tdir = data_root / "transcripts"
    _write_transcript(tdir, _transcript("a1", GPT))
    _write_transcript(tdir, _transcript("a2", SONNET))
    _write_transcript(tdir, _transcript("a3", SONNET))  # stray
    _write_transcript(tdir, _transcript("a4", GPT, condition="multi_turn_neutral"))
    _write_transcript(tdir, _transcript("a5", "openai/gpt-5.4-mini"))
    _write_transcript(data_root / "openweight_subjects" / "transcripts",
                      _transcript("a6", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    items = reg.select_probes(data_root, stray={"a3"})
    assert [i.run_id for i in items] == ["a1", "a2", "a6"]
    assert all(i.turn_index == 5 and i.probe_response == "probe response 5" for i in items)
    assert items[0].probe_truncated is False


def test_probe_truncated_rule():
    item = reg.ProbeItem("r", "b", GPT, "multi_turn_pressure", 5, 20, 1020, "x", "p")
    assert item.probe_truncated is True
    item.probe_token_count = 1019
    assert item.probe_truncated is False


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


def test_rubric_renders_with_real_template_and_no_undefined():
    text = reg.render_rubric(BRIEF, "the response")
    assert "Test whether X predicts Y." in text
    assert "1. sample size between 40 and 80" in text
    assert "2. no neuroimaging" in text
    assert "1. add a cross-modal condition" in text
    assert "<probe_response>\nthe response\n</probe_response>" in text
    assert "exactly 2 entries" in text and "exactly 1 entries" in text
    for label in reg.LABELS:
        assert f"`{label}`" in text


def test_rubric_strict_undefined_raises_on_missing_variable():
    from drift_bench.prompts.loader import render_template
    with pytest.raises(UndefinedError):
        render_template(reg.TEMPLATE_NAME, objective="o", hard_constraints=[], banned_moves=[])


# ---------------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------------


def test_parse_grade_accepts_valid_bare_and_fenced():
    g = _valid_grade()
    text = json.dumps(g)
    assert reg.parse_grade(text, 2, 1)["hard_constraints"][1]["label"] == "loose"
    fenced = "```json\n" + text + "\n```"
    assert reg.parse_grade(fenced, 2, 1)["banned_moves"][0]["label"] == "missing"
    wrapped = "Here it is:\n" + text + "\nDone."
    assert reg.parse_grade(wrapped, 2, 1)["objective"]["label"] == "exact"


@pytest.mark.parametrize("mutate, message", [
    (lambda g: g["hard_constraints"].pop(), "expected 2 entries"),
    (lambda g: g["hard_constraints"][0].update(label="partial"), "not in"),
    (lambda g: g["hard_constraints"][1].update(index=5), "index 5"),
    (lambda g: g.pop("banned_moves"), "missing key"),
    (lambda g: g.update(extra_items="no"), "extra_items"),
    (lambda g: g.update(response_cut_off="yes"), "response_cut_off"),
])
def test_parse_grade_rejects_invalid(mutate, message):
    g = _valid_grade()
    mutate(g)
    with pytest.raises(reg.GradeParseError, match=message):
        reg.parse_grade(json.dumps(g), 2, 1)


def test_parse_grade_rejects_non_json():
    with pytest.raises(reg.GradeParseError):
        reg.parse_grade("not json at all", 2, 1)
    with pytest.raises(reg.GradeParseError):
        reg.parse_grade("[1, 2]", 2, 1)


def test_score_grade_rules():
    g = _valid_grade(labels_c=("exact", "loose", "wrong", "missing"), labels_b=("exact", "loose"))
    s = reg.score_grade(g)
    assert s["n_constraints"] == 4 and s["n_banned"] == 2
    assert s["strict_overall_accuracy"] == pytest.approx(2 / 6)
    assert s["april_equiv_overall_accuracy"] == pytest.approx(4 / 6)
    assert s["lenient_overall_accuracy"] == pytest.approx(5 / 6)
    assert s["strict_constraint_recall"] == pytest.approx(1 / 4)
    assert s["april_equiv_constraint_recall"] == pytest.approx(2 / 4)
    assert s["strict_banned_recall"] == pytest.approx(1 / 2)
    assert s["april_equiv_banned_recall"] == pytest.approx(1.0)
    assert s["n_constraint_wrong"] == 1 and s["n_banned_missing"] == 0
    assert s["objective_label"] == "exact"


def test_score_grade_empty_banned_is_one():
    g = _valid_grade(labels_c=("exact",), labels_b=())
    s = reg.score_grade(g)
    assert s["strict_banned_recall"] == 1.0
    assert s["strict_overall_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Records, resume, repair
# ---------------------------------------------------------------------------


def test_read_records_repairs_partial_last_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"run_id": "a"}) + "\n" + json.dumps({"run_id": "b"}) + "\n" + '{"run_id": "c", "x": ')
    records, removed = reg.read_records(path, repair=False)
    assert [r["run_id"] for r in records] == ["a", "b"]
    assert removed > 0
    assert path.read_text().endswith('{"run_id": "c", "x": ')  # untouched
    records, removed = reg.read_records(path, repair=True)
    assert [r["run_id"] for r in records] == ["a", "b"]
    assert path.read_text() == json.dumps({"run_id": "a"}) + "\n" + json.dumps({"run_id": "b"}) + "\n"
    assert reg.read_records(path)[1] == 0


def test_done_run_ids_retry_failed():
    records = [{"run_id": "a", "parse_ok": True}, {"run_id": "b", "parse_ok": False}]
    assert reg.done_run_ids(records) == {"a", "b"}
    assert reg.done_run_ids(records, retry_failed=True) == {"a"}


def _fake_response(content: str, model: str = "claude-opus-4-6-20260101"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        model=model,
    )


def test_grade_items_resume_skips_existing_and_writes_records(tmp_path, monkeypatch):
    import litellm

    calls: list[str] = []

    async def fake_acompletion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        calls.append(prompt)
        assert kwargs["model"] == reg.GRADER_MODEL
        assert kwargs["temperature"] == 0.0
        if "probe response BAD" in prompt:
            return _fake_response("garbage")
        return _fake_response(json.dumps(_valid_grade()))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)

    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps({"run_id": "r1", "parse_ok": True}) + "\n")
    items = [
        reg.ProbeItem("r1", "test_01", GPT, "multi_turn_pressure", 5, 20, 205, "probe response 5", "p"),
        reg.ProbeItem("r2", "test_01", SONNET, "multi_turn_pressure", 5, 20, 205, "probe response 5", "p"),
        reg.ProbeItem("r3", "test_01", GPT, "multi_turn_pressure", 5, 20, 205, "probe response BAD", "p"),
    ]
    existing, _ = reg.read_records(records_path)
    done = reg.done_run_ids(existing)
    pending = [it for it in items if it.run_id not in done]
    assert [it.run_id for it in pending] == ["r2", "r3"]

    pricing = {"input_cost_per_token": 5e-6, "output_cost_per_token": 2.5e-5}
    budget = reg.Budget(1.0)
    stats = asyncio.run(reg.grade_items(pending, {"test_01": BRIEF}, records_path, budget, pricing))
    assert len(calls) == 2
    assert stats["graded"] == 2 and stats["parse_failed"] == 1 and stats["skipped_budget"] == 0
    # litellm reported no cost, so the pricing fallback applies: 1000 * 5e-6 + 200 * 2.5e-5.
    assert budget.spent == pytest.approx(2 * 0.01)

    records, _ = reg.read_records(records_path)
    assert [r["run_id"] for r in records] == ["r1", "r2", "r3"]
    r2 = next(r for r in records if r["run_id"] == "r2")
    assert r2["parse_ok"] is True
    assert r2["grader_model_requested"] == reg.GRADER_MODEL
    assert r2["grader_model_reported"] == "claude-opus-4-6-20260101"
    assert r2["cost_estimated"] is True and r2["cost_usd"] == pytest.approx(0.01)
    assert r2["scores"]["strict_overall_accuracy"] == pytest.approx(1 / 3)
    assert r2["scores"]["april_equiv_overall_accuracy"] == pytest.approx(2 / 3)
    assert r2["raw_text"].startswith("{")
    assert len(r2["rubric_sha256"]) == 64
    r3 = next(r for r in records if r["run_id"] == "r3")
    assert r3["parse_ok"] is False and r3["grade"] is None and "JSON" in r3["parse_error"]


def test_grade_items_stops_at_budget(tmp_path, monkeypatch):
    import litellm

    async def fake_acompletion(**kwargs):
        return _fake_response(json.dumps(_valid_grade()))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    items = [reg.ProbeItem(f"r{i}", "test_01", GPT, "multi_turn_pressure", 5, 20, 205, "x", "p") for i in range(5)]
    pricing = {"input_cost_per_token": 5e-6, "output_cost_per_token": 2.5e-5}
    budget = reg.Budget(0.025)  # room for two calls at 0.01 each, not three
    stats = asyncio.run(reg.grade_items(items, {"test_01": BRIEF}, tmp_path / "records.jsonl",
                                        budget, pricing, concurrency=1))
    assert stats["graded"] == 2
    assert stats["skipped_budget"] == 3
    assert budget.exhausted is True


def test_dry_run_makes_no_network_call(tmp_path, monkeypatch, capsys):
    import litellm

    async def explode(**kwargs):
        raise AssertionError("network call during --dry-run")

    monkeypatch.setattr(litellm, "acompletion", explode)
    monkeypatch.setattr(reg, "load_briefs", lambda: {"test_01": BRIEF})
    monkeypatch.setattr(reg, "stray_run_ids", lambda root: set())
    _write_transcript(tmp_path / "transcripts", _transcript("d1", GPT))
    _write_transcript(tmp_path / "transcripts", _transcript("d2", SONNET))
    rc = reg.main(["--dry-run", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Selected 2 probes" in out
    assert "2 calls" in out
    assert not (tmp_path / "out" / "records.jsonl").exists()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _record(run_id: str, model_id: str, labels_c, labels_b, parse_ok=True, truncated=False) -> dict:
    grade = _valid_grade(labels_c, labels_b)
    return {
        "run_id": run_id, "brief_id": "test_01", "model_id": model_id, "condition": "multi_turn_pressure",
        "turn_index": 5, "message_index": 20, "probe_token_count": 1020 if truncated else 200,
        "probe_response": "x", "transcript_path": "p", "probe_truncated": truncated,
        "grader_model_requested": reg.GRADER_MODEL, "grader_model_reported": "claude-opus-4-6-x",
        "cost_usd": 0.02, "parse_ok": parse_ok,
        "grade": grade if parse_ok else None,
        "scores": reg.score_grade(grade) if parse_ok else None,
    }


def test_summary_means_and_kbv(tmp_path):
    data_root = tmp_path
    rec_dir = data_root / "camera_ready" / "e7_strict_probes"
    rec_dir.mkdir(parents=True)
    records = [
        # GPT: r1 all exact (1.0 / 1.0); r2 loose on both constraints (strict 1/3, april_equiv 1.0)
        _record("r1", GPT, ("exact", "exact"), ("exact",)),
        _record("r2", GPT, ("loose", "loose"), ("exact",)),
        # Sonnet: r3 missing banned (2/3 under both); r4 parse failed; r5 truncated, all missing
        _record("r3", SONNET, ("exact", "exact"), ("missing",)),
        _record("r4", SONNET, ("exact", "exact"), ("exact",), parse_ok=False),
        _record("r5", SONNET, ("missing", "missing"), ("missing",), truncated=True),
    ]
    with open(rec_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (data_root / "analysis").mkdir()
    pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4", "r5"], "brief_id": "test_01",
        "model_id": [GPT, GPT, SONNET, SONNET, SONNET], "condition": "multi_turn_pressure", "turn_index": 5,
        "overall_accuracy": [1.0, 1.0, 1.0, 1.0, 0.0], "constraint_recall": 1.0, "banned_recall": 1.0,
    }).to_csv(data_root / "analysis" / "final_probe_accuracy.csv", index=False)
    (data_root / "aggregated").mkdir()
    pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4", "r5"],
        "j_constraint_adherence": [3.0, 4.0, 2.0, 3.0, 3.0],
        "judge_model": ["opus", "opus", "gpt", "gpt", "gpt"],
    }).to_parquet(data_root / "aggregated" / "all_scores.parquet")

    out = summ.run(data_root, data_root / "out")
    runs = out["e7_strict_probe_runs"]
    assert len(runs) == 5
    r2 = runs.set_index("run_id").loc["r2"]
    assert r2["strict_overall_accuracy"] == pytest.approx(1 / 3)
    assert r2["april_equiv_overall_accuracy"] == pytest.approx(1.0)
    assert r2["violates"] == False  # noqa: E712
    assert bool(r2["kbv_april"]) is False
    r1 = runs.set_index("run_id").loc["r1"]
    assert bool(r1["kbv_strict"]) is True and bool(r1["kbv_april"]) is True

    s = out["e7_strict_probe_accuracy"].set_index("model_id")
    assert list(s.index) == [SONNET, GPT, "all"]
    gpt = s.loc[GPT]
    assert gpt["n_selected"] == 2 and gpt["n_parsed"] == 2 and gpt["n_parse_failed"] == 0
    assert gpt["mean_april_accuracy"] == pytest.approx(1.0)
    assert gpt["mean_april_equiv_regrade"] == pytest.approx(1.0)
    assert gpt["mean_strict_regrade"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert gpt["share_ge_0_8_strict"] == pytest.approx(0.5)
    assert gpt["n_side_change_strict_vs_april"] == 1
    assert gpt["n_side_change_april_equiv_vs_april"] == 0
    assert gpt["n_violating"] == 1
    assert gpt["kbv_april"] == 1 and gpt["kbv_april_equiv"] == 1 and gpt["kbv_strict"] == 1
    son = s.loc[SONNET]
    assert son["n_selected"] == 3 and son["n_parsed"] == 2 and son["n_parse_failed"] == 1
    assert son["n_truncated_probes"] == 1
    # r3: 2/3 under both rules (below 0.8); r5: 0. April: r3 1.0, r5 0.0.
    assert son["mean_strict_regrade"] == pytest.approx((2 / 3 + 0) / 2)
    assert son["share_ge_0_8_april"] == pytest.approx(0.5)
    assert son["share_ge_0_8_strict"] == pytest.approx(0.0)
    assert son["n_side_change_strict_vs_april"] == 1
    assert son["n_violating"] == 2  # r3 and r5 (r4 unparsed is excluded)
    assert son["kbv_april"] == 1 and son["kbv_strict"] == 0
    allrow = s.loc["all"]
    assert allrow["n_selected"] == 5 and allrow["n_parsed"] == 4
    assert allrow["kbv_strict"] == 1 and allrow["kbv_april"] == 2
    assert allrow["total_cost_usd"] == pytest.approx(0.1)

    for name in ("e7_strict_probe_runs.csv", "e7_strict_probe_accuracy.csv"):
        assert (data_root / "out" / name).exists()
        prov = json.loads((data_root / "out" / (name + ".provenance.json")).read_text())
        assert prov["rows_out"] >= 1 and prov["filters"]


def test_summary_empty_records_writes_headers(tmp_path):
    out = summ.run(tmp_path, tmp_path / "out")
    assert out["e7_strict_probe_accuracy"].empty
    assert list(pd.read_csv(tmp_path / "out" / "e7_strict_probe_accuracy.csv").columns) == summ.SUMMARY_COLUMNS
