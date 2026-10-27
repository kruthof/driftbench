"""Camera-ready X3: probe replay runner and the E3 accuracy script.

Pure-function tests. ``_call_llm`` is replaced by a recorder, so no network
and no API keys. The main check runs the real ``MultiTurnRunner.run`` with
the fake call, captures every probe call it makes, and asserts that
``reprobe.build_probe_context`` rebuilds the same message list from the
saved transcript.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from drift_bench.analysis.camera_ready import reprobe_accuracy
from drift_bench.analysis.probes import extract_probes
from drift_bench.prompts.loader import get_restatement_probe
from drift_bench.runners import reprobe
from drift_bench.runners.multi_turn import MultiTurnRunner
from drift_bench.runners.reprobe import (
    ProbeSite,
    ReprobeRunner,
    build_probe_context,
    find_probe_sites,
    infer_finish_reason,
    is_truncated,
    load_transcript,
    replay_transcripts,
    select_sites,
)

GEMINI = "gemini/gemini-3.1-pro-preview"


class CallRecorder:
    """Stands in for BaseRunner._call_llm. Records every call."""

    def __init__(self, probe_tokens: int = 1020, finish_reason: str | None = "stop", cost: float = 0.01):
        self.calls: list[dict] = []
        self.probe_tokens = probe_tokens
        self.finish_reason = finish_reason
        self.cost = cost

    def install(self, monkeypatch: pytest.MonkeyPatch, cls=MultiTurnRunner) -> None:
        recorder = self

        async def fake_call_llm(self_, model_id, messages, temperature=0.7, max_tokens=4096,
                                response_format=None):
            call = {
                "model_id": model_id,
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            recorder.calls.append(call)
            n = len(recorder.calls)
            is_probe = messages[-1]["content"] == get_restatement_probe()
            tokens = recorder.probe_tokens if is_probe else 300
            return {
                "content": f"{'probe' if is_probe else 'answer'} {n}",
                "input_tokens": 100 * n,
                "output_tokens": tokens,
                "cost": recorder.cost,
                "latency_ms": 1.0,
                "finish_reason": recorder.finish_reason,
                "response_model": "fake-model-001",
            }

        monkeypatch.setattr(cls, "_call_llm", fake_call_llm)

    def probe_calls(self) -> list[dict]:
        return [c for c in self.calls if c["messages"][-1]["content"] == get_restatement_probe()]


def _run_original(config_path: Path, brief: dict, condition: str, out_dir: Path,
                  recorder: CallRecorder, model_id: str = GEMINI) -> Path:
    """Run the real MultiTurnRunner with the fake call and return the saved transcript path."""
    runner = MultiTurnRunner(config_path)
    transcript = asyncio.run(runner.run(
        brief=brief, model_id=model_id, condition=condition, repetition=0, output_dir=out_dir,
    ))
    return out_dir / f"{transcript.metadata.run_id}_{brief['id']}_{condition}_{model_id.replace('/', '_')}.jsonl"


# --- context rebuild ---------------------------------------------------------

@pytest.mark.parametrize("condition", ["multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"])
def test_rebuild_matches_original_probe_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict, condition: str,
) -> None:
    """build_probe_context must return byte-identical messages to what multi_turn.py sent."""
    recorder = CallRecorder()
    recorder.install(monkeypatch)
    path = _run_original(config_path, sample_brief, condition, tmp_path, recorder)

    original_probe_calls = recorder.probe_calls()
    expected_probes = {"multi_turn_neutral": 5, "multi_turn_pressure": 5, "checkpointed_pressure": 5}
    assert len(original_probe_calls) == expected_probes[condition]

    transcript = load_transcript(path)
    sites = find_probe_sites(transcript)
    assert len(sites) == len(original_probe_calls)

    for site, call in zip(sites, original_probe_calls):
        rebuilt = build_probe_context(transcript["messages"], site)
        assert rebuilt == call["messages"]
        # The original call used temperature 0.0 and the config cap (1024).
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 1024
        assert call["model_id"] == GEMINI

    # The rebuilt context never contains a probe answer and starts with the stored system prompt.
    for site in sites:
        rebuilt = build_probe_context(transcript["messages"], site)
        assert rebuilt[0]["role"] == "system"
        assert rebuilt[0]["content"] == transcript["messages"][0]["content"]
        assert all(not m["content"].startswith("probe ") for m in rebuilt)
        assert rebuilt[-1] == {"role": "user", "content": get_restatement_probe()}
        assert set(rebuilt[-1].keys()) == {"role", "content"}


def test_rebuild_matches_build_messages_list_on_fixture(
    config_path: Path, sample_transcript,
) -> None:
    """On the shared fixture the rebuild equals BaseRunner._build_messages_list + probe prompt."""
    runner = MultiTurnRunner(config_path)
    transcript = sample_transcript.to_dict()
    sites = find_probe_sites(transcript)
    assert len(sites) == 1
    site = sites[0]
    expected = runner._build_messages_list(sample_transcript.messages[:site.probe_user_index]) + [
        {"role": "user", "content": "restate constraints"}
    ]
    assert build_probe_context(transcript["messages"], site) == expected


def test_turn_index_matches_extract_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    """turn_index must equal the value analysis/probes.py assigns (join key to the April CSV)."""
    recorder = CallRecorder()
    recorder.install(monkeypatch)
    path = _run_original(config_path, sample_brief, "checkpointed_pressure", tmp_path, recorder)
    transcript = load_transcript(path)
    ours = [s.turn_index for s in find_probe_sites(transcript)]
    theirs = [p["turn_index"] for p in extract_probes(transcript)]
    assert ours == theirs
    # Checkpoint turns count as user turns: the final probe sits at turn 7 (matches the CSV).
    assert ours[-1] == 7
    assert select_sites(find_probe_sites(transcript), "final")[0].turn_index == 7


def test_find_probe_sites_skips_unanswered_probe() -> None:
    transcript = {
        "metadata": {"run_id": "r"},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "token_count": 10},
            {"role": "user", "content": "p", "is_probe": True},
            # no assistant probe answer: the run crashed here
        ],
    }
    assert find_probe_sites(transcript) == []


# --- flags and selection -----------------------------------------------------

def test_truncation_flag() -> None:
    assert is_truncated(1020) is True
    assert is_truncated(1024) is True
    assert is_truncated(1019) is False
    assert is_truncated(None) is False


def test_select_sites_final_and_all() -> None:
    sites = [ProbeSite(1, 3, 4, 500, False), ProbeSite(2, 7, 8, 1020, True, is_final=True)]
    assert select_sites(sites, "final") == [sites[1]]
    assert select_sites(sites, "all") == sites
    with pytest.raises(ValueError):
        select_sites(sites, "last")


def test_infer_finish_reason_fallback() -> None:
    assert infer_finish_reason(4092, 4096) == "length"
    assert infer_finish_reason(4096, 4096) == "length"
    assert infer_finish_reason(4091, 4096) == "stop"
    assert infer_finish_reason(None, 4096) == "stop"


def test_multi_turn_probe_cap_comes_from_config(config_path: Path) -> None:
    """config.yaml documents 1024 so the main runner's behaviour is unchanged."""
    runner = MultiTurnRunner(config_path)
    assert runner._get_probe_max_tokens() == 1024
    runner.config["experiment"]["probe_max_tokens"] = 4096
    assert runner._get_probe_max_tokens() == 4096


# --- replay loop -------------------------------------------------------------

def _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs: int = 2) -> Path:
    """Write n multi-turn transcripts plus one single-shot transcript with fake calls."""
    tdir = tmp_path / "transcripts"
    recorder = CallRecorder(probe_tokens=1020)
    recorder.install(monkeypatch)
    for _ in range(n_runs):
        _run_original(config_path, sample_brief, "multi_turn_pressure", tdir, recorder)
    # Single-shot transcript (no probes).
    single = {
        "metadata": {"run_id": "ss0001", "brief_id": sample_brief["id"], "model_id": GEMINI,
                     "condition": "single_shot", "repetition": 0},
        "messages": [
            {"role": "system", "content": "s", "is_probe": False},
            {"role": "user", "content": "u", "is_probe": False},
            {"role": "assistant", "content": "a", "token_count": 5, "is_probe": False},
        ],
    }
    (tdir / "ss0001_x_single_shot_gemini.jsonl").write_text(json.dumps(single) + "\n")
    return tdir


def test_replay_writes_records_skips_single_shot_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=2)
    out_dir = tmp_path / "out"

    recorder = CallRecorder(probe_tokens=2000, finish_reason="stop", cost=0.01)
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, models=[GEMINI], conditions=None)
    assert len(transcripts) == 3

    summary = asyncio.run(replay_transcripts(
        runner, transcripts, out_dir, turn="final", probe_max_tokens=4096,
        max_budget=10.0, concurrency=2,
    ))
    assert summary["runs_scanned"] == 3
    assert summary["runs_single_shot_skipped"] == 1
    assert summary["probes_selected"] == 2
    assert summary["probes_replayed"] == 2
    assert summary["probes_skipped_budget"] == 0
    assert summary["cost_usd"] == pytest.approx(0.02)
    assert len(recorder.calls) == 2

    # Each replay call used the April temperature (0.0) and the new cap: only the cap changes.
    for call in recorder.calls:
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 4096
        assert call["messages"][-1]["content"] == get_restatement_probe()

    files = sorted(out_dir.glob("reprobe_*.jsonl"))
    assert len(files) == 2
    record = json.loads(files[0].read_text().splitlines()[0])
    required = {
        "run_id", "brief_id", "model_id", "condition", "turn_index",
        "original_probe_tokens", "original_probe_truncated", "new_probe_content",
        "new_probe_tokens", "new_finish_reason", "cost_usd", "timestamp", "probe_max_tokens",
    }
    assert required <= set(record)
    assert record["turn_index"] == 5
    assert record["original_probe_tokens"] == 1020
    assert record["original_probe_truncated"] is True
    assert record["new_probe_tokens"] == 2000
    assert record["new_finish_reason"] == "stop"
    assert record["finish_reason_source"] == "api"
    assert record["probe_max_tokens"] == 4096
    assert record["temperature"] == 0.0
    assert record["is_final_probe"] is True
    assert record["probe_prompt_matches_template"] is True
    assert record["cost_usd"] == pytest.approx(0.01)

    # Second invocation: everything exists, no call is made.
    summary2 = asyncio.run(replay_transcripts(
        runner, transcripts, out_dir, turn="final", probe_max_tokens=4096,
        max_budget=10.0, concurrency=2,
    ))
    assert summary2["runs_already_complete"] == 2
    assert summary2["probes_already_done"] == 2
    assert summary2["probes_replayed"] == 0
    assert len(recorder.calls) == 2

    # --turn all on the same output: the 4 earlier probes per run are still missing.
    summary3 = asyncio.run(replay_transcripts(
        runner, transcripts, out_dir, turn="all", probe_max_tokens=4096,
        max_budget=10.0, concurrency=2,
    ))
    assert summary3["probes_selected"] == 10
    assert summary3["probes_already_done"] == 2
    assert summary3["probes_replayed"] == 8
    lines = files[0].read_text().splitlines()
    assert sorted(json.loads(line)["turn_index"] for line in lines) == [1, 2, 3, 4, 5]


def test_budget_cap_stops_further_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=3)
    recorder = CallRecorder(cost=0.01)
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, models=None, conditions=["multi_turn_pressure"])
    assert len(transcripts) == 3
    summary = asyncio.run(replay_transcripts(
        runner, transcripts, tmp_path / "out", turn="final", probe_max_tokens=4096,
        max_budget=0.015, concurrency=1,
    ))
    assert summary["probes_replayed"] == 2
    assert summary["probes_skipped_budget"] == 1
    assert len(recorder.calls) == 2


def test_finish_reason_inferred_when_api_gives_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=1)
    recorder = CallRecorder(probe_tokens=4092, finish_reason=None)
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, None, ["multi_turn_pressure"])
    asyncio.run(replay_transcripts(
        runner, transcripts, tmp_path / "out", turn="final", probe_max_tokens=4096, max_budget=1.0,
    ))
    record = json.loads(next((tmp_path / "out").glob("reprobe_*.jsonl")).read_text())
    assert record["new_finish_reason"] == "length"
    assert record["finish_reason_source"] == "inferred_from_tokens"


def test_dry_run_makes_no_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=1)
    recorder = CallRecorder()
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, None, None)
    summary = asyncio.run(replay_transcripts(
        runner, transcripts, tmp_path / "out", turn="all", probe_max_tokens=4096,
        max_budget=1.0, dry_run=True,
    ))
    assert summary["probes_replayed"] == 5
    assert recorder.calls == []
    assert list((tmp_path / "out").glob("reprobe_*.jsonl")) == []


def test_partial_last_line_is_repaired_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    """A crash mid-write leaves a partial line. Resume must drop it and not glue the next record to it."""
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=1)
    out_dir = tmp_path / "out"
    recorder = CallRecorder(cost=0.01)
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, None, ["multi_turn_pressure"])
    run_id = transcripts[0][1]["metadata"]["run_id"]
    out_path = reprobe.output_path(out_dir, run_id)

    # One good record (turn 1) then a truncated record (turn 2) with no newline.
    out_dir.mkdir()
    good = {"run_id": run_id, "turn_index": 1, "new_probe_content": "ok"}
    partial = json.dumps({"run_id": run_id, "turn_index": 2, "new_probe_content": "cut"})[:-7]
    out_path.write_text(json.dumps(good) + "\n" + partial)

    assert reprobe.repair_partial_file(out_path) == 1
    assert out_path.read_bytes().endswith(b"\n")
    assert [r["turn_index"] for r in reprobe.read_existing_records(out_path)] == [1]

    # Resume replays turns 2 to 5; every line parses; turn 2 is recovered.
    out_path.write_text(json.dumps(good) + "\n" + partial)  # damage it again
    summary = asyncio.run(replay_transcripts(
        runner, transcripts, out_dir, turn="all", probe_max_tokens=4096, max_budget=1.0,
    ))
    assert summary["probes_already_done"] == 1
    assert summary["probes_replayed"] == 4
    lines = out_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["turn_index"] for r in records] == [1, 2, 3, 4, 5]
    assert records[1]["new_probe_content"].startswith("probe ")


def test_configured_temperature_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=1)
    recorder = CallRecorder()
    recorder.install(monkeypatch, cls=ReprobeRunner)
    runner = ReprobeRunner(config_path)
    transcripts = reprobe.list_transcripts(tdir, None, ["multi_turn_pressure"])
    asyncio.run(replay_transcripts(
        runner, transcripts, tmp_path / "out", turn="final", probe_max_tokens=4096,
        max_budget=1.0, temperature_spec="configured",
    ))
    assert recorder.calls[0]["temperature"] == 1.0  # Gemini Pro in config.yaml


def test_cli_budget_required_unless_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_path: Path, sample_brief: dict,
) -> None:
    tdir = _make_transcripts(monkeypatch, tmp_path, config_path, sample_brief, n_runs=1)
    recorder = CallRecorder()
    recorder.install(monkeypatch, cls=ReprobeRunner)
    base = ["--transcripts-dir", str(tdir), "--output-dir", str(tmp_path / "out"),
            "--config", str(config_path)]
    with pytest.raises(SystemExit):
        reprobe.main(base)
    summary = reprobe.main(base + ["--dry-run"])
    assert summary["probes_replayed"] == 1
    assert summary["temperature"] == "original"
    assert recorder.calls == []


def test_temperature_spec() -> None:
    class Stub:
        def _get_temperature(self, model_id):
            return 1.0

    assert reprobe.resolve_temperature(Stub(), GEMINI, "configured") == 1.0
    assert reprobe.resolve_temperature(Stub(), GEMINI, "original") == 0.0
    assert reprobe.resolve_temperature(Stub(), GEMINI, "0.3") == 0.3


# --- reprobe_accuracy --------------------------------------------------------

def test_accuracy_script_refuses_missing_reprobe_dir(tmp_path: Path, sample_brief: dict) -> None:
    """A missing replay directory must raise, never write empty CSVs over real ones."""
    data_root = tmp_path / "data"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    (briefs_dir / "test_01.yaml").write_text(yaml.safe_dump(sample_brief))
    out_dir = tmp_path / "analysis_out"
    with pytest.raises(FileNotFoundError):
        reprobe_accuracy.run(data_root, out_dir, briefs_dir=briefs_dir)
    assert not (out_dir / "reprobe_accuracy.csv").exists()
    assert not (out_dir / "reprobe_summary.csv").exists()


def test_accuracy_script_scores_and_joins(tmp_path: Path, sample_brief: dict) -> None:
    data_root = tmp_path / "data"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    (briefs_dir / "test_01.yaml").write_text(yaml.safe_dump(sample_brief))
    reprobe_dir = data_root / "camera_ready" / "e3_reprobe_gemini_pro"
    reprobe_dir.mkdir(parents=True)
    (data_root / "analysis").mkdir()
    pd.DataFrame([
        {"run_id": "r1", "brief_id": "test_01", "model_id": GEMINI, "condition": "multi_turn_pressure",
         "turn_index": 5, "overall_accuracy": 0.4, "constraint_recall": 1 / 3, "banned_recall": 0.5},
        {"run_id": "r2", "brief_id": "test_01", "model_id": GEMINI, "condition": "multi_turn_pressure",
         "turn_index": 5, "overall_accuracy": 1.0, "constraint_recall": 1.0, "banned_recall": 1.0},
        {"run_id": "r3", "brief_id": "test_01", "model_id": GEMINI, "condition": "multi_turn_neutral",
         "turn_index": 5, "overall_accuracy": 1.0, "constraint_recall": 1.0, "banned_recall": 1.0},
    ]).to_csv(data_root / "analysis" / "final_probe_accuracy.csv", index=False)

    def line(run_id, condition, content, orig_tokens, finish):
        return json.dumps({
            "run_id": run_id, "brief_id": "test_01", "model_id": GEMINI, "condition": condition,
            "turn_index": 5, "original_probe_tokens": orig_tokens,
            "original_probe_truncated": orig_tokens >= 1020, "new_probe_content": content,
            "new_probe_tokens": 1500, "new_finish_reason": finish, "cost_usd": 0.02,
            "timestamp": "t", "probe_max_tokens": 4096, "temperature": 1.0,
        })

    full = "constraint one, constraint two, constraint three; banned one, banned two"
    (reprobe_dir / "reprobe_r1.jsonl").write_text(line("r1", "multi_turn_pressure", full, 1020, "stop") + "\n")
    (reprobe_dir / "reprobe_r2.jsonl").write_text(
        line("r2", "multi_turn_pressure", "constraint one only", 900, "length") + "\n")
    # r3 is a non-final replay (turn 3): the April CSV has no row for that turn_index.
    r3 = json.loads(line("r3", "multi_turn_neutral", full, 1020, "stop"))
    r3["turn_index"] = 3
    (reprobe_dir / "reprobe_r3.jsonl").write_text(json.dumps(r3) + "\n")

    out = reprobe_accuracy.run(data_root, tmp_path / "out", briefs_dir=briefs_dir)
    acc = out["reprobe_accuracy"].set_index("run_id")
    assert len(acc) == 3
    assert acc.loc["r1", "new_overall_accuracy"] == pytest.approx(1.0)
    assert acc.loc["r1", "original_overall_accuracy"] == pytest.approx(0.4)
    assert acc.loc["r1", "accuracy_delta"] == pytest.approx(0.6)
    assert bool(acc.loc["r1", "original_truncated"]) is True
    assert acc.loc["r2", "new_overall_accuracy"] == pytest.approx(0.2)
    assert acc.loc["r2", "new_constraint_recall"] == pytest.approx(1 / 3)
    assert acc.loc["r2", "new_banned_recall"] == pytest.approx(0.0)
    assert bool(acc.loc["r2", "original_truncated"]) is False
    assert pd.isna(acc.loc["r3", "original_overall_accuracy"])

    summ = out["reprobe_summary"].set_index("condition")
    p = summ.loc["multi_turn_pressure"]
    assert p["n"] == 2 and p["n_with_original"] == 2
    assert p["mean_original_accuracy"] == pytest.approx(0.7)
    assert p["mean_new_accuracy"] == pytest.approx(0.6)
    assert p["share_original_truncated"] == pytest.approx(0.5)
    assert p["share_new_finish_length"] == pytest.approx(0.5)
    assert p["share_original_ge_0_8"] == pytest.approx(0.5)
    assert p["share_new_ge_0_8"] == pytest.approx(0.5)
    n = summ.loc["multi_turn_neutral"]
    assert n["n"] == 1 and n["n_with_original"] == 0
    assert pd.isna(n["mean_original_accuracy"])
    # Shares at the 0.8 threshold use n_with_original as denominator: NaN when it is 0.
    assert pd.isna(n["share_original_ge_0_8"]) and pd.isna(n["share_new_ge_0_8"])
    assert n["share_original_truncated"] == pytest.approx(1.0)

    prov = json.loads((tmp_path / "out" / "reprobe_accuracy.csv.provenance.json").read_text())
    assert prov["rows_in"] == 3 and prov["rows_out"] == 3 and prov["rows_joined_to_original"] == 2


def test_accuracy_script_ignores_run_log(tmp_path: Path, sample_brief: dict) -> None:
    """The runner's summary log matches the reprobe_*.jsonl glob but holds no probes."""
    data_root = tmp_path / "data"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    (briefs_dir / "test_01.yaml").write_text(yaml.safe_dump(sample_brief))
    reprobe_dir = data_root / "camera_ready" / "e3_reprobe_gemini_pro"
    reprobe_dir.mkdir(parents=True)
    (reprobe_dir / "reprobe_r1.jsonl").write_text(json.dumps({
        "run_id": "r1", "brief_id": "test_01", "model_id": GEMINI, "condition": "multi_turn_pressure",
        "turn_index": 5, "original_probe_tokens": 1020, "original_probe_truncated": True,
        "new_probe_content": "objective", "new_probe_tokens": 1500, "new_finish_reason": "stop",
        "cost_usd": 0.02, "timestamp": "t", "probe_max_tokens": 4096, "temperature": 0.0,
    }) + "\n")
    (reprobe_dir / reprobe_accuracy.RUN_LOG_NAME).write_text(json.dumps({
        "runs_scanned": 1, "probes_replayed": 1, "cost_usd": 0.02, "dry_run": False,
    }) + "\n")
    records, files = reprobe_accuracy.load_reprobe_records(reprobe_dir)
    assert len(files) == 1 and files[0].endswith("reprobe_r1.jsonl")
    assert len(records) == 1
    out = reprobe_accuracy.run(data_root, tmp_path / "out", reprobe_dir=reprobe_dir, briefs_dir=briefs_dir)
    assert len(out["reprobe_accuracy"]) == 1
    prov = json.loads((tmp_path / "out" / "reprobe_accuracy.csv.provenance.json").read_text())
    assert prov["rows_in"] == 1
    assert prov["rows_without_brief"] == 0
    assert prov["inputs"]["reprobe_files"] == 1


def test_write_csv_names_sidecar_after_csv(tmp_path: Path) -> None:
    """Plan P12: the sidecar is ``<name>.csv.provenance.json``."""
    reprobe_accuracy._write_csv(pd.DataFrame({"a": [1]}), tmp_path / "reprobe_summary.csv", {"inputs": {}})
    assert not (tmp_path / "reprobe_summary.provenance.json").exists()
    prov = json.loads((tmp_path / "reprobe_summary.csv.provenance.json").read_text())
    assert prov["rows_out"] == 1
