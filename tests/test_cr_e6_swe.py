"""E6 result tables (drift_bench.analysis.camera_ready.e6_swe) on synthetic data.

Pure file checks: the joins, the KBV flag, the cell means, the cost sums
and the executable-arm side-by-side table. No network, no API keys.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import e6_swe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIEFS_SWE = PROJECT_ROOT / "drift_bench" / "briefs_swe"
PRO = "gemini/gemini-3.1-pro-preview"
MINI = "openai/gpt-5.4-mini"
OPUS = "anthropic/claude-opus-4-6"


def _probe_text(brief: dict) -> str:
    return "\n".join(["Objective: " + brief["objective"], *brief["hard_constraints"], *brief["banned_moves"]])


def _transcript(run_id, brief, model, condition, rep, cost, finish="stop", probe_text=None):
    msgs = [{"role": "system", "content": "s", "timestamp": 0},
            {"role": "user", "content": "u", "timestamp": 0}]
    n_turns = 1 if condition == "single_shot" else 6
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"turn {i}\n```python\nx = {i}\n```", "timestamp": 0,
                     "token_count": 100, "finish_reason": finish, "response_model": model.split("/")[-1] + "-2026"})
        if condition != "single_shot" and i < n_turns - 1:
            msgs.append({"role": "user", "content": "probe", "timestamp": 0, "is_probe": True})
            msgs.append({"role": "assistant", "content": probe_text or "nothing", "timestamp": 0, "is_probe": True})
        if i < n_turns - 1:
            msgs.append({"role": "user", "content": "next", "timestamp": 0})
    return {"metadata": {"run_id": run_id, "brief_id": brief["id"], "model_id": model, "condition": condition,
                         "repetition": rep, "start_time": 1.0, "end_time": 2.0, "total_input_tokens": 1000,
                         "total_output_tokens": 500, "total_cost_usd": cost, "error": None,
                         "temperature": 1.0, "max_tokens": 4096, "litellm_version": "1.83.8"},
            "messages": msgs}


def _judge(run_id, brief_id, model, condition, adherence, fidelity=4, cost=0.04, turn_index=None):
    return {"run_id": run_id, "brief_id": brief_id, "model_id": model, "condition": condition,
            "judge_model": OPUS, "judge_response_model": "claude-opus-4-6-20260301",
            "objective_fidelity": fidelity, "constraint_adherence": adherence, "alternative_coverage": 3,
            "complexity_inflation": 1, "summary": "", "violations": [] if adherence == 4 else ["v"],
            "optional_extras_flagged": ["e"], "timestamp": 1.0, "cost_usd": cost, "cost_estimated": False,
            "turn_index": turn_index}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


@pytest.fixture
def e6_data(tmp_path: Path) -> Path:
    import yaml

    data_root = tmp_path / "data"
    cr = data_root / "camera_ready"
    b02 = yaml.safe_load((BRIEFS_SWE / "swe_02.yaml").read_text())
    b10 = yaml.safe_load((BRIEFS_SWE / "swe_10.yaml").read_text())
    good = _probe_text(b02)
    # Main arm: Pro and mini, swe_02 and swe_10, SS and MT-P, one rep each.
    main_t, main_s = cr / "e6_swe" / "transcripts", cr / "e6_swe" / "scores"
    _write(main_t / "a1_swe_02_single_shot_x.jsonl", _transcript("a1", b02, PRO, "single_shot", 0, 0.20))
    _write(main_s / "judge_a1.jsonl", _judge("a1", "swe_02", PRO, "single_shot", 4))
    _write(main_t / "a2_swe_02_multi_turn_pressure_x.jsonl", _transcript("a2", b02, PRO, "multi_turn_pressure", 0, 0.30, probe_text=good))
    _write(main_s / "judge_a2.jsonl", _judge("a2", "swe_02", PRO, "multi_turn_pressure", 3))  # KBV
    _write(main_t / "a3_swe_10_multi_turn_pressure_x.jsonl", _transcript("a3", b10, PRO, "multi_turn_pressure", 0, 0.30, probe_text=_probe_text(b10)))
    _write(main_s / "judge_a3.jsonl", _judge("a3", "swe_10", PRO, "multi_turn_pressure", 4))  # knows, complies
    _write(main_t / "a4_swe_02_multi_turn_pressure_x.jsonl", _transcript("a4", b02, MINI, "multi_turn_pressure", 0, 0.05, probe_text="forgot"))
    _write(main_s / "judge_a4.jsonl", _judge("a4", "swe_02", MINI, "multi_turn_pressure", 3))  # violates, no recall
    _write(main_t / "a5_swe_02_single_shot_x.jsonl", _transcript("a5", b02, MINI, "single_shot", 0, 0.02))
    # unjudged run
    _write(main_s / "judge_t3_a5.jsonl", _judge("a5", "swe_02", MINI, "single_shot", 2, turn_index=3))  # per-turn file, ignored
    # Exec arm: one Pro run, judged under contract and plain briefs.
    ex_t, ex_s, ex_p = cr / "e6_swe_verify" / "transcripts", cr / "e6_swe_verify" / "scores", cr / "e6_swe_verify" / "scores_plain_brief"
    _write(ex_t / "c1_swe_10_single_shot_x.jsonl", _transcript("c1", b10, PRO, "single_shot", 0, 0.40, finish="length"))
    _write(ex_s / "judge_c1.jsonl", _judge("c1", "swe_10", PRO, "single_shot", 4, fidelity=4))
    _write(ex_p / "judge_c1.jsonl", _judge("c1", "swe_10", PRO, "single_shot", 4, fidelity=3, cost=0.03))
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame([{"run_id": "c1", "passed": 7, "failed": 2, "errors": 0, "total": 9, "all_passed": False,
                   "timed_out": False, "truncated": True, "no_code": False, "refused": False, "refused_reason": "",
                   "static_checks": 5, "static_failed": 0}]).to_csv(out / "swe_verify_results.csv", index=False)
    return data_root


def test_runs_table_joins_and_kbv(e6_data: Path, tmp_path: Path) -> None:
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    runs = out["e6_swe_runs"].set_index("run_id")
    assert sorted(runs.index) == ["a1", "a2", "a3", "a4", "a5", "c1"]
    assert runs.loc["a2", "kbv"] and runs.loc["a2", "knows"] and runs.loc["a2", "violates"]
    assert not runs.loc["a3", "kbv"] and runs.loc["a3", "knows"]
    assert not runs.loc["a4", "kbv"] and runs.loc["a4", "violates"] and not runs.loc["a4", "knows"]
    assert not runs.loc["a1", "kbv"]  # single shot never KBV
    assert not runs.loc["a5", "judged"] and pd.isna(runs.loc["a5", "j_constraint_adherence"])
    assert runs.loc["a1", "subject_response_model"] == "gemini-3.1-pro-preview-2026"
    assert runs.loc["c1", "final_finish_reason"] == "length" and runs.loc["c1", "arm"] == "exec"
    assert runs.loc["a2", "n_probes"] == 5 and runs.loc["a1", "n_probes"] == 0
    assert runs.loc["a1", "litellm_version"] == "1.83.8"
    assert runs.loc["a2", "judge_response_model"] == "claude-opus-4-6-20260301"


def test_cell_table(e6_data: Path, tmp_path: Path) -> None:
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    cells = out["e6_swe_cells"].set_index(["arm", "model_id", "condition"])
    pro_mtp = cells.loc[("main", PRO, "multi_turn_pressure")]
    assert pro_mtp["n"] == 2 and pro_mtp["n_kbv"] == 1 and pro_mtp["kbv_rate"] == 0.5
    assert pro_mtp["mean_constraint_adherence"] == 3.5 and pro_mtp["noncompliance_rate"] == 0.5
    assert 0.0 <= pro_mtp["kbv_ci_lo"] < 0.5 < pro_mtp["kbv_ci_hi"] <= 1.0
    mini_mtp = cells.loc[("main", MINI, "multi_turn_pressure")]
    assert mini_mtp["n_kbv"] == 0 and mini_mtp["knows_rate"] == 0.0
    ss = cells.loc[("main", PRO, "single_shot")]
    assert ss["n"] == 1 and pd.isna(ss["kbv_rate"])
    # a5 is unjudged: not in the mini single_shot cell.
    assert ("main", MINI, "single_shot") not in cells.index
    assert ("exec", PRO, "single_shot") in cells.index


def test_cost_table_sums_litellm_fields(e6_data: Path, tmp_path: Path) -> None:
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    costs = out["e6_swe_costs"]
    subj = costs[(costs["phase"] == "subjects") & (costs["arm"] == "main")].set_index("model_id")
    assert abs(subj.loc[PRO, "cost_usd"] - 0.80) < 1e-9 and subj.loc[PRO, "n"] == 3
    assert abs(subj.loc[MINI, "cost_usd"] - 0.07) < 1e-9
    judge = costs[(costs["phase"] == "judge") & (costs["arm"] == "main")].iloc[0]
    assert judge["n"] == 4 and abs(judge["cost_usd"] - 0.16) < 1e-9 and judge["model_id"] == OPUS
    plain = costs[costs["phase"] == "judge_plain_brief"].iloc[0]
    assert plain["n"] == 1 and abs(plain["cost_usd"] - 0.03) < 1e-9
    total = costs[costs["phase"] == "total"].iloc[0]
    assert abs(total["cost_usd"] - (0.80 + 0.07 + 0.16 + 0.40 + 0.04 + 0.03)) < 1e-9


def test_exec_fidelity_table(e6_data: Path, tmp_path: Path) -> None:
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    f = out["e6_swe_exec_fidelity"].set_index("run_id")
    assert list(f.index) == ["c1"]
    assert f.loc["c1", "contract_objective_fidelity"] == 4 and f.loc["c1", "plain_objective_fidelity"] == 3
    assert f.loc["c1", "delta_objective_fidelity"] == 1
    assert f.loc["c1", "passed"] == 7 and f.loc["c1", "truncated"]


def test_outputs_and_provenance(e6_data: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    e6_swe.run(e6_data, out_dir, BRIEFS_SWE)
    for name in ("e6_swe_runs", "e6_swe_cells", "e6_swe_costs", "e6_swe_exec_fidelity"):
        assert (out_dir / f"{name}.csv").is_file()
        prov = json.loads((out_dir / f"{name}.csv.provenance.json").read_text())
        assert "kbv_definition" in prov and prov["rows_out"] >= 1 and prov["filters"]


def test_missing_arm_raises_clear_file_not_found(tmp_path: Path, e6_data: Path) -> None:
    """run_analysis item 6 must see FileNotFoundError naming the arm, not a KeyError from a merge."""
    with pytest.raises(FileNotFoundError, match="arm 'main'.*transcripts"):
        e6_swe.run(tmp_path / "nothing", tmp_path / "out", BRIEFS_SWE)
    # Transcripts present, scores dir empty (the partial state after a failed judge phase).
    import shutil
    scores = e6_data / "camera_ready" / "e6_swe" / "scores"
    shutil.rmtree(scores)
    scores.mkdir()
    with pytest.raises(FileNotFoundError, match="arm 'main'.*judge_\\*\\.jsonl"):
        e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    # Exec arm without transcripts.
    shutil.rmtree(e6_data / "camera_ready" / "e6_swe_verify" / "transcripts")
    (e6_data / "camera_ready" / "e6_swe" / "scores" / "judge_a1.jsonl").write_text(
        json.dumps(_judge("a1", "swe_02", PRO, "single_shot", 4)) + "\n")
    with pytest.raises(FileNotFoundError, match="arm 'exec'.*transcripts"):
        e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)


def test_replayed_probe_becomes_knows_input(e6_data: Path, tmp_path: Path) -> None:
    """A reprobe record for a4 (original probe 'forgot') with a full restatement flips knows; probe_source=replay."""
    import yaml
    b02 = yaml.safe_load((BRIEFS_SWE / "swe_02.yaml").read_text())
    rdir = e6_data / "camera_ready" / "e6_swe" / "reprobe_gemini_pro"
    rdir.mkdir()
    _write(rdir / "reprobe_a4.jsonl", {
        "run_id": "a4", "brief_id": "swe_02", "model_id": MINI, "condition": "multi_turn_pressure", "turn_index": 5,
        "original_probe_tokens": 1020, "original_probe_truncated": True, "new_probe_content": _probe_text(b02),
        "new_probe_tokens": 1500, "new_finish_reason": "stop", "cost_usd": 0.03, "probe_max_tokens": 4096,
        "temperature": 0.0, "response_model": "x", "is_final_probe": True, "probe_prompt_matches_template": True})
    (rdir / "reprobe_run_log.jsonl").write_text(json.dumps({"dry_run": False}) + "\n")
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    runs = out["e6_swe_runs"].set_index("run_id")
    assert runs.loc["a4", "probe_source"] == "replay" and runs.loc["a2", "probe_source"] == "original"
    assert runs.loc["a4", "original_probe_accuracy"] < 0.8 and runs.loc["a4", "final_probe_accuracy"] == 1.0
    assert runs.loc["a4", "knows"] and runs.loc["a4", "kbv"], "adherence 3 plus replayed recall is KBV"
    assert runs.loc["a4", "replay_probe_tokens"] == 1500
    cells = out["e6_swe_cells"].set_index(["arm", "model_id", "condition"])
    mini = cells.loc[("main", MINI, "multi_turn_pressure")]
    assert mini["n_kbv"] == 1 and mini["n_probe_replay"] == 1 and mini["mean_original_probe_accuracy"] < 0.8
    assert mini["n_kbv_capped_as_graded"] == 0 and mini["kbv_rate_capped_as_graded"] == 0.0, \
        "with the capped probe graded as-is, a4 does not 'know'"
    pro = cells.loc[("main", PRO, "multi_turn_pressure")]
    assert pro["n_kbv_capped_as_graded"] == pro["n_kbv"] == 1, "no replay for Pro here: both KBV columns agree"
    assert pd.isna(cells.loc[("main", PRO, "single_shot")]["n_kbv_capped_as_graded"])
    cell_prov = json.loads((tmp_path / "out" / "e6_swe_cells.csv.provenance.json").read_text())
    assert "kbv_capped_as_graded" in cell_prov
    prov = json.loads((tmp_path / "out" / "e6_swe_runs.csv.provenance.json").read_text())
    assert prov["probe_source"].startswith("1 runs use the replayed final probe")
    assert sum("reprobe_" in i for i in prov["inputs"]) == 2, "both arms' replay directories are inputs"
    # The replay's litellm cost is a phase of the cost table and part of the total.
    costs = out["e6_swe_costs"]
    rp = costs[costs["phase"] == "probe_replay"]
    assert len(rp) == 1 and rp.iloc[0]["arm"] == "main" and rp.iloc[0]["model_id"] == MINI
    assert rp.iloc[0]["n"] == 1 and abs(rp.iloc[0]["cost_usd"] - 0.03) < 1e-9 and rp.iloc[0]["output_tokens"] == 1500
    assert rp.iloc[0]["response_models"] == "x"
    total = costs[costs["phase"] == "total"].iloc[0]
    assert abs(total["cost_usd"] - (0.80 + 0.07 + 0.16 + 0.40 + 0.04 + 0.03 + 0.03)) < 1e-9
    assert runs.loc["a4", "replay_cost_usd"] == 0.03 and runs.loc["a4", "replay_response_model"] == "x"


def test_exec_arm_replay_is_read_from_its_own_directory(e6_data: Path, tmp_path: Path) -> None:
    """A replay under e6_swe_verify/reprobe_gemini_pro/ feeds the exec arm's knows input and cost row."""
    import yaml
    b10 = yaml.safe_load((BRIEFS_SWE / "swe_10.yaml").read_text())
    cr = e6_data / "camera_ready"
    _write(cr / "e6_swe_verify" / "transcripts" / "c2_swe_10_multi_turn_pressure_x.jsonl",
           _transcript("c2", b10, PRO, "multi_turn_pressure", 0, 0.50, probe_text="forgot"))
    _write(cr / "e6_swe_verify" / "scores" / "judge_c2.jsonl", _judge("c2", "swe_10", PRO, "multi_turn_pressure", 3))
    rdir = cr / "e6_swe_verify" / "reprobe_gemini_pro"
    rdir.mkdir()
    _write(rdir / "reprobe_c2.jsonl", {
        "run_id": "c2", "brief_id": "swe_10", "model_id": PRO, "condition": "multi_turn_pressure", "turn_index": 5,
        "original_probe_tokens": 1020, "original_probe_truncated": True, "new_probe_content": _probe_text(b10),
        "new_probe_tokens": 1300, "new_finish_reason": "stop", "cost_usd": 0.033, "probe_max_tokens": 4096,
        "temperature": 0.0, "response_model": "gemini-3.1-pro-preview", "is_final_probe": True,
        "probe_prompt_matches_template": True})
    (rdir / "reprobe_run_log.jsonl").write_text(json.dumps({"dry_run": False}) + "\n")
    out = e6_swe.run(e6_data, tmp_path / "out", BRIEFS_SWE)
    runs = out["e6_swe_runs"].set_index("run_id")
    assert runs.loc["c2", "arm"] == "exec" and runs.loc["c2", "probe_source"] == "replay"
    assert runs.loc["c2", "original_probe_accuracy"] < 0.8 and runs.loc["c2", "knows"] and runs.loc["c2", "kbv"]
    assert runs.loc["a2", "probe_source"] == "original", "a main-arm run is untouched by the exec replay"
    cells = out["e6_swe_cells"].set_index(["arm", "model_id", "condition"])
    ex = cells.loc[("exec", PRO, "multi_turn_pressure")]
    assert ex["n_kbv"] == 1 and ex["n_probe_replay"] == 1
    costs = out["e6_swe_costs"]
    rp = costs[(costs["phase"] == "probe_replay") & (costs["arm"] == "exec")].iloc[0]
    assert rp["n"] == 1 and abs(rp["cost_usd"] - 0.033) < 1e-9 and rp["response_models"] == "gemini-3.1-pro-preview"
    assert not ((costs["phase"] == "probe_replay") & (costs["arm"] == "main")).any()
    fid = out["e6_swe_exec_fidelity"].set_index("run_id")
    assert "c2" in fid.index


def test_wilson() -> None:
    lo, hi = e6_swe.wilson(6, 8)
    assert abs(lo - 0.41) < 0.01 and abs(hi - 0.93) < 0.01  # pilot report: 75% (6/8) [41-93]
    assert e6_swe.wilson(0, 0) == (pytest.approx(float("nan"), nan_ok=True), pytest.approx(float("nan"), nan_ok=True))
