"""Tests for drift_bench.analysis.camera_ready.stretch_violate (synthetic inputs, no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import stretch_violate as sv


def _scores(rows):
    return pd.DataFrame(rows, columns=["run_id", "brief_id", "model_id", "condition", "judge_model", "j_constraint_adherence"])


def _probes(d):
    return pd.DataFrame({"run_id": list(d), "probe_turn_index": 5, "overall_accuracy": list(d.values())})


def test_load_judge_adherence_absent_and_prefix(tmp_path):
    assert sv.load_judge_adherence(tmp_path / "nope", "judge_opus").empty
    (tmp_path / "judge_opus_r1.jsonl").write_text(json.dumps({"run_id": "r1", "judge_model": "opus", "constraint_adherence": 2}) + "\n")
    (tmp_path / "judge_gpt_r1.jsonl").write_text(json.dumps({"run_id": "r1", "judge_model": "gpt", "constraint_adherence": 3}) + "\n")
    df = sv.load_judge_adherence(tmp_path, "judge_opus")
    assert len(df) == 1 and df["constraint_adherence"].iloc[0] == 2 and df["source"].iloc[0] == "judge_opus"


def test_build_runs_flags():
    scores = _scores([
        ["a", "b1", "m", "multi_turn_pressure", "j", 3],
        ["b", "b1", "m", "multi_turn_pressure", "j", 2],
        ["c", "b1", "m", "multi_turn_pressure", "j", 4],
        ["d", "b1", "m", "single_shot", "j", 3],
        ["e", "b1", "m", "multi_turn_pressure_rigor", "j", 1],
    ])
    probes = _probes({"a": 1.0, "b": 0.7, "c": 1.0})
    judges = {"skeptical": pd.DataFrame({"source": ["skeptical"], "run_id": ["a"], "judge_model": ["g"], "constraint_adherence": [2]})}
    runs = sv.build_runs(scores, probes, judges, pd.DataFrame())
    assert "e" not in set(runs["run_id"])  # non-core condition dropped
    r = runs.set_index("run_id")
    assert r.loc["a", "kbv_positive"] and r.loc["a", "at_3"] and not r.loc["a", "below_3"]
    assert not r.loc["b", "kbv_positive"] and r.loc["b", "below_3"]  # knows < 0.8
    assert not r.loc["c", "kbv_positive"]  # adherence 4
    assert not r.loc["d", "kbv_positive"]  # no probe row
    assert r.loc["a", "skeptical_adherence"] == 2 and pd.isna(r.loc["b", "skeptical_adherence"])


def _synthetic_runs():
    rows = []
    # Model X: pressure runs at 3 (stretch only). Sonnet: pressure runs below 3.
    for i in range(4):
        rows.append([f"x_ss{i}", f"b{i}", "x", "single_shot", "j", 4])
        rows.append([f"x_mp{i}", f"b{i}", "x", "multi_turn_pressure", "j", 3])
        rows.append([f"s_ss{i}", f"b{i}", sv.SONNET, "single_shot", "g", 4])
        rows.append([f"s_mp{i}", f"b{i}", sv.SONNET, "multi_turn_pressure", "g", 2])
    scores = _scores(rows)
    probes = _probes({r[0]: 1.0 for r in rows if r[3] == "multi_turn_pressure"})
    return scores, probes


def test_summary_split_and_group_rows():
    scores, probes = _synthetic_runs()
    runs = sv.build_runs(scores, probes, {}, pd.DataFrame())
    checks = pd.DataFrame({"run_id": ["x_mp0", "x_mp0", "s_mp0"], "judge_model": ["j", "j", "g"],
                           "status": ["satisfied", "stretched", "violated"]})
    s = sv.summarise_runs_split(runs, checks).set_index(["model_group", "condition"])
    x = s.loc[("x", "multi_turn_pressure")]
    assert x["n_kbv_positive"] == 4 and x["share_kbv_at_3"] == 1.0 and x["n_kbv_below_3"] == 0
    assert x["share_below_3"] == 0.0 and x["n_adh_3"] == 4
    assert x["structured_n_checks"] == 2 and x["structured_share_stretched"] == 0.5
    son = s.loc[(sv.SONNET, "multi_turn_pressure")]
    assert son["share_kbv_below_3"] == 1.0 and son["structured_share_violated"] == 1.0
    allp = s.loc[(sv.GROUP_ALL, "multi_turn_pressure")]
    assert allp["n_runs"] == 8 and allp["n_below_3"] == 4 and allp["share_kbv_at_3"] == 0.5
    excl = s.loc[(sv.GROUP_EXCL, "multi_turn_pressure")]
    assert excl["n_runs"] == 4 and excl["n_below_3"] == 0


def test_movement_table_excluding_sonnet_llama():
    scores, probes = _synthetic_runs()
    runs = sv.build_runs(scores, probes, {}, pd.DataFrame())
    m = sv.movement_table(runs).set_index("model_group")
    assert m.loc[sv.GROUP_ALL, "n_below_3_pressure"] == 4 and m.loc[sv.GROUP_ALL, "n_below_3_single_shot"] == 0
    assert m.loc[sv.GROUP_EXCL, "n_below_3_pressure"] == 0
    assert m.loc[sv.GROUP_EXCL, "delta_mean_adherence"] == pytest.approx(-1.0)
    assert m.loc[sv.GROUP_EXCL, "n_briefs_paired"] == 4
    assert 0 <= m.loc[sv.GROUP_ALL, "below_3_fisher_p"] <= 1
    assert m.loc[sv.GROUP_EXCL, "welch_t_p"] < 0.05 or np.isnan(m.loc[sv.GROUP_EXCL, "welch_t_p"])


def test_by_judge_table_moves():
    scores, probes = _synthetic_runs()
    # The skeptical judge scores the four model-x pressure runs: three move to 2, one stays at 3.
    sk = pd.DataFrame({"source": "skeptical", "run_id": [f"x_mp{i}" for i in range(4)], "judge_model": "g",
                       "constraint_adherence": [2, 2, 2, 3]})
    runs = sv.build_runs(scores, probes, {"skeptical": sk}, pd.DataFrame())
    bj = sv.by_judge_table(runs, ["skeptical", "judge_opus"])
    assert set(bj["judge_source"]) == {"published", "skeptical"}
    row = bj[(bj["judge_source"] == "skeptical") & (bj["model_id"] == "x") & (bj["condition"] == "multi_turn_pressure")].iloc[0]
    assert row["n_runs"] == 4
    assert row["n_published_kbv_at_3"] == 4
    assert row["share_published_kbv_at_3_moved_below_3"] == pytest.approx(0.75)
    assert row["share_published_kbv_below_3_under_judge"] == pytest.approx(0.75)
    pub = bj[(bj["judge_source"] == "published") & (bj["model_id"] == "x") & (bj["condition"] == "multi_turn_pressure")].iloc[0]
    assert pub["share_below_3"] == 0.0 and pub["n_runs"] == 4


def test_run_end_to_end(tmp_path):
    root = tmp_path / "data"
    (root / "aggregated").mkdir(parents=True)
    (root / "analysis").mkdir()
    scores, probes = _synthetic_runs()
    full = scores.assign(j_objective_fidelity=4, j_alternative_coverage=4, j_complexity_inflation=0)
    full.to_parquet(root / "aggregated" / "all_scores.parquet", index=False)
    probes.rename(columns={"probe_turn_index": "turn_index"}).assign(brief_id="b", model_id="x", condition="multi_turn_pressure",
                                                                     constraint_recall=1.0, banned_recall=1.0).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)
    (root / "scores").mkdir()
    (root / "scores" / "structured_x_mp0.jsonl").write_text(json.dumps({
        "run_id": "x_mp0", "brief_id": "b0", "model_id": "x", "condition": "multi_turn_pressure", "judge_model": "j",
        "constraint_adherence": 3, "constraint_checks": [{"constraint": "c", "status": "stretched", "reason": ""}]}) + "\n")
    out = tmp_path / "out"
    frames = sv.run(root, out)
    for name, df in frames.items():
        assert (out / f"{name}.csv").exists()
        prov = json.loads((out / f"{name}.csv.provenance.json").read_text())
        assert prov["rows_out"] == len(df)
    assert prov["inputs"]["judge_sources"]["judge_opus"]["present"] is False
    assert prov["inputs"]["judge_sources"]["structured"]["files"] == 1
    assert [f["name"] for f in prov["filters"]] == ["drop_stray_sonnet_runs"] * 2
    assert prov["filters"][0]["rows_dropped"] == 0 and prov["rows_in"]["parquet_after_stray_filter"] == 16
    assert "sample" in prov["rules"] and "Sonnet pressure n = 4" in prov["rules"]["denominators"]
    runs = frames["stretch_violate_runs"].set_index("run_id")
    assert runs.loc["x_mp0", "structured_adherence"] == 3
    assert runs.loc["x_mp0", "structured_n_not_satisfied"] == 1
