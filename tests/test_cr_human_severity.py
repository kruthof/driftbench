"""Tests for drift_bench.analysis.camera_ready.human_severity (synthetic inputs, no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import human_severity as hs


def _human_json(tmp_path):
    data = {
        "r1": {"run_id": "r1", "brief_id": "b1", "n_raters": 3, "rater_ids": ["A", "B", "C"],
               "constraint_statuses": ["Satisfied", "Satisfied", "Violated", "Stretched"],
               "violated_count": 1, "stretched_count": 1, "computed_score": 2, "computed_score_from_constraint_median": 2, "per_rater_scores": {}},
        "r2": {"run_id": "r2", "brief_id": "b1", "n_raters": 3, "rater_ids": ["A", "B", "C"],
               "constraint_statuses": ["Satisfied", "Satisfied", "Satisfied", "Satisfied"],
               "violated_count": 0, "stretched_count": 0, "computed_score": 4, "computed_score_from_constraint_median": 4, "per_rater_scores": {}},
        "r3": {"run_id": "r3", "brief_id": "b2", "n_raters": 3, "rater_ids": ["A", "B", "C"],
               "constraint_statuses": ["Violated", "Violated"],
               "violated_count": 2, "stretched_count": 0, "computed_score": 1, "computed_score_from_constraint_median": 1, "per_rater_scores": {}},
    }
    p = tmp_path / "human_scores.json"
    p.write_text(json.dumps(data))
    return p


def _scores():
    return pd.DataFrame({
        "run_id": ["r1", "r2", "r3"], "model_id": ["m", "m", "m"],
        "condition": ["multi_turn_pressure", "single_shot", "multi_turn_pressure"],
        "judge_model": ["j", "j", "j"], "j_constraint_adherence": [3, 4, 2],
    })


def _checks():
    # r1 judge: satisfied, satisfied, stretched, stretched. r3 judge: satisfied, violated. r2 has no structured file.
    rows = []
    for i, s in enumerate(["satisfied", "satisfied", "stretched", "stretched"]):
        rows.append({"run_id": "r1", "judge_model": "j", "constraint_index": i, "status": s})
    for i, s in enumerate(["satisfied", "violated"]):
        rows.append({"run_id": "r3", "judge_model": "j", "constraint_index": i, "status": s})
    return pd.DataFrame(rows)


def test_load_human_scores_counts(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    r = h.set_index("run_id")
    assert r.loc["r1", "n_items"] == 4 and r.loc["r1", "n_violated"] == 1 and r.loc["r1", "n_not_satisfied"] == 2
    assert r.loc["r1", "any_violation"] and not r.loc["r2", "any_violation"]


def test_load_rater_labels(tmp_path):
    d = tmp_path / "raters"
    d.mkdir()
    (d / "rater_A.json").write_text(json.dumps({"r1": {"run_id": "r1", "rater_id": "A", "constraint_statuses": ["Violated", "Satisfied"]}}))
    (d / "rater_B.json").write_text(json.dumps({"r1": {"run_id": "r1", "rater_id": "B", "constraint_statuses": ["Stretched", "Satisfied"]}}))
    rl = hs.load_rater_labels(d)
    assert len(rl) == 4 and set(rl["rater_id"]) == {"A", "B"}
    assert hs.load_rater_labels(tmp_path / "absent").empty


def test_by_condition_pooled_vs_run_mean(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    runs = hs.build_runs(h, _scores(), _checks(), pd.DataFrame(columns=["rater_id", "run_id", "item", "status"]), pd.DataFrame(columns=["run_id", "judge_violations_listed"]))
    t = hs.by_condition_table(runs, pd.DataFrame(columns=["rater_id", "run_id", "item", "status"]))
    pooled = t[(t["aggregation"] == "median_label_pooled") & (t["condition"] == "multi_turn_pressure")].iloc[0]
    # r1: 1 violated of 4, r3: 2 of 2. Pooled: 3 / 6. Run mean: (0.25 + 1.0) / 2.
    assert pooled["n_runs"] == 2 and pooled["n_items"] == 6
    assert pooled["share_violated"] == pytest.approx(0.5)
    assert pooled["share_not_satisfied"] == pytest.approx(4 / 6)
    assert pooled["violations_per_run"] == pytest.approx(1.5)
    assert pooled["share_runs_with_violation"] == 1.0
    rm = t[(t["aggregation"] == "median_label_run_mean") & (t["condition"] == "multi_turn_pressure")].iloc[0]
    assert rm["share_violated"] == pytest.approx(0.625)
    ss = t[(t["aggregation"] == "median_label_pooled") & (t["condition"] == "single_shot")].iloc[0]
    assert ss["share_violated"] == 0.0 and ss["share_runs_with_violation"] == 0.0
    assert "per_rater_pooled" not in set(t["aggregation"])  # no rater labels given


def test_per_rater_pooled_uses_rater_run_pairs(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    rl = pd.DataFrame([
        {"rater_id": "A", "run_id": "r2", "item": 0, "status": "violated"},
        {"rater_id": "A", "run_id": "r2", "item": 1, "status": "satisfied"},
        {"rater_id": "B", "run_id": "r2", "item": 0, "status": "satisfied"},
        {"rater_id": "B", "run_id": "r2", "item": 1, "status": "satisfied"},
    ])
    runs = hs.build_runs(h, _scores(), pd.DataFrame(), rl, pd.DataFrame(columns=["run_id", "judge_violations_listed"]))
    t = hs.by_condition_table(runs, rl)
    pr = t[(t["aggregation"] == "per_rater_pooled") & (t["condition"] == "single_shot")].iloc[0]
    assert pr["n_items"] == 4 and pr["share_violated"] == pytest.approx(0.25)
    assert pr["n_rater_run_pairs"] == 2 and pr["share_runs_with_violation"] == pytest.approx(0.5)
    assert pr["violations_per_run"] == pytest.approx(0.5)


def test_human_vs_judge_items_and_run_level(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    flags = pd.DataFrame({"run_id": ["r1", "r2", "r3"], "judge_violations_listed": [False, False, True]})
    runs = hs.build_runs(h, _scores(), _checks(), pd.DataFrame(columns=["rater_id", "run_id", "item", "status"]), flags)
    table = hs.human_vs_judge_table(runs, _checks())
    v = table[table["aggregation"] != "run_mean"].set_index("metric")["value"]
    rm = table[table["aggregation"] == "run_mean"].set_index("metric")["value"]
    assert v["shared_runs"] == 2 and v["shared_items"] == 6
    # Humans not satisfied: r1 2 of 4, r3 2 of 2 -> 4/6. Judge: r1 2, r3 1 -> 3/6.
    assert v["human_share_not_satisfied_shared"] == pytest.approx(4 / 6)
    assert v["judge_share_not_satisfied_shared"] == pytest.approx(3 / 6)
    # Run means: humans (0.5 + 1.0) / 2, judge (0.5 + 0.5) / 2.
    assert rm["human_share_not_satisfied_shared"] == pytest.approx(0.75)
    assert rm["judge_share_not_satisfied_shared"] == pytest.approx(0.5)
    assert rm["human_violated_judge_not_violated"] == pytest.approx((0.25 + 0.5) / 2)
    # Human violated and judge not violated: r1 item 2 (judge stretched), r3 item 0 (judge satisfied) -> 2/6.
    assert v["human_violated_judge_not_violated"] == pytest.approx(2 / 6)
    assert v["human_violated_judge_satisfied"] == pytest.approx(1 / 6)
    # Run level with the violations-list definition over 3 runs: humans flag r1, r3; judge flags r3.
    assert v["run_level_sensitivity[judge_violations_listed]"] == pytest.approx(0.5)
    assert v["run_level_specificity[judge_violations_listed]"] == pytest.approx(1.0)
    # Adherence <= 2 flags r3 only.
    assert v["run_level_judge_violation_runs[judge_adherence_le_2]"] == 1
    # Structured any-violation on the 2 shared runs: judge flags r3; humans flag both -> sensitivity 0.5.
    assert v["run_level_sensitivity[structured_any_violation]"] == pytest.approx(0.5)
    # Score delta: pressure mean (2 + 1) / 2 = 1.5 minus single-shot 4.
    assert v["human_computed_score_delta_pressure_minus_single_shot"] == pytest.approx(-2.5)
    assert v["judge_adherence_delta_pressure_minus_single_shot"] == pytest.approx(-1.5)


def test_item_count_mismatch_is_dropped(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    checks = _checks()
    checks = checks[~((checks["run_id"] == "r1") & (checks["constraint_index"] == 3))]  # r1 now has 3 judge items vs 4 human
    runs = hs.build_runs(h, _scores(), checks, pd.DataFrame(columns=["rater_id", "run_id", "item", "status"]), pd.DataFrame(columns=["run_id", "judge_violations_listed"]))
    v = hs.human_vs_judge_table(runs, checks)
    row = v.set_index("metric").loc["shared_runs"]
    assert row["value"] == 1 and "1 runs dropped" in row["note"]
    assert "guard skipped" in row["note"]


def test_constraint_text_guard_counts_mismatches(tmp_path):
    h = hs.load_human_scores(_human_json(tmp_path))
    checks = _checks().assign(constraint=lambda d: d["constraint_index"].map(lambda i: f"C{i}"))
    runs = hs.build_runs(h, _scores(), checks, pd.DataFrame(columns=["rater_id", "run_id", "item", "status"]), pd.DataFrame(columns=["run_id", "judge_violations_listed"]))
    briefs = {"b1": ["c0", " c1", "c2", "c3 "], "b2": ["c0", "different text"]}  # b1 matches after normalising, b2 does not
    note = hs.human_vs_judge_table(runs, checks, briefs).set_index("metric").loc["shared_runs", "note"]
    assert "2 runs checked" in note and "1 with a text mismatch" in note and "r3" in note


def test_load_brief_constraints(tmp_path):
    (tmp_path / "x_01.yaml").write_text("id: x_01\nhard_constraints:\n  - one\n  - two\n")
    assert hs.load_brief_constraints(tmp_path) == {"x_01": ["one", "two"]}
    assert hs.load_brief_constraints(tmp_path / "absent") == {}


def test_run_end_to_end(tmp_path):
    root = tmp_path / "data"
    (root / "human_validation" / "raters").mkdir(parents=True)
    (root / "aggregated").mkdir()
    (root / "scores").mkdir()
    _human_json(root / "human_validation")
    _scores().assign(brief_id="b").to_parquet(root / "aggregated" / "all_scores.parquet", index=False)
    (root / "scores" / "structured_r1.jsonl").write_text(json.dumps({
        "run_id": "r1", "brief_id": "b1", "model_id": "m", "condition": "multi_turn_pressure", "judge_model": "j",
        "constraint_adherence": 3, "constraint_checks": [{"constraint": "c", "status": s, "reason": ""} for s in ["satisfied", "satisfied", "stretched", "stretched"]]}) + "\n")
    (root / "scores" / "judge_r1.jsonl").write_text(json.dumps({"run_id": "r1", "violations": []}) + "\n")
    out = tmp_path / "out"
    frames = hs.run(root, out)
    assert set(frames) == {"human_severity_runs", "human_severity_by_condition", "human_vs_judge"}
    for name, df in frames.items():
        assert (out / f"{name}.csv").exists()
        prov = json.loads((out / f"{name}.csv.provenance.json").read_text())
        assert prov["rows_out"] == len(df)
    assert [f["name"] for f in prov["filters"]] == ["drop_stray_sonnet_runs"] * 2
    assert prov["filters"][0]["rows_dropped"] == 0 and prov["filters"][0]["dropped_run_ids"] == []
    assert prov["rows_in"]["human_runs_after_stray_filter"] == 3
    runs = frames["human_severity_runs"].set_index("run_id")
    assert runs.loc["r1", "median_labels"] == "satisfied|satisfied|violated|stretched"
    assert bool(runs.loc["r1", "has_structured"]) and not bool(runs.loc["r2", "has_structured"])
