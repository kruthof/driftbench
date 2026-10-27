"""Tests for drift_bench.analysis.camera_ready.auditor_onset.

Synthetic auditor records and a synthetic parquet. No network, no API.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import auditor_onset as ao


def _record(run_id, condition, cls, turns, model="openai/gpt-5.4"):
    return {
        "run_id": run_id, "brief_id": "b1", "model_id": model,
        "condition": condition, "drift_classification": cls,
        "drift_events": [{"turn": t, "description": "d"} for t in turns],
    }


def test_event_turns_keeps_integers_and_drops_junk():
    rec = {"drift_events": [
        {"turn": 3, "description": "x"},
        {"turn": "2", "description": "y"},
        {"turn": None, "description": "z"},
        {"turn": "later", "description": "w"},
        {"turn": True, "description": "v"},
        "not a dict",
    ]}
    assert ao.event_turns(rec) == [3, 2]
    assert ao.event_turns({"drift_events": None}) == []
    assert ao.event_turns({}) == []


def test_summarise_record_first_flag_and_thresholds():
    s = ao.summarise_record(_record("r", "multi_turn_pressure", "trajectory_drift", [4, 2, 5]))
    assert s["n_events"] == 3
    assert s["first_flag_turn"] == 2
    assert s["last_flag_turn"] == 5
    assert s["flag_at_turn_1"] == 0
    assert s["flag_by_turn_2"] == 1
    assert s["flag_by_turn_3"] == 1
    assert s["event_turns"] == "4;2;5"
    assert s["is_drift_case"] == 1


def test_summarise_record_without_events():
    s = ao.summarise_record(_record("r", "single_shot", "no_drift", []))
    assert s["n_events"] == 0
    assert s["first_flag_turn"] is None
    assert s["flag_by_turn_2"] is None
    assert s["is_drift_case"] == 0


def test_build_summary_shares():
    recs = [
        _record("a", "multi_turn_pressure", "trajectory_drift", [1, 3, 4]),
        _record("b", "multi_turn_pressure", "trajectory_drift", [2, 3, 4]),
        _record("c", "multi_turn_pressure", "trajectory_lock_in", [3, 4, 5]),
        _record("d", "multi_turn_pressure", "mild_drift", [5, 6, 6]),
        _record("e", "multi_turn_neutral", "no_drift", []),
        _record("f", "multi_turn_neutral", "mild_drift", [2, 2]),
    ]
    runs = pd.DataFrame([ao.summarise_record(r) for r in recs])
    s = ao.build_summary(runs)

    ev = s[(s["section"] == "events_per_run") & (s["condition"] == "multi_turn_pressure")
           & (s["statistic"] == "n_events == 3 (share)")].iloc[0]
    assert ev["share"] == 1.0 and ev["n"] == 4 and ev["denominator"] == 4

    ev_n = s[(s["section"] == "events_per_run") & (s["condition"] == "multi_turn_neutral")
             & (s["statistic"] == "n_events == 0")].iloc[0]
    assert ev_n["n"] == 1 and ev_n["share"] == pytest.approx(0.5)

    ff = s[(s["section"] == "first_flag_by_condition")
           & (s["condition"] == "multi_turn_pressure")].iloc[0]
    assert ff["n"] == 4
    assert ff["share_at_turn_1"] == pytest.approx(1 / 4)
    assert ff["share_by_turn_2"] == pytest.approx(2 / 4)
    assert ff["share_by_turn_3"] == pytest.approx(3 / 4)

    # Neutral: only the run with events counts.
    ffn = s[(s["section"] == "first_flag_by_condition")
            & (s["condition"] == "multi_turn_neutral")].iloc[0]
    assert ffn["n"] == 1 and ffn["share_by_turn_2"] == 1.0

    # Drift cases: pressure runs classified trajectory_drift or lock_in.
    dc = s[s["section"] == "first_flag_pressure_drift_cases"].iloc[0]
    assert dc["n"] == 3
    assert dc["n_by_turn_2"] == 2
    assert dc["share_by_turn_2"] == pytest.approx(2 / 3)
    assert dc["share_by_turn_3"] == pytest.approx(1.0)

    td = s[s["section"] == "first_flag_pressure_trajectory_drift"].iloc[0]
    assert td["n"] == 2

    mix = s[(s["section"] == "classification_mix") & (s["condition"] == "multi_turn_pressure")]
    assert dict(zip(mix["statistic"], mix["n"])) == {
        "trajectory_drift": 2, "trajectory_lock_in": 1, "mild_drift": 1}

    turns = s[(s["section"] == "event_turn_mix") & (s["condition"] == "multi_turn_pressure")]
    assert turns["n"].sum() == 12
    assert turns["denominator"].iloc[0] == 12


def test_run_filters_to_parquet_and_writes_provenance(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "scores").mkdir(parents=True)
    (data_root / "openweight_subjects" / "scores").mkdir(parents=True)
    (data_root / "aggregated").mkdir()
    recs = {
        "in1": _record("in1", "multi_turn_pressure", "trajectory_drift", [2, 3, 4]),
        "in2": _record("in2", "single_shot", "no_drift", [],
                       model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "out": _record("out", "multi_turn_pressure", "trajectory_drift", [1, 1, 1]),
    }
    for rid in ("in1", "out"):
        (data_root / "scores" / f"auditor_{rid}.jsonl").write_text(
            json.dumps(recs[rid]) + "\n")
    (data_root / "openweight_subjects" / "scores" / "auditor_in2.jsonl").write_text(
        json.dumps(recs["in2"]) + "\n")
    pd.DataFrame({"run_id": ["in1", "in2"]}).to_parquet(
        data_root / "aggregated" / "all_scores.parquet")

    out_dir = tmp_path / "out"
    tables = ao.run(data_root, out_dir)
    runs = tables["auditor_onset_runs"]
    assert set(runs["run_id"]) == {"in1", "in2"}
    assert (out_dir / "auditor_onset_runs.csv").exists()
    prov = json.loads((out_dir / "auditor_onset_runs.csv.provenance.json").read_text())
    assert prov["rows_written"] == 2
    assert prov["filters"][-1]["rows_in"] == 3
    assert prov["filters"][-1]["rows_out"] == 2
    assert "date" in prov
    assert (out_dir / "auditor_onset_summary.csv.provenance.json").exists()
