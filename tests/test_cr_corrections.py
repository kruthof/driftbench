"""Tests for drift_bench.analysis.camera_ready.corrections and failed_runs.

Pure functions on synthetic frames. No real data, no network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import corrections as cr
from drift_bench.analysis.camera_ready import failed_runs as fr

SONNET = cr.SONNET
GPT = "openai/gpt-5.4"
PRESSURE = cr.PRESSURE
CHECKPOINT = cr.CHECKPOINT


# ---------------------------------------------------------------------------
# B1 and C15: drift events
# ---------------------------------------------------------------------------


def test_first_event_turn_ignores_bad_entries() -> None:
    assert cr.first_event_turn([{"turn": 3}, {"turn": 2}, {"turn": "x"}, "junk", {"turn": True}]) == 2
    assert cr.first_event_turn([]) is None
    assert cr.first_event_turn(None) is None


def _auditor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d", "e"],
            "condition": [PRESSURE] * 4 + ["single_shot"],
            "model_id": [GPT] * 5,
            "drift_classification": ["trajectory_drift", "trajectory_lock_in", "mild_drift", "trajectory_drift", "trajectory_drift"],
            "drift_events": [
                [{"turn": 2}, {"turn": 4}, {"turn": 5}],
                [{"turn": 3}, {"turn": 4}, {"turn": 5}],
                [{"turn": 1}, {"turn": 2}, {"turn": 3}],
                [{"turn": 4}, {"turn": 5}],
                [{"turn": 1}, {"turn": 2}, {"turn": 3}],
            ],
        }
    )


def _scores_for_events() -> pd.DataFrame:
    return pd.DataFrame({"run_id": ["a", "b", "c", "d", "e"], "condition": [PRESSURE] * 4 + ["single_shot"]})


def test_drift_onset_shares_use_pressure_drift_classes_only() -> None:
    events = cr.auditor_event_table(_auditor())
    rows = cr.drift_onset_shares(events, _scores_for_events())
    by2, by3 = rows
    # drift cases: a (turn 2), b (turn 3), d (turn 4); c is mild, e is single-shot
    assert (by2["numerator"], by2["denominator"]) == (1, 3)
    assert (by3["numerator"], by3["denominator"]) == (2, 3)
    assert by2["recomputed_value"] == pytest.approx(33.3, abs=0.1)


def test_exact_three_events_over_pressure_rows() -> None:
    events = cr.auditor_event_table(_auditor())
    row = cr.exact_three_events(events, _scores_for_events())
    assert (row["numerator"], row["denominator"]) == (3, 4)


# ---------------------------------------------------------------------------
# B3: checkpoint recovery under three truncation rules
# ---------------------------------------------------------------------------


def test_sonnet_checkpoint_recovery_rules() -> None:
    flags = pd.DataFrame(
        {
            "run_id": list("abcdef"),
            "model_id": [SONNET] * 5 + [GPT],
            "condition": [PRESSURE, PRESSURE, CHECKPOINT, CHECKPOINT, CHECKPOINT, CHECKPOINT],
            "has_final_probe": [True, True, True, True, False, True],
            "kbv": [True, True, True, False, None, True],
            "final_assistant_at_cap": [True, False, True, False, False, False],
            "any_assistant_at_cap": [True, True, True, False, False, False],
        }
    )
    rows = cr.sonnet_checkpoint_recovery(flags, "t")
    got = {(r["quantity"].split(", ")[1], r["quantity"].split(", ")[2]): (r["numerator"], r["denominator"]) for r in rows}
    assert got[(PRESSURE, "all runs")] == (2, 2)
    assert got[(CHECKPOINT, "all runs")] == (1, 2)  # e has no probe, f is GPT
    assert got[(PRESSURE, "untruncated final (final assistant token_count < cap)")] == (1, 1)
    assert got[(CHECKPOINT, "untruncated final (final assistant token_count < cap)")] == (0, 1)
    assert got[(PRESSURE, "untruncated any turn (no assistant token_count == cap)")] == (0, 0)


# ---------------------------------------------------------------------------
# B8, B9, B17: human validation helpers
# ---------------------------------------------------------------------------


def test_rater_violation_counts(tmp_path) -> None:
    (tmp_path / "human_rater_set1_rater_A.json").write_text(
        json.dumps({"r1": {"rater_id": "rater_A", "violated_count": 2, "stretched_count": 1}, "r2": {"rater_id": "rater_A", "violated_count": 3, "stretched_count": 0}})
    )
    df = cr.rater_violation_counts(tmp_path)
    assert df.loc[0, ["rater_id", "n_runs", "violated", "stretched"]].tolist() == ["rater_A", 2, 5, 1]


def test_human_condition_means_and_delta() -> None:
    human = pd.DataFrame({"condition": ["single_shot", "single_shot", PRESSURE, "multi_turn_neutral"], "computed_score": [4, 3, 2, 1]})
    m = cr.human_condition_means(human, "computed_score")
    assert m["single_shot"] == 3.5 and m["pressure"] == 2.0 and m["delta"] == -1.5
    assert (m["n_ss"], m["n_pr"]) == (2, 1)


def test_human_label_counts_case_insensitive() -> None:
    human = pd.DataFrame({"constraint_statuses": [["Satisfied", "Violated"], ["stretched", "Violated"]]})
    c = cr.human_label_counts(human)
    assert (c["satisfied"], c["stretched"], c["violated"]) == (1, 1, 2)


# ---------------------------------------------------------------------------
# B16: truncation
# ---------------------------------------------------------------------------


def _transcripts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d", "e"],
            "brief_id": ["b1", "b1", "b1", "b2", "b1"],
            "model_id": [SONNET, SONNET, SONNET, SONNET, GPT],
            "condition": [PRESSURE, PRESSURE, "single_shot", "multi_turn_pressure_rigor", PRESSURE],
            "repetition": [0, 0, 0, 0, 0],
            "any_assistant_at_cap": [True, False, False, True, True],
            "final_assistant_at_cap": [False, False, False, True, True],
            "n_assistant": [6, 6, 1, 6, 6],
            "error": [None, None, None, None, None],
        }
    )


def test_truncation_by_condition_restricts_to_sonnet_main_conditions() -> None:
    tb = cr.truncation_by_condition(_transcripts()).set_index("condition")
    assert tb.loc["all_main", ["n", "truncated_any_turn", "truncated_final"]].tolist() == [3, 1, 0]
    assert "multi_turn_pressure_rigor" not in tb.index


def test_truncation_adherence_test_picks_column() -> None:
    flags = pd.DataFrame(
        {
            "model_id": [SONNET] * 4,
            "condition": [PRESSURE] * 4,
            "j_constraint_adherence": [2, 2, 4, 4],
            "any_assistant_at_cap": [True, True, True, False],
            "final_assistant_at_cap": [True, True, False, False],
        }
    )
    t_any = cr.truncation_adherence_test(flags, column="any_assistant_at_cap")
    t_fin = cr.truncation_adherence_test(flags, column="final_assistant_at_cap")
    assert (t_any["n_truncated"], t_any["n_untruncated"]) == (3, 1)
    assert (t_fin["n_truncated"], t_fin["n_untruncated"]) == (2, 2)
    assert t_fin["mean_truncated"] == 2.0 and t_fin["mean_untruncated"] == 4.0
    assert 0 <= t_fin["p"] <= 1


# ---------------------------------------------------------------------------
# B17: structured checks, B14: cells
# ---------------------------------------------------------------------------


def test_structured_status_counts(tmp_path) -> None:
    rec = {"constraint_checks": [{"status": "satisfied"}, {"status": "Stretched"}, {"status": "violated"}, {"status": "stretched"}]}
    (tmp_path / "structured_x.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "judge_x.jsonl").write_text(json.dumps({"constraint_checks": [{"status": "violated"}]}) + "\n")
    c = cr.structured_status_counts(tmp_path)
    assert (c["satisfied"], c["stretched"], c["violated"]) == (1, 2, 1)


def test_sonnet_cell_counts() -> None:
    scores = pd.DataFrame(
        {
            "run_id": list("abcdef"),
            "brief_id": ["b1", "b1", "b1", "b2", "b2", "b1"],
            "model_id": [SONNET] * 5 + [GPT],
            "condition": ["single_shot"] * 3 + [PRESSURE] * 2 + ["single_shot"],
        }
    )
    tr = _transcripts()
    cc = cr.sonnet_cell_counts(scores, tr)
    assert cc["cells_with_3_runs"] == 1 and cc["n_cells"] == 2
    assert cc["runs_per_condition"] == {"single_shot": 3, PRESSURE: 2}
    assert cc["transcript_cells_with_duplicate_repetition"] == 1  # a and b share (b1, pressure, rep 0)


# ---------------------------------------------------------------------------
# B2: Gemini Pro probe cap
# ---------------------------------------------------------------------------

PRO = cr.GEMINI_PRO


def _gemini_flags() -> pd.DataFrame:
    # p1..p4 pressure: p1 capped+violates, p2 capped+adheres, p3 untruncated
    # full recall+violates, p4 untruncated low recall+violates; p5 neutral at
    # 1019 tokens; g1 GPT at the cap (never waived).
    scores = pd.DataFrame(
        {
            "run_id": ["p1", "p2", "p3", "p4", "p5", "g1"],
            "brief_id": ["b1"] * 6,
            "model_id": [PRO] * 5 + [GPT],
            "condition": [PRESSURE] * 4 + ["multi_turn_neutral", PRESSURE],
            "j_constraint_adherence": [2, 4, 2, 2, 2, 2],
        }
    )
    probes = pd.DataFrame({"run_id": scores["run_id"], "overall_accuracy": [0.5, 0.5, 1.0, 0.5, 1.0, 0.5]})
    flags = cr.kbv_flags(scores, probes)
    flags["final_probe_token_count"] = [1020, 1024, 900, 800, 1019, 1020]
    return flags


def test_gemini_pro_kbv_variants() -> None:
    v = cr.gemini_pro_kbv_variants(_gemini_flags())
    assert v["n_capped"] == 2
    assert v["waived"][:2] == (2, 4)  # p1 (waived, violates) + p3
    assert v["as_is"][:2] == (1, 4)  # p3 only
    assert v["excluded"][:2] == (1, 2)  # p3 of {p3, p4}
    lo, hi = cr.wilson_ci(1, 2)
    assert v["excluded"][2:] == pytest.approx((lo, hi))


def test_gemini_pro_probe_truncation_counts_one_below_cap() -> None:
    t = cr.gemini_pro_probe_truncation(_gemini_flags())
    assert t == {"n_probes": 5, "n_capped": 2, "n_one_below_cap": 1, "n_untruncated": 3, "n_untruncated_full_recall": 2}


def test_ledger_entry_has_note_column() -> None:
    e = cr._entry("X", "q", None, 1, "r", note="say this")
    assert list(e) == cr.LEDGER_COLUMNS and e["note"] == "say this"
    assert cr._entry("X", "q", None, 1, "r")["note"] == ""


# ---------------------------------------------------------------------------
# failed_runs
# ---------------------------------------------------------------------------


def test_split_main_conditions_sets_rigor_runs_aside() -> None:
    main, foreign = fr.split_main_conditions(_transcripts())
    assert set(main["run_id"]) == {"a", "b", "c", "e"}
    assert foreign["condition"].tolist() == ["multi_turn_pressure_rigor"]


def test_rows_in_from_summary() -> None:
    summary = pd.DataFrame({"quantity": ["rows_in:transcripts", "design_rows", "rows_in:judge_files"], "value": [5, 8, 4], "detail": [""] * 3})
    assert fr.rows_in_from_summary(summary) == {"transcripts": 5, "judge_files": 4}


def test_design_shortfall_counts_missing_reps() -> None:
    tr = pd.DataFrame({"brief_id": ["b1", "b1", "b2"], "condition": ["single_shot"] * 3, "model_id": [GPT] * 3})
    rows = fr.design_shortfall(tr, models=[GPT], briefs=["b1", "b2", "b3"], conditions=["single_shot"], reps=2)
    detail = sorted((r["brief_id"], r["detail"]) for r in rows)
    assert detail == [("b2", "cell has 1 transcripts, design intends 2"), ("b3", "cell has 0 transcripts, design intends 2"), ("b3", "cell has 0 transcripts, design intends 2")]
    assert all(r["run_id"] == "" for r in rows)


def test_short_transcripts_uses_condition_requirement() -> None:
    tr = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "brief_id": ["b1"] * 3,
            "model_id": [GPT] * 3,
            "condition": ["checkpointed_pressure", "checkpointed_pressure", "single_shot"],
            "n_assistant": [8, 6, 1],
        }
    )
    rows = fr.short_transcripts(tr)
    assert [r["run_id"] for r in rows] == ["b"]
    assert "needs 8" in rows[0]["detail"]


def test_metadata_errors_and_null_scores() -> None:
    tr = pd.DataFrame({"run_id": ["a", "b"], "brief_id": ["b1"] * 2, "model_id": [GPT] * 2, "condition": [PRESSURE] * 2, "error": [None, "timeout"]})
    assert [r["run_id"] for r in fr.metadata_errors(tr)] == ["b"]
    scores = pd.DataFrame({"run_id": ["a", "b"], "brief_id": ["b1"] * 2, "model_id": [GPT] * 2, "condition": [PRESSURE] * 2, "j_constraint_adherence": [4, None], "a_recoverability": [3, 3]})
    rows = fr.null_scores(scores)
    assert len(rows) == 1 and rows[0]["detail"] == "j_constraint_adherence"


def test_load_run_summaries(tmp_path) -> None:
    (tmp_path / "run_summary.json").write_text(json.dumps({"total_runs": 3, "failed_runs": 1, "conditions": ["single_shot"], "models": ["m"]}))
    (tmp_path / "monitored").mkdir()
    (tmp_path / "monitored" / "run_summary.json").write_text(json.dumps({"total_runs": 2, "failed_runs": 0}))
    df = fr.load_run_summaries(tmp_path)
    assert df["file"].tolist() == ["run_summary.json", "monitored/run_summary.json"]
    assert df["failed_runs"].tolist() == [1, 0]
