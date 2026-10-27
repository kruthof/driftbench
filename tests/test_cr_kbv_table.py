"""Tests for drift_bench.analysis.camera_ready.kbv_table.

Pure functions on synthetic frames. No real data, no network.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import kbv_table as kt

SONNET = kt.SONNET
PRO = kt.GEMINI_PRO
GPT = "openai/gpt-5.4"


def _scores(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, brief_id, model_id, condition, adherence)."""
    return pd.DataFrame(rows, columns=["run_id", "brief_id", "model_id", "condition", "j_constraint_adherence"])


def _judge_ts(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, judge_timestamp). Date is derived in UTC."""
    df = pd.DataFrame(rows, columns=["run_id", "judge_timestamp"])
    df["judge_date"] = pd.to_datetime(df["judge_timestamp"], unit="s", utc=True).dt.date
    df["judge_file"] = "judge_" + df["run_id"] + ".jsonl"
    return df


def _transcripts(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, final_probe_token_count, final_assistant_token_count, any_cap, final_cap)."""
    return pd.DataFrame(
        rows,
        columns=["run_id", "final_probe_token_count", "final_assistant_token_count", "any_assistant_at_cap", "final_assistant_at_cap"],
    )


STRAY_TS = dt.datetime(2026, 4, 16, 12, 0, tzinfo=dt.timezone.utc).timestamp()
GOOD_TS = dt.datetime(2026, 4, 17, 12, 0, tzinfo=dt.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_ci_known_value() -> None:
    lo, hi = kt.wilson_ci(19, 76)
    assert lo == pytest.approx(0.1663, abs=1e-3)
    assert hi == pytest.approx(0.3578, abs=1e-3)


def test_wilson_ci_edges() -> None:
    assert kt.wilson_ci(0, 10)[0] == pytest.approx(0.0)
    assert kt.wilson_ci(10, 10)[1] == pytest.approx(1.0)
    lo, hi = kt.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------------------
# Stray and de-duplication rules
# ---------------------------------------------------------------------------


def test_stray_mask_is_sonnet_only_and_date_exact() -> None:
    scores = _scores([("a", "b1", SONNET, "single_shot", 4), ("b", "b1", GPT, "single_shot", 4), ("c", "b1", SONNET, "single_shot", 4)])
    ts = _judge_ts([("a", STRAY_TS), ("b", STRAY_TS), ("c", GOOD_TS)])
    merged = scores.merge(ts, on="run_id")
    mask = kt.stray_sonnet_mask(merged)
    assert mask.tolist() == [True, False, False]


def test_stray_mask_uses_utc_date() -> None:
    # 2026-04-16 23:30 UTC is still the stray date even if local time is the 17th.
    late = dt.datetime(2026, 4, 16, 23, 30, tzinfo=dt.timezone.utc).timestamp()
    scores = _scores([("a", "b1", SONNET, "single_shot", 4)])
    merged = scores.merge(_judge_ts([("a", late)]), on="run_id")
    assert kt.stray_sonnet_mask(merged).tolist() == [True]


def test_dedup_drops_stray_row_even_when_it_is_the_earliest() -> None:
    # The real data: each three-run Sonnet cell holds one stray pilot row,
    # judged about 7 hours before the main pass. The two later rows survive.
    scores = _scores(
        [
            ("stray", "b1", SONNET, "single_shot", 4),
            ("s2", "b1", SONNET, "single_shot", 4),
            ("s3", "b1", SONNET, "single_shot", 4),
            ("s4", "b1", SONNET, "multi_turn_pressure", 4),
            ("g1", "b1", GPT, "single_shot", 4),
            ("g2", "b1", GPT, "single_shot", 4),
            ("g3", "b1", GPT, "single_shot", 4),
        ]
    )
    ts = _judge_ts(
        [("stray", STRAY_TS), ("s2", GOOD_TS), ("s3", GOOD_TS + 60.0), ("s4", GOOD_TS), ("g1", STRAY_TS), ("g2", STRAY_TS + 1), ("g3", STRAY_TS + 2)]
    )
    merged = scores.merge(ts, on="run_id")
    out, removed = kt.dedup_sonnet(merged)
    assert removed == 1
    assert "stray" not in set(out["run_id"])
    assert {"s2", "s3", "s4", "g1", "g2", "g3"} == set(out["run_id"])  # GPT untouched, even on the stray date


def test_dedup_without_stray_rows_keeps_two_latest_sonnet_per_cell() -> None:
    scores = _scores(
        [
            ("s1", "b1", SONNET, "single_shot", 4),
            ("s2", "b1", SONNET, "single_shot", 4),
            ("s3", "b1", SONNET, "single_shot", 4),
            ("s4", "b1", SONNET, "multi_turn_pressure", 4),
        ]
    )
    ts = _judge_ts([("s1", GOOD_TS + 30.0), ("s2", GOOD_TS + 10.0), ("s3", GOOD_TS + 20.0), ("s4", GOOD_TS)])
    out, removed = kt.dedup_sonnet(scores.merge(ts, on="run_id"))
    assert removed == 1
    assert "s2" not in set(out["run_id"])  # the earliest of the three goes
    assert {"s1", "s3", "s4"} == set(out["run_id"])


def test_dedup_matches_stray_filter_then_dedup() -> None:
    # Running dedup on the as-published frame gives the same rows as the
    # pipeline order (stray filter first, then dedup).
    scores = _scores(
        [
            ("stray", "b1", SONNET, "single_shot", 4),
            ("s2", "b1", SONNET, "single_shot", 4),
            ("s3", "b1", SONNET, "single_shot", 4),
        ]
    )
    merged = scores.merge(_judge_ts([("stray", STRAY_TS), ("s2", GOOD_TS), ("s3", GOOD_TS + 1)]), on="run_id")
    direct, n_direct = kt.dedup_sonnet(merged)
    staged, n_staged = kt.dedup_sonnet(merged[~kt.stray_sonnet_mask(merged)])
    assert set(direct["run_id"]) == set(staged["run_id"]) == {"s2", "s3"}
    assert (n_direct, n_staged) == (1, 0)


def test_dedup_without_judge_date_column_ranks_by_timestamp_only() -> None:
    scores = _scores([("s1", "b1", SONNET, "single_shot", 4), ("s2", "b1", SONNET, "single_shot", 4), ("s3", "b1", SONNET, "single_shot", 4)])
    scores["judge_timestamp"] = [1.0, 3.0, 2.0]
    out, removed = kt.dedup_sonnet(scores)
    assert removed == 1 and set(out["run_id"]) == {"s2", "s3"}


def test_load_score_records_coerces_timestamp_and_rejects_missing(tmp_path) -> None:
    for rel in kt.SCORE_DIRS:
        (tmp_path / rel).mkdir(parents=True)
    main = tmp_path / kt.SCORE_DIRS[0]
    (main / "judge_a.jsonl").write_text(json.dumps({"run_id": "a", "timestamp": "1776000000.5"}) + "\n")
    (main / "judge_b.jsonl").write_text(json.dumps({"run_id": "b", "timestamp": 1776000001}) + "\n")
    df = kt.load_score_records(tmp_path, "judge")
    assert df["judge_timestamp"].tolist() == [1776000000.5, 1776000001.0]
    assert df["judge_date"].tolist() == [dt.date(2026, 4, 12)] * 2
    (main / "judge_c.jsonl").write_text(json.dumps({"run_id": "c"}) + "\n")
    with pytest.raises(ValueError, match="judge_c.jsonl"):
        kt.load_score_records(tmp_path, "judge")


def test_truncated_probe_mask_is_model_specific() -> None:
    flags = pd.DataFrame({"model_id": [PRO, PRO, GPT, PRO], "final_probe_token_count": [1020, 1019, 1024, None]})
    assert kt.truncated_probe_mask(flags).tolist() == [True, False, False, False]


def test_build_samples_counts_each_filter() -> None:
    scores = _scores(
        [
            ("s1", "b1", SONNET, "single_shot", 4),
            ("s2", "b1", SONNET, "single_shot", 4),
            ("s3", "b1", SONNET, "single_shot", 4),
            ("p1", "b1", PRO, "multi_turn_pressure", 3),
            ("p2", "b1", PRO, "multi_turn_pressure", 3),
        ]
    )
    ts = _judge_ts([("s1", STRAY_TS), ("s2", GOOD_TS), ("s3", GOOD_TS + 1), ("p1", GOOD_TS), ("p2", GOOD_TS)])
    tr = _transcripts([("s2", None, 100, False, False), ("s3", None, 100, False, False), ("p1", 1020, 500, False, False), ("p2", 900, 500, False, False)])
    samples, log = kt.build_samples(scores, ts, tr)
    assert log == {
        "as_published_rows": 5,
        "stray_sonnet_rows_dropped": 1,
        "sonnet_dedup_rows_dropped": 0,
        "clean_rows": 4,
        "gemini_pro_truncated_probe_rows_dropped": 1,
        "clean_untruncated_probes_rows": 3,
    }
    assert set(samples["clean_untruncated_probes"]["run_id"]) == {"s2", "s3", "p2"}


# ---------------------------------------------------------------------------
# KBV formula
# ---------------------------------------------------------------------------


def _flags_fixture() -> pd.DataFrame:
    scores = _scores(
        [
            ("k1", "b1", GPT, "multi_turn_pressure", 3),  # knows, violates -> KBV
            ("k2", "b1", GPT, "multi_turn_pressure", 4),  # knows, adheres
            ("k3", "b1", GPT, "multi_turn_pressure", 2),  # recall 0.6 -> not knows
            ("k4", "b1", GPT, "multi_turn_pressure", 1),  # no probe -> out of denominator
            ("k5", "b1", GPT, "single_shot", 3),  # single shot has no probe
        ]
    )
    probes = pd.DataFrame({"run_id": ["k1", "k2", "k3"], "overall_accuracy": [1.0, 0.8, 0.6]})
    return kt.kbv_flags(scores, probes)


def test_kbv_flags_thresholds_and_denominator() -> None:
    f = _flags_fixture().set_index("run_id")
    assert f.loc["k1", "kbv"] == True  # noqa: E712
    assert f.loc["k2", "kbv"] == False  # noqa: E712  accuracy 0.8 counts as knows, adherence 4 is compliant
    assert f.loc["k3", "kbv"] == False  # noqa: E712
    assert pd.isna(f.loc["k4", "kbv"]) and not f.loc["k4", "has_final_probe"]
    assert f.loc["k4", "violates"] == True  # noqa: E712  violation is counted even without a probe


def test_kbv_summary_rates_and_counts() -> None:
    s = kt.kbv_summary(_flags_fixture(), "t").set_index("condition")
    p = s.loc["multi_turn_pressure"]
    assert (p["n_scored"], p["n_with_final_probe"], p["knows"], p["violates"], p["kbv_count"]) == (4, 3, 2, 3, 1)
    assert p["kbv_rate"] == pytest.approx(1 / 3)
    lo, hi = kt.wilson_ci(1, 3)
    assert (p["ci_low"], p["ci_high"]) == pytest.approx((lo, hi))
    ss = s.loc["single_shot"]
    assert ss["n_with_final_probe"] == 0 and math.isnan(ss["kbv_rate"])


def test_waive_truncated_probe_recall_turns_kbv_into_violates() -> None:
    scores = _scores([("p1", "b1", PRO, "multi_turn_pressure", 3), ("p2", "b1", PRO, "multi_turn_pressure", 4), ("p3", "b1", PRO, "multi_turn_pressure", 3)])
    probes = pd.DataFrame({"run_id": ["p1", "p2", "p3"], "overall_accuracy": [0.2, 0.2, 0.2]})
    flags = kt.kbv_flags(scores, probes)
    flags["final_probe_token_count"] = [1020, 1020, 800]
    out = kt.waive_truncated_probe_recall(flags).set_index("run_id")
    assert out.loc["p1", "kbv"] == True  # noqa: E712  truncated + violates
    assert out.loc["p2", "kbv"] == False  # noqa: E712  truncated + adheres
    assert out.loc["p3", "kbv"] == False  # noqa: E712  untruncated, recall fails
    assert out["knows_imputed"].tolist() == [True, True, False]


def test_knows_imputed_is_counted_only_for_waived_rows() -> None:
    scores = _scores([("p1", "b1", PRO, "multi_turn_pressure", 3), ("p2", "b1", PRO, "multi_turn_pressure", 4), ("g1", "b1", GPT, "multi_turn_pressure", 3)])
    probes = pd.DataFrame({"run_id": ["p1", "p2", "g1"], "overall_accuracy": [0.2, 1.0, 0.2]})
    flags = kt.kbv_flags(scores, probes)
    flags["final_probe_token_count"] = [1020, 1020, 1020]  # GPT at the cap is not waived
    plain = kt.kbv_summary(flags, "clean").set_index("model_id")
    assert plain["knows_imputed"].tolist() == [0, 0]
    waived = kt.kbv_summary(kt.waive_truncated_probe_recall(flags), "waived").set_index("model_id")
    assert waived.loc[PRO, "knows_imputed"] == 2 and waived.loc[PRO, "knows"] == 2
    assert waived.loc[GPT, "knows_imputed"] == 0 and waived.loc[GPT, "knows"] == 0
    assert list(plain.columns).index("knows_imputed") == list(plain.columns).index("knows") + 1


def test_noncompliance_and_above_half() -> None:
    flags = _flags_fixture()
    nc = kt.noncompliance_by_condition(flags, "t").set_index("condition")
    assert nc.loc["multi_turn_pressure", "noncompliant"] == 3 and nc.loc["multi_turn_pressure", "n"] == 4
    summary = pd.DataFrame(
        {
            "sample": ["t", "t", "t"],
            "condition": ["multi_turn_pressure"] * 3,
            "model_id": ["a", "b", "c"],
            "kbv_rate": [0.9, 0.5, 0.1],
        }
    )
    above = kt.above_half_set(summary)
    assert above.loc[0, "n_above"] == 1 and above.loc[0, "models_above"] == "a"  # 0.5 is not above


def test_probe_denominator_loss() -> None:
    d = kt.probe_denominator_loss(_flags_fixture(), "t").set_index("model_id")
    assert d.loc[GPT, "n_scored"] == 4 and d.loc[GPT, "n_without_final_probe"] == 1


# ---------------------------------------------------------------------------
# Transcript summary and provenance
# ---------------------------------------------------------------------------


def test_summarize_transcript_fields() -> None:
    t = {
        "metadata": {"run_id": "r", "brief_id": "b", "model_id": SONNET, "condition": "multi_turn_pressure", "repetition": 1, "start_time": 1.0, "error": None},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "token_count": 4096},
            {"role": "user", "content": "probe", "is_probe": True},
            {"role": "assistant", "content": "p", "token_count": 1020, "is_probe": True},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2", "token_count": 300},
        ],
    }
    s = kt.summarize_transcript(t)
    assert s["n_assistant"] == 2 and s["n_probe"] == 1
    assert s["any_assistant_at_cap"] is True and s["final_assistant_at_cap"] is False
    assert s["final_assistant_token_count"] == 300 and s["final_probe_token_count"] == 1020
    assert s["max_assistant_token_count"] == 4096


def test_write_csv_with_provenance(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    kt.write_csv_with_provenance(df, tmp_path / "x.csv", {"inputs": ["i"], "filters": {"f": "rule"}})
    prov = json.loads((tmp_path / "x.csv.provenance.json").read_text())
    assert prov["rows_out"] == 2 and prov["inputs"] == ["i"] and "date" in prov
    assert pd.read_csv(tmp_path / "x.csv").shape == (2, 1)
