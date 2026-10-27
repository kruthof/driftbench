"""Tests for drift_bench.analysis.camera_ready.rebuttal_reproduction (X16). Synthetic inputs only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import rebuttal_reproduction as rr


# ---------------------------------------------------------------- precision

@pytest.mark.parametrize("value,posted,expected", [
    (12.19, "12.2", True),
    (12.24, "12.2", True),
    (12.26, "12.2", False),
    (0.0394, ".039", True),
    (0.0395, ".039", False),     # half up: 0.040
    (-1.118, "-1.12", True),
    (-1.125, "-1.12", False),    # half up away from zero: -1.13
    (72.6, "73", True),
    (72.4, "73", False),
    (99.9, "100", True),
    (2.02e-08, "2.0e-08", True),
    (2.06e-08, "2.0e-08", False),
    (2.02e-07, "2.0e-08", False),
    (float("nan"), "12.2", False),
    (None, "12.2", False),
])
def test_matches_at_precision(value, posted, expected):
    assert rr.matches_at_precision(value, posted) is expected


def test_format_at_precision_keeps_posted_decimals():
    assert rr.format_at_precision(0.0394, ".039") == "0.039"
    assert rr.format_at_precision(25.0, "25") == "25"
    assert rr.format_at_precision(13.92, "13.9") == "13.9"
    assert rr.format_at_precision(2.024e-08, "2.0e-08") == "2.0e-08"
    assert rr.format_at_precision(None, "1.0") == ""


def test_sci_parts_rolls_over_mantissa():
    mant, exp = rr.sci_parts(9.96e-05, 1)
    assert (float(mant), exp) == (1.0, -4)
    with pytest.raises(ValueError):
        rr.sci_parts(0.0, 1)


# ---------------------------------------------------------------- formulas

def test_holm_adjust_step_down_and_monotone():
    adj = rr.holm_adjust([0.01, 0.04, 0.03, 0.20])
    # sorted p: .01 (x4), .03 (x3), .04 (x2), .20 (x1), running max
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert rr.holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_run_mean_differs_from_pooled():
    # Run a: 1 of 2 items flagged; run b: 0 of 8.
    df = pd.DataFrame({"run_id": ["a"] * 2 + ["b"] * 8, "flag": [True, False] + [False] * 8})
    assert rr.run_mean_share(df, "flag") == pytest.approx(0.25)
    assert rr.pooled_share(df, "flag") == pytest.approx(0.10)


def test_brief_mean_deltas_averages_repetitions_and_drops_unpaired():
    runs = pd.DataFrame({
        "brief_id": ["b1", "b1", "b1", "b2", "b3"],
        "condition": [rr.SS, rr.MTP, rr.MTP, rr.SS, rr.MTP],
        "g": [0.0, 0.2, 0.4, 0.1, 0.5],
    })
    m = rr.brief_mean_deltas(runs, "g")
    assert list(m.index) == ["b1"]
    assert m.loc["b1", rr.MTP] == pytest.approx(0.3)


def test_kbv_flag_needs_knowing_and_violating():
    acc = pd.Series([0.8, 0.79, 1.0, np.nan])
    adh = pd.Series([3, 3, 4, 2])
    assert rr.kbv_flag(acc, adh).tolist() == [True, False, False, False]


# ---------------------------------------------------------------- rows

def test_make_row_count_must_match_numerator_and_denominator():
    ok = rr.make_row("K", "i", "p", "36.6% (15/41)", "36.6", "d", "s", 100 * 15 / 41,
                     numerator=15, denominator=41, posted_numerator=15, posted_denominator=41)
    assert ok["matches_at_posted_precision"] == "yes"
    assert ok["reproduced_value"] == "36.6"
    bad = rr.make_row("K", "i", "p", "7.4% of 27 runs", "7.4", "d", "s", 100 * 2 / 27,
                      numerator=2, denominator=26, posted_denominator=27)
    assert bad["matches_at_posted_precision"] == "no"


def test_make_row_scientific_value_gets_mantissa_and_exponent():
    row = rr.make_row("P", "i", "p", "p = 2.0e-08", "2.0e-08", "d", "s", 2.0242e-08)
    assert row["matches_at_posted_precision"] == "yes"
    assert (row["value_mantissa"], row["value_exponent"]) == (2.0, -8)


def test_lost_row_has_no_value_and_says_why():
    row = rr.lost_row("L", "i", "p", "0.557", "offset", "files lost")
    assert row["matches_at_posted_precision"] == "no"
    assert row["basis"] == rr.LOST
    assert math.isnan(row["value"]) and row["reproduced_value"] == ""
    assert row["note"] == "files lost"
    assert set(row) == set(rr.COLUMNS)


# ---------------------------------------------------------------- macros

def test_rebuttal_macros_require_a_match():
    from drift_bench.analysis.camera_ready import macros

    specs = [s for s in macros.static_specs() if s.csv == rr.CSV_NAME]
    assert specs
    for s in specs:
        assert s.name.startswith("\\crRebut")
        assert s.required is False
        assert s.where.get("matches_at_posted_precision") == "yes"
        assert len(s.name) < 60


def test_rebuttal_macro_missing_when_row_does_not_match(tmp_path):
    from drift_bench.analysis.camera_ready import macros

    rows = [rr.make_row("GkbvGpt", "i", "p", ".039", ".039", "d", "s", 0.0394, denominator=76),
            rr.make_row("GkbvMini", "i", "p", ".029", ".029", "d", "s", 0.0310, denominator=76)]
    pd.DataFrame(rows, columns=rr.COLUMNS).to_csv(tmp_path / rr.CSV_NAME, index=False)
    specs = [s for s in macros.static_specs() if s.csv == rr.CSV_NAME and "Gkbv" in s.name]
    emitted, missing = macros.resolve_all(specs, tmp_path)
    names = {e["name"] for e in emitted}
    assert "\\crRebutGkbvGpt" in names
    assert "\\crRebutGkbvMini" not in names
    assert any(m["name"] == "\\crRebutGkbvMini" for m in missing)


def test_probe_e7_rows_fails_loudly_without_inputs(tmp_path):
    data_root, out_dir = tmp_path / "data", tmp_path / "out"
    (data_root / "analysis").mkdir(parents=True)
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="e7_summary"):
        rr.probe_e7_rows(data_root, out_dir)
    pd.DataFrame({"run_id": ["r1"]}).to_csv(out_dir / "e7_strict_probe_runs.csv", index=False)
    with pytest.raises(FileNotFoundError):
        rr.probe_e7_rows(data_root, out_dir)


def test_probe_e7_rows_turn_one_mean(tmp_path):
    data_root, out_dir = tmp_path / "data", tmp_path / "out"
    (data_root / "analysis").mkdir(parents=True)
    out_dir.mkdir()
    pd.DataFrame({"run_id": ["r1", "r2"]}).to_csv(out_dir / "e7_strict_probe_runs.csv", index=False)
    pd.DataFrame({"run_id": ["r1", "r1", "r2", "r2", "r3"], "turn_index": [1, 3, 1, 3, 1],
                  "overall_accuracy": [0.9, 0.1, 1.0, 0.1, 0.0]}).to_csv(data_root / "analysis" / "probe_accuracy.csv", index=False)
    row = rr.probe_e7_rows(data_root, out_dir)[0]
    assert row["key"] == "SubstringTurnOne"
    assert row["value"] == pytest.approx(95.0)
    assert row["denominator"] == 2


def test_post_labels_name_no_reviewer():
    for label in (rr.R1, rr.R2, rr.R3, rr.REPLY):
        assert label.startswith(("rebuttal", "discussion reply"))
