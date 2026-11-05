"""Tests for drift_bench.analysis.camera_ready.macros.

Synthetic CSVs in tmp_path. Every format, every failure mode, LaTeX name
validity, the None/nan guard, and check_against_tex on a small tex string.
No real data, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import macros as m

NAME_RE = re.compile(r"^\\cr[A-Za-z]+$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    d = tmp_path / "csv"
    _write(d / "rates.csv", [
        {"sample": "clean", "model": "a", "rate": 0.25, "lo": 0.166, "hi": 0.358, "k": 19, "n": 76, "p": 0.0004, "txt": "15 to 86"},
        {"sample": "clean", "model": "b", "rate": 0.5, "lo": 0.4, "hi": 0.6, "k": 38, "n": 76, "p": 0.0510, "txt": "x_y"},
        {"sample": "pub", "model": "a", "rate": 0.3, "lo": 0.2, "hi": 0.4, "k": 30, "n": 100, "p": 0.2, "txt": ""},
        {"sample": "pub", "model": "b", "rate": "", "lo": "", "hi": "", "k": 2116, "n": 2146, "p": "", "txt": "9393"},
    ])
    _write(d / "corrections_ledger.csv", [
        {"id": "B1", "quantity": "share by turn 2", "published_value": "74%", "recomputed_value": 72.6, "numerator": 239, "denominator": 329, "rule": "r"},
        {"id": "B16", "quantity": "runs truncated, all_main", "published_value": "226 of 315", "recomputed_value": 226, "numerator": 127, "denominator": 315, "rule": "r"},
        {"id": "B8", "quantity": "per-rater min to max", "published_value": "", "recomputed_value": "15 to 86", "numerator": "", "denominator": "", "rule": "r"},
    ])
    return d


def spec(name: str, where: dict, column, fmt: str, **kw) -> m.MacroSpec:
    return m.MacroSpec(name, "rates.csv", where, column, fmt, **kw)


A = {"sample": "clean", "model": "a"}
B = {"sample": "clean", "model": "b"}
PUB_B = {"sample": "pub", "model": "b"}


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, column, where, expected",
    [
        ("int", "k", PUB_B, "2{,}116"),
        ("int", "k", A, "19"),
        ("round", "rate", A, "0"),
        ("pct0", "rate", A, "25"),
        ("pct1", "rate", A, "25.0"),
        ("dec1", "rate", A, "0.3"),
        ("dec2", "lo", A, "0.17"),
        ("dec3", "lo", A, "0.166"),
        ("pval", "p", A, "<0.001"),
        ("pval", "p", B, "0.051"),
        ("pval", "p", {"sample": "pub", "model": "a"}, "0.200"),
        ("ci", ("lo", "hi"), A, "[16.6, 35.8]"),
        ("cidec2", ("lo", "hi"), A, "[0.17, 0.36]"),
        ("cidec3", ("lo", "hi"), A, "[0.166, 0.358]"),
        ("frac", ("k", "n"), A, "19/76"),
        ("frac", ("k", "n"), PUB_B, "2{,}116/2{,}146"),
        ("auto", "txt", A, "15 to 86"),
        ("auto", "txt", B, r"x\_y"),
        ("auto", "txt", PUB_B, "9{,}393"),
        ("auto", "rate", A, "0.25"),
        ("auto", "k", A, "19"),
    ],
)
def test_every_format(csv_dir: Path, fmt, column, where, expected) -> None:
    value, _ = m.resolve(spec("\\crX", where, column, fmt), csv_dir, {})
    assert value == expected


def test_int_rejects_non_integer() -> None:
    with pytest.raises(m.MacroError, match="not an integer"):
        m.fmt_int("0.25")


def test_int_accepts_float_text_that_is_integral() -> None:
    assert m.fmt_int("239.0") == "239"
    assert m.fmt_int(2128.0) == "2{,}128"


def test_pval_edges() -> None:
    assert m.fmt_pval(0.00049) == "<0.001"
    assert m.fmt_pval(0.0005) == "<0.001"
    assert m.fmt_pval("0.00099") == "<0.001"
    assert m.fmt_pval(0.001) == "0.001"
    assert m.fmt_pval("0.0010") == "0.001"
    assert m.fmt_pval(0.0865) == "0.087"
    assert m.fmt_pval(0.0510) == "0.051"
    assert m.fmt_pval(1.0) == "1.000"
    with pytest.raises(m.MacroError, match="above 1"):
        m.fmt_pval(1.0001)
    with pytest.raises(m.MacroError, match="negative"):
        m.fmt_pval(-0.1)
    assert m.fmt_pval_tex("<0.001") == "$p < 0.001$"
    assert m.fmt_pval_tex("0.086") == "$p = 0.086$"
    assert m.fmt_pval_val("<0.001") == "$<$0.001"
    assert m.fmt_pval_val("0.086") == "0.086"


def test_round_and_pct_use_half_up_on_the_decimal_text() -> None:
    assert m.fmt_round("2.5") == "3"
    assert m.fmt_round("0.5") == "1"
    assert m.fmt_round("8048.3278") == "8{,}048"
    assert m.fmt_pct("0.0565", 1) == "5.7"   # float 0.0565 * 100 = 5.6499999... would give 5.6
    assert m.fmt_pct("0.145", 1) == "14.5"
    assert m.fmt_pct("0.25", 0) == "25"
    assert m.fmt_dec("2.675", 2) == "2.68"
    assert m.fmt_ci("0.1665", "0.3585", 100, 1) == "[16.7, 35.9]"


def test_text_cells_with_innocent_words_are_kept(csv_dir: Path, tmp_path: Path) -> None:
    _write(csv_dir / "words.csv", [
        {"k": "a", "val": "maintenance budget"},
        {"k": "b", "val": "Finance and nonempty"},
        {"k": "c", "val": "x < y | z > w"},
    ])
    specs = [m.MacroSpec("\\crWord" + k.upper(), "words.csv", {"k": k}, "val", "auto") for k in "abc"]
    tables = m.run(tmp_path, tmp_path / "out", csv_dir=csv_dir, specs=specs)
    values = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    assert values["\\crWordA"] == "maintenance budget"
    assert values["\\crWordB"] == "Finance and nonempty"
    assert values["\\crWordC"] == r"x \textless{} y \textbar{} z \textgreater{} w"


def test_pval_macros_are_self_contained_and_have_val_twins(csv_dir: Path, tmp_path: Path) -> None:
    specs = [spec("\\crSmallP", A, "p", "pval"), spec("\\crBigP", B, "p", "pval")]
    out = tmp_path / "out"
    tables = m.run(tmp_path, out, csv_dir=csv_dir, specs=specs)
    values = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    assert values == {
        "\\crSmallP": "$p < 0.001$",
        "\\crSmallPVal": "$<$0.001",
        "\\crBigP": "$p = 0.051$",
        "\\crBigPVal": "0.051",
    }
    tex = (out / "cr_macros.tex").read_text()
    body = "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))
    assert "<" not in m.MATH_RE.sub("", body)
    payload = json.loads((out / "cr_macros.json").read_text())
    twins = [r for r in payload["macros"] if r["variant"] == "Val"]
    assert [r["name"] for r in twins] == ["\\crSmallPVal", "\\crBigPVal"]
    assert twins[0]["raw"] == "0.0004"


def test_val_twin_name_collision_is_rejected() -> None:
    specs = [spec("\\crP", A, "p", "pval"), spec("\\crPVal", A, "p", "dec3")]
    with pytest.raises(m.MacroError, match="Val twin"):
        m.check_unique_names(specs)


def test_default_fills_an_empty_selection(csv_dir: Path, tmp_path: Path) -> None:
    specs = [
        spec("\\crZero", {"sample": "clean", "model": "zzz"}, "rate", "pct1", required=False, default="0.0"),
        spec("\\crReal", A, "rate", "pct1", default="0.0"),
    ]
    out = tmp_path / "out"
    tables = m.run(tmp_path, out, csv_dir=csv_dir, specs=specs)
    values = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    assert values == {"\\crZero": "0.0", "\\crReal": "25.0"}
    assert tables["cr_macros_missing"].empty
    payload = json.loads((out / "cr_macros.json").read_text())
    zero = next(r for r in payload["macros"] if r["name"] == "\\crZero")
    assert zero["raw"] == "0.0" and zero["default"] == "0.0"
    # A default never masks a multi-row selection or a blank cell.
    with pytest.raises(m.MacroError, match="2 rows"):
        m.resolve(spec("\\crMulti", {"sample": "clean"}, "rate", "pct1", default="0.0"), csv_dir, {})
    with pytest.raises(m.MacroError, match="blank"):
        m.resolve(spec("\\crBlank", PUB_B, "rate", "pct1", default="0.0"), csv_dir, {})


def test_verify_against_csvs(csv_dir: Path, tmp_path: Path) -> None:
    specs = [
        spec("\\crRate", A, "rate", "pct1"),
        spec("\\crP", A, "p", "pval"),
        spec("\\crFrac", B, ("k", "n"), "frac"),
        spec("\\crGone", {"sample": "clean", "model": "zzz"}, "rate", "pct1", required=False),
    ]
    out = tmp_path / "out"
    m.run(tmp_path, out, csv_dir=csv_dir, specs=specs)
    assert m.verify_against_csvs(out / "cr_macros.json", csv_dir) == []
    # Change one cell, drop one row, and add the row the missing macro wants.
    df = pd.read_csv(csv_dir / "rates.csv", dtype=str, keep_default_na=False)
    df.loc[(df["sample"] == "clean") & (df["model"] == "a"), "rate"] = "0.26"
    df = df[~((df["sample"] == "clean") & (df["model"] == "b"))]
    df = pd.concat([df, pd.DataFrame([{"sample": "clean", "model": "zzz", "rate": "0.5"}])], ignore_index=True)
    df.to_csv(csv_dir / "rates.csv", index=False)
    diffs = m.verify_against_csvs(out / "cr_macros.json", csv_dir)
    kinds = {d["name"]: d["kind"] for d in diffs}
    assert kinds == {"\\crRate": "raw_changed", "\\crFrac": "no_longer_resolves", "\\crGone": "now_resolves"}
    assert next(d for d in diffs if d["name"] == "\\crRate")["detail"] == "0.25 -> 0.26"
    # The p-value row (unchanged) is compared through its prose form and its twin is skipped.
    assert "\\crP" not in kinds and "\\crPVal" not in kinds


def test_spec_record_round_trips() -> None:
    s = spec("\\crX", A, ("k", "n"), "frac", required=False, agg=None, note="n", default="1/2")
    assert m.spec_from_record(m.spec_record(s)) == s


def test_auto_rounds_long_floats_and_keeps_written_decimals() -> None:
    assert m.fmt_auto("2.683") == "2.683"
    assert m.fmt_auto("100.0") == "100.0"
    assert m.fmt_auto("-1.1184") == "-1.118"
    assert m.fmt_auto("2146") == "2{,}146"
    assert m.fmt_auto("a%b") == r"a\%b"


def test_agg_sum_mean_count(csv_dir: Path) -> None:
    s = spec("\\crSum", {"sample": "clean"}, "k", "int", agg="sum")
    assert m.resolve(s, csv_dir, {})[0] == "57"
    s = spec("\\crMean", {"sample": "clean"}, "rate", "pct1", agg="mean")
    assert m.resolve(s, csv_dir, {})[0] == "37.5"
    s = spec("\\crCount", {"sample": "clean"}, "model", "int", agg="count")
    assert m.resolve(s, csv_dir, {})[0] == "2"


def test_agg_min_max_and_the_label_of_the_extreme_row(csv_dir: Path) -> None:
    clean = {"sample": "clean"}
    assert m.resolve(spec("\\crMin", clean, "rate", "pct1", agg="min"), csv_dir, {})[0] == "25.0"
    assert m.resolve(spec("\\crMax", clean, "rate", "pct1", agg="max"), csv_dir, {})[0] == "50.0"
    # The label of the row with the extreme value, found from the data.
    assert m.resolve(spec("\\crMinLabel", clean, "model", "auto", agg="min", by="rate"), csv_dir, {})[0] == "a"
    assert m.resolve(spec("\\crMaxLabel", clean, "model", "auto", agg="max", by="rate"), csv_dir, {})[0] == "b"
    # The raw cell is kept, not a float re-rendering of it.
    assert m.resolve(spec("\\crMaxK", clean, "k", "int", agg="max", by="rate"), csv_dir, {})[1] == "38"


def test_agg_min_max_tie_makes_the_label_ambiguous(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "rates.csv", [{"g": "x", "pair": "A-B", "kappa": 0.2}, {"g": "x", "pair": "C-D", "kappa": 0.2},
                             {"g": "x", "pair": "B-C", "kappa": 0.7}])
    w = {"g": "x"}
    with pytest.raises(m.MacroError, match="tie for the min of 'kappa'"):
        m.resolve(spec("\\crTie", w, "pair", "auto", agg="min", by="kappa"), d, {})
    # A tie does not matter for the value itself.
    assert m.resolve(spec("\\crTieValue", w, "kappa", "dec2", agg="min"), d, {})[0] == "0.20"
    assert m.resolve(spec("\\crMaxPair", w, "pair", "auto", agg="max", by="kappa"), d, {})[0] == "B-C"


def test_spec_record_carries_by_only_when_set() -> None:
    s = spec("\\crX", A, "model", "auto", agg="min", by="rate")
    assert m.spec_record(s)["by"] == "rate"
    assert m.spec_from_record(m.spec_record(s)) == s
    assert "by" not in m.spec_record(spec("\\crY", A, "k", "int"))


def test_numeric_where_matches_int_and_float_text(csv_dir: Path) -> None:
    s = spec("\\crK", {"k": "19"}, "n", "int")
    assert m.resolve(s, csv_dir, {})[0] == "76"
    s = spec("\\crK", {"k": 19.0}, "n", "int")
    assert m.resolve(s, csv_dir, {})[0] == "76"


def test_list_where_matches_any_item_and_survives_the_json_record(csv_dir: Path) -> None:
    """X13: a list or tuple value in `where` selects rows matching any item, numerically too."""
    s = spec("\\crAny", {"k": ("19", "38", "999")}, "n", "int", agg="count")
    assert m.resolve(s, csv_dir, {})[0] == "2"
    s = spec("\\crAnyNum", {"k": (19.0, 2116)}, "n", "int", agg="sum")
    assert m.resolve(s, csv_dir, {})[0] == "2{,}222"
    # No item matches: the default fills the empty selection.
    s = spec("\\crNone", {"k": ("1", "2")}, "n", "int", agg="count", default="0")
    assert m.resolve(s, csv_dir, {})[0] == "0"
    # The JSON record stores the tuple as a list; the list resolves the same way.
    back = m.spec_from_record(json.loads(json.dumps(m.spec_record(spec("\\crAny", {"k": ("19", "38")}, "n", "int", agg="count")))))
    assert back.where == {"k": ["19", "38"]}
    assert m.resolve(back, csv_dir, {})[0] == "2"


def test_x13_swe_strict_kbv_single_shot_complexity_max_and_handcheck_sample(tmp_path: Path) -> None:
    """X13: KBV counting only adherence below 3, the single-shot complexity maximum, the hand-check sample size."""
    d = tmp_path / "csv"
    base = {"arm": "main", "model_id": PRO, "final_finish_reason": "stop", "violates": "False"}
    _write(d / "e6_swe_runs.csv", [
        {**base, "run_id": "a", "condition": "single_shot", "knows": "", "j_constraint_adherence": 4, "j_complexity_inflation": 0},
        {**base, "run_id": "b", "condition": "single_shot", "knows": "", "j_constraint_adherence": 2, "j_complexity_inflation": 1},
        {**base, "run_id": "c", "condition": "multi_turn_pressure", "knows": "True", "j_constraint_adherence": 3, "j_complexity_inflation": 2},
        {**base, "run_id": "d", "condition": "multi_turn_pressure", "knows": "True", "j_constraint_adherence": 2, "j_complexity_inflation": 2},
        {**base, "run_id": "e", "condition": "multi_turn_pressure", "knows": "False", "j_constraint_adherence": 1, "j_complexity_inflation": 3},
        {**base, "run_id": "f", "arm": "exec", "condition": "multi_turn_pressure", "knows": "True", "j_constraint_adherence": 0,
         "j_complexity_inflation": 1},
    ])
    _write(d / "mechanism_handcheck_sample.csv", [{"run_id": f"r{i}", "constraint": "c"} for i in range(7)])
    wanted = {"\\crESixKbvBelowThree", "\\crESixMaxComplexitySingleShot", "\\crMechHandcheckSampleN"}
    specs = [s for s in m._e6_swe_specs() + m._mechanism_handcheck_specs() if s.name in wanted]
    assert {s.name for s in specs} == wanted
    v = _values(tmp_path, d, specs)
    # d and f: pressure, knows, below 3 (both arms); c is 3; e fails the knows filter.
    assert v["\\crESixKbvBelowThree"] == "2"
    # The maximum, not the mean, over single-shot runs of both arms.
    assert v["\\crESixMaxComplexitySingleShot"] == "1"
    assert v["\\crMechHandcheckSampleN"] == "7"
    # With no pressure run below 3 the count is the default 0.
    runs = pd.read_csv(d / "e6_swe_runs.csv")
    runs.loc[runs["j_constraint_adherence"] < 3, "j_constraint_adherence"] = 3
    runs.to_csv(d / "e6_swe_runs.csv", index=False)
    assert _values(tmp_path, d, specs)["\\crESixKbvBelowThree"] == "0"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_csv_raises_with_spec_name(csv_dir: Path) -> None:
    s = m.MacroSpec("\\crGone", "absent.csv", {}, "x", "int")
    with pytest.raises(m.MacroError, match=r"\\crGone: CSV not found"):
        m.resolve(s, csv_dir, {})


def test_empty_selection_raises(csv_dir: Path) -> None:
    s = spec("\\crEmpty", {"sample": "clean", "model": "zzz"}, "rate", "pct1")
    with pytest.raises(m.MacroError, match=r"\\crEmpty: no row in rates.csv matches"):
        m.resolve(s, csv_dir, {})


def test_multi_row_selection_raises(csv_dir: Path) -> None:
    s = spec("\\crMulti", {"sample": "clean"}, "rate", "pct1")
    with pytest.raises(m.MacroError, match=r"\\crMulti: 2 rows in rates.csv match"):
        m.resolve(s, csv_dir, {})


def test_blank_cell_raises(csv_dir: Path) -> None:
    s = spec("\\crBlank", PUB_B, "rate", "pct1")
    with pytest.raises(m.MacroError, match="blank"):
        m.resolve(s, csv_dir, {})


def test_unknown_column_raises(csv_dir: Path) -> None:
    s = spec("\\crCol", A, "nope", "pct1")
    with pytest.raises(m.MacroError, match="column 'nope' not in rates.csv"):
        m.resolve(s, csv_dir, {})


def test_bad_names_and_formats_rejected() -> None:
    with pytest.raises(m.MacroError, match="bad macro name"):
        m.MacroSpec("\\crHas2Digits", "a.csv", {}, "x", "int")
    with pytest.raises(m.MacroError, match="bad macro name"):
        m.MacroSpec("\\NoPrefix", "a.csv", {}, "x", "int")
    with pytest.raises(m.MacroError, match="unknown fmt"):
        m.MacroSpec("\\crX", "a.csv", {}, "x", "pct9")
    with pytest.raises(m.MacroError, match="column pair"):
        m.MacroSpec("\\crX", "a.csv", {}, "x", "ci")
    with pytest.raises(m.MacroError, match="one column name"):
        m.MacroSpec("\\crX", "a.csv", {}, ("x", "y"), "int")
    with pytest.raises(m.MacroError, match="unknown agg"):
        m.MacroSpec("\\crX", "a.csv", {}, "x", "int", agg="median")
    with pytest.raises(m.MacroError, match="`by` needs agg min or max"):
        m.MacroSpec("\\crX", "a.csv", {}, "x", "int", agg="mean", by="y")


def test_required_failure_stops_run(csv_dir: Path, tmp_path: Path) -> None:
    specs = [spec("\\crEmpty", {"sample": "clean", "model": "zzz"}, "rate", "pct1")]
    with pytest.raises(m.MacroError, match=r"required macro failed: \\crEmpty"):
        m.run(tmp_path, tmp_path / "out", csv_dir=csv_dir, specs=specs)


def test_optional_failure_lands_in_missing(csv_dir: Path, tmp_path: Path) -> None:
    specs = [
        spec("\\crOk", A, "rate", "pct1"),
        spec("\\crEmpty", {"sample": "clean", "model": "zzz"}, "rate", "pct1", required=False),
        spec("\\crBlank", PUB_B, "rate", "pct1", required=False),
        m.MacroSpec("\\crGone", "absent.csv", {}, "x", "int", required=False),
    ]
    out = tmp_path / "out"
    tables = m.run(tmp_path, out, csv_dir=csv_dir, specs=specs)
    assert tables["cr_macros"]["name"].tolist() == ["\\crOk"]
    assert sorted(tables["cr_macros_missing"]["name"]) == ["\\crBlank", "\\crEmpty", "\\crGone"]
    prov = json.loads((out / "cr_macros.tex.provenance.json").read_text())
    assert prov["row_counts"] == {"macros_registered": 4, "macros_emitted": 1, "macros_missing": 3, "macros_required": 1}
    assert {r["name"] for r in prov["missing"]} == {"\\crBlank", "\\crEmpty", "\\crGone"}
    assert "absent.csv" in prov["inputs_missing_on_disk"]
    tex = (out / "cr_macros.tex").read_text()
    assert "\\crEmpty" not in tex and "\\crGone" not in tex
    assert "\\newcommand{\\crOk}{25.0}" in tex


# ---------------------------------------------------------------------------
# Output files and the None/nan guard
# ---------------------------------------------------------------------------


def test_run_writes_tex_json_and_provenance(csv_dir: Path, tmp_path: Path) -> None:
    specs = [
        spec("\\crRate", A, "rate", "pct1"),
        spec("\\crRateFrac", A, ("k", "n"), "frac"),
        spec("\\crBig", PUB_B, "k", "int"),
    ]
    out = tmp_path / "out"
    m.run(tmp_path, out, csv_dir=csv_dir, specs=specs)
    tex = (out / "cr_macros.tex").read_text()
    lines = [ln for ln in tex.splitlines() if ln.startswith("\\newcommand")]
    assert lines == [
        "\\newcommand{\\crRate}{25.0}",
        "\\newcommand{\\crRateFrac}{19/76}",
        "\\newcommand{\\crBig}{2{,}116}",
    ]
    payload = json.loads((out / "cr_macros.json").read_text())
    row = payload["macros"][1]
    assert row["name"] == "\\crRateFrac"
    assert row["csv"] == "rates.csv"
    assert row["where"] == A
    assert row["column"] == ["k", "n"]
    assert row["fmt"] == "frac"
    assert row["raw"] == ["19", "76"]
    assert payload["missing"] == []
    assert (out / "cr_macros.tex.provenance.json").exists()


def test_output_never_contains_none_or_nan(csv_dir: Path, tmp_path: Path) -> None:
    _write(csv_dir / "weird.csv", [{"k": "v", "val": "None"}, {"k": "w", "val": "nan"}, {"k": "x", "val": "NaN"}])
    for key in ("v", "w", "x"):
        s = m.MacroSpec("\\crWeird", "weird.csv", {"k": key}, "val", "auto", required=False)
        tables = m.run(tmp_path, tmp_path / "out", csv_dir=csv_dir, specs=[s])
        assert tables["cr_macros"].empty
        tex = (tmp_path / "out" / "cr_macros.tex").read_text()
        assert "None" not in tex
        assert not re.search(r"\bnan\b", tex, flags=re.IGNORECASE)


def test_registry_names_are_valid_latex_and_unique(csv_dir: Path) -> None:
    specs = m.build_registry(csv_dir)
    names = [s.name for s in specs]
    assert all(NAME_RE.match(n) for n in names)
    assert len(names) == len(set(names))
    assert len(specs) > 500


def test_static_registry_covers_the_headline_macros() -> None:
    names = {s.name for s in m.static_specs()}
    for expected in (
        "\\crCleanN", "\\crPublishedN", "\\crStrayN", "\\crFailedRuns", "\\crSubjectFailures", "\\crScoringFailures",
        "\\crDesignRows", "\\crKbvSonnetPressure", "\\crKbvSonnetPressureFrac", "\\crKbvSonnetPressureCi",
        "\\crKbvPubSonnetPressure", "\\crKbvProPressureWaived", "\\crKbvProPressureWaivedFrac",
        "\\crKbvProPressureExcluded", "\\crKbvProPressureExcludedFrac", "\\crKbvAboveHalfModels",
        "\\crNoncompliantPressure", "\\crMeanAdherenceSonnetPressure", "\\crGkbvSonnetPressure",
        "\\crGkbvSonnetWilcoxonP", "\\crGkbvSonnetHolmP", "\\crBelowThreeAllPressure", "\\crKbvAtThreeAllPressure",
        "\\crHumanNotSatisfiedPooled", "\\crJudgeNotSatisfiedPooled", "\\crHumanScoreDelta",
        "\\crMaxContextShare", "\\crMinHeadroom", "\\crVerbosityPressureVsNeutralTotal",
        "\\crOnsetByTurnTwo", "\\crOnsetByTurnThree", "\\crOrKbvHardConstraints", "\\crOrKbvBannedMoves",
        "\\crRhoAdherenceHardConstraints", "\\crTypologyTopFourShare", "\\crTypologyBestSplitOr",
        "\\crTypologyOmnibusP", "\\crMechAppearsInChecklist", "\\crMechComplianceLanguage",
        "\\crMechAcknowledge", "\\crMechSilent", "\\crMonitorReflagNamed", "\\crMonitorFlagNotNamed",
        "\\crCheckableShare", "\\crCheckerJudgeAgreement", "\\crCheckerJudgeKappa",
    ):
        assert expected in names, expected
    # Every cell mean: 4 dimensions x 7 models x 4 conditions, judge arm.
    means = [s for s in m.static_specs() if s.csv == "per_model_condition_means.csv"
             and s.name.startswith("\\crMean") and s.column.endswith("_mean")]
    assert len(means) == 4 * 7 * 4


def test_required_specs_are_the_b1_b2_streams(csv_dir: Path) -> None:
    required = {s.csv for s in m.build_registry(csv_dir) if s.required}
    assert required == {
        "kbv_published_arm.csv", "kbv_probe_denominator.csv", "kbv_above50_pressure.csv",
        "noncompliance_by_condition.csv", "failed_runs_summary.csv", "corrections_ledger.csv",
    }
    assert all(s.required for s in m.ledger_specs(csv_dir))


# ---------------------------------------------------------------------------
# Ledger macros
# ---------------------------------------------------------------------------


def test_slug_and_number_words() -> None:
    assert m.slug("share by turn 2") == "ShareByTurnTwo"
    assert m.slug("B16") == "BSixteen"
    assert m.slug("C15") == "CFifteen"
    assert m.slug("B12") == "BTwelve"
    assert m.slug("set1_rater_A.json") == "SetOneRaterAJson"
    assert m.slug("token_count == 4096") == "TokenCountFourZeroNineSix"
    assert m.number_words("42") == "FortyTwo"
    assert m.number_words("20") == "Twenty"
    assert m.number_words("07") == "ZeroSeven"
    with pytest.raises(m.MacroError):
        m.slug("--")


def test_ledger_specs_one_macro_per_row_with_variants(csv_dir: Path) -> None:
    specs = m.ledger_specs(csv_dir)
    names = [s.name for s in specs]
    assert names == [
        "\\crLedgerBOneShareByTurnTwo",
        "\\crLedgerBOneShareByTurnTwoFrac",
        "\\crLedgerBOneShareByTurnTwoDen",
        "\\crLedgerBOneShareByTurnTwoPub",
        "\\crLedgerBSixteenRunsTruncatedAllMain",
        "\\crLedgerBSixteenRunsTruncatedAllMainFrac",
        "\\crLedgerBSixteenRunsTruncatedAllMainDen",
        "\\crLedgerBSixteenRunsTruncatedAllMainPub",
        "\\crLedgerBEightPerRaterMinToMax",
    ]
    assert all(s.required for s in specs)
    values = {s.name: m.resolve(s, csv_dir, {})[0] for s in specs}
    assert values["\\crLedgerBOneShareByTurnTwo"] == "72.6"
    assert values["\\crLedgerBOneShareByTurnTwoFrac"] == "239/329"
    assert values["\\crLedgerBOneShareByTurnTwoPub"] == r"74\%"
    assert values["\\crLedgerBSixteenRunsTruncatedAllMain"] == "226"
    assert values["\\crLedgerBEightPerRaterMinToMax"] == "15 to 86"


def test_ledger_duplicate_rows_raise(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "corrections_ledger.csv", [
        {"id": "B1", "quantity": "x", "recomputed_value": 1},
        {"id": "B1", "quantity": "x", "recomputed_value": 2},
    ])
    with pytest.raises(m.MacroError, match="same macro"):
        m.ledger_specs(d)


def test_ledger_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(m.MacroError, match="ledger CSV not found"):
        m.ledger_specs(tmp_path)


# ---------------------------------------------------------------------------
# check_against_tex
# ---------------------------------------------------------------------------


def test_check_against_tex(tmp_path: Path) -> None:
    tex = tmp_path / "paper.tex"
    tex.write_text("Across $\\crCleanN$ runs, KBV is \\crKbvSonnetPressure\\% (\\crKbvSonnetPressureFrac). \\crUndefinedOne here.\n")
    macros = tmp_path / "cr_macros.tex"
    macros.write_text(
        "% header\n\\newcommand{\\crCleanN}{2{,}116}\n\\newcommand{\\crKbvSonnetPressure}{98.7}\n"
        "\\newcommand{\\crKbvSonnetPressureFrac}{75/76}\n\\newcommand{\\crUnused}{1}\n"
    )
    report = m.check_against_tex(tex, macros)
    assert report == {"used_not_defined": ["\\crUndefinedOne"], "defined_not_used": ["\\crUnused"]}
    report = m.check_against_tex(tex, defined={"\\crCleanN"})
    assert report["used_not_defined"] == ["\\crKbvSonnetPressure", "\\crKbvSonnetPressureFrac", "\\crUndefinedOne"]
    assert report["defined_not_used"] == []


def test_check_against_tex_ignores_other_macros(tmp_path: Path) -> None:
    tex = tmp_path / "paper.tex"
    tex.write_text("\\BenchmarkRuns and \\crFoo and \\cr and \\cref{fig:x} and \\crefrange{a}{b}\n")
    report = m.check_against_tex(tex, defined=set())
    assert report["used_not_defined"] == ["\\crFoo"]
    assert not m.NAME_RE.match("\\crfoo")


def test_tex_header_says_how_to_use_macros_in_prose() -> None:
    tex = m.render_tex([])
    assert "\\crCleanN{}" in tex and "no xspace" in tex


# ---------------------------------------------------------------------------
# release_manifest hook (plan X6)
# ---------------------------------------------------------------------------


def test_release_manifest_appends_one_input_line(tmp_path: Path) -> None:
    from drift_bench.analysis import release_manifest as rm

    root = tmp_path / "repo"
    cr_tex = root / "drift_bench" / "data" / "camera_ready" / "analysis" / "cr_macros.tex"
    cr_tex.parent.mkdir(parents=True)
    cr_tex.write_text("% cr\n")
    tex_output = root / "drift_bench" / "data" / "analysis" / "release_macros.tex"
    tex_output.parent.mkdir(parents=True)
    tex_output.write_text("\\newcommand{\\BenchmarkRuns}{2{,}146}\n")
    line = rm._append_cr_macros_input(tex_output, cr_tex, root=root)
    assert line == "\\input{drift_bench/data/camera_ready/analysis/cr_macros.tex}"
    text = tex_output.read_text()
    assert text.startswith("\\newcommand{\\BenchmarkRuns}{2{,}146}\n")
    assert text.count(line) == 1
    rm._append_cr_macros_input(tex_output, cr_tex, root=root)
    assert tex_output.read_text().count(line) == 1
    # Outside the root the path is absolute.
    other = tmp_path / "elsewhere" / "cr_macros.tex"
    other.parent.mkdir()
    other.write_text("")
    assert rm._append_cr_macros_input(tex_output, other, root=root) == f"\\input{{{other.resolve().as_posix()}}}"
    # A data directory linked into an HF cache outside the root keeps the repo path.
    cache = tmp_path / "hf_cache" / "camera_ready" / "analysis"
    cache.mkdir(parents=True)
    (cache / "cr_macros.tex").write_text("% cr\n")
    (root / "drift_bench" / "data" / "linked").symlink_to(cache)
    linked = root / "drift_bench" / "data" / "linked" / "cr_macros.tex"
    assert rm._append_cr_macros_input(tex_output, linked, root=root) == "\\input{drift_bench/data/linked/cr_macros.tex}"


# ---------------------------------------------------------------------------
# Experiment table groups (E1 crossed judge, E2, E3, E5, E6, E7)
# ---------------------------------------------------------------------------


EXPERIMENT_GROUPS = (
    m._crossed_judge_specs, m._e7_strict_probe_specs, m._reprobe_specs,
    m._e2_uncapped_specs, m._e5_per_turn_specs, m._e6_swe_specs, m._sensitivity_specs, m._followup_runs_specs,
    m._gkbv_opus_arm_specs, m._surface_gap_specs, m._probe_accuracy_specs, m._length_fidelity_specs,
    m._sonnet_truncation_specs, m._judge_sensitivity_specs,
)
PRO = "gemini/gemini-3.1-pro-preview"
FLASH = "gemini/gemini-3.1-flash-lite-preview"
GPT = "openai/gpt-5.4"


def _values(tmp_path: Path, csv_dir: Path, specs: list[m.MacroSpec]) -> dict[str, str]:
    tables = m.run(tmp_path, tmp_path / "out", csv_dir=csv_dir, specs=specs)
    return dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))


def test_experiment_group_names_are_short_unique_and_optional() -> None:
    names: list[str] = []
    for group in EXPERIMENT_GROUPS:
        specs = group()
        assert specs, group.__name__
        assert all(not s.required for s in specs), group.__name__
        names += [s.name for s in specs]
    assert all(NAME_RE.match(n) for n in names)
    assert len(names) == len(set(names))
    too_long = [n for n in names if len(n) >= 60]
    assert not too_long, too_long
    # The experiment names do not collide with the rest of the registry.
    static = {s.name for s in m.static_specs()}
    assert set(names) <= static
    assert len(static) == len(m.static_specs())


def test_crossed_judge_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "kbv_common_judge.csv", [
        {"judge_arm": "opus", "variant": "clean", "model_id": GPT, "condition": "multi_turn_pressure",
         "n": 76, "n_with_probe": 76, "n_truncated_waived": 0, "kbv_count": 6, "kbv_rate": 6 / 76, "ci_low": 0.0367, "ci_high": 0.1617},
        {"judge_arm": "published", "variant": "clean_truncated_probes_waived", "model_id": PRO, "condition": "multi_turn_pressure",
         "n": 76, "n_with_probe": 76, "n_truncated_waived": 62, "kbv_count": 19, "kbv_rate": 0.25, "ci_low": 0.166, "ci_high": 0.358},
    ])
    _write(d / "kbv_common_judge_range.csv", [
        {"judge_arm": "opus", "variant": "clean", "n_models": 7, "min_kbv_rate": 0.0789, "argmin_model": GPT,
         "max_kbv_rate": 0.934, "argmax_model": "llama", "cut": 0.5, "n_above_cut": 4, "models_above_cut": "a;b"},
    ])
    _write(d / "judge_offset.csv", [
        {"block": "offset_overall", "dimension": "constraint_adherence", "model_id": "all", "condition": "ss+mtp", "judge": "opus_minus_gpt",
         "estimate": 0.5354, "ci_low": 0.4876, "ci_high": 0.5851, "coef": "", "se": "", "n_runs": 1057},
        {"block": "interaction_judge_x_condition", "dimension": "complexity_inflation", "model_id": "all", "condition": "ss+mtp", "judge": "opus_vs_gpt",
         "estimate": 0.0327, "ci_low": "", "ci_high": "", "coef": -0.0931, "se": 0.0436, "n_runs": 1057},
        {"block": "pressure_share_in_cells_3_4", "dimension": "constraint_adherence", "model_id": "all", "condition": "multi_turn_pressure", "judge": "gpt",
         "estimate": 0.734, "ci_low": "", "ci_high": "", "coef": "", "se": "", "n_runs": 530},
    ])
    _write(d / "common_judge_aggregate.csv", [
        {"table_arm": "opus", "judge_arm": "opus", "row_type": "condition_mean", "model_id": "all", "condition": "multi_turn_pressure",
         "n": 530, "constraint_adherence_mean": 3.4453, "constraint_adherence_se": 0.0273},
        {"table_arm": "opus", "judge_arm": "opus", "row_type": "delta_pressure_minus_single_shot", "model_id": "all", "condition": "mtp-ss",
         "n": 1057, "constraint_adherence_mean": -0.3213, "constraint_adherence_se": 0.0333},
    ])
    _write(d / "auditor_agreement_by_arm.csv", [
        {"judge_arm": "published", "definition": "kbv_vs_auditor", "condition": "multi_turn_pressure", "n": 530,
         "judge_flag_rate": 0.5226, "auditor_drift_rate": 0.5925, "percent_agreement": 0.7943, "spearman_rho": 0.5911, "spearman_p": 3e-51,
         "both_positive": 241, "both_negative": 180, "judge_only": 36, "auditor_only": 73},
    ])
    _write(d / "stretch_violate_by_judge_arm.csv", [
        {"judge_arm": "opus", "model_id": "all", "n_ss": 527, "n_mtp": 530, "below3_ss": 3, "below3_mtp": 23,
         "share_below3_ss": 3 / 527, "share_below3_mtp": 23 / 530, "n_kbv_positive_mtp": 259, "kbv_at3": 236, "kbv_below3": 23,
         "share_kbv_at3": 236 / 259, "share_kbv_below3": 23 / 259},
    ])
    _write(d / "judge_drift_control.csv", [
        {"control_judge": "both", "dimension": "constraint_adherence", "n": 49, "mean_april": 3.1429, "mean_today": 3.1837,
         "mean_diff_today_minus_april": 0.0408, "exact_agreement": 0.9592, "within_1": 1.0, "spearman_rho": 0.9686},
    ])
    v = _values(tmp_path, d, m._crossed_judge_specs())
    assert v["\\crKbvOpusArmPressureGpt"] == "7.9"
    assert v["\\crKbvOpusArmPressureGptFrac"] == "6/76"
    assert v["\\crKbvPublishedArmPressureProWaived"] == "25.0"
    assert v["\\crKbvPublishedArmPressureProWaivedCi"] == "[16.6, 35.8]"
    assert v["\\crKbvPublishedArmPressureProWaivedImputed"] == "62"
    assert v["\\crKbvOpusArmMin"] == "7.9"
    assert v["\\crKbvOpusArmMax"] == "93.4"
    assert v["\\crKbvOpusArmAboveHalfFrac"] == "4/7"
    assert v["\\crKbvOpusArmMinModel"] == "openai/gpt-5.4"
    assert v["\\crJudgeOffsetAdherence"] == "0.535"
    assert v["\\crJudgeOffsetAdherenceCi"] == "[0.488, 0.585]"
    assert v["\\crJudgeInteractionComplexityCoef"] == "-0.093"
    assert v["\\crJudgeInteractionComplexitySe"] == "0.044"
    assert v["\\crJudgeInteractionComplexityP"] == "$p = 0.033$"
    assert v["\\crPressureCellsThreeFourGptArm"] == "73.4"
    assert v["\\crMeanAdherenceAllPressureOpusArm"] == "3.45"
    assert v["\\crSeAdherenceAllPressureOpusArm"] == "0.03"
    assert v["\\crDeltaAdherenceAllOpusArm"] == "-0.32"
    assert v["\\crNAllPressureOpusArm"] == "530"
    assert v["\\crAuditorAgreePublishedArmKbv"] == "79.4"
    assert v["\\crAuditorAgreePublishedArmKbvRho"] == "0.59"
    assert v["\\crAuditorAgreePublishedArmKbvRhoP"] == "$p < 0.001$"
    assert v["\\crBelowThreeOpusArmAllPressureFrac"] == "23/530"
    assert v["\\crKbvAtThreeOpusArmAll"] == "91.1"
    assert v["\\crDriftControlBothAdherenceExact"] == "95.9"
    assert v["\\crDriftControlBothAdherenceDiff"] == "0.04"
    # Rows the synthetic CSVs do not carry are optional, never fatal.
    assert "\\crKbvGptArmPressureSonnet" not in v


def test_e7_strict_probe_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "e7_strict_probe_accuracy.csv", [
        {"model_id": "all", "n_selected": 304, "n_truncated_probes": 62, "n_with_adherence": 304,
         "mean_april_accuracy": 0.9406, "mean_april_equiv_regrade": 0.9384, "mean_strict_regrade": 0.9368,
         "share_ge_0_8_strict": 0.9211, "n_side_change_strict_vs_april": 1, "kbv_april": 166, "kbv_strict": 165,
         "kbv_rate_strict": 165 / 304, "grader_models_reported": "claude-opus-4-6", "total_cost_usd": 5.4294},
    ])
    _write(d / "e7_strict_probe_runs.csv", [
        {"run_id": "a", "probe_truncated": "False", "parse_ok": "True", "kbv_april": "True", "kbv_strict": "True"},
        {"run_id": "b", "probe_truncated": "True", "parse_ok": "True", "kbv_april": "True", "kbv_strict": "False"},
        {"run_id": "c", "probe_truncated": "False", "parse_ok": "True", "kbv_april": "False", "kbv_strict": "False"},
    ])
    v = _values(tmp_path, d, m._e7_strict_probe_specs())
    assert v["\\crESevenAprilRecallPooled"] == "94.1"
    assert v["\\crESevenEquivRecallPooled"] == "93.8"
    assert v["\\crESevenStrictRecallPooled"] == "93.7"
    assert v["\\crESevenStrictAboveCutPooled"] == "92.1"
    assert v["\\crESevenSideChangeStrictPooled"] == "1"
    assert v["\\crESevenKbvStrictFracPooled"] == "165/304"
    assert v["\\crESevenKbvAprilFracPooled"] == "166/304"
    assert v["\\crESevenKbvStrictRatePooled"] == "54.3"
    assert v["\\crESevenTruncatedPooled"] == "62"
    assert v["\\crESevenCostPooled"] == "5.43"
    assert v["\\crESevenGrader"] == "claude-opus-4-6"
    assert v["\\crESevenRuns"] == "3"
    assert v["\\crESevenRunsTruncated"] == "1"
    assert v["\\crESevenRunsKbvApril"] == "2"
    assert v["\\crESevenRunsKbvStrict"] == "1"
    assert v["\\crESevenRunsParseFailed"] == "0"


def test_reprobe_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "reprobe_summary.csv", [
        {"model_id": PRO, "condition": "multi_turn_pressure", "n": 76, "mean_original_accuracy": 0.7623, "mean_new_accuracy": 1.0,
         "mean_accuracy_delta": 0.2377, "share_original_truncated": 0.8158, "share_new_finish_length": 0.0,
         "share_original_ge_0_8": 0.6974, "share_new_ge_0_8": 1.0, "mean_original_tokens": 996.6, "mean_new_tokens": 1215.4},
    ])
    _write(d / "reprobe_accuracy.csv", [
        {"run_id": "a", "model_id": PRO, "condition": "multi_turn_pressure", "original_truncated": "True", "new_finish_reason": "stop",
         "original_overall_accuracy": 0.5, "new_overall_accuracy": 1.0, "probe_max_tokens": 4096, "cost_usd": 0.03},
        {"run_id": "b", "model_id": PRO, "condition": "multi_turn_neutral", "original_truncated": "False", "new_finish_reason": "stop",
         "original_overall_accuracy": 1.0, "new_overall_accuracy": 1.0, "probe_max_tokens": 4096, "cost_usd": 0.02},
    ])
    v = _values(tmp_path, d, m._reprobe_specs())
    assert v["\\crEThreeReplaysPressure"] == "76"
    assert v["\\crEThreeOriginalRecallPressure"] == "76.2"
    assert v["\\crEThreeReplayRecallPressure"] == "100.0"
    assert v["\\crEThreeRecallGainPressure"] == "23.8"
    assert v["\\crEThreeOriginalCapHitsPressure"] == "81.6"
    assert v["\\crEThreeReplayCapHitsPressure"] == "0.0"
    assert v["\\crEThreeOriginalTokensPressure"] == "997"
    assert v["\\crEThreeReplayTokensPressure"] == "1{,}215"
    assert v["\\crEThreeReplays"] == "2"
    assert v["\\crEThreeOriginalCapHits"] == "1"
    assert v["\\crEThreeReplayCapHits"] == "0"
    assert v["\\crEThreeOriginalCapHitsPressureN"] == "1"
    assert v["\\crEThreeOriginalCapHitsNeutralN"] == "0"
    assert v["\\crEThreeOriginalRecall"] == "75.0"
    assert v["\\crEThreeReplayRecall"] == "100.0"
    assert v["\\crEThreeReplayCap"] == "4{,}096"
    assert v["\\crEThreeCost"] == "0.05"


def test_e2_uncapped_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    cell = {"source": "e2_uncapped", "condition": "multi_turn_pressure", "n_runs": 38, "n_briefs": 38, "max_tokens": 16384,
            "cap_turns": 107, "cap_turn_share": 0.4693, "runs_any_cap": 36, "runs_any_cap_share": 0.9474, "final_cap": 23,
            "final_cap_share": 0.6053, "final_tokens_mean": 4493.76, "output_tokens_per_run_mean": 25908.9,
            "judge_date_utc": "2026-10-26", "constraint_adherence_mean": 3.3421, "constraint_adherence_sd": 0.5,
            "constraint_adherence_ci_low": 3.1842, "constraint_adherence_ci_high": 3.5, "adherence_lt4": 25, "adherence_lt3": 0,
            "adherence_eq4": 13, "n_with_final_probe": 38.0, "knows": 38.0, "kbv_count": 25.0, "kbv_rate": 25 / 38,
            "kbv_ci_low": 0.4989, "kbv_ci_high": 0.7879}
    _write(d / "e2_uncapped_cells.csv", [{**cell, "judge_arm": "opus"}, {**cell, "judge_arm": "gpt", "kbv_count": 37.0, "kbv_rate": 37 / 38}])
    _write(d / "e2_uncapped_deltas.csv", [
        {"condition": "multi_turn_pressure", "judge_arm": "opus", "metric": "kbv_rate", "design": "unpaired_runs", "n_e2": 38, "n_april": 76,
         "delta": -0.0921, "ci_low": -0.2718, "ci_high": 0.0767, "p_value": 0.3772, "briefs_up": "", "briefs_down": "", "briefs_tie": "",
         "ci_excludes_zero": "False"},
        {"condition": "multi_turn_pressure", "judge_arm": "gpt", "metric": "constraint_adherence", "design": "brief_paired", "n_e2": 38, "n_april": 76,
         "delta": 0.2632, "ci_low": 0.0892, "ci_high": 0.4372, "p_value": 0.0057, "briefs_up": 19.0, "briefs_down": 6.0, "briefs_tie": 13.0,
         "ci_excludes_zero": "True"},
        {"condition": "multi_turn_pressure", "judge_arm": "opus", "metric": "output_tokens_per_run", "design": "brief_paired", "n_e2": 38, "n_april": 76,
         "delta": 2997.45, "ci_low": 2084.2, "ci_high": 3910.7, "p_value": 1.5e-06, "briefs_up": 34.0, "briefs_down": 4.0, "briefs_tie": 0.0,
         "ci_excludes_zero": "True"},
    ])
    _write(d / "e2_uncapped_offset.csv", [
        {"scope": "overall", "dimension": "constraint_adherence", "n_runs": 76, "n_briefs": 38, "e2_offset": 0.7237, "e2_ci_low": 0.6184,
         "e2_ci_high": 0.8421, "e1_offset": 0.8092, "e1_ci_low": 0.7171, "e1_ci_high": 0.8947, "e1_n_runs": 152.0},
    ])
    v = _values(tmp_path, d, m._e2_uncapped_specs())
    assert v["\\crETwoNUncappedPressure"] == "38"
    assert v["\\crETwoMaxTokensUncappedPressure"] == "16{,}384"
    assert v["\\crETwoFinalCapShareUncappedPressure"] == "60.5"
    assert v["\\crETwoFinalCapFracUncappedPressure"] == "23/38"
    assert v["\\crETwoOutputTokensUncappedPressure"] == "25{,}909"
    assert v["\\crETwoMeanAdherenceUncappedPressureOpusArm"] == "3.34"
    assert v["\\crETwoMeanAdherenceUncappedPressureOpusArmCi"] == "[3.18, 3.50]"
    assert v["\\crETwoAdherenceLtFourUncappedPressureOpusArm"] == "25"
    assert v["\\crETwoJudgeDateUncappedPressureOpusArm"] == "2026-10-26"
    assert v["\\crETwoKbvUncappedOpusArm"] == "65.8"
    assert v["\\crETwoKbvUncappedOpusArmFrac"] == "25/38"
    assert v["\\crETwoKbvUncappedOpusArmCi"] == "[49.9, 78.8]"
    assert v["\\crETwoKbvUncappedGptArm"] == "97.4"
    assert v["\\crETwoDeltaKbvOpusArmUnpaired"] == "-9.2"
    assert v["\\crETwoDeltaKbvOpusArmUnpairedCi"] == "[-27.2, 7.7]"
    assert v["\\crETwoDeltaKbvOpusArmUnpairedP"] == "$p = 0.377$"
    assert v["\\crETwoDeltaKbvOpusArmUnpairedExcludes"] == "False"
    assert v["\\crETwoDeltaAdherencePressureGptArmPaired"] == "0.26"
    assert v["\\crETwoDeltaAdherencePressureGptArmPairedCi"] == "[0.09, 0.44]"
    assert v["\\crETwoDeltaAdherencePressureGptArmPairedBriefsUp"] == "19"
    assert v["\\crETwoDeltaOutputTokensPressureOpusArmPaired"] == "2{,}997"
    assert v["\\crETwoDeltaOutputTokensPressureOpusArmPairedCiLow"] == "2{,}084"
    assert v["\\crETwoDeltaOutputTokensPressureOpusArmPairedCiHigh"] == "3{,}911"
    assert v["\\crETwoOffsetAdherenceOverall"] == "0.72"
    assert v["\\crETwoOffsetAdherenceOverallCi"] == "[0.62, 0.84]"
    assert v["\\crETwoOffsetAdherenceOverallEOne"] == "0.81"
    assert v["\\crETwoOffsetAdherenceOverallEOneN"] == "152"
    # No KBV macro is registered for single-shot cells.
    assert not [n for n in (s.name for s in m._e2_uncapped_specs()) if "Kbv" in n and "SingleShot" in n]


def test_e5_per_turn_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "e5_per_turn_cells.csv", [
        {"model_id": FLASH, "condition": "multi_turn_pressure", "turn": 2, "n": 38, "mean_constraint_adherence": 3.1579,
         "ci_lo_constraint_adherence": 2.9953, "ci_hi_constraint_adherence": 3.3205, "share_ca_lt4": 0.78947, "share_ca_lt4_ci_lo": 0.64,
         "share_ca_lt4_ci_hi": 0.89, "kbv_rate": 0.78947, "kbv_ci_lo": 0.64, "kbv_ci_hi": 0.89, "probe_recall_mean": 0.9971},
    ])
    _write(d / "e5_per_turn_deltas.csv", [
        {"model_id": FLASH, "condition": "multi_turn_pressure", "dimension": "constraint_adherence", "contrast": "t2-t1", "n": 38,
         "mean_delta": -0.39474, "ci_lo": -0.5746, "ci_hi": -0.2149, "p_paired_t": 7.7e-05, "p_wilcoxon": 0.000275,
         "step_share": 0.8824, "step_share_denominator_ci_excludes_zero": "True"},
        {"model_id": "pooled", "condition": "pooled", "dimension": "constraint_adherence", "contrast": "t5-t2", "n": 152,
         "mean_delta": 0.0132, "ci_lo": -0.0421, "ci_hi": 0.0684, "p_paired_t": 0.6389, "p_wilcoxon": 0.6374, "step_share": "",
         "step_share_denominator_ci_excludes_zero": ""},
        {"model_id": "pooled", "condition": "multi_turn_pressure", "dimension": "alternative_coverage", "contrast": "t6april-t5", "n": 76,
         "mean_delta": 1.4211, "ci_lo": 1.2, "ci_hi": 1.64, "p_paired_t": 1e-20, "p_wilcoxon": 1e-10, "step_share": "",
         "step_share_denominator_ci_excludes_zero": ""},
    ])
    v = _values(tmp_path, d, m._e5_per_turn_specs())
    assert v["\\crEFiveNFlashLitePressureTurnTwo"] == "38"
    assert v["\\crEFiveMeanAdherenceFlashLitePressureTurnTwo"] == "3.16"
    assert v["\\crEFiveMeanAdherenceFlashLitePressureTurnTwoCi"] == "[3.00, 3.32]"
    assert v["\\crEFiveBelowFourFlashLitePressureTurnTwo"] == "78.9"
    assert v["\\crEFiveKbvFlashLitePressureTurnTwoCi"] == "[64.0, 89.0]"
    assert v["\\crEFiveRecallFlashLitePressureTurnTwo"] == "99.7"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOne"] == "-0.39"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneExact"] == "-0.395"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneCi"] == "[-0.57, -0.21]"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneP"] == "$p < 0.001$"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneWilcoxonP"] == "$p < 0.001$"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneStepShare"] == "0.88"
    assert v["\\crEFiveAdherenceFlashLitePressureTwoOneStepExcludesZero"] == "True"
    assert v["\\crEFiveAdherencePooledFiveTwo"] == "0.01"
    assert v["\\crEFiveAdherencePooledFiveTwoN"] == "152"
    assert v["\\crEFiveCoveragePooledPressureSixAprilFive"] == "1.42"
    assert v["\\crEFiveCoveragePooledPressureSixAprilFiveCi"] == "[1.20, 1.64]"
    names = {s.name for s in m._e5_per_turn_specs()}
    # Step share only on the t2-t1 contrast; no per-model rows for the pooled condition.
    assert "\\crEFiveAdherencePooledFiveTwoStepShare" not in names
    assert "\\crEFiveAdherenceGptTwoOne" not in names
    assert "\\crEFiveAdherencePooledTwoOneStepShare" in names


def test_e6_swe_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "e6_swe_cells.csv", [
        {"arm": "main", "model_id": FLASH, "condition": "multi_turn_pressure", "n": 20, "n_briefs": 10, "mean_objective_fidelity": 3.9,
         "mean_constraint_adherence": 3.2, "mean_alternative_coverage": 3.4, "mean_complexity_inflation": 1.0, "noncompliance_rate": 0.8,
         "n_noncompliant": 16, "n_with_probe": 20, "n_kbv": 16.0, "kbv_rate": 0.8, "kbv_ci_lo": 0.584, "kbv_ci_hi": 0.9193,
         "mean_final_probe_accuracy": 1.0, "knows_rate": 1.0, "n_probe_replay": 0, "mean_original_probe_accuracy": 1.0,
         "n_kbv_capped_as_graded": 16.0, "kbv_rate_capped_as_graded": 0.8},
        {"arm": "main", "model_id": PRO, "condition": "multi_turn_pressure", "n": 20, "n_briefs": 10, "mean_objective_fidelity": 4.0,
         "mean_constraint_adherence": 3.75, "mean_alternative_coverage": 4.0, "mean_complexity_inflation": 0.95, "noncompliance_rate": 0.25,
         "n_noncompliant": 5, "n_with_probe": 20, "n_kbv": 5.0, "kbv_rate": 0.25, "kbv_ci_lo": 0.1119, "kbv_ci_hi": 0.4687,
         "mean_final_probe_accuracy": 1.0, "knows_rate": 1.0, "n_probe_replay": 20, "mean_original_probe_accuracy": 0.8214,
         "n_kbv_capped_as_graded": 4.0, "kbv_rate_capped_as_graded": 0.2},
        {"arm": "main", "model_id": FLASH, "condition": "single_shot", "n": 20, "n_briefs": 10, "mean_objective_fidelity": 3.9,
         "mean_constraint_adherence": 3.05, "mean_alternative_coverage": 3.05, "mean_complexity_inflation": 0.0, "noncompliance_rate": 0.95,
         "n_noncompliant": 19, "n_with_probe": 0, "n_kbv": "", "kbv_rate": "", "kbv_ci_lo": "", "kbv_ci_hi": "",
         "mean_final_probe_accuracy": "", "knows_rate": "", "n_probe_replay": 0, "mean_original_probe_accuracy": "",
         "n_kbv_capped_as_graded": "", "kbv_rate_capped_as_graded": ""},
    ])
    run = {"arm": "main", "model_id": PRO, "condition": "single_shot", "final_finish_reason": "stop", "violates": "False", "j_constraint_adherence": 4}
    _write(d / "e6_swe_runs.csv", [
        {**run, "run_id": "a", "final_finish_reason": "length", "violates": "True", "j_constraint_adherence": 3},
        {**run, "run_id": "b", "final_finish_reason": "length", "violates": "False"},
        {**run, "run_id": "c", "violates": "True", "j_constraint_adherence": 3},
        {**run, "run_id": "d"},
        {**run, "run_id": "e", "arm": "exec"},
    ])
    _write(d / "e6_swe_exec_fidelity.csv", [
        {"run_id": "e", "model_id": PRO, "condition": "single_shot", "contract_constraint_adherence": 3, "plain_constraint_adherence": 4,
         "delta_constraint_adherence": -1, "contract_objective_fidelity": 4, "plain_objective_fidelity": 4, "delta_objective_fidelity": 0},
        {"run_id": "f", "model_id": PRO, "condition": "single_shot", "contract_constraint_adherence": 4, "plain_constraint_adherence": 4,
         "delta_constraint_adherence": 0, "contract_objective_fidelity": 4, "plain_objective_fidelity": 4, "delta_objective_fidelity": 0},
    ])
    _write(d / "swe_verify_results.csv", [
        {"run_id": "e", "all_passed": "True", "passed": 9, "failed": 0, "errors": 0, "total": 9, "truncated": "True", "refused": "False",
         "no_code": "False", "static_checks": 5, "static_failed": 0},
        {"run_id": "f", "all_passed": "True", "passed": 7, "failed": 0, "errors": 0, "total": 7, "truncated": "False", "refused": "False",
         "no_code": "False", "static_checks": 3, "static_failed": 0},
    ])
    _write(d / "e6_swe_costs.csv", [
        {"arm": "all", "phase": "total", "model_id": "", "n": 312, "cost_usd": 20.26},
        {"arm": "main", "phase": "subjects", "model_id": PRO, "n": 40, "cost_usd": 8.2874},
        {"arm": "exec", "phase": "judge_plain_brief", "model_id": "anthropic/claude-opus-4-6", "n": 16, "cost_usd": 0.8594},
    ])
    v = _values(tmp_path, d, m._e6_swe_specs())
    assert v["\\crESixKbvFlashLite"] == "80.0"
    assert v["\\crESixKbvFlashLiteFrac"] == "16/20"
    assert v["\\crESixKbvFlashLiteCi"] == "[58.4, 91.9]"
    assert v["\\crESixKbvProFrac"] == "5/20"
    assert v["\\crESixKbvCappedProFrac"] == "4/20"
    assert v["\\crESixKbvCappedPro"] == "20.0"
    assert v["\\crESixOriginalRecallPro"] == "82.1"
    assert v["\\crESixProbeReplaysPro"] == "20"
    assert v["\\crESixNoncompliantFlashLiteSingleShotFrac"] == "19/20"
    assert v["\\crESixMeanAdherenceFlashLiteSingleShot"] == "3.05"
    assert v["\\crESixRuns"] == "5"
    assert v["\\crESixMainRuns"] == "4"
    assert v["\\crESixExecRuns"] == "1"
    assert v["\\crESixProSingleShotCapHits"] == "2"
    assert v["\\crESixProSingleShotCapNoncompliant"] == "1"
    assert v["\\crESixProSingleShotNoncompliant"] == "2"
    assert v["\\crESixRunsAdherenceThree"] == "2"
    assert v["\\crESixRunsAdherenceFour"] == "3"
    assert v["\\crESixRunsAdherenceTwo"] == "0"
    assert v["\\crESixExecCompared"] == "2"
    assert v["\\crESixExecAgreeAdherence"] == "1"
    assert v["\\crESixExecAgreeFidelity"] == "2"
    assert v["\\crESixExecContractAdherence"] == "3.50"
    assert v["\\crESixExecPlainAdherence"] == "4.00"
    assert v["\\crESixExecContractAdherenceProSingleShot"] == "3.50"
    assert v["\\crESixHarnessRuns"] == "2"
    assert v["\\crESixHarnessAllPassed"] == "2"
    assert v["\\crESixHarnessTests"] == "16"
    assert v["\\crESixHarnessFailed"] == "0"
    assert v["\\crESixHarnessTruncated"] == "1"
    assert v["\\crESixHarnessRefused"] == "0"
    assert v["\\crESixHarnessStaticChecks"] == "8"
    assert v["\\crESixCostTotal"] == "20.26"
    assert v["\\crESixCostCalls"] == "312"
    assert v["\\crESixCostSubjectsPro"] == "8.29"
    assert v["\\crESixCostExecJudgePlain"] == "0.86"
    # The single-shot Flash-Lite cell has no probe, so no KBV macro resolves for it and nothing is fatal.
    assert "\\crESixKbvFlashLiteSingleShot" not in v


REAL_CSV_DIR = m.DEFAULT_OUT_DIR
REAL_INPUTS = (
    "kbv_common_judge.csv", "judge_offset.csv", "e2_uncapped_cells.csv", "e5_per_turn_deltas.csv",
    "e6_swe_cells.csv", "reprobe_accuracy.csv", "e7_strict_probe_accuracy.csv", "corrections_ledger.csv",
)


@pytest.mark.skipif(not all((REAL_CSV_DIR / f).exists() for f in REAL_INPUTS), reason="camera-ready CSVs not on disk")
def test_real_generation_defines_the_decision_headline_macros(tmp_path: Path) -> None:
    """Decisions P1, P3, P8, P14, P15, P16: the macros the Phase 3 text quotes resolve on the real CSVs.

    Writes to tmp_path only; the committed cr_macros.tex is not touched.
    """
    tables = m.run(m.DEFAULT_DATA_ROOT, tmp_path / "out", csv_dir=REAL_CSV_DIR)
    v = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    expected = {
        # P1: Gemini Pro KBV 25.0% (19/76, Wilson [16.6, 35.8]) with recall measured; 18.4% as graded; 14.3% excluded.
        "\\crKbvProPressureWaived": "25.0",
        "\\crKbvProPressureWaivedFrac": "19/76",
        "\\crKbvPublishedArmPressureProWaivedCi": "[16.6, 35.8]",
        "\\crKbvProPressure": "18.4",
        "\\crKbvProPressureExcluded": "14.3",
        "\\crEThreeReplays": "228",
        "\\crEThreeReplayRecall": "100.0",
        # Judge offset (A4): Opus minus GPT-5.4 on adherence, +0.535.
        "\\crJudgeOffsetAdherence": "0.535",
        "\\crJudgeOffsetAdherenceCi": "[0.488, 0.585]",
        # A2 / A12: the KBV range and the above-50% set per arm.
        "\\crKbvOpusArmMin": "7.9",
        "\\crKbvOpusArmMax": "93.4",
        "\\crKbvGptArmMin": "42.1",
        "\\crKbvGptArmMax": "98.7",
        "\\crKbvOpusArmAboveHalfFrac": "4/7",
        "\\crKbvGptArmAboveHalfFrac": "6/7",
        # P16: E2 Sonnet uncapped.
        "\\crETwoKbvUncappedOpusArm": "65.8",
        "\\crETwoKbvUncappedOpusArmCi": "[49.9, 78.8]",
        "\\crETwoKbvAprilOpusArm": "75.0",
        "\\crETwoDeltaKbvOpusArmUnpaired": "-9.2",
        "\\crETwoDeltaKbvOpusArmUnpairedCi": "[-27.2, 7.7]",
        "\\crETwoDeltaAdherencePressureGptArmPaired": "0.26",
        "\\crETwoOffsetAdherenceOverall": "0.72",
        "\\crETwoOffsetAdherenceOverallEOne": "0.81",
        # P14: E5 per-turn.
        "\\crEFiveAdherenceFlashLitePressureTwoOne": "-0.39",
        "\\crEFiveAdherenceFlashLitePressureTwoOneExact": "-0.395",
        "\\crEFiveAdherenceFlashLitePressureTwoOneCi": "[-0.57, -0.21]",
        "\\crEFiveAdherenceFlashLitePressureTwoOneStepShare": "0.88",
        "\\crEFiveAdherencePooledPressureTwoOneStepShare": "0.89",
        "\\crEFiveCoveragePooledPressureSixAprilFive": "1.42",
        "\\crEFiveCoveragePooledPressureSixAprilFiveCi": "[1.20, 1.64]",
        # P15: E6 SWE pilot.
        "\\crESixKbvFlashLiteFrac": "16/20",
        "\\crESixKbvProFrac": "5/20",
        "\\crESixKbvCappedProFrac": "4/20",
        "\\crESixKbvMiniFrac": "2/20",
        "\\crESixNoncompliantFlashLiteSingleShotFrac": "19/20",
        "\\crESixProSingleShotCapHits": "9",
        "\\crESixProSingleShotCapNoncompliant": "7",
        "\\crESixHarnessAllPassed": "16",
        "\\crESixRunsAdherenceTwo": "0",
        # B1 (ledger alias).
        "\\crOnsetPubByTurnTwoFrac": "239/329",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # P3 and P8 are covered by the existing typology and mechanism groups; they must still resolve.
    for name in ("\\crTypologyOmnibusP", "\\crTypologyOmnibusClusters", "\\crMechDeparturesN", "\\crMechAppearsInChecklistStrictFrac"):
        assert name in v, name


def test_sensitivity_group(tmp_path: Path) -> None:
    """X7: the sensitivity cells (follow-up A temperature, follow-up B rigor) resolve beside the April cells."""
    d = tmp_path / "csv"
    SONNET = "anthropic/claude-sonnet-4-6"

    def cell(section, model, cond, variant, n, k, lo, hi):
        return {"section": section, "model_id": model, "condition": cond, "variant": variant, "setting": "s", "source": "src",
                "judge_arm": "published", "judge_model": "j", "n_runs": n, "n_briefs": 38, "n_with_final_probe": n, "knows": n,
                "violates": k, "kbv_count": k, "kbv_rate": k / n, "ci_low": lo, "ci_high": hi, "n_truncated_probes": 0,
                "stray_rows_dropped": 0}

    _write(d / "sensitivity_cells.csv", [
        cell("temperature", FLASH, "multi_turn_neutral", "april", 76, 21, 0.1884, 0.3858),
        cell("temperature", FLASH, "multi_turn_neutral", "followup", 76, 18, 0.1554, 0.3436),
        cell("temperature", FLASH, "multi_turn_pressure", "april", 76, 58, 0.6564, 0.8446),
        cell("temperature", FLASH, "multi_turn_pressure", "followup", 76, 63, 0.7290, 0.8972),
        cell("temperature", FLASH, "checkpointed_pressure", "april", 76, 55, 0.6142, 0.8116),
        cell("temperature", FLASH, "checkpointed_pressure", "followup", 76, 45, 0.4798, 0.6956),
        cell("pressure_type", GPT, "multi_turn_pressure", "april", 76, 6, 0.0367, 0.1617),
        cell("pressure_type", GPT, "multi_turn_pressure_rigor", "followup", 22, 4, 0.0731, 0.3852),
        cell("pressure_type", SONNET, "multi_turn_pressure", "april", 76, 75, 0.9292, 0.9977),
        cell("pressure_type", SONNET, "multi_turn_pressure_rigor", "followup", 21, 20, 0.7733, 0.9915),
        cell("pressure_type", FLASH, "multi_turn_pressure", "april", 76, 58, 0.6564, 0.8446),
        cell("pressure_type", FLASH, "multi_turn_pressure_rigor", "followup", 38, 19, 0.3485, 0.6515),
    ])
    specs = m._sensitivity_specs()
    assert len(specs) == 3 * (4 + 3) + 3 * (4 + 3)
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crSensFlashLiteTempPressure": "82.9",
        "\\crSensFlashLiteTempPressureFrac": "63/76",
        "\\crSensFlashLiteTempPressureCi": "[72.9, 89.7]",
        "\\crSensFlashLiteTempPressureN": "76",
        "\\crSensFlashLiteTempAprilPressure": "76.3",
        "\\crSensFlashLiteTempAprilPressureFrac": "58/76",
        "\\crSensFlashLiteTempNeutralFrac": "18/76",
        "\\crSensFlashLiteTempCheckpointFrac": "45/76",
        "\\crSensFlashLiteTempAprilCheckpointFrac": "55/76",
        "\\crSensRigorGpt": "18.2",
        "\\crSensRigorGptFrac": "4/22",
        "\\crSensRigorGptCi": "[7.3, 38.5]",
        "\\crSensRigorGptN": "22",
        "\\crSensNoveltyGptFrac": "6/76",
        "\\crSensRigorSonnet": "95.2",
        "\\crSensRigorSonnetFrac": "20/21",
        "\\crSensNoveltySonnetFrac": "75/76",
        "\\crSensRigorFlashLite": "50.0",
        "\\crSensRigorFlashLiteFrac": "19/38",
        "\\crSensRigorFlashLiteCi": "[34.9, 65.2]",
        "\\crSensNoveltyFlashLite": "76.3",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # The April pressure cell of the temperature section and of the pressure-type section are the same numbers.
    assert v["\\crSensFlashLiteTempAprilPressure"] == v["\\crSensNoveltyFlashLite"]
    # No April-variant macro carries an N: the April denominators are the \crKbv* macros' business.
    assert "\\crSensNoveltyGptN" not in v
    assert "\\crSensFlashLiteTempAprilPressureN" not in v



# ---------------------------------------------------------------------------
# X8 groups: E4 gKBV per arm, surface gap, probe accuracy, B10, B16, judge sensitivity
# ---------------------------------------------------------------------------

SONNET_ID = "anthropic/claude-sonnet-4-6"
LLAMA_ID = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
OPUS_ID = "anthropic/claude-opus-4-6"


def test_gkbv_opus_arm_group(tmp_path: Path) -> None:
    """E4: structured_opus rows for Sonnet/Llama/Qwen, structured (Opus-judged) rows for the other four; one Holm family."""
    d = tmp_path / "csv"

    def summ(source, model, judge, cond, g, lo, hi, n):
        return {"source": source, "model_id": model, "judge_model": judge, "condition": cond, "n_runs": n,
                "mean_g_kbv": g, "g_kbv_ci_lo": lo, "g_kbv_ci_hi": hi}

    _write(d / "graded_severity_summary.csv", [
        summ("structured_opus", SONNET_ID, OPUS_ID, "multi_turn_pressure", 0.062234, 0.042025, 0.084242, 76),
        summ("structured_opus", SONNET_ID, OPUS_ID, "single_shot", 0.028947, 0.015351, 0.043871, 76),
        summ("structured_opus", LLAMA_ID, OPUS_ID, "multi_turn_pressure", 0.152412, 0.127283, 0.177010, 76),
        # The GPT-judged Sonnet row of source structured must not feed the Opus arm.
        summ("structured", SONNET_ID, GPT, "multi_turn_pressure", 0.138252, 0.109616, 0.166719, 76),
        summ("structured", GPT, OPUS_ID, "multi_turn_pressure", 0.039411, 0.023935, 0.057898, 76),
    ])

    def test_row(source, model, delta, p, holm, sig):
        return {"source": source, "model_id": model, "judge_model": OPUS_ID, "n_briefs_paired": 38, "mean_brief_delta": delta,
                "wilcoxon_p": p, "holm_p_opus_arm": holm, "significant_holm_05_opus_arm": sig, "n_models_in_opus_arm_family": 3}

    _write(d / "graded_severity_opus_arm_tests.csv", [
        test_row("structured_opus", LLAMA_ID, 0.114192, 0.000008, 0.000024, 1),
        test_row("structured_opus", SONNET_ID, 0.033286, 0.060265, 0.060265, 0),
        test_row("structured", GPT, 0.024373, 0.011109, 0.022218, 1),
    ])
    specs = m._gkbv_opus_arm_specs()
    assert len(specs) == 7 * (2 * 3 + 4) + 2
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crGkbvOpusArmSonnetPressure": "6.2",
        "\\crGkbvOpusArmSonnetPressureCi": "[4.2, 8.4]",
        "\\crGkbvOpusArmSonnetPressureN": "76",
        "\\crGkbvOpusArmSonnetSingleShot": "2.9",
        "\\crGkbvOpusArmLlamaPressure": "15.2",
        "\\crGkbvOpusArmGptPressure": "3.9",
        "\\crGkbvOpusArmDeltaLlama": "0.114",
        "\\crGkbvOpusArmRawPLlama": "$p < 0.001$",
        "\\crGkbvOpusArmHolmPLlamaVal": "$<$0.001",
        "\\crGkbvOpusArmRawPSonnet": "$p = 0.060$",
        "\\crGkbvOpusArmHolmPGpt": "$p = 0.022$",
        "\\crGkbvOpusArmPairedBriefsGpt": "38",
        "\\crGkbvOpusArmHolmFamily": "3",
        "\\crGkbvOpusArmHolmSignificant": "2",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # Models without a test row produce no test macro.
    assert "\\crGkbvOpusArmRawPQwen" not in v


def test_surface_gap_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    rows = []
    for arm, model, cond, sfg, lo, hi in (
        ("opus", LLAMA_ID, "multi_turn_pressure", 0.802632, 0.664309, 0.947368),
        ("opus", SONNET_ID, "multi_turn_pressure", -0.296053, -0.414474, -0.171053),
        ("opus", "all", "multi_turn_pressure", 0.083962, 0.036721, 0.133716),
        ("published", SONNET_ID, "single_shot", 0.368421, 0.282730, 0.454112),
        ("gpt", LLAMA_ID, "multi_turn_pressure", 0.552632, 0.388158, 0.717105),
    ):
        rows.append({"judge_arm": arm, "judge_model": "j", "condition": cond, "model_id": model, "n": 76, "n_briefs": 38,
                     "fidelity_mean": 3.0, "adherence_mean": 3.0, "coverage_mean": 3.0, "sfg": sfg, "sfg_se": 0.05,
                     "sfg_ci_low": lo, "sfg_ci_high": hi})
    _write(d / "surface_gap.csv", rows)
    specs = m._surface_gap_specs()
    assert len(specs) == 3 * 8 * 2 * 2
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crSfgOpusArmLlamaPressure": "0.80",
        "\\crSfgOpusArmLlamaPressureCi": "[0.66, 0.95]",
        "\\crSfgOpusArmSonnetPressure": "-0.30",
        "\\crSfgOpusArmAllPressure": "0.08",
        "\\crSfgPublishedArmSonnetSingleShot": "0.37",
        "\\crSfgGptArmLlamaPressureCi": "[0.39, 0.72]",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))


def test_probe_accuracy_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"

    def row(sample, scope, model, agg, n, mean, k=None, share=None, lo=None, hi=None):
        return {"sample": sample, "scope": scope, "model_id": model, "aggregation": agg, "n": n, "n_models": 7, "n_briefs": 38,
                "mean_accuracy": mean, "ci_low": lo, "ci_high": hi, "n_at_or_above_cut": k, "share_at_or_above_cut": share,
                "mean_constraint_recall": mean, "mean_banned_recall": 1.0, "n_final_probe_truncated": 0}

    _write(d / "probe_accuracy_summary.csv", [
        row("as_published", "multi_turn", "all", "run_mean", 1595, 0.972982, 1541, 0.966144, 0.964297, 0.980875),
        row("as_published", "multi_turn", "all", "model_mean", 1595, 0.972999, None, 0.966165),
        row("clean", "multi_turn_pressure", "all", "run_mean", 530, 0.965916, 507, 0.956604, 0.949452, 0.979947),
        row("clean_untruncated_probes", "multi_turn", "all", "run_mean", 1423, 0.999844, 1423, 1.0, 0.999528, 1.0),
        row("clean", "multi_turn_pressure", PRO, "run_mean", 76, 0.762307, 53, 0.697368),
        row("probe_file", "all_rows", "all", "run_mean", 1683, 0.974395, 1629, 0.967914, 0.966, 0.982),
    ])
    specs = m._probe_accuracy_specs()
    assert len(specs) == 3 * 2 * (6 + 7 * 3) + 2
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crProbeAccPooledAsPublished": "97.3",
        "\\crProbeAccPooledAsPublishedCi": "[96.4, 98.1]",
        "\\crProbeAccPooledAsPublishedN": "1{,}595",
        "\\crProbeAccShareAboveCutPooledAsPublishedFrac": "1{,}541/1{,}595",
        "\\crProbeAccModelMeanAsPublished": "97.3",
        "\\crProbeAccPooledPressureClean": "96.6",
        "\\crProbeAccShareAboveCutPooledPressureClean": "95.7",
        "\\crProbeAccShareAboveCutPooledCleanUntrunc": "100.0",
        "\\crProbeAccProPressureClean": "76.2",
        "\\crProbeAccShareAboveCutProPressureCleanFrac": "53/76",
        "\\crProbeAccPooledAllRowsFile": "97.4",
        "\\crProbeAccPooledAllRowsFileN": "1{,}683",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # The model_mean row has no count, so no Frac is built from it.
    assert not any(n.startswith("\\crProbeAccModelMean") and n.endswith("Frac") for n in v)


def test_length_fidelity_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"

    def row(name, group, est, lo, hi, p, n):
        return {"section": "fidelity_length", "name": name, "group": group, "estimate": est, "ci_low": lo, "ci_high": hi,
                "p_value": p, "n": n, "note": "x"}

    _write(d / "tokens_context_summary.csv", [
        row("ols fidelity pressure log_output_tokens: log_output_tokens", "open_weight", 0.249820, -0.379163, 0.878803, 0.4362894, 150),
        row("ols fidelity pressure log_output_tokens: log_output_tokens", "commercial", -0.442716, -0.828737, -0.056695, 0.02458472, 380),
        row("ols fidelity pressure_raw log_output_tokens: log_output_tokens", "open_weight", -0.032634, -0.844994, 0.779725, 0.9372417, 150),
        row("ols fidelity pressure log_output_tokens: complexity", "open_weight", -0.412757, -0.543028, -0.282487, 5.3e-10, 150),
        row("ols fidelity published log_final_words: log_final_words", "open_weight", 0.232546, 0.097860, 0.367232, 0.0007141, 602),
        row("ols fidelity published log_final_words: complexity", "all", -0.248025, -0.284298, -0.211752, 5.9e-41, 2116),
        # A row of another section with the same name must not be picked up.
        {"section": "pressure_tokens", "name": "ols fidelity pressure log_output_tokens: log_output_tokens", "group": "open_weight",
         "estimate": 9.0, "ci_low": 8.0, "ci_high": 10.0, "p_value": 0.5, "n": 1, "note": "x"},
    ])
    specs = m._length_fidelity_specs()
    assert len(specs) == 3 * (3 * 4 + 2 * 3)
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crLengthFidelityOpenWeight": "0.25",
        "\\crLengthFidelityOpenWeightCi": "[-0.38, 0.88]",
        "\\crLengthFidelityOpenWeightP": "$p = 0.436$",
        "\\crLengthFidelityOpenWeightPVal": "0.436",
        "\\crLengthFidelityOpenWeightN": "150",
        "\\crLengthFidelityCommercial": "-0.44",
        "\\crLengthFidelityCommercialP": "$p = 0.025$",
        "\\crLengthFidelityRawOpenWeightP": "$p = 0.937$",
        "\\crLengthFidelityComplexityOpenWeightP": "$p < 0.001$",
        "\\crLengthFidelityPubOpenWeight": "0.23",
        "\\crLengthFidelityPubOpenWeightN": "602",
        "\\crLengthFidelityPubComplexityAll": "-0.25",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    assert "\\crLengthFidelityComplexityOpenWeightN" not in {s.name for s in specs}


def test_sonnet_truncation_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"

    def row(sample, rule, cond, n, k, lo, hi):
        return {"sample": sample, "model_id": SONNET_ID, "rule": rule, "condition": cond, "n_runs": n, "n_truncated": k,
                "n_untruncated": n - k, "share_truncated": k / n, "ci_low": lo, "ci_high": hi}

    _write(d / "sonnet_truncation.csv", [
        # Gemini Pro rows share sample/rule/condition with the Sonnet rows; model_id tells them apart; listed first, so a
        # Sonnet filter without model_id would resolve to them.
        {**row("clean", "any_turn", "all_main", 303, 48, 0.12, 0.20), "model_id": "gemini/gemini-3.1-pro-preview"},
        {**row("clean", "final_turn", "all_main", 303, 2, 0.0, 0.02), "model_id": "gemini/gemini-3.1-pro-preview"},
        {**row("clean", "final_turn", "single_shot", 75, 2, 0.0, 0.09), "model_id": "gemini/gemini-3.1-pro-preview"},
        row("clean", "any_turn", "all_main", 304, 221, 0.674293, 0.773989),
        row("as_published", "any_turn", "all_main", 315, 226, 0.665351, 0.764329),
        row("clean", "final_turn", "all_main", 304, 125, 0.357316, 0.467269),
        row("as_published", "final_turn", "all_main", 315, 127, 0.350485, 0.458197),
        row("clean", "any_turn", "multi_turn_pressure", 76, 72, 0.872343, 0.979345),
        row("clean", "final_turn", "multi_turn_pressure", 76, 50, 0.545959, 0.754637),
    ])
    specs = m._sonnet_truncation_specs()
    assert len(specs) == 5 + 1 + 3 + 3 + 4 * 2 + 2 + 1 + 4
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crSonnetTruncatedClean": "221",
        "\\crSonnetTruncatedCleanFrac": "221/304",
        "\\crSonnetTruncatedCleanPct": "72.7",
        "\\crSonnetTruncatedCleanDen": "304",
        "\\crSonnetTruncatedCleanCi": "[67.4, 77.4]",
        "\\crSonnetTruncatedPubPct": "71.7",
        "\\crSonnetFinalCapCleanFrac": "125/304",
        "\\crSonnetFinalCapPubFrac": "127/315",
        "\\crSonnetFinalCapPubPct": "40.3",
        "\\crSonnetTruncatedCleanPressureFrac": "72/76",
        "\\crSonnetFinalCapCleanPressureFrac": "50/76",
        "\\crGeminiProCapCleanFrac": "48/303",
        "\\crGeminiProFinalCapClean": "2",
        "\\crGeminiProFinalCapCleanSingleShot": "2",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))


def test_judge_sensitivity_group(tmp_path: Path) -> None:
    d = tmp_path / "csv"

    def row(source, compared, cond, n, below3, mean, kpos, kbelow, kat3, kmoved):
        return {"judge_source": source, "compared_with": compared, "model_id": "all", "condition": cond, "judge_model": "j",
                "n_runs": n, "share_below_3": below3, "mean_adherence": mean, "n_published_kbv_positive": kpos,
                "n_published_kbv_below_3_under_judge": kbelow,
                "share_published_kbv_below_3_under_judge": (kbelow / kpos) if kpos else None,
                "n_published_kbv_at_3": kat3, "n_published_kbv_at_3_moved_below_3": kmoved,
                "share_published_kbv_at_3_moved_below_3": (kmoved / kat3) if kat3 else None}

    _write(d / "stretch_violate_by_judge.csv", [
        row("skeptical", "skeptical", "multi_turn_pressure", 304, 0.759868, 1.865132, 200, 174, 143, 124),
        row("skeptical", "skeptical", "single_shot", 302, 0.566225, 2.437086, 0, 0, 0, 0),
        row("published", "skeptical", "multi_turn_pressure", 304, 0.1875, 3.115132, 200, 57, 143, 0),
        row("judge_opus", "judge_opus", "multi_turn_pressure", 76, 0.026316, 3.223684, 75, 2, 31, 0),
        # A per-model row with the same keys otherwise must not be picked up.
        {**row("skeptical", "skeptical", "multi_turn_pressure", 76, 0.9, 1.0, 75, 70, 60, 55), "model_id": SONNET_ID},
    ])
    specs = m._judge_sensitivity_specs()
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crSvJudgeSkepticalPressureKbvBelowThree": "87.0",
        "\\crSvJudgeSkepticalPressureKbvBelowThreeFrac": "174/200",
        "\\crSvJudgeSkepticalPressureKbvAtThreeMoved": "86.7",
        "\\crSvJudgeSkepticalPressureKbvAtThreeMovedFrac": "124/143",
        "\\crSvJudgeSkepticalPressureBelowThree": "76.0",
        "\\crSvJudgeSkepticalPressureMeanAdherence": "1.87",
        "\\crSvJudgeSkepticalPressureN": "304",
        "\\crSvJudgeSkepticalSingleShotBelowThree": "56.6",
        "\\crSvJudgePublishedVsSkepticalPressureKbvBelowThreeFrac": "57/200",
        "\\crSvJudgeCrossedOpusPressureKbvBelowThreeFrac": "2/75",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    names = {s.name for s in specs}
    # Single-shot has no KBV-positive runs, so no KBV macro is registered for it.
    assert not any("SingleShotKbv" in n for n in names)
    # The published scores never move, so no moved-share macro exists for them.
    assert not any(n.startswith("\\crSvJudgePublishedVs") and "Moved" in n for n in names)
    # The crossed judges only scored single-shot and pressure.
    assert not any(n.startswith("\\crSvJudgeCrossedOpusNeutral") for n in names)


def test_monitoring_group(tmp_path: Path) -> None:
    """X10: KBV, adherence and interventions under monitoring resolve beside the unmonitored cell."""
    d = tmp_path / "csv"
    flash, sonnet = "gemini/gemini-3.1-flash-lite-preview", "anthropic/claude-sonnet-4-6"

    def cell(model, n_tr, n, k, lo, hi, adh, alo, ahi, inter, n_int, un_k, un_n, un_lo, un_hi, d_lo, d_hi, capped=0, final_capped=0):
        return {"model_id": model, "n_transcripts": n_tr, "n_judged": n, "n_with_final_probe": n, "kbv_count": k,
                "kbv_rate": k / n, "ci_low": lo, "ci_high": hi, "mean_adherence": adh, "adherence_ci_low": alo,
                "adherence_ci_high": ahi, "mean_complexity": 1.157895, "complexity_ci_low": 0.948245,
                "complexity_ci_high": 1.367544, "mean_interventions": inter, "n_with_intervention": n_int,
                "n_output_capped": capped, "n_final_output_capped": final_capped,
                "unmonitored_kbv_count": un_k, "unmonitored_n_with_final_probe": un_n, "unmonitored_kbv_rate": un_k / un_n,
                "unmonitored_ci_low": un_lo, "unmonitored_ci_high": un_hi, "kbv_diff": k / n - un_k / un_n,
                "kbv_diff_ci_low": d_lo, "kbv_diff_ci_high": d_hi}

    _write(d / "monitoring_cells.csv", [
        cell(flash, 38, 38, 19, 0.348499, 0.651501, 3.5, 3.333448, 3.666552, 2.184211, 32, 58, 76, 0.656386, 0.844607,
             -0.435165, -0.077813),
        cell(sonnet, 38, 38, 37, 0.865051, 0.995339, 2.894737, 2.767103, 3.02237, 3.578947, 38, 75, 76, 0.929163, 0.997674,
             -0.12233, 0.048452, capped=34, final_capped=21),
    ])
    specs = m._monitoring_specs()
    assert specs and all(not s.required for s in specs)
    assert {s.name for s in specs} <= {s.name for s in m.static_specs()}
    tables = m.run(tmp_path, tmp_path / "out", csv_dir=d, specs=specs)
    v = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    expected = {
        "\\crMonKbvFlashLite": "50.0",
        "\\crMonKbvFlashLiteFrac": "19/38",
        "\\crMonKbvFlashLiteCi": "[34.8, 65.2]",
        "\\crMonKbvFlashLiteN": "38",
        "\\crMonAdherenceFlashLite": "3.50",
        "\\crMonAdherenceFlashLiteCi": "[3.33, 3.67]",
        "\\crMonComplexityFlashLite": "1.16",
        "\\crMonInterventionsFlashLite": "2.2",
        "\\crMonInterventionRunsFlashLiteFrac": "32/38",
        "\\crMonKbvUnmonitoredFlashLite": "76.3",
        "\\crMonKbvUnmonitoredFlashLiteFrac": "58/76",
        "\\crMonKbvDiffFlashLite": "-26.3",
        "\\crMonKbvDiffFlashLiteCi": "[-43.5, -7.8]",
        "\\crMonKbvSonnet": "97.4",
        "\\crMonAdherenceSonnet": "2.89",
        "\\crMonOutputCappedSonnetFrac": "34/38",
        "\\crMonFinalOutputCappedSonnetFrac": "21/38",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # Qwen and Llama have no row here: they land in the missing list, not in the tex.
    assert "\\crMonKbvQwen" not in v
    assert "\\crMonKbvQwen" in set(tables["cr_macros_missing"]["name"])


def test_checker_type_group(tmp_path: Path) -> None:
    """C11: per-checker-type coverage (TOTAL row) and checker-judge agreement (checker_type rows)."""
    d = tmp_path / "csv"
    types = list(m.CHECKER_TYPES)
    total = {"brief_id": "TOTAL", "n_hard_constraints": 234, "n_checkable": 104} | {f"n_{t}": 0 for t in types}
    total |= {"n_numeric_cap_count": 37, "n_threshold_restatement": 7}
    _write(d / "checker_coverage.csv", [{**total, "brief_id": "agri_01", "n_numeric_cap_count": 1}, total])
    comp = "structured_judge:value_pairs:not_satisfied=stretched+violated"
    _write(d / "checker_agreement.csv", [
        {"comparison": comp, "group": "overall", "level": "all", "n_pairs": 2981, "agreement": 0.9634},
        {"comparison": comp, "group": "checker_type", "level": "numeric_cap_count", "n_pairs": 824, "agreement": 0.951456,
         "kappa": 0.037327},
        {"comparison": "structured_judge:value_pairs:not_satisfied=violated_only", "group": "checker_type",
         "level": "numeric_cap_count", "n_pairs": 824, "agreement": 0.991505},
    ])
    specs = m._checker_type_specs()
    assert all(not s.required for s in specs)
    names = {s.name for s in specs}
    assert names <= {s.name for s in m.static_specs()}
    # Restatement-only constraints yield no value pairs, so no agreement macro is registered for them.
    assert "\\crCheckTypeThresholdRestatementAgree" not in names
    v = _values(tmp_path, d, specs)
    assert v["\\crCheckTypeNumericCapCountCheckable"] == "37"
    assert v["\\crCheckTypeNumericCapCountFrac"] == "37/234"
    assert v["\\crCheckTypeNumericCapCountFracCheckable"] == "37/104"
    assert v["\\crCheckTypeThresholdRestatementCheckable"] == "7"
    assert v["\\crCheckTypeNumericCapCountAgree"] == "95.1"
    assert v["\\crCheckTypeNumericCapCountPairs"] == "824"
    assert v["\\crCheckTypeNumericCapCountKappa"] == "0.04"


# ---------------------------------------------------------------------------
# X9 fix and X11: mechanism hand-check arms, token coverage, follow-up run counts
# ---------------------------------------------------------------------------


def test_mechanism_handcheck_group(tmp_path: Path) -> None:
    """X9: agreement macros for appears/verdict, LLM departs labels overall and per judge arm, flagged count."""
    d = tmp_path / "csv"
    labels = ("yes", "no", "no_checklist", "unclear", "compliant", "hedged", "flagged_noncompliant", "absent")

    def row(comparison, n, n_agree, kappa, b=None, a=None):
        r = {"comparison": comparison, "rater_a": "x", "rater_b": "llm", "n_rows": n, "n": n, "n_missing": 0,
             "n_agree": n_agree, "frac_agree": n_agree / n, "pct_agree": 100 * n_agree / n, "kappa": kappa}
        r |= {f"a_n_{lab}": (a or {}).get(lab, 0) for lab in labels}
        r |= {f"b_n_{lab}": (b or {}).get(lab, 0) for lab in labels}
        return r

    _write(d / "mechanism_handcheck_agreement.csv", [
        row("appears", 50, 50, 1.0, b={"yes": 44, "no": 2, "no_checklist": 4}),
        row("appears_binary", 50, 50, 1.0),
        row("verdict", 50, 46, 0.7840172786177106, b={"compliant": 37, "hedged": 7, "absent": 6, "flagged_noncompliant": 0}),
        row("verdict_both_appear", 44, 40, 0.557788944723618, a={"hedged": 3}, b={"hedged": 7}),
        row("departs", 50, 6, 0.0, b={"yes": 6, "no": 19, "unclear": 25}),
        row("departs_gpt_arm", 23, 2, 0.0, b={"yes": 2, "no": 16, "unclear": 5}),
        row("departs_opus_arm", 27, 4, 0.0, b={"yes": 4, "no": 3, "unclear": 20}),
    ])
    specs = m._mechanism_handcheck_specs()
    names = {s.name for s in specs}
    assert all(not s.required for s in specs) and names <= {s.name for s in m.static_specs()}
    # The departs rows count LLM labels; no macro reads their n_agree as agreement.
    for gone in ("\\crMechHandcheckDeparts", "\\crMechHandcheckDepartsAgree", "\\crMechHandcheckDepartsN",
                 "\\crMechHandcheckDepartsFrac", "\\crMechHandcheckDepartsKappa"):
        assert gone not in names, gone
    assert not [s for s in specs if s.where.get("comparison", "").startswith("departs") and "n_agree" in s.column]
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crMechHandcheckAppearsFrac": "50/50",
        "\\crMechHandcheckAppearsKappa": "1.00",
        "\\crMechHandcheckVerdict": "92.0",
        "\\crMechHandcheckVerdictKappa": "0.78",
        "\\crMechHandcheckVerdictBothAppearFrac": "40/44",
        "\\crMechHandcheckLlmDepartsYesFrac": "6/50",
        "\\crMechHandcheckLlmDepartsNo": "19",
        "\\crMechHandcheckLlmDepartsUnclearFrac": "25/50",
        "\\crMechHandcheckDepartsYesGptArmFrac": "2/23",
        "\\crMechHandcheckDepartsNoGptArmFrac": "16/23",
        "\\crMechHandcheckDepartsUnclearGptArmFrac": "5/23",
        "\\crMechHandcheckDepartsYesOpusArmFrac": "4/27",
        "\\crMechHandcheckDepartsNoOpusArmFrac": "3/27",
        "\\crMechHandcheckDepartsUnclearOpusArmFrac": "20/27",
        "\\crMechHandcheckLlmFlaggedN": "0",
        "\\crMechHandcheckDetHedged": "3",
        "\\crMechHandcheckLlmHedged": "7",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))
    # Per-arm counts add up to the overall departs labels.
    for lab in ("Yes", "No", "Unclear"):
        gpt = v[f"\\crMechHandcheckDeparts{lab}GptArmFrac"].split("/")
        opus = v[f"\\crMechHandcheckDeparts{lab}OpusArmFrac"].split("/")
        assert int(gpt[0]) + int(opus[0]) == int(v[f"\\crMechHandcheckLlmDeparts{lab}"])
        assert int(gpt[1]) + int(opus[1]) == 50


def test_tokens_pressure_runs_reads_the_coverage_row(tmp_path: Path) -> None:
    """X11: \\crTokensPressureRuns is the n of the pressure coverage row, beside \\crTokensRuns."""
    d = tmp_path / "csv"
    _write(d / "tokens_context_summary.csv", [
        {"section": "coverage", "name": "runs with usage records", "group": "all", "estimate": 2116.0, "n": 2116},
        {"section": "coverage", "name": "pressure runs with usage records", "group": "multi_turn_pressure",
         "estimate": 530.0, "n": 530},
    ])
    specs = [s for s in m._tokens_specs() if s.name in ("\\crTokensRuns", "\\crTokensPressureRuns")]
    assert len(specs) == 2 and all(not s.required for s in specs)
    v = _values(tmp_path, d, specs)
    assert v == {"\\crTokensRuns": "2{,}116", "\\crTokensPressureRuns": "530"}


def test_followup_runs_group(tmp_path: Path) -> None:
    """X11: planned, completed and failed runs of follow-ups A and B (sensitivity_runs.csv)."""
    d = tmp_path / "csv"
    _write(d / "sensitivity_runs.csv", [
        {"followup": "followup_a", "section": "temperature", "planned": 304, "completed": 304, "failed": 0, "unaccounted": 0},
        {"followup": "followup_b", "section": "pressure_type", "planned": 228, "completed": 176, "failed": 52, "unaccounted": 0},
    ])
    specs = m._followup_runs_specs()
    assert len(specs) == 6 and all(not s.required for s in specs)
    v = _values(tmp_path, d, specs)
    assert v == {
        "\\crFollowupAPlannedRuns": "304", "\\crFollowupACompletedRuns": "304", "\\crFollowupAFailedRuns": "0",
        "\\crFollowupBPlannedRuns": "228", "\\crFollowupBCompletedRuns": "176", "\\crFollowupBFailedRuns": "52",
    }


X11_REAL_INPUTS = ("mechanism_handcheck_agreement.csv", "tokens_context_summary.csv", "sensitivity_runs.csv")


@pytest.mark.skipif(not all((REAL_CSV_DIR / f).exists() for f in X11_REAL_INPUTS), reason="camera-ready CSVs not on disk")
def test_real_generation_defines_the_x9_fix_and_x11_macros(tmp_path: Path) -> None:
    """The new macros resolve on the real CSVs with the values the review expects. Writes to tmp_path only."""
    specs = [s for s in m.static_specs() if s.csv in X11_REAL_INPUTS]
    v = _values(tmp_path, REAL_CSV_DIR, specs)
    expected = {
        "\\crMechHandcheckDepartsYesGptArmFrac": "2/23",
        "\\crMechHandcheckDepartsNoGptArmFrac": "16/23",
        "\\crMechHandcheckDepartsUnclearGptArmFrac": "5/23",
        "\\crMechHandcheckDepartsYesOpusArmFrac": "4/27",
        "\\crMechHandcheckDepartsNoOpusArmFrac": "3/27",
        "\\crMechHandcheckDepartsUnclearOpusArmFrac": "20/27",
        "\\crMechHandcheckLlmFlaggedN": "0",
        "\\crTokensPressureRuns": "530",
        "\\crFollowupBPlannedRuns": "228",
        "\\crFollowupBCompletedRuns": "176",
        "\\crFollowupBFailedRuns": "52",
        "\\crFollowupAPlannedRuns": "304",
        "\\crFollowupACompletedRuns": "304",
        "\\crFollowupAFailedRuns": "0",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))


# ---------------------------------------------------------------------------
# X12: Gemini Pro perfect-recall gKBV, checkable-constraint human-human ceiling
# ---------------------------------------------------------------------------


PRO_ID = "gemini/gemini-3.1-pro-preview"


def test_x12_gkbv_perfect_recall_and_checkable_ceiling(tmp_path: Path) -> None:
    """Pressure-only perfect-recall gKBV macros; ceiling range found from the rows, not from pair names."""
    d = tmp_path / "csv"
    _write(d / "graded_severity_summary.csv", [
        {"source": "structured", "model_id": PRO_ID, "condition": "multi_turn_pressure", "n_runs": 76, "mean_g_kbv": 0.026441,
         "g_kbv_ci_lo": 0.013158, "g_kbv_ci_hi": 0.04117, "n_perfect_recall": 52, "mean_g_kbv_perfect_recall": 0.019231},
        {"source": "structured", "model_id": PRO_ID, "condition": "single_shot", "n_runs": 75, "mean_g_kbv": 0.013,
         "g_kbv_ci_lo": 0.004, "g_kbv_ci_hi": 0.023, "n_perfect_recall": 0, "mean_g_kbv_perfect_recall": ""},
        {"source": "structured_opus", "model_id": PRO_ID, "condition": "multi_turn_pressure", "n_runs": 99, "mean_g_kbv": 0.5,
         "g_kbv_ci_lo": 0.4, "g_kbv_ci_hi": 0.6, "n_perfect_recall": 99, "mean_g_kbv_perfect_recall": 0.5},
    ])
    ceiling = "human_human_ceiling:not_satisfied=stretched+violated"
    rows = []
    for pair, k_all, k_chk, n_chk in (("A-B", 0.224, 0.4437, 151), ("A-C", 0.594, 0.6529, 309), ("A-D", 0.315, 0.2161, 158),
                                      ("B-C", 0.465, 0.7321, 151), ("C-D", 0.328, 0.2793, 158)):
        rows.append({"comparison": ceiling, "group": "rater_pair", "level": pair, "n_pairs": 300, "kappa": k_all, "agreement": 0.8})
        rows.append({"comparison": ceiling, "group": "rater_pair_checkable_constraints", "level": pair, "n_pairs": n_chk,
                     "kappa": k_chk, "agreement": 0.9})
    _write(d / "checker_agreement.csv", rows)
    specs = [s for s in m._graded_severity_specs() if s.csv == "graded_severity_summary.csv"]
    specs += [s for s in m._checker_specs() if "HumanHumanCeiling" in s.name]
    # Only the pressure condition carries the perfect-recall and N macros.
    names = {s.name for s in specs}
    assert "\\crGkbvProPressurePerfectRecall" in names
    assert not any(n.endswith(("SingleShotN", "NeutralN", "CheckpointN")) or ("PerfectRecall" in n and "Pressure" not in n)
                   for n in names)
    v = _values(tmp_path, d, specs)
    expected = {
        "\\crGkbvProPressure": "2.6",
        "\\crGkbvProPressureN": "76",
        "\\crGkbvProPressurePerfectRecall": "1.9",
        "\\crGkbvProPressurePerfectRecallN": "52",
        "\\crHumanHumanCeilingCheckableKappaMin": "0.22",
        "\\crHumanHumanCeilingCheckableKappaMinPair": "A-D",
        "\\crHumanHumanCeilingCheckableKappaMax": "0.73",
        "\\crHumanHumanCeilingCheckableKappaMaxPair": "B-C",
        "\\crHumanHumanCeilingCheckableKappa": "0.46",
        "\\crHumanHumanCeilingCheckableRaterPairs": "5",
        "\\crHumanHumanCeilingCheckablePairs": "927",
        # The all-constraints macros are unchanged and still read the rater_pair rows.
        "\\crHumanHumanCeilingKappaAB": "0.22",
        "\\crHumanHumanCeilingRaterPairs": "5",
        "\\crHumanHumanCeilingPairs": "1{,}500",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))


X12_REAL_INPUTS = ("graded_severity_summary.csv", "checker_agreement.csv")


@pytest.mark.skipif(not all((REAL_CSV_DIR / f).exists() for f in X12_REAL_INPUTS), reason="camera-ready CSVs not on disk")
def test_real_generation_defines_the_x12_macros(tmp_path: Path) -> None:
    """The X12 macros resolve on the real CSVs with the values the review expects. Writes to tmp_path only."""
    names = {"\\crGkbvProPressureN", "\\crGkbvProPressurePerfectRecall", "\\crGkbvProPressurePerfectRecallN"}
    specs = [s for s in m.static_specs() if s.name in names or "HumanHumanCeilingCheckable" in s.name]
    v = _values(tmp_path, REAL_CSV_DIR, specs)
    expected = {
        "\\crGkbvProPressureN": "76",
        "\\crGkbvProPressurePerfectRecall": "1.9",
        "\\crGkbvProPressurePerfectRecallN": "52",
        "\\crHumanHumanCeilingCheckableKappaMin": "0.22",
        "\\crHumanHumanCeilingCheckableKappaMinPair": "A-D",
        "\\crHumanHumanCeilingCheckableKappaMax": "0.73",
        "\\crHumanHumanCeilingCheckableKappaMaxPair": "B-C",
        "\\crHumanHumanCeilingCheckableKappa": "0.46",
        "\\crHumanHumanCeilingCheckableRaterPairs": "5",
        "\\crHumanHumanCeilingCheckablePairs": "927",
    }
    for name, value in expected.items():
        assert v.get(name) == value, (name, v.get(name))


# ---------------------------------------------------------------------------
# X14: providers among the Opus-arm models above 50%
# ---------------------------------------------------------------------------

LLAMA_ID = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
QWEN_ID = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
SONNET_ID = "anthropic/claude-sonnet-4-6"
FLASH_ID = "gemini/gemini-3.1-flash-lite-preview"


def test_model_provider_maps_every_subject_model_to_five_providers() -> None:
    providers = {mid: m.model_provider(mid) for mid in m.MODEL_LABELS}
    assert providers[LLAMA_ID] == "Meta"
    assert providers[QWEN_ID] == "Alibaba"
    assert providers[FLASH_ID] == providers[PRO_ID] == "Google"
    assert providers["openai/gpt-5.4-mini"] == providers[GPT] == "OpenAI"
    assert len(set(providers.values())) == 5
    # Gemini under another prefix is still Google; the host prefix alone names no developer.
    assert m.model_provider("google/gemini-3.1-pro") == m.model_provider("Vertex_AI/gemini-3.1-pro") == "Google"
    for bad in ("together_ai/mistralai/x", "vertex_ai/claude-x", "", "llama"):
        with pytest.raises(m.MacroError, match="no provider"):
            m.model_provider(bad)


def test_nproviders_format_counts_distinct_providers(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    _write(d / "kbv_common_judge_range.csv", [
        {"judge_arm": "opus", "variant": "clean", "n_models": 7, "cut": 0.5, "n_above_cut": 4,
         "models_above_cut": ";".join([LLAMA_ID, FLASH_ID, SONNET_ID, QWEN_ID])},
        {"judge_arm": "gpt", "variant": "clean", "n_models": 7, "cut": 0.5, "n_above_cut": 3,
         "models_above_cut": f"{FLASH_ID};{PRO_ID};google/gemini-x"},
        {"judge_arm": "published", "variant": "clean", "n_models": 7, "cut": 0.5, "n_above_cut": 0, "models_above_cut": ""},
    ])
    specs = [s for s in m._crossed_judge_specs() if s.name == "\\crKbvOpusArmAboveHalfProviders"]
    assert len(specs) == 1 and specs[0].where == {"judge_arm": "opus", "variant": "clean"}
    assert specs[0].fmt == "nproviders" and not specs[0].required
    specs += [
        m.MacroSpec("\\crGeminiOnly", "kbv_common_judge_range.csv", {"judge_arm": "gpt"}, "models_above_cut", "nproviders"),
        m.MacroSpec("\\crNone", "kbv_common_judge_range.csv", {"judge_arm": "published"}, "models_above_cut", "nproviders",
                    required=False),
    ]
    tables = m.run(tmp_path, tmp_path / "out", csv_dir=d, specs=specs)
    v = dict(zip(tables["cr_macros"]["name"], tables["cr_macros"]["value"]))
    assert v["\\crKbvOpusArmAboveHalfProviders"] == "4"
    assert v["\\crGeminiOnly"] == "1"
    assert list(tables["cr_macros_missing"]["name"]) == ["\\crNone"]
    # The record round-trips, so --verify re-resolves the same value.
    assert m.verify_against_csvs(tmp_path / "out" / "cr_macros.json", d) == []


@pytest.mark.skipif(not (REAL_CSV_DIR / "kbv_common_judge_range.csv").exists(), reason="camera-ready CSVs not on disk")
def test_real_opus_arm_above_half_set_and_providers(tmp_path: Path) -> None:
    """Section 4 names the four Opus-arm models above 50%; the set and the provider count must match the CSV."""
    names = {"\\crKbvOpusArmAboveHalfFrac", "\\crKbvOpusArmAboveHalfProviders"}
    specs = [s for s in m.static_specs() if s.name in names]
    v = _values(tmp_path, REAL_CSV_DIR, specs)
    assert v == {"\\crKbvOpusArmAboveHalfFrac": "4/7", "\\crKbvOpusArmAboveHalfProviders": "4"}
    df = m.read_csv_str(REAL_CSV_DIR / "kbv_common_judge_range.csv")
    row = df[(df["judge_arm"] == "opus") & (df["variant"] == "clean")]
    assert set(row["models_above_cut"].iloc[0].split(";")) == {LLAMA_ID, FLASH_ID, SONNET_ID, QWEN_ID}


def test_sv_constraint_opus_arm_sums_opus_rows_only(tmp_path: Path) -> None:
    """X15 (C3): constraint-level stretched/violated counts sum the Opus-judged rows of both
    sources; the GPT-judged Sonnet row and other conditions do not enter."""
    d = tmp_path / "csv"

    def row(source, model, judge, cond, checks, stretched, violated):
        return {"source": source, "model_id": model, "judge_model": judge, "condition": cond,
                "total_checks": checks, "total_stretched": stretched, "total_violated": violated}

    _write(d / "graded_severity_summary.csv", [
        row("structured", GPT, OPUS_ID, "multi_turn_pressure", 468, 18, 0),
        row("structured_opus", SONNET_ID, OPUS_ID, "multi_turn_pressure", 468, 28, 1),
        row("structured_opus", LLAMA_ID, OPUS_ID, "multi_turn_pressure", 468, 62, 9),
        # Excluded: GPT-5.4 judges Sonnet under source structured; neutral is not an arm condition.
        row("structured", SONNET_ID, GPT, "multi_turn_pressure", 468, 56, 9),
        row("structured", GPT, OPUS_ID, "multi_turn_neutral", 468, 13, 0),
        row("structured", GPT, OPUS_ID, "single_shot", 468, 7, 0),
        row("structured_opus", SONNET_ID, OPUS_ID, "single_shot", 468, 13, 0),
    ])
    specs = m._sv_constraint_opus_arm_specs()
    assert len(specs) == 6
    assert all(not s.required for s in specs)
    v = _values(tmp_path, d, specs)
    assert v["\\crSvConstraintOpusArmChecksPressure"] == "1{,}404"
    assert v["\\crSvConstraintOpusArmStretchedPressure"] == "108"
    assert v["\\crSvConstraintOpusArmViolatedPressure"] == "10"
    assert v["\\crSvConstraintOpusArmChecksSingleShot"] == "936"
    assert v["\\crSvConstraintOpusArmStretchedSingleShot"] == "20"
    assert v["\\crSvConstraintOpusArmViolatedSingleShot"] == "0"
