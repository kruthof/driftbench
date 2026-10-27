"""Tests for drift_bench.analysis.camera_ready.checkers. Synthetic inputs only."""
from __future__ import annotations

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import checkers as ck

S, V, U = ck.SATISFIED, ck.VIOLATED, ck.UNDETERMINED


def entry(*clauses, **kw):
    e = {"brief_id": "t", "index": 0, "constraint": "x", "checker_type": "t", "checkable": True, "clauses": list(clauses)}
    e.update(kw)
    return e


def check(text, *clauses):
    return ck.check_constraint(text, entry(*clauses))


def verdict(text, *clauses):
    return check(text, *clauses)["verdict"]


# ---------------------------------------------------------------- classifier

@pytest.mark.parametrize("text,ctype,checkable", [
    ("budget under 50000 GBP", "numeric_cap_budget", True),
    ("minimum sample size of 30 clusters", "numeric_cap_count", True),
    ("between-subjects design with exactly 2 conditions (expanding vs uniform spacing)", "exact_count", True),
    ("sample size between 80 and 200 undergraduate participants", "numeric_range", True),
    ("retention test at exactly 30 days post-learning, no intermediate tests", "numeric_cap_duration", True),
    ("analysis must cover at least 12 consecutive months", "numeric_cap_duration", True),
    ("must report transit depth precision per spectral bin of 50 ppm or better", "numeric_cap_quantity", True),
    ("Film thickness held at 50 +/- 5 micrometers for all specimens", "numeric_range", True),
    ("analysis window must be 2015-2023 inclusive", "numeric_cap_duration", True),
    ("the worker pool must be capped at 16 concurrent workers", "numeric_cap_count", True),
    ("no use of ground-based photometry or spectroscopy as supplementary data", "forbidden_term", False),
    ("must include state and quarter fixed effects", "required_term", False),
    ("sampling protocol must use pan traps or netting, not both combined", "enumerated_choice", False),
    ("energy cost burden defined as annual electricity expenditure divided by annual household income", "none", False),
    ("retrospective cohort design only, no randomization of exposure", "none", False),
])
def test_classifier(text, ctype, checkable):
    lab = ck.classify_constraint(text)
    assert lab["checker_type"] == ctype
    assert lab["checkable"] is checkable


def test_classifier_types_are_in_the_fixed_set():
    for text in ["budget under 1 USD", "no X", "must Y", "anything else"]:
        assert ck.classify_constraint(text)["checker_type"] in ck.ALL_TYPES


# ---------------------------------------------------------------- text preparation

def test_strip_alternatives_removes_only_that_section():
    text = ("## Minimal viable design\nWe recruit 150 participants.\n"
            "## Two plausible alternatives not chosen\n1. A 300-participant design.\n2. A three-condition design.\n"
            "## Final constraint checklist\n- Sample size: met\n")
    out, stripped = ck.strip_alternatives(text)
    assert stripped
    assert "300-participant" not in out
    assert "150 participants" in out and "Final constraint checklist" in out


def test_strip_alternatives_refuses_a_cut_over_half_the_text():
    text = "Alternative explanation of the confound\n" + "Design text with 150 participants.\n" * 20
    out, stripped = ck.strip_alternatives(text)
    assert not stripped and out == text


def test_normalise_text_maps_typography():
    assert ck.normalise_text("**80–200** ≥ 3 ± 2") == "80-200 >= 3 +/- 2"


def test_normalise_text_maps_latex_comparators_and_math_spans():  # item 5
    assert ck.normalise_text(r"counties with $\ge$ 20 cases") == "counties with >= 20 cases"
    assert ck.normalise_text(r"$N \geq 150$ patients, $\sigma \leq 50$ ppm") == "N >= 150 patients, \\sigma <= 50 ppm"
    assert ck.normalise_text(r"$N=20$ per arm") == "N=20 per arm"
    assert ck.normalise_text("costs $20,000 and $30,000") == "costs $20,000 and $30,000"  # currency is not a math span


def test_latex_bound_is_a_bound_not_a_value():  # item 5
    mn = {"kind": "count", "comparator": "min", "limit": 20, "noun": r"counties", "scope": "total"}
    res = check(r"Sample: counties with $\ge$ 3 counties is too few; we require $\ge$ 20 counties.", mn)
    assert res["verdict"] == U and res["evidence_kind"] == "bound"


# ---------------------------------------------------------------- count checkers

MIN_SITES = {"kind": "count", "comparator": "min", "limit": 8, "noun": r"sites|farms", "scope": "total"}


def test_count_min_positive_negative_undetermined():
    assert verdict("We will sample 12 working farms across two regions.", MIN_SITES) == S
    assert verdict("We will sample 6 sites across two regions.", MIN_SITES) == V
    assert verdict("We will sample several sites across two regions.", MIN_SITES) == U


def test_restated_bound_is_undetermined_with_a_bound_verdict():  # item 4
    res = check("Minimum of 8 paired sites: met.", MIN_SITES)
    assert res["verdict"] == U and res["evidence_kind"] == "bound"
    assert res["undetermined_reason"] == "restatement_only" and res["bound_verdict"] == S
    res = check("Up to 6 sites will be used.", MIN_SITES)
    assert res["verdict"] == U and res["bound_verdict"] == V


def test_value_outranks_restated_bound():  # item 1
    assert verdict("Minimum of 8 paired sites: met. We will sample 6 sites.", MIN_SITES) == V
    assert verdict("At least 8 sites. We will sample 12 sites.", MIN_SITES) == S
    budget = {"kind": "budget", "comparator": "max", "limit": 50000, "currency": "GBP", "strict": True}
    res = check("Total: £51,729. Budget under £50,000: met.", budget)
    assert res["verdict"] == V and res["evidence_kind"] == "value" and res["bound_verdict"] == S


def test_count_min_takes_the_largest_value_as_the_total():
    assert verdict("We use 10 sites: 5 sites on clay and 5 sites on sand.", MIN_SITES) == S


def test_count_max_exact_and_range():
    mx = {"kind": "count", "comparator": "max", "limit": 2, "noun": r"conditions?", "scope": "total"}
    ex = {"kind": "count", "comparator": "exact", "limit": 2, "noun": r"conditions?", "scope": "total"}
    rg = {"kind": "count", "comparator": "range", "low": 80, "high": 200, "noun": r"participants", "scope": "total", "allow_n": True}
    assert verdict("A design with three conditions.", mx) == V
    assert verdict("A design with two conditions.", mx) == S
    assert verdict("A design with exactly 3 conditions.", ex) == V
    assert verdict("Participants: N = 150.", rg) == S
    assert verdict("We recruit 60 participants.", rg) == V
    assert verdict("We recruit 60-100 participants.", rg) == V
    res = check("We recruit 100-180 participants.", rg)
    assert res["verdict"] == S and res["evidence_kind"] == "value_range"


def test_count_conflicting_values_are_undetermined():
    ex = {"kind": "count", "comparator": "exact", "limit": 2, "noun": r"conditions?", "scope": "total"}
    res = check("Two conditions in the main study. Three conditions in a follow-up.", ex)
    assert res["verdict"] == U and res["undetermined_reason"] == "conflict"


def test_count_ignores_ordinals_decimals_identifiers_and_bare_one():
    mx = {"kind": "count", "comparator": "max", "limit": 2, "noun": r"conditions?|arms?", "scope": "total"}
    assert verdict("No 3rd condition is added.", mx) == U
    assert verdict("We expect d = 0.20 between arms.", mx) == U
    assert verdict("R1 excludes arms with attrition.", mx) == U
    assert verdict("One arm could be more engaging.", mx) == U
    assert verdict("Exactly one arm receives the intervention.", mx) == S


def test_count_per_group_scope():
    pg = {"kind": "count", "comparator": "min", "limit": 200, "noun": r"participants", "scope": "per_group", "group_noun": r"arms?|conditions?", "allow_n": True}
    assert verdict("We enrol 250 participants per arm.", pg) == S
    assert verdict("Participants: 300 (150 per arm).", pg) == V
    assert verdict("We enrol 600 participants in total.", pg) == U


def test_pilot_samples_are_ignored():
    rg = {"kind": "count", "comparator": "range", "low": 800, "high": 2000, "noun": r"participants|users", "scope": "total", "allow_n": True}
    assert verdict("A pilot test (n=25) precedes the study.", rg) == U


def test_years_are_not_counts():  # item 7
    mn = {"kind": "count", "comparator": "min", "limit": 20, "noun": r"counties", "scope": "total", "allow_n": True}
    assert verdict("A 2015-2023 panel of counties.", mn) == U
    assert verdict("Counties observed 2015 to 2023 quarterly.", mn) == U
    assert verdict("Panel: 2019 quarterly observations of counties.", mn) == U
    hh = {"kind": "count", "comparator": "min", "limit": 1000, "noun": r"households", "scope": "total"}
    assert verdict("We enrol 2000 households.", hh) == S


# ---------------------------------------------------------------- budget

BUDGET = {"kind": "budget", "comparator": "max", "limit": 50000, "currency": "GBP", "strict": True,
          "exclude_after_values": r"^\s*(?:hard\s+)?(?:cap|limit|ceiling)", "exclude_before": r"exceed(?:s|ing|ed)?\s+(?:the\s+)?$"}


def test_budget_uses_the_largest_amount_in_the_right_currency():
    assert verdict("Field costs £20,000; lab costs £15,000; total £45,000.", BUDGET) == S
    assert verdict("Lab costs £15,000; total £52,000.", BUDGET) == V
    assert verdict("Total cost 45,000 USD.", BUDGET) == U
    assert verdict("Total cost 45k GBP.", BUDGET) == S


def test_budget_strict_cap_compares_exactly():  # item 2
    assert verdict("Total: £50,000.", BUDGET) == V
    assert verdict("Total: £49,944.", BUDGET) == S
    assert verdict("Total: £50,001.", BUDGET) == V
    inclusive = {**BUDGET, "strict": False}
    assert verdict("Total: £50,000.", inclusive) == S


def test_budget_multiplier_needs_a_whole_token():  # item 3
    assert verdict("£20,000 materials and £10,000 labour.", BUDGET) == S
    assert verdict("£20,000 million would be absurd.", BUDGET) == V
    assert verdict("£45k total.", BUDGET) == S
    assert verdict("£30,000 bn", BUDGET) == V


def test_budget_restated_cap_is_bound_evidence():
    res = check("This stays under the £50,000 cap.", BUDGET)
    assert res["verdict"] == U and res["evidence_kind"] == "bound" and res["bound_verdict"] == S
    assert check("The £50,000 cap applies.", BUDGET)["evidence_kind"] == "none"
    assert check("This exceeds the £50,000 limit.", BUDGET)["evidence_kind"] == "none"


# ---------------------------------------------------------------- duration

def test_duration_exact_with_unit_conversion_tolerance_and_context():  # item 2: tolerance only here
    d = {"kind": "duration", "comparator": "exact", "unit": "weeks", "limit": 2, "context": r"follow-?up", "units": ["weeks", "days"]}
    assert verdict("Follow-up at 14 days.", d) == S
    assert verdict("Follow-up at 3 weeks.", d) == V
    assert verdict("Sessions last 2 weeks.", d) == U  # no follow-up context near the mention
    m = {"kind": "duration", "comparator": "exact", "unit": "months", "limit": 6, "units": ["months", "weeks"]}
    assert verdict("Measured over 26 weeks.", m) == S  # 182 days vs 182.6: within the duration tolerance


def test_duration_context_window_is_local():
    d = {"kind": "duration", "comparator": "range", "unit": "hours", "low": 7, "high": 9, "context": r"control", "units": ["hours"], "context_window": 30}
    long = "The deprived group stays awake for 36 hours while, after a long clause of many words, the control group sleeps 8 hours."
    assert verdict(long, d) == S


def test_duration_max_and_min_values():
    mx = {"kind": "duration", "comparator": "max", "unit": "minutes", "limit": 90, "units": ["minutes", "hours"]}
    assert verdict("The session lasts 2 hours.", mx) == V
    assert verdict("The session lasts 75 minutes.", mx) == S
    mn = {"kind": "duration", "comparator": "min", "unit": "months", "limit": 12, "units": ["months", "years"]}
    assert verdict("Data span 2 full calendar years.", mn) == S
    assert verdict("Data span 6 months.", mn) == V


def test_frequency():
    f = {"kind": "frequency", "comparator": "max", "limit": 7, "unit": "days", "context": r"CO2"}
    assert verdict("CO2 is measured weekly by titration.", f) == S
    assert verdict("CO2 is measured monthly.", f) == V
    assert verdict("CO2 is measured regularly.", f) == U


# ---------------------------------------------------------------- quantity, year, clock, dims

def test_quantity_with_units_and_prefix():
    q = {"kind": "quantity", "comparator": "max", "units": {r"km\b": 1, r"\bm\b": 0.001}, "limit": 200, "context": r"diameter"}
    assert verdict("A region of 150 km diameter.", q) == S
    assert verdict("A region of 250 km diameter.", q) == V
    assert verdict("A region of 200 km diameter.", q) == S  # inclusive cap, exact comparison
    age = {"kind": "quantity", "comparator": "range", "units": {r"years?": 1}, "low": 18, "high": 35, "prefix": r"aged?"}
    res = check("Healthy adults aged 18-35.", age)
    assert res["verdict"] == S and res["evidence_kind"] == "value_range"
    assert verdict("Healthy adults aged 18-45.", age) == V
    assert verdict("Healthy adults.", age) == U


def test_threshold_restatement_and_exclusions():
    t = {"kind": "quantity", "comparator": "exact", "units": {r"residents": 1}, "limit": 10000, "ignore_cmp": True,
         "exclude_before": r"per\s*$"}
    assert verdict("Exclude counties with fewer than 10,000 residents.", t) == S
    assert verdict("Exclude counties with fewer than 1,000 residents.", t) == V
    assert verdict("Rate per 100,000 residents.", t) == U


def test_year_range_exact_and_span():
    ex = {"kind": "year_range", "comparator": "exact", "low": 2015, "high": 2023}
    assert verdict("Panel 2015-2023.", ex) == S
    assert verdict("Panel 2012-2023.", ex) == V
    mx = {"kind": "year_range", "comparator": "max", "limit": 10}
    assert verdict("Analysis from 2010 to 2019.", mx) == S
    assert verdict("Analysis from 2010 to 2022.", mx) == V
    assert verdict("Analysis of a decade.", mx) == U


def test_clock_range_and_dims():
    c = {"kind": "clock_range", "low": 8, "high": 10}
    assert verdict("Testing between 08:00 and 10:00.", c) == S
    assert verdict("Testing between 09:00 and 11:00.", c) == V
    assert verdict("Testing in the morning.", c) == U
    d = {"kind": "dims", "dims": [5, 7], "unit": r"cm"}
    assert verdict("Electrodes of 7 x 5 cm.", d) == S
    assert verdict("Electrodes of 5 x 5 cm.", d) == V
    assert verdict("Standard sponge electrodes.", d) == U


# ---------------------------------------------------------------- multi-clause and agreement

def test_multi_clause_violation_then_conflict_then_satisfied():  # item 6
    c1 = {"kind": "count", "comparator": "min", "limit": 100, "noun": r"trials", "scope": "total"}
    c2 = {"kind": "count", "comparator": "exact", "limit": 5, "noun": r"layouts", "scope": "total"}
    assert verdict("200 trials across 3 layouts.", c1, c2) == V
    assert verdict("200 trials across 5 layouts.", c1, c2) == S
    assert verdict("200 trials in the test set.", c1, c2) == S
    res = check("200 trials across 5 layouts; a stress test uses 8 layouts.", c1, c2)
    assert res["verdict"] == U and res["undetermined_reason"] == "conflict"
    assert verdict("50 trials across 5 layouts; a stress test uses 8 layouts.", c1, c2) == V


def test_cohen_kappa_and_agreement_table():
    a = pd.Series(["satisfied", "satisfied", "not_satisfied", "not_satisfied"])
    assert ck.cohen_kappa(a, a) == pytest.approx(1.0)
    assert ck.cohen_kappa(a, a[::-1].reset_index(drop=True)) == pytest.approx(-1.0)
    pairs = pd.DataFrame({
        "verdict": [S, S, V, V, S], "ref": ["satisfied", "not_satisfied", "not_satisfied", "satisfied", "satisfied"],
        "checker_type": ["a", "a", "a", "b", "b"],
    })
    tab = ck.agreement_table(pairs, "ref", [["checker_type"]], "x")
    overall = tab[tab.group == "overall"].iloc[0]
    assert overall.n_pairs == 5 and overall.agreement == pytest.approx(3 / 5)
    assert overall.checker_sat_ref_not == 1 and overall.checker_viol_ref_sat == 1
    pairs["bound_verdict"] = [S, V, S, S, S]
    tab2 = ck.agreement_table(pairs, "ref", [], "y", verdict_col="bound_verdict")
    assert tab2.iloc[0].agreement == pytest.approx(4 / 5)


def test_match_checks_to_brief_exact_and_fuzzy():
    cons = ["paired design with at least 8 pairs", "removal method must be mechanical only"]
    checks = [{"constraint": "paired design with at least 8 pairs", "status": "satisfied"},
              {"constraint": "remove method must be mechanical only", "status": "stretched"},
              {"constraint": "something unrelated (BANNED)", "status": "violated"}]
    out = ck._match_checks_to_brief(checks, cons)
    assert out == [(0, "satisfied", "exact_text"), (1, "stretched", "fuzzy_text")]


def test_rules_file_covers_every_constraint_in_order():
    briefs = ck.load_briefs()
    rules = ck.load_rules()
    assert len(rules) == sum(len(v) for v in briefs.values())
    for e in rules:
        assert briefs[e["brief_id"]][e["index"]] == e["constraint"]
        assert e["checker_type"] in ck.ALL_TYPES
        assert e["checkable"] == (e["checker_type"] in ck.CHECKABLE_TYPES)
        if e["checkable"]:
            assert e["clauses"], e
        if e["checker_type"] == "threshold_restatement":
            assert any(c.get("ignore_cmp") for c in e["clauses"])
    assert "threshold_restatement" not in ck.AGREEMENT_TYPES


# ---------------------------------------------------------------- round 3

BUDGET_R3 = {"kind": "budget", "comparator": "max", "limit": 50000, "currency": "GBP", "strict": True,
             "exclude_after_values": r"^\s*(?:hard\s+)?(?:cap|limit|ceiling|maximum|threshold|constraint|budget\b)",
             "exclude_before": r"(?:exceed(?:s|ing|ed)?|above|over|beyond|breach(?:es|ing)?)\s+(?:the\s+)?$"}
HH_MIN = {"kind": "count", "comparator": "min", "noun": r"households", "limit": 1000, "scope": "total"}


def test_subsample_counts_are_not_totals():  # item 1(a)
    # a subset size on its own leaves only bound evidence
    r = check("We require at least 1,000 households. The training set (400 households) is held apart.", HH_MIN)
    assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only" and r["bound_verdict"] == S
    assert verdict("The calibration sample 300 households is drawn first.", HH_MIN) == U
    # the brief's regex has no "of": "calibration sample of 300 households" is still read as a value
    assert verdict("The calibration sample of 300 households is drawn first.", HH_MIN) == V
    assert verdict("A random subsample of 300 households is re-surveyed.", HH_MIN) == U
    assert verdict("We keep 20% of the households (400 households) as a hold-out fold.", HH_MIN) == U
    # the total itself still decides
    assert verdict("We enrol 2,000 households; the training split (800 households) is used for tuning.", HH_MIN) == S
    assert verdict("We enrol 600 households.", HH_MIN) == V


def test_subset_words_can_be_narrowed_per_clause():  # item 1(a), nlp_01/4
    c = dict(HH_MIN, subset_words=r"training|train")
    assert verdict("The held-out set (600 households) is scored.", HH_MIN) == U
    assert verdict("The held-out set (600 households) is scored.", c) == V
    assert verdict("The training set (600 households) is scored.", c) == U


def test_cap_echo_is_a_bound_not_a_value():  # item 1(b)
    r = check("Within a budget of £50,000 we run two waves.", BUDGET_R3)
    assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only" and r["bound_verdict"] == S
    r = check("The £50,000 budget covers staff and travel.", BUDGET_R3)
    assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only"
    for txt in ("The £50,000 GBP budget is allocated as follows.", "Budget (£50,000): staff and travel.", "Budget: £50,000 across two waves."):
        r = check(txt, BUDGET_R3)
        assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only", txt
    # a smaller value next to the cap echo still decides
    r = check("Within a budget of £50,000 the total comes to £42,000.", BUDGET_R3)
    assert r["verdict"] == S and r["n_value_mentions"] == 1 and r["n_bound_mentions"] == 1
    # an amount equal to the cap that is not an echo stays a value (and violates the strict cap)
    assert verdict("Total cost: £50,000.", BUDGET_R3) == V


def test_at_cap_flag():  # item 1(c)
    r = check("Total cost: £50,000.", BUDGET_R3)
    assert r["verdict"] == V and r["at_cap"] is True
    r = check("Total cost: approximately £50,000.", BUDGET_R3)
    assert r["verdict"] == V and r["at_cap"] is True
    r = check("Total cost: £52,000.", BUDGET_R3)
    assert r["verdict"] == V and r["at_cap"] is False
    r = check("Total cost: £48,000.", BUDGET_R3)
    assert r["verdict"] == S and r["at_cap"] is False
    # non-strict caps: equality satisfies, so at_cap stays false
    c = {"kind": "count", "comparator": "max", "noun": r"sites", "limit": 12, "scope": "total"}
    r = check("We use 12 sites.", c)
    assert r["verdict"] == S and r["at_cap"] is False


def test_nlp_01_0_is_a_range_rule():  # item 1(d)
    rules = {(e["brief_id"], e["index"]): e for e in ck.load_rules()}
    e = rules[("nlp_01", 0)]
    c = e["clauses"][0]
    assert c["comparator"] == "range" and c["low"] == 6.5 and c["high"] == 7.5
    assert ck.check_constraint("We use a frozen 7B decoder LLM.", e)["verdict"] == S
    assert ck.check_constraint("We use a frozen 6.7B decoder LLM.", e)["verdict"] == S
    assert ck.check_constraint("We use a frozen 13B decoder LLM.", e)["verdict"] == V
    for b in ("agri_01", "climate_01", "hci_01", "molbio_01", "pubhealth_01"):
        bc = next(e for (bid, _), e in rules.items() if bid == b and e["checker_type"] == "numeric_cap_budget")
        assert "budget\\b" in bc["clauses"][0]["exclude_after_values"]
    assert "held" not in rules[("nlp_01", 4)]["clauses"][0]["subset_words"]


def test_comparator_propagates_across_and():  # item 1(e)
    r = check("We need at least 1,000 treated and 1,000 control households.", HH_MIN)
    assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only" and r["bound_verdict"] == S
    assert verdict("We enrol 1,000 treated and 1,200 control households.", HH_MIN) == S
    d = {"kind": "duration", "comparator": "min", "unit": "years", "limit": 2}
    r = check("We take at least 2 years before and 2 years after the event.", d)
    assert r["verdict"] == U and r["undetermined_reason"] == "restatement_only" and r["bound_verdict"] == S
    assert verdict("We take 1 year before and 1 year after the event.", d) == V
    # no 'and' within reach: nothing is inherited
    assert verdict("At least 2 years of records are needed. We take 1 year.", d) == V
    # unit-aware (item 7): the earlier span must carry the clause's unit, so ">10cm lift and 3s hold" keeps 3 s as a value
    sec = {"kind": "duration", "comparator": "min", "unit": "seconds", "limit": 3}
    r = check("Grasp success: >10cm lift and 3s hold.", sec)
    assert r["verdict"] == S and r["evidence_kind"] == "value"
    r = check("Grasp success: >10cm lift and 2s hold.", sec)
    assert r["verdict"] == V
    # counts: the earlier span may be elliptical ("1,000 treated") or carry the noun
    r = check("We need at least 1,000 treated households and 1,000 control households.", HH_MIN)
    assert r["verdict"] == U and r["bound_verdict"] == S
    # a long unrelated earlier span does not carry the comparator
    assert verdict("We need at least 8 fully instrumented rural field sites and 1,000 households.", HH_MIN) == S


def test_restrict_to_parquet_runs(tmp_path):  # item 2
    pq = tmp_path / "all_scores.parquet"
    pd.DataFrame({"run_id": ["a", "b"], "x": [1, 2]}).to_parquet(pq)
    df = pd.DataFrame({"run_id": ["a", "a", "b", "c", "d"], "condition": ["s", "s", "p", "multi_turn_pressure_rigor", "s"]})
    out, info = ck.restrict_to_parquet_runs(df, pq)
    assert list(out.run_id) == ["a", "a", "b"]
    assert info["rows_in"] == 5 and info["rows_out"] == 3 and info["runs_dropped"] == 2
    assert info["runs_dropped_by_condition"] == {"multi_turn_pressure_rigor": 1, "s": 1}


def test_ceiling_rows_carry_note(tmp_path):  # item 3
    rd = tmp_path / "human_validation" / "raters"
    rd.mkdir(parents=True)
    import json
    for rid, labs in (("rater_A", ["Satisfied", "Violated"]), ("rater_B", ["Satisfied", "Satisfied"])):
        (rd / f"{rid}.json").write_text(json.dumps({"r1": {"rater_id": rid, "brief_id": "b", "constraint_statuses": labs}}))
    rules = [{"brief_id": "b", "index": 0, "checkable": True, "checker_type": "numeric_cap_count"},
             {"brief_id": "b", "index": 1, "checkable": True, "checker_type": "numeric_cap_count"}]
    t, info = ck.human_ceiling_table(tmp_path, {"b": ["x", "y"]}, rules, stray=set())
    assert set(t.level) == {"A-B"} and "note" in t.columns
    k = t[t.group == "rater_pair_checkable_constraints"].kappa.iloc[0]
    assert t.note.str.contains(f"kappa {k:.2f} to {k:.2f}").all() and t.note.str.contains("names A-B as the full-set pair").all()
    assert info["rows_dropped"] == 0 and info["rows_in"] == 2
    # a stray run is dropped from every rater
    t2, info2 = ck.human_ceiling_table(tmp_path, {"b": ["x", "y"]}, rules, stray={"r1"})
    assert len(t2) == 0 and info2["rows_dropped"] == 2


def test_structured_checks_prefer_scores_then_e4(tmp_path):  # round 4 item 1
    import json
    briefs = {"b": ["at least 3 sites", "budget under 100 GBP"]}
    (tmp_path / "scores").mkdir()
    (tmp_path / "openweight_subjects" / "scores").mkdir(parents=True)
    e4 = tmp_path / "e4"
    e4.mkdir()
    rec = lambda rid, model, st: {"run_id": rid, "brief_id": "b", "judge_model": model, "constraint_checks": [
        {"constraint": "at least 3 sites", "status": st}, {"constraint": "budget under 100 GBP", "status": "satisfied"}]}
    (tmp_path / "scores" / "structured_r1.jsonl").write_text(json.dumps(rec("r1", "gpt", "violated")) + "\n")
    (e4 / "structured_opus_r1.jsonl").write_text(json.dumps(rec("r1", "opus", "satisfied")) + "\n")
    d = rec("r2", "opus", "stretched")
    d["manual_override"] = {"note": "x"}
    (e4 / "structured_opus_r2.jsonl").write_text(json.dumps(d) + "\n")
    df, info = ck.load_structured_checks(tmp_path, briefs, e4_dir=e4)
    assert len(df) == 4
    r1 = df[(df.run_id == "r1") & (df.constraint_index == 0)].iloc[0]
    assert r1.judge_source == "scores" and r1.judge_status == "violated" and r1.judge_model == "gpt"
    r2 = df[(df.run_id == "r2") & (df.constraint_index == 0)].iloc[0]
    assert r2.judge_source == "e4_structured_opus" and r2.judge_status == "stretched"
    assert info["judge_sources"]["e4_structured_opus"]["n_files_skipped_already_covered"] == 1
    assert info["judge_sources"]["e4_structured_opus"]["n_files_used"] == 1


# --- set-1 items as the rater app showed them (R3) ---------------------------

_S1_BRIEFS = {"b": ["at least 3 sites", "budget under 100 GBP"], "c": ["exactly 2 arms"]}


def _s1_item(run_id, brief_id, text, constraints=None, **extra):
    it = {"run_id": run_id, "item_number": 1 if run_id == "r1" else 2, "brief_id": brief_id,
          "hard_constraints": list(_S1_BRIEFS[brief_id] if constraints is None else constraints),
          "final_proposal": text, "domain": "d", "objective": "o", "banned_moves": []}
    it.update(extra)
    return it


def _s1_pair():
    full = [_s1_item("r1", "b", "We use 4 sites.\n\n## Final Constraint Checklist\n- 4 sites: yes", judge_score=4, auditor_score=3),
            _s1_item("r2", "c", "Two arms, no self-check.", judge_score=2, auditor_score=2)]
    shown = [_s1_item("r1", "b", "We use 4 sites.", judge_score=4, auditor_score=3),
             _s1_item("r2", "c", "Two arms, no self-check.", judge_score=2, auditor_score=2)]
    return shown, full


def test_check_set1_items_accepts_prefixes_and_counts_cuts():
    shown, full = _s1_pair()
    assert ck.check_set1_items(shown, full, _S1_BRIEFS) == {"n_shown_shorter_than_full": 1, "n_shown_equal_to_full": 1}


def test_check_set1_items_rejects_a_text_that_is_not_a_prefix():
    shown, full = _s1_pair()
    shown[0]["final_proposal"] = "We use 5 sites."
    with pytest.raises(ValueError, match="not a prefix"):
        ck.check_set1_items(shown, full, _S1_BRIEFS)
    shown, full = _s1_pair()
    shown[1]["final_proposal"] = "Two arms, no self-check. Plus text the full message lacks."
    with pytest.raises(ValueError, match="not a prefix"):
        ck.check_set1_items(shown, full, _S1_BRIEFS)


def test_check_set1_items_rejects_constraint_order_and_run_mismatch():
    shown, full = _s1_pair()
    shown[0]["hard_constraints"] = list(reversed(_S1_BRIEFS["b"]))
    with pytest.raises(ValueError, match="constraint order"):
        ck.check_set1_items(shown, full, _S1_BRIEFS)
    shown, full = _s1_pair()
    with pytest.raises(ValueError, match="different runs"):
        ck.check_set1_items(shown[:1], full, _S1_BRIEFS)


def test_export_set1_items_writes_shown_text_without_scores(tmp_path):
    import json
    shown, full = _s1_pair()
    (tmp_path / "shown.json").write_text(json.dumps(shown))
    (tmp_path / "full.json").write_text(json.dumps(full))
    dst = tmp_path / "out" / "set1.csv"
    df = ck.export_set1_items(tmp_path / "shown.json", dst, full_json=tmp_path / "full.json", briefs=_S1_BRIEFS)
    back = pd.read_csv(dst)
    assert list(back.columns) == ck.ITEM_COLUMNS == list(df.columns)
    assert back.set_index("run_id").final_proposal.to_dict() == {"r1": "We use 4 sites.", "r2": "Two arms, no self-check."}
    assert json.loads(back.hard_constraints.iloc[0]) == _S1_BRIEFS["b"]
    prov = json.loads((tmp_path / "out" / "set1.csv.provenance.json").read_text())
    assert prov["rows_in"] == prov["rows_out"] == 2 and prov["n_shown_shorter_than_full"] == 1
    # a failed check writes nothing
    shown[0]["final_proposal"] = "We use 5 sites."
    (tmp_path / "shown.json").write_text(json.dumps(shown))
    with pytest.raises(ValueError):
        ck.export_set1_items(tmp_path / "shown.json", tmp_path / "bad.csv", full_json=tmp_path / "full.json", briefs=_S1_BRIEFS)
    assert not (tmp_path / "bad.csv").exists()


def test_load_human_labels_reads_set1_from_the_items_csv(tmp_path):
    import json
    hv = tmp_path / "human_validation"
    hv.mkdir()
    scores = {"r1": {"brief_id": "b", "constraint_statuses": ["Satisfied", "Violated"]},
              "r2": {"brief_id": "c", "constraint_statuses": ["Stretched"]}}
    (hv / "human_scores.json").write_text(json.dumps(scores))
    shown, full = _s1_pair()
    # the full-message file must not be read for the as-rated text
    (hv / "scoring_items.json").write_text(json.dumps(full))
    s1 = tmp_path / "set1.csv"
    ck._items_frame(shown[:1]).to_csv(s1, index=False)
    s2 = tmp_path / "set2.csv"
    ck._items_frame(shown[1:]).to_csv(s2, index=False)
    df, info = ck.load_human_labels(tmp_path, _S1_BRIEFS, set2_csv=s2, set1_csv=s1)
    r1 = df[df.run_id == "r1"]
    assert set(r1.as_rated_text) == {"We use 4 sites."} and set(r1.human_set) == {"set1"}
    assert list(r1.human_status) == ["satisfied", "violated"]
    assert set(df[df.run_id == "r2"].human_set) == {"set2"}
    assert info["item_files"] == [str(s1), str(s2)] and info["runs_with_as_rated_text"] == 2
    with pytest.raises(FileNotFoundError, match="set1 items missing"):
        ck.load_human_labels(tmp_path, _S1_BRIEFS, set2_csv=s2, set1_csv=tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="set2 items missing"):
        ck.load_human_labels(tmp_path, _S1_BRIEFS, set2_csv=tmp_path / "missing.csv", set1_csv=s1)
    # the loader checks constraint order against the brief
    bad = ck._items_frame(shown[:1])
    bad["hard_constraints"] = json.dumps(list(reversed(_S1_BRIEFS["b"])))
    bad.to_csv(s1, index=False)
    with pytest.raises(AssertionError, match="constraint order"):
        ck.load_human_labels(tmp_path, _S1_BRIEFS, set2_csv=s2, set1_csv=s1)
