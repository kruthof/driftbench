"""Tests for the constraint typology analysis. Synthetic data only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import typology as T
from drift_bench.analysis.camera_ready import typology_annotate as TA


# --- matcher ---------------------------------------------------------------


def test_match_exact_after_normalisation():
    cands = ["budget under 50000 GBP", "minimum of 8 paired sites"]
    i, how, r = T.match_constraint("  Budget under   50000 gbp ", cands)
    assert (i, how, r) == (0, "exact", 1.0)


def test_match_fuzzy_above_threshold():
    cands = ["must measure soil organic carbon by dry combustion, not loss-on-ignition", "budget under 50000 GBP"]
    i, how, r = T.match_constraint("must measure soil organic carbon by dry combustion, not loss on ignition", cands)
    assert i == 0 and how == "fuzzy" and r > 0.9


def test_match_none_below_threshold():
    cands = ["budget under 50000 GBP", "no irrigation differences permitted"]
    i, how, r = T.match_constraint("sample size of at least 200 participants", cands)
    assert i is None and how == "none" and r <= 0.9


def test_match_all_attaches_indices():
    constraints = pd.DataFrame(
        {"brief_id": ["b1", "b1", "b2"], "constraint_index": [0, 1, 0], "constraint_text": ["alpha beta gamma", "delta epsilon", "zeta eta theta"]}
    )
    checks = pd.DataFrame({"brief_id": ["b1", "b1", "b2"], "check_text": ["delta epsilon", "something else entirely", "Zeta eta theta"]})
    out = T.match_all(checks, constraints)
    assert out["constraint_index"].tolist()[0] == 1
    assert pd.isna(out["constraint_index"].tolist()[1])
    assert out["constraint_index"].tolist()[2] == 0
    assert out["match_how"].tolist() == ["exact", "none", "exact"]


# --- label resolution --------------------------------------------------------


def test_resolve_labels_prefers_author():
    sheet = pd.DataFrame({"label_author": ["prohibition", "", ""], "label_llm": ["sample_floor", "sample_floor", ""]})
    out = T.resolve_labels(sheet)
    assert out["label"].tolist() == ["prohibition", "sample_floor", ""]
    assert out["label_source"].tolist() == ["author", "llm", "none"]


def test_resolve_labels_rejects_unknown_id():
    sheet = pd.DataFrame({"label_author": ["not_a_category"], "label_llm": [""]})
    with pytest.raises(ValueError):
        T.resolve_labels(sheet)


# --- rates and excess --------------------------------------------------------


def test_wilson_ci_matches_known_value():
    lo, hi = T.wilson_ci(10, 100)
    assert abs(lo - 0.0552) < 1e-3 and abs(hi - 0.1744) < 1e-3
    assert T.wilson_ci(0, 0)[0] != T.wilson_ci(0, 0)[0]  # nan


def _synthetic_checks(seed: int = 0, n_per_cell: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rates = {"resource_limit": (0.05, 0.30), "prohibition": (0.01, 0.02)}
    for t in TA.CATEGORY_IDS:
        ss_rate, pr_rate = rates.get(t, (0.03, 0.08))
        for cond, rate in (("single_shot", ss_rate), ("multi_turn_pressure", pr_rate)):
            for i in range(n_per_cell):
                # One run per (condition, i); each run holds one check of every type.
                rows.append({"run_id": f"{cond}_{i}", "brief_id": f"b{i % 10}", "condition": cond, "label": t, "constraint_key": f"b{i % 10}:{t}", "non_satisfied": bool(rng.random() < rate)})
    return pd.DataFrame(rows)


def test_excess_by_type_counts_and_share():
    checks = pd.DataFrame(
        {
            "label": ["a", "a", "a", "b", "b", "b"],
            "condition": ["single_shot", "multi_turn_pressure", "multi_turn_pressure"] * 2,
            "non_satisfied": [False, True, True, False, True, False],
        }
    )
    ex = T.excess_by_type(checks, ["a"])
    a = ex[ex["type"] == "a"].iloc[0]
    b = ex[ex["type"] == "b"].iloc[0]
    assert a["excess"] == 2 and b["excess"] == 1
    assert abs(a["excess_share"] - 2 / 3) < 1e-9
    assert bool(a["fragile_top4"]) and not bool(b["fragile_top4"])
    assert abs(ex["check_share"].sum() - 1.0) < 1e-9
    # Rate-based excess: (1.0 - 0.0) * 2 = 2 for a, (0.5 - 0.0) * 2 = 1 for b.
    assert abs(a["excess_rate_based"] - 2.0) < 1e-9 and abs(b["excess_rate_based"] - 1.0) < 1e-9
    assert abs(a["share_rate_based"] - 2 / 3) < 1e-9
    assert abs(ex["share_rate_based"].sum() - 1.0) < 1e-9


def test_excess_rate_based_differs_from_count_when_denominators_differ():
    # Type a: 2 SS checks (1 bad), 4 pressure checks (2 bad). Count excess 1, rate excess 0.
    checks = pd.DataFrame(
        {
            "label": ["a"] * 6,
            "condition": ["single_shot"] * 2 + ["multi_turn_pressure"] * 4,
            "non_satisfied": [True, False, True, True, False, False],
        }
    )
    ex = T.excess_by_type(checks, ["a"]).iloc[0]
    assert ex["excess"] == 1 and abs(ex["excess_rate_based"]) < 1e-9


def test_per_type_rates_overall_and_by_condition():
    checks = _synthetic_checks()
    out = T.per_type_rates(checks)
    assert set(out["condition"]) == {"all", "single_shot", "multi_turn_pressure"}
    overall = out[out["condition"] == "all"]
    assert len(overall) == 8
    assert (overall["n_checks"] == 120).all()
    assert ((overall["ci_low"] <= overall["rate"]) & (overall["rate"] <= overall["ci_high"])).all()


# --- split enumeration -------------------------------------------------------


def test_balanced_splits_enumerates_35_unordered():
    types = TA.CATEGORY_IDS
    splits = T.balanced_splits(types)
    assert len(splits) == 35
    seen = set()
    for a, b in splits:
        assert len(a) == 4 and len(b) == 4 and not set(a) & set(b)
        assert set(a) | set(b) == set(types)
        key = frozenset([frozenset(a), frozenset(b)])
        assert key not in seen
        seen.add(key)


def test_balanced_splits_requires_eight():
    with pytest.raises(ValueError):
        T.balanced_splits(["a", "b"])


def test_best_balanced_split_finds_planted_group():
    checks = _synthetic_checks(seed=1, n_per_cell=150)
    splits = T.best_balanced_split(checks, TA.CATEGORY_IDS)
    assert len(splits) == 35
    assert (splits["interaction_or"] >= 1).all()
    assert splits["rank"].tolist() == list(range(1, 36))
    assert splits["p_value_cluster_run"].is_monotonic_increasing
    # resource_limit carries the planted interaction, so it sits in group A of the best split.
    assert "resource_limit" in splits.iloc[0]["group_a"].split("|")
    # Both cluster levels are reported, with their cluster counts.
    for level in ("run", "brief"):
        for col in ("interaction_se_log", "ci_low", "ci_high", "p_value", "n_clusters"):
            assert f"{col}_cluster_{level}" in splits
    assert (splits["n_clusters_cluster_run"] == 300).all()
    assert (splits["n_clusters_cluster_brief"] == 10).all()
    assert ((splits["ci_low_cluster_run"] <= splits["interaction_or"]) & (splits["interaction_or"] <= splits["ci_high_cluster_run"])).all()


def test_quant_vs_rest_flag_survives_orientation():
    # Plant the interaction in a non-quant type, so quant-vs-rest has OR < 1 and
    # the enumeration orients the quant set into group B.
    rng = np.random.default_rng(5)
    rows = []
    for t in TA.CATEGORY_IDS:
        rates = (0.05, 0.40) if t == "prohibition" else (0.05, 0.06)
        for cond, rate in (("single_shot", rates[0]), ("multi_turn_pressure", rates[1])):
            for i in range(150):
                rows.append({"run_id": f"{cond}_{i}", "brief_id": f"b{i % 10}", "condition": cond, "label": t, "non_satisfied": bool(rng.random() < rate)})
    checks = pd.DataFrame(rows)
    splits = T.best_balanced_split(checks, TA.CATEGORY_IDS)
    q = splits[splits["is_quant_vs_rest"]]
    assert len(q) == 1
    assert q.iloc[0]["group_b"] == "|".join(sorted(T.QUANT_TYPES))
    direct = T.quant_vs_rest(checks)
    assert direct["interaction_or"] < 1
    assert abs(direct["interaction_or"] * q.iloc[0]["interaction_or"] - 1) < 1e-9
    assert abs(direct["p_value_cluster_run"] - q.iloc[0]["p_value_cluster_run"]) < 1e-9


def test_split_interaction_clustering_changes_se_not_estimate():
    checks = _synthetic_checks(seed=3, n_per_cell=120)
    group_a = ("resource_limit", "sample_floor", "duration_window", "count_scope_limit")
    none = T.split_interaction(checks, group_a, groups_col=None)
    run = T.split_interaction(checks, group_a, groups_col="run_id")
    brief = T.split_interaction(checks, group_a, groups_col="brief_id")
    assert abs(none["interaction_or"] - run["interaction_or"]) < 1e-9
    assert abs(none["interaction_or"] - brief["interaction_or"]) < 1e-9
    ses = {none["interaction_se_log"], run["interaction_se_log"], brief["interaction_se_log"]}
    assert len(ses) == 3, "clustering must change the SE"
    assert (none["groups_col"], run["groups_col"], brief["groups_col"]) == ("none", "run_id", "brief_id")
    assert (none["n_clusters"], run["n_clusters"], brief["n_clusters"]) == (0, 240, 10)
    for r in (none, run, brief):
        assert r["ci_low"] <= r["interaction_or"] <= r["ci_high"]
        assert 0 <= r["p_value"] <= 1


def test_omnibus_interaction_rows_and_roles():
    checks = _synthetic_checks(seed=2, n_per_cell=100)
    out = T.omnibus_interaction(checks)
    assert out["test"].tolist() == ["wald_cluster_run", "likelihood_ratio", "wald_unclustered", "wald_cluster_brief"]
    assert out["role"].tolist() == ["primary", "sensitivity", "sensitivity", "sensitivity"]
    assert out["n_clusters"].tolist() == [200, 0, 0, 10]
    assert (out["df"] == 7).all()
    assert ((out["p_value"] >= 0) & (out["p_value"] <= 1)).all()
    # The three Wald statistics differ because only the covariance changes.
    wald = out[out["test"].str.startswith("wald")]["statistic"]
    assert wald.nunique() == 3


# --- agreement ---------------------------------------------------------------


def test_cohens_kappa_perfect_and_chance():
    a = pd.Series(["x", "y", "x", "y"])
    assert T.cohens_kappa(a, a) == 1.0
    b = pd.Series(["x", "x", "y", "y"])
    assert abs(T.cohens_kappa(a, b)) < 1e-9


def test_agreement_empty_when_author_missing():
    sheet = pd.DataFrame({"label_author": ["", ""], "label_llm": ["prohibition", "sample_floor"]})
    summary, conf = T.agreement(sheet)
    assert summary.iloc[0]["n_rows_both_labelled"] == 0 and conf.empty


def test_agreement_confusion_shape():
    sheet = pd.DataFrame({"label_author": ["prohibition", "sample_floor", "sample_floor"], "label_llm": ["prohibition", "sample_floor", "prohibition"]})
    summary, conf = T.agreement(sheet)
    assert summary.iloc[0]["n_rows_both_labelled"] == 3
    assert abs(summary.iloc[0]["percent_agreement"] - 2 / 3) < 1e-9
    assert conf.shape == (8, 9)  # 8 author rows, 1 index column + 8 llm columns


# --- sheet and annotator parsing (no network) --------------------------------


def test_build_sheet_keeps_existing_labels():
    constraints = pd.DataFrame({"brief_id": ["b1", "b1"], "constraint_index": [0, 1], "constraint_text": ["c0", "c1"], "objective": ["o", "o"]})
    existing = pd.DataFrame({"brief_id": ["b1"], "constraint_index": [1], "constraint_text": ["c1"], "label_author": ["prohibition"], "label_llm": ["data_source"], "llm_rationale": ["r"]})
    sheet = TA.build_sheet(constraints, existing)
    assert sheet.columns.tolist() == TA.SHEET_COLUMNS
    assert sheet["label_author"].tolist() == ["", "prohibition"]
    assert sheet["label_llm"].tolist() == ["", "data_source"]


def test_parse_response_handles_prose_and_bad_ids():
    cat, rat = TA.parse_response('Sure: {"category": "sample_floor", "rationale": "n floor"}')
    assert (cat, rat) == ("sample_floor", "n floor")
    cat, rat = TA.parse_response('{"category": "made_up", "rationale": "x"}')
    assert cat == "" and "invalid category" in rat
    cat, rat = TA.parse_response("no json here")
    assert cat == "" and rat.startswith("unparseable")


def test_cache_read_tokens_reads_both_spellings():
    class Details:
        cached_tokens = 12

    class UsageA:
        prompt_tokens_details = Details()

    class UsageB:
        prompt_tokens_details = None
        cache_read_input_tokens = 7

    class UsageC:
        pass

    assert TA.cache_read_tokens(UsageA()) == 12
    assert TA.cache_read_tokens(UsageB()) == 7
    assert TA.cache_read_tokens(UsageC()) == 0


def test_archive_if_stale_renames_only_on_sha_mismatch(tmp_path):
    import json

    path = tmp_path / "responses.jsonl"
    rec = {"brief_id": "b1", "constraint_index": 0, "label_llm": "prohibition", "codebook_sha256": "a" * 64}
    path.write_text(json.dumps(rec) + "\n")
    # Same sha: nothing happens.
    assert TA.archive_if_stale(path, "a" * 64) is None and path.exists()
    # Different sha: archived as v1 with the records' own sha prefix.
    out = TA.archive_if_stale(path, "b" * 64)
    assert out == tmp_path / "responses_v1_codebook_aaaaaaaa.jsonl" and out.exists() and not path.exists()
    # Next stale file takes the next pass number; records without a sha give 'unknown'.
    path.write_text(json.dumps({"brief_id": "b1", "constraint_index": 0, "label_llm": "prohibition"}) + "\n")
    out2 = TA.archive_if_stale(path, "b" * 64)
    assert out2 == tmp_path / "responses_v2_codebook_unknown.jsonl" and out2.exists()
    # Missing or empty file: nothing happens.
    assert TA.archive_if_stale(path, "b" * 64) is None
    path.write_text("")
    assert TA.archive_if_stale(path, "b" * 64) is None


def test_archived_passes_counts_changed_labels(tmp_path):
    sheet = pd.DataFrame({"brief_id": ["b1", "b1"], "constraint_index": [0, 1], "label_llm": ["prohibition", "data_source"]})
    recs = [
        {"brief_id": "b1", "constraint_index": 0, "label_llm": "prohibition", "cost_usd": 0.01, "timestamp": "t1"},
        {"brief_id": "b1", "constraint_index": 1, "label_llm": "design_requirement", "cost_usd": 0.02, "timestamp": "t2"},
    ]
    (tmp_path / "responses_v1_codebook_abcdef01.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in recs) + "\n")
    hist = TA.archived_passes(tmp_path, sheet)
    assert len(hist) == 1
    h = hist[0]
    assert h["codebook_sha256_prefix"] == "abcdef01" and h["n_calls"] == 2
    assert abs(h["cost_usd"] - 0.03) < 1e-9
    assert h["n_labels_changed_vs_current"] == 1 and h["changed"] == [("b1", 1, "design_requirement", "data_source")]


# Integration check on the real briefs. It reads files on disk, so it is skipped when
# the briefs directory is absent (synthetic-only environments).
@pytest.mark.skipif(not TA.BRIEFS_DIR.is_dir(), reason="real briefs directory not present")
def test_real_briefs_yield_234_constraints():
    assert len(TA.load_constraints()) == 234


def test_run_fills_llm_labels_from_raw_responses(tmp_path, monkeypatch):
    """A fresh output dir gets label_llm from responses.jsonl, not from an old sheet."""
    import json
    raw_dir = tmp_path / "camera_ready" / "e8_typology_llm"
    raw_dir.mkdir(parents=True)
    sha = TA.codebook_sha256(TA.CODEBOOK_PATH.read_text())
    constraints = TA.load_constraints()
    first = constraints.iloc[0]
    recs = [
        {"brief_id": first.brief_id, "constraint_index": int(first.constraint_index),
         "codebook_sha256": sha, "label_llm": "resource_limit", "llm_rationale": "r1"},
        {"brief_id": first.brief_id, "constraint_index": int(first.constraint_index),
         "codebook_sha256": "0" * 64, "label_llm": "stale_label", "llm_rationale": "old codebook"},
    ]
    (raw_dir / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    out = tmp_path / "out"
    sheet = TA.run(tmp_path, out)["typology_sheet"]
    row = sheet[(sheet.brief_id == first.brief_id) & (sheet.constraint_index == int(first.constraint_index))].iloc[0]
    assert row.label_llm == "resource_limit" and row.llm_rationale == "r1"
    # Rows the raw file does not cover stay empty; nothing is invented.
    assert (sheet.label_llm != "").sum() == 1
    # Running again keeps label_author from the existing sheet.
    sheet.loc[sheet.index[0], "label_author"] = "author_x"
    TA.write_sheet(sheet, out / "typology_sheet.csv")
    again = TA.run(tmp_path, out)["typology_sheet"]
    assert again.iloc[0].label_author == "author_x"
    assert again.iloc[0].label_llm == "resource_limit"


# --- provenance sidecar naming (plan P12) ---------------------------------


def test_provenance_sidecar_is_named_after_the_csv_with_extension(tmp_path):
    """Both typology writers put the sidecar at ``<name>.csv.provenance.json``."""
    import json
    csv_path = tmp_path / "typology_by_type.csv"
    csv_path.write_text("a\n1\n")
    T._provenance(csv_path, ["in.csv"], ["no filter"], 1, 1)
    assert (tmp_path / "typology_by_type.csv.provenance.json").exists()
    assert not (tmp_path / "typology_by_type.provenance.json").exists()
    assert json.loads((tmp_path / "typology_by_type.csv.provenance.json").read_text())["csv"] == "typology_by_type.csv"

    sheet_path = tmp_path / "typology_sheet.csv"
    sheet_path.write_text("a\n1\n")
    TA.write_provenance(sheet_path, ["briefs/*.yaml"], ["one row per constraint"], 1, 1)
    assert (tmp_path / "typology_sheet.csv.provenance.json").exists()
    assert not (tmp_path / "typology_sheet.provenance.json").exists()
