"""Tests for drift_bench.analysis.camera_ready.kbv_crossed.

Pure functions on synthetic frames. No real data, no network.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import kbv_crossed as kc

SONNET = kc.SONNET
GPT = "openai/gpt-5.4"
LLAMA = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
PRO = kc.GEMINI_PRO
SS, MTP, MTN = kc.SS, kc.MTP, kc.MTN


def _published(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, brief_id, model_id, condition, fid, adh, alt, cmp, auditor_class)."""
    df = pd.DataFrame(
        rows,
        columns=["run_id", "brief_id", "model_id", "condition", "j_objective_fidelity", "j_constraint_adherence",
                 "j_alternative_coverage", "j_complexity_inflation", "a_drift_classification"],
    )
    df["judge_model"] = df["model_id"].map(kc.published_judge_of)
    return df


def _crossed(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, model_id, arm, fid, adh, alt, cmp). Judge follows the arm."""
    recs = []
    for run_id, model_id, arm, fid, adh, alt, cmp in rows:
        judge = kc.OPUS if arm == "opus" else kc.GPT
        recs.append(
            {
                "run_id": run_id, "brief_id": "b1", "model_id": model_id, "condition": MTP,
                "judge_model": judge, "judge_response_model": judge, "timestamp": 1.0,
                "objective_fidelity": fid, "constraint_adherence": adh, "alternative_coverage": alt,
                "complexity_inflation": cmp, "source_file": f"judge_{arm}_{run_id}.jsonl",
                "crossed_arm": arm, "source_dir": arm,
            }
        )
    return pd.DataFrame(recs)


def _probes(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["run_id", "overall_accuracy"])


@pytest.fixture
def small():
    pub = _published(
        [
            ("s1", "b1", SONNET, MTP, 4, 2, 4, 2, "trajectory_drift"),   # published GPT
            ("s2", "b2", SONNET, SS, 4, 3, 4, 1, "mild_drift"),
            ("s3", "b1", SONNET, MTN, 4, 3, 4, 1, "mild_drift"),         # neutral: no crossed arm
            ("g1", "b1", GPT, MTP, 4, 4, 4, 1, "no_drift"),              # published Opus
            ("g2", "b2", GPT, SS, 4, 4, 4, 0, "no_drift"),
            ("l1", "b1", LLAMA, MTP, 3, 3, 3, 2, "trajectory_lock_in"),
            ("l2", "b2", LLAMA, SS, 4, 4, 4, 0, "mild_drift"),
            ("l3", "b3", LLAMA, MTP, 3, 3, 3, 2, "mild_drift"),         # no crossed file
        ]
    )
    crossed = _crossed(
        [
            ("s1", SONNET, "opus", 4, 3, 4, 1),
            ("s2", SONNET, "opus", 4, 4, 4, 0),
            ("g1", GPT, "gpt", 4, 3, 3, 2),
            ("g2", GPT, "gpt", 4, 3, 3, 1),
            ("l1", LLAMA, "gpt", 3, 2, 2, 3),
            ("l2", LLAMA, "gpt", 4, 4, 3, 1),
            ("zz", LLAMA, "gpt", 4, 4, 4, 0),                            # no parquet row
        ]
    )
    probes = _probes([("s1", 1.0), ("s2", 1.0), ("g1", 1.0), ("l1", 0.9), ("l3", 1.0), ("s3", 1.0)])
    return pub, crossed, probes


# ---------------------------------------------------------------- arm assembly


def test_judge_routing_rules():
    assert kc.published_judge_of(SONNET) == kc.GPT
    assert kc.published_judge_of(GPT) == kc.OPUS
    assert kc.crossed_judge_of(SONNET) == kc.OPUS
    assert kc.crossed_judge_of(LLAMA) == kc.GPT


def test_assemble_arms_which_file_feeds_which_arm(small):
    pub, crossed, probes = small
    runs, log = kc.assemble_runs(pub, crossed, probes)
    r = runs.set_index("run_id")
    # Sonnet: opus arm = crossed file, gpt arm = published.
    assert r.loc["s1", "opus_constraint_adherence"] == 3
    assert r.loc["s1", "gpt_constraint_adherence"] == 2
    # Non-Sonnet: opus arm = published, gpt arm = crossed file.
    assert r.loc["g1", "opus_constraint_adherence"] == 4
    assert r.loc["g1", "gpt_constraint_adherence"] == 3
    assert r.loc["l1", "gpt_alternative_coverage"] == 2
    # Unjoined crossed file is counted and listed, never added.
    assert "zz" not in r.index
    assert log["crossed_files_without_clean_parquet_row"] == 1
    assert log["crossed_files_without_clean_parquet_row_ids"] == ["zz"]
    assert log["crossed_judge_mismatch_rows"] == 0


def test_missing_cells_are_not_fabricated(small):
    pub, crossed, probes = small
    runs, log = kc.assemble_runs(pub, crossed, probes)
    r = runs.set_index("run_id")
    # Neutral cell: only the published arm exists.
    assert not r.loc["s3", "has_crossed"]
    assert math.isnan(r.loc["s3", "opus_constraint_adherence"])
    assert r.loc["s3", "gpt_constraint_adherence"] == 3  # Sonnet published is GPT; still only via published
    # A pressure run without a crossed file is excluded from both common arms.
    assert math.isnan(r.loc["l3", "gpt_constraint_adherence"])
    assert log["rows_in_crossed_cells_without_crossed_file"] == 1
    opus = kc.arm_frame(runs, "opus")
    gpt = kc.arm_frame(runs, "gpt")
    assert set(opus["run_id"]) == {"s1", "s2", "g1", "g2", "l1", "l2"}
    assert set(gpt["run_id"]) == set(opus["run_id"])
    assert (opus["judge_model"] == kc.OPUS).all()
    assert (gpt["judge_model"] == kc.GPT).all()
    published = kc.arm_frame(runs, "published")
    assert len(published) == len(pub)
    assert (published["judge_arm"] == "published").all()
    runs_df = kc.runs_table(runs)
    assert len(runs_df) == 6 and "s3" not in set(runs_df["run_id"])


def test_routing_check_flags_wrong_directory():
    crossed = _crossed([("a", SONNET, "gpt", 4, 4, 4, 0), ("b", GPT, "opus", 4, 4, 4, 0)])
    chk = kc.check_crossed_routing(crossed)
    assert chk["gpt_dir_sonnet"] == 1 and chk["opus_dir_non_sonnet"] == 1


# ---------------------------------------------------------------- KBV


def test_kbv_counts_per_arm(small):
    pub, crossed, probes = small
    runs, _ = kc.assemble_runs(pub, crossed, probes)
    opus = kc.kbv_by_model(kc.kbv_flags(kc.arm_frame(runs, "opus")), "opus", "clean")
    o = opus.set_index("model_id")
    # Sonnet s1 under Opus: adherence 3, knows -> KBV 1 of 1.
    assert o.loc[SONNET, "kbv_count"] == 1 and o.loc[SONNET, "n_with_probe"] == 1
    # GPT g1 under Opus (published): adherence 4 -> 0 of 1.
    assert o.loc[GPT, "kbv_count"] == 0
    gpt = kc.kbv_by_model(kc.kbv_flags(kc.arm_frame(runs, "gpt")), "gpt", "clean").set_index("model_id")
    # GPT g1 under GPT (crossed): adherence 3 -> 1 of 1.
    assert gpt.loc[GPT, "kbv_count"] == 1
    # Llama l1: probe 0.9 knows, adherence 2 -> KBV; l3 excluded (no crossed file).
    assert gpt.loc[LLAMA, "kbv_count"] == 1 and gpt.loc[LLAMA, "n"] == 1
    lo, hi = gpt.loc[GPT, ["ci_low", "ci_high"]]
    assert 0 <= lo <= 1.0 <= hi <= 1


def test_kbv_flags_runs_without_probe_leave_denominator():
    df = pd.DataFrame(
        {
            "run_id": ["a", "b"], "model_id": [GPT, GPT], "condition": [MTP, MTP], "judge_model": [kc.OPUS] * 2,
            "j_constraint_adherence": [2, 2], "final_probe_accuracy": [1.0, np.nan], "has_final_probe": [True, False],
        }
    )
    s = kc.kbv_by_model(kc.kbv_flags(df), "opus", "clean").iloc[0]
    assert s["n"] == 2 and s["n_with_probe"] == 1 and s["kbv_count"] == 1 and s["violates"] == 2


def test_waiver_imputes_knows_only_for_flagged_rows():
    df = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"], "model_id": [PRO] * 3, "condition": [MTP] * 3, "judge_model": [kc.OPUS] * 3,
            "j_constraint_adherence": [3, 3, 3], "final_probe_accuracy": [0.5, 0.5, 1.0], "has_final_probe": [True] * 3,
        }
    )
    plain = kc.kbv_by_model(kc.kbv_flags(df), "opus", "clean").iloc[0]
    assert plain["kbv_count"] == 1 and plain["n_truncated_waived"] == 0 and plain["knows_flipped_by_waiver"] == 0
    # a: below threshold and waived (flipped); b: below, not waived; c: at ceiling and waived (not flipped).
    waived = kc.kbv_by_model(kc.kbv_flags(df, waive_truncated=pd.Series([True, False, True], index=df.index)), "opus", "w").iloc[0]
    assert waived["kbv_count"] == 2
    assert waived["n_truncated_waived"] == 2 and waived["knows_flipped_by_waiver"] == 1


def test_membership_set_and_range():
    summary = pd.DataFrame(
        {
            "judge_arm": ["opus"] * 3 + ["gpt"] * 3,
            "variant": "clean",
            "model_id": ["m1", "m2", "m3"] * 2,
            "kbv_rate": [0.1, 0.6, 0.9, 0.5, 0.7, 0.95],
        }
    )
    r = kc.kbv_range(summary).set_index("judge_arm")
    assert r.loc["opus", "n_above_cut"] == 2 and r.loc["opus", "models_above_cut"] == "m3;m2"
    assert r.loc["gpt", "n_above_cut"] == 2  # exactly 0.5 is not above
    assert r.loc["opus", "min_kbv_rate"] == 0.1 and r.loc["opus", "argmin_model"] == "m1"
    assert r.loc["opus", "argmax_model"] == "m3" and r.loc["gpt", "max_kbv_rate"] == 0.95


# ---------------------------------------------------------------- offset


def test_offset_sign_and_bootstrap_ci_shape(small):
    pub, crossed, probes = small
    runs, _ = kc.assemble_runs(pub, crossed, probes)
    paired = kc.paired_frame(runs)
    assert len(paired) == 6
    tab = kc.offset_table(paired)
    row = tab[(tab["block"] == "offset_overall") & (tab["dimension"] == "constraint_adherence")].iloc[0]
    # Opus minus GPT per run: s1 +1, s2 +1, g1 +1, g2 +1, l1 +1, l2 0 -> mean 5/6.
    assert row["estimate"] == pytest.approx(5 / 6)
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
    assert row["n_runs"] == 6 and row["n_briefs"] == 2
    excess = tab[(tab["block"] == "anthropic_excess") & (tab["dimension"] == "constraint_adherence")].iloc[0]
    # Sonnet offset 1.0; GPT 1.0; Llama 0.5 -> excess 1 - 0.75 = 0.25.
    assert excess["estimate"] == pytest.approx(0.25)
    assert excess["ci_low"] <= excess["estimate"] <= excess["ci_high"]


def test_cluster_bootstrap_is_deterministic_and_brackets_mean():
    rng = np.random.default_rng(1)
    vals = pd.Series(rng.normal(0.5, 1.0, 200))
    clusters = pd.Series(np.repeat(np.arange(20), 10))
    a = kc.cluster_bootstrap_mean(vals, clusters, draws=500, seed=3)
    b = kc.cluster_bootstrap_mean(vals, clusters, draws=500, seed=3)
    assert a == b
    assert a[0] < vals.mean() < a[1]
    empty = kc.cluster_bootstrap_mean(pd.Series([], dtype=float), pd.Series([], dtype=object))
    assert len(empty) == 2 and all(math.isnan(x) for x in empty)


def test_interaction_test_returns_probability():
    rng = np.random.default_rng(7)
    n = 120
    base = pd.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(n)],
            "brief_id": [f"b{i % 10}" for i in range(n)],
            "model_id": [GPT if i % 2 else LLAMA for i in range(n)],
            "condition": [SS if i % 4 < 2 else MTP for i in range(n)],
        }
    )
    for d in kc.DIMS:
        base[f"opus_{d}"] = rng.integers(2, 5, n)
        base[f"gpt_{d}"] = base[f"opus_{d}"] - rng.integers(0, 2, n)
    long = kc.long_scores(base)
    assert len(long) == 2 * n and set(long["judge"]) == {"gpt", "opus"}
    r = kc.interaction_test(long, "constraint_adherence")
    assert 0 <= r["p"] <= 1 and r["n_clusters"] == 10 and r["n_obs"] == 2 * n


def test_delta_rows_and_max_abs_difference():
    base = pd.DataFrame(
        {
            "run_id": list("abcd"),
            "brief_id": ["b1", "b2", "b1", "b2"],
            "model_id": [GPT] * 4,
            "condition": [SS, SS, MTP, MTP],
        }
    )
    for d in kc.DIMS:
        base[f"opus_{d}"] = [4, 4, 3, 3]   # delta -1
        base[f"gpt_{d}"] = [4, 4, 4, 2]    # delta -1
    base["opus_constraint_adherence"] = [4, 4, 3, 1]  # delta -2 under Opus
    rows = pd.DataFrame(kc.delta_rows(base))
    mx = rows[(rows["block"] == "delta_max_abs_judge_difference") & (rows["dimension"] == "constraint_adherence")].iloc[0]
    assert mx["estimate"] == pytest.approx(1.0) and mx["model_id"] == GPT
    fid = rows[(rows["block"] == "delta_max_abs_judge_difference") & (rows["dimension"] == "objective_fidelity")].iloc[0]
    assert fid["estimate"] == pytest.approx(0.0)


def test_rubric_cell_share():
    base = pd.DataFrame(
        {
            "run_id": list("abcd"), "brief_id": ["b1"] * 4, "model_id": [GPT] * 4,
            "condition": [MTP, MTP, MTP, SS],
            "opus_constraint_adherence": [4, 3, 2, 4],
            "gpt_constraint_adherence": [3, 2, 1, 4],
        }
    )
    rows = {r["judge"]: r["estimate"] for r in kc.rubric_cell_rows(base)}
    assert rows["opus"] == pytest.approx(2 / 3)
    assert rows["gpt"] == pytest.approx(1 / 3)
    assert rows["both_judges_pooled"] == pytest.approx(0.5)


# ---------------------------------------------------------------- aggregate, auditor, stretch, control


def test_aggregate_table_flags_published_rows_under_common_arms(small):
    pub, crossed, probes = small
    runs, _ = kc.assemble_runs(pub, crossed, probes)
    agg = kc.aggregate_table(runs)
    opus_all = agg[(agg["table_arm"] == "opus") & (agg["model_id"] == "all")].set_index("condition")
    assert opus_all.loc[MTN, "judge_arm"] == "published"
    assert opus_all.loc[MTP, "judge_arm"] == "opus"
    assert opus_all.loc[MTP, "n"] == 3  # s1, g1, l1 (l3 lacks a crossed file)
    # Opus-arm pressure adherence: s1 3 (crossed), g1 4, l1 3 -> 10/3.
    assert opus_all.loc[MTP, "constraint_adherence_mean"] == pytest.approx(10 / 3)
    delta = opus_all.loc["mtp-ss"]
    assert delta["row_type"] == "delta_pressure_minus_single_shot"
    pub_all = agg[(agg["table_arm"] == "published") & (agg["model_id"] == "all")].set_index("condition")
    assert pub_all.loc[MTP, "n"] == 4


def test_auditor_agreement_counts(small):
    pub, crossed, probes = small
    runs, _ = kc.assemble_runs(pub, crossed, probes)
    flags = kc.kbv_flags(kc.arm_frame(runs, "published"))
    a = kc.auditor_agreement(flags, "published").set_index("definition")
    k = a.loc["kbv_vs_auditor"]
    # Pressure runs with a probe: s1 (kbv 1, aud 1), g1 (0, 0), l1 (1, 1), l3 (1, 0).
    assert k["n"] == 4 and k["percent_agreement"] == pytest.approx(0.75)
    assert k["both_positive"] == 2 and k["judge_only"] == 1 and k["auditor_only"] == 0
    assert -1 <= k["spearman_rho"] <= 1


def test_stretch_violate_shares(small):
    pub, crossed, probes = small
    runs, _ = kc.assemble_runs(pub, crossed, probes)
    flags = kc.kbv_flags(kc.arm_frame(runs, "gpt"))
    sv = kc.stretch_violate(flags, "gpt").set_index("model_id")
    # GPT arm pressure: s1 adh 2 (published GPT), g1 3, l1 2 -> below-3 share 2/3; SS: s2 3, g2 3, l2 4 -> 0.
    assert sv.loc["all", "share_below3_mtp"] == pytest.approx(2 / 3)
    assert sv.loc["all", "share_below3_ss"] == 0
    # KBV-positive pressure runs: s1 (2), g1 (3), l1 (2) -> at-3 share 1/3.
    assert sv.loc["all", "n_kbv_positive_mtp"] == 3 and sv.loc["all", "share_kbv_at3"] == pytest.approx(1 / 3)


def test_drift_control_stats():
    pub = _published([("c1", "b1", GPT, SS, 4, 4, 4, 0, "no_drift"), ("c2", "b1", GPT, MTP, 3, 2, 3, 2, "mild_drift")])
    controls = _crossed([("c1", GPT, "opus", 4, 3, 4, 0), ("c2", GPT, "opus", 3, 2, 3, 3), ("c9", GPT, "opus", 4, 4, 4, 4)])
    controls = controls.rename(columns={"crossed_arm": "control_judge"})
    out, log = kc.drift_control(controls, pub)
    assert log["control_files"] == 3 and log["control_files_with_clean_parquet_row"] == 2
    o = out[out["control_judge"] == "opus"].set_index("dimension")
    assert o.loc["constraint_adherence", "mad"] == pytest.approx(0.5)
    assert o.loc["constraint_adherence", "exact_agreement"] == pytest.approx(0.5)
    assert o.loc["objective_fidelity", "exact_agreement"] == 1.0
    assert o.loc["complexity_inflation", "mean_diff_today_minus_april"] == pytest.approx(0.5)


# ---------------------------------------------------------------- loaders and end to end


def _write_judge(dir_, prefix, run_id, model_id, condition, judge, adh, brief="b1"):
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_id": run_id, "brief_id": brief, "model_id": model_id, "condition": condition, "judge_model": judge,
        "objective_fidelity": 4, "constraint_adherence": adh, "alternative_coverage": 4, "complexity_inflation": 1,
        "timestamp": 1793031106.0, "judge_response_model": judge,
    }
    with open(dir_ / f"{prefix}_{run_id}.jsonl", "w") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_load_judge_dir_and_missing_dir(tmp_path):
    d = tmp_path / "opus"
    _write_judge(d, "judge_opus", "r1", SONNET, MTP, kc.OPUS, 3)
    df = kc.load_judge_dir(d, "judge_opus")
    assert len(df) == 1 and df.loc[0, "constraint_adherence"] == 3 and df.loc[0, "source_file"] == "judge_opus_r1.jsonl"
    empty = kc.load_judge_dir(tmp_path / "absent", "judge_gpt")
    assert empty.empty and "run_id" in empty.columns


def test_run_end_to_end_on_synthetic_tree(tmp_path):
    root = tmp_path / "data"
    (root / "aggregated").mkdir(parents=True)
    (root / "analysis").mkdir()
    (root / "scores").mkdir()
    (root / "openweight_subjects" / "scores").mkdir(parents=True)
    (root / "transcripts").mkdir()
    (root / "openweight_subjects" / "transcripts").mkdir()
    # Eight briefs and a small seeded perturbation, so the brief-clustered
    # covariance in the interaction model is not degenerate.
    rng = np.random.default_rng(20261026)
    rows = []
    briefs = [f"b{i}" for i in range(1, 9)]
    k = 0
    for model in (SONNET, GPT, LLAMA):
        for cond in (SS, MTP, MTN):
            for b in briefs:
                k += 1
                adh = 4 if cond != MTP else (2 if model != GPT else 4)
                adh = int(np.clip(adh - rng.integers(0, 2), 0, 4))
                fid = int(np.clip(4 - rng.integers(0, 2), 0, 4))
                alt = int(np.clip(4 - rng.integers(0, 2), 0, 4))
                cmp_ = int(rng.integers(0, 3))
                rows.append((f"r{k}", b, model, cond, fid, adh, alt, cmp_, "trajectory_drift" if adh < 4 else "no_drift"))
    pub = _published(rows)
    pub.to_parquet(root / "aggregated" / "all_scores.parquet")
    # Sonnet pressure stays non-compliant under both arms; the GPT model is
    # compliant under its Opus (published) score and non-compliant under GPT.
    pub_adh = {r["run_id"]: int(r["j_constraint_adherence"]) for _, r in pub.iterrows()}
    for _, r in pub.iterrows():
        # Published judge files with a non-stray timestamp (the stray rule reads them).
        _write_judge(root / "scores", "judge", r["run_id"], r["model_id"], r["condition"], r["judge_model"], pub_adh[r["run_id"]], r["brief_id"])
        if r["condition"] in (SS, MTP):
            jitter = int(rng.integers(0, 2))
            if r["model_id"] == SONNET:
                crossed_adh = min(3, pub_adh[r["run_id"]] + jitter) if r["condition"] == MTP else min(4, pub_adh[r["run_id"]] + jitter)
                _write_judge(root / kc.E1_DIR / "opus", "judge_opus", r["run_id"], r["model_id"], r["condition"], kc.OPUS, crossed_adh, r["brief_id"])
            else:
                crossed_adh = max(0, min(3, pub_adh[r["run_id"]] - jitter)) if r["condition"] == MTP else max(0, pub_adh[r["run_id"]] - jitter)
                _write_judge(root / kc.E1_DIR / "gpt", "judge_gpt", r["run_id"], r["model_id"], r["condition"], kc.GPT, crossed_adh, r["brief_id"])
    gpt_pub_pressure = [pub_adh[r["run_id"]] for _, r in pub.iterrows() if r["model_id"] == GPT and r["condition"] == MTP]
    expected_gpt_opus_rate = float(np.mean([a < 4 for a in gpt_pub_pressure]))
    # Controls: the original judge again, identical scores to April.
    first_sonnet = pub[(pub["model_id"] == SONNET) & (pub["condition"] == SS)].iloc[0]
    first_gpt = pub[(pub["model_id"] == GPT) & (pub["condition"] == SS)].iloc[0]
    _write_judge(root / kc.E1_DIR / "control_gpt", "judge_gpt", first_sonnet["run_id"], SONNET, SS, kc.GPT, pub_adh[first_sonnet["run_id"]])
    _write_judge(root / kc.E1_DIR / "control_opus", "judge_opus", first_gpt["run_id"], GPT, SS, kc.OPUS, pub_adh[first_gpt["run_id"]])
    pd.DataFrame({"run_id": pub["run_id"], "brief_id": pub["brief_id"], "model_id": pub["model_id"], "condition": pub["condition"],
                  "turn_index": 5, "overall_accuracy": 1.0, "constraint_recall": 1.0, "banned_recall": 1.0}).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)
    out = tmp_path / "out"
    tables = kc.run(root, out)
    assert len(tables["crossed_scores_runs"]) == 3 * 2 * len(briefs)
    for name in tables:
        assert (out / f"{name}.csv").exists() and (out / f"{name}.csv.provenance.json").exists()
    prov = json.loads((out / "kbv_common_judge.csv.provenance.json").read_text())
    assert prov["filters"]["unjoined_crossed_files"]["n_dropped"] == 0
    assert set(prov["columns"]) == set(tables["kbv_common_judge"].columns)
    kbv = tables["kbv_common_judge"]
    kbv = kbv[kbv["variant"] == "clean"].set_index(["judge_arm", "model_id"])
    # Sonnet pressure is non-compliant under both arms (published <= 2, Opus crossed <= 3).
    assert kbv.loc[("opus", SONNET), "kbv_rate"] == 1.0 and kbv.loc[("gpt", SONNET), "kbv_rate"] == 1.0
    # GPT model: Opus arm = published (3 or 4); GPT arm = crossed (<= 3) -> 1.0.
    assert kbv.loc[("opus", GPT), "kbv_rate"] == pytest.approx(expected_gpt_opus_rate)
    assert kbv.loc[("gpt", GPT), "kbv_rate"] == 1.0
    # Adherence scores only differ from the fidelity constant, so the fidelity
    # control columns are exact; adherence was written from April's value too.
    ctrl = tables["judge_drift_control"]
    assert (ctrl.loc[ctrl["dimension"] == "constraint_adherence", "exact_agreement"] == 1.0).all()
    rng_ = tables["kbv_common_judge_range"].set_index(["judge_arm", "variant"])
    assert rng_.loc[("gpt", "clean"), "n_above_cut"] == 3
    assert rng_.loc[("opus", "clean"), "n_above_cut"] == 2 + int(expected_gpt_opus_rate > 0.5)
    offs = tables["judge_offset"]
    inter = offs[offs["block"] == "interaction_judge_x_condition"]
    assert inter["coef"].notna().all() and inter["se"].notna().all()
    assert offs.loc[offs["block"] == "offset_overall", "coef"].isna().all()
