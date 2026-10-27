"""Tests for drift_bench.analysis.camera_ready.graded_severity (synthetic inputs, no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import graded_severity as gs
from drift_bench.analysis.camera_ready.sample import SONNET as SAMPLE_SONNET


def _write_structured(d: Path, prefix: str, run_id: str, statuses: list[str], model="m", cond="single_shot", brief="b1", judge="j"):
    rec = {
        "run_id": run_id, "brief_id": brief, "model_id": model, "condition": cond, "judge_model": judge,
        "constraint_adherence": 4 if all(s.lower() == "satisfied" for s in statuses) else 3,
        "constraint_checks": [{"constraint": f"c{i}", "status": s, "reason": ""} for i, s in enumerate(statuses)],
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}_{run_id}.jsonl").write_text(json.dumps(rec) + "\n")


def test_normalise_status_is_case_insensitive_and_strict():
    assert gs.normalise_status("SATISFIED") == "satisfied"
    assert gs.normalise_status(" Stretched ") == "stretched"
    with pytest.raises(ValueError):
        gs.normalise_status("maybe")


def test_loader_handles_absent_directory(tmp_path):
    df = gs.load_structured_runs(tmp_path / "missing", "structured_opus")
    assert df.empty
    assert "g_kbv" in df.columns


def test_loader_respects_prefix(tmp_path):
    _write_structured(tmp_path, "structured", "r1", ["satisfied", "violated"])
    _write_structured(tmp_path, "structured_opus", "r1", ["satisfied", "satisfied"])
    pub = gs.load_structured_runs(tmp_path, "structured")
    opus = gs.load_structured_runs(tmp_path, "structured_opus")
    # The plain prefix glob must not pick up the structured_opus file.
    assert len(pub) == 1 and len(opus) == 1
    assert pub["g_kbv"].iloc[0] == 0.5
    assert opus["g_kbv"].iloc[0] == 0.0


def test_g_kbv_counts_stretched_and_violated(tmp_path):
    _write_structured(tmp_path, "structured", "r1", ["satisfied", "STRETCHED", "violated", "satisfied"])
    runs = gs.load_structured_runs(tmp_path, "structured")
    r = runs.iloc[0]
    assert r["n_checks"] == 4 and r["n_stretched"] == 1 and r["n_violated"] == 1
    assert r["n_not_satisfied"] == 2 and r["g_kbv"] == 0.5


def test_file_without_checks_is_recorded_not_counted(tmp_path):
    _write_structured(tmp_path, "structured", "r1", [])
    _write_structured(tmp_path, "structured", "r2", ["satisfied"])
    runs = gs.load_structured_runs(tmp_path, "structured")
    assert list(runs["run_id"]) == ["r2"]
    assert runs.attrs["files_without_checks"] == ["structured_r1.jsonl"]


def test_recall_flags():
    runs = pd.DataFrame({"run_id": ["a", "b", "c"], "g_kbv": [0.0, 0.2, 0.5]})
    probes = pd.DataFrame({"run_id": ["a", "b"], "probe_turn_index": [5, 5], "overall_accuracy": [1.0, 0.85]})
    out = gs.attach_recall(runs, probes, {"a": True})
    assert out["perfect_recall"].tolist() == [True, False, False]
    assert out["knows"].tolist() == [True, True, False]
    assert out["has_probe"].tolist() == [True, True, False]
    assert bool(out.loc[0, "final_probe_truncated"]) is True
    assert pd.isna(out.loc[2, "final_probe_truncated"])


def test_probe_truncation_rule(tmp_path):
    def write(run_id, tokens):
        rec = {"metadata": {"run_id": run_id},
               "messages": [{"role": "assistant", "content": "x", "is_probe": False, "token_count": 50},
                            {"role": "assistant", "content": "p", "is_probe": True, "token_count": tokens}]}
        (tmp_path / f"{run_id}.jsonl").write_text(json.dumps(rec) + "\n")
    write("t", 1020)
    write("u", 1019)
    (tmp_path / "noprobe.jsonl").write_text(json.dumps({"metadata": {"run_id": "n"}, "messages": []}) + "\n")
    # Run v: an earlier probe is truncated, the final one is not.
    rec = {"metadata": {"run_id": "v"},
           "messages": [{"role": "assistant", "content": "p", "is_probe": True, "token_count": 1020},
                        {"role": "assistant", "content": "p", "is_probe": True, "token_count": 10}]}
    (tmp_path / "v.jsonl").write_text(json.dumps(rec) + "\n")
    st = {}
    out = gs.final_probe_truncation([tmp_path, tmp_path / "absent"], stats_out=st)
    assert out == {"t": True, "u": False, "v": False}
    assert st["transcript_files_scanned"][str(tmp_path)] == 4 and st["runs_with_probe"] == 3
    assert gs.final_probe_truncation([tmp_path], any_probe=True)["v"] is True


def test_holm_correction_matches_hand_computation():
    adj = gs.holm_correction([0.01, 0.04, 0.03, np.nan])
    # Sorted: 0.01 (x3), 0.03 (x2), 0.04 (x1) -> 0.03, 0.06, 0.06 (monotone), NaN stays.
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])
    assert gs.holm_correction([0.9, 0.8]) == [1.0, 1.0]


def test_bootstrap_ci_brackets_mean():
    v = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    lo, hi = gs.bootstrap_mean_ci(v, n_resamples=500)
    assert lo <= v.mean() <= hi
    assert np.isnan(gs.bootstrap_mean_ci(np.array([0.3]))[0])


def test_brief_paired_test_uses_brief_means_and_common_briefs():
    rows = []
    for b in ["b1", "b2", "b3", "b4", "b5", "b6"]:
        rows.append({"brief_id": b, "condition": "single_shot", "g_kbv": 0.0})
        rows.append({"brief_id": b, "condition": "single_shot", "g_kbv": 0.2})  # repetition
        rows.append({"brief_id": b, "condition": "multi_turn_pressure", "g_kbv": 0.5})
    rows.append({"brief_id": "b7", "condition": "multi_turn_pressure", "g_kbv": 1.0})  # unpaired brief
    res = gs.brief_paired_test(pd.DataFrame(rows))
    assert res["n_briefs_paired"] == 6
    assert res["mean_a"] == pytest.approx(0.1)
    assert res["mean_brief_delta"] == pytest.approx(0.4)
    assert res["wilcoxon_p"] < 0.05
    assert res["n_zero_deltas"] == 0
    assert not np.isnan(res["wilcoxon_p_exact"])


def test_brief_paired_test_rounds_ties_and_drops_zeros():
    # Deltas 1/3 - 0 and 2/3 - 1/3 differ by float noise; after rounding they tie.
    rows = []
    for b in ["b1", "b2", "b3", "b4", "b5", "b6"]:
        rows.append({"brief_id": b, "condition": "single_shot", "g_kbv": 0.0})
        rows.append({"brief_id": b, "condition": "multi_turn_pressure", "g_kbv": 1 / 3})
    rows.append({"brief_id": "b7", "condition": "single_shot", "g_kbv": 1 / 3})
    rows.append({"brief_id": "b7", "condition": "multi_turn_pressure", "g_kbv": 2 / 3})
    rows.append({"brief_id": "b8", "condition": "single_shot", "g_kbv": 0.2})
    rows.append({"brief_id": "b8", "condition": "multi_turn_pressure", "g_kbv": 0.2})
    res = gs.brief_paired_test(pd.DataFrame(rows))
    assert res["n_briefs_paired"] == 8 and res["n_zero_deltas"] == 1
    diffs = np.round(np.array([1 / 3] * 6 + [2 / 3 - 1 / 3, 0.0]), gs.WILCOXON_ROUND_DECIMALS)
    expected = gs.stats.wilcoxon(diffs, zero_method="wilcox", method=gs.WILCOXON_METHOD).pvalue
    assert res["wilcoxon_p"] == pytest.approx(expected)
    assert len(set(diffs[diffs != 0])) == 1  # all seven non-zero deltas tie after rounding


def test_brief_paired_test_all_zero_differences_gives_nan():
    rows = [{"brief_id": b, "condition": c, "g_kbv": 0.1} for b in ["b1", "b2"] for c in ["single_shot", "multi_turn_pressure"]]
    res = gs.brief_paired_test(pd.DataFrame(rows))
    assert np.isnan(res["wilcoxon_p"])


def test_summary_and_tests_end_to_end(tmp_path):
    scores = tmp_path / "scores"
    for i, b in enumerate(["b1", "b2", "b3", "b4", "b5"]):
        _write_structured(scores, "structured", f"ss{i}", ["satisfied"] * 4, cond="single_shot", brief=b)
        _write_structured(scores, "structured", f"mp{i}", ["satisfied", "satisfied", "stretched", "satisfied"], cond="multi_turn_pressure", brief=b)
    probes = pd.DataFrame({"run_id": [f"mp{i}" for i in range(5)], "turn_index": 5, "overall_accuracy": [1.0, 1.0, 1.0, 0.5, 1.0]})
    probes = probes.rename(columns={"turn_index": "probe_turn_index"})
    runs = gs.attach_recall(gs.load_structured_runs(scores, "structured"), probes, {"mp0": True})
    summary = gs.summarise(runs)
    mp = summary[summary["condition"] == "multi_turn_pressure"].iloc[0]
    assert mp["n_runs"] == 5 and mp["mean_g_kbv"] == pytest.approx(0.25)
    assert mp["n_perfect_recall"] == 4 and mp["n_knows"] == 4
    assert mp["n_perfect_recall_untruncated"] == 3
    assert mp["modal_n_not_satisfied"] == 1 and mp["share_1_not_satisfied"] == 1.0
    assert mp["mean_n_checks"] == 4.0
    ss = summary[summary["condition"] == "single_shot"].iloc[0]
    assert ss["n_perfect_recall"] == 0 and ss["mean_g_kbv"] == 0.0
    tests = gs.test_table(runs)
    assert tests["n_briefs_paired"].iloc[0] == 5
    assert tests["mean_brief_delta"].iloc[0] == pytest.approx(0.25)
    assert tests["holm_p"].iloc[0] == pytest.approx(tests["wilcoxon_p"].iloc[0])
    assert tests["holm_p_within_judge"].iloc[0] == pytest.approx(tests["wilcoxon_p"].iloc[0])
    assert tests["wilcoxon_method"].iloc[0] == gs.WILCOXON_METHOD


def test_holm_within_judge_family_is_per_judge():
    rows = []
    for m, j, delta in (("m1", "opus", 0.3), ("m2", "opus", 0.3), ("m3", "gpt", 0.3)):
        for i, b in enumerate(["b1", "b2", "b3", "b4", "b5", "b6", "b7"]):
            rows.append({"source": "s", "run_id": f"{m}_ss{i}", "brief_id": b, "model_id": m, "condition": "single_shot", "judge_model": j, "g_kbv": 0.0})
            rows.append({"source": "s", "run_id": f"{m}_mp{i}", "brief_id": b, "model_id": m, "condition": "multi_turn_pressure", "judge_model": j, "g_kbv": delta + 0.01 * i})
    t = gs.test_table(pd.DataFrame(rows)).set_index("model_id")
    assert t.loc["m3", "n_models_in_within_judge_family"] == 1
    assert t.loc["m3", "holm_p_within_judge"] == pytest.approx(t.loc["m3", "wilcoxon_p"])
    assert t.loc["m1", "n_models_in_within_judge_family"] == 2
    assert t["n_models_in_holm_family"].iloc[0] == 3


def test_opus_arm_family_prefers_e4_rows_and_holms_over_all_opus_models():
    opus, gpt = gs.OPUS_JUDGE, "openai/gpt-5.4"

    def row(source, model, judge, p, t):
        return {"source": source, "model_id": model, "judge_model": judge, "n_runs_single_shot": 76, "n_runs_pressure": 76,
                "n_briefs_paired": 38, "mean_g_kbv_single_shot_brief": 0.01, "mean_g_kbv_pressure_brief": 0.02,
                "mean_brief_delta": 0.01, "n_zero_deltas": 0, "wilcoxon_method": "asymptotic", "wilcoxon_stat": 1.0,
                "wilcoxon_p": p, "wilcoxon_p_exact": p, "paired_t_p": t}

    tests = pd.DataFrame([
        row("structured", "sonnet", gpt, 0.0001, 0.0001),        # GPT-judged: never in the Opus family
        row("structured_opus", "sonnet", opus, 0.04, 0.03),
        row("structured", "gpt", opus, 0.01, 0.02),
        row("structured", "llama", opus, 0.50, 0.40),
        row("structured_opus", "llama", opus, 0.02, 0.01),       # E4 row wins over the structured one
    ])
    t = gs.opus_arm_test_table(tests).set_index("model_id")
    assert sorted(t.index) == ["gpt", "llama", "sonnet"]
    assert t.loc["llama", "source"] == "structured_opus" and t.loc["llama", "wilcoxon_p"] == pytest.approx(0.02)
    assert t.loc["sonnet", "judge_model"] == opus
    # Holm over 0.01, 0.02, 0.04: 0.03, 0.04, 0.04.
    assert t.loc["gpt", "holm_p_opus_arm"] == pytest.approx(0.03)
    assert t.loc["llama", "holm_p_opus_arm"] == pytest.approx(0.04)
    assert t.loc["sonnet", "holm_p_opus_arm"] == pytest.approx(0.04)
    assert t["significant_holm_05_opus_arm"].tolist() == [1, 1, 1]
    assert (t["n_models_in_opus_arm_family"] == 3).all()
    assert gs.opus_arm_test_table(tests[tests["judge_model"] == gpt]).empty


def test_run_writes_csvs_and_provenance(tmp_path):
    root = tmp_path / "data"
    _write_structured(root / "scores", "structured", "r1", ["satisfied", "stretched"], cond="multi_turn_pressure")
    _write_structured(root / "scores", "structured", "r2", ["satisfied"], cond="single_shot")  # not in the parquet
    _write_structured(root / "scores", "structured", "r3", [], cond="single_shot")  # no checks
    # r4 is a stray Sonnet pilot run: in the parquet, judged on 2026-04-16 UTC. The sample rule drops it.
    _write_structured(root / "scores", "structured", "r4", ["violated"], model=SAMPLE_SONNET, cond="multi_turn_pressure")
    (root / "scores" / "judge_r4.jsonl").write_text(json.dumps({"run_id": "r4", "model_id": SAMPLE_SONNET, "timestamp": 1776340800.0}) + "\n")
    (root / "analysis").mkdir(parents=True)
    (root / "aggregated").mkdir(parents=True)
    pd.DataFrame({"run_id": ["r1"], "brief_id": ["b1"], "model_id": ["m"], "condition": ["multi_turn_pressure"],
                  "turn_index": [5], "overall_accuracy": [1.0], "constraint_recall": [1.0], "banned_recall": [1.0]}).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)
    pd.DataFrame({"run_id": ["r1", "r4"], "model_id": ["m", SAMPLE_SONNET], "condition": ["multi_turn_pressure"] * 2}).to_parquet(root / "aggregated" / "all_scores.parquet", index=False)
    (root / "transcripts").mkdir()
    (root / "transcripts" / "r1.jsonl").write_text(json.dumps({"metadata": {"run_id": "r1"}, "messages": [
        {"role": "assistant", "content": "p", "is_probe": True, "token_count": 5}]}) + "\n")
    out = tmp_path / "out"
    frames = gs.run(root, out)
    assert set(frames) == {"graded_severity_runs", "graded_severity_summary", "graded_severity_tests",
                           "graded_severity_opus_arm_tests"}
    for name in frames:
        assert (out / f"{name}.csv").exists()
        prov = json.loads((out / f"{name}.csv.provenance.json").read_text())
        assert prov["rows_out"] == len(frames[name])
        assert "rules" in prov and "inputs" in prov
    # No Opus-judged row in this tree: the seven-test family table is empty but written.
    assert frames["graded_severity_opus_arm_tests"].empty
    prov = json.loads((out / "graded_severity_tests.csv.provenance.json").read_text())
    assert prov["inputs"]["opus_rescoring_present"] is False
    # The parquet filter drops r2, the sample rule drops r4; both are recorded. File counts survive the merge.
    assert list(frames["graded_severity_runs"]["run_id"]) == ["r1"]
    assert [f["name"] for f in prov["filters"]] == ["parquet_filter", "drop_stray_sonnet_runs"]
    assert prov["filters"][0]["dropped_not_in_parquet"] == ["r2"]
    assert prov["filters"][1]["rows_dropped"] == 1 and prov["filters"][1]["n_stray_run_ids"] == 1
    assert prov["rows_in"] == {"structured_runs_loaded": 3, "structured_runs_after_parquet_filter": 2,
                               "structured_runs_after_stray_filter": 1, "probe_rows": 1}
    assert "Sonnet pressure n = 0" in prov["rules"]["denominators"]
    assert prov["inputs"]["structured_files_found"] == {"structured": 4, "structured_opus": 0}
    assert prov["inputs"]["structured_files_without_checks"] == {"structured": ["structured_r3.jsonl"], "structured_opus": []}
    assert prov["inputs"]["transcript_files_scanned"][str(root / "transcripts")] == 1
    assert prov["inputs"]["runs_with_probe_in_transcripts"] == 1
    assert "wilcoxon" in prov["rules"] and gs.WILCOXON_METHOD in prov["rules"]["wilcoxon"]
    assert "denominators" in prov["rules"]


# ---------------------------------------------------------------------------
# Extra constraint checks appended by the judge (E4 review)
# ---------------------------------------------------------------------------

def _write_structured_with_checks(d: Path, prefix: str, run_id: str, checks: list[tuple[str, str]], brief="b1"):
    rec = {
        "run_id": run_id, "brief_id": brief, "model_id": "m", "condition": "single_shot", "judge_model": "j",
        "constraint_adherence": 4,
        "constraint_checks": [{"constraint": c, "status": s, "reason": ""} for c, s in checks],
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}_{run_id}.jsonl").write_text(json.dumps(rec) + "\n")


def test_match_checks_exact_text_drops_extras_and_keeps_brief_order():
    hc = ["alpha", "beta"]
    checks = [{"constraint": "beta", "status": "satisfied"}, {"constraint": "alpha", "status": "violated"},
              {"constraint": "no gamma (banned)", "status": "satisfied"}]
    kept, dropped, method = gs.match_checks_to_brief(checks, hc)
    assert method == "exact_text"
    assert [(i, c["constraint"]) for i, c in kept] == [(0, "alpha"), (1, "beta")]
    assert [c["constraint"] for c in dropped] == ["no gamma (banned)"]


def test_match_checks_falls_back_to_truncation_when_text_differs():
    hc = ["alpha", "beta"]
    checks = [{"constraint": "Alpha, paraphrased", "status": "satisfied"}, {"constraint": "beta", "status": "stretched"},
              {"constraint": "extra", "status": "violated"}]
    kept, dropped, method = gs.match_checks_to_brief(checks, hc)
    assert method == "truncated"
    assert [c["constraint"] for _, c in kept] == ["Alpha, paraphrased", "beta"]
    assert [c["constraint"] for c in dropped] == ["extra"]


def test_match_checks_keeps_everything_without_brief_or_without_extras():
    checks = [{"constraint": "a", "status": "satisfied"}, {"constraint": "b", "status": "violated"}]
    assert gs.match_checks_to_brief(checks, None) == (list(enumerate(checks)), [], "none")
    assert gs.match_checks_to_brief(checks, ["a", "b", "c"]) == (list(enumerate(checks)), [], "none")


def test_loader_drops_appended_extras_and_records_them(tmp_path):
    briefs = {"b1": ["alpha", "beta"]}
    _write_structured_with_checks(tmp_path, "structured", "r1",
                                  [("alpha", "satisfied"), ("beta", "satisfied"), ("banned x", "violated"), ("criterion y", "stretched")])
    _write_structured_with_checks(tmp_path, "structured", "r2", [("alpha", "satisfied"), ("beta", "stretched")])
    checks = gs.load_structured_checks(tmp_path, "structured", briefs=briefs)
    assert checks[checks["run_id"] == "r1"]["constraint"].tolist() == ["alpha", "beta"]
    runs = gs.summarise_runs(checks)
    assert runs.set_index("run_id").loc["r1", "g_kbv"] == 0.0  # the violated extra does not count
    assert runs.set_index("run_id").loc["r2", "g_kbv"] == 0.5
    dropped = checks.attrs["dropped_extra_checks"]
    assert dropped == [{"run_id": "r1", "source": "structured", "brief_id": "b1", "method": "exact_text",
                        "n_dropped": 2, "texts": ["banned x", "criterion y"]}]
    # Without briefs the old behaviour holds: every check counts.
    all_checks = gs.load_structured_checks(tmp_path, "structured")
    assert len(all_checks[all_checks["run_id"] == "r1"]) == 4
    assert all_checks.attrs["dropped_extra_checks"] == []


def test_run_records_dropped_extras_in_provenance(tmp_path):
    root = tmp_path / "data"
    briefs_dir = root / "briefs"
    briefs_dir.mkdir(parents=True)
    # The brief must pass the brief schema: id pattern, objective length, list sizes.
    (briefs_dir / "test_01.yaml").write_text(
        "id: test_01\ndomain: test\nobjective: a synthetic objective for the test\n"
        "hard_constraints:\n  - alpha\n  - beta\n  - gamma\n"
        "success_criteria:\n  - s1\n  - s2\nplausible_directions:\n  - d1\n  - d2\nbanned_moves:\n  - m1\n  - m2\n")
    _write_structured_with_checks(root / "scores", "structured", "r1",
                                  [("alpha", "satisfied"), ("beta", "satisfied"), ("gamma", "satisfied"), ("extra", "violated")],
                                  brief="test_01")
    (root / "analysis").mkdir()
    (root / "aggregated").mkdir()
    pd.DataFrame({"run_id": ["r1"], "brief_id": ["test_01"], "model_id": ["m"], "condition": ["single_shot"],
                  "turn_index": [5], "overall_accuracy": [1.0], "constraint_recall": [1.0], "banned_recall": [1.0]}).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)
    pd.DataFrame({"run_id": ["r1"], "model_id": ["m"], "condition": ["single_shot"]}).to_parquet(root / "aggregated" / "all_scores.parquet", index=False)
    (root / "transcripts").mkdir()
    loaded = gs.load_all_structured_runs(root, briefs_dir)
    assert loaded.attrs["dropped_extra_checks"]["structured"][0]["texts"] == ["extra"]
    assert loaded["n_checks"].tolist() == [3]


def test_write_csv_with_provenance_names_sidecar_after_csv(tmp_path):
    """Plan P12: the sidecar is ``<name>.csv.provenance.json``, also used by stretch_violate and human_severity."""
    gs.write_csv_with_provenance(pd.DataFrame({"a": [1, 2]}), tmp_path / "x.csv", {"inputs": ["i"]})
    assert (tmp_path / "x.csv").exists()
    assert not (tmp_path / "x.provenance.json").exists()
    prov = json.loads((tmp_path / "x.csv.provenance.json").read_text())
    assert prov["csv"] == "x.csv" and prov["rows_out"] == 2
