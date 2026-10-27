"""Tests for drift_bench.analysis.camera_ready.figures_cr (plan item D10).

Synthetic CSVs, parquet files, transcripts and briefs under tmp_path. No
network, no API keys, no reads of drift_bench/data or .env. Every figure
function must write a PDF and a provenance sidecar, and two runs on the same
inputs must produce byte-identical PDFs.
"""

from __future__ import annotations

import builtins
import json
import math
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from drift_bench.analysis.camera_ready import figures_cr as fc
from drift_bench.run_analysis import discover_camera_ready_modules

GPT, SONNET, PRO, FLASH = fc.GPT, fc.SONNET, fc.GEMINI_PRO, fc.FLASH_LITE
SS, MTN, MTP, CKP = fc.SS, fc.MTN, fc.MTP, fc.CKP
OPUS, GPTJ = fc.OPUS_JUDGE, fc.GPT_JUDGE
DATE = "2026-10-26"

BRIEF = {
    "id": "b1",
    "domain": "test",
    "objective": "Study something",
    "hard_constraints": [
        "Budget must not exceed 25,000 GBP in total",
        "Use only the public open dataset released before 2020",
    ],
    "success_criteria": ["works"],
    "plausible_directions": ["x"],
    "banned_moves": ["No new data collection from human participants"],
}
GOOD_PROBE = (
    "1. Objective: study something. 2. Constraints: budget 25,000 gbp; only the public open "
    "dataset released before 2020. 3. Banned: new data collection from human participants."
)
BAD_PROBE = "I recall nothing in particular."


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _kbv_row(arm, variant, model, judge, n, k, waived=0):
    lo, hi = fc.wilson(k, n)
    return {
        "judge_arm": arm, "variant": variant, "model_id": model, "condition": MTP, "judge_model": judge,
        "n": n, "n_with_probe": n, "knows": n, "n_truncated_waived": waived, "knows_flipped_by_waiver": 0,
        "violates": k, "kbv_count": k, "kbv_rate": k / n, "ci_low": lo, "ci_high": hi,
    }


def _write_kbv_common_judge(d: Path) -> None:
    rows = []
    counts = {  # (arm, variant) -> {model: kbv_count}
        ("opus", "clean"): {GPT: 6, SONNET: 57, PRO: 14},
        ("opus", "clean_truncated_probes_waived"): {GPT: 6, SONNET: 57, PRO: 19},
        ("gpt", "clean"): {GPT: 32, SONNET: 75, PRO: 41},
        ("gpt", "clean_truncated_probes_waived"): {GPT: 32, SONNET: 75, PRO: 61},
        ("published", "clean"): {GPT: 6, SONNET: 75, PRO: 14},
        ("published", "clean_truncated_probes_waived"): {GPT: 6, SONNET: 75, PRO: 19},
    }
    for (arm, variant), per_model in counts.items():
        for model, k in per_model.items():
            if arm == "opus":
                judge = OPUS
            elif arm == "gpt":
                judge = GPTJ
            else:
                judge = GPTJ if model == SONNET else OPUS
            rows.append(_kbv_row(arm, variant, model, judge, 76, k, waived=62 if (model == PRO and "waived" in variant) else 0))
    pd.DataFrame(rows).to_csv(d / "kbv_common_judge.csv", index=False)
    pd.DataFrame([
        {"judge_arm": "opus", "variant": "clean", "n_models": 3, "min_kbv_rate": 6 / 76, "argmin_model": GPT,
         "max_kbv_rate": 57 / 76, "argmax_model": SONNET, "cut": 0.5, "n_above_cut": 1, "models_above_cut": SONNET},
    ]).to_csv(d / "kbv_common_judge_range.csv", index=False)


def _write_published_arm(d: Path) -> None:
    rows = []
    cells = {
        (GPT, MTP): 6, (GPT, CKP): 3, (GPT, MTN): 3,
        (SONNET, MTP): 75, (SONNET, CKP): 66, (SONNET, MTN): 72,
        (PRO, MTP): 14, (PRO, CKP): 15, (PRO, MTN): 5,
        (FLASH, MTP): 58, (FLASH, CKP): 55, (FLASH, MTN): 21,
    }
    for sample in ("as_published", "clean"):
        for (model, cond), k in cells.items():
            lo, hi = fc.wilson(k, 76)
            rows.append({"sample": sample, "model_id": model, "condition": cond, "n_scored": 76, "n_with_final_probe": 76,
                         "knows": 76, "knows_imputed": 0, "violates": k, "kbv_count": k, "kbv_rate": k / 76,
                         "ci_low": lo, "ci_high": hi})
        for model in (GPT, SONNET, PRO, FLASH):
            rows.append({"sample": sample, "model_id": model, "condition": SS, "n_scored": 76, "n_with_final_probe": 0,
                         "knows": 0, "knows_imputed": 0, "violates": 5, "kbv_count": 0, "kbv_rate": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan")})
    pd.DataFrame(rows).to_csv(d / "kbv_published_arm.csv", index=False)


def _write_corrections(d: Path) -> None:
    base = "KBV among Sonnet runs with a final probe; cap = 4096; untruncated final (final assistant token_count < cap)"
    rows = [
        {"id": "B3", "quantity": "Sonnet KBV, multi_turn_pressure, untruncated final (final assistant token_count < cap), sample=as_published",
         "published_value": "100%", "recomputed_value": 100.0, "rule": base, "numerator": 30.0, "denominator": 30.0, "source_files": "x", "note": ""},
        {"id": "B3", "quantity": "Sonnet KBV, multi_turn_pressure, untruncated final (final assistant token_count < cap), sample=clean",
         "published_value": "100%", "recomputed_value": 100.0, "rule": base, "numerator": 26.0, "denominator": 26.0, "source_files": "x", "note": ""},
        {"id": "B3", "quantity": "Sonnet KBV, checkpointed_pressure, untruncated final (final assistant token_count < cap), sample=clean",
         "published_value": "55% (11/20)", "recomputed_value": 55.0, "rule": base, "numerator": 11.0, "denominator": 20.0, "source_files": "x", "note": ""},
        {"id": "B2", "quantity": "something else", "published_value": "", "recomputed_value": 1.0, "rule": "", "numerator": 1.0,
         "denominator": 2.0, "source_files": "x", "note": ""},
    ]
    pd.DataFrame(rows).to_csv(d / "corrections_ledger.csv", index=False)


def _write_per_turn(d: Path) -> None:
    rows = []
    dims = ["objective_fidelity", "constraint_adherence", "alternative_coverage", "complexity_inflation"]
    for model in (GPT, FLASH):
        for cond in (MTN, MTP):
            for turn in range(1, 7):
                row = {"model_id": model, "condition": cond, "turn": turn,
                       "source": "published_april_opus" if turn == 6 else "per_turn_opus_20261026",
                       "format": "final_sectioned" if turn == 6 else "refinement_turn", "n": 38}
                for dim in dims:
                    base = 3.5 if dim != "complexity_inflation" else 0.2 * turn
                    if cond == MTP and dim == "constraint_adherence":
                        base -= 0.1 * turn
                    row[f"mean_{dim}"] = base
                    row[f"se_{dim}"] = 0.05
                    row[f"ci_lo_{dim}"] = base - 0.1
                    row[f"ci_hi_{dim}"] = base + 0.1
                row.update({"share_ca_lt4": 0.2, "share_ca_lt4_ci_lo": 0.1, "share_ca_lt4_ci_hi": 0.3, "kbv_rate": 0.2,
                            "kbv_n": 38, "kbv_ci_lo": 0.1, "kbv_ci_hi": 0.3, "probe_recall_mean": 1.0})
                rows.append(row)
    pd.DataFrame(rows).to_csv(d / "e5_per_turn_cells.csv", index=False)


def _write_classification(d: Path) -> None:
    rows = []
    counts = {SS: (527, 175), MTN: (531, 166), MTP: (530, 282), CKP: (528, 246)}
    for sample in ("as_published", "clean"):
        for cond, (n, k) in counts.items():
            lo, hi = fc.wilson(k, n)
            rows.append({"sample": sample, "condition": cond, "n": n, "noncompliant": k, "rate": k / n, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(d / "noncompliance_by_condition.csv", index=False)
    mix = []
    shares = {SS: {"mild_drift": 437, "no_drift": 88, "trajectory_drift": 1, "trajectory_lock_in": 1},
              MTN: {"mild_drift": 405, "no_drift": 61, "trajectory_drift": 52, "trajectory_lock_in": 13},
              MTP: {"trajectory_drift": 249, "mild_drift": 215, "trajectory_lock_in": 65, "no_drift": 1},
              CKP: {"mild_drift": 287, "trajectory_drift": 222, "trajectory_lock_in": 19}}
    for cond, per in shares.items():
        denom = sum(per.values())
        for cls, n in per.items():
            mix.append({"section": "classification_mix", "condition": cond, "model_id": "", "subset": "", "statistic": cls,
                        "n": float(n), "share": n / denom, "denominator": float(denom)})
    mix.append({"section": "events_per_run", "condition": SS, "model_id": "", "subset": "", "statistic": "n_events == 0",
                "n": 26.0, "share": 0.05, "denominator": 527.0})
    pd.DataFrame(mix).to_csv(d / "auditor_onset_summary.csv", index=False)


def _write_tokens(d: Path) -> None:
    rows = []
    i = 0
    for model in (GPT, SONNET, FLASH):
        for cond in (SS, MTN, MTP, CKP):
            for rep in range(8):
                i += 1
                rows.append({"run_id": f"r{i:04d}", "brief_id": "b1", "model_id": model, "condition": cond,
                             "last_output_tokens": 400 + 90 * rep + (300 if cond == MTP else 0),
                             "j_complexity_inflation": (rep % 3) + (1 if cond == MTP else 0)})
    runs = pd.DataFrame(rows)
    runs.to_csv(d / "tokens_context_runs.csv", index=False)
    crossed = runs[(runs["model_id"] == SONNET) & (runs["condition"].isin([SS, MTP]))][["run_id", "model_id", "condition"]].copy()
    crossed["opus_complexity_inflation"] = 2.0
    crossed.to_csv(d / "crossed_scores_runs.csv", index=False)


def _transcript(run_id, brief_id, model_id, condition, probe_text, n_turns=2):
    msgs = [{"role": "system", "content": "sys", "timestamp": 1.0, "token_count": None, "latency_ms": None, "is_probe": False}]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"turn {t}", "timestamp": 2.0 + t, "token_count": None, "latency_ms": None, "is_probe": False})
        msgs.append({"role": "assistant", "content": f"answer {t}", "timestamp": 2.5 + t, "token_count": 100, "latency_ms": 1.0, "is_probe": False})
        if t < n_turns - 1:
            msgs.append({"role": "user", "content": "restate", "timestamp": 2.6 + t, "token_count": None, "latency_ms": None, "is_probe": True})
            msgs.append({"role": "assistant", "content": probe_text, "timestamp": 2.7 + t, "token_count": 50, "latency_ms": 1.0, "is_probe": True})
    return {"metadata": {"run_id": run_id, "brief_id": brief_id, "model_id": model_id, "condition": condition, "repetition": 0,
                         "start_time": 1.0, "end_time": 9.0, "total_input_tokens": 1, "total_output_tokens": 1,
                         "total_cost_usd": 0.0, "error": None}, "messages": msgs}


def _write_followup(root: Path, runs: list[tuple[str, str, str, int, str, str]]) -> None:
    """runs: (run_id, model_id, condition, adherence, probe_text, judge_model)."""
    (root / "aggregated").mkdir(parents=True)
    (root / "transcripts").mkdir()
    (root / "scores").mkdir()
    rows = []
    for run_id, model, cond, adh, probe, judge in runs:
        rows.append({"run_id": run_id, "brief_id": "b1", "model_id": model, "condition": cond, "judge_model": judge,
                     "j_objective_fidelity": 4, "j_constraint_adherence": adh, "j_alternative_coverage": 4,
                     "j_complexity_inflation": 1, "j_summary": "", "a_objective_fidelity": 4, "a_constraint_adherence": 4,
                     "a_alternative_coverage": 4, "a_complexity_inflation": 1, "a_recoverability": 4, "a_drift_classification": "mild_drift"})
        with open(root / "transcripts" / f"{run_id}_b1_{cond}_{model.replace('/', '_')}.jsonl", "w") as fh:
            fh.write(json.dumps(_transcript(run_id, "b1", model, cond, probe)) + "\n")
    pd.DataFrame(rows).to_parquet(root / "aggregated" / "all_scores.parquet", index=False)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    analysis = root / "camera_ready" / "analysis"
    analysis.mkdir(parents=True)
    _write_kbv_common_judge(analysis)
    _write_published_arm(analysis)
    _write_corrections(analysis)
    _write_per_turn(analysis)
    _write_classification(analysis)
    _write_tokens(analysis)
    (root / "briefs").mkdir()
    with open(root / "briefs" / "b1.yaml", "w") as fh:
        yaml.safe_dump(BRIEF, fh)
    # follow-up A: Flash-Lite at temperature 0.7, three multi-turn conditions
    fa = []
    for tag, cond in (("n", MTN), ("p", MTP), ("c", CKP)):
        fa += [(f"a{tag}1", FLASH, cond, 3, GOOD_PROBE, OPUS),   # kbv
               (f"a{tag}2", FLASH, cond, 4, GOOD_PROBE, OPUS),   # knows, complies
               (f"a{tag}3", FLASH, cond, 2, BAD_PROBE, OPUS)]    # violates without recall
    _write_followup(root / "followup_a", fa)
    fb = [("b1", GPT, fc.MTP_RIGOR, 3, GOOD_PROBE, OPUS), ("b2", GPT, fc.MTP_RIGOR, 4, GOOD_PROBE, OPUS),
          ("b3", SONNET, fc.MTP_RIGOR, 3, GOOD_PROBE, GPTJ), ("b4", SONNET, fc.MTP_RIGOR, 3, GOOD_PROBE, GPTJ),
          ("b5", FLASH, fc.MTP_RIGOR, 3, GOOD_PROBE, OPUS), ("b6", FLASH, fc.MTP_RIGOR, 4, BAD_PROBE, OPUS),
          ("b7", FLASH, SS, 3, GOOD_PROBE, OPUS)]
    _write_followup(root / "followup_b", fb)
    return root


@pytest.fixture
def analysis_dir(data_root: Path) -> Path:
    return data_root / "camera_ready" / "analysis"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wilson_matches_csv_values() -> None:
    lo, hi = fc.wilson(57, 76)
    assert math.isclose(lo, 0.6422324806394772, abs_tol=1e-9)
    assert math.isclose(hi, 0.8337107264667001, abs_tol=1e-9)
    assert all(math.isnan(v) for v in fc.wilson(0, 0))


def test_module_opts_out_of_item_6() -> None:
    assert fc.RUN_IN_ITEM_6 is False
    assert "figures_cr" not in [name for name, _ in discover_camera_ready_modules()]


@pytest.mark.parametrize("name", ["kbv_by_judge_arm", "trajectories", "drift_checkpoint", "drift_classification", "complexity_vs_tokens"])
def test_each_analysis_figure_writes_pdf_and_provenance(name: str, analysis_dir: Path, tmp_path: Path) -> None:
    figures = tmp_path / "figs"
    numbers = fc.FIGURE_FUNCTIONS[name](analysis_dir, figures, DATE)
    pdf = figures / f"fig_{name}.pdf"
    prov_path = figures / f"fig_{name}.provenance.json"
    assert pdf.exists() and pdf.stat().st_size > 1000
    assert pdf.read_bytes().startswith(b"%PDF")
    prov = json.loads(prov_path.read_text())
    for key in ("inputs", "arm", "sample", "date", "numbers_drawn", "output"):
        assert key in prov, key
    assert prov["date"] == DATE
    assert prov["output"] == pdf.name
    assert isinstance(numbers, pd.DataFrame) and len(numbers) == len(prov["numbers_drawn"]) > 0
    assert prov["sample"]["rule"].startswith("drop runs with model_id == anthropic/claude-sonnet-4-6")
    assert b"CreationDate" not in pdf.read_bytes()


def test_sensitivity_writes_pdf_and_provenance(data_root: Path, analysis_dir: Path, tmp_path: Path) -> None:
    figures = tmp_path / "figs"
    numbers = fc.fig_sensitivity(data_root, analysis_dir, figures, DATE)
    prov = json.loads((figures / "fig_sensitivity.provenance.json").read_text())
    assert (figures / "fig_sensitivity.pdf").exists()
    assert prov["sample"]["followup_a"]["rows_with_final_probe"] == 9
    assert prov["sample"]["followup_b"]["rows_with_final_probe"] == 6
    b = numbers[numbers["panel"] == "b_pressure_type"].set_index(["model_id", "series"])
    assert b.loc[(GPT, "Rigor pressure (rigor follow-up)"), "kbv_count"] == 1
    assert b.loc[(GPT, "Rigor pressure (rigor follow-up)"), "n"] == 2
    assert b.loc[(SONNET, "Rigor pressure (rigor follow-up)"), "kbv_rate"] == 1.0
    assert b.loc[(SONNET, "Rigor pressure (rigor follow-up)"), "judge_model"] == GPTJ
    assert b.loc[(FLASH, "Rigor pressure (rigor follow-up)"), "kbv_count"] == 1  # b6 violates? no: adherence 4; b5 kbv
    a = numbers[numbers["panel"] == "a_temperature"].set_index(["condition", "series"])
    assert a.loc[(MTP, "Temperature follow-up, temperature 0.7"), "kbv_count"] == 1
    assert a.loc[(MTP, "Temperature follow-up, temperature 0.7"), "n"] == 3
    assert a.loc[(MTP, "Main benchmark, temperature 1.0"), "kbv_count"] == 58


def test_followup_kbv_drops_stray_and_unscored(data_root: Path) -> None:
    cells, info = fc.followup_kbv(data_root / "followup_b", data_root / "briefs", [fc.MTP_RIGOR], main_stray={"b3"})
    assert info["rows_in"] == 7
    assert info["stray_main_ids_matched"] == 1
    assert info["rows_after_stray"] == 6
    son = cells[cells["model_id"] == SONNET].iloc[0]
    assert son["n"] == 1 and son["kbv_count"] == 1
    assert set(cells["condition"]) == {fc.MTP_RIGOR}


def test_kbv_figure_numbers(analysis_dir: Path, tmp_path: Path) -> None:
    numbers = fc.fig_kbv_by_judge_arm(analysis_dir, tmp_path / "figs", DATE)
    # Gemini Pro carries a measured and a capped-as-graded row under both arms; the others only measured.
    pro = numbers[numbers["model_id"] == PRO]
    assert sorted(pro["variant"]) == ["capped_as_graded", "capped_as_graded", "measured", "measured"]
    assert set(numbers[numbers["model_id"] == GPT]["variant"]) == {"measured"}
    meas = numbers[numbers["variant"] == "measured"].set_index(["model_id", "judge_arm"])
    assert math.isclose(meas.loc[(PRO, "opus"), "kbv_rate"], 19 / 76)
    assert meas.loc[(SONNET, "gpt"), "published_arm"] and not meas.loc[(SONNET, "opus"), "published_arm"]
    assert meas.loc[(GPT, "opus"), "published_arm"] and not meas.loc[(GPT, "gpt"), "published_arm"]
    # Ordered by the Opus-arm rate: GPT (6) < Pro (19) < Sonnet (57), top row = highest y.
    y = meas.xs("opus", level="judge_arm")["y"]
    assert y[GPT] < y[PRO] < y[SONNET]
    prov = json.loads((tmp_path / "figs" / "fig_kbv_by_judge_arm.provenance.json").read_text())
    assert prov["reference_line"] == "50% of runs"
    assert prov["range"]


def test_trajectories_numbers(analysis_dir: Path, tmp_path: Path) -> None:
    numbers = fc.fig_trajectories(analysis_dir, tmp_path / "figs", DATE)
    assert set(numbers["dimension"]) == {"constraint_adherence", "complexity_inflation"}
    assert "alternative_coverage" not in set(numbers["dimension"])
    t6 = numbers[numbers["turn"] == 6]
    assert set(t6["format"]) == {"final_sectioned"} and len(t6) == 2 * 2 * 2
    assert len(numbers) == 2 * 2 * 2 * 6
    prov = json.loads((tmp_path / "figs" / "fig_trajectories.provenance.json").read_text())
    assert prov["x_positions"]["final"] > prov["x_positions"]["break"] > 5


def test_checkpoint_numbers(analysis_dir: Path, tmp_path: Path) -> None:
    numbers = fc.fig_drift_checkpoint(analysis_dir, tmp_path / "figs", DATE)
    extra = numbers[numbers["row"] == "Sonnet 4.6, untruncated finals"].set_index("condition")
    assert extra.loc[MTP, "kbv_count"] == 26 and extra.loc[MTP, "n"] == 26 and extra.loc[MTP, "kbv_rate"] == 1.0
    assert extra.loc[CKP, "kbv_count"] == 11 and extra.loc[CKP, "n"] == 20 and math.isclose(extra.loc[CKP, "kbv_rate"], 0.55)
    lo, hi = fc.wilson(11, 20)
    assert math.isclose(extra.loc[CKP, "ci_low"], lo) and math.isclose(extra.loc[CKP, "ci_high"], hi)
    all_runs = numbers[numbers["subset"] == "all runs"]
    assert len(all_runs) == 2 * 4
    assert set(all_runs[all_runs["model_id"] == SONNET]["judge_model"]) == {GPTJ}
    assert set(all_runs[all_runs["model_id"] == GPT]["judge_model"]) == {OPUS}


def test_classification_numbers(analysis_dir: Path, tmp_path: Path) -> None:
    numbers = fc.fig_drift_classification(analysis_dir, tmp_path / "figs", DATE)
    a = numbers[numbers["panel"] == "a_noncompliance"].set_index("condition")
    assert math.isclose(a.loc[MTP, "share"], 282 / 530)
    b = numbers[numbers["panel"] == "b_classification"]
    assert len(b) == 4 * 4  # every class is listed, missing ones with count 0
    ckp_nd = b[(b["condition"] == CKP) & (b["class"] == "no_drift")].iloc[0]
    assert ckp_nd["count"] == 0 and ckp_nd["share"] == 0.0
    for cond in (SS, MTN, MTP, CKP):
        assert math.isclose(b[b["condition"] == cond]["share"].sum(), 1.0, abs_tol=1e-9)


def test_complexity_sonnet_handling(analysis_dir: Path, tmp_path: Path) -> None:
    numbers = fc.fig_complexity_vs_tokens(analysis_dir, tmp_path / "figs", DATE)
    prov = json.loads((tmp_path / "figs" / "fig_complexity_vs_tokens.provenance.json").read_text())
    s = prov["sample"]
    assert s["rows_in"] == 3 * 4 * 8
    assert s["sonnet_rows_replaced_by_crossed_opus"] == 16
    assert s["sonnet_rows_dropped_no_opus_score"] == 16
    assert s["rows_drawn"] == 96 - 16 == prov["scatter_points"]
    assert s["rows_drawn_by_condition"][MTN] == 16 and s["rows_drawn_by_condition"][MTP] == 24
    assert set(numbers["condition"]) == {SS, MTN, MTP, CKP}
    assert (numbers["n"] > 0).all()


def test_run_writes_all_six(data_root: Path, tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    results = fc.run(data_root, tmp_path / "unused_out", figures_dir=figures, date=DATE)
    assert set(results) == set(fc.FIGURE_FUNCTIONS)
    for name in fc.FIGURE_FUNCTIONS:
        assert (figures / f"fig_{name}.pdf").exists()
        assert (figures / f"fig_{name}.provenance.json").exists()
        assert isinstance(results[name], pd.DataFrame)
    assert not (tmp_path / "unused_out").exists()


def test_run_is_deterministic(data_root: Path, tmp_path: Path) -> None:
    fc.run(data_root, None, figures_dir=tmp_path / "one", date=DATE)
    fc.run(data_root, None, figures_dir=tmp_path / "two", date=DATE)
    for name in fc.FIGURE_FUNCTIONS:
        a = (tmp_path / "one" / f"fig_{name}.pdf").read_bytes()
        b = (tmp_path / "two" / f"fig_{name}.pdf").read_bytes()
        assert a == b, f"{name}: PDF bytes differ between two runs"
        assert (tmp_path / "one" / f"fig_{name}.provenance.json").read_text() == (tmp_path / "two" / f"fig_{name}.provenance.json").read_text()


def test_run_reads_nothing_outside_tmp(data_root: Path, tmp_path: Path, monkeypatch) -> None:
    """No reads of drift_bench/data or .env; the repo figures directory is untouched."""
    repo_data = (fc.REPO_ROOT / "drift_bench" / "data").resolve()
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            opened.append(os.fspath(file) if not isinstance(file, bytes) else file.decode())
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    before = {p: p.stat().st_mtime_ns for p in fc.FIGURES_DIR.glob("*")} if fc.FIGURES_DIR.exists() else {}
    fc.run(data_root, None, figures_dir=tmp_path / "figs", date=DATE)
    monkeypatch.undo()
    for path in opened:
        p = Path(path)
        assert not p.name == ".env" and not p.name.endswith(".env"), path
        try:
            assert not p.resolve().is_relative_to(repo_data), path
        except OSError:
            pass
    after = {p: p.stat().st_mtime_ns for p in fc.FIGURES_DIR.glob("*")} if fc.FIGURES_DIR.exists() else {}
    assert before == after
