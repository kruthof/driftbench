"""Tests for drift_bench.analysis.camera_ready.per_model_tables.

Synthetic score frames. No network, no API keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready import per_model_tables as pm


def _scores():
    rows = []
    dims = list(pm.JUDGE_DIMS) + list(pm.AUDITOR_DIMS)
    # Model a, single_shot: adherence 4, 3, 2 -> mean 3, sd 1, se 1/sqrt(3).
    for i, v in enumerate([4, 3, 2]):
        r = {"run_id": f"a_ss_{i}", "model_id": "openai/gpt-5.4",
             "condition": "single_shot", "a_drift_classification": "no_drift"}
        r.update({d: v for d in dims})
        rows.append(r)
    # Model a, pressure: two runs, one trajectory_drift.
    for i, (v, cls) in enumerate([(4, "mild_drift"), (1, "trajectory_drift")]):
        r = {"run_id": f"a_mp_{i}", "model_id": "openai/gpt-5.4",
             "condition": "multi_turn_pressure", "a_drift_classification": cls}
        r.update({d: v for d in dims})
        rows.append(r)
    # Model with an underscore-bearing id, one neutral run.
    r = {"run_id": "b_mn_0", "model_id": "together_ai/some_org/model_x",
         "condition": "multi_turn_neutral", "a_drift_classification": "trajectory_lock_in"}
    r.update({d: 2 for d in dims})
    rows.append(r)
    return pd.DataFrame(rows)


def test_cell_means_se_is_sample_sd_over_sqrt_n():
    cells = pm.cell_means(_scores())
    ss = cells[(cells["model_id"] == "openai/gpt-5.4") & (cells["condition"] == "single_shot")].iloc[0]
    assert ss["n"] == 3
    assert ss["j_constraint_adherence_mean"] == 3.0
    assert ss["j_constraint_adherence_se"] == np.std([4, 3, 2], ddof=1) / np.sqrt(3)
    assert ss["share_no_drift"] == 1.0
    mp = cells[(cells["model_id"] == "openai/gpt-5.4") & (cells["condition"] == "multi_turn_pressure")].iloc[0]
    assert mp["share_trajectory_drift"] == 0.5
    assert mp["share_mild_drift"] == 0.5
    # A single-run cell has no sample SD.
    one = cells[cells["model_id"] == "together_ai/some_org/model_x"].iloc[0]
    assert one["n"] == 1 and np.isnan(one["j_constraint_adherence_se"])
    # Model order follows MODEL_ORDER, conditions follow CONDITIONS.
    assert list(cells["condition"][:2]) == ["single_shot", "multi_turn_pressure"]


def test_attach_kbv_rates_leaves_single_shot_nan():
    cells = pm.cell_means(_scores())
    kbv = pd.DataFrame([{
        "condition_score": "multi_turn_pressure", "model_id_score": "openai/gpt-5.4",
        "total": 2, "knows_count": 2, "violates_count": 1, "kbv_count": 1,
        "kbv_rate": 0.5, "knows_rate": 1.0, "violates_rate": 0.5,
    }])
    out = pm.attach_kbv_rates(cells, kbv)
    ss = out[(out["model_id"] == "openai/gpt-5.4") & (out["condition"] == "single_shot")].iloc[0]
    mp = out[(out["model_id"] == "openai/gpt-5.4") & (out["condition"] == "multi_turn_pressure")].iloc[0]
    assert np.isnan(ss["kbv_rate"])
    assert mp["kbv_rate"] == 0.5 and mp["kbv_n"] == 2
    assert np.isnan(out[out["model_id"] == "together_ai/some_org/model_x"].iloc[0]["kbv_rate"])


def test_render_latex_one_row_per_cell_and_escaped_underscores():
    cells = pm.attach_kbv_rates(pm.cell_means(_scores()), pd.DataFrame(
        columns=["condition_score", "model_id_score", "total", "kbv_count",
                 "kbv_rate", "knows_rate"]))
    tex = pm.render_latex(cells)
    body_rows = [ln for ln in tex.splitlines() if ln.endswith("\\\\") and "&" in ln
                 and not ln.startswith("Model")]
    # Two tabulars, three cells each.
    assert len(body_rows) == 2 * len(cells) == 6
    assert "some\\_org/model\\_x" in tex
    assert "some_org" not in tex.replace("some\\_org", "")
    assert tex.count(r"\multirow{2}{*}{GPT-5.4}") == 2
    assert tex.count(r"\begin{tabular}") == 2
    assert r"\resizebox{\textwidth}{!}" in tex
    assert r"\footnotesize" in tex
    assert "booktabs,multirow,graphicx" in tex
    assert " & SS & 3 & " in tex and " & MT-P & 2 & " in tex
    # Single-shot KBV renders as a dash.
    ss_line = [ln for ln in body_rows if " & SS & " in ln][0]
    assert " & -- & " in ss_line


def test_kbv_summary_from_probes_matches_definition():
    scores = pd.DataFrame([
        {"run_id": "a", "model_id": "m", "condition": "multi_turn_pressure", "j_constraint_adherence": 3},
        {"run_id": "b", "model_id": "m", "condition": "multi_turn_pressure", "j_constraint_adherence": 4},
        {"run_id": "c", "model_id": "m", "condition": "multi_turn_pressure", "j_constraint_adherence": 2},
        {"run_id": "d", "model_id": "m", "condition": "single_shot", "j_constraint_adherence": 2},
    ])
    probes = pd.DataFrame([
        {"run_id": "a", "overall_accuracy": 1.0},   # knows, violates -> kbv
        {"run_id": "b", "overall_accuracy": 0.9},   # knows, complies
        {"run_id": "c", "overall_accuracy": 0.5},   # does not know, violates
    ])
    k = pm.kbv_summary_from_probes(scores, probes)
    assert len(k) == 1  # single_shot has no probe row
    r = k.iloc[0]
    assert r["condition_score"] == "multi_turn_pressure" and r["total"] == 3
    assert r["knows_count"] == 2 and r["violates_count"] == 2 and r["kbv_count"] == 1
    assert r["kbv_rate"] == 1 / 3


def test_kbv_denominator_sentence_adapts_to_cells():
    cells = pd.DataFrame([
        {"model_id": "openai/gpt-5.4", "condition": "multi_turn_pressure", "n": 10, "kbv_n": 10.0},
        {"model_id": "openai/gpt-5.4", "condition": "single_shot", "n": 10, "kbv_n": np.nan},
    ])
    assert pm.kbv_denominator_sentence(cells).endswith("equals $n$ in every cell shown.")
    cells.loc[0, "kbv_n"] = 8.0
    s = pm.kbv_denominator_sentence(cells)
    assert "GPT-5.4 MT-P: 8 of 10" in s
