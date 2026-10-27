"""Tests for drift_bench.analysis.camera_ready.surface_gap.

Pure functions on synthetic frames and one synthetic data tree in tmp_path.
No real data, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import kbv_crossed as kc
from drift_bench.analysis.camera_ready import surface_gap as sg

SONNET = kc.SONNET
GPT = "openai/gpt-5.4"
LLAMA = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
SS, MTP, MTN = kc.SS, kc.MTP, kc.MTN


def _runs(rows: list[tuple]) -> pd.DataFrame:
    """rows: (run_id, brief, model, condition, opus (fid, adh, cov) or None, gpt (...) or None)."""
    out = []
    for run_id, brief, model, cond, opus, gpt in rows:
        rec = {"run_id": run_id, "brief_id": brief, "model_id": model, "condition": cond,
               "published_judge": kc.published_judge_of(model)}
        for arm, vals in (("opus", opus), ("gpt", gpt)):
            for dim, v in zip(sg.GAP_DIMS, vals or (np.nan, np.nan, np.nan)):
                rec[f"{arm}_{dim}"] = v
        pub = gpt if model == SONNET else opus
        for dim, v in zip(sg.GAP_DIMS, pub or (np.nan, np.nan, np.nan)):
            rec[f"published_{dim}"] = v
        out.append(rec)
    return pd.DataFrame(out)


def test_run_sfg_is_fidelity_minus_mean_of_adherence_and_coverage() -> None:
    df = _runs([("a", "b1", GPT, MTP, (4, 2, 3), (3, 3, 3))])
    assert sg.run_sfg(df, "opus").iloc[0] == pytest.approx(4 - 2.5)
    assert sg.run_sfg(df, "gpt").iloc[0] == pytest.approx(0.0)


def test_table_cells_pool_models_and_equal_the_mean_run_gap() -> None:
    df = _runs([
        ("l1", "b1", LLAMA, MTP, (4, 2, 2), (4, 3, 3)),
        ("l2", "b2", LLAMA, MTP, (3, 2, 3), (3, 2, 2)),
        ("g1", "b1", GPT, MTP, (4, 4, 4), (4, 3, 4)),
        ("s1", "b1", SONNET, MTP, (3, 3, 4), (3, 2, 4)),
        ("x1", "b1", GPT, MTN, (4, 0, 0), (4, 0, 0)),     # neutral: outside the crossed cells
        ("m1", "b2", GPT, SS, (4, 4, 4), None),            # no GPT-arm scores: out of that arm only
    ])
    t = sg.surface_gap_table(df, draws=100)
    cell = t.set_index(["judge_arm", "condition", "model_id"])
    llama = cell.loc[("opus", MTP, LLAMA)]
    assert llama["n"] == 2
    assert llama["sfg"] == pytest.approx(np.mean([4 - 2, 3 - 2.5]))
    assert llama["fidelity_mean"] == pytest.approx(3.5)
    assert llama["sfg_ci_low"] <= llama["sfg"] <= llama["sfg_ci_high"]
    pooled = cell.loc[("opus", MTP, "all")]
    assert pooled["n"] == 4
    assert pooled["sfg"] == pytest.approx(np.mean([2.0, 0.5, 0.0, -0.5]))
    assert MTN not in set(t["condition"])
    assert ("opus", SS, GPT) in cell.index and ("gpt", SS, GPT) not in cell.index
    # The published arm takes GPT-5.4 for Sonnet and Opus for the rest.
    assert cell.loc[("published", MTP, SONNET), "sfg"] == pytest.approx(3 - 3)
    assert cell.loc[("published", MTP, "all"), "judge_model"] == ";".join(sorted({kc.OPUS, kc.GPT}))
    assert cell.loc[("opus", MTP, "all"), "judge_model"] == kc.OPUS


def _write_judge(dir_: Path, prefix: str, run_id: str, model_id: str, condition: str, judge: str,
                 scores: tuple[int, int, int], brief: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    fid, adh, cov = scores
    rec = {"run_id": run_id, "brief_id": brief, "model_id": model_id, "condition": condition, "judge_model": judge,
           "objective_fidelity": fid, "constraint_adherence": adh, "alternative_coverage": cov, "complexity_inflation": 1,
           "timestamp": 1793031106.0, "judge_response_model": judge}
    (dir_ / f"{prefix}_{run_id}.jsonl").write_text(json.dumps(rec) + "\n")


def test_run_rebuilds_the_crossed_table_from_raw_files_and_writes_only_to_out_dir(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "aggregated").mkdir(parents=True)
    (root / "analysis").mkdir()
    pub_rows = [
        # run_id, brief, model, condition, published (fid, adh, cov), crossed (fid, adh, cov) or None
        ("s1", "b1", SONNET, MTP, (4, 2, 4), (4, 3, 4)),
        ("s2", "b2", SONNET, SS, (4, 3, 4), (4, 4, 4)),
        ("l1", "b1", LLAMA, MTP, (4, 2, 2), (3, 2, 2)),
        ("l2", "b2", LLAMA, SS, (4, 3, 3), (4, 3, 3)),
        ("l3", "b2", LLAMA, MTP, (3, 3, 3), None),   # no crossed file: not in the crossed table
        ("g1", "b1", GPT, MTN, (4, 4, 4), None),     # neutral: not in the crossed table
    ]
    recs = []
    for run_id, brief, model, cond, pub, crossed in pub_rows:
        recs.append({"run_id": run_id, "brief_id": brief, "model_id": model, "condition": cond,
                     "judge_model": kc.published_judge_of(model),
                     "j_objective_fidelity": pub[0], "j_constraint_adherence": pub[1], "j_alternative_coverage": pub[2],
                     "j_complexity_inflation": 1})
        _write_judge(root / "scores", "judge", run_id, model, cond, kc.published_judge_of(model), pub, brief)
        if crossed is not None:
            arm = "opus" if model == SONNET else "gpt"
            _write_judge(root / kc.E1_DIR / arm, f"judge_{arm}", run_id, model, cond, kc.crossed_judge_of(model), crossed, brief)
    pd.DataFrame(recs).to_parquet(root / "aggregated" / "all_scores.parquet")
    pd.DataFrame({"run_id": [r[0] for r in pub_rows], "overall_accuracy": 1.0}).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)
    before = {p for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "out"
    t = sg.run(root, out)["surface_gap"]
    assert {p for p in root.rglob("*") if p.is_file()} == before
    assert {p.name for p in out.iterdir()} == {sg.CSV_NAME, sg.CSV_NAME + ".provenance.json"}
    cell = t.set_index(["judge_arm", "condition", "model_id"])
    # Opus arm: Sonnet takes its crossed (Opus) file, Llama its published (Opus) score.
    assert cell.loc[("opus", MTP, SONNET), "sfg"] == pytest.approx(4 - 3.5)
    assert cell.loc[("opus", MTP, LLAMA), "sfg"] == pytest.approx(4 - 2)
    # GPT arm: Sonnet takes its published (GPT) score, Llama its crossed (GPT) file.
    assert cell.loc[("gpt", MTP, SONNET), "sfg"] == pytest.approx(4 - 3)
    assert cell.loc[("gpt", MTP, LLAMA), "sfg"] == pytest.approx(3 - 2)
    assert cell.loc[("opus", MTP, "all"), "n"] == 2   # l3 has no crossed file
    prov = json.loads((out / (sg.CSV_NAME + ".provenance.json")).read_text())
    assert prov["rows_out"] == len(t)
    assert prov["filters"][1]["rows_out"] == 4


def test_item_6_discovers_the_module() -> None:
    from drift_bench.run_analysis import discover_camera_ready_modules

    found = dict(discover_camera_ready_modules())
    assert found.get("surface_gap") is sg
    # macros reads this module's CSV, so it must run after it.
    names = list(found)
    assert names.index("surface_gap") < names.index("macros")
