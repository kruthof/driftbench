"""Tests for drift_bench.analysis.camera_ready.sonnet_truncation.

Pure functions on synthetic frames and one synthetic data tree in tmp_path.
No real data, no network.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import sonnet_truncation as st
from drift_bench.analysis.camera_ready.kbv_table import wilson_ci

SONNET = st.SONNET
GPT = "openai/gpt-5.4"
SS, MTN, MTP, CKP = "single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"
STRAY_TS = dt.datetime(2026, 4, 16, 12, 0, tzinfo=dt.timezone.utc).timestamp()
GOOD_TS = dt.datetime(2026, 4, 17, 12, 0, tzinfo=dt.timezone.utc).timestamp()


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (model_id, condition, any_cap, final_cap); None means no transcript."""
    return pd.DataFrame(rows, columns=["model_id", "condition", "any_assistant_at_cap", "final_assistant_at_cap"])


def _cell(t: pd.DataFrame, rule: str, cond: str) -> pd.Series:
    sel = t[(t["rule"] == rule) & (t["condition"] == cond)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_both_rules_per_condition_and_pooled() -> None:
    df = _frame([
        (SONNET, SS, False, False),
        (SONNET, MTP, True, True),
        (SONNET, MTP, True, False),   # capped earlier, not on the final turn
        (SONNET, MTP, False, False),
        (SONNET, MTN, True, False),
        (SONNET, CKP, None, None),    # no transcript: out of numerator and denominator
        (GPT, MTP, True, True),       # other model: ignored
        (SONNET, "multi_turn_pressure_rigor", True, True),  # not a main condition
    ])
    t = st.truncation_table(df, "s")
    assert set(t["rule"]) == {"any_turn", "final_turn"}
    assert len(t) == 2 * 5
    any_p = _cell(t, "any_turn", MTP)
    assert (any_p["n_truncated"], any_p["n_runs"], any_p["n_untruncated"]) == (2, 3, 1)
    fin_p = _cell(t, "final_turn", MTP)
    assert (fin_p["n_truncated"], fin_p["n_runs"]) == (1, 3)
    pooled_any = _cell(t, "any_turn", st.ALL_MAIN)
    assert (pooled_any["n_truncated"], pooled_any["n_runs"]) == (3, 5)
    assert pooled_any["share_truncated"] == pytest.approx(3 / 5)
    lo, hi = wilson_ci(3, 5)
    assert (pooled_any["ci_low"], pooled_any["ci_high"]) == pytest.approx((lo, hi))
    ck = _cell(t, "any_turn", CKP)
    assert ck["n_runs"] == 0 and pd.isna(ck["share_truncated"])


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


def test_run_on_synthetic_tree_drops_stray_runs_from_clean_and_writes_only_to_out_dir(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "aggregated").mkdir(parents=True)
    runs = [
        # run_id, brief, condition, judge ts, assistant token counts (non-probe, in order)
        ("a", "b1", MTP, GOOD_TS, [500, 4096, 800]),    # any-turn capped, final not
        ("b", "b2", MTP, GOOD_TS, [500, 800, 4096]),    # final capped
        ("c", "b1", MTP, STRAY_TS, [4096, 4096, 4096]),  # stray
        ("d", "b1", SS, GOOD_TS, [4095]),               # one short of the cap
    ]
    pd.DataFrame([{"run_id": r[0], "brief_id": r[1], "model_id": SONNET, "condition": r[2], "j_constraint_adherence": 3}
                  for r in runs]).to_parquet(root / "aggregated" / "all_scores.parquet")
    for run_id, brief, cond, ts, toks in runs:
        _write_json(root / "scores" / f"judge_{run_id}.jsonl", {"run_id": run_id, "model_id": SONNET, "timestamp": ts})
        msgs = []
        for tok in toks:
            msgs += [{"role": "user", "content": "u", "token_count": 1, "is_probe": False},
                     {"role": "assistant", "content": "a", "token_count": tok, "is_probe": False},
                     {"role": "assistant", "content": "probe", "token_count": 4096, "is_probe": True}]  # probes never count
        _write_json(root / "transcripts" / f"{run_id}.jsonl",
                    {"metadata": {"run_id": run_id, "brief_id": brief, "model_id": SONNET, "condition": cond}, "messages": msgs})
    before = {p for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "out"
    t = st.run(root, out)["sonnet_truncation"]
    assert {p for p in root.rglob("*") if p.is_file()} == before
    assert {p.name for p in out.iterdir()} == {"sonnet_truncation.csv", "sonnet_truncation.csv.provenance.json"}
    sel = t.set_index(["sample", "rule", "condition"])
    assert (sel.loc[("as_published", "any_turn", st.ALL_MAIN), ["n_truncated", "n_runs"]] == [3, 4]).all()
    assert (sel.loc[("clean", "any_turn", st.ALL_MAIN), ["n_truncated", "n_runs"]] == [2, 3]).all()
    assert (sel.loc[("clean", "final_turn", st.ALL_MAIN), ["n_truncated", "n_runs"]] == [1, 3]).all()
    assert (sel.loc[("as_published", "final_turn", MTP), ["n_truncated", "n_runs"]] == [2, 3]).all()
    prov = json.loads((out / "sonnet_truncation.csv.provenance.json").read_text())
    assert prov["rows_out"] == len(t)
    assert prov["row_counts"]["sonnet_rows_by_sample"]["clean"]["rows_out"] == 3


def test_item_6_discovers_the_module() -> None:
    from drift_bench.run_analysis import discover_camera_ready_modules

    found = dict(discover_camera_ready_modules())
    assert found.get("sonnet_truncation") is st
    # macros reads this module's CSV, so it must run after it.
    names = list(found)
    assert names.index("sonnet_truncation") < names.index("macros")
