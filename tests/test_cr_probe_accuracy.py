"""Tests for drift_bench.analysis.camera_ready.probe_accuracy.

Pure functions on synthetic frames and one synthetic data tree in tmp_path.
No real data, no network.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import probe_accuracy as pa

SONNET = "anthropic/claude-sonnet-4-6"
PRO = "gemini/gemini-3.1-pro-preview"
GPT = "openai/gpt-5.4"
MTN, MTP, CKP = "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"
STRAY_TS = dt.datetime(2026, 4, 16, 12, 0, tzinfo=dt.timezone.utc).timestamp()
GOOD_TS = dt.datetime(2026, 4, 17, 12, 0, tzinfo=dt.timezone.utc).timestamp()


def _rows(recs: list[tuple]) -> pd.DataFrame:
    """recs: (run_id, brief_id, model_id, condition, overall_accuracy[, final_probe_token_count])."""
    out = []
    for r in recs:
        row = {"run_id": r[0], "brief_id": r[1], "model_id": r[2], "condition": r[3], "overall_accuracy": r[4],
               "constraint_recall": r[4], "banned_recall": 1.0}
        if len(r) > 5:
            row["final_probe_token_count"] = r[5]
        out.append(row)
    return pd.DataFrame(out)


def _by(table: pd.DataFrame, **where) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for k, v in where.items():
        mask &= table[k] == v
    sel = table[mask]
    assert len(sel) == 1, (where, len(sel))
    return sel.iloc[0]


def test_run_mean_pools_runs_and_model_mean_weights_models_equally() -> None:
    rows = _rows([
        ("a1", "b1", GPT, MTP, 1.0), ("a2", "b2", GPT, MTP, 1.0), ("a3", "b3", GPT, MTP, 1.0),
        ("p1", "b1", PRO, MTP, 0.5),
    ])
    t = pa.summarize(rows, "s", draws=50)
    pooled = _by(t, scope="multi_turn", model_id="all", aggregation="run_mean")
    assert pooled["mean_accuracy"] == pytest.approx(3.5 / 4)
    assert pooled["n"] == 4 and pooled["n_models"] == 2
    model_mean = _by(t, scope="multi_turn", model_id="all", aggregation="model_mean")
    assert model_mean["mean_accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert model_mean["share_at_or_above_cut"] == pytest.approx((1.0 + 0.0) / 2)
    # A share averaged over models has no single numerator.
    assert math.isnan(model_mean["n_at_or_above_cut"]) and math.isnan(model_mean["ci_low"])


def test_recall_cut_is_inclusive() -> None:
    rows = _rows([("a", "b1", GPT, MTP, 0.8), ("b", "b2", GPT, MTP, 0.79), ("c", "b3", GPT, MTP, 1.0)])
    cell = _by(pa.summarize(rows, "s", draws=10), scope=MTP, model_id=GPT, aggregation="run_mean")
    assert cell["n_at_or_above_cut"] == 2
    assert cell["share_at_or_above_cut"] == pytest.approx(2 / 3)


def test_scopes_filter_conditions_and_skip_empty_scopes() -> None:
    rows = _rows([
        ("a", "b1", GPT, MTP, 1.0), ("b", "b1", GPT, MTN, 0.5),
        ("c", "b1", GPT, "multi_turn_pressure_rigor", 0.0),
    ])
    t = pa.summarize(rows, "s", draws=10)
    assert set(t["scope"]) == {"all_rows", "multi_turn", MTP, MTN}  # no checkpointed rows
    assert _by(t, scope="all_rows", model_id="all", aggregation="run_mean")["n"] == 3
    multi = _by(t, scope="multi_turn", model_id="all", aggregation="run_mean")
    assert multi["n"] == 2 and multi["mean_accuracy"] == pytest.approx(0.75)
    assert _by(t, scope=MTP, model_id="all", aggregation="run_mean")["mean_accuracy"] == 1.0


def test_truncated_probe_count_is_gemini_pro_at_or_above_the_cap() -> None:
    rows = _rows([
        ("p1", "b1", PRO, MTP, 0.5, 1020), ("p2", "b2", PRO, MTP, 1.0, 1019),
        ("g1", "b1", GPT, MTP, 1.0, 1020),
    ])
    t = pa.summarize(rows, "s", draws=10)
    assert _by(t, scope=MTP, model_id="all", aggregation="run_mean")["n_final_probe_truncated"] == 1
    assert _by(t, scope=MTP, model_id=GPT, aggregation="run_mean")["n_final_probe_truncated"] == 0


def test_probe_rows_inner_join_keeps_the_sample_labels() -> None:
    sample = pd.DataFrame({"run_id": ["a", "b"], "brief_id": ["b1", "b2"], "model_id": [GPT, GPT], "condition": [MTP, MTN]})
    probes = pd.DataFrame({"run_id": ["a", "c"], "brief_id": ["x", "x"], "model_id": ["other", "other"], "condition": [CKP, CKP],
                           "turn_index": [5, 5], "overall_accuracy": [0.9, 1.0], "constraint_recall": [1.0, 1.0], "banned_recall": [1.0, 1.0]})
    out = pa.probe_rows(sample, probes)
    assert out["run_id"].tolist() == ["a"]
    assert out.loc[0, "model_id"] == GPT and out.loc[0, "condition"] == MTP and out.loc[0, "brief_id"] == "b1"


def test_cluster_bootstrap_degenerate_and_deterministic() -> None:
    v = pd.Series([1.0, 1.0, 1.0])
    c = pd.Series(["b1", "b2", "b3"])
    assert pa.cluster_bootstrap_mean(v, c, draws=20) == (1.0, 1.0)
    lo, hi = pa.cluster_bootstrap_mean(pd.Series([0.0, 1.0, 0.5, 1.0]), pd.Series(["b1", "b2", "b3", "b4"]), draws=200)
    assert (lo, hi) == pa.cluster_bootstrap_mean(pd.Series([0.0, 1.0, 0.5, 1.0]), pd.Series(["b1", "b2", "b3", "b4"]), draws=200)
    assert 0.0 <= lo <= 0.625 <= hi <= 1.0
    assert all(math.isnan(x) for x in pa.cluster_bootstrap_mean(pd.Series([float("nan")]), pd.Series(["b1"])))


# ---------------------------------------------------------------- synthetic tree


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


def _build_tree(root: Path) -> None:
    """Six parquet runs, one stray Sonnet run, one capped Gemini Pro probe, one probe row outside the parquet."""
    runs = [
        # run_id, brief, model, condition, judge ts, final probe tokens, accuracy
        ("g1", "b1", GPT, MTP, GOOD_TS, 300, 1.0),
        ("g2", "b2", GPT, MTN, GOOD_TS, 300, 1.0),
        ("s1", "b1", SONNET, MTP, GOOD_TS, 300, 1.0),
        ("s0", "b1", SONNET, MTP, STRAY_TS, 300, 0.6),  # stray: dropped from clean
        ("p1", "b1", PRO, MTP, GOOD_TS, 1020, 0.2),    # capped probe
        ("p2", "b2", PRO, MTP, GOOD_TS, 500, 1.0),
    ]
    pd.DataFrame([{"run_id": r[0], "brief_id": r[1], "model_id": r[2], "condition": r[3], "j_constraint_adherence": 3}
                  for r in runs]).to_parquet(root / "aggregated" / "all_scores.parquet")
    for run_id, brief, model, cond, ts, ptok, _acc in runs:
        _write_json(root / "scores" / f"judge_{run_id}.jsonl", {"run_id": run_id, "model_id": model, "timestamp": ts})
        msgs = [
            {"role": "user", "content": "u", "token_count": 1, "is_probe": False},
            {"role": "assistant", "content": "a", "token_count": 100, "is_probe": False},
            {"role": "user", "content": "probe", "token_count": 1, "is_probe": True},
            {"role": "assistant", "content": "p", "token_count": ptok, "is_probe": True},
        ]
        _write_json(root / "transcripts" / f"{run_id}.jsonl",
                    {"metadata": {"run_id": run_id, "brief_id": brief, "model_id": model, "condition": cond, "repetition": 0},
                     "messages": msgs})
    probes = [{"run_id": r[0], "brief_id": r[1], "model_id": r[2], "condition": r[3], "turn_index": 5,
               "overall_accuracy": r[6], "constraint_recall": r[6], "banned_recall": 1.0} for r in runs]
    probes.append({"run_id": "x1", "brief_id": "b1", "model_id": GPT, "condition": "multi_turn_pressure_rigor", "turn_index": 5,
                   "overall_accuracy": 1.0, "constraint_recall": 1.0, "banned_recall": 1.0})
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probes).to_csv(root / "analysis" / "final_probe_accuracy.csv", index=False)


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_run_on_synthetic_tree_applies_each_sample_and_writes_only_to_out_dir(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "aggregated").mkdir(parents=True)
    _build_tree(root)
    before = _files(root)
    out = tmp_path / "out"
    t = pa.run(root, out)["probe_accuracy_summary"]
    assert _files(root) == before
    assert {p.name for p in out.iterdir()} == {pa.CSV_NAME, pa.CSV_NAME + ".provenance.json"}

    def pooled(sample: str, scope: str = "multi_turn") -> pd.Series:
        return _by(t, sample=sample, scope=scope, model_id="all", aggregation="run_mean")

    # The probe file keeps the rigor run in all_rows only.
    assert pooled("probe_file", "all_rows")["n"] == 7 and pooled("probe_file")["n"] == 6
    # as_published: the six parquet runs.
    ap = pooled("as_published")
    assert ap["n"] == 6 and ap["mean_accuracy"] == pytest.approx((1 + 1 + 1 + 0.6 + 0.2 + 1) / 6)
    assert ap["n_final_probe_truncated"] == 1
    # clean: the stray Sonnet run is gone.
    assert pooled("clean")["n"] == 5
    assert _by(t, sample="clean", scope="multi_turn", model_id=SONNET, aggregation="run_mean")["mean_accuracy"] == 1.0
    # clean_untruncated_probes: the capped Pro run is gone as well.
    cu = pooled("clean_untruncated_probes")
    assert cu["n"] == 4 and cu["mean_accuracy"] == 1.0 and cu["share_at_or_above_cut"] == 1.0
    prov = json.loads((out / (pa.CSV_NAME + ".provenance.json")).read_text())
    assert prov["rows_out"] == len(t)
    assert prov["row_counts"]["probe_rows_in_file"] == 7
    assert prov["row_counts"]["probe_rows_in_file_with_parquet_row"] == 6
    assert prov["row_counts"]["probe_rows_joined_by_sample"] == {
        "probe_file": 7, "as_published": 6, "clean": 5, "clean_untruncated_probes": 4}


def test_item_6_discovers_the_module() -> None:
    from drift_bench.run_analysis import discover_camera_ready_modules

    found = dict(discover_camera_ready_modules())
    assert found.get("probe_accuracy") is pa
    # macros reads this module's CSV, so it must run after it.
    names = list(found)
    assert names.index("probe_accuracy") < names.index("macros")
