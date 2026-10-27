"""Tests for drift_bench.analysis.camera_ready.sensitivity (X7).

Synthetic fixtures under tmp_path. No network, no real data, no API keys.
The tests cover the KBV rule, the Wilson CI, the variant labelling, the
stray rule on a follow-up tree, the probe scorer path, and the two checks
(April CSV of record, figure provenance).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

from drift_bench.analysis.camera_ready import sensitivity as s

FLASH, GPT, SONNET = s.FLASH_LITE, s.GPT, s.SONNET
MTN, MTP, CKP, RIGOR = s.MTN, s.MTP, s.CKP, s.MTP_RIGOR
OPUS = "anthropic/claude-opus-4-6"

BRIEF = {
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
BAD_PROBE = "I do not remember the brief."
APRIL_JUDGE_TS = 1776384000.0 + 3600  # 2026-04-17 01:00 UTC
STRAY_JUDGE_TS = 1776297600.0 + 3600  # 2026-04-16 01:00 UTC


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _transcript(run_id: str, brief_id: str, model_id: str, condition: str, probes: list[str], probe_tokens: int = 50) -> dict:
    """Two-turn conversation; ``probes`` gives the probe text after each assistant turn."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "brief"}]
    for i, probe in enumerate(probes, start=1):
        msgs.append({"role": "assistant", "content": "answer " * 5, "token_count": 300})
        msgs.append({"role": "user", "content": "probe?", "is_probe": True})
        msgs.append({"role": "assistant", "content": probe, "is_probe": True, "token_count": probe_tokens})
        if i < len(probes):
            msgs.append({"role": "user", "content": "Continue."})
    if not probes:
        msgs.append({"role": "assistant", "content": "answer", "token_count": 300})
    return {"metadata": {"run_id": run_id, "brief_id": brief_id, "model_id": model_id, "condition": condition, "repetition": 0,
                         "error": None}, "messages": msgs}


def _judge(run_id: str, model_id: str, ts: float) -> dict:
    return {"run_id": run_id, "model_id": model_id, "judge_model": GPT if model_id == SONNET else OPUS,
            "constraint_adherence": 3, "timestamp": ts}


def _score_row(run_id: str, brief_id: str, model_id: str, condition: str, adherence: int) -> dict:
    return {"run_id": run_id, "brief_id": brief_id, "model_id": model_id, "condition": condition,
            "judge_model": GPT if model_id == SONNET else OPUS, "j_constraint_adherence": adherence}


class Tree:
    """A tiny data root: briefs, main run, followup_a, followup_b."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "data"
        self.briefs = tmp_path / "briefs"
        self.out = tmp_path / "out"
        self.briefs.mkdir()
        for bid in ("b1", "b2"):
            (self.briefs / f"{bid}.yaml").write_text(yaml.safe_dump({"id": bid, **BRIEF}))
        self.counter = 0
        self.rows: dict[str, list[dict]] = {}

    def _id(self) -> str:
        self.counter += 1
        return f"r{self.counter:04d}"

    def add_run(self, sub: str, brief_id: str, model_id: str, condition: str, adherence: int, probes: list[str],
                judge_ts: float = APRIL_JUDGE_TS, probe_tokens: int = 50, run_id: str | None = None) -> str:
        run_id = run_id or self._id()
        d = self.root / sub if sub else self.root
        (d / "transcripts").mkdir(parents=True, exist_ok=True)
        (d / "scores").mkdir(parents=True, exist_ok=True)
        (d / "transcripts" / f"{run_id}_{brief_id}_{condition}_{model_id.replace('/', '_')}.jsonl").write_text(
            json.dumps(_transcript(run_id, brief_id, model_id, condition, probes, probe_tokens)) + "\n")
        (d / "scores" / f"judge_{run_id}.jsonl").write_text(json.dumps(_judge(run_id, model_id, judge_ts)) + "\n")
        self.rows.setdefault(sub, []).append(_score_row(run_id, brief_id, model_id, condition, adherence))
        return run_id

    def write_parquets(self) -> None:
        for sub, rows in self.rows.items():
            d = self.root / sub if sub else self.root
            (d / "aggregated").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(d / "aggregated" / "all_scores.parquet", index=False)

    def write_run_summary(self, sub: str, total: int, failed: int, briefs: int, conditions: list[str], models: list[str],
                          repetitions: int) -> None:
        d = self.root / sub if sub else self.root
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_summary.json").write_text(json.dumps({
            "total_runs": total, "failed_runs": failed, "judge_scores": total, "auditor_scores": total,
            "briefs": briefs, "conditions": conditions, "models": models, "repetitions": repetitions}))

    def write_april_record(self, final_probe_rows: list[dict], published_rows: list[dict]) -> None:
        (self.root / "analysis").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(final_probe_rows, columns=["run_id", "brief_id", "model_id", "condition", "turn_index", "overall_accuracy",
                                                "constraint_recall", "banned_recall"]).to_csv(
            self.root / "analysis" / "final_probe_accuracy.csv", index=False)
        self.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(published_rows).to_csv(self.out / "kbv_published_arm.csv", index=False)


def _pub_row(sample: str, model_id: str, condition: str, n_scored: int, n_probe: int, knows: int, violates: int, kbv: int) -> dict:
    lo, hi = s.wilson_ci(kbv, n_probe) if n_probe else (math.nan, math.nan)
    return {"sample": sample, "model_id": model_id, "condition": condition, "n_scored": n_scored, "n_with_final_probe": n_probe,
            "knows": knows, "knows_imputed": 0, "violates": violates, "kbv_count": kbv,
            "kbv_rate": (kbv / n_probe) if n_probe else math.nan, "ci_low": lo, "ci_high": hi}


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    """April: Flash-Lite neutral/pressure/checkpointed, GPT pressure, Sonnet pressure (one stray).
    Follow-up A: Flash-Lite, three conditions. Follow-up B: three models, rigor pressure, plus one stray Sonnet run."""
    t = Tree(tmp_path)
    fpa = []
    pub = []
    # April cells: two runs each; KBV counts chosen per cell.
    april_cells = {
        (FLASH, MTN): [(3, GOOD_PROBE), (4, GOOD_PROBE)],      # 1 KBV of 2
        (FLASH, MTP): [(2, GOOD_PROBE), (3, GOOD_PROBE)],      # 2 of 2
        (FLASH, CKP): [(4, GOOD_PROBE), (4, BAD_PROBE)],       # 0 of 2
        (GPT, MTP): [(4, GOOD_PROBE), (3, BAD_PROBE)],         # 0 of 2 (knows 1, violates 1, never both)
        (SONNET, MTP): [(3, GOOD_PROBE), (3, GOOD_PROBE)],     # 2 of 2
    }
    for (model, cond), runs in april_cells.items():
        knows = violates = kbv = 0
        for i, (adh, probe) in enumerate(runs):
            rid = t.add_run("", f"b{i + 1}", model, cond, adh, [GOOD_PROBE, probe])
            acc = 1.0 if probe == GOOD_PROBE else 0.0
            fpa.append({"run_id": rid, "brief_id": f"b{i + 1}", "model_id": model, "condition": cond, "turn_index": 2,
                        "overall_accuracy": acc, "constraint_recall": acc, "banned_recall": acc})
            knows += acc >= 0.8
            violates += adh < 4
            kbv += (acc >= 0.8) and adh < 4
        n_scored = len(runs) + (1 if model == SONNET else 0)
        pub.append(_pub_row("clean", model, cond, len(runs), len(runs), knows, violates, kbv))
        pub.append(_pub_row("as_published", model, cond, n_scored, n_scored, knows + (model == SONNET), violates + (model == SONNET), kbv + (model == SONNET)))
    # One stray Sonnet pressure run in the main tree (judged 2026-04-16).
    stray = t.add_run("", "b1", SONNET, MTP, 2, [GOOD_PROBE, GOOD_PROBE], judge_ts=STRAY_JUDGE_TS)
    fpa.append({"run_id": stray, "brief_id": "b1", "model_id": SONNET, "condition": MTP, "turn_index": 2, "overall_accuracy": 1.0,
                "constraint_recall": 1.0, "banned_recall": 1.0})
    t.write_april_record(fpa, pub)
    # Follow-up A: Flash-Lite at 0.7; two runs per condition.
    t.add_run("followup_a", "b1", FLASH, MTN, 4, [GOOD_PROBE, GOOD_PROBE])
    t.add_run("followup_a", "b2", FLASH, MTN, 3, [GOOD_PROBE, BAD_PROBE])           # violates, does not know: not KBV
    t.add_run("followup_a", "b1", FLASH, MTP, 3, [BAD_PROBE, GOOD_PROBE])           # KBV (final probe counts)
    t.add_run("followup_a", "b2", FLASH, MTP, 2, [GOOD_PROBE, GOOD_PROBE])           # KBV
    t.add_run("followup_a", "b1", FLASH, CKP, 4, [GOOD_PROBE, GOOD_PROBE])           # knows, adheres
    t.add_run("followup_a", "b2", FLASH, CKP, 3, [GOOD_PROBE, GOOD_PROBE], probe_tokens=1020)  # KBV, truncated probe
    t.add_run("followup_a", "b1", FLASH, "single_shot", 3, [])                        # out of scope
    # Follow-up B: rigor pressure; a stray Sonnet row that the rule must drop.
    t.add_run("followup_b", "b1", GPT, RIGOR, 3, [GOOD_PROBE, GOOD_PROBE])           # KBV
    t.add_run("followup_b", "b2", GPT, RIGOR, 4, [GOOD_PROBE, GOOD_PROBE])
    t.add_run("followup_b", "b1", SONNET, RIGOR, 3, [GOOD_PROBE, GOOD_PROBE])        # KBV
    t.add_run("followup_b", "b2", SONNET, RIGOR, 3, [GOOD_PROBE, GOOD_PROBE], judge_ts=STRAY_JUDGE_TS)  # stray
    t.add_run("followup_b", "b1", FLASH, RIGOR, 3, [GOOD_PROBE, BAD_PROBE])          # violates, forgot
    t.add_run("followup_b", "b2", FLASH, RIGOR, 3, [])                                # no probe: out of the denominator
    t.add_run("followup_b", "b1", GPT, "single_shot", 3, [])                          # out of scope
    t.write_parquets()
    # Run summaries: A plans 2 briefs x 4 conditions x 1 model x 1 rep = 8 (7 done, 1 failed);
    # B plans 2 x 2 x 3 x 1 = 12 (7 done, 5 failed).
    t.write_run_summary("followup_a", 7, 1, 2, ["single_shot", MTN, MTP, CKP], [FLASH], 1)
    t.write_run_summary("followup_b", 7, 5, 2, ["single_shot", RIGOR], [GPT, SONNET, FLASH], 1)
    return t


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_kbv_rule_boundaries() -> None:
    scores = pd.DataFrame({
        "run_id": ["a", "b", "c", "d", "e", "f"],
        "j_constraint_adherence": [3, 3, 4, 3, 2, 3],
    })
    probes = pd.DataFrame({
        "run_id": ["a", "b", "c", "d", "e"],
        "final_probe_accuracy": [0.8, 0.79, 1.0, 1.0, 0.5],
        "final_probe_turn": [5] * 5,
        "final_probe_token_count": [50, 50, 50, 1020, 1019],
    })
    f = s.kbv_flags(scores, probes).set_index("run_id")
    assert f.loc["a", "kbv"] is True or bool(f.loc["a", "kbv"])          # 0.8 counts as knows
    assert not bool(f.loc["b", "kbv"])                                   # 0.79 does not
    assert not bool(f.loc["c", "kbv"]) and not bool(f.loc["c", "violates"])  # adherence 4 adheres
    assert bool(f.loc["d", "kbv"]) and bool(f.loc["d", "probe_truncated"])   # truncated probes count, never waived
    assert not bool(f.loc["e", "probe_truncated"])                       # 1019 is below the cap
    assert not f.loc["f", "has_final_probe"] and pd.isna(f.loc["f", "kbv"]) and pd.isna(f.loc["f", "knows"])
    assert bool(f.loc["f", "violates"])                                  # violates is defined without a probe


def test_wilson_ci_matches_published_values() -> None:
    lo, hi = s.wilson_ci(63, 76)
    assert (round(lo, 4), round(hi, 4)) == (0.7290, 0.8972)
    lo, hi = s.wilson_ci(4, 22)
    assert (round(lo, 4), round(hi, 4)) == (0.0731, 0.3852)
    lo, hi = s.wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 0.31
    assert all(math.isnan(v) for v in s.wilson_ci(0, 0))


def test_cell_row_counts_from_flags() -> None:
    scores = pd.DataFrame({
        "run_id": ["a", "b", "c", "d"], "brief_id": ["x", "y", "x", "y"], "judge_model": [OPUS] * 4,
        "j_constraint_adherence": [3, 3, 4, 3],
    })
    probes = pd.DataFrame({"run_id": ["a", "b", "c"], "final_probe_accuracy": [1.0, 0.0, 1.0], "final_probe_turn": [2] * 3,
                           "final_probe_token_count": [50, 1500, 50]})
    row = s.cell_row(s.kbv_flags(scores, probes), "temperature", FLASH, MTP, "followup", "temperature 0.7", "followup_a", 2)
    assert row["n_runs"] == 4 and row["n_with_final_probe"] == 3 and row["n_briefs"] == 2
    assert row["knows"] == 2 and row["violates"] == 3 and row["kbv_count"] == 1
    assert row["kbv_rate"] == pytest.approx(1 / 3)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx(s.wilson_ci(1, 3))
    assert row["n_truncated_probes"] == 1 and row["stray_rows_dropped"] == 2
    assert row["judge_arm"] == "published" and row["judge_model"] == OPUS
    assert row["variant"] == "followup" and row["setting"] == "temperature 0.7" and row["source"] == "followup_a"
    assert list(row) == s.COLUMNS


def test_final_probes_scores_the_last_probe_with_the_repo_scorer(tmp_path: Path) -> None:
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    (briefs_dir / "b1.yaml").write_text(yaml.safe_dump({"id": "b1", **BRIEF}))
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    for rid, probes, tok in (("p1", [GOOD_PROBE, BAD_PROBE], 50), ("p2", [BAD_PROBE, GOOD_PROBE], 1020), ("p3", [], 50), ("p4", [GOOD_PROBE], 50)):
        (tdir / f"{rid}_b1_{MTP}_m.jsonl").write_text(json.dumps(_transcript(rid, "b1", FLASH, MTP, probes, tok)) + "\n")
    (tdir / "broken.jsonl").write_text("{not json\n")
    df, info = s.final_probes(tdir, s.load_briefs(briefs_dir), {"p1", "p2", "p3"})
    got = df.set_index("run_id")
    assert got.loc["p1", "final_probe_accuracy"] == 0.0 and got.loc["p1", "final_probe_turn"] == 2
    assert got.loc["p2", "final_probe_accuracy"] == 1.0 and got.loc["p2", "final_probe_token_count"] == 1020
    assert "p3" not in got.index and "p4" not in got.index
    assert info["transcripts_matched"] == 3 and info["runs_with_final_probe"] == 2 and info["transcripts_unreadable"] == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_writes_cells_with_variant_labels(tree: Tree) -> None:
    out = s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=None)["sensitivity_cells"]
    assert list(out.columns) == s.COLUMNS
    assert len(out) == 12
    # Row order: each April cell is followed by its follow-up cell.
    assert out["variant"].tolist() == ["april", "followup"] * 6
    assert out["section"].tolist() == ["temperature"] * 6 + ["pressure_type"] * 6
    assert out["condition"].tolist()[:6] == [MTN, MTN, MTP, MTP, CKP, CKP]
    assert out["condition"].tolist()[6:] == [MTP, RIGOR] * 3
    assert out["model_id"].tolist()[6:] == [GPT, GPT, SONNET, SONNET, FLASH, FLASH]
    assert set(out["judge_arm"]) == {"published"}
    assert out.loc[out["variant"] == "april", "setting"].tolist() == ["temperature 1.0"] * 3 + ["novelty pressure"] * 3
    assert out.loc[out["variant"] == "followup", "setting"].tolist() == ["temperature 0.7"] * 3 + ["rigor pressure"] * 3
    assert out.loc[out["variant"] == "followup", "source"].tolist() == ["followup_a"] * 3 + ["followup_b"] * 3

    cells = out.set_index(["section", "model_id", "condition", "variant"])
    fa = cells.loc[("temperature", FLASH, MTN, "followup")]
    assert (fa["n_runs"], fa["n_with_final_probe"], fa["knows"], fa["violates"], fa["kbv_count"]) == (2, 2, 1, 1, 0)
    fa = cells.loc[("temperature", FLASH, MTP, "followup")]
    assert (fa["kbv_count"], fa["n_with_final_probe"], fa["kbv_rate"]) == (2, 2, 1.0)
    fa = cells.loc[("temperature", FLASH, CKP, "followup")]
    assert (fa["kbv_count"], fa["n_truncated_probes"]) == (1, 1)
    assert (fa["ci_low"], fa["ci_high"]) == pytest.approx(s.wilson_ci(1, 2))
    # Follow-up B: the stray Sonnet run is dropped and counted; the probe-less Flash-Lite run leaves the denominator.
    fb = cells.loc[("pressure_type", SONNET, RIGOR, "followup")]
    assert (fb["n_runs"], fb["kbv_count"], fb["stray_rows_dropped"]) == (1, 1, 1)
    fb = cells.loc[("pressure_type", FLASH, RIGOR, "followup")]
    assert (fb["n_runs"], fb["n_with_final_probe"], fb["violates"], fb["kbv_count"]) == (2, 1, 2, 0)
    fb = cells.loc[("pressure_type", GPT, RIGOR, "followup")]
    assert (fb["kbv_count"], fb["n_with_final_probe"], fb["judge_model"]) == (1, 2, OPUS)
    # April rows carry the CSV of record; Sonnet pressure records its 1 stray row.
    ap = cells.loc[("pressure_type", SONNET, MTP, "april")]
    assert (ap["n_runs"], ap["kbv_count"], ap["stray_rows_dropped"], ap["judge_model"]) == (2, 2, 1, GPT)
    ap = cells.loc[("temperature", FLASH, MTN, "april")]
    assert (ap["kbv_count"], ap["n_with_final_probe"], ap["stray_rows_dropped"]) == (1, 2, 0)
    assert ap["kbv_rate"] == pytest.approx(0.5)
    # The two April Flash-Lite pressure rows (one per section) are the same cell.
    a1 = cells.loc[("temperature", FLASH, MTP, "april")]
    a2 = cells.loc[("pressure_type", FLASH, MTP, "april")]
    assert a1["kbv_count"] == a2["kbv_count"] == 2 and a1["ci_low"] == a2["ci_low"]

    csv = pd.read_csv(tree.out / "sensitivity_cells.csv")
    assert len(csv) == 12
    prov = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())
    assert prov["rows_out"] == 12 and "date" in prov and prov["kbv"] == s.KBV_RULE
    assert prov["sections"]["temperature"]["april"]["all_checks_agree"] is True
    assert prov["sections"]["pressure_type"]["april"]["all_checks_agree"] is True
    assert prov["sections"]["pressure_type"]["april"]["stray_rows_dropped"] == 1
    fb_filters = {f["name"]: f for f in prov["sections"]["pressure_type"]["followup"]["filters"]}
    # 7 follow-up B rows in (6 rigor + 1 single-shot), 6 after the stray Sonnet run is dropped.
    assert fb_filters["drop_stray_sonnet_runs"]["rows_out"] == 6 and len(fb_filters["drop_stray_sonnet_runs"]["stray_local_ids"]) == 1
    assert fb_filters["cells"]["rows_out"] == 5 and fb_filters["final_probe_present"]["rows_out"] == 4
    assert prov["figure_provenance_check"] == {"checked": False}


def test_april_check_warns_when_csv_of_record_disagrees(tree: Tree) -> None:
    pub = pd.read_csv(tree.out / "kbv_published_arm.csv")
    sel = (pub["sample"] == "clean") & (pub["model_id"] == GPT) & (pub["condition"] == MTP)
    pub.loc[sel, "kbv_count"] = 2  # the recomputation gives 0
    pub.to_csv(tree.out / "kbv_published_arm.csv", index=False)
    with pytest.warns(UserWarning, match="differ"):
        out = s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=None)["sensitivity_cells"]
    row = out[(out["model_id"] == GPT) & (out["variant"] == "april")].iloc[0]
    assert row["kbv_count"] == 2  # the CSV of record is reported, never replaced
    prov = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())
    checks = {c["model_id"]: c for c in prov["sections"]["pressure_type"]["april"]["checks"]}
    assert checks[GPT]["agree"] is False and checks[GPT]["recomputed"]["kbv_count"] == 0
    assert prov["sections"]["pressure_type"]["april"]["all_checks_agree"] is False


def test_figure_provenance_check_with_renamed_series(tree: Tree, tmp_path: Path) -> None:
    # The figure's series were renamed on 2026-10-27: no label says "April" any more.
    def drawn(panel, series, model, cond, n, k):
        return {"panel": panel, "series": series, "model_id": model, "condition": cond, "n": n, "kbv_count": k}

    main_t, fu_t = "Main benchmark, temperature 1.0", "Temperature follow-up, temperature 0.7"
    main_p, fu_p = "Novelty pressure (main benchmark)", "Rigor pressure (rigor follow-up)"
    numbers = [
        drawn("a_temperature", main_t, FLASH, MTN, 2, 1),
        drawn("a_temperature", fu_t, FLASH, MTN, 2, 0),
        drawn("a_temperature", main_t, FLASH, MTP, 2, 2),
        drawn("a_temperature", fu_t, FLASH, MTP, 2, 2),
        drawn("a_temperature", main_t, FLASH, CKP, 2, 0),
        drawn("a_temperature", fu_t, FLASH, CKP, 2, 1),
        drawn("b_pressure_type", main_p, GPT, MTP, 2, 0),
        drawn("b_pressure_type", fu_p, GPT, RIGOR, 2, 1),
        drawn("b_pressure_type", main_p, SONNET, MTP, 2, 2),
        drawn("b_pressure_type", fu_p, SONNET, RIGOR, 1, 1),
        drawn("b_pressure_type", main_p, FLASH, MTP, 2, 2),
        drawn("b_pressure_type", fu_p, FLASH, RIGOR, 1, 0),
    ]
    fig = tmp_path / "fig_sensitivity.provenance.json"
    fig.write_text(json.dumps({"numbers_drawn": numbers}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=fig)
    check = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())["figure_provenance_check"]
    assert check["agree"] is True and check["cells_compared"] == 12 and check["mismatches"] == []

    # A changed main-benchmark count is caught on the April cell, not on the follow-up cell.
    numbers[0]["kbv_count"] = 2
    fig.write_text(json.dumps({"numbers_drawn": numbers}))
    with pytest.warns(UserWarning, match="differ from"):
        s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=fig)
    check = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())["figure_provenance_check"]
    assert check["agree"] is False and check["cells_compared"] == 12
    assert [tuple(mm["cell"]) for mm in check["mismatches"]] == [("temperature", FLASH, MTN, "april")]


def test_figure_provenance_check(tree: Tree, tmp_path: Path) -> None:
    def drawn(panel, series, model, cond, n, k):
        return {"panel": panel, "series": series, "model_id": model, "condition": cond, "n": n, "kbv_count": k}

    numbers = [
        drawn("a_temperature", "April, temperature 1.0", FLASH, MTN, 2, 1),
        drawn("a_temperature", "Follow-up A, temperature 0.7", FLASH, MTN, 2, 0),
        drawn("a_temperature", "April, temperature 1.0", FLASH, MTP, 2, 2),
        drawn("a_temperature", "Follow-up A, temperature 0.7", FLASH, MTP, 2, 2),
        drawn("a_temperature", "April, temperature 1.0", FLASH, CKP, 2, 0),
        drawn("a_temperature", "Follow-up A, temperature 0.7", FLASH, CKP, 2, 1),
        drawn("b_pressure_type", "Novelty pressure (April)", GPT, MTP, 2, 0),
        drawn("b_pressure_type", "Rigor pressure (follow-up B)", GPT, RIGOR, 2, 1),
        drawn("b_pressure_type", "Novelty pressure (April)", SONNET, MTP, 2, 2),
        drawn("b_pressure_type", "Rigor pressure (follow-up B)", SONNET, RIGOR, 1, 1),
        drawn("b_pressure_type", "Novelty pressure (April)", FLASH, MTP, 2, 2),
        drawn("b_pressure_type", "Rigor pressure (follow-up B)", FLASH, RIGOR, 1, 0),
    ]
    fig = tmp_path / "fig_sensitivity.provenance.json"
    fig.write_text(json.dumps({"numbers_drawn": numbers}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=fig)
    prov = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())
    assert prov["figure_provenance_check"]["agree"] is True and prov["figure_provenance_check"]["cells_compared"] == 12

    numbers[7]["kbv_count"] = 2
    fig.write_text(json.dumps({"numbers_drawn": numbers[:-1]}))
    with pytest.warns(UserWarning, match="differ from"):
        s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=fig)
    check = json.loads((tree.out / "sensitivity_cells.csv.provenance.json").read_text())["figure_provenance_check"]
    assert check["agree"] is False and check["cells_compared"] == 11
    reasons = {tuple(mm["cell"]): mm for mm in check["mismatches"]}
    assert reasons[("pressure_type", GPT, RIGOR, "followup")]["figure"]["kbv_count"] == 2
    assert reasons[("pressure_type", FLASH, RIGOR, "followup")]["reason"] == "not drawn"

    missing = s.check_against_figure(pd.DataFrame(columns=s.COLUMNS), tmp_path / "nope.json")
    assert missing["checked"] is False


# ---------------------------------------------------------------------------
# Follow-up run counts (X11)
# ---------------------------------------------------------------------------


def test_planned_runs_multiplies_the_design() -> None:
    summary = {"briefs": 38, "conditions": ["single_shot", RIGOR], "models": [GPT, SONNET, FLASH], "repetitions": 1}
    assert s.planned_runs(summary) == 228
    assert s.planned_runs({"briefs": 38, "conditions": ["a", "b", "c", "d"], "models": [FLASH], "repetitions": 2}) == 304


def test_run_writes_sensitivity_runs_with_provenance(tree: Tree) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = s.run(tree.root, tree.out, briefs_dir=tree.briefs, figure_provenance=None)
    runs = out["sensitivity_runs"]
    assert list(runs.columns) == s.RUNS_COLUMNS
    assert runs["followup"].tolist() == ["followup_a", "followup_b"]
    assert runs["section"].tolist() == ["temperature", "pressure_type"]
    assert runs["setting"].tolist() == ["temperature 0.7", "rigor pressure"]
    a, b = runs.iloc[0], runs.iloc[1]
    assert (a["planned"], a["completed"], a["failed"], a["unaccounted"]) == (8, 7, 1, 0)
    assert (b["planned"], b["completed"], b["failed"], b["unaccounted"]) == (12, 7, 5, 0)
    assert (b["briefs"], b["n_conditions"], b["n_models"], b["repetitions"]) == (2, 2, 3, 1)
    assert b["models"] == f"{GPT};{SONNET};{FLASH}" and b["conditions"] == f"single_shot;{RIGOR}"
    # 7 transcripts on disk per follow-up; the parquets hold every added run (B keeps the stray row).
    assert (a["transcripts_on_disk"], a["parquet_rows"]) == (7, 7)
    assert (b["transcripts_on_disk"], b["parquet_rows"]) == (7, 7)
    csv = pd.read_csv(tree.out / s.RUNS_CSV)
    assert csv["failed"].tolist() == [1, 5]
    prov = json.loads((tree.out / f"{s.RUNS_CSV}.provenance.json").read_text())
    assert prov["rows_out"] == 2 and prov["filters"] == s.RUNS_RULES and "date" in prov
    assert prov["followups"]["followup_b"]["completed_plus_failed_equals_planned"] is True
    cells_b = {(c["model_id"], c["condition"]): c for c in prov["followups"]["followup_b"]["parquet_rows_by_cell"]}
    assert cells_b[(SONNET, RIGOR)] == {"model_id": SONNET, "condition": RIGOR, "parquet_rows": 2, "planned": 2}
    assert prov["other_rigor_pass"] is None  # the main tree has no run_summary.json
    # The KBV cells are still written and unchanged in shape.
    assert len(out["sensitivity_cells"]) == 12 and (tree.out / "sensitivity_cells.csv").exists()


def test_run_count_row_warns_on_gaps_and_needs_the_summary(tree: Tree) -> None:
    fb = tree.root / "followup_b"
    tree.write_run_summary("followup_b", 7, 4, 2, ["single_shot", RIGOR], [GPT, SONNET, FLASH], 1)
    with pytest.warns(UserWarning, match="planned 12 != completed 7 \\+ failed 4"):
        row, info = s.run_count_row(fb, s.SECTION_PRESSURE_TYPE)
    assert row["unaccounted"] == 1 and info["completed_plus_failed_equals_planned"] is False
    tree.write_run_summary("followup_b", 8, 4, 2, ["single_shot", RIGOR], [GPT, SONNET, FLASH], 1)
    with pytest.warns(UserWarning, match="7 transcripts on disk"):
        row, info = s.run_count_row(fb, s.SECTION_PRESSURE_TYPE)
    assert row["unaccounted"] == 0 and info["transcripts_equal_completed"] is False
    (fb / "run_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_summary.json"):
        s.run_count_row(fb, s.SECTION_PRESSURE_TYPE)


def _set_start_times(paths: list[Path], starts: list[float]) -> None:
    for path, start in zip(paths, starts):
        obj = json.loads(path.read_text())
        obj["metadata"]["start_time"] = start
        path.write_text(json.dumps(obj) + "\n")


def test_main_rigor_pass_is_reported_for_context(tree: Tree) -> None:
    assert s.main_rigor_pass(tree.root) is None
    # A pass over 3 briefs x {single_shot, rigor} x 3 models x 1 rep = 18 cells. The 9 single-shot
    # cells are already on disk (reused). It writes 4 rigor transcripts: 3 cells that followup_b
    # also has (followup_b has all 6 rigor cells of b1, b2) and 1 cell (b3) that followup_b lacks.
    for bid in ("b1", "b2", "b3"):
        for model in (GPT, SONNET, FLASH):
            tree.add_run("", bid, model, "single_shot", 3, [])
    rigor = [tree.add_run("", bid, model, RIGOR, 3, []) for bid, model in
             (("b1", GPT), ("b2", GPT), ("b1", SONNET), ("b3", FLASH))]
    for rid in rigor[1:]:  # only the first rigor run was scored
        (tree.root / "scores" / f"judge_{rid}.jsonl").unlink()
    # The skeptical-prompt judge scored one rigor run that the main judge did not.
    (tree.root / "scores_skeptical").mkdir(exist_ok=True)
    (tree.root / "scores_skeptical" / f"skeptical_{rigor[1]}.jsonl").write_text(
        json.dumps({"run_id": rigor[1], "condition": RIGOR, "constraint_adherence": 3}) + "\n")
    tree.write_run_summary("", 13, 5, 3, ["single_shot", RIGOR], [GPT, SONNET, FLASH], 1)
    info = s.main_rigor_pass(tree.root)
    assert (info["planned"], info["completed"], info["failed"]) == (18, 13, 5)
    assert info["rigor_transcripts_in_main_transcripts_dir"] == 4
    assert (info["rigor_completed"], info["rigor_failed"], info["reused_existing_transcripts"]) == (4, 5, 9)
    assert info["single_shot_cells_on_disk"] == 9
    assert (info["n_rigor_scored"], info["n_rigor_skeptical_scored"], info["overlap_with_followup_b"]) == (1, 1, 0)
    assert (info["followup_b_rigor_cells"], info["followup_b_rigor_cells_rerun"], info["cells_not_in_followup_b"]) == (6, 3, 1)
    assert info["start_gap_seconds"] is None and info["start_windows_overlap"] is None
    assert info["rules"] == s.MAIN_RIGOR_RULES
    note = info["note"]
    assert "`completed` (13) counts reused transcripts" in note and "9 of the 13 are such reused transcripts" in note
    assert "5 of the 9 planned rigor runs failed" in note and "start times are missing" in note
    assert "re-ran 3 of followup_b's 6 rigor cells plus 1 cells followup_b lacks" in note
    assert ("It has no main-judge scores: 1 of its 4 rigor runs have a scores/judge_<run_id>.jsonl; "
            "1 have a skeptical-prompt judge file (scores_skeptical/skeptical_<run_id>.jsonl).") in note
    assert "never scored" not in note
    # Start times: followup_b begins 73 s after the pass, and the two start windows overlap.
    t0 = 1776460255.0
    main_rigor = sorted((tree.root / "transcripts").glob(f"*_{RIGOR}_*.jsonl"))
    fb_files = sorted((tree.root / "followup_b" / "transcripts").glob("*.jsonl"))
    _set_start_times(main_rigor, [t0 + 100 * i for i in range(len(main_rigor))])
    _set_start_times(fb_files, [t0 + 73 + i for i in range(len(fb_files))])
    info = s.main_rigor_pass(tree.root)
    assert info["start_gap_seconds"] == 73.0 and info["start_windows_overlap"] is True
    assert info["rigor_first_start_utc"] == "2026-04-17T21:10:55+00:00"
    assert "started 73 s apart and their start windows overlap" in info["note"]
    _set_start_times(fb_files, [t0 + 1000 + i for i in range(len(fb_files))])
    assert s.main_rigor_pass(tree.root)["start_windows_overlap"] is False
    # A rigor run id that followup_b also holds counts as overlap.
    shared = fb_files[0].name.split("_")[0]
    tree.add_run("", "b2", SONNET, RIGOR, 3, [], run_id=shared)
    info = s.main_rigor_pass(tree.root)
    assert (info["rigor_completed"], info["overlap_with_followup_b"], info["followup_b_rigor_cells_rerun"]) == (5, 1, 4)
    tree.write_run_summary("", 10, 0, 2, [MTP], [GPT], 1)
    assert s.main_rigor_pass(tree.root) is None  # a summary without the rigor condition is not reported
