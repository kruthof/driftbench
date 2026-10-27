"""Tests for drift_bench.analysis.camera_ready.e2_uncapped.

Synthetic fixtures under tmp_path. No network, no real data, no .env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from drift_bench.analysis.camera_ready import e2_uncapped as e2

SONNET = e2.SONNET
SS, MTP = e2.SS, e2.MTP

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
APRIL_JUDGE_TS = 1776384000.0 + 3600  # 2026-04-17 01:00 UTC
STRAY_JUDGE_TS = 1776297600.0 + 3600  # 2026-04-16 01:00 UTC
OCT_TS = 1792972800.0 + 3600  # 2026-10-26 01:00 UTC


def _transcript(run_id, brief_id, condition, turn_tokens, max_tokens, repetition=0, finish="stop", error=None):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "brief"}]
    for i, tc in enumerate(turn_tokens, start=1):
        msgs.append({"role": "assistant", "content": "x" * 10, "token_count": tc, "finish_reason": finish, "response_model": "claude-sonnet-4-6"})
        if condition == MTP and i < len(turn_tokens):
            msgs.append({"role": "user", "content": "probe?", "is_probe": True})
            msgs.append({"role": "assistant", "content": GOOD_PROBE, "is_probe": True, "token_count": 50, "finish_reason": "stop"})
            msgs.append({"role": "user", "content": "Continue."})
    return {
        "metadata": {
            "run_id": run_id, "brief_id": brief_id, "model_id": SONNET, "condition": condition, "repetition": repetition,
            "total_input_tokens": 1000, "total_output_tokens": int(sum(turn_tokens)), "total_cost_usd": 0.5,
            "error": error, "temperature": 0.7, "max_tokens": max_tokens, "litellm_version": "1.83.8",
        },
        "messages": msgs,
    }


def _judge(run_id, brief_id, condition, judge, scores, ts, resp=None, cost_estimated=False):
    return {
        "run_id": run_id, "brief_id": brief_id, "model_id": SONNET, "condition": condition,
        "judge_model": judge, "judge_response_model": resp,
        "objective_fidelity": scores[0], "constraint_adherence": scores[1],
        "alternative_coverage": scores[2], "complexity_inflation": scores[3],
        "timestamp": ts, "cost_usd": 0.05, "cost_estimated": cost_estimated,
    }


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec) + "\n")


class Fixture:
    """Two briefs. April: 2 reps per cell plus one stray single-shot run. E2: 1 run per cell."""

    def __init__(self, tmp_path: Path, e2_finish="stop", with_offset_csv=True):
        self.root = tmp_path / "data"
        self.out = tmp_path / "out"
        self.briefs_dir = tmp_path / "briefs"
        self.briefs_dir.mkdir()
        self.briefs = ["b1", "b2"]
        for b in self.briefs:
            (self.briefs_dir / f"{b}.yaml").write_text(yaml.safe_dump({"id": b, **BRIEF}))
        self.e2 = self.root / e2.E2_DIR
        self.april_ids: list[str] = []
        self.e2_ids: list[str] = []
        pq_rows = []
        probe_rows = []
        # April: adherence 3 and 4 alternate so the brief mean is 3.5 under GPT; Opus (E1) one point higher.
        for bi, b in enumerate(self.briefs):
            for cond in (SS, MTP):
                for rep in (0, 1):
                    rid = f"a{bi}{'s' if cond == SS else 'p'}{rep}"
                    self.april_ids.append(rid)
                    tokens = [3000] if cond == SS else [2000, 4096, 4096, 3000, 4096, 4096 if rep == 0 else 3900]
                    _write(self.root / "transcripts" / f"{rid}_{b}_{cond}_anthropic_claude-sonnet-4-6.jsonl", _transcript(rid, b, cond, tokens, 4096, rep))
                    gpt_scores = (4, 3 + rep, 4, 2)
                    _write(self.root / "scores" / f"judge_{rid}.jsonl", _judge(rid, b, cond, e2.GPT, gpt_scores, APRIL_JUDGE_TS))
                    pq_rows.append({"run_id": rid, "brief_id": b, "model_id": SONNET, "condition": cond, "judge_model": e2.GPT,
                                    **{f"j_{d}": s for d, s in zip(e2.DIMS, gpt_scores)}})
                    _write(self.root / e2.E1_OPUS_DIR / f"judge_opus_{rid}.jsonl", _judge(rid, b, cond, e2.OPUS, (4, 4, 4, 1), OCT_TS, "claude-opus-4-6"))
                    if cond == MTP:
                        probe_rows.append({"run_id": rid, "brief_id": b, "model_id": SONNET, "condition": cond, "turn_index": 5, "overall_accuracy": 1.0, "constraint_recall": 1.0, "banned_recall": 1.0})
        # stray: single-shot run of b1, judged on 2026-04-16
        rid = "stray0"
        _write(self.root / "transcripts" / f"{rid}_b1_{SS}_anthropic_claude-sonnet-4-6.jsonl", _transcript(rid, "b1", SS, [3100], 4096, 2))
        _write(self.root / "scores" / f"judge_{rid}.jsonl", _judge(rid, "b1", SS, e2.GPT, (0, 0, 0, 0), STRAY_JUDGE_TS))
        pq_rows.append({"run_id": rid, "brief_id": "b1", "model_id": SONNET, "condition": SS, "judge_model": e2.GPT, **{f"j_{d}": 0 for d in e2.DIMS}})
        _write(self.root / e2.E1_OPUS_DIR / f"judge_opus_{rid}.jsonl", _judge(rid, "b1", SS, e2.OPUS, (0, 0, 0, 0), OCT_TS, "claude-opus-4-6"))
        (self.root / "aggregated").mkdir(parents=True)
        pd.DataFrame(pq_rows).to_parquet(self.root / "aggregated" / "all_scores.parquet", index=False)
        (self.root / "analysis").mkdir(parents=True)
        pd.DataFrame(probe_rows).to_csv(self.root / "analysis" / "final_probe_accuracy.csv", index=False)
        # E2
        for bi, b in enumerate(self.briefs):
            for cond in (SS, MTP):
                rid = f"e{bi}{'s' if cond == SS else 'p'}"
                self.e2_ids.append(rid)
                tokens = [3100] if cond == SS else [2500, 4500, 6000, 3000, 7000, 5000]
                _write(self.e2 / "transcripts" / f"{rid}_{b}_{cond}_anthropic_claude-sonnet-4-6.jsonl",
                       _transcript(rid, b, cond, tokens, 16384, 0, finish=e2_finish))
                _write(self.e2 / "scores_opus" / f"judge_opus_{rid}.jsonl", _judge(rid, b, cond, e2.OPUS, (4, 4, 4, 2), OCT_TS, "claude-opus-4-6"))
                _write(self.e2 / "scores_gpt" / f"judge_gpt_{rid}.jsonl", _judge(rid, b, cond, e2.GPT, (4, 3, 4, 3), OCT_TS, "gpt-5.4-2026-03-05"))
        self.out.mkdir()
        if with_offset_csv:
            _write_comparators(self.out)

    def run(self, out_dir: Path | None = None, comparators_dir: Path | None = None):
        # Never fall back to the default comparators directory (real data).
        return e2.run(
            self.root,
            out_dir if out_dir is not None else self.out,
            self.briefs_dir,
            n_briefs=len(self.briefs),
            comparators_dir=comparators_dir if comparators_dir is not None else self.out,
        )


def _write_comparators(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"block": "offset_by_model", "dimension": dim, "model_id": SONNET, "condition": "ss+mtp", "judge": "opus_minus_gpt",
          "estimate": 0.8, "ci_low": 0.7, "ci_high": 0.9, "n_runs": 152, "n_briefs": 38} for dim in e2.DIMS]
    ).to_csv(d / e2.JUDGE_OFFSET_CSV, index=False)
    pd.DataFrame(
        [{"control_judge": "gpt", "judge_model": e2.GPT, "dimension": "constraint_adherence", "n": 25,
          "mean_april": 2.76, "mean_today": 2.84, "mean_diff_today_minus_april": 0.08, "mad": 0.08, "exact_agreement": 0.92, "within_1": 1.0}]
    ).to_csv(d / e2.JUDGE_DRIFT_CSV, index=False)


def test_cap_rule_asymmetry() -> None:
    t = _transcript("r", "b1", MTP, [4096, 4097, 100], 4096)
    april = e2.summarize_run(t, e2.cap_rule_april)
    uncapped = e2.summarize_run(t, e2.cap_rule_e2)
    assert april["cap_flags"] == [True, False, False]
    assert uncapped["cap_flags"] == [False, True, False]
    assert april["cap_turns"] == 1 and uncapped["cap_turns"] == 1
    assert april["final_cap"] is False and uncapped["final_cap"] is False
    assert april["run_max"] == 4097
    assert april["probe_tokens_max"] == 50  # probes do not enter the turn statistics


def test_brief_paired_delta_averages_april_repetitions() -> None:
    e2f = pd.DataFrame({"brief_id": ["b1", "b2"], "v": [5.0, 2.0]})
    april = pd.DataFrame({"brief_id": ["b1", "b1", "b2", "b2", "b3"], "v": [1.0, 3.0, 2.0, 2.0, 9.0]})
    out = e2.brief_paired_deltas(e2f, april, "v")
    assert out["brief_id"].tolist() == ["b1", "b2"]  # b3 has no E2 run
    assert out["april"].tolist() == [2.0, 2.0]
    assert out["delta"].tolist() == [3.0, 0.0]
    with pytest.raises(ValueError):
        e2.brief_paired_deltas(pd.DataFrame({"brief_id": ["b1", "b1"], "v": [1.0, 2.0]}), april, "v")


def test_wilson_and_newcombe_hand_values() -> None:
    lo, hi = e2.wilson_ci(25, 38)
    assert lo == pytest.approx(0.4993, abs=1e-3)
    assert hi == pytest.approx(0.7879, abs=1e-3)
    lo, hi = e2.wilson_ci(57, 76)
    assert lo == pytest.approx(0.6423, abs=1e-3)
    assert hi == pytest.approx(0.8337, abs=1e-3)
    # Newcombe method 10 from those Wilson bounds: d = 0.6579 - 0.75 = -0.0921
    lo, hi = e2.newcombe_ci(25, 38, 57, 76)
    assert lo == pytest.approx(-0.2714, abs=2e-3)
    assert hi == pytest.approx(0.0767, abs=2e-3)
    assert e2.fisher_p(25, 38, 57, 76) == pytest.approx(0.38, abs=0.02)


def test_run_drops_strays_from_april(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    tables = fx.run()
    cells = tables["e2_uncapped_cells"]
    assert len(cells) == 8  # 2 sources x 2 cells x 2 arms
    april_ss = cells[(cells["source"] == e2.SOURCE_APRIL) & (cells["condition"] == SS)]
    assert april_ss["n_runs"].tolist() == [4, 4]  # 2 briefs x 2 reps; the stray is gone
    assert april_ss["repetitions"].tolist() == [2.0, 2.0]
    prov = json.loads((fx.out / "e2_uncapped_cells.csv.provenance.json").read_text())
    assert prov["filters"]["april_sample"]["rows_dropped"] == 1
    assert prov["filters"]["april_sample"]["n_stray_run_ids"] == 1
    # cap rules: April counts the 4096 turns, E2 counts the > 4096 turns
    april_p = cells[(cells["source"] == e2.SOURCE_APRIL) & (cells["condition"] == MTP)].iloc[0]
    e2_p = cells[(cells["source"] == e2.SOURCE_E2) & (cells["condition"] == MTP)].iloc[0]
    assert april_p["cap_turns"] == 14 and april_p["n_turns"] == 24  # 3 per rep-0 run, 4 per rep-1 run
    assert e2_p["cap_turns"] == 8 and e2_p["n_turns"] == 12
    assert april_p["final_cap"] == 2 and e2_p["final_cap"] == 2
    assert april_p["max_tokens"] == 4096 and e2_p["max_tokens"] == 16384
    # KBV: Opus E2 adherence 4 everywhere -> 0 KBV; GPT E2 adherence 3 with full recall -> 2/2
    e2_opus = cells[(cells["source"] == e2.SOURCE_E2) & (cells["condition"] == MTP) & (cells["judge_arm"] == "opus")].iloc[0]
    e2_gpt = cells[(cells["source"] == e2.SOURCE_E2) & (cells["condition"] == MTP) & (cells["judge_arm"] == "gpt")].iloc[0]
    assert e2_opus["kbv_count"] == 0 and e2_gpt["kbv_count"] == 2 and e2_gpt["kbv_rate"] == 1.0
    # brief-paired adherence delta under GPT: E2 3 minus the April mean 3.5 = -0.5 on both briefs
    d = tables["e2_uncapped_deltas"]
    row = d[(d["condition"] == MTP) & (d["judge_arm"] == "gpt") & (d["metric"] == "constraint_adherence") & (d["design"] == e2.DESIGN_PAIRED)].iloc[0]
    assert row["delta"] == pytest.approx(-0.5)
    assert row["briefs_down"] == 2 and row["briefs_up"] == 0
    assert row["n_briefs"] == 2
    assert row["judge_comparability"].startswith("judge-date gap")
    tok = d[(d["metric"] == "output_tokens_per_run")]
    assert set(tok["judge_comparability"]) == {"n/a: token metric, no judge involved"}
    # offset: Opus - GPT adherence is +1 on every E2 run; the E1 value comes from the CSV
    off = tables["e2_uncapped_offset"]
    row = off[(off["scope"] == e2.SCOPE_OVERALL) & (off["dimension"] == "constraint_adherence")].iloc[0]
    assert row["e2_offset"] == pytest.approx(1.0) and row["n_runs"] == 4
    assert row["e1_offset"] == pytest.approx(0.8)
    cell_row = off[(off["scope"] == MTP) & (off["dimension"] == "constraint_adherence")].iloc[0]
    assert pd.isna(cell_row["e1_offset"]) and "no per-cell" in cell_row["e1_source"]


def test_comparators_dir_is_independent_of_out_dir(tmp_path: Path) -> None:
    fx = Fixture(tmp_path, with_offset_csv=False)
    cmp = tmp_path / "cmp"
    _write_comparators(cmp)
    out2 = tmp_path / "out2"
    tables = fx.run(out_dir=out2, comparators_dir=cmp)
    off = tables["e2_uncapped_offset"]
    overall = off[off["scope"] == e2.SCOPE_OVERALL]
    assert len(overall) == 4
    assert overall["e1_offset"].tolist() == pytest.approx([0.8] * 4)
    assert (out2 / "e2_uncapped_offset.csv").exists() and not (out2 / e2.JUDGE_OFFSET_CSV).exists()
    d = tables["e2_uncapped_deltas"]
    gpt_text = d[(d["judge_arm"] == "gpt") & (d["metric"] == "constraint_adherence")]["judge_comparability"].iloc[0]
    assert "control (judge_drift_control.csv): GPT-5.4 re-scored n = 25" in gpt_text
    prov = json.loads((out2 / "e2_uncapped_offset.csv.provenance.json").read_text())
    assert any(str(cmp) in s for s in prov["inputs"])
    # missing comparators: warn, keep the graceful text, NaN E1 columns
    with pytest.warns(UserWarning, match="not found"):
        tables = fx.run(out_dir=tmp_path / "out3", comparators_dir=tmp_path / "nowhere")
    off = tables["e2_uncapped_offset"]
    assert off["e1_offset"].isna().all()
    assert set(off["e1_source"]) == {"no judge_offset.csv"}
    d = tables["e2_uncapped_deltas"]
    assert "judge_drift_control.csv not found" in d[d["judge_arm"] == "gpt"]["judge_comparability"].iloc[0]


def test_length_finish_reason_raises(tmp_path: Path) -> None:
    fx = Fixture(tmp_path, e2_finish="length")
    with pytest.raises(ValueError, match="finish_reason"):
        fx.run()


def test_provenance_keys_and_columns(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    tables = fx.run()
    for name, table in tables.items():
        csv = fx.out / f"{name}.csv"
        prov = json.loads((fx.out / f"{name}.csv.provenance.json").read_text())
        for key in ("inputs", "filters", "cap_rules", "kbv_rule", "ci_methods", "judge_dates_utc", "design", "wording_limit", "columns", "date", "rows_out"):
            assert key in prov, (name, key)
        assert prov["filters"]["stray_rule"] == e2.STRAY_RULE
        assert prov["ci_methods"]["seed"] == e2.BOOT_SEED
        assert "does not raise KBV" in prov["wording_limit"]
        assert list(pd.read_csv(csv).columns) == list(prov["columns"])
        assert prov["rows_out"] == len(table)
    assert "2026-04-17" in json.loads((fx.out / "e2_uncapped_deltas.csv.provenance.json").read_text())["judge_dates_utc"]["april_clean/gpt"]
