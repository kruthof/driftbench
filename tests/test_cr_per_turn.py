"""Tests for drift_bench.analysis.camera_ready.per_turn.

Synthetic fixtures under tmp_path. No network, no real data, no .env.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from drift_bench.analysis.camera_ready import per_turn as pt

GPT, FL = pt.GPT, pt.FLASH_LITE
MTN, MTP = pt.MTN, pt.MTP
DIMS = pt.DIMS

# One brief with constraints the probe scorer can find by keyword.
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


def _judge_record(run_id, brief_id, model_id, condition, turn, scores, judge=pt.OPUS, cost_estimated=False):
    return {
        "run_id": run_id, "brief_id": brief_id, "model_id": model_id, "condition": condition,
        "judge_model": judge, "judge_response_model": "claude-opus-4-6",
        "objective_fidelity": scores[0], "constraint_adherence": scores[1],
        "alternative_coverage": scores[2], "complexity_inflation": scores[3],
        "summary": "", "violations": [], "optional_extras_flagged": [],
        "timestamp": 1793042366.0 + turn, "turn_index": turn, "cost_usd": 0.03, "cost_estimated": cost_estimated,
    }


def _write_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec) + "\n")


def _transcript(run_id, brief_id, model_id, condition, probe_texts, repetition=0):
    msgs = [{"role": "system", "content": "sys", "is_probe": False}, {"role": "user", "content": "brief", "is_probe": False}]
    for t in range(1, 7):
        msgs.append({"role": "assistant", "content": f"turn {t}", "is_probe": False, "token_count": 100})
        if t <= 5:
            msgs.append({"role": "user", "content": "probe?", "is_probe": True})
            msgs.append({"role": "assistant", "content": probe_texts[t - 1], "is_probe": True, "token_count": 50})
            msgs.append({"role": "user", "content": "Continue.", "is_probe": False})
    return {
        "metadata": {"run_id": run_id, "brief_id": brief_id, "model_id": model_id, "condition": condition, "repetition": repetition, "error": None},
        "messages": msgs,
    }


class Fixture:
    """A small E5-shaped data root: 2 models x 2 conditions x n_briefs runs."""

    def __init__(self, tmp_path: Path, n_briefs: int = 3, scores=None, probes=None, check_ids=None):
        self.root = tmp_path / "data"
        self.briefs_dir = tmp_path / "briefs"
        self.briefs_dir.mkdir()
        self.runs: list[tuple[str, str, str, str]] = []
        for b in range(n_briefs):
            brief = dict(BRIEF, id=f"b{b}")
            (self.briefs_dir / f"b{b}.yaml").write_text(yaml.safe_dump(brief))
        k = 0
        for model in (GPT, FL):
            for cond in (MTN, MTP):
                for b in range(n_briefs):
                    self.runs.append((f"r{k:03d}", f"b{b}", model, cond))
                    k += 1
        self.scores = scores or (lambda rid, model, cond, turn: (4, 4 if cond == MTN else 3, 3, turn - 1 if turn <= 5 else 2))
        self.probes = probes or (lambda rid, turn: GOOD_PROBE)
        self.check_ids = check_ids if check_ids is not None else [r[0] for r in self.runs[:2]]
        self.write()

    @property
    def e5(self) -> Path:
        return self.root / pt.E5_DIR

    def write(self) -> None:
        april_rows = []
        for rid, brief, model, cond in self.runs:
            for t in pt.PER_TURN_TURNS:
                rec = _judge_record(rid, brief, model, cond, t, self.scores(rid, model, cond, t))
                _write_jsonl(self.e5 / pt.PER_TURN_SUBDIR / f"{pt.FILE_PREFIX}_t{t}_{rid}.jsonl", rec)
            s6 = self.scores(rid, model, cond, 6)
            april_rows.append({"run_id": rid, "brief_id": brief, "model_id": model, "condition": cond, "judge_model": pt.OPUS,
                               **{f"j_{d}": s6[i] for i, d in enumerate(DIMS)}})
            # April judge file: 2026-04-17 12:00 UTC
            april_rec = dict(_judge_record(rid, brief, model, cond, 6, s6), timestamp=1776427200.0)
            april_rec.pop("turn_index")
            _write_jsonl(self.root / "scores" / f"judge_{rid}.jsonl", april_rec)
            if rid in self.check_ids:
                s7 = self.scores(rid, model, cond, 7)
                _write_jsonl(self.e5 / pt.T6_CHECK_SUBDIR / f"{pt.FILE_PREFIX}_t6_{rid}.jsonl", _judge_record(rid, brief, model, cond, 6, s7))
            _write_jsonl(self.root / "transcripts" / f"{rid}_{brief}_{cond}_{model.replace('/', '_')}.jsonl",
                         _transcript(rid, brief, model, cond, [self.probes(rid, t) for t in range(1, 6)]))
        (self.root / "aggregated").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(april_rows).to_parquet(self.root / "aggregated" / "all_scores.parquet", index=False)
        (self.e5 / pt.RUN_IDS_FILE).write_text("# ids\n" + "\n".join(r[0] for r in self.runs) + "\n")
        (self.e5 / pt.T6_CHECK_IDS_FILE).write_text("\n".join(self.check_ids) + "\n")

    def run(self, out_dir: Path):
        return pt.run(self.root, out_dir, self.briefs_dir)


@pytest.fixture(autouse=True)
def no_stray(monkeypatch):
    """The synthetic root has no judge_*.jsonl tree; keep the stray set explicit."""
    monkeypatch.setattr(pt, "stray_run_ids", lambda data_root: set())


# ---------------------------------------------------------------- parsing and guards


def test_parse_turn_filename():
    assert pt.parse_turn_filename("judge_opus_t3_0abf473b87ad.jsonl") == (3, "0abf473b87ad")
    with pytest.raises(ValueError):
        pt.parse_turn_filename("judge_gpt_t3_abc.jsonl")
    with pytest.raises(ValueError):
        pt.parse_turn_filename("judge_opus_abc.jsonl")


def test_turn_index_mismatch_raises(tmp_path):
    d = tmp_path / "opus"
    rec = _judge_record("r1", "b0", GPT, MTN, 2, (4, 4, 4, 0))
    _write_jsonl(d / "judge_opus_t3_r1.jsonl", rec)
    with pytest.raises(ValueError, match="turn_index"):
        pt.load_turn_dir(d)


def test_wrong_judge_or_estimated_cost_raises(tmp_path):
    d = tmp_path / "opus"
    _write_jsonl(d / "judge_opus_t1_r1.jsonl", _judge_record("r1", "b0", GPT, MTN, 1, (4, 4, 4, 0), judge="openai/gpt-5.4"))
    with pytest.raises(ValueError, match="judge_model"):
        pt.load_turn_dir(d)
    _write_jsonl(d / "judge_opus_t1_r1.jsonl", _judge_record("r1", "b0", GPT, MTN, 1, (4, 4, 4, 0), cost_estimated=True))
    with pytest.raises(ValueError, match="cost_estimated"):
        pt.load_turn_dir(d)
    with pytest.raises(FileNotFoundError):
        pt.load_turn_dir(tmp_path / "nope")


def test_load_turn_dir_reads_good_files(tmp_path):
    d = tmp_path / "opus"
    _write_jsonl(d / "judge_opus_t1_r1.jsonl", _judge_record("r1", "b0", GPT, MTN, 1, (4, 3, 2, 1)))
    _write_jsonl(d / "judge_opus_t5_r1.jsonl", _judge_record("r1", "b0", GPT, MTN, 5, (3, 2, 1, 0)))
    df = pt.load_turn_dir(d)
    assert df["turn"].tolist() == [1, 5]
    assert df.set_index("turn").loc[5, "constraint_adherence"] == 2


def test_missing_run_turn_raises(tmp_path):
    fx = Fixture(tmp_path)
    (fx.e5 / pt.PER_TURN_SUBDIR / f"{pt.FILE_PREFIX}_t4_r002.jsonl").unlink()
    with pytest.raises(ValueError, match=r"missing \(run, turn\)"):
        fx.run(tmp_path / "out")


def test_missing_april_row_raises(tmp_path):
    fx = Fixture(tmp_path)
    p = fx.root / "aggregated" / "all_scores.parquet"
    df = pd.read_parquet(p)
    df[df["run_id"] != "r001"].to_parquet(p, index=False)
    with pytest.raises(ValueError, match="April row"):
        fx.run(tmp_path / "out")


def test_april_judge_guard(tmp_path):
    fx = Fixture(tmp_path)
    p = fx.root / "aggregated" / "all_scores.parquet"
    df = pd.read_parquet(p)
    df.loc[df["run_id"] == "r001", "judge_model"] = "openai/gpt-5.4"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="judged by"):
        fx.run(tmp_path / "out")


def test_missing_t6_check_file_raises(tmp_path):
    fx = Fixture(tmp_path)
    (fx.e5 / pt.T6_CHECK_SUBDIR / f"{pt.FILE_PREFIX}_t6_{fx.check_ids[0]}.jsonl").unlink()
    with pytest.raises(ValueError, match="turn-6 check"):
        fx.run(tmp_path / "out")


# ---------------------------------------------------------------- stray filter


def test_stray_filter_applied_and_recorded(tmp_path, monkeypatch):
    fx = Fixture(tmp_path)
    monkeypatch.setattr(pt, "stray_run_ids", lambda data_root: {"r000"})
    out = tmp_path / "out"
    tables = fx.run(out)
    cells = tables["e5_per_turn_cells"]
    assert cells[(cells["model_id"] == GPT) & (cells["condition"] == MTN)]["n"].unique().tolist() == [2]
    prov = json.loads((out / "e5_per_turn_cells.csv.provenance.json").read_text())
    assert prov["filters"]["sample"]["rows_dropped"] == 6          # 6 turns of r000
    assert prov["filters"]["sample"]["rule"] == pt.STRAY_RULE
    assert prov["filters"]["run_ids"]["n_after_stray"] == 11
    # r000 was a check id; it must leave the t6check contrast too.
    d = tables["e5_per_turn_deltas"]
    chk = d[(d["contrast"] == "t6check-t5") & (d["model_id"] == pt.POOLED) & (d["condition"] == pt.POOLED)]
    assert chk["n"].unique().tolist() == [1]


# ---------------------------------------------------------------- cells


def test_cell_means_and_n_by_hand(tmp_path):
    def scores(rid, model, cond, turn):
        # adherence: r-index parity decides; complexity grows with turn
        base = int(rid[1:]) % 3
        return (4, 4 - base if turn > 1 else 4, 2, min(turn, 4))

    fx = Fixture(tmp_path, scores=scores)
    tables = fx.run(tmp_path / "out")
    cells = tables["e5_per_turn_cells"].set_index(["model_id", "condition", "turn"])
    # GPT neutral runs r000, r001, r002 -> base 0,1,2 -> adherence at turn 2: 4,3,2 -> mean 3
    assert cells.loc[(GPT, MTN, 2), "n"] == 3
    assert cells.loc[(GPT, MTN, 2), "mean_constraint_adherence"] == pytest.approx(3.0)
    assert cells.loc[(GPT, MTN, 1), "mean_constraint_adherence"] == pytest.approx(4.0)
    assert cells.loc[(GPT, MTN, 2), "share_ca_lt4"] == pytest.approx(2 / 3)
    assert cells.loc[(GPT, MTN, 1), "share_ca_lt4"] == 0
    assert cells.loc[(GPT, MTN, 3), "mean_complexity_inflation"] == pytest.approx(3.0)
    assert cells.loc[(GPT, MTN, 6), "source"] == pt.SOURCE_T6_APRIL
    assert cells.loc[(GPT, MTN, 6), "format"] == pt.FORMAT_FINAL
    assert cells.loc[(GPT, MTN, 1), "format"] == pt.FORMAT_REFINEMENT
    assert len(tables["e5_per_turn_cells"]) == 24
    # t interval by hand on [4, 3, 2]
    x = np.array([4.0, 3.0, 2.0])
    se = x.std(ddof=1) / math.sqrt(3)
    q = stats.t.ppf(0.975, 2)
    assert cells.loc[(GPT, MTN, 2), "se_constraint_adherence"] == pytest.approx(se)
    assert cells.loc[(GPT, MTN, 2), "ci_lo_constraint_adherence"] == pytest.approx(3 - q * se)
    assert cells.loc[(GPT, MTN, 2), "ci_hi_constraint_adherence"] == pytest.approx(3 + q * se)
    # Wilson by hand for 2 of 3
    lo, hi = pt.wilson_ci(2, 3)
    assert cells.loc[(GPT, MTN, 2), "share_ca_lt4_ci_lo"] == pytest.approx(lo)
    assert cells.loc[(GPT, MTN, 2), "share_ca_lt4_ci_hi"] == pytest.approx(hi)


def test_t6_rows_come_from_parquet_not_check_dir(tmp_path):
    def scores(rid, model, cond, turn):
        if turn == 6:
            return (4, 4, 4, 0)       # April (parquet)
        if turn == 7:
            return (1, 1, 1, 3)       # today's t6 check, deliberately different
        return (4, 4, 2, 1)

    fx = Fixture(tmp_path, scores=scores)
    tables = fx.run(tmp_path / "out")
    cells = tables["e5_per_turn_cells"]
    t6 = cells[cells["turn"] == 6]
    assert (t6["mean_alternative_coverage"] == 4).all()
    assert (t6["mean_constraint_adherence"] == 4).all()
    d = tables["e5_per_turn_deltas"]
    a = d[(d["contrast"] == "t6april-t5") & (d["dimension"] == "alternative_coverage") & (d["model_id"] == GPT) & (d["condition"] == MTN)].iloc[0]
    assert a["mean_delta"] == pytest.approx(2.0)
    c = d[(d["contrast"] == "t6check-t5") & (d["dimension"] == "alternative_coverage") & (d["model_id"] == pt.POOLED) & (d["condition"] == pt.POOLED)].iloc[0]
    assert c["mean_delta"] == pytest.approx(-1.0)
    assert c["n"] == 2


def test_t6check_uses_exactly_the_check_ids(tmp_path):
    fx = Fixture(tmp_path, n_briefs=4)
    assert len(fx.check_ids) == 2
    # An extra, unlisted check file must be ignored, not counted.
    rid, brief, model, cond = fx.runs[5]
    _write_jsonl(fx.e5 / pt.T6_CHECK_SUBDIR / f"{pt.FILE_PREFIX}_t6_{rid}.jsonl", _judge_record(rid, brief, model, cond, 6, (4, 4, 4, 0)))
    out = tmp_path / "out"
    tables = fx.run(out)
    d = tables["e5_per_turn_deltas"]
    chk = d[(d["contrast"] == "t6check-t5") & (d["model_id"] == pt.POOLED) & (d["condition"] == pt.POOLED)]
    assert chk["n"].unique().tolist() == [2]
    prov = json.loads((out / "e5_per_turn_deltas.csv.provenance.json").read_text())
    assert prov["filters"]["t6_check_ids"]["ids"] == sorted(fx.check_ids)
    assert prov["filters"]["t6_check_files_outside_check_ids_ignored"] == [rid]


# ---------------------------------------------------------------- statistics


def test_paired_ci_matches_scipy_t_by_hand():
    later = np.array([3, 2, 4, 3, 1], dtype=float)
    earlier = np.array([4, 4, 4, 3, 3], dtype=float)
    st = pt.paired_delta(later, earlier)
    d = later - earlier
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    assert st["n"] == 5
    assert st["mean_delta"] == pytest.approx(d.mean())
    assert st["se"] == pytest.approx(se)
    assert st["ci_lo"] == pytest.approx(d.mean() - q * se)
    assert st["ci_hi"] == pytest.approx(d.mean() + q * se)
    assert st["p_paired_t"] == pytest.approx(stats.ttest_rel(later, earlier).pvalue)
    assert st["p_wilcoxon"] == pytest.approx(stats.wilcoxon(d, zero_method="wilcox", method="approx").pvalue)
    assert (st["n_down"], st["n_same"], st["n_up"]) == (3, 2, 0)  # deltas -1, -2, 0, 0, -2


def test_wilcoxon_all_zero_deltas_gives_p_one():
    st = pt.paired_delta(np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0]))
    assert st["p_wilcoxon"] == 1.0
    assert st["p_paired_t"] == 1.0
    assert st["mean_delta"] == 0.0 and st["ci_lo"] == 0.0 and st["ci_hi"] == 0.0
    assert st["n_same"] == 3


def test_step_share_hand_value_and_nan():
    assert pt.step_share(-0.4, -0.45) == pytest.approx(0.4 / 0.45)
    assert math.isnan(pt.step_share(0.1, 0.0))
    assert math.isnan(pt.step_share(0.1, math.nan))


def test_step_share_in_deltas_table(tmp_path):
    def scores(rid, model, cond, turn):
        # adherence 4 at t1, 3 at t2..t4, 2 at t5 -> t2-t1 = -1, t5-t1 = -2, share 0.5
        adh = {1: 4, 2: 3, 3: 3, 4: 3, 5: 2}.get(turn, 3)
        return (4, adh, 3, 0)

    fx = Fixture(tmp_path, scores=scores)
    d = fx.run(tmp_path / "out")["e5_per_turn_deltas"]
    row = d[(d["contrast"] == "t2-t1") & (d["dimension"] == "constraint_adherence") & (d["model_id"] == GPT) & (d["condition"] == MTP)].iloc[0]
    assert row["step_share"] == pytest.approx(0.5)
    other = d[(d["contrast"] != "t2-t1")]
    assert other["step_share"].isna().all()
    # objective_fidelity never moves: denominator 0 -> NaN, no exception
    flat = d[(d["contrast"] == "t2-t1") & (d["dimension"] == "objective_fidelity")]
    assert flat["step_share"].isna().all()
    assert (flat["p_wilcoxon"] == 1.0).all()


def test_ci_excludes_zero():
    assert pt.ci_excludes_zero(-0.63, -0.26) is True
    assert pt.ci_excludes_zero(0.1, 0.5) is True
    assert pt.ci_excludes_zero(-0.1, 0.5) is False
    assert pt.ci_excludes_zero(0.0, 0.5) is False
    assert pt.ci_excludes_zero(math.nan, 0.5) is False


def test_step_share_denominator_flag_in_deltas(tmp_path):
    def scores(rid, model, cond, turn):
        # pressure: adherence falls 4 -> 3 at t2 and stays; neutral: flat 4.
        # alternative_coverage: 3 at t1, 2 at t2, 4 at t5 on one run only -> t5-t1 CI contains 0.
        adh = 4 if cond == MTN or turn == 1 else 3
        alt = {1: 3, 2: 2}.get(turn, 3)
        if rid == "r003" and turn == 5:
            alt = 4
        return (4, adh, alt, 0)

    fx = Fixture(tmp_path, n_briefs=4)
    fx.scores = scores
    fx.write()
    d = fx.run(tmp_path / "out")["e5_per_turn_deltas"]
    t21 = d[d["contrast"] == "t2-t1"].set_index(["model_id", "condition", "dimension"])
    # pressure adherence: t5-t1 = -1 on every run -> CI [-1, -1] excludes 0
    assert t21.loc[(GPT, MTP, "constraint_adherence"), "step_share_denominator_ci_excludes_zero"] == True  # noqa: E712
    assert t21.loc[(GPT, MTP, "constraint_adherence"), "step_share"] == pytest.approx(1.0)
    # neutral adherence: all-zero deltas -> CI [0, 0] does not exclude 0
    assert t21.loc[(GPT, MTN, "constraint_adherence"), "step_share_denominator_ci_excludes_zero"] == False  # noqa: E712
    # GPT neutral alt coverage: t5-t1 = [0, 0, 0, 1] -> CI straddles 0, flag False, share NaN-free but not quotable
    assert t21.loc[(GPT, MTN, "alternative_coverage"), "step_share_denominator_ci_excludes_zero"] == False  # noqa: E712
    other = d[d["contrast"] != "t2-t1"]
    assert other["step_share_denominator_ci_excludes_zero"].isna().all()
    assert str(d["step_share_denominator_ci_excludes_zero"].dtype) == "boolean"


def test_april_judge_dates_computed_not_hardcoded(tmp_path):
    fx = Fixture(tmp_path)
    info = pt.april_judge_dates(fx.root, [r[0] for r in fx.runs] + ["missing"])
    assert info == {"dates": ["2026-04-17"], "n_files": 12, "n_missing_files": 1, "n_without_timestamp": 0}
    out = tmp_path / "out"
    fx.run(out)
    prov = json.loads((out / "e5_per_turn_cells.csv.provenance.json").read_text())
    assert prov["dates"]["t6_april"]["judge_dates_utc"] == ["2026-04-17"]
    assert prov["dates"]["t6_april"]["n_judge_files"] == 12
    assert "kbv_note" in prov["filters"] and "t6check_note" in prov["filters"]
    assert "n = 2" in prov["filters"]["t6check_note"]


def test_contrast_flags():
    d = pt.deltas_table(pt.wide_scores(_tiny_long(), pd.DataFrame(columns=["run_id", *DIMS])))
    flags = d.drop_duplicates("contrast").set_index("contrast")
    assert flags.loc["t6april-t5", "same_judge_date"] == False  # noqa: E712
    assert flags.loc["t6april-t5", "format_matched"] == False  # noqa: E712
    assert flags.loc["t6check-t5", "format_matched"] == False  # noqa: E712
    assert flags.loc["t6check-t5", "same_judge_date"] == True  # noqa: E712
    assert flags.loc["t5-t1", "same_judge_date"] == True and flags.loc["t5-t1", "format_matched"] == True  # noqa: E712
    # no check files: t6check rows have n = 0 and NaN statistics
    chk = d[d["contrast"] == "t6check-t5"]
    assert (chk["n"] == 0).all() and chk["mean_delta"].isna().all()


def _tiny_long() -> pd.DataFrame:
    rows = []
    for rid in ("a", "b"):
        for t in range(1, 7):
            rows.append({"run_id": rid, "brief_id": "b0", "model_id": GPT, "condition": MTP, "turn": t,
                         **{d: 4 for d in DIMS}, "source": "x", "format": "y", "probe_accuracy": 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- KBV and probes


def test_kbv_uses_probe_after_turn_t(tmp_path):
    def scores(rid, model, cond, turn):
        return (4, 3, 3, 0)  # every turn violates (adherence 3)

    def probes(rid, turn):
        return GOOD_PROBE if turn in (1, 3, 5) else BAD_PROBE  # knows after turns 1, 3, 5 only

    fx = Fixture(tmp_path, scores=scores, probes=probes)
    cells = fx.run(tmp_path / "out")["e5_per_turn_cells"].set_index(["model_id", "condition", "turn"])
    for turn, expect in [(1, 1.0), (2, 0.0), (3, 1.0), (4, 0.0), (5, 1.0), (6, 1.0)]:  # turn 6 uses probe 5
        assert cells.loc[(GPT, MTP, turn), "kbv_rate"] == pytest.approx(expect), turn
        assert cells.loc[(GPT, MTP, turn), "kbv_n"] == 3
    assert cells.loc[(GPT, MTP, 2), "probe_recall_mean"] == pytest.approx(0.0)
    assert cells.loc[(GPT, MTP, 1), "probe_recall_mean"] == pytest.approx(1.0)
    lo, hi = pt.wilson_ci(3, 3)
    assert cells.loc[(GPT, MTP, 1), "kbv_ci_lo"] == pytest.approx(lo)
    assert cells.loc[(GPT, MTP, 1), "kbv_ci_hi"] == pytest.approx(hi)


def test_probe_scorer_turn_alignment(tmp_path):
    (tmp_path / "briefs").mkdir()
    (tmp_path / "briefs" / "b0.yaml").write_text(yaml.safe_dump(dict(BRIEF, id="b0")))
    root = tmp_path / "data"
    texts = [GOOD_PROBE, BAD_PROBE, GOOD_PROBE, BAD_PROBE, GOOD_PROBE]
    _write_jsonl(root / "transcripts" / "r1_b0_x_y.jsonl", _transcript("r1", "b0", GPT, MTP, texts))
    df = pt.probe_accuracy_per_turn(root, ["r1"], tmp_path / "briefs")
    assert df["turn"].tolist() == [1, 2, 3, 4, 5]
    assert df["overall_accuracy"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    with pytest.raises(FileNotFoundError):
        pt.probe_accuracy_per_turn(root, ["r9"], tmp_path / "briefs")


def test_missing_probe_raises(tmp_path):
    fx = Fixture(tmp_path)
    path = next((fx.root / "transcripts").glob("r000_*.jsonl"))
    tr = json.loads(path.read_text())
    tr["messages"] = [m for m in tr["messages"] if not m.get("is_probe")]
    path.write_text(json.dumps(tr) + "\n")
    with pytest.raises(ValueError, match="no probe"):
        fx.run(tmp_path / "out")


# ---------------------------------------------------------------- pooled rows and provenance


def test_pooled_rows_n_is_twice_cell_n(tmp_path):
    fx = Fixture(tmp_path, n_briefs=38)
    out = tmp_path / "out"
    tables = fx.run(out)
    cells = tables["e5_per_turn_cells"]
    assert cells["n"].unique().tolist() == [38]
    d = tables["e5_per_turn_deltas"]
    pooled = d[(d["model_id"] == pt.POOLED) & (d["condition"] == MTP) & (d["contrast"] == "t5-t1")]
    assert pooled["n"].unique().tolist() == [76]
    allp = d[(d["model_id"] == pt.POOLED) & (d["condition"] == pt.POOLED) & (d["contrast"] == "t5-t1")]
    assert allp["n"].unique().tolist() == [152]
    # (4 cells + 2 pooled-per-condition + 1 pooled) x 4 dims x 5 contrasts
    assert len(d) == 7 * 4 * 5
    assert list(d.columns) == list(pt.COLUMNS["e5_per_turn_deltas"])
    assert list(cells.columns) == list(pt.COLUMNS["e5_per_turn_cells"])


def test_provenance_keys(tmp_path):
    fx = Fixture(tmp_path)
    out = tmp_path / "out"
    fx.run(out)
    for name in ("e5_per_turn_cells", "e5_per_turn_deltas"):
        assert (out / f"{name}.csv").exists()
        prov = json.loads((out / f"{name}.csv.provenance.json").read_text())
        for key in ("inputs", "filters", "ci_methods", "judges", "dates", "row_counts", "columns", "date", "rows_out", "output"):
            assert key in prov, (name, key)
        for key in ("sample", "stray_rule", "repetition_rule", "run_ids", "t6_check_ids", "kbv", "kbv_note", "t6check_note"):
            assert key in prov["filters"], (name, key)
        assert prov["filters"]["run_ids"]["n"] == 12
        assert prov["filters"]["t6_check_ids"]["n"] == 2
        assert prov["filters"]["repetitions_observed"] == {"0": 12}
        assert prov["judges"]["per_turn"]["judge_model"] == pt.OPUS
        assert set(prov["columns"]) == set(pd.read_csv(out / f"{name}.csv").columns)
    cells_prov = json.loads((out / "e5_per_turn_cells.csv.provenance.json").read_text())
    assert "run-clustered" in cells_prov["ci_methods"]["mean"]
    assert cells_prov["row_counts"]["rows_in"] == 72 and cells_prov["rows_out"] == 24
