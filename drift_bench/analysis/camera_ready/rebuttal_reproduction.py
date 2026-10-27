"""X16: reproduce the posted rebuttal values from the released data.

The author posted three rebuttals on 26 July and a two-part discussion reply on 31 July.
The camera-ready corrections list (Appendix ``app:corrections``) tags the values
in them ``told`` or ``regenerated``. This module recomputes each posted number
under the definition and sample the rebuttal used, from files on disk only. It
then says whether the result matches the posted value at the posted precision.

Rows whose inputs were lost, or that need new model calls, are listed with the
reason and no reproduced value. No filter here was chosen to hit a posted value
without saying so: where a posted value matches only under one of several
plausible definitions, the ``note`` column gives the others.

Output: ``rebuttal_reproduction.csv`` and its provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
CSV_NAME = "rebuttal_reproduction.csv"

R1, R2, R3, REPLY = "rebuttal 1, 26 Jul", "rebuttal 2, 26 Jul", "rebuttal 3, 26 Jul", "discussion reply to rebuttal 3, 31 Jul"

GPT = "openai/gpt-5.4"
MINI = "openai/gpt-5.4-mini"
SONNET = "anthropic/claude-sonnet-4-6"
PRO = "gemini/gemini-3.1-pro-preview"
FLASH = "gemini/gemini-3.1-flash-lite-preview"
LLAMA = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
QWEN = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
COMMERCIAL = [GPT, MINI, PRO, FLASH, SONNET]

SS, MTN, MTP, CKP = "single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"
MAIN_CONDITIONS = [SS, MTN, MTP, CKP]
MULTI_TURN = [MTN, MTP, CKP]
DRIFT_CASES = ("trajectory_drift", "trajectory_lock_in")

KNOWS = 0.8          # final-probe overall_accuracy at or above this "knows"
VIOLATES = 4         # judge constraint_adherence below this "violates"
PROBE_CAP = 1020     # Gemini Pro probe replies stop at 1,020 tokens (cap 1,024)
OUTPUT_CAP = 4096    # Sonnet assistant output cap
DESIGN_ROWS = 38 * 7 * 2 * 4  # briefs x models x repetitions x conditions

COLUMNS = ["key", "item", "post", "posted_value", "definition", "sample", "reproduced_value",
           "matches_at_posted_precision", "value", "numerator", "denominator", "value_mantissa",
           "value_exponent", "basis", "note"]

RELEASED = "released files"
RERUN = "camera-ready re-run"
LOST = "not reproducible"


# ---------------------------------------------------------------------------
# Precision helpers (pure)
# ---------------------------------------------------------------------------

def _decimals(number: str) -> int:
    s = number.strip().lstrip("+-")
    return len(s.split(".", 1)[1]) if "." in s else 0


def _quant(value: float, places: int) -> Decimal:
    q = Decimal(1).scaleb(-places)
    return Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP)


def sci_parts(value: float, places: int) -> tuple[Decimal, int]:
    """Mantissa (rounded half up to ``places``) and exponent of ``value``."""
    if value == 0 or not math.isfinite(value):
        raise ValueError(f"no scientific form for {value!r}")
    exp = math.floor(math.log10(abs(value)))
    mant = _quant(value / 10 ** exp, places)
    if abs(mant) >= 10:
        exp += 1
        mant = _quant(value / 10 ** exp, places)
    return mant, exp


def format_at_precision(value: float | None, posted_number: str) -> str:
    """``value`` printed with the decimals of ``posted_number`` ('' when absent)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = posted_number.strip()
    if "e" in s.lower():
        mant_s, _ = s.lower().split("e")
        mant, exp = sci_parts(value, _decimals(mant_s))
        return f"{mant}e{exp:+03d}"
    return str(_quant(value, _decimals(s)))


def matches_at_precision(value: float | None, posted_number: str) -> bool:
    """True when ``value`` rounds (half up) to ``posted_number`` at its printed precision.

    ``posted_number`` is the bare number as posted: '12.2', '.039', '-1.12', '2.0e-08'.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    s = posted_number.strip()
    if "e" in s.lower():
        mant_s, exp_s = s.lower().split("e")
        mant, exp = sci_parts(value, _decimals(mant_s))
        return exp == int(exp_s) and mant == Decimal(mant_s)
    return _quant(value, _decimals(s)) == Decimal(s)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def run_mean_share(df: pd.DataFrame, flag: str, run_col: str = "run_id") -> float:
    """Mean over runs of the per-run share of items with ``flag`` true."""
    return float(df.groupby(run_col)[flag].mean().mean())


def pooled_share(df: pd.DataFrame, flag: str) -> float:
    """Share of all items with ``flag`` true, pooled over runs."""
    return float(df[flag].mean())


def brief_mean_deltas(runs: pd.DataFrame, value: str, cond_a: str = SS, cond_b: str = MTP) -> pd.DataFrame:
    """Brief means of ``value`` in two conditions, briefs present in both."""
    m = runs[runs["condition"].isin([cond_a, cond_b])].groupby(["brief_id", "condition"])[value].mean().unstack("condition")
    if cond_a not in m or cond_b not in m:
        return pd.DataFrame(columns=[cond_a, cond_b])
    return m.dropna(subset=[cond_a, cond_b])


def kbv_flag(accuracy: pd.Series, adherence: pd.Series) -> pd.Series:
    """KBV per run: knows (accuracy >= 0.8) and violates (adherence < 4). NaN accuracy is not KBV."""
    return (accuracy >= KNOWS) & (adherence < VIOLATES)


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def make_row(key: str, item: str, post: str, posted_value: str, posted_number: str | None,
             definition: str, sample: str, value: float | None = None,
             numerator: float | None = None, denominator: float | None = None,
             posted_numerator: float | None = None, posted_denominator: float | None = None,
             basis: str = RELEASED, note: str = "") -> dict:
    """One CSV row. A count posted as k/n must match both k and n."""
    has_value = value is not None and not (isinstance(value, float) and math.isnan(value))
    match = False
    if has_value and posted_number is not None:
        match = matches_at_precision(value, posted_number)
        if posted_numerator is not None and numerator != posted_numerator:
            match = False
        if posted_denominator is not None and denominator != posted_denominator:
            match = False
    mant = exp = None
    if has_value and posted_number is not None and "e" in posted_number.lower():
        m, e = sci_parts(value, _decimals(posted_number.lower().split("e")[0]))
        mant, exp = float(m), int(e)
    return {
        "key": key, "item": item, "post": post, "posted_value": posted_value,
        "definition": definition, "sample": sample,
        "reproduced_value": format_at_precision(value, posted_number) if has_value and posted_number else "",
        "matches_at_posted_precision": "yes" if match else "no",
        "value": float(value) if has_value else np.nan,
        "numerator": numerator, "denominator": denominator,
        "value_mantissa": mant, "value_exponent": exp,
        "basis": basis if has_value or basis != RELEASED else LOST, "note": note,
    }


def lost_row(key: str, item: str, post: str, posted_value: str, definition: str, reason: str,
             sample: str = "") -> dict:
    return make_row(key, item, post, posted_value, None, definition, sample, basis=LOST, note=reason)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _last_json_line(path: Path) -> dict | None:
    with path.open() as fh:
        lines = [ln for ln in fh if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def load_raw_judge(data_root: Path) -> pd.DataFrame:
    """Published-arm judge files (last line per file): run_id, model_id, condition, adherence."""
    rows = []
    for sub in ("scores", "openweight_subjects/scores"):
        for path in sorted((data_root / sub).glob("judge_*.jsonl")):
            rec = _last_json_line(path)
            if rec is None or path.name != f"judge_{rec['run_id']}.jsonl":
                continue
            rows.append({"run_id": rec["run_id"], "model_id": rec.get("model_id"),
                         "condition": rec.get("condition"), "brief_id": rec.get("brief_id"),
                         "adherence": rec.get("constraint_adherence")})
    out = pd.DataFrame(rows, columns=["run_id", "model_id", "condition", "brief_id", "adherence"])
    out["adherence"] = pd.to_numeric(out["adherence"], errors="coerce")
    return out


def load_auditor(data_root: Path) -> pd.DataFrame:
    from drift_bench.analysis.camera_ready.auditor_onset import summarise_record
    from drift_bench.analysis.camera_ready.b4_common import read_json_lines, score_files

    records = []
    for f in score_files(data_root, "auditor"):
        records.extend(read_json_lines(f))
    df = pd.DataFrame([summarise_record(r) for r in records])
    return df.drop_duplicates(subset="run_id", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def human_rows(data_root: Path, briefs_dir: Path, scores: pd.DataFrame, checks_raw: pd.DataFrame) -> list[dict]:
    from drift_bench.analysis.camera_ready import human_severity as hs

    hv = data_root / "human_validation"
    human = hs.load_human_scores(hv / "human_scores.json")
    rl = hs.load_rater_labels(hv / "raters")
    jf = hs.load_judge_violations_flag([data_root / "scores", data_root / "openweight_subjects" / "scores"])
    runs = hs.build_runs(human, scores, checks_raw, rl, jf)
    byc = hs.by_condition_table(runs, rl)
    vj = hs.human_vs_judge_table(runs, checks_raw, hs.load_brief_constraints(briefs_dir))

    def vjv(metric: str, agg: str) -> pd.Series:
        return vj[(vj["metric"] == metric) & (vj["aggregation"] == agg)].iloc[0]

    def bc(cond: str, agg: str) -> pd.Series:
        return byc[(byc["condition"] == cond) & (byc["aggregation"] == agg)].iloc[0]

    item = "Human severity"
    samp = "as published: 110 human-rated runs, condition from the 2,146-row parquet; structured files incl. stray runs"
    rows = []
    shared = vjv("shared_runs", "run_level")
    n_sh = int(shared["value"])
    for key, metric, posted, num, defn in [
        ("HumanNotSatShared", "human_share_not_satisfied_shared", "12.2%", "12.2", "human median label not satisfied"),
        ("JudgeNotSatShared", "judge_share_not_satisfied_shared", "5.5%", "5.5", "structured judge status not satisfied"),
        ("HumanViolJudgeNot", "human_violated_judge_not_violated", "5.1%", "5.1",
         "human median label violated where the structured judge marks stretched or satisfied"),
    ]:
        r = vjv(metric, "run_mean")
        rows.append(make_row(key, item, f"{R1}; {R2}" if key == "HumanNotSatShared" else R1, posted, num,
                             f"{defn}; mean over shared runs of the per-run item share", samp + f"; {n_sh} shared runs",
                             100 * float(r["value"]), denominator=n_sh,
                             note="pooled over items: " + f"{100 * float(vjv(metric, 'pooled')['value']):.2f}%"))
    ss_m, ss_p = bc(SS, "median_label_run_mean"), bc(SS, "median_label_pooled")
    p_m, p_p = bc(MTP, "median_label_run_mean"), bc(MTP, "median_label_pooled")
    rows += [
        make_row("SevNotSatSingle", item, R1, "5.3%", "5.3", "per-constraint median label not satisfied; run mean",
                 samp + f"; single-shot {int(ss_m['n_runs'])} runs / {int(ss_m['n_items'])} items",
                 100 * float(ss_m["share_not_satisfied"]), denominator=int(ss_m["n_runs"]),
                 note=f"pooled gives {100 * float(ss_p['share_not_satisfied']):.2f}%"),
        make_row("SevViolSingle", item, f"{R1}; {R2}", "1.2%", "1.2", "per-constraint median label violated; pooled over items",
                 samp + "; single-shot", 100 * float(ss_p["share_violated"]),
                 numerator=int(round(float(ss_p["share_violated"]) * int(ss_p["n_items"]))), denominator=int(ss_p["n_items"]),
                 note=(f"the run mean gives {100 * float(ss_m['share_violated']):.2f}% (no match); the other cells of the "
                       "posted table match the run mean, so the table mixes the two aggregations")),
        make_row("SevNotSatPressure", item, R1, "24.0%", "24.0", "per-constraint median label not satisfied; run mean",
                 samp + f"; pressure {int(p_m['n_runs'])} runs / {int(p_m['n_items'])} items",
                 100 * float(p_m["share_not_satisfied"]), denominator=int(p_m["n_runs"]),
                 note=f"pooled gives {100 * float(p_p['share_not_satisfied']):.2f}%"),
        make_row("SevViolPressure", item, f"{R1}; {R2}", "13.9%", "13.9", "per-constraint median label violated; run mean",
                 samp + "; pressure", 100 * float(p_m["share_violated"]), denominator=int(p_m["n_runs"]),
                 note=f"pooled gives {100 * float(p_p['share_violated']):.2f}%"),
        make_row("SevViolPerRunSingle", item, f"{R1}; {R2}", "0.07", "0.07", "violated median labels per run", samp + "; single-shot",
                 float(ss_m["violations_per_run"])),
        make_row("SevViolPerRunPressure", item, f"{R1}; {R2}", "0.93", "0.93", "violated median labels per run", samp + "; pressure",
                 float(p_m["violations_per_run"])),
        make_row("SevRunsViolSingle", item, f"{R1}; {R2}", "7.4% (runs / items 27 / 167)", "7.4",
                 "share of runs with at least one violated median label", samp + "; single-shot",
                 100 * float(ss_m["share_runs_with_violation"]), numerator=int(ss_m["n_runs_with_violation"]),
                 denominator=int(ss_m["n_runs"]), posted_denominator=27,
                 note="the clean sample drops one stray single-shot run"),
        make_row("SevRunsViolPressure", item, f"{R1}; {R2}", "36.6% (15/41)", "36.6",
                 "share of runs with at least one violated median label", samp + "; pressure",
                 100 * float(p_m["share_runs_with_violation"]), numerator=int(p_m["n_runs_with_violation"]),
                 denominator=int(p_m["n_runs"]), posted_numerator=15, posted_denominator=41),
    ]
    d = vj.set_index("metric")
    rows += [
        make_row("HumanDelta", "Human delta", R1, "-1.12", "-1.12",
                 "mean computed_score (median of raters' scores), pressure minus single-shot", samp,
                 float(d.loc["human_computed_score_delta_pressure_minus_single_shot", "value"]),
                 denominator=int(d.loc["human_computed_score_delta_pressure_minus_single_shot", "n_runs"]),
                 note="the clean sample gives the paper's \\crHumanScoreDelta"),
        make_row("HumanMedianDelta", "Human delta", R1, "-1.02", "-1.02",
                 "mean score from the per-constraint median labels, pressure minus single-shot", samp,
                 float(d.loc["human_score_from_constraint_median_delta_pressure_minus_single_shot", "value"]),
                 denominator=int(d.loc["human_score_from_constraint_median_delta_pressure_minus_single_shot", "n_runs"]),
                 note="the clean sample gives the paper's \\crHumanMedianScoreDelta"),
    ]
    # Protocol counts and rater strictness, from the raw rater files.
    n_items = int(human["n_items"].sum())
    flagged = {}
    for path in sorted((hv / "raters").glob("*.json")):
        data = json.loads(path.read_text())
        flagged[path.name] = sum(1 for rec in data.values() for s in rec["constraint_statuses"]
                                 if hs.normalise_status(s) == "violated")
    raters = sorted({str(r) for r in rl["rater_id"].dropna()})
    rows += [
        make_row("HumanRuns", "Human raters", R1, "110 runs", "110", "runs in human_scores.json", "all human-rated runs",
                 float(len(human))),
        make_row("HumanItems", "Human raters", R1, "684 items", "684", "constraint items over the human-rated runs",
                 "all human-rated runs", float(n_items)),
        make_row("HumanLabels", "Human raters", R1, "2,052 labels", "2052", "rater labels in the raw rater files",
                 "all rater files", float(len(rl))),
        make_row("RaterFiles", "Human raters", f"{R1}; {R2}; {R3}", "six rater-set files", "6",
                 "rater export files", "human_validation/raters/*.json", float(len(flagged))),
        make_row("Raters", "Human raters", f"{R1}; {R2}; {R3}", "four raters", "4",
                 "distinct rater_id over the rater files", "human_validation/raters/*.json", float(len(raters)),
                 note="rater ids: " + ", ".join(raters)),
        make_row("RaterFlagMin", "Human raters", f"{R1}; {R2}", "15 to 86 flagged violations", "15",
                 "fewest violated labels in one rater file", "human_validation/raters/*.json",
                 float(min(flagged.values())) if flagged else None),
        make_row("RaterFlagMax", "Human raters", f"{R1}; {R2}", "15 to 86 flagged violations", "86",
                 "most violated labels in one rater file", "human_validation/raters/*.json",
                 float(max(flagged.values())) if flagged else None),
    ]
    return rows


def graded_rows(data_root: Path, briefs: dict[str, list[str]], fp: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    from drift_bench.analysis.camera_ready import graded_severity as gs

    checks = gs.load_structured_checks(data_root / "scores", "structured", "structured", briefs)
    runs = gs.summarise_runs(checks)
    item = "Graded-severity tests"
    samp = ("published structured arm, every structured_*.jsonl file (stray runs and three unaggregated runs included), "
            "checks matched to the brief's hard constraints")
    rows = []
    p = runs[runs["condition"] == MTP]
    for model, posted, n_posted, key in [(GPT, ".039", 76, "GkbvGpt"), (MINI, ".029", 76, "GkbvMini"),
                                         (PRO, ".026", 76, "GkbvPro"), (FLASH, ".076", 75, "GkbvFlashLite"),
                                         (SONNET, ".132", 82, "GkbvSonnet")]:
        g = p[p["model_id"] == model]
        rows.append(make_row(key, item, R1, f"{posted} (n {n_posted})", posted,
                             "mean over pressure runs of the share of hard-constraint checks not satisfied (gKBV)",
                             samp, float(g["g_kbv"].mean()), denominator=int(len(g)), posted_denominator=n_posted))
    r = p.merge(fp[["run_id", "overall_accuracy", "constraint_recall"]], on="run_id", how="left")
    pro = r[r["model_id"] == PRO]
    cr1 = pro[pro["constraint_recall"] == 1.0]
    oa1 = pro[pro["overall_accuracy"] == 1.0]
    rows.append(make_row("GkbvProPerfect", "Gemini 3.1 Pro gKBV at perfect recall", R1, ".023 (n 76 to 58)", ".023",
                         "Gemini Pro pressure gKBV over runs whose final probe recalls every hard constraint (constraint_recall == 1)",
                         samp + "; final_probe_accuracy.csv", float(cr1["g_kbv"].mean()), denominator=int(len(cr1)),
                         posted_denominator=58,
                         note=(f"overall_accuracy == 1 (constraints and banned moves) gives {oa1['g_kbv'].mean():.4f} "
                               f"on {len(oa1)} runs")))
    others = []
    for model in (GPT, MINI, FLASH, SONNET):
        g = r[r["model_id"] == model]
        others.append(int((g["constraint_recall"] != 1.0).sum()))
    rows.append(make_row("GkbvOtherRowsImperfect", "Gemini 3.1 Pro gKBV at perfect recall", R1,
                         "leaves four rows unchanged", "0",
                         "pressure runs of the other four models without perfect constraint recall", samp,
                         float(sum(others)), note="0 means conditioning leaves the four rows unchanged"))
    s = p[p["model_id"] == SONNET]
    n_hc = sum(len(v) for v in briefs.values()) / len(briefs)
    rows.append(make_row("HardPerBrief", item, f"{R1}; {R3}", "6.16", "6.16", "hard constraints per brief (234/38)",
                         "38 briefs", float(n_hc)))
    rows.append(make_row("SonnetNotSatPerRun", item, R1, "0.81 of the 6.16 per run", "0.81",
                         "Sonnet pressure gKBV times hard constraints per brief", samp,
                         float(s["g_kbv"].mean() * n_hc),
                         note=(f"the mean count of non-satisfied checks per run is {s['n_not_satisfied'].mean():.3f} "
                               f"of {s['n_checks'].mean():.3f} checks (0.82 at the posted precision: no match)")))
    # Tests over brief means, single-shot vs pressure.
    pw, pt, pw_round = {}, {}, {}
    for model in COMMERCIAL:
        m = brief_mean_deltas(runs[runs["model_id"] == model], "g_kbv")
        d = (m[MTP] - m[SS]).to_numpy(dtype=float)
        pw[model] = float(stats.wilcoxon(d, zero_method="wilcox", method="asymptotic").pvalue)
        pw_round[model] = float(stats.wilcoxon(np.round(d, 9), zero_method="wilcox", method="asymptotic").pvalue)
        pt[model] = float(stats.ttest_rel(m[MTP], m[SS]).pvalue)
    hw = dict(zip(COMMERCIAL, holm_adjust([pw[m] for m in COMMERCIAL])))
    ht = dict(zip(COMMERCIAL, holm_adjust([pt[m] for m in COMMERCIAL])))
    surv_t = sorted(m for m in COMMERCIAL if ht[m] < 0.05)
    surv_w = sorted(m for m in COMMERCIAL if hw[m] < 0.05)
    rows.append(make_row("SonnetWilcoxonP", item, R1, "Wilcoxon p = 0.051", "0.051",
                         ("Wilcoxon signed-rank test of Sonnet's brief-mean gKBV, pressure minus single-shot; "
                          "asymptotic; differences not rounded"), samp, pw[SONNET],
                         note=(f"rounding the differences to 9 decimals, so that equal deltas tie as in graded_severity.py, "
                               f"gives p = {pw_round[SONNET]:.4f}")))
    for model, key in ((GPT, "HolmTGpt"), (FLASH, "HolmTFlashLite")):
        rows.append(make_row(key, item, R1, "under Holm correction only GPT-5.4 and Flash-Lite survive", None,
                             "Holm-adjusted p over five brief-paired t-tests of brief-mean gKBV (single-shot vs pressure)",
                             samp, ht[model],
                             note=(f"survivors at 0.05 under paired t: {surv_t}; under Holm over Wilcoxon tests: "
                                   f"{surv_w or 'none'}. The rebuttal does not name the test behind Holm")))
    for r_, model in ((rows[-2], GPT), (rows[-1], FLASH)):
        r_["matches_at_posted_precision"] = "yes" if surv_t == sorted([GPT, FLASH]) else "no"
        r_["reproduced_value"] = f"Holm p {ht[model]:.3f}"
    return rows, checks


def stretched_rows(checks: pd.DataFrame, data_root: Path) -> list[dict]:
    n_files = len(list((data_root / "scores").glob("structured_*.jsonl")))
    ns = checks[checks["status"].isin(["stretched", "violated"])]
    st = int((ns["status"] == "stretched").sum())
    samp = f"all {n_files} structured_*.jsonl files (stray runs and three unaggregated GPT-5.4-mini runs included)"
    return [
        make_row("StretchedOfNonSat", "Stretched versus violated", f"{R1}; {R2}", "450 of 469", "450",
                 "structured-judge checks matched to the brief's hard constraints; stretched among stretched or violated",
                 samp, float(st), numerator=st, denominator=int(len(ns)), posted_numerator=450, posted_denominator=469,
                 note="without matching to the brief the files give 450 of 470 (the corrections ledger's B17 count)"),
        make_row("TypologyChecks", "Constraint typology", R2, "9,386 checks", "9386",
                 "structured-judge checks matched to the brief's hard constraints", samp, float(len(checks))),
        make_row("HardConstraints", "Constraint typology", R2, "234 hard constraints", "234",
                 "hard constraints over the 38 briefs", "drift_bench/briefs/*.yaml", float(checks.attrs.get("n_brief_constraints", np.nan))
                 if checks.attrs.get("n_brief_constraints") else None),
    ]


def split_rows(scores: pd.DataFrame, fp: pd.DataFrame, raw: pd.DataFrame) -> list[dict]:
    item = "Stretch-versus-violate split"
    d = scores.merge(fp[["run_id", "overall_accuracy"]], on="run_id", how="left")
    d["kbv"] = kbv_flag(d["overall_accuracy"], d["j_constraint_adherence"])
    samp = "as published: the 2,146-row parquet, published judge arm; final_probe_accuracy.csv"
    p = d[d["condition"] == MTP]
    k = p[p["kbv"]]
    s = d[d["condition"] == SS]
    b3 = k[k["j_constraint_adherence"] < 3]
    sl = int(b3["model_id"].isin([SONNET, LLAMA]).sum())
    rows = [
        make_row("KbvAtThree", item, REPLY, "76.3% of the 283 KBV-positive pressure runs", "76.3",
                 "KBV-positive pressure runs with adherence 3", samp, 100 * float((k["j_constraint_adherence"] == 3).mean()),
                 numerator=int((k["j_constraint_adherence"] == 3).sum()), denominator=int(len(k)), posted_denominator=283),
        make_row("BelowThreeSingle", item, REPLY, "2.0% (11/542)", "2.0", "single-shot runs with adherence below 3", samp,
                 100 * float((s["j_constraint_adherence"] < 3).mean()), numerator=int((s["j_constraint_adherence"] < 3).sum()),
                 denominator=int(len(s)), posted_numerator=11, posted_denominator=542),
        make_row("BelowThreePressure", item, REPLY, "13.0% (71/545)", "13.0", "pressure runs with adherence below 3", samp,
                 100 * float((p["j_constraint_adherence"] < 3).mean()), numerator=int((p["j_constraint_adherence"] < 3).sum()),
                 denominator=int(len(p)), posted_numerator=71, posted_denominator=545),
        make_row("SonnetLlamaBelowThree", item, REPLY, "66 of 67 KBV-positive runs from Sonnet 4.6 and Llama", "66",
                 "KBV-positive pressure runs below 3 that come from Sonnet 4.6 or Llama", samp, float(sl), numerator=sl,
                 denominator=int(len(b3)), posted_numerator=66, posted_denominator=67),
        make_row("CellsThreeFour", "Rubric cells at the KBV cut", R3, "87%", "87", "pressure runs with adherence 3 or 4",
                 samp, 100 * float(p["j_constraint_adherence"].isin([3, 4]).mean()),
                 numerator=int(p["j_constraint_adherence"].isin([3, 4]).sum()), denominator=int(len(p))),
    ]
    sp = d[(d["model_id"] == SONNET) & (d["condition"] == MTP) & d["overall_accuracy"].notna()]
    rows.append(make_row("SonnetGptArm", "Rubric cells at the KBV cut", R3, "Sonnet moving 99% to 72%", "99",
                         "Sonnet pressure KBV on the published arm (GPT-5.4 judge), runs with a final probe", samp,
                         100 * float(sp["kbv"].mean()), numerator=int(sp["kbv"].sum()), denominator=int(len(sp))))
    rows.append(lost_row("SonnetOpusArm", "Rubric cells at the KBV cut", R3, "Sonnet moving 99% to 72%",
                         "Sonnet pressure KBV under an Opus 4.6 judge",
                         "the July Opus judgements of Sonnet were lost; the E1 re-run gives \\crKbvOpusArmPressureSonnet"))
    # Without Sonnet and Llama, raw judge files (they hold three GPT-5.4-mini runs absent from the parquet).
    ex = raw[~raw["model_id"].isin([SONNET, LLAMA])]
    rs, rp = ex[ex["condition"] == SS], ex[ex["condition"] == MTP]
    samp_raw = ("raw published-arm judge files (2,149 runs: the parquet plus three GPT-5.4-mini runs it lacks), "
                "five models other than Sonnet 4.6 and Llama")
    pq_ex = d[~d["model_id"].isin([SONNET, LLAMA])]
    pqs, pqp = pq_ex[pq_ex["condition"] == SS], pq_ex[pq_ex["condition"] == MTP]
    rows += [
        make_row("ExclBelowThreeSingle", item, REPLY, "2/378 against 1/378", None, "single-shot runs below 3", samp_raw,
                 float((rs["adherence"] < 3).sum()), numerator=int((rs["adherence"] < 3).sum()), denominator=int(len(rs)),
                 note=f"the parquet gives {int((pqs['j_constraint_adherence'] < 3).sum())}/{len(pqs)}"),
        make_row("ExclBelowThreePressure", item, REPLY, "2/378 against 1/378", None, "pressure runs below 3", samp_raw,
                 float((rp["adherence"] < 3).sum()), numerator=int((rp["adherence"] < 3).sum()), denominator=int(len(rp)),
                 note=f"the parquet gives {int((pqp['j_constraint_adherence'] < 3).sum())}/{len(pqp)}"),
    ]
    for r_, k_, d_ in ((rows[-2], 2, 378), (rows[-1], 1, 378)):
        r_["matches_at_posted_precision"] = "yes" if (r_["numerator"], r_["denominator"]) == (k_, d_) else "no"
        r_["reproduced_value"] = f"{r_['numerator']}/{r_['denominator']}"
    delta = float(rp["adherence"].mean() - rs["adherence"].mean())
    pmw = float(stats.mannwhitneyu(rp["adherence"].dropna(), rs["adherence"].dropna(), alternative="two-sided").pvalue)
    pq_p = float(stats.mannwhitneyu(pqp["j_constraint_adherence"], pqs["j_constraint_adherence"], alternative="two-sided").pvalue)
    rows += [
        make_row("ExclAdhDelta", item, REPLY, "-0.18", "-0.18", "mean adherence, pressure minus single-shot", samp_raw, delta,
                 note=f"the parquet gives {float(pqp['j_constraint_adherence'].mean() - pqs['j_constraint_adherence'].mean()):.4f}"),
        make_row("ExclAdhDeltaP", item, REPLY, "p = 2.0e-08", "2.0e-08",
                 "two-sided Mann-Whitney test of run adherence, pressure vs single-shot", samp_raw, pmw,
                 note=f"the parquet gives p = {pq_p:.3g}"),
    ]
    return rows


def kbv_rows(scores: pd.DataFrame, fp: pd.DataFrame, tindex: pd.DataFrame, auditor: pd.DataFrame,
             data_root: Path) -> list[dict]:
    d = scores.merge(fp[["run_id", "overall_accuracy"]], on="run_id", how="left")
    d = d.merge(tindex[["run_id", "final_probe_token_count"]], on="run_id", how="left")
    d["kbv"] = kbv_flag(d["overall_accuracy"], d["j_constraint_adherence"])
    samp = "as published: the 2,146-row parquet, published judge arm; final_probe_accuracy.csv"
    rows = []

    def cell(model: str, cond: str = MTP) -> pd.DataFrame:
        return d[(d["model_id"] == model) & (d["condition"] == cond) & d["overall_accuracy"].notna()]

    g, ll = cell(GPT), cell(LLAMA)
    rows += [
        make_row("OpusArmMin", "Headline KBV range", f"{R1}; {R2}; {R3}", "8% to 93% under a common Opus judge", "8",
                 "lowest pressure KBV under Opus 4.6: GPT-5.4, published Opus judgements", samp,
                 100 * float(g["kbv"].mean()), numerator=int(g["kbv"].sum()), denominator=int(len(g)),
                 note="Sonnet's Opus-judged value comes from the lost July files; its E1 re-run value lies inside the range"),
        make_row("OpusArmMax", "Headline KBV range", f"{R1}; {R2}; {R3}", "8% to 93% under a common Opus judge", "93",
                 "highest pressure KBV under Opus 4.6: Llama, published Opus judgements", samp,
                 100 * float(ll["kbv"].mean()), numerator=int(ll["kbv"].sum()), denominator=int(len(ll))),
        lost_row("GptArmMin", "Headline KBV range", f"{R1}; {R2}; {R3}", "43% to 99% under a common GPT-5.4 judge",
                 "lowest pressure KBV under GPT-5.4", "July GPT-5.4 judgements of the six non-Anthropic subjects were lost; "
                 "the E1 re-run gives \\crKbvGptArmMin"),
        lost_row("AboveHalf", "Headline KBV range", R2, "four of seven above 50% under Opus and six of seven under GPT-5.4",
                 "models above 50% pressure KBV per common-judge arm",
                 "needs the lost July crossed judgements; the E1 re-run gives \\crKbvOpusArmAboveHalfFrac and \\crKbvGptArmAboveHalfFrac"),
        lost_row("SonnetWorstProb", "Headline KBV range", f"{R1}; {R2}; {R3}", "0.0% under a common Opus judge and 67% under GPT-5.4",
                 "bootstrap probability that Sonnet is the worst model", "needs the lost July crossed judgements; not recomputed"),
        make_row("GptKbv", "GPT-5.4 drifts at only 4%", f"{R1}; {R2}; {R3}", "its KBV is 8%", "8",
                 "GPT-5.4 pressure KBV, published arm (Opus 4.6)", samp, 100 * float(g["kbv"].mean()),
                 numerator=int(g["kbv"].sum()), denominator=int(len(g))),
    ]
    a = auditor.merge(scores[["run_id", "a_drift_classification"]], on="run_id", how="inner")
    ag = a[(a["model_id"] == GPT) & (a["condition"] == MTP)]
    k = int(ag["a_drift_classification"].isin(DRIFT_CASES).sum())
    rows.append(make_row("GptAuditor", "GPT-5.4 drifts at only 4%", f"{R1}; {R2}; {R3}",
                         "the value from our separate drift-event auditor (4%)", "4",
                         "GPT-5.4 pressure runs the auditor classifies trajectory_drift or trajectory_lock_in", samp,
                         100 * k / len(ag), numerator=k, denominator=int(len(ag))))
    # Gemini Pro.
    pro = cell(PRO)
    trunc = pro["final_probe_token_count"] >= PROBE_CAP
    waived = ((pro["overall_accuracy"] >= KNOWS) | trunc) & (pro["j_constraint_adherence"] < VIOLATES)
    rows += [
        make_row("ProKbvWaived", "Gemini 3.1 Pro KBV", f"{R2}; {R3}", "25% not 18%", "25",
                 "Gemini Pro pressure KBV, a capped final probe (token_count >= 1,020) counted as knows", samp,
                 100 * float(waived.mean()), numerator=int(waived.sum()), denominator=int(len(pro))),
        make_row("ProKbvGraded", "Gemini 3.1 Pro KBV", f"{R2}; {R3}", "25% not 18%", "18",
                 "Gemini Pro pressure KBV, capped probes graded as stored", samp, 100 * float(pro["kbv"].mean()),
                 numerator=int(pro["kbv"].sum()), denominator=int(len(pro))),
    ]
    fpm = fp[fp["condition"].isin(MULTI_TURN)].merge(tindex[["run_id", "final_probe_token_count"]], on="run_id", how="left")
    fpm = fpm[fpm["run_id"].isin(scores["run_id"])]
    fpm["trunc"] = fpm["final_probe_token_count"] >= PROBE_CAP
    fpro = fpm[fpm["model_id"] == PRO]
    rows.append(make_row("ProProbesTruncated", "Gemini 3.1 Pro gKBV at perfect recall", R1, "truncated 73% of its probes", "73",
                         "Gemini Pro final probes with token_count >= 1,020, three multi-turn conditions",
                         samp, 100 * float(fpro["trunc"].mean()), numerator=int(fpro["trunc"].sum()), denominator=int(len(fpro))))
    pp = fpm[(fpm["model_id"] == PRO) & (fpm["condition"] == MTP)]
    below = pp[pp["overall_accuracy"] < KNOWS]
    imperfect = pp[pp["overall_accuracy"] < 1.0]
    rows.append(make_row("ProImperfect", "Probe recall", f"{R1}; {R2}; {R3}",
                         "Gemini Pro's 23 imperfect runs of 76 all have truncated probes", "23",
                         "Gemini Pro pressure runs whose final probe scores below 0.8; all have a capped probe",
                         samp, float(len(below)) if bool(below["trunc"].all()) else np.nan,
                         numerator=int(len(below)), denominator=int(len(pp)), posted_denominator=76,
                         note=(f"{int(below['trunc'].sum())} of the {len(below)} are capped. Runs below 1.0 number "
                               f"{len(imperfect)}, {int(imperfect['trunc'].sum())} of them capped")))
    rows.append(make_row("ProbePooled", "Probe recall", f"{R1}; {R2}; {R3}", "the paper's 97.3%", "97.3",
                         "mean final-probe overall_accuracy, three multi-turn conditions, capped probes graded as stored",
                         samp + f"; {len(fpm)} runs", 100 * float(fpm["overall_accuracy"].mean()), denominator=int(len(fpm))))
    unt = fpm[~fpm["trunc"]]
    by_model = unt.groupby("model_id")["overall_accuracy"].mean()
    by_model_p = unt[unt["condition"] == MTP].groupby("model_id")["overall_accuracy"].mean()
    rows.append(make_row("UntruncMin", "Probe recall", f"{R1}; {R2}; {R3}",
                         "untruncated probes give 100% for all seven models", "100",
                         "lowest per-model mean final-probe accuracy over uncapped probes, three multi-turn conditions",
                         samp, 100 * float(by_model.min()), denominator=int(len(by_model)),
                         note=(f"lowest model {by_model.idxmin()}; under pressure alone the lowest is "
                               f"{100 * float(by_model_p.min()):.2f}%. rebuttal 3's 'at every turn' was not checked")))
    return rows


def probe_e7_rows(data_root: Path, out_dir: Path) -> list[dict]:
    """Turn-1 substring accuracy of the E7 re-graded runs. Missing inputs raise."""
    name = "e7_strict_probe_runs.csv"
    candidates = [out_dir / name, data_root / "camera_ready" / "analysis" / name]
    e7 = next((c for c in candidates if c.exists()), None)
    pa_path = data_root / "analysis" / "probe_accuracy.csv"
    if e7 is None:
        raise FileNotFoundError(f"{name} not found in {[str(c) for c in candidates]}; run e7_summary first")
    if not pa_path.exists():
        raise FileNotFoundError(pa_path)
    ids = pd.read_csv(e7, usecols=["run_id"])["run_id"].unique()
    pa = pd.read_csv(pa_path)
    t1 = pa[pa["run_id"].isin(ids) & (pa["turn_index"] == pa.groupby("run_id")["turn_index"].transform("min"))]
    rows = [make_row("SubstringTurnOne", "Strict re-grading of the probes", f"{R1}; {R2}; {R3}",
                     "the substring scorer's 97.1%", "97.1",
                     "mean substring-scorer accuracy of the first probe of each re-graded run",
                     f"the {len(ids)} runs of {name}; probe_accuracy.csv",
                     100 * float(t1["overall_accuracy"].mean()), denominator=int(len(t1)))]
    rows += [
        lost_row("StrictRegrade", "Strict re-grading of the probes", f"{R1}; {R2}; {R3}", "96.9%",
                 "strict LLM re-grading of 304 probes", "the July grader outputs were lost; the E7 re-grade gives \\crESevenStrictRecallPooled"),
        lost_row("SideChanges", "Strict re-grading of the probes", f"{R1}; {R2}", "none changing side at the 0.8 threshold",
                 "probes whose knows flag changes between scorers", "needs the lost July grader outputs; E7 finds \\crESevenSideChangeStrictPooled"),
    ]
    return rows


def onset_rows(scores: pd.DataFrame, auditor: pd.DataFrame) -> list[dict]:
    a = auditor.merge(scores[["run_id", "a_drift_classification"]], on="run_id", how="inner")
    p = a[(a["condition"] == MTP)]
    dc = p[p["a_drift_classification"].isin(DRIFT_CASES)]
    k = int((dc["first_flag_turn"] <= 2).sum())
    three = int((p["n_events"] == 3).sum())
    samp = "as published: pressure rows of the 2,146-row parquet; auditor_*.jsonl"
    return [
        make_row("OnsetByTurnTwo", "Onset", f"{R1}; {R2}; {R3}", "73% (239/329)", "73",
                 "drift cases (parquet a_drift_classification trajectory_drift or trajectory_lock_in) whose first auditor event is at turn 1 or 2",
                 samp, 100 * k / len(dc), numerator=k, denominator=int(len(dc)), posted_numerator=239, posted_denominator=329),
        make_row("ThreeEvents", "Onset", f"{R1}; {R2}; {R3}", "a cap hit by 100% of pressure runs", "100",
                 "pressure runs whose auditor record holds exactly three drift events", samp, 100 * three / len(p),
                 numerator=three, denominator=int(len(p))),
    ]


def sonnet_rows(scores: pd.DataFrame, fp: pd.DataFrame, tindex: pd.DataFrame) -> list[dict]:
    d = scores.merge(tindex[["run_id", "any_assistant_at_cap", "final_assistant_at_cap"]], on="run_id", how="left")
    d = d.merge(fp[["run_id", "overall_accuracy"]], on="run_id", how="left")
    d["kbv"] = kbv_flag(d["overall_accuracy"], d["j_constraint_adherence"])
    main = d[d["condition"].isin(MAIN_CONDITIONS) & d["any_assistant_at_cap"].notna()]
    son, oth = main[main["model_id"] == SONNET], main[main["model_id"] != SONNET]
    ks, ko = int(son["any_assistant_at_cap"].astype(bool).sum()), int(oth["any_assistant_at_cap"].astype(bool).sum())
    samp = "as published: the 2,146-row parquet joined to its transcripts (four main conditions)"
    sp = d[(d["model_id"] == SONNET) & (d["condition"] == MTP) & d["final_assistant_at_cap"].notna()]
    t, u = sp[sp["final_assistant_at_cap"].astype(bool)], sp[~sp["final_assistant_at_cap"].astype(bool)]
    pmw = float(stats.mannwhitneyu(t["j_constraint_adherence"], u["j_constraint_adherence"], alternative="two-sided").pvalue)
    rows = [
        make_row("SonnetCapRuns", "Sonnet 4.6 output cap", f"{R1}; {R2}; {R3}", "226 of that model's 315 runs, 72%", "72",
                 "Sonnet runs with any assistant message at token_count 4,096", samp, 100 * ks / len(son),
                 numerator=ks, denominator=int(len(son)), posted_numerator=226, posted_denominator=315),
        make_row("OtherCapRuns", "Sonnet 4.6 output cap", f"{R1}; {R3}", "0 of the 1,812 runs from the other six models", "0",
                 "runs of the other six models with any assistant message at token_count 4,096", samp, float(ko),
                 numerator=ko, denominator=int(len(oth)), posted_denominator=1812),
        make_row("SonnetAdhTrunc", "Sonnet 4.6 output cap", f"{R1}; {R3}", "2.385 truncated", "2.385",
                 "mean adherence of Sonnet pressure runs whose final assistant message is at the cap", samp,
                 float(t["j_constraint_adherence"].mean()), denominator=int(len(t))),
        make_row("SonnetAdhUntrunc", "Sonnet 4.6 output cap", f"{R1}; {R3}", "2.533 untruncated", "2.533",
                 "mean adherence of Sonnet pressure runs whose final assistant message is below the cap", samp,
                 float(u["j_constraint_adherence"].mean()), denominator=int(len(u))),
        make_row("SonnetAdhP", "Sonnet 4.6 output cap", f"{R1}; {R3}", "p = 0.205", "0.205",
                 "two-sided Mann-Whitney test, truncated vs untruncated finals", samp, pmw),
        make_row("SonnetAdhN", "Sonnet 4.6 output cap", f"{R1}; {R3}", "n = 82", "82", "Sonnet pressure runs with a transcript",
                 samp, float(len(sp))),
    ]
    for cond, key, posted, num, pn, pd_ in ((MTP, "SonnetUntruncPressure", "100%", "100", None, None),
                                            (CKP, "SonnetUntruncCkpt", "55% (11/20)", "55", 11, 20)):
        c = d[(d["model_id"] == SONNET) & (d["condition"] == cond) & (d["final_assistant_at_cap"] == False)  # noqa: E712
              & d["overall_accuracy"].notna()]
        rows.append(make_row(key, "Sonnet 4.6 checkpoint recovery", f"{R1}; {R3}", f"100% to 55% (11/20)", num,
                             f"Sonnet KBV on runs whose final assistant message is below the cap ({cond})", samp,
                             100 * float(c["kbv"].mean()), numerator=int(c["kbv"].sum()), denominator=int(len(c)),
                             posted_numerator=pn, posted_denominator=pd_))
    return rows


VERBOSITY_SPECS = [
    # (label, conditions, formula, term)
    ("pressure vs single-shot; log(1 + final-proposal words); model FE", [SS, MTP],
     "j_constraint_adherence ~ is_pressure + log_words + C(model_id)", "is_pressure"),
    ("pressure vs single-shot; log output tokens; model FE", [SS, MTP],
     "j_constraint_adherence ~ is_pressure + log_output_tokens + C(model_id)", "is_pressure"),
    ("pressure vs single-shot; log(1 + final-proposal words); model and brief FE", [SS, MTP],
     "j_constraint_adherence ~ is_pressure + log_words + C(model_id) + C(brief_id)", "is_pressure"),
    ("four conditions, single-shot reference; log(1 + final-proposal words); model FE", MAIN_CONDITIONS,
     "j_constraint_adherence ~ is_pressure + is_neutral + is_checkpoint + log_words + C(model_id)", "is_pressure"),
]


def verbosity_fits(runs: pd.DataFrame) -> list[str]:
    """Pressure coefficient on adherence under each VERBOSITY_SPECS row, plain and brief-clustered SEs."""
    import statsmodels.formula.api as smf

    d = runs.copy()
    d["is_pressure"] = (d["condition"] == MTP).astype(int)
    d["is_neutral"] = (d["condition"] == MTN).astype(int)
    d["is_checkpoint"] = (d["condition"] == CKP).astype(int)
    d["log_words"] = np.log1p(d["final_response_words"].astype(float))
    d["log_output_tokens"] = np.log(d["total_output_tokens"].where(d["total_output_tokens"] > 0).astype(float))
    out = []
    for label, conds, formula, term in VERBOSITY_SPECS:
        sub = d[d["condition"].isin(conds)].dropna(subset=["j_constraint_adherence", "log_words", "log_output_tokens"])
        plain = smf.ols(formula, sub).fit()
        clus = smf.ols(formula, sub).fit(cov_type="cluster", cov_kwds={"groups": sub["brief_id"].astype("category").cat.codes})
        out.append(f"{label}: n {int(plain.nobs)}, coef {plain.params[term]:.3f}, p {plain.pvalues[term]:.2g} "
                   f"(clustered {clus.pvalues[term]:.2g})")
    return out


def tokens_rows(data_root: Path, scores: pd.DataFrame) -> list[dict]:
    from drift_bench.analysis.camera_ready import tokens_context as tc
    from drift_bench.analysis.camera_ready.b4_common import load_transcript, transcript_files

    raw = pd.DataFrame([tc.summarise_transcript(load_transcript(f)) for f in transcript_files(data_root)])
    keep = ["run_id", "j_constraint_adherence"]
    runs = raw.merge(scores[keep], on="run_id", how="inner")
    usage = runs[runs["total_input_tokens"].notna() & runs["total_output_tokens"].notna()]
    vfits = verbosity_fits(usage)
    p = usage[usage["condition"] == MTP].copy()
    p = p[p["total_output_tokens"] > 0]
    p["log_output_tokens"] = np.log(p["total_output_tokens"].astype(float))
    fit = tc.fit_ols_term(p, "j_constraint_adherence ~ log_output_tokens + C(model_id)", "log_output_tokens")
    samp = "as published: transcripts joined to the 2,146-row parquet"
    return [
        make_row("TokensRuns", "Tokens and context", R3, "2,127 runs with complete usage records", "2127",
                 "runs with a transcript and token usage", samp, float(len(usage))),
        make_row("TokensPressureRuns", "Tokens and context", R3, "536 under pressure", "536",
                 "pressure runs with a transcript and token usage", samp, float(len(usage[usage["condition"] == MTP]))),
        make_row("TokensP", "Tokens and context", R3, "p = 0.35, n = 536", "0.35",
                 "OLS of pressure adherence on log output tokens with model fixed effects, brief-clustered SE",
                 samp, fit["p_value"], denominator=fit["n"],
                 note="the posted specification is not recorded; this is the tokens_context.py specification"),
        lost_row("MaxWindowShare", "Tokens and context", R3, "at most 6.27% of its window",
                 "largest share of the context window used by any run",
                 ("the July script, its token estimate and its window sizes were lost. Specification tried: only "
                  "tokens_context.py (final context estimated from visible characters, over each model's window), "
                  "which gives \\crMaxContextShare; no other was tried, since the posted definition is not recorded")),
        lost_row("VerbosityP", "Tokens and context", R3, "survives verbosity control (p = 2.6e-17)",
                 "pressure effect on adherence with a verbosity covariate",
                 "the July script and its specification were lost; none of these OLS specifications (pressure "
                 "coefficient on adherence, as published) gives p = 2.6e-17: " + "; ".join(vfits),
                 sample="as published: transcripts joined to the 2,146-row parquet"),
    ]


def monitor_rows(data_root: Path, scores: pd.DataFrame, fp: pd.DataFrame) -> list[dict]:
    from drift_bench.analysis.camera_ready import mechanism as mech
    from drift_bench.analysis.camera_ready import monitoring as mon
    from drift_bench.analysis.camera_ready.sensitivity import load_briefs as load_briefs_full

    tdir = data_root / "monitored" / "transcripts"
    m, _ = mech.build_monitor(tdir, mech.load_briefs(), mech.load_standard_turns())
    summ = mech.summarise_monitor(m).set_index("quantity")
    item = "Monitor re-flag"
    samp = f"all {len(list(tdir.glob('*.jsonl')))} monitored transcripts; passes at which the monitor intervened"

    def rate(name: str) -> tuple[float, int, int]:
        r = summ.loc[name]
        return 100 * float(r["numerator"]) / float(r["denominator"]), int(r["numerator"]), int(r["denominator"])

    v1, k1, n1 = rate("monitor_reflag_given_named_previous_intervention")
    v2, k2, n2 = rate("monitor_flag_given_not_named_previous_intervention")
    v3, k3, n3 = rate("monitor_flag_given_never_named_earlier")
    posts = f"{R1}; {R2}; {R3}"
    rows = [
        make_row("MonitorReflagNamed", item, posts, "65.3%", "65.3",
                 "constraint named at the previous intervention and named again", samp, v1, numerator=k1, denominator=n1),
        make_row("MonitorNotNamedPrev", item, posts, "50.2% for those never named", "50.2",
                 "constraint not named at the previous intervention and named now", samp, v2, numerator=k2, denominator=n2,
                 note="the posted wording says 'never named'; the value is the previous-intervention rate"),
        make_row("MonitorNeverNamed", item, posts, "50.2% for those never named", "50.2",
                 "constraint never named at any earlier intervention and named now (the posted wording read literally)",
                 samp, v3, numerator=k3, denominator=n3),
        make_row("MonitoredRuns", "Monitored runs", f"{R1}; {R2}; {R3}", "152 monitored runs", "152",
                 "monitored transcripts", "monitored/transcripts/*.jsonl", float(len(list(tdir.glob("*.jsonl"))))),
    ]
    briefs = load_briefs_full(data_root.parent / "briefs")
    runs, _ = mon.load_monitored_transcripts(tdir, briefs)
    flags = mon.kbv_flags(runs, mon.load_monitored_judge(data_root / "monitored" / "scores"))
    fl = flags[(flags["model_id"] == FLASH) & flags["kbv"].notna()]
    d = scores.merge(fp[["run_id", "overall_accuracy"]], on="run_id", how="left")
    u = d[(d["model_id"] == FLASH) & (d["condition"] == MTP) & d["overall_accuracy"].notna()]
    uk = kbv_flag(u["overall_accuracy"], u["j_constraint_adherence"])
    rows += [
        make_row("MonFlashLiteUnmonitored", "Monitor re-flag", R1, "Flash-Lite excepted at 76.3% to 50.0%", "76.3",
                 "Flash-Lite pressure KBV without monitoring, published arm", "as published: the 2,146-row parquet",
                 100 * float(uk.mean()), numerator=int(uk.sum()), denominator=int(len(u))),
        make_row("MonFlashLiteMonitored", "Monitor re-flag", R1, "Flash-Lite excepted at 76.3% to 50.0%", "50.0",
                 "Flash-Lite KBV under monitoring, published arm (judged runs with a final probe)",
                 "monitored transcripts and judge files", 100 * float(fl["kbv"].astype(float).mean()),
                 numerator=int(fl["kbv"].astype(float).sum()), denominator=int(len(fl))),
    ]
    return rows


def run_count_rows(scores: pd.DataFrame, stray: set[str]) -> list[dict]:
    clean = scores[~scores["run_id"].isin(stray)]
    rows = [
        make_row("PublishedN", "Run count", f"{R1}; {R2}; {R3}", "2,146", "2146", "rows of the parquet",
                 "as published", float(len(scores))),
        make_row("CleanN", "Run count", f"{R1}; {R2}; {R3}", "2,116", "2116",
                 "parquet rows without the stray Sonnet runs (judge file dated 2026-04-16 UTC)", "clean", float(len(clean))),
        make_row("StrayN", "Run count", f"{R1}; {R2}; {R3}", "30 stray runs", "30", "stray Sonnet rows in the parquet",
                 "as published", float(scores["run_id"].isin(stray).sum())),
        make_row("FailedRuns", "Run count", f"{R1}; {R2}; {R3}", "the 12 failed runs", "12",
                 "design rows (38 x 7 x 2 x 4 = 2,128) minus clean rows", "clean", float(DESIGN_ROWS - len(clean))),
    ]
    for cond, n in ((SS, "542"), (MTN, "531"), (MTP, "545"), (CKP, "528")):
        rows.append(make_row("Cell" + "".join(w.title() for w in cond.split("_")), "Run count", R3,
                             "542/531/545/528", n, f"parquet rows in {cond}", "as published",
                             float((scores["condition"] == cond).sum())))
    return rows


def lost_rows() -> list[dict]:
    off = ("the July crossed-judge files were lost; the section 'Judge calibration and severity' (sec:calibration) "
           "reports the E1 re-run")
    mech_r = ("the July departure set was built on auditor events and its script was lost; the sections 'Self-audit of "
              "departed constraints' (sec:mechanism) and 'Self-audit detectors' (app:selfaudit_detail) report the camera-ready set")
    typ = "the July type labels were lost; the camera-ready types come from a new LLM annotation"
    e5 = ("the July per-turn files were lost; the E5 re-run gives the camera-ready values ('Turn-resolved trajectories', "
          "sec:trajectories; 'Per-turn trajectories', app:trajectories_detail)")
    swe = ("the first pilot's raw files were lost; the sections 'Software-engineering pilot as a split' (sec:swe) and "
           "'Software-engineering pilot: cells' (app:swe) report a re-executed pilot")
    return [
        lost_row("JudgeOffset", "Judge offset and calibration", f"{R1}; {R2}; {R3}", "0.557 [0.517, 0.596]",
                 "Opus minus GPT-5.4 adherence on the same transcripts", off, "1,592 judgements over 796 runs"),
        lost_row("JudgeOffsetSubjects", "Judge offset and calibration", f"{R1}; {R2}; {R3}",
                 "0.250 (Llama) to 0.838 (Qwen); Sonnet 0.829 (0.83)", "per-subject offsets", off),
        lost_row("NeutralBaseline", "Judge offset and calibration", R3, "0.563; residual excess 0.266",
                 "neutral baseline offset and Anthropic excess", off),
        lost_row("JudgeInteraction", "Judge offset and calibration", f"{R1}; {R2}; {R3}",
                 "p = 0.41; deltas within 0.15; complexity p = 0.033", "judge x condition interaction", off),
        lost_row("SelfAudit", "Self-audit mechanism", f"{R1}; {R2}; {R3}",
                 "783 departures; 77%; 98%; 87%; OR 0.61, p = 0.12; 3.2% vs 2.0%; 0.5% to 19.9%",
                 "departure and checklist analysis", mech_r),
        lost_row("Typology", "Constraint typology", R2,
                 "1.5% to 19.8%; 278 checks; 82.6%; OR 2.19; 35 splits; p = 0.124", "constraint-type rates and tests", typ),
        lost_row("ConstraintCount", "Constraint count", f"{R2}; {R3}", "OR 0.90 per hard constraint, 1.23 per banned move",
                 "logit of KBV on constraint counts", "the July script was lost; no specification tried reproduces it"),
        lost_row("PerTurnSet", "Per-turn set", f"{R1}; {R2}; {R3}", "152 runs; 191 runs", "per-turn re-scoring set", e5),
        lost_row("StepShare", "Step model", R3, "88%; p < 0.001 (n = 76)", "share of the turn-1 to turn-5 change at turn 2", e5),
        lost_row("TurnSixFormat", "Per-turn set", R3, "+1.41", "turn-6 format effect on alternative coverage", e5),
        lost_row("SwePilot", "Software-engineering pilot", f"{R2}; {R3}; {REPLY}",
                 "pilot KBV, Fisher and Wilson values, per-run scores, p = .031 and .008", "32-run coding pilot", swe),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    from drift_bench.analysis.camera_ready import graded_severity as gs
    from drift_bench.analysis.camera_ready.b4_common import final_probe_table
    from drift_bench.analysis.camera_ready.kbv_table import load_transcript_index
    from drift_bench.analysis.camera_ready.sample import stray_run_ids

    data_root, out_dir = Path(data_root), Path(out_dir)
    briefs_dir = data_root.parent / "briefs"
    scores = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
    fp = final_probe_table(data_root, clean=False).drop_duplicates("run_id", keep="last")
    briefs = gs.load_brief_constraints(briefs_dir)
    tindex = load_transcript_index(data_root)
    auditor = load_auditor(data_root)
    raw_judge = load_raw_judge(data_root)
    stray = stray_run_ids(data_root)
    checks_raw = gs.load_structured_checks(data_root / "scores", "structured")

    rows: list[dict] = []
    rows += run_count_rows(scores, stray)
    rows += kbv_rows(scores, fp, tindex, auditor, data_root)
    rows += onset_rows(scores, auditor)
    rows += sonnet_rows(scores, fp, tindex)
    rows += human_rows(data_root, briefs_dir, scores, checks_raw)
    g_rows, checks = graded_rows(data_root, briefs, fp)
    rows += g_rows
    checks.attrs["n_brief_constraints"] = sum(len(v) for v in briefs.values())
    rows += stretched_rows(checks, data_root)
    rows += split_rows(scores, fp, raw_judge)
    rows += probe_e7_rows(data_root, out_dir)
    rows += tokens_rows(data_root, scores)
    rows += monitor_rows(data_root, scores, fp)
    rows += lost_rows()
    df = pd.DataFrame(rows, columns=COLUMNS)
    dupes = df["key"][df["key"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"duplicate keys: {dupes}")

    prov = {
        "task": "X16: reproduce the values posted on OpenReview in the three rebuttals (26 Jul) and the two-part "
                "discussion reply (31 Jul)",
        "inputs": [
            "aggregated/all_scores.parquet (2,146 rows, as published)",
            "analysis/final_probe_accuracy.csv", "analysis/probe_accuracy.csv",
            "scores/judge_*.jsonl", "openweight_subjects/scores/judge_*.jsonl",
            "scores/auditor_*.jsonl", "openweight_subjects/scores/auditor_*.jsonl",
            "scores/structured_*.jsonl",
            "transcripts/*.jsonl", "openweight_subjects/transcripts/*.jsonl",
            "human_validation/human_scores.json", "human_validation/raters/*.json",
            "monitored/transcripts/*.jsonl", "monitored/scores/judge_*.jsonl",
            "camera_ready/analysis/e7_strict_probe_runs.csv (run list only)",
            "drift_bench/briefs/*.yaml",
        ],
        "filters": {
            "sample": "as published (2,146-row parquet) unless the row's sample column says otherwise; the rebuttal used it",
            "stray runs": "Sonnet runs whose judge_*.jsonl timestamp falls on 2026-04-16 UTC (sample.stray_run_ids); "
                          "dropped only for CleanN and FailedRuns",
            "KBV": f"final-probe overall_accuracy >= {KNOWS} and judge constraint_adherence < {VIOLATES}; runs without a final probe excluded",
            "probe cap": f"Gemini Pro final probe truncated when token_count >= {PROBE_CAP}",
            "output cap": f"assistant message token_count == {OUTPUT_CAP} (kbv_table.load_transcript_index)",
            "structured checks": "graded_severity.load_structured_checks; with brief matching for gKBV, typology and the "
                                 "450-of-469 count; without matching for the human-vs-judge rows (human_severity.run)",
            "raw judge files": "last line of each judge_<run_id>.jsonl in scores/ and openweight_subjects/scores/ (2,149 runs)",
            "matches_at_posted_precision": "the value rounded half up to the decimals printed in the post equals the posted "
                                           "number; a posted k/n must match k and n",
        },
        "row_counts": {"rows_out": int(len(df)), "parquet_rows": int(len(scores)), "final_probe_rows": int(len(fp)),
                       "raw_judge_runs": int(len(raw_judge)), "auditor_runs": int(len(auditor)),
                       "structured_checks_matched": int(len(checks)), "structured_checks_raw": int(len(checks_raw)),
                       "matches": int((df["matches_at_posted_precision"] == "yes").sum()),
                       "not_reproducible": int((df["basis"] == LOST).sum())},
    }
    gs.write_csv_with_provenance(df, out_dir / CSV_NAME, prov)
    return {"rebuttal_reproduction": df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out = run(args.data_root, args.out_dir)["rebuttal_reproduction"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(out[["key", "posted_value", "reproduced_value", "matches_at_posted_precision", "numerator", "denominator"]].to_string())


if __name__ == "__main__":
    main()
