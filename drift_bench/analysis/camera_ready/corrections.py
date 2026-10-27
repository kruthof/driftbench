"""Corrections ledger: every corrected number with its rule and denominator.

Plan items B1, B2, B3, B5, B8, B9, B12, B14, B15, B16, B17 and C15. One CSV,
``corrections_ledger.csv``, with columns
``id, quantity, published_value, recomputed_value, rule, numerator,
denominator, source_files, note``. ``note`` carries what the paper must say
about a row, or a caveat on the test behind it.

Published values are copied from the camera-ready plan (section 2; not part
of the public repository).
They are sanity targets, not inputs to any computation.

Run: ``python -m drift_bench.analysis.camera_ready.corrections``
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready import failed_runs
from drift_bench.analysis.camera_ready.kbv_table import (
    ADHERENCE_THRESHOLD,
    CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    GEMINI_PRO,
    GEMINI_PRO_PROBE_CAP,
    MAIN_CONDITIONS,
    PRESSURE,
    SONNET,
    SONNET_OUTPUT_CAP,
    build_samples,
    kbv_flags,
    load_final_probes,
    load_judge_timestamps,
    load_score_records,
    load_scores,
    load_transcript_index,
    read_jsonl_first,
    truncated_probe_mask,
    waive_truncated_probe_recall,
    wilson_ci,
    write_csv_with_provenance,
)

DRIFT_CLASSES = ("trajectory_drift", "trajectory_lock_in")
MULTI_TURN_CONDITIONS = [c for c in MAIN_CONDITIONS if c != "single_shot"]
LEDGER_COLUMNS = [
    "id",
    "quantity",
    "published_value",
    "recomputed_value",
    "rule",
    "numerator",
    "denominator",
    "source_files",
    "note",
]

# What the paper must say about the Sonnet truncation test (B16).
B16_PAPER_WORDING = (
    "the paper prints the final-turn rule as the adherence test: 52 truncated vs 30 untruncated, "
    "Mann-Whitney U, two-sided, tie-corrected, p = 0.205; wording must be "
    "'no detectable difference at n = 30', never 'does not move'"
)
B16_ANY_TURN_NOTE = (
    "the any-turn split leaves only 4-5 untruncated runs, so this test is uninformative; "
    "the any-turn count (226 of 315) is reported only as the truncation prevalence"
)


def _entry(id_, quantity, published, recomputed, rule, numerator=None, denominator=None, sources="", note="") -> dict:
    return {
        "id": id_,
        "quantity": quantity,
        "published_value": published,
        "recomputed_value": recomputed,
        "rule": rule,
        "numerator": numerator,
        "denominator": denominator,
        "source_files": sources,
        "note": note,
    }


def _pct(k, n) -> float:
    return 100.0 * k / n if n else math.nan


# ---------------------------------------------------------------------------
# B1 and C15: auditor drift events
# ---------------------------------------------------------------------------


def first_event_turn(drift_events) -> int | None:
    """The smallest integer ``turn`` in a list of auditor drift events."""
    turns = []
    for ev in drift_events or []:
        if isinstance(ev, dict) and isinstance(ev.get("turn"), (int, float)) and not isinstance(ev.get("turn"), bool):
            turns.append(int(ev["turn"]))
    return min(turns) if turns else None


def auditor_event_table(auditor: pd.DataFrame) -> pd.DataFrame:
    """``run_id, n_events, first_turn`` from the raw auditor records."""
    out = auditor[["run_id", "condition", "model_id", "drift_classification"]].copy()
    out["n_events"] = auditor["drift_events"].apply(lambda e: len(e) if isinstance(e, list) else 0)
    out["first_turn"] = auditor["drift_events"].apply(first_event_turn)
    return out


def drift_onset_shares(events: pd.DataFrame, scores: pd.DataFrame, by_turns=(2, 3)) -> list[dict]:
    """Share of pressure drift cases (parquet rows) with a first event by turn k."""
    pressure_ids = set(scores.loc[scores["condition"] == PRESSURE, "run_id"])
    sub = events[events["run_id"].isin(pressure_ids) & events["drift_classification"].isin(DRIFT_CLASSES)]
    n = int(len(sub))
    rows = []
    for k in by_turns:
        num = int((sub["first_turn"] <= k).sum())
        rows.append(
            _entry(
                "B1",
                f"share of pressure drift cases with first auditor event by turn {k}",
                {2: "74%", 3: None}[k] if k in (2, 3) else None,
                round(_pct(num, n), 1),
                f"pressure rows in the parquet with a_drift_classification in {DRIFT_CLASSES}; "
                "first event = min(turn) over auditor drift_events",
                num,
                n,
                "scores/auditor_*.jsonl; openweight_subjects/scores/auditor_*.jsonl; aggregated/all_scores.parquet",
            )
        )
    return rows


def exact_three_events(events: pd.DataFrame, scores: pd.DataFrame) -> dict:
    pressure_ids = set(scores.loc[scores["condition"] == PRESSURE, "run_id"])
    sub = events[events["run_id"].isin(pressure_ids)]
    num = int((sub["n_events"] == 3).sum())
    return _entry(
        "C15",
        "share of pressure runs where the auditor emitted exactly 3 drift events",
        "100%",
        round(_pct(num, len(sub)), 1),
        "len(drift_events) == 3 over parquet pressure rows",
        num,
        int(len(sub)),
        "scores/auditor_*.jsonl; openweight_subjects/scores/auditor_*.jsonl",
    )


# ---------------------------------------------------------------------------
# B3: Sonnet checkpoint recovery
# ---------------------------------------------------------------------------


def sonnet_checkpoint_recovery(flags: pd.DataFrame, sample: str) -> list[dict]:
    """Sonnet KBV under pressure vs checkpointed, three truncation rules."""
    rules = {
        "all runs": pd.Series(True, index=flags.index),
        "untruncated final (final assistant token_count < cap)": ~flags["final_assistant_at_cap"].fillna(False).astype(bool),
        "untruncated any turn (no assistant token_count == cap)": ~flags["any_assistant_at_cap"].fillna(False).astype(bool),
    }
    published = {
        ("all runs", PRESSURE): "99%",
        ("all runs", CHECKPOINT): "87%",
        ("untruncated final (final assistant token_count < cap)", PRESSURE): "100%",
        ("untruncated final (final assistant token_count < cap)", CHECKPOINT): "55% (11/20)",
    }
    rows = []
    sonnet = flags[(flags["model_id"] == SONNET) & flags["has_final_probe"]]
    for rule_name, mask in rules.items():
        for condition in (PRESSURE, CHECKPOINT):
            g = sonnet[(sonnet["condition"] == condition) & mask.loc[sonnet.index]]
            k, n = int(g["kbv"].fillna(False).sum()), int(len(g))
            rows.append(
                _entry(
                    "B3",
                    f"Sonnet KBV, {condition}, {rule_name}, sample={sample}",
                    published.get((rule_name, condition)),
                    round(_pct(k, n), 1),
                    f"KBV among Sonnet runs with a final probe; cap = {SONNET_OUTPUT_CAP}; {rule_name}",
                    k,
                    n,
                    "aggregated/all_scores.parquet; analysis/final_probe_accuracy.csv; transcripts/*.jsonl",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# B8, B9, B17 (human part): human validation
# ---------------------------------------------------------------------------


def rater_violation_counts(raters_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raters_dir.glob("*.json")):
        with open(path) as fh:
            data = json.load(fh)
        rows.append(
            {
                "file": path.name,
                "rater_id": next(iter(data.values()))["rater_id"] if data else None,
                "n_runs": len(data),
                "violated": int(sum(v.get("violated_count", 0) for v in data.values())),
                "stretched": int(sum(v.get("stretched_count", 0) for v in data.values())),
            }
        )
    return pd.DataFrame(rows)


def human_condition_means(human: pd.DataFrame, column: str) -> dict:
    """Mean of ``column`` for single-shot and pressure plus the delta."""
    ss = human.loc[human["condition"] == "single_shot", column]
    pr = human.loc[human["condition"] == PRESSURE, column]
    return {"single_shot": float(ss.mean()), "pressure": float(pr.mean()), "delta": float(pr.mean() - ss.mean()), "n_ss": int(len(ss)), "n_pr": int(len(pr))}


def human_label_counts(human: pd.DataFrame) -> Counter:
    return Counter(str(x).lower() for labels in human["constraint_statuses"] for x in labels)


# ---------------------------------------------------------------------------
# B16: Sonnet truncation
# ---------------------------------------------------------------------------


def truncation_by_condition(transcripts: pd.DataFrame, model_id: str = SONNET) -> pd.DataFrame:
    sub = transcripts[(transcripts["model_id"] == model_id) & transcripts["condition"].isin(MAIN_CONDITIONS)]
    out = sub.groupby("condition").agg(n=("run_id", "size"), truncated_any_turn=("any_assistant_at_cap", "sum"), truncated_final=("final_assistant_at_cap", "sum")).reset_index()
    total = pd.DataFrame([{"condition": "all_main", "n": int(len(sub)), "truncated_any_turn": int(sub["any_assistant_at_cap"].sum()), "truncated_final": int(sub["final_assistant_at_cap"].sum())}])
    return pd.concat([out, total], ignore_index=True)


def truncation_adherence_test(
    flags: pd.DataFrame, condition: str = PRESSURE, column: str = "any_assistant_at_cap"
) -> dict:
    """Mean adherence for truncated vs untruncated Sonnet runs, Mann-Whitney U.

    ``column`` picks the truncation rule: ``any_assistant_at_cap`` (any turn)
    or ``final_assistant_at_cap`` (the judged final message only).
    """
    g = flags[(flags["model_id"] == SONNET) & (flags["condition"] == condition) & flags[column].notna()]
    trunc = g[g[column].astype(bool)]["j_constraint_adherence"]
    untr = g[~g[column].astype(bool)]["j_constraint_adherence"]
    if len(trunc) and len(untr):
        # Normal approximation with tie and continuity correction. scipy
        # picks this method itself whenever ties exist; naming it keeps the
        # rule explicit. Integer scores always tie.
        p = float(stats.mannwhitneyu(trunc, untr, alternative="two-sided", method="asymptotic").pvalue)
    else:
        p = math.nan
    return {"mean_truncated": float(trunc.mean()) if len(trunc) else math.nan, "mean_untruncated": float(untr.mean()) if len(untr) else math.nan, "n_truncated": int(len(trunc)), "n_untruncated": int(len(untr)), "p": p}


# ---------------------------------------------------------------------------
# B2: Gemini Pro KBV under the probe cap
# ---------------------------------------------------------------------------


def gemini_pro_probe_truncation(
    flags: pd.DataFrame,
    conditions: list[str] = MULTI_TURN_CONDITIONS,
    cap: int = GEMINI_PRO_PROBE_CAP,
) -> dict:
    """Final-probe truncation for Gemini Pro over the multi-turn conditions.

    ``flags`` must carry ``has_final_probe, final_probe_token_count,
    overall_accuracy``. Returns the probe count, the number at or above the
    cap, the number one token short of it, and how many untruncated probes
    score full recall.
    """
    g = flags[(flags["model_id"] == GEMINI_PRO) & flags["condition"].isin(conditions) & flags["has_final_probe"]]
    tok = pd.to_numeric(g["final_probe_token_count"], errors="coerce")
    untr = g[tok < cap]
    return {
        "n_probes": int(len(g)),
        "n_capped": int((tok >= cap).sum()),
        "n_one_below_cap": int((tok == cap - 1).sum()),
        "n_untruncated": int(len(untr)),
        "n_untruncated_full_recall": int((untr["overall_accuracy"] >= 1.0).sum()),
    }


def gemini_pro_kbv_variants(flags: pd.DataFrame, condition: str = PRESSURE, cap: int = GEMINI_PRO_PROBE_CAP) -> dict:
    """Gemini Pro KBV under ``condition`` for three treatments of a capped probe.

    ``waived``: a capped probe passes the recall filter (the July rule).
    ``as_is``: capped probes are graded on what they contain.
    ``excluded``: capped runs leave the denominator.
    Each value is ``(kbv_count, n, ci_low, ci_high)``.
    """
    g = flags[(flags["model_id"] == GEMINI_PRO) & (flags["condition"] == condition) & flags["has_final_probe"]]
    capped = truncated_probe_mask(g, GEMINI_PRO, cap)
    waived = waive_truncated_probe_recall(g, GEMINI_PRO, cap)
    out = {}
    for name, sub in (("waived", waived), ("as_is", g), ("excluded", g[~capped])):
        k, n = int(sub["kbv"].fillna(False).sum()), int(len(sub))
        lo, hi = wilson_ci(k, n)
        out[name] = (k, n, lo, hi)
    out["n_capped"] = int(capped.sum())
    return out


# ---------------------------------------------------------------------------
# B17: structured judge checks
# ---------------------------------------------------------------------------


def structured_status_counts(scores_dir: Path) -> Counter:
    counts: Counter = Counter()
    for path in sorted(scores_dir.glob("structured_*.jsonl")):
        rec = read_jsonl_first(path)
        for check in rec.get("constraint_checks") or []:
            counts[str(check.get("status", "")).lower()] += 1
    return counts


# ---------------------------------------------------------------------------
# B12: monitored counts
# ---------------------------------------------------------------------------


def monitored_counts(data_root: Path) -> dict:
    m = data_root / "monitored"
    agg = m / "aggregated" / "scores.parquet"
    return {
        "transcripts": len(list((m / "transcripts").glob("*.jsonl"))),
        "judge_files": len(list((m / "scores").glob("judge_*.jsonl"))),
        "auditor_files": len(list((m / "scores").glob("auditor_*.jsonl"))),
        "aggregated_rows": int(len(pd.read_parquet(agg))) if agg.exists() else None,
    }


# ---------------------------------------------------------------------------
# B14: Sonnet over-sampling
# ---------------------------------------------------------------------------


def sonnet_cell_counts(scores: pd.DataFrame, transcripts: pd.DataFrame) -> dict:
    s = scores[scores["model_id"] == SONNET]
    cells = s.groupby(["brief_id", "condition"]).size()
    per_condition = s.groupby("condition").size().to_dict()
    t = transcripts[(transcripts["model_id"] == SONNET) & transcripts["condition"].isin(MAIN_CONDITIONS)]
    rep_dups = t.groupby(["brief_id", "condition", "repetition"]).size()
    return {
        "runs_per_condition": per_condition,
        "cells_with_3_runs": int((cells == 3).sum()),
        "n_cells": int(len(cells)),
        "cell_size_counts": {int(k): int(v) for k, v in cells.value_counts().items()},
        "transcript_cells_with_duplicate_repetition": int((rep_dups > 1).sum()),
    }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_ledger(data_root: Path) -> tuple[pd.DataFrame, dict]:
    data_root = Path(data_root)
    scores = load_scores(data_root)
    judge_ts = load_judge_timestamps(data_root)
    transcripts = load_transcript_index(data_root)
    final_probes = load_final_probes(data_root)
    samples, log = build_samples(scores, judge_ts, transcripts)
    flags = {name: kbv_flags(df, final_probes) for name, df in samples.items() if name in ("as_published", "clean")}

    rows: list[dict] = []
    src_scores = "aggregated/all_scores.parquet"

    # B1, C15
    auditor = load_score_records(data_root, "auditor")
    events = auditor_event_table(auditor)
    rows += drift_onset_shares(events, scores)
    rows.append(exact_three_events(events, scores))

    # B2: Gemini Pro KBV under pressure. Gemini Pro has no stray rows, so the
    # clean and as-published samples hold the same 76 runs.
    gp = gemini_pro_kbv_variants(flags["clean"])
    pt = gemini_pro_probe_truncation(flags["clean"])
    gp_src = src_scores + "; analysis/final_probe_accuracy.csv; transcripts/*.jsonl"
    gp_rule = (
        f"{gp['n_capped']} of {gp['waived'][1]} final pressure probes hit the {GEMINI_PRO_PROBE_CAP:,}-token cap; "
        f"all {pt['n_untruncated_full_recall']} of {pt['n_untruncated']} untruncated Gemini Pro probes across the three multi-turn "
        "conditions score 100% recall, so a capped probe is counted as passing the recall filter"
    )
    k, n, lo, hi = gp["waived"]
    rows.append(_entry("B2", "Gemini Pro KBV, pressure, capped probes waived (published rule), sample=clean", "25.0% [16.6, 35.8] (19/76)", round(_pct(k, n), 1), gp_rule, k, n, gp_src, f"matches the July kbv_summary.csv violates_rate; Wilson CI [{_pct(lo, 1):.1f}, {_pct(hi, 1):.1f}]; knows is imputed for the {gp['n_capped']} capped runs"))
    k, n, lo, hi = gp["as_is"]
    rows.append(_entry("B2", "Gemini Pro KBV, pressure, capped probes graded as-is, sample=clean", None, round(_pct(k, n), 1), f"overall_accuracy >= 0.8 on the probe text as returned, even when it hit the {GEMINI_PRO_PROBE_CAP}-token cap", k, n, gp_src, f"Wilson CI [{_pct(lo, 1):.1f}, {_pct(hi, 1):.1f}]"))
    k, n, lo, hi = gp["excluded"]
    rows.append(_entry("B2", "Gemini Pro KBV, pressure, capped runs excluded, sample=clean_untruncated_probes", None, round(_pct(k, n), 1), f"drop runs whose final probe has token_count >= {GEMINI_PRO_PROBE_CAP}; KBV over the rest", k, n, gp_src, f"Wilson CI [{_pct(lo, 1):.1f}, {_pct(hi, 1):.1f}]; n = {n} is too small to support a rate"))
    rows.append(
        _entry(
            "B2",
            "Gemini Pro final probes truncated, three multi-turn conditions",
            "73.2% (167/228)",
            round(_pct(pt["n_capped"], pt["n_probes"]), 1),
            f"final probe assistant message token_count >= {GEMINI_PRO_PROBE_CAP} over Gemini Pro runs with a final probe in {MULTI_TURN_CONDITIONS}",
            pt["n_capped"],
            pt["n_probes"],
            "transcripts/*.jsonl; analysis/final_probe_accuracy.csv",
            f"off by one against July: {pt['n_one_below_cap']} probe sits at {GEMINI_PRO_PROBE_CAP - 1} tokens and the >= {GEMINI_PRO_PROBE_CAP} rule was not tuned to recover it; "
            f"{pt['n_untruncated_full_recall']} of {pt['n_untruncated']} untruncated probes score 100% recall",
        )
    )

    # B3
    for name in ("as_published", "clean"):
        rows += sonnet_checkpoint_recovery(flags[name], name)

    # B5
    fr = failed_runs.compute(data_root)
    fr_summary = fr["failed_runs_summary"].set_index("quantity")["value"]
    rows_in = failed_runs.rows_in_from_summary(fr["failed_runs_summary"])
    n_design = int(fr_summary["design_rows"])
    rows.append(_entry("B5", "total scored runs, as published", "2,146", int(log["as_published_rows"]), "rows in all_scores.parquet", None, None, src_scores))
    rows.append(_entry("B5", "total scored runs, clean", "2,116", int(log["clean_rows"]), f"drop {log['stray_sonnet_rows_dropped']} stray Sonnet rows (judge date 2026-04-16 UTC); Sonnet de-dup to 2 per cell removed {log['sonnet_dedup_rows_dropped']}", None, None, src_scores + "; scores/judge_*.jsonl"))
    rows.append(
        _entry(
            "B5",
            "failed runs",
            "12",
            int(fr_summary["failed_runs"]),
            f"intended design rows (38 x 4 x 7 x 2 = {n_design:,}) absent from the parquet: {int(fr_summary['subject_failures'])} missing transcripts "
            f"+ {int(fr_summary['scoring_failures'])} scoring failures (transcript without judge, or judge without auditor)",
            int(fr_summary["failed_runs"]),
            n_design,
            "camera_ready/analysis/failed_runs.csv",
            "the missing transcripts are inferred from the design, not logged; see failed_runs_summary.csv",
        )
    )
    rows.append(_entry("B5", "non-Sonnet design shortfall", "12", int(fr_summary["non_sonnet_design_shortfall"]), "1,824 intended non-Sonnet rows minus non-Sonnet rows in the parquet", None, None, src_scores))

    # B8
    raters = rater_violation_counts(data_root / "human_validation" / "raters")
    rows.append(_entry("B8", "per-rater flagged violations, min to max over rater files", "15 to 86", f"{int(raters['violated'].min())} to {int(raters['violated'].max())}", "sum of violated_count per rater file", None, int(len(raters)), "human_validation/raters/*.json"))
    for r in raters.itertuples(index=False):
        rows.append(_entry("B8", f"flagged violations, {r.file}", None, int(r.violated), "sum of violated_count over the file's runs; stretched in numerator column", int(r.stretched), int(r.n_runs), f"human_validation/raters/{r.file}"))

    # B9
    with open(data_root / "human_validation" / "human_scores.json") as fh:
        human = pd.DataFrame(json.load(fh).values())
    human = human.merge(scores[["run_id", "condition", "model_id"]], on="run_id", how="left")
    hs = "human_validation/human_scores.json; " + src_scores
    for col, pub in (("computed_score", ("3.704", "2.585", "-1.118")), ("computed_score_from_constraint_median", (None, None, "-1.02"))):
        m = human_condition_means(human, col)
        rows.append(_entry("B9", f"human adherence single-shot mean ({col})", pub[0], round(m["single_shot"], 3), f"mean {col} over runs joined to parquet condition == single_shot", None, m["n_ss"], hs))
        rows.append(_entry("B9", f"human adherence pressure mean ({col})", pub[1], round(m["pressure"], 3), f"mean {col} over runs joined to parquet condition == multi_turn_pressure", None, m["n_pr"], hs))
        rows.append(_entry("B9", f"human adherence delta pressure minus single-shot ({col})", pub[2], round(m["delta"], 3), "pressure mean minus single-shot mean", m["n_pr"], m["n_ss"], hs))

    # B12
    mc = monitored_counts(data_root)
    for key, pub in (("transcripts", "151"), ("judge_files", "151"), ("auditor_files", "151"), ("aggregated_rows", "151")):
        rows.append(_entry("B12", f"monitored {key}", pub, mc[key], "file or row count under monitored/", None, None, "monitored/"))

    # B14
    cc = sonnet_cell_counts(scores, transcripts)
    for condition in MAIN_CONDITIONS:
        rows.append(_entry("B14", f"Sonnet scored runs, {condition}", "91 (SS, pressure)" if condition in ("single_shot", PRESSURE) else "76", int(cc["runs_per_condition"].get(condition, 0)), "parquet rows with model_id == Sonnet", None, 76, src_scores))
    rows.append(
        _entry(
            "B14",
            "Sonnet (brief, condition) cells with 3 scored runs",
            None,
            cc["cells_with_3_runs"],
            "cells in the parquet with 3 Sonnet rows; design intends 2; equals the 30 stray rows (one stray pilot row per three-run cell)",
            cc["cells_with_3_runs"],
            cc["n_cells"],
            src_scores,
            "not the published '11 duplicate design cells', which counts transcript cells with a duplicated repetition index (next row)",
        )
    )
    rows.append(_entry("B14", "Sonnet transcript cells with a duplicated repetition index", "11 duplicate design cells", cc["transcript_cells_with_duplicate_repetition"], "(brief, condition, repetition) triples with more than one Sonnet transcript, main conditions", None, None, "transcripts/*.jsonl"))

    # B15
    son = scores[scores["model_id"] == SONNET]
    no_transcript = int((~son["run_id"].isin(transcripts["run_id"])).sum())
    rows.append(_entry("B15", "Sonnet scored runs without a transcript", "19", no_transcript, "parquet Sonnet run_id with no file under transcripts/", no_transcript, int(len(son)), src_scores + "; transcripts/*.jsonl"))
    f_pub = flags["as_published"]
    sp = f_pub[(f_pub["model_id"] == SONNET) & (f_pub["condition"] == PRESSURE)]
    rows.append(_entry("B15", "Sonnet pressure runs without a usable final probe", "9 of 91", int((~sp["has_final_probe"]).sum()), "parquet Sonnet pressure rows with no row in final_probe_accuracy.csv", int((~sp["has_final_probe"]).sum()), int(len(sp)), src_scores + "; analysis/final_probe_accuracy.csv"))
    judge = load_score_records(data_root, "judge")
    ss_raw = int((judge["condition"] == "single_shot").sum())
    ss_pq = int((scores["condition"] == "single_shot").sum())
    rows.append(_entry("B15", "single-shot denominator, parquet", "542", ss_pq, "parquet rows with condition == single_shot", None, None, src_scores))
    rows.append(_entry("B15", "single-shot denominator, raw judge files", "544", ss_raw, "judge_*.jsonl files with condition == single_shot (main + open-weight score dirs)", None, None, "scores/judge_*.jsonl; openweight_subjects/scores/judge_*.jsonl"))

    # B16
    tb = truncation_by_condition(transcripts)
    for r in tb.itertuples(index=False):
        pub = "226 of 315" if r.condition == "all_main" else None
        note = "reported in the paper only as the truncation prevalence; the adherence test uses the final-turn rule" if r.condition == "all_main" else ""
        rows.append(_entry("B16", f"Sonnet runs truncated (any assistant message token_count == {SONNET_OUTPUT_CAP}), {r.condition}", pub, int(r.truncated_any_turn), "transcripts on the four main conditions; final-turn truncations in numerator column", int(r.truncated_final), int(r.n), "transcripts/*.jsonl", note))
    # Two truncation rules on two samples. The July values (2.385 vs 2.533,
    # n = 82) equal 124/52 and 76/30, the final-turn split on the
    # as-published sample. That is the paper's test. The any-turn split
    # leaves 4-5 untruncated runs and is kept for the record only.
    trunc_rules = (
        ("final_assistant_at_cap", "final turn", ("2.385", "2.533", "0.205")),
        ("any_assistant_at_cap", "any turn", (None, None, None)),
    )
    mw_rule = "scipy.stats.mannwhitneyu, two-sided, asymptotic (normal approximation, tie-corrected, continuity-corrected), truncated vs untruncated under {column}"
    for name in ("as_published", "clean"):
        for column, label, pub in trunc_rules:
            pub = pub if name == "as_published" else (None, None, None)
            t = truncation_adherence_test(flags[name], column=column)
            src = src_scores + "; transcripts/*.jsonl"
            if label == "final turn":
                note = B16_PAPER_WORDING if name == "as_published" else "final-turn rule on the clean sample; the paper prints the as-published final-turn test"
            else:
                note = B16_ANY_TURN_NOTE
            rows.append(_entry("B16", f"Sonnet pressure adherence, truncated mean, rule={label}, sample={name}", pub[0], round(t["mean_truncated"], 3), f"mean j_constraint_adherence, Sonnet pressure runs with a transcript and {column} == True", None, t["n_truncated"], src, note))
            rows.append(_entry("B16", f"Sonnet pressure adherence, untruncated mean, rule={label}, sample={name}", pub[1], round(t["mean_untruncated"], 3), f"mean j_constraint_adherence, Sonnet pressure runs with a transcript and {column} == False", None, t["n_untruncated"], src, note))
            rows.append(_entry("B16", f"Sonnet pressure adherence, Mann-Whitney p (two-sided), rule={label}, sample={name}", pub[2], round(t["p"], 3) if not math.isnan(t["p"]) else None, mw_rule.format(column=column), t["n_truncated"], t["n_truncated"] + t["n_untruncated"], src, note))

    # B17
    sc = structured_status_counts(data_root / "scores")
    non_sat = sc.get("stretched", 0) + sc.get("violated", 0)
    rows.append(_entry("B17", "structured judge: stretched among non-satisfied constraint checks", "450 of 470", sc.get("stretched", 0), "status == stretched over checks with status != satisfied; 5 commercial models, all conditions", sc.get("stretched", 0), non_sat, "scores/structured_*.jsonl"))
    rows.append(_entry("B17", "structured judge: violated among non-satisfied constraint checks", "20 of 470", sc.get("violated", 0), "status == violated over checks with status != satisfied", sc.get("violated", 0), non_sat, "scores/structured_*.jsonl"))
    rows.append(_entry("B17", "structured judge: total constraint checks", "9,393", sum(sc.values()), "all constraint_checks entries", None, None, "scores/structured_*.jsonl"))
    hl = human_label_counts(human)
    h_non = hl.get("stretched", 0) + hl.get("violated", 0)
    rows.append(_entry("B17", "human median labels: stretched among non-satisfied", "47 of 106", hl.get("stretched", 0), "constraint_statuses median label == Stretched over labels != Satisfied, 110 runs", hl.get("stretched", 0), h_non, "human_validation/human_scores.json"))
    rows.append(_entry("B17", "human median labels: violated among non-satisfied", "59 of 106", hl.get("violated", 0), "constraint_statuses median label == Violated over labels != Satisfied, 110 runs", hl.get("violated", 0), h_non, "human_validation/human_scores.json"))

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    extras = {"samples": log, "monitored": mc, "sonnet_cells": cc, "raters": raters, "truncation": tb, "rows_in": rows_in}
    return ledger, extras


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root, out_dir = Path(data_root), Path(out_dir)
    ledger, extras = build_ledger(data_root)
    write_csv_with_provenance(
        ledger,
        out_dir / "corrections_ledger.csv",
        {
            "inputs": sorted(set(s for cell in ledger["source_files"] for s in cell.split("; ") if s)),
            "rows_in": extras["rows_in"],
            "filters": {
                "samples": extras["samples"],
                "drift_cases": f"pressure parquet rows with a_drift_classification in {DRIFT_CLASSES}",
                "truncation": f"assistant message token_count == {SONNET_OUTPUT_CAP}",
                "probe_truncation": f"Gemini Pro final probe assistant message token_count >= {GEMINI_PRO_PROBE_CAP}",
                "kbv": f"overall_accuracy >= 0.8 and j_constraint_adherence < {ADHERENCE_THRESHOLD}",
                "mann_whitney": "scipy.stats.mannwhitneyu, two-sided, method=asymptotic (tie-corrected, continuity-corrected)",
                "b16_paper_wording": B16_PAPER_WORDING,
                "b16_any_turn": B16_ANY_TURN_NOTE,
                "published_values": "copied from the camera-ready plan (section 2; not part of the public repository); not used in any computation",
            },
            "description": "corrections ledger: one row per corrected quantity; the note column says what the paper must print",
        },
    )
    return {"corrections_ledger": ledger}


def main() -> None:
    ledger = run()["corrections_ledger"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70, "display.max_rows", 200):
        print(ledger[["id", "quantity", "published_value", "recomputed_value", "numerator", "denominator", "note"]])


if __name__ == "__main__":
    main()
