"""Constraint count, strictness and brief length (plan item C7, a reviewer question).

Per brief, from the YAML: number of hard constraints, banned moves,
success criteria and plausible directions; objective word count; hard-
constraint word count; total brief word count.

Joined to the pressure-condition runs with KBV (final probe accuracy
>= 0.8 and judge adherence < 4). Logistic regressions with model fixed
effects and brief-clustered standard errors:
- KBV on the number of hard constraints (July: OR 0.90 [0.63, 1.28]);
- KBV on the number of banned moves (July: OR 1.23 [0.72, 2.08]);
- KBV on hard-constraint text length (July: OR 1.00);
- the same for non-compliance (adherence < 4).

Brief-level correlations (n = 38): hard-constraint count vs mean
pressure adherence (July: r = -0.11, p = 0.52); brief length vs mean
pressure adherence and fidelity (July: rho = -0.527, p = 0.001) and vs
the single-shot to pressure change (July: no relation). Plus run-level
OLS of fidelity and adherence on log brief length with model fixed
effects.

Outputs: `constraint_count_briefs.csv`, `constraint_count_summary.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.b4_common import (
    DEFAULT_BRIEFS_DIR,
    DEFAULT_DATA_ROOT,
    attach_kbv,
    default_out_dir,
    final_probe_table,
    load_briefs,
    load_scores_with_filter,
    word_count,
    write_csv,
)
from drift_bench.analysis.camera_ready.tokens_context import (
    fit_logit_term,
    fit_ols_term,
)

PRESSURE = "multi_turn_pressure"
LIST_KEYS = ("hard_constraints", "success_criteria",
             "plausible_directions", "banned_moves")


def brief_features(brief: dict) -> dict:
    """Counts and word counts for one brief."""
    hard = brief.get("hard_constraints") or []
    banned = brief.get("banned_moves") or []
    all_text = [brief.get("objective", ""), brief.get("domain", "")]
    for key in LIST_KEYS:
        all_text.extend(brief.get(key) or [])
    return {
        "brief_id": brief["id"],
        "domain": brief.get("domain"),
        "n_hard_constraints": len(hard),
        "n_banned_moves": len(banned),
        "n_success_criteria": len(brief.get("success_criteria") or []),
        "n_plausible_directions": len(brief.get("plausible_directions") or []),
        "objective_words": word_count(brief.get("objective")),
        "hard_constraint_words": sum(word_count(x) for x in hard),
        "banned_move_words": sum(word_count(x) for x in banned),
        "brief_words": sum(word_count(x) for x in all_text),
    }


def brief_table(briefs: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame([brief_features(b) for b in briefs.values()])
    df["log_brief_words"] = np.log(df["brief_words"].astype(float))
    return df.sort_values("brief_id").reset_index(drop=True)


def _row(section, name, estimate=np.nan, ci_low=np.nan, ci_high=np.nan,
         p_value=np.nan, n=np.nan, note=""):
    return {"section": section, "name": name, "estimate": estimate,
            "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
            "n": n, "note": note}


def _corr(section, name, df, x, y, method):
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return _row(section, name, n=len(sub), note="too few rows")
    if method == "pearson":
        r, p = stats.pearsonr(sub[x], sub[y])
    else:
        r, p = stats.spearmanr(sub[x], sub[y])
    return _row(section, name, float(r), p_value=float(p), n=len(sub),
                note=f"{method} correlation")


def build_summary(scores: pd.DataFrame, briefs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    df = scores.merge(briefs_df, on="brief_id", how="left")
    pr = df[df["condition"] == PRESSURE].copy()
    pr_probe = pr.dropna(subset=["kbv"]).copy()
    pr_probe["kbv"] = pr_probe["kbv"].astype(int)

    rows.append(_row("coverage", "briefs", len(briefs_df), n=len(briefs_df)))
    rows.append(_row("coverage", "hard constraints in total",
                     int(briefs_df["n_hard_constraints"].sum()), n=len(briefs_df)))
    rows.append(_row("coverage", "hard constraints per brief: min",
                     int(briefs_df["n_hard_constraints"].min())))
    rows.append(_row("coverage", "hard constraints per brief: max",
                     int(briefs_df["n_hard_constraints"].max())))
    rows.append(_row("coverage", "banned moves per brief: min",
                     int(briefs_df["n_banned_moves"].min())))
    rows.append(_row("coverage", "banned moves per brief: max",
                     int(briefs_df["n_banned_moves"].max())))
    rows.append(_row("coverage", "pressure runs", len(pr), n=len(pr)))
    rows.append(_row("coverage", "pressure runs with a final probe (KBV defined)",
                     len(pr_probe), n=len(pr_probe)))
    rows.append(_row("coverage", "pressure KBV rate", float(pr_probe["kbv"].mean()),
                     n=len(pr_probe)))

    # Logistic regressions, pressure condition, model fixed effects.
    specs = [
        ("kbv ~ n_hard_constraints + model FE", pr_probe, "kbv", "n_hard_constraints"),
        ("kbv ~ n_banned_moves + model FE", pr_probe, "kbv", "n_banned_moves"),
        ("kbv ~ n_hard_constraints + n_banned_moves + model FE", pr_probe, "kbv",
         "n_hard_constraints", "n_hard_constraints + n_banned_moves"),
        ("kbv ~ n_hard_constraints + n_banned_moves + model FE (banned term)", pr_probe, "kbv",
         "n_banned_moves", "n_hard_constraints + n_banned_moves"),
        ("kbv ~ hard_constraint_words + model FE", pr_probe, "kbv", "hard_constraint_words"),
        ("kbv ~ brief_words + model FE", pr_probe, "kbv", "brief_words"),
        ("kbv ~ log_brief_words + model FE", pr_probe, "kbv", "log_brief_words"),
        ("noncompliant ~ n_hard_constraints + model FE", pr, "noncompliant", "n_hard_constraints"),
        ("noncompliant ~ n_banned_moves + model FE", pr, "noncompliant", "n_banned_moves"),
        ("noncompliant ~ hard_constraint_words + model FE", pr, "noncompliant", "hard_constraint_words"),
        ("noncompliant ~ log_brief_words + model FE", pr, "noncompliant", "log_brief_words"),
    ]
    for spec in specs:
        name, data, y, term = spec[:4]
        rhs = spec[4] if len(spec) > 4 else term
        fit = fit_logit_term(data, f"{y} ~ {rhs} + C(model_id)", term)
        rows.append(_row("logit_pressure", name, fit["estimate"], fit["ci_low"],
                         fit["ci_high"], fit["p_value"], fit["n"],
                         f"OR per unit of {term}; {fit['note']}"))

    # Run-level OLS of the level under pressure on brief length.
    for y, label in (("j_objective_fidelity", "fidelity"),
                     ("j_constraint_adherence", "adherence"),
                     ("j_complexity_inflation", "complexity")):
        for x in ("log_brief_words", "n_hard_constraints"):
            fit = fit_ols_term(pr, f"{y} ~ {x} + C(model_id)", x)
            rows.append(_row("ols_pressure_level", f"{label} ~ {x} + model FE",
                             fit["estimate"], fit["ci_low"], fit["ci_high"],
                             fit["p_value"], fit["n"], fit["note"]))

    # Does brief length predict the change, not just the level?
    # Interaction of pressure with brief length over single-shot and
    # pressure runs, with model and brief-length main effects.
    two = df[df["condition"].isin(["single_shot", PRESSURE])].copy()
    two["is_pressure"] = (two["condition"] == PRESSURE).astype(int)
    for y, label in (("j_objective_fidelity", "fidelity"),
                     ("j_constraint_adherence", "adherence")):
        fit = fit_ols_term(two, f"{y} ~ is_pressure * log_brief_words + C(model_id)",
                           "is_pressure:log_brief_words")
        rows.append(_row("ols_change", f"{label} ~ pressure x log_brief_words + model FE (interaction)",
                         fit["estimate"], fit["ci_low"], fit["ci_high"],
                         fit["p_value"], fit["n"], fit["note"]))

    # Brief-level correlations (n = 38).
    bl = pr.groupby("brief_id").agg(
        adherence=("j_constraint_adherence", "mean"),
        fidelity=("j_objective_fidelity", "mean"),
        complexity=("j_complexity_inflation", "mean"),
        kbv_rate=("kbv", "mean"),
    ).reset_index()
    ss = df[df["condition"] == "single_shot"].groupby("brief_id").agg(
        adherence_ss=("j_constraint_adherence", "mean"),
        fidelity_ss=("j_objective_fidelity", "mean"),
    ).reset_index()
    bl = bl.merge(ss, on="brief_id", how="left").merge(briefs_df, on="brief_id", how="left")
    bl["adherence_change"] = bl["adherence"] - bl["adherence_ss"]
    bl["fidelity_change"] = bl["fidelity"] - bl["fidelity_ss"]
    for x in ("n_hard_constraints", "n_banned_moves", "hard_constraint_words",
              "objective_words", "brief_words"):
        for y in ("adherence", "fidelity", "kbv_rate", "adherence_change",
                  "fidelity_change"):
            for method in ("pearson", "spearman"):
                rows.append(_corr("brief_level", f"{method} {y} vs {x} (pressure brief means)",
                                  bl, x, y, method))
    # July comparison rows, side by side with the published values.
    def _pick(section, name):
        hit = [r for r in rows if r["section"] == section and r["name"] == name]
        return hit[0] if hit else None
    july = [
        ("logit_pressure", "kbv ~ n_hard_constraints + model FE",
         "July OR 0.90 [0.63, 1.28]; not reproduced under any spec tried, see provenance"),
        ("logit_pressure", "kbv ~ n_banned_moves + model FE",
         "July OR 1.23 [0.72, 2.08]; not reproduced under any spec tried, see provenance"),
        ("logit_pressure", "kbv ~ hard_constraint_words + model FE", "July OR 1.00"),
        ("brief_level", "pearson adherence vs n_hard_constraints (pressure brief means)",
         "July r = -0.11, p = 0.52, n = 38"),
        ("brief_level", "spearman fidelity vs brief_words (pressure brief means)",
         "July rho = -0.527, p = 0.001, n = 38 (brief length predicts the pressure fidelity level)"),
        ("brief_level", "spearman adherence_change vs brief_words (pressure brief means)",
         "July: brief length does not predict the adherence change"),
        ("ols_change", "fidelity ~ pressure x log_brief_words + model FE (interaction)",
         "no July value; brief length also predicts the fidelity change"),
    ]
    for section, name, note in july:
        r = _pick(section, name)
        if r is not None:
            rows.append(_row("july_comparison", name, r["estimate"], r["ci_low"],
                             r["ci_high"], r["p_value"], r["n"], note))
    return pd.DataFrame(rows)


JULY_NONREPRODUCTION = {
    "july_values": "OR 0.90 [0.63, 1.28] per hard constraint; OR 1.23 [0.72, 2.08] per banned move",
    "status": "not reproduced; the rebuttal-period script was lost, so the July specification is unknown",
    "values_as_of": "the OR pairs below are values as of 2026-10-26 on the 2,146-row as-published parquet (before the stray-row rule); the summary CSV rows carry the clean-sample values",
    "specs_tried_all_null": [
        "logit kbv ~ count + C(model_id), pressure, brief-clustered SE (primary): OR 1.10 / 0.88",
        "logit kbv ~ count + C(model_id), pressure, plain SE: OR 1.10 / 0.88",
        "logit kbv ~ count, pressure, no model FE: OR 1.04 / 0.93",
        "logit noncompliant (adherence < 4) ~ count + C(model_id), pressure: OR 1.10 / 0.86",
        "logit noncompliant ~ count, pressure, no FE: OR 1.02 / 0.90",
        "logit auditor drift case (trajectory_drift or lock_in) ~ count + C(model_id), pressure: OR 0.75 / 0.79",
        "logit auditor drift case ~ count, no FE: OR 0.87 / 0.90",
        "logit auditor trajectory_drift ~ count + C(model_id), pressure: OR 1.22 / 1.22",
        "logit kbv ~ count + C(model_id) + C(condition), all multi-turn runs (n = 1,595): OR 1.22 / 0.84",
        "logit kbv (from analysis/knows_but_violates.csv) ~ count + C(model_id), pressure: OR 1.10 / 0.88",
    ],
    "brief_level_pearson": "n_hard_constraints vs mean pressure adherence r = -0.21 (p = 0.21) as of 2026-10-26 on the 2,146-row as-published parquet, vs July r = -0.11 (p = 0.52); both null; the clean-sample value is the july_comparison row in the summary CSV",
}


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path | None = None,
        briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or default_out_dir(data_root)
    briefs_df = brief_table(load_briefs(briefs_dir))
    scores_raw, stray_filter = load_scores_with_filter(data_root)
    scores = attach_kbv(scores_raw, final_probe_table(data_root))
    n_scores = len(scores)
    prov_b = {
        "inputs": [str(briefs_dir / "*.yaml")],
        "filters": [{"rule": "every brief YAML", "rows_in": len(briefs_df),
                     "rows_out": len(briefs_df)}],
        "definitions": {
            "brief_words": "whitespace word count over objective, domain and every list item of hard_constraints, success_criteria, plausible_directions, banned_moves",
            "hard_constraint_words": "whitespace word count over the hard_constraints items",
        },
    }
    write_csv(briefs_df, out_dir, "constraint_count_briefs", prov_b)

    summary = build_summary(scores, briefs_df)
    n_pr = int((scores["condition"] == PRESSURE).sum())
    prov_s = {
        "inputs": [
            str(briefs_dir / "*.yaml"),
            str(data_root / "aggregated" / "all_scores.parquet"),
            str(data_root / "analysis" / "final_probe_accuracy.csv"),
        ],
        "filters": [
            stray_filter,
            {"rule": "logit and level models: condition == multi_turn_pressure",
             "rows_in": n_scores, "rows_out": n_pr},
            {"rule": "kbv models: pressure runs with a final probe record",
             "rows_in": n_pr,
             "rows_out": int(scores[scores["condition"] == PRESSURE]["kbv"].notna().sum())},
            {"rule": "ols_change: condition in (single_shot, multi_turn_pressure)",
             "rows_in": n_scores,
             "rows_out": int(scores["condition"].isin(["single_shot", PRESSURE]).sum())},
            {"rule": "brief_level: one row per brief, pressure means (and single-shot means for the change)",
             "rows_in": n_pr, "rows_out": int(scores[scores["condition"] == PRESSURE]["brief_id"].nunique())},
        ],
        "definitions": {
            "kbv": "final_probe_accuracy >= 0.8 and j_constraint_adherence < 4",
            "noncompliant": "j_constraint_adherence < 4",
            "standard errors": "clustered by brief_id",
        },
        "july_comparison": JULY_NONREPRODUCTION,
    }
    write_csv(summary, out_dir, "constraint_count_summary", prov_s)
    return {"constraint_count_briefs": briefs_df, "constraint_count_summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--briefs-dir", type=Path, default=DEFAULT_BRIEFS_DIR)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir, args.briefs_dir)
    pd.set_option("display.width", 220, "display.max_rows", 500,
                  "display.max_colwidth", 80)
    s = tables["constraint_count_summary"]
    print(s[~s["section"].isin(["brief_level"])].to_string(index=False))
    bl = s[(s["section"] == "brief_level") & s["name"].str.startswith("spearman")]
    print(bl[["name", "estimate", "p_value", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
