"""Constraint typology analysis (camera-ready items C6, E8).

Reads the annotation sheet, matches every structured-judge constraint check to its
brief constraint, and computes per-type non-satisfaction rates, the pressure-induced
excess, the condition-by-type interaction test, the best balanced 4-vs-4 split, and
author-vs-LLM agreement.

Label rule: ``label_author`` when filled, else ``label_llm``. The column
``label_source`` records which one was used for each constraint.

Match rule: a check matches a constraint of the same brief when the texts are equal
after whitespace normalisation and case folding. Otherwise the constraint with the
highest ``difflib.SequenceMatcher`` ratio wins if that ratio exceeds 0.9. Checks
with no match are reported in ``typology_unmatched.csv`` and excluded.

Non-satisfied means status in {stretched, violated}.

Sample: the camera-ready sample drops the 30 stray Sonnet pilot runs (plan B5/Q5) via
the shared ``sample.drop_stray`` helper. 11 of them have structured score files.

Inference: checks within one run come from one judge call, so they are correlated.
The primary interaction tests cluster standard errors by ``run_id`` (757 runs in the
SS and pressure cells after the stray filter). The unclustered likelihood-ratio test, the unclustered Wald
test, and the brief-clustered Wald test (38 clusters, few-clusters caveat) are
sensitivity rows.

Usage::

    python -m drift_bench.analysis.camera_ready.typology
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import itertools
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready.sample import drop_stray, stray_run_ids
from drift_bench.analysis.camera_ready.typology_annotate import (
    CATEGORY_IDS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    load_constraints,
    read_sheet,
)

log = logging.getLogger(__name__)

COMMERCIAL_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
]
CONDITIONS = ["single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"]
NON_SATISFIED = {"stretched", "violated"}
MATCH_THRESHOLD = 0.9
QUANT_TYPES = ["resource_limit", "sample_floor", "duration_window", "count_scope_limit"]
N_FRAGILE = 4

# July 2026 values from the rebuttal, for the report only. Never used in a computation.
JULY = {
    "n_checks": 9386,
    "rate_min": 0.0154,
    "rate_max": 0.1978,
    "resource_limit_n": 278,
    "excess_share_top4": 0.826,
    "split_or": 2.19,
    "omnibus_p": 0.124,  # most likely the unclustered Wald test
    "n_non_satisfied": 469,
}


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def match_constraint(check_text: str, candidates: list[str], threshold: float = MATCH_THRESHOLD) -> tuple[int | None, str, float]:
    """Return (index, how, ratio) for the best candidate.

    ``how`` is 'exact', 'fuzzy' or 'none'. ``index`` is None when nothing matches.
    """
    norm = normalise(check_text)
    norms = [normalise(c) for c in candidates]
    for i, c in enumerate(norms):
        if c == norm:
            return i, "exact", 1.0
    best_i, best_r = None, 0.0
    for i, c in enumerate(norms):
        r = difflib.SequenceMatcher(None, norm, c).ratio()
        if r > best_r:
            best_i, best_r = i, r
    if best_i is not None and best_r > threshold:
        return best_i, "fuzzy", best_r
    return None, "none", best_r


def resolve_labels(sheet: pd.DataFrame) -> pd.DataFrame:
    """Add ``label`` and ``label_source`` columns. Author label wins when filled."""
    out = sheet.copy()
    author = out["label_author"].fillna("").astype(str).str.strip()
    llm = out["label_llm"].fillna("").astype(str).str.strip()
    out["label"] = np.where(author != "", author, llm)
    out["label_source"] = np.where(author != "", "author", np.where(llm != "", "llm", "none"))
    bad = set(out.loc[out["label"] != "", "label"]) - set(CATEGORY_IDS)
    if bad:
        raise ValueError(f"unknown category ids in sheet: {sorted(bad)}")
    return out


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_type_rates(checks: pd.DataFrame) -> pd.DataFrame:
    """n, non-satisfied count, rate and Wilson CI per type, overall and by condition.

    ``checks`` needs columns: label, condition, non_satisfied (bool).
    """
    rows = []
    groups = [("all", checks)] + [(c, checks[checks["condition"] == c]) for c in CONDITIONS if (checks["condition"] == c).any()]
    for cond, sub in groups:
        for t, g in sub.groupby("label"):
            n = len(g)
            k = int(g["non_satisfied"].sum())
            lo, hi = wilson_ci(k, n)
            rows.append(
                {
                    "condition": cond,
                    "type": t,
                    "n_checks": n,
                    "n_non_satisfied": k,
                    "rate": k / n if n else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_constraints": g["constraint_key"].nunique() if "constraint_key" in g else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df["quantitative_envelope"] = df["type"].isin(QUANT_TYPES)
    return df.sort_values(["condition", "rate"], ascending=[True, False]).reset_index(drop=True)


def excess_by_type(checks: pd.DataFrame, fragile_types: list[str]) -> pd.DataFrame:
    """Pressure-minus-single-shot non-satisfied checks per type, and the fragile share.

    ``excess`` is the raw count difference. The two cells have near-identical check
    counts (same briefs, same models) but not identical ones, so the table also carries
    ``excess_rate_based = (rate_pressure - rate_single_shot) * n_checks_pressure`` and
    its share, so a reader can see the arithmetic for both versions.
    """
    ss = checks[checks["condition"] == "single_shot"]
    pr = checks[checks["condition"] == "multi_turn_pressure"]
    rows = []
    for t in sorted(checks["label"].unique()):
        s = ss[ss["label"] == t]
        p = pr[pr["label"] == t]
        rows.append(
            {
                "type": t,
                "n_checks_single_shot": len(s),
                "n_checks_pressure": len(p),
                "non_satisfied_single_shot": int(s["non_satisfied"].sum()),
                "non_satisfied_pressure": int(p["non_satisfied"].sum()),
                "rate_single_shot": s["non_satisfied"].mean() if len(s) else float("nan"),
                "rate_pressure": p["non_satisfied"].mean() if len(p) else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df["excess"] = df["non_satisfied_pressure"] - df["non_satisfied_single_shot"]
    total = df["excess"].sum()
    df["excess_share"] = df["excess"] / total if total else float("nan")
    df["excess_rate_based"] = (df["rate_pressure"] - df["rate_single_shot"]) * df["n_checks_pressure"]
    total_rb = df["excess_rate_based"].sum()
    df["share_rate_based"] = df["excess_rate_based"] / total_rb if total_rb else float("nan")
    df["fragile_top4"] = df["type"].isin(fragile_types)
    df["check_share"] = (df["n_checks_single_shot"] + df["n_checks_pressure"]) / (
        df["n_checks_single_shot"].sum() + df["n_checks_pressure"].sum()
    )
    return df


def balanced_splits(types: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """All unordered 4-vs-4 splits of 8 types. 70 / 2 = 35."""
    types = sorted(types)
    if len(types) != 8:
        raise ValueError("balanced_splits needs exactly 8 types")
    anchor = types[0]
    out = []
    for combo in itertools.combinations(types, 4):
        if anchor in combo:
            rest = tuple(t for t in types if t not in combo)
            out.append((combo, rest))
    return out


def _fit_logit(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None):
    """Binomial GLM. ``groups`` None gives ordinary SEs, else cluster-robust SEs."""
    import statsmodels.api as sm

    model = sm.GLM(y, X, family=sm.families.Binomial())
    if groups is None:
        return model.fit()
    codes = pd.factorize(pd.Series(groups))[0]
    return model.fit(cov_type="cluster", cov_kwds={"groups": codes})


CLUSTER_LEVELS = {"run": "run_id", "brief": "brief_id"}
Z95 = 1.959964


def _design(checks: pd.DataFrame, group_a: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sub = checks[checks["condition"].isin(["single_shot", "multi_turn_pressure"])].copy()
    sub["pressure"] = (sub["condition"] == "multi_turn_pressure").astype(float)
    sub["grp"] = sub["label"].isin(group_a).astype(float)
    X = np.column_stack([np.ones(len(sub)), sub["pressure"], sub["grp"], sub["pressure"] * sub["grp"]])
    y = sub["non_satisfied"].astype(float).to_numpy()
    return sub, X, y


def split_interaction(checks: pd.DataFrame, group_a: tuple[str, ...], groups_col: str | None = "run_id") -> dict:
    """Logit non_satisfied ~ pressure * in_group_a on the SS and pressure cells.

    ``groups_col`` names the cluster column for the robust SE (``run_id`` by default,
    ``brief_id`` for the few-clusters sensitivity fit, None for ordinary SEs). The
    point estimate does not depend on the clustering; only the SE, CI and p do. The
    interaction OR is the ratio of pressure odds ratios between the two groups. Group
    A is the group whose OR is the numerator.
    """
    sub, X, y = _design(checks, group_a)
    groups = sub[groups_col].to_numpy() if groups_col else None
    res = _fit_logit(y, X, groups)
    beta = float(res.params[3])
    se = float(res.bse[3])
    return {
        "interaction_or": math.exp(beta),
        "interaction_se_log": se,
        "ci_low": math.exp(beta - Z95 * se),
        "ci_high": math.exp(beta + Z95 * se),
        "p_value": float(res.pvalues[3]),
        "n": len(sub),
        "groups_col": groups_col or "none",
        "n_clusters": int(pd.Series(groups).nunique()) if groups is not None else 0,
    }


def _split_both_levels(checks: pd.DataFrame, group_a: tuple[str, ...]) -> dict:
    """One row with the OR and, per cluster level, the SE, CI, p and cluster count."""
    out: dict = {}
    for level, col in CLUSTER_LEVELS.items():
        r = split_interaction(checks, group_a, groups_col=col)
        out["interaction_or"] = r["interaction_or"]
        out["n"] = r["n"]
        for k in ("interaction_se_log", "ci_low", "ci_high", "p_value", "n_clusters"):
            out[f"{k}_cluster_{level}"] = r[k]
    return out


def best_balanced_split(checks: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Fit all 35 splits at both cluster levels. Orient each so the OR is at least 1.

    Sorted by the run-clustered p (primary). The brief-clustered columns are the
    sensitivity fit.
    """
    rows = []
    for a, b in balanced_splits(types):
        r = _split_both_levels(checks, a)
        if r["interaction_or"] < 1:
            a, b = b, a
            r = _split_both_levels(checks, a)
        rows.append({"group_a": "|".join(a), "group_b": "|".join(b), **r})
    df = pd.DataFrame(rows).sort_values("p_value_cluster_run").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["n_splits_searched"] = len(df)
    quant = "|".join(sorted(QUANT_TYPES))
    # The flag must not depend on the orientation: the quant set may sit in group B.
    df["is_quant_vs_rest"] = (df["group_a"] == quant) | (df["group_b"] == quant)
    return df


def quant_vs_rest(checks: pd.DataFrame) -> dict:
    """The pre-specified split with the 4 quantitative-envelope types as the numerator.

    Not oriented, so the OR can be below 1.
    """
    return _split_both_levels(checks, tuple(sorted(QUANT_TYPES)))


def omnibus_interaction(checks: pd.DataFrame) -> pd.DataFrame:
    """Condition (SS vs pressure) by type (8 levels) interaction, four tests.

    Primary: Wald chi-square on the 7 interaction terms with cluster-robust SEs by
    run (``wald_cluster_run``). Sensitivity: the likelihood-ratio test with ordinary
    likelihood, the unclustered Wald test, and the brief-clustered Wald test (38
    clusters, so the asymptotics are shaky). The ``role`` column says which is which.
    """
    from scipy import stats

    sub = checks[checks["condition"].isin(["single_shot", "multi_turn_pressure"])].copy()
    sub["pressure"] = (sub["condition"] == "multi_turn_pressure").astype(float)
    types = sorted(sub["label"].unique())
    dummies = pd.get_dummies(pd.Categorical(sub["label"], categories=types), drop_first=True).astype(float)
    base = np.column_stack([np.ones(len(sub)), sub["pressure"].to_numpy(), dummies.to_numpy()])
    inter = (dummies.to_numpy() * sub["pressure"].to_numpy()[:, None])
    full = np.column_stack([base, inter])
    y = sub["non_satisfied"].astype(float).to_numpy()
    df_inter = inter.shape[1]
    k = base.shape[1]
    R = np.zeros((df_inter, full.shape[1]))
    for i in range(df_inter):
        R[i, k + i] = 1.0

    def wald_row(test: str, groups_col: str | None, role: str, note: str) -> dict:
        groups = sub[groups_col].to_numpy() if groups_col else None
        res = _fit_logit(y, full, groups)
        w = res.wald_test(R, scalar=True)
        return {
            "test": test,
            "statistic": float(w.statistic),
            "df": df_inter,
            "p_value": float(w.pvalue),
            "n": len(sub),
            "role": role,
            "n_clusters": int(pd.Series(groups).nunique()) if groups is not None else 0,
            "note": note,
        }

    r0 = _fit_logit(y, base)
    r1 = _fit_logit(y, full)
    lr = 2 * (r1.llf - r0.llf)
    rows = [
        wald_row("wald_cluster_run", "run_id", "primary", "chi-square Wald on the 7 interaction terms; cluster-robust SEs by run_id (checks within a run share one judge call)"),
        {"test": "likelihood_ratio", "statistic": lr, "df": df_inter, "p_value": float(stats.chi2.sf(lr, df_inter)), "n": len(sub), "role": "sensitivity", "n_clusters": 0, "note": "logit with vs without condition x type terms; ordinary likelihood; treats checks as independent"},
        wald_row("wald_unclustered", None, "sensitivity", "chi-square Wald on the 7 interaction terms; ordinary SEs; treats checks as independent (the July 2026 report most likely used this)"),
        wald_row("wald_cluster_brief", "brief_id", "sensitivity", "chi-square Wald on the 7 interaction terms; cluster-robust SEs by brief; 38 clusters, few-clusters caveat"),
    ]
    return pd.DataFrame(rows)


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement(sheet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kappa and confusion matrix on rows where both labels are filled."""
    both = sheet[(sheet["label_author"].astype(str).str.strip() != "") & (sheet["label_llm"].astype(str).str.strip() != "")]
    if both.empty:
        summary = pd.DataFrame([{"n_rows_both_labelled": 0, "percent_agreement": np.nan, "cohens_kappa": np.nan, "note": "label_author is empty; agreement not computed"}])
        return summary, pd.DataFrame()
    a, b = both["label_author"].astype(str), both["label_llm"].astype(str)
    summary = pd.DataFrame(
        [{"n_rows_both_labelled": len(both), "percent_agreement": float((a.to_numpy() == b.to_numpy()).mean()), "cohens_kappa": cohens_kappa(a, b), "note": "author vs claude-opus-4-6 annotator"}]
    )
    conf = pd.crosstab(a.rename("author"), b.rename("llm")).reindex(index=CATEGORY_IDS, columns=CATEGORY_IDS, fill_value=0)
    return summary, conf.reset_index()


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def load_checks(scores_dir: Path) -> tuple[pd.DataFrame, int]:
    """Every constraint check from structured_*.jsonl. Returns (df, n_files)."""
    rows = []
    files = sorted(scores_dir.glob("structured_*.jsonl"))
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            for j, c in enumerate(d.get("constraint_checks") or []):
                rows.append(
                    {
                        "run_id": d["run_id"],
                        "brief_id": d["brief_id"],
                        "model_id": d["model_id"],
                        "condition": d["condition"],
                        "judge_model": d.get("judge_model"),
                        "check_index": j,
                        "check_text": str(c.get("constraint", "")),
                        "status": str(c.get("status", "")).strip().lower(),
                    }
                )
    return pd.DataFrame(rows), len(files)


def match_all(checks: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    """Attach constraint_index, match_how, match_ratio to every check."""
    by_brief = {b: g.sort_values("constraint_index")["constraint_text"].tolist() for b, g in constraints.groupby("brief_id")}
    cache: dict[tuple[str, str], tuple[int | None, str, float]] = {}
    idx, how, ratio = [], [], []
    for b, t in zip(checks["brief_id"], checks["check_text"]):
        key = (b, t)
        if key not in cache:
            cache[key] = match_constraint(t, by_brief.get(b, []))
        i, h, r = cache[key]
        idx.append(i)
        how.append(h)
        ratio.append(r)
    out = checks.copy()
    out["constraint_index"] = idx
    out["match_how"] = how
    out["match_ratio"] = ratio
    return out


def _provenance(csv_path: Path, inputs: list[str], filters: list[str], n_in: int, n_out: int, extra: dict | None = None) -> None:
    prov = {"csv": csv_path.name, "inputs": inputs, "filters": filters, "rows_in": n_in, "rows_out": n_out, "date": dt.date.today().isoformat()}
    if extra:
        prov.update(extra)
    csv_path.with_name(csv_path.name + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = data_root / "scores"
    sheet_path = out_dir / "typology_sheet.csv"
    if not sheet_path.exists() and (DEFAULT_OUT_DIR / "typology_sheet.csv").exists():
        # The sheet lives in the default out dir. Fall back to it and record that.
        sheet_path = DEFAULT_OUT_DIR / "typology_sheet.csv"
        log.info("no sheet in %s; using %s", out_dir, sheet_path)

    sheet = resolve_labels(read_sheet(sheet_path))
    constraints = load_constraints()
    raw, n_files = load_checks(scores_dir)
    n_raw = len(raw)
    log.info("%d structured files, %d checks on disk", n_files, n_raw)

    filters = [
        f"files: {scores_dir}/structured_*.jsonl (5 commercial models, 4 conditions); {n_files} files",
        "one row per entry of constraint_checks",
    ]
    raw = raw[raw["model_id"].isin(COMMERCIAL_MODELS) & raw["condition"].isin(CONDITIONS)]
    filters.append(f"model_id in {COMMERCIAL_MODELS} and condition in {CONDITIONS}: {len(raw)} rows kept")
    # Camera-ready sample: drop the stray Sonnet pilot runs (shared rule, plan B5/Q5).
    stray = stray_run_ids(data_root)
    n_stray_with_checks = int(raw.loc[raw["run_id"].isin(stray), "run_id"].nunique())
    raw, stray_info = drop_stray(raw, data_root, run_col="run_id", stray=stray)
    stray_info["n_stray_runs_with_structured_checks"] = n_stray_with_checks
    filters.append(stray_info)
    log.info("stray filter: %s", stray_info)

    matched = match_all(raw, constraints)
    unmatched = matched[matched["constraint_index"].isna()].copy()
    matched = matched[matched["constraint_index"].notna()].copy()
    matched["constraint_index"] = matched["constraint_index"].astype(int)
    filters.append(
        f"match rule: exact text match after whitespace/case normalisation within the same brief, else best difflib ratio > {MATCH_THRESHOLD}; "
        f"{int((matched['match_how']=='exact').sum())} exact, {int((matched['match_how']=='fuzzy').sum())} fuzzy, {len(unmatched)} unmatched (dropped)"
    )

    # Attach labels.
    lab = sheet[["brief_id", "constraint_index", "label", "label_source", "constraint_text"]]
    matched = matched.merge(lab, on=["brief_id", "constraint_index"], how="left")
    n_before = len(matched)
    unlabelled = matched[matched["label"].fillna("") == ""]
    matched = matched[matched["label"].fillna("") != ""].copy()
    filters.append(f"constraint has a label (author or llm): {n_before - len(matched)} checks dropped for missing labels")
    matched["non_satisfied"] = matched["status"].isin(NON_SATISFIED)
    matched["constraint_key"] = matched["brief_id"] + ":" + matched["constraint_index"].astype(str)
    filters.append("non_satisfied = status in {stretched, violated}")

    label_source_counts = sheet["label_source"].value_counts().to_dict()
    common_inputs = [str(scores_dir / "structured_*.jsonl"), f"sheet_path_used: {sheet_path}", "drift_bench/briefs/*.yaml"]

    # 1. Per-type rates.
    by_type = per_type_rates(matched)
    path = out_dir / "typology_by_type.csv"
    by_type.to_csv(path, index=False)
    _provenance(path, common_inputs, filters, n_raw, len(by_type), {"label_source_counts": label_source_counts, "n_checks_used": len(matched)})

    # 2. Excess and fragile share.
    overall = by_type[by_type["condition"] == "all"].sort_values("rate", ascending=False)
    fragile = overall["type"].head(N_FRAGILE).tolist()
    excess = excess_by_type(matched, fragile)
    path = out_dir / "typology_excess.csv"
    excess.to_csv(path, index=False)
    _provenance(path, common_inputs, filters + ["cells: single_shot and multi_turn_pressure only", "excess = non-satisfied(pressure) - non-satisfied(single_shot) per type", "excess_rate_based = (rate_pressure - rate_single_shot) * n_checks_pressure; share_rate_based = excess_rate_based / sum(excess_rate_based)"], n_raw, len(excess), {"fragile_top4_by_overall_rate": fragile})

    # 3. Interaction tests and split enumeration.
    types = sorted(matched["label"].unique())
    omni = omnibus_interaction(matched)
    splits = best_balanced_split(matched, types) if len(types) == 8 else pd.DataFrame()
    inter_rows = omni.to_dict("records")
    if len(splits):
        best = splits.iloc[0]
        inter_rows.append({"test": "best_balanced_split", "statistic": best["interaction_or"], "df": 1, "p_value": best["p_value_cluster_run"], "n": best["n"], "role": "exploratory", "n_clusters": int(best["n_clusters_cluster_run"]), "note": f"selected minimum run-clustered p over {int(best['n_splits_searched'])} balanced 4-vs-4 splits; group A = {best['group_a']}; OR CI [{best['ci_low_cluster_run']:.3f}, {best['ci_high_cluster_run']:.3f}] by run; brief-clustered p = {best['p_value_cluster_brief']:.4g}; not a confirmatory test"})
        q = splits[splits["is_quant_vs_rest"]]
        if len(q):
            rank = int(q.iloc[0]["rank"])
            qr = quant_vs_rest(matched)
            inter_rows.append({"test": "quant_vs_rest_split", "statistic": qr["interaction_or"], "df": 1, "p_value": qr["p_value_cluster_run"], "n": qr["n"], "role": "pre_specified", "n_clusters": int(qr["n_clusters_cluster_run"]), "note": f"pre-specified codebook split; OR = pressure odds ratio of the 4 quantitative-envelope types over the 4 others (not oriented, may be < 1); run-clustered SE; OR CI [{qr['ci_low_cluster_run']:.3f}, {qr['ci_high_cluster_run']:.3f}]; brief-clustered p = {qr['p_value_cluster_brief']:.4g}; rank {rank} of {int(splits['n_splits_searched'].iloc[0])} by run-clustered p"})
    interaction = pd.DataFrame(inter_rows)
    path = out_dir / "typology_interaction.csv"
    interaction.to_csv(path, index=False)
    _provenance(path, common_inputs, filters + ["cells: single_shot and multi_turn_pressure only", "model: logit(non_satisfied) ~ pressure * type", "primary test: Wald on the 7 interaction terms with cluster-robust SEs by run_id; the other omnibus rows are sensitivity (role column)"], n_raw, len(interaction))
    if len(splits):
        path = out_dir / "typology_splits.csv"
        splits.to_csv(path, index=False)
        _provenance(path, common_inputs, filters + ["all 35 balanced 4-vs-4 splits of the 8 types; OR oriented >= 1", "SE, CI and p at two cluster levels: run_id (primary, sorted by this p) and brief_id (sensitivity)"], n_raw, len(splits))

    # 4. Agreement.
    agree, conf = agreement(sheet)
    path = out_dir / "typology_agreement.csv"
    agree.to_csv(path, index=False)
    _provenance(path, [str(sheet_path)], ["rows where both label_author and label_llm are non-empty"], len(sheet), len(agree))
    if len(conf):
        conf.to_csv(out_dir / "typology_confusion.csv", index=False)
        _provenance(out_dir / "typology_confusion.csv", [str(sheet_path)], ["rows where both labels are non-empty; rows = author, columns = llm"], len(sheet), len(conf))

    # 5. Unmatched.
    unmatched_out = pd.concat([unmatched, unlabelled.drop(columns=["constraint_key", "non_satisfied"], errors="ignore")], ignore_index=True)
    unmatched_out["reason"] = np.where(unmatched_out["constraint_index"].isna(), "no text match above threshold", "constraint has no label")
    cols = ["run_id", "brief_id", "model_id", "condition", "check_index", "check_text", "status", "match_ratio", "reason"]
    unmatched_out = unmatched_out[[c for c in cols if c in unmatched_out]]
    path = out_dir / "typology_unmatched.csv"
    unmatched_out.to_csv(path, index=False)
    _provenance(path, common_inputs, filters, n_raw, len(unmatched_out))

    return {
        "typology_by_type": by_type,
        "typology_excess": excess,
        "typology_interaction": interaction,
        "typology_splits": splits,
        "typology_agreement": agree,
        "typology_confusion": conf,
        "typology_unmatched": unmatched_out,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 200)
    print(out["typology_by_type"][out["typology_by_type"]["condition"] == "all"].to_string(index=False))
    print(out["typology_excess"].to_string(index=False))
    print(out["typology_interaction"].to_string(index=False))
    print(out["typology_agreement"].to_string(index=False))


if __name__ == "__main__":
    main()
