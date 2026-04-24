"""Inter-rater reliability analysis between judge and auditor scores.

Priority 1: This gates all other analyses. If agreement is weak,
the measurement apparatus is suspect.
"""

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import cohen_kappa_score


SCORE_PAIRS = [
    ("j_objective_fidelity", "a_objective_fidelity"),
    ("j_constraint_adherence", "a_constraint_adherence"),
    ("j_alternative_coverage", "a_alternative_coverage"),
    ("j_complexity_inflation", "a_complexity_inflation"),
]

LABEL_SET = list(range(5))  # 0-4 scale


def compute_cohens_kappa(
    df: pd.DataFrame,
    per_condition: bool = True,
) -> pd.DataFrame:
    """Quadratic-weighted Cohen's kappa between judge and auditor.

    Args:
        df: Aggregated DataFrame with j_* and a_* columns.
        per_condition: If True, also compute kappa per condition.

    Returns:
        DataFrame with columns: metric, condition, kappa, n.
    """
    rows = []

    for j_col, a_col in SCORE_PAIRS:
        metric_name = j_col.replace("j_", "")
        # Overall
        valid = df[[j_col, a_col]].dropna()
        if len(valid) >= 10:
            kappa = cohen_kappa_score(
                valid[j_col].astype(int),
                valid[a_col].astype(int),
                labels=LABEL_SET,
                weights="quadratic",
            )
            rows.append({
                "metric": metric_name,
                "condition": "all",
                "kappa": kappa,
                "n": len(valid),
            })

        # Per condition
        if per_condition:
            for cond in sorted(df["condition"].dropna().unique()):
                cond_df = df[df["condition"] == cond]
                valid_c = cond_df[[j_col, a_col]].dropna()
                if len(valid_c) >= 10:
                    kappa_c = cohen_kappa_score(
                        valid_c[j_col].astype(int),
                        valid_c[a_col].astype(int),
                        labels=LABEL_SET,
                        weights="quadratic",
                    )
                    rows.append({
                        "metric": metric_name,
                        "condition": cond,
                        "kappa": kappa_c,
                        "n": len(valid_c),
                    })

    return pd.DataFrame(rows)


def compute_krippendorff_alpha(
    df: pd.DataFrame,
    per_condition: bool = True,
) -> pd.DataFrame:
    """Krippendorff's alpha (interval-weighted) between judge and auditor.

    Args:
        df: Aggregated DataFrame with j_* and a_* columns.
        per_condition: If True, also compute alpha per condition.

    Returns:
        DataFrame with columns: metric, condition, alpha, n.
    """
    rows = []

    for j_col, a_col in SCORE_PAIRS:
        metric_name = j_col.replace("j_", "")
        # Overall
        valid = df[[j_col, a_col]].dropna()
        if len(valid) >= 10:
            # krippendorff expects a reliability matrix: raters x units
            reliability_data = np.array([
                valid[j_col].values,
                valid[a_col].values,
            ], dtype=float)
            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="interval",
            )
            rows.append({
                "metric": metric_name,
                "condition": "all",
                "alpha": alpha,
                "n": len(valid),
            })

        if per_condition:
            for cond in sorted(df["condition"].dropna().unique()):
                cond_df = df[df["condition"] == cond]
                valid_c = cond_df[[j_col, a_col]].dropna()
                if len(valid_c) >= 10:
                    reliability_data_c = np.array([
                        valid_c[j_col].values,
                        valid_c[a_col].values,
                    ], dtype=float)
                    alpha_c = krippendorff.alpha(
                        reliability_data=reliability_data_c,
                        level_of_measurement="interval",
                    )
                    rows.append({
                        "metric": metric_name,
                        "condition": cond,
                        "alpha": alpha_c,
                        "n": len(valid_c),
                    })

    return pd.DataFrame(rows)


def bootstrap_reliability(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for both Cohen's kappa and Krippendorff's alpha.

    Returns:
        DataFrame with columns: metric, kappa, kappa_ci_lower, kappa_ci_upper,
                                alpha, alpha_ci_lower, alpha_ci_upper, n.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for j_col, a_col in SCORE_PAIRS:
        metric_name = j_col.replace("j_", "")
        valid = df[[j_col, a_col]].dropna()
        n = len(valid)
        if n < 10:
            continue

        j_vals = valid[j_col].astype(int).values
        a_vals = valid[a_col].astype(int).values

        # Point estimates
        kappa = cohen_kappa_score(j_vals, a_vals, labels=LABEL_SET, weights="quadratic")
        rel_data = np.array([j_vals.astype(float), a_vals.astype(float)])
        alpha = krippendorff.alpha(reliability_data=rel_data, level_of_measurement="interval")

        # Bootstrap
        kappa_boots = []
        alpha_boots = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            j_boot = j_vals[idx]
            a_boot = a_vals[idx]
            try:
                k = cohen_kappa_score(j_boot, a_boot, labels=LABEL_SET, weights="quadratic")
                kappa_boots.append(k)
            except Exception:
                pass
            try:
                rd = np.array([j_boot.astype(float), a_boot.astype(float)])
                a_val = krippendorff.alpha(reliability_data=rd, level_of_measurement="interval")
                alpha_boots.append(a_val)
            except Exception:
                pass

        kappa_boots = np.array(kappa_boots)
        alpha_boots = np.array(alpha_boots)

        rows.append({
            "metric": metric_name,
            "kappa": kappa,
            "kappa_ci_lower": np.percentile(kappa_boots, 2.5) if len(kappa_boots) > 0 else np.nan,
            "kappa_ci_upper": np.percentile(kappa_boots, 97.5) if len(kappa_boots) > 0 else np.nan,
            "alpha": alpha,
            "alpha_ci_lower": np.percentile(alpha_boots, 2.5) if len(alpha_boots) > 0 else np.nan,
            "alpha_ci_upper": np.percentile(alpha_boots, 97.5) if len(alpha_boots) > 0 else np.nan,
            "n": n,
        })

    return pd.DataFrame(rows)


def drift_classification_agreement(df: pd.DataFrame) -> dict:
    """Agreement on drift classification between judge-derived and auditor.

    The judge does not produce a drift classification directly.
    We derive one from judge scores: if any dimension (constraint_adherence
    or alternative_coverage) <= 2, mark as potential drift.
    Then compare against auditor's drift_classification.

    Returns dict with percent agreement and Cohen's kappa.
    """
    if "a_drift_classification" not in df.columns:
        return {"error": "no auditor drift classification"}

    valid = df[["j_constraint_adherence", "j_alternative_coverage",
                "j_complexity_inflation", "a_drift_classification"]].dropna()
    if len(valid) < 10:
        return {"error": "insufficient data", "n": len(valid)}

    # Derive judge-side binary drift indicator
    # "drifted" = constraint_adherence <= 2 OR alternative_coverage <= 2
    judge_drift = (
        (valid["j_constraint_adherence"] <= 2) |
        (valid["j_alternative_coverage"] <= 2)
    ).astype(int)

    # Auditor binary: trajectory_drift or trajectory_lock_in = 1, else 0
    auditor_drift = valid["a_drift_classification"].isin(
        ["trajectory_drift", "trajectory_lock_in"]
    ).astype(int)

    agreement = (judge_drift == auditor_drift).mean()
    kappa = cohen_kappa_score(judge_drift, auditor_drift)

    return {
        "percent_agreement": agreement,
        "kappa": kappa,
        "n": len(valid),
        "judge_drift_rate": judge_drift.mean(),
        "auditor_drift_rate": auditor_drift.mean(),
    }


def run_reliability_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    """Run all reliability analyses and return results."""
    return {
        "cohens_kappa": compute_cohens_kappa(df),
        "krippendorff_alpha": compute_krippendorff_alpha(df),
        "bootstrap_ci": bootstrap_reliability(df),
        "drift_agreement": drift_classification_agreement(df),
    }
