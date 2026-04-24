"""Statistical analysis for drift_bench results."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = [
    "j_objective_fidelity",
    "j_constraint_adherence",
    "j_alternative_coverage",
    "j_complexity_inflation",
    "a_recoverability",
]


def load_aggregated(path: Path) -> pd.DataFrame:
    """Load the aggregated Parquet table."""
    return pd.read_parquet(path)


def compute_condition_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition summary statistics for all scoring dimensions."""
    rows = []
    for condition in df["condition"].unique():
        cond_df = df[df["condition"] == condition]
        for metric in METRICS:
            if metric not in cond_df.columns:
                continue
            values = cond_df[metric].dropna()
            n = len(values)
            if n == 0:
                continue
            mean = values.mean()
            std = values.std(ddof=1)
            if n > 1:
                ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / (n**0.5))
            else:
                ci = (mean, mean)
            rows.append({
                "condition": condition,
                "metric": metric,
                "mean": mean,
                "std": std,
                "n": n,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            })
    return pd.DataFrame(rows)


def pairwise_condition_tests(
    df: pd.DataFrame,
    baseline: str = "single_shot",
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests comparing each condition to baseline.

    Uses Holm-Bonferroni correction for multiple comparisons.
    Effect size: rank-biserial correlation r = 1 - (2W)/(n*(n+1)).
    """
    test_metrics = [m for m in METRICS if m != "a_recoverability"]
    rows = []

    baseline_df = df[df["condition"] == baseline]

    for condition in df["condition"].unique():
        if condition == baseline:
            continue
        cond_df = df[df["condition"] == condition]

        bl_grouped = baseline_df.groupby(["brief_id", "model_id"])[test_metrics].mean()
        cd_grouped = cond_df.groupby(["brief_id", "model_id"])[test_metrics].mean()
        paired = bl_grouped.join(cd_grouped, lsuffix="_bl", rsuffix="_cd", how="inner")

        for metric in test_metrics:
            bl_col = f"{metric}_bl"
            cd_col = f"{metric}_cd"
            if bl_col not in paired.columns or cd_col not in paired.columns:
                continue
            bl_vals = paired[bl_col].values
            cd_vals = paired[cd_col].values
            n = len(bl_vals)

            if n < 5:
                rows.append({
                    "condition": condition,
                    "metric": metric,
                    "statistic": None,
                    "p_value": None,
                    "p_value_corrected": None,
                    "effect_size_r": None,
                    "n_pairs": n,
                    "note": "too few pairs",
                })
                continue

            diffs = bl_vals - cd_vals
            if np.all(diffs == 0):
                rows.append({
                    "condition": condition,
                    "metric": metric,
                    "statistic": None,
                    "p_value": None,
                    "p_value_corrected": None,
                    "effect_size_r": None,
                    "n_pairs": n,
                    "note": "identical values",
                })
                continue

            try:
                stat_result = stats.wilcoxon(bl_vals, cd_vals, zero_method="wilcox")
                p = stat_result.pvalue
                # Compute T+ manually from ranked differences.
                # scipy.stats.wilcoxon returns min(T+, T-), not T+.
                nonzero_diffs = diffs[diffs != 0]
                n_eff = len(nonzero_diffs)
                ranks = stats.rankdata(np.abs(nonzero_diffs))
                W_plus = np.sum(ranks[nonzero_diffs > 0])
                total_rank_sum = n_eff * (n_eff + 1) / 2
                W_minus = total_rank_sum - W_plus
                # Rank-biserial r: r = (W+ - W-) / (W+ + W-)
                r = (W_plus - W_minus) / total_rank_sum if total_rank_sum > 0 else 0.0
                rows.append({
                    "condition": condition,
                    "metric": metric,
                    "statistic": W_plus,
                    "p_value": p,
                    "p_value_corrected": None,  # filled below
                    "effect_size_r": r,
                    "n_pairs": n,
                    "n_effective": n_eff,
                })
            except ValueError:
                rows.append({
                    "condition": condition,
                    "metric": metric,
                    "statistic": None,
                    "p_value": None,
                    "p_value_corrected": None,
                    "effect_size_r": None,
                    "n_pairs": n,
                    "note": "test failed",
                })

    result_df = pd.DataFrame(rows)

    # Holm-Bonferroni correction on all valid p-values
    valid_mask = result_df["p_value"].notna()
    if valid_mask.any():
        from statsmodels.stats.multitest import multipletests
        p_vals = result_df.loc[valid_mask, "p_value"].values
        _, corrected, _, _ = multipletests(p_vals, method="holm")
        result_df.loc[valid_mask, "p_value_corrected"] = corrected

    return result_df


def compute_inter_rater_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quadratic-weighted Cohen's kappa between judge and auditor scores."""
    from sklearn.metrics import cohen_kappa_score

    metric_pairs = [
        ("j_objective_fidelity", "a_objective_fidelity", list(range(5))),
        ("j_constraint_adherence", "a_constraint_adherence", list(range(5))),
        ("j_alternative_coverage", "a_alternative_coverage", list(range(5))),
        ("j_complexity_inflation", "a_complexity_inflation", list(range(5))),
    ]
    rows = []
    for j_col, a_col, label_set in metric_pairs:
        if j_col not in df.columns or a_col not in df.columns:
            continue
        valid = df[[j_col, a_col]].dropna()
        if len(valid) < 10:
            rows.append({"metric": j_col.replace("j_", ""), "kappa": None, "n": len(valid)})
            continue
        kappa = cohen_kappa_score(
            valid[j_col].astype(int),
            valid[a_col].astype(int),
            labels=label_set,
            weights="quadratic",
        )
        rows.append({
            "metric": j_col.replace("j_", ""),
            "kappa": kappa,
            "n": len(valid),
        })
    return pd.DataFrame(rows)


def compute_drift_classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize drift classifications by condition and model."""
    if "a_drift_classification" not in df.columns:
        return pd.DataFrame()
    group = (
        df.groupby(["condition", "model_id", "a_drift_classification"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby(["condition", "model_id"]).size().reset_index(name="total")
    merged = group.merge(totals, on=["condition", "model_id"])
    merged["proportion"] = merged["count"] / merged["total"]
    return merged
