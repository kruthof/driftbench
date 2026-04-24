"""Compare blind judge (no transcript) vs transcript-aware judge scores.

If scores agree, the positional bias does not drive substantive findings.
If they diverge, we learn what the transcript contributes to scoring.
"""

from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm
from statsmodels.formula.api import ols


SCORE_DIMS = [
    "objective_fidelity",
    "constraint_adherence",
    "alternative_coverage",
    "complexity_inflation",
]


def load_blind_scores(scores_dir: Path) -> pd.DataFrame:
    """Load all blind judge score files."""
    records = []
    for path in sorted(scores_dir.glob("blind_*.jsonl")):
        try:
            with jsonlines.open(path) as reader:
                for record in reader:
                    records.append({
                        "run_id": record["run_id"],
                        "blind_objective_fidelity": record["objective_fidelity"],
                        "blind_constraint_adherence": record["constraint_adherence"],
                        "blind_alternative_coverage": record["alternative_coverage"],
                        "blind_complexity_inflation": record["complexity_inflation"],
                    })
        except Exception:
            continue
    return pd.DataFrame(records)


def compare_blind_vs_aware(
    scores_df: pd.DataFrame,
    blind_df: pd.DataFrame,
) -> dict:
    """Compare blind judge scores with transcript-aware judge scores.

    Returns dict with agreement metrics and bias comparison.
    """
    # Merge on run_id
    merged = scores_df.merge(blind_df, on="run_id", how="inner")
    if merged.empty:
        return {"error": "no matching runs"}

    results = {"n": len(merged)}

    # Per-dimension agreement
    agreement_rows = []
    for dim in SCORE_DIMS:
        j_col = f"j_{dim}"
        b_col = f"blind_{dim}"
        if j_col not in merged.columns or b_col not in merged.columns:
            continue

        valid = merged[[j_col, b_col]].dropna()
        if len(valid) < 10:
            continue

        j_vals = valid[j_col].astype(int).values
        b_vals = valid[b_col].astype(int).values

        # Cohen's kappa
        kappa = cohen_kappa_score(j_vals, b_vals, labels=list(range(5)), weights="quadratic")

        # Spearman correlation
        rho, p = sp_stats.spearmanr(j_vals, b_vals)

        # Mean difference (bias direction)
        mean_diff = (b_vals - j_vals).mean()
        std_diff = (b_vals - j_vals).std()

        # Mean absolute error
        mae = np.abs(b_vals - j_vals).mean()

        agreement_rows.append({
            "metric": dim,
            "kappa": kappa,
            "spearman_rho": rho,
            "spearman_p": p,
            "mean_diff_blind_minus_aware": mean_diff,
            "std_diff": std_diff,
            "mae": mae,
            "n": len(valid),
        })

    results["agreement"] = pd.DataFrame(agreement_rows)

    # Check if blind judge has less positional bias
    if "token_count" in merged.columns:
        bias_rows = []
        cond_col = None
        for c in ["condition", "condition_score", "condition_feat"]:
            if c in merged.columns:
                cond_col = c
                break

        if cond_col:
            cond_dummies = pd.get_dummies(merged[cond_col], drop_first=True).astype(float)
            X_cond = sm.add_constant(cond_dummies.values)

        for dim in SCORE_DIMS:
            j_col = f"j_{dim}"
            b_col = f"blind_{dim}"
            if j_col not in merged.columns or b_col not in merged.columns:
                continue

            valid_idx = merged[[j_col, b_col, "token_count"]].dropna().index
            n = len(valid_idx)
            if n < 20:
                continue

            tokens = merged.loc[valid_idx, "token_count"].values.astype(float)

            if cond_col:
                X = X_cond[:n]
                resid_tokens = sm.OLS(tokens, X).fit().resid
                resid_j = sm.OLS(merged.loc[valid_idx, j_col].values.astype(float), X).fit().resid
                resid_b = sm.OLS(merged.loc[valid_idx, b_col].values.astype(float), X).fit().resid
            else:
                resid_tokens = tokens - tokens.mean()
                resid_j = merged.loc[valid_idx, j_col].values.astype(float)
                resid_b = merged.loc[valid_idx, b_col].values.astype(float)

            rho_j, _ = sp_stats.spearmanr(resid_tokens, resid_j)
            rho_b, _ = sp_stats.spearmanr(resid_tokens, resid_b)

            bias_rows.append({
                "metric": dim,
                "aware_bias_rho": rho_j,
                "blind_bias_rho": rho_b,
                "bias_reduction": abs(rho_j) - abs(rho_b),
                "n": n,
            })

        results["bias_comparison"] = pd.DataFrame(bias_rows)

    # Re-run key condition comparisons with blind scores
    if "condition" in merged.columns or "condition_score" in merged.columns:
        cond_col = "condition_score" if "condition_score" in merged.columns else "condition"
        condition_rows = []
        for dim in SCORE_DIMS:
            j_col = f"j_{dim}"
            b_col = f"blind_{dim}"

            for cond in ["single_shot", "multi_turn_pressure"]:
                cond_data = merged[merged[cond_col] == cond]
                if len(cond_data) < 5:
                    continue
                condition_rows.append({
                    "metric": dim,
                    "condition": cond,
                    "aware_mean": cond_data[j_col].mean(),
                    "blind_mean": cond_data[b_col].mean(),
                    "diff": cond_data[b_col].mean() - cond_data[j_col].mean(),
                    "n": len(cond_data),
                })

        results["condition_means"] = pd.DataFrame(condition_rows)

    return results


def run_blind_comparison(
    scores_dir: Path,
    aggregated_df: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """Run the full blind vs aware comparison."""
    blind_df = load_blind_scores(scores_dir)
    if blind_df.empty:
        return {"error": "no blind scores found", "n_blind": 0}

    # Merge features for token counts if available
    if features_df is not None:
        merged_scores = aggregated_df.merge(
            features_df[["run_id", "token_count"]],
            on="run_id",
            how="left",
        )
    else:
        merged_scores = aggregated_df

    results = compare_blind_vs_aware(merged_scores, blind_df)
    results["n_blind"] = len(blind_df)

    return results
