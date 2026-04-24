"""Score debiasing and averaging for judge/auditor scores.

Addresses positional bias (correlation between transcript length and scores)
and inter-rater noise by:
1. Averaging judge + auditor scores (reduces rater-specific bias)
2. Residualizing scores on token count (removes length effect)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm


SCORE_PAIRS = [
    ("j_objective_fidelity", "a_objective_fidelity", "objective_fidelity"),
    ("j_constraint_adherence", "a_constraint_adherence", "constraint_adherence"),
    ("j_alternative_coverage", "a_alternative_coverage", "alternative_coverage"),
    ("j_complexity_inflation", "a_complexity_inflation", "complexity_inflation"),
]


def compute_averaged_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Average judge and auditor scores for each dimension.

    Returns the input DataFrame with added avg_* columns.
    """
    df = df.copy()
    for j_col, a_col, name in SCORE_PAIRS:
        if j_col in df.columns and a_col in df.columns:
            df[f"avg_{name}"] = df[[j_col, a_col]].mean(axis=1)
    return df


def residualize_on_tokens(
    df: pd.DataFrame,
    token_col: str = "token_count",
    score_cols: list[str] | None = None,
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Remove the linear effect of token count from scores.

    For each score column, regresses score on log(token_count) within
    each condition (or overall if no condition column), then replaces
    the score with the residual + the grand mean. This preserves the
    original scale while removing length bias.

    Args:
        df: DataFrame with scores and token_count.
        token_col: Column with token counts.
        score_cols: Columns to residualize. If None, uses all j_*, a_*, avg_* columns.
        condition_col: If provided, residualize within each condition.

    Returns:
        DataFrame with added debiased_* columns.
    """
    df = df.copy()

    if score_cols is None:
        score_cols = [
            c for c in df.columns
            if c.startswith(("j_", "a_", "avg_"))
            and not c.endswith(("_summary", "_classification"))
            and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    if token_col not in df.columns:
        return df

    log_tokens = np.log1p(df[token_col].astype(float))

    for col in score_cols:
        if col not in df.columns:
            continue

        valid_mask = df[col].notna() & log_tokens.notna()
        debiased = df[col].copy().astype(float)
        grand_mean = df.loc[valid_mask, col].mean()

        if condition_col and condition_col in df.columns:
            # Residualize within each condition
            for cond in df[condition_col].unique():
                cond_mask = valid_mask & (df[condition_col] == cond)
                if cond_mask.sum() < 10:
                    continue
                y = df.loc[cond_mask, col].astype(float).values
                X = sm.add_constant(log_tokens[cond_mask].values)
                try:
                    resid = sm.OLS(y, X).fit().resid
                    cond_mean = y.mean()
                    debiased.loc[cond_mask] = resid + cond_mean
                except Exception:
                    pass
        else:
            # Residualize overall
            y = df.loc[valid_mask, col].astype(float).values
            X = sm.add_constant(log_tokens[valid_mask].values)
            try:
                resid = sm.OLS(y, X).fit().resid
                debiased.loc[valid_mask] = resid + grand_mean
            except Exception:
                pass

        # Clip to original scale bounds
        col_min = df[col].min()
        col_max = df[col].max()
        debiased = debiased.clip(lower=col_min, upper=col_max)

        short_name = col.replace("j_", "").replace("a_", "").replace("avg_", "")
        prefix = col.split("_")[0] + "_" if col.startswith(("j_", "a_")) else "avg_"
        df[f"debiased_{prefix}{short_name}"] = debiased

    return df


def check_debiasing_effectiveness(
    df: pd.DataFrame,
    token_col: str = "token_count",
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Compare positional bias before and after debiasing.

    Returns DataFrame showing partial Spearman rho with token count
    for original, averaged, and debiased scores.
    """
    rows = []

    # Build condition control matrix
    if condition_col and condition_col in df.columns:
        cond_dummies = pd.get_dummies(df[condition_col], drop_first=True).astype(float)
        X_cond = sm.add_constant(cond_dummies.values)
    else:
        X_cond = sm.add_constant(np.ones(len(df)))

    for j_col, a_col, name in SCORE_PAIRS:
        variants = {}
        # Original judge
        if j_col in df.columns:
            variants[f"judge_{name}"] = j_col
        # Original auditor
        if a_col in df.columns:
            variants[f"auditor_{name}"] = a_col
        # Averaged
        avg_col = f"avg_{name}"
        if avg_col in df.columns:
            variants[f"averaged_{name}"] = avg_col
        # Debiased judge
        deb_j = f"debiased_j_{name}"
        if deb_j in df.columns:
            variants[f"debiased_judge_{name}"] = deb_j
        # Debiased averaged
        deb_avg = f"debiased_avg_{name}"
        if deb_avg in df.columns:
            variants[f"debiased_avg_{name}"] = deb_avg

        for variant_name, col in variants.items():
            valid = df[[token_col, col]].dropna()
            if len(valid) < 20:
                continue

            n = len(valid)
            try:
                resid_tokens = sm.OLS(
                    valid[token_col].values.astype(float),
                    X_cond[:n],
                ).fit().resid
                resid_score = sm.OLS(
                    valid[col].values.astype(float),
                    X_cond[:n],
                ).fit().resid
                rho, p = sp_stats.spearmanr(resid_tokens, resid_score)
            except Exception:
                rho, p = np.nan, np.nan

            rows.append({
                "metric": name,
                "variant": variant_name.split("_")[0] if "_" in variant_name else variant_name,
                "column": col,
                "partial_rho": rho,
                "p_value": p,
                "n": n,
            })

    return pd.DataFrame(rows)


def rerun_reliability_with_averaged(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-rater reliability using averaged vs original scores.

    Since averaging removes the concept of two independent raters,
    we instead compare: does averaging improve the correlation with
    the auditor's drift classification (external criterion validity)?
    """
    from sklearn.metrics import cohen_kappa_score

    rows = []

    for j_col, a_col, name in SCORE_PAIRS:
        if j_col not in df.columns or a_col not in df.columns:
            continue

        valid = df[[j_col, a_col]].dropna()
        if len(valid) < 10:
            continue

        # Original kappa
        kappa_orig = cohen_kappa_score(
            valid[j_col].astype(int), valid[a_col].astype(int),
            labels=list(range(5)), weights="quadratic",
        )

        # Compute correlation of each version with drift classification
        # (as external criterion)
        avg_col = f"avg_{name}"
        if avg_col in df.columns and "a_drift_classification" in df.columns:
            drift_binary = df["a_drift_classification"].isin(
                ["trajectory_drift", "trajectory_lock_in"]
            ).astype(float)
            valid2 = df[[j_col, a_col, avg_col]].dropna()
            drift_valid = drift_binary[valid2.index]

            corr_j = sp_stats.spearmanr(valid2[j_col], drift_valid)[0]
            corr_a = sp_stats.spearmanr(valid2[a_col], drift_valid)[0]
            corr_avg = sp_stats.spearmanr(valid2[avg_col], drift_valid)[0]
        else:
            corr_j = corr_a = corr_avg = np.nan

        rows.append({
            "metric": name,
            "kappa_judge_auditor": kappa_orig,
            "judge_drift_corr": corr_j,
            "auditor_drift_corr": corr_a,
            "averaged_drift_corr": corr_avg,
            "n": len(valid),
        })

    return pd.DataFrame(rows)


def run_debiasing_analysis(
    scores_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict:
    """Run the full debiasing analysis pipeline.

    1. Merge scores with features (for token counts)
    2. Compute averaged scores
    3. Residualize on token count
    4. Check effectiveness
    5. Compare reliability

    Returns dict with all results.
    """
    # Merge
    merged = features_df.merge(scores_df, on="run_id", suffixes=("_feat", "_score"))
    if merged.empty:
        return {"error": "empty merge"}

    # Resolve condition column
    cond_col = None
    for candidate in ["condition", "condition_score", "condition_feat"]:
        if candidate in merged.columns:
            cond_col = candidate
            break

    # Step 1: Average scores
    merged = compute_averaged_scores(merged)

    # Step 2: Residualize
    merged = residualize_on_tokens(
        merged,
        token_col="token_count",
        condition_col=cond_col,
    )

    # Step 3: Check effectiveness
    effectiveness = check_debiasing_effectiveness(
        merged,
        token_col="token_count",
        condition_col=cond_col,
    )

    # Step 4: Reliability comparison
    reliability_comparison = rerun_reliability_with_averaged(merged)

    return {
        "merged_debiased": merged,
        "effectiveness": effectiveness,
        "reliability_comparison": reliability_comparison,
    }
