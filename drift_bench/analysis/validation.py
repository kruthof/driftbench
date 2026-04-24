"""LLM-judge validation analyses.

Priority 4: Positional bias check, structural count validation,
and human spot-check infrastructure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def positional_bias_check(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """Check for positional bias in LLM judge scoring.

    Tests whether transcript length (token count) predicts judge scores
    after controlling for condition. A significant residual correlation
    indicates the judge is biased by transcript length.

    Args:
        merged_df: DataFrame with token_count and j_* score columns.
                   Expected from verbosity.merge_features_with_scores().

    Returns:
        DataFrame with one row per metric: metric, raw_spearman_rho,
        raw_p_value, partial_spearman_rho, partial_p_value, n.
    """
    score_cols = {
        "j_objective_fidelity": "objective_fidelity",
        "j_constraint_adherence": "constraint_adherence",
        "j_alternative_coverage": "alternative_coverage",
        "j_complexity_inflation": "complexity_inflation",
    }

    # Resolve column names (may have _score suffix from merge)
    resolved = {}
    for col, label in score_cols.items():
        if col + "_score" in merged_df.columns:
            resolved[col + "_score"] = label
        elif col in merged_df.columns:
            resolved[col] = label

    # Find the condition column — may have suffix from merge
    cond_col = None
    for candidate in ["condition", "condition_score", "condition_feat"]:
        if candidate in merged_df.columns:
            cond_col = candidate
            break
    if cond_col is None:
        return pd.DataFrame()

    rows = []
    for col, label in resolved.items():
        valid = merged_df[["token_count", col, cond_col]].dropna().copy()
        if len(valid) < 20:
            continue

        # Raw Spearman correlation
        rho_raw, p_raw = sp_stats.spearmanr(valid["token_count"], valid[col])

        # Partial correlation: residualize both on condition dummies
        cond_dummies = pd.get_dummies(valid[cond_col], drop_first=True).astype(float)
        X_cond = sm.add_constant(cond_dummies.values.astype(float))

        try:
            resid_tokens = sm.OLS(
                valid["token_count"].values.astype(float), X_cond
            ).fit().resid
            resid_score = sm.OLS(
                valid[col].values.astype(float), X_cond
            ).fit().resid
            rho_partial, p_partial = sp_stats.spearmanr(resid_tokens, resid_score)
        except Exception:
            rho_partial, p_partial = np.nan, np.nan

        rows.append({
            "metric": label,
            "raw_spearman_rho": rho_raw,
            "raw_p_value": p_raw,
            "partial_spearman_rho": rho_partial,
            "partial_p_value": p_partial,
            "n": len(valid),
        })

    return pd.DataFrame(rows)


def select_human_spotcheck_sample(
    scores_df: pd.DataFrame,
    n_per_condition: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a stratified sample for human spot-checking.

    Stratified by condition, balanced across models and drift severity.

    Args:
        scores_df: Aggregated scores DataFrame.
        n_per_condition: Number of runs to sample per condition.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with selected run_ids and their metadata.
    """
    rng = np.random.RandomState(seed)
    selected = []

    conditions = sorted(scores_df["condition"].dropna().unique())

    for cond in conditions:
        cond_df = scores_df[scores_df["condition"] == cond]
        models = sorted(cond_df["model_id"].dropna().unique())

        if len(models) == 0:
            continue

        # Distribute n_per_condition across models as evenly as possible
        per_model = max(1, n_per_condition // len(models))
        remainder = n_per_condition - per_model * len(models)

        for i, model in enumerate(models):
            n_this = per_model + (1 if i < remainder else 0)
            model_df = cond_df[cond_df["model_id"] == model]

            if len(model_df) <= n_this:
                selected.append(model_df)
            else:
                # Stratify by drift severity: try to get a mix
                if "a_drift_classification" in model_df.columns:
                    # Sample proportionally from each drift class
                    sampled = model_df.groupby("a_drift_classification").apply(
                        lambda x: x.sample(
                            n=min(len(x), max(1, n_this // len(model_df["a_drift_classification"].unique()))),
                            random_state=rng,
                        ),
                        include_groups=False,
                    ).reset_index(drop=True)
                    if len(sampled) < n_this:
                        remaining = model_df[~model_df["run_id"].isin(sampled["run_id"])]
                        extra = remaining.sample(
                            n=min(len(remaining), n_this - len(sampled)),
                            random_state=rng,
                        )
                        sampled = pd.concat([sampled, extra])
                    selected.append(sampled.head(n_this))
                else:
                    selected.append(model_df.sample(n=n_this, random_state=rng))

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)
    return result


def validate_structural_counts(
    merged_df: pd.DataFrame,
    sample_size: int = 30,
    seed: int = 42,
) -> dict:
    """Validate LLM-extracted structural counts against regex heuristic.

    Compares regex_total_structural against LLM-extracted counts
    (if available) on a random sample.

    Args:
        merged_df: DataFrame with both regex and LLM structural counts.
        sample_size: Number of runs to validate.
        seed: Random seed.

    Returns:
        dict with correlation metrics and sample details.
    """
    # Check for regex columns
    regex_col = "regex_total_structural"
    if regex_col not in merged_df.columns:
        return {"error": "no regex structural counts found"}

    # Check for LLM columns
    llm_col = "llm_total_structural"
    if llm_col not in merged_df.columns:
        # If no LLM counts yet, just report regex distribution
        valid = merged_df[regex_col].dropna()
        return {
            "status": "regex_only",
            "regex_mean": valid.mean(),
            "regex_std": valid.std(),
            "regex_median": valid.median(),
            "n": len(valid),
            "note": "LLM structural counts not yet extracted. Regex baseline available.",
        }

    # Both available — compute agreement
    valid = merged_df[[regex_col, llm_col]].dropna()
    if len(valid) < 10:
        return {"error": f"insufficient data: {len(valid)} rows"}

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(valid), size=min(sample_size, len(valid)), replace=False)
    sample = valid.iloc[sample_idx]

    rho, p_rho = sp_stats.spearmanr(sample[regex_col], sample[llm_col])
    r, p_r = sp_stats.pearsonr(sample[regex_col], sample[llm_col])
    mae = np.abs(sample[regex_col] - sample[llm_col]).mean()

    return {
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "pearson_r": r,
        "pearson_p": p_r,
        "mean_absolute_error": mae,
        "sample_size": len(sample),
    }


def run_validation_analysis(
    scores_df: pd.DataFrame,
    merged_df: pd.DataFrame | None = None,
) -> dict:
    """Run all validation analyses.

    Args:
        scores_df: Aggregated scores DataFrame.
        merged_df: Optional features+scores merged DataFrame
                   (from verbosity analysis).

    Returns:
        dict with validation results.
    """
    results = {}

    # Human spot-check sample
    results["spotcheck_sample"] = select_human_spotcheck_sample(scores_df)

    # Positional bias check (requires merged_df with token counts)
    if merged_df is not None and "token_count" in merged_df.columns:
        results["positional_bias"] = positional_bias_check(merged_df)
    else:
        results["positional_bias"] = pd.DataFrame()

    # Structural count validation (requires merged_df with regex counts)
    if merged_df is not None:
        results["structural_validation"] = validate_structural_counts(merged_df)
    else:
        results["structural_validation"] = {"status": "no merged data available"}

    return results
