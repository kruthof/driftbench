"""Verbosity-controlled complexity analysis.

Priority 2: Separates complexity inflation from raw verbosity.
Shows that pressure increases structural complexity even after
controlling for output length.
"""

import re
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.formula.api import ols


def extract_length_features(text: str) -> dict:
    """Extract length-based features from a text response.

    Returns:
        dict with token_count (whitespace), sentence_count, section_count.
    """
    # Token count (whitespace-split approximation)
    tokens = text.split()
    token_count = len(tokens)

    # Sentence count (split on sentence-ending punctuation)
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Section/header count (markdown headers)
    headers = re.findall(r'^#{1,6}\s+.+', text, re.MULTILINE)
    # Also count numbered section headers like "1." or "1)" (no leading whitespace)
    numbered_headers = re.findall(r'^\d+[\.\)]\s+', text, re.MULTILINE)
    section_count = len(headers) + len(numbered_headers)

    # Bullet/list item count (indented numbered items only, to avoid
    # double-counting top-level numbered headers)
    bullets = re.findall(r'^[\s]*[-*+]\s+', text, re.MULTILINE)
    indented_numbered = re.findall(r'^\s+\d+[\.\)]\s+', text, re.MULTILINE)
    list_item_count = len(bullets) + len(indented_numbered)

    return {
        "token_count": token_count,
        "sentence_count": sentence_count,
        "section_count": section_count,
        "list_item_count": list_item_count,
    }


def extract_final_response(transcript: dict) -> str:
    """Extract the last assistant message (non-probe) from a transcript."""
    for msg in reversed(transcript.get("messages", [])):
        if msg.get("role") == "assistant" and not msg.get("is_probe", False):
            return msg.get("content", "")
    return ""


def extract_transcript_features(transcripts_dir: Path) -> pd.DataFrame:
    """Extract length features from all transcripts.

    Returns DataFrame with run_id plus length features.
    """
    rows = []
    for path in sorted(transcripts_dir.glob("*.jsonl")):
        try:
            with jsonlines.open(path) as reader:
                transcript = next(iter(reader))
        except Exception:
            continue

        final_text = extract_final_response(transcript)
        if not final_text:
            continue

        features = extract_length_features(final_text)
        meta = transcript["metadata"]

        # Also count total output tokens from metadata if available
        total_output_tokens = meta.get("total_output_tokens", 0)

        rows.append({
            "run_id": meta["run_id"],
            "brief_id": meta["brief_id"],
            "model_id": meta["model_id"],
            "condition": meta["condition"],
            "total_output_tokens": total_output_tokens,
            **features,
        })

    return pd.DataFrame(rows)


def extract_structural_counts_regex(text: str) -> dict:
    """Extract structural complexity counts using regex heuristics.

    This serves as a validation baseline for LLM-extracted counts.
    """
    text_lower = text.lower()

    # Count stages/phases
    stage_patterns = [
        r'\bstage\s+\d+',
        r'\bphase\s+\d+',
        r'\bstep\s+\d+',
        r'\bround\s+\d+',
    ]
    stage_count = sum(
        len(re.findall(p, text_lower)) for p in stage_patterns
    )
    # Also count numbered sections as a proxy
    numbered_sections = re.findall(r'^#{1,3}\s+\d+\.', text, re.MULTILINE)
    stage_count = max(stage_count, len(numbered_sections))

    # Count sub-experiments / analyses
    analysis_terms = [
        r'\bsub-experiment',
        r'\bsub-study',
        r'\bsecondary\s+analysis',
        r'\bexploratory\s+analysis',
        r'\bsensitivity\s+analysis',
        r'\brobustness\s+check',
        r'\bmediation\s+analysis',
        r'\bmoderation\s+analysis',
    ]
    analysis_count = sum(
        1 for p in analysis_terms if re.search(p, text_lower)
    )

    # Count datasets/resources mentioned
    resource_terms = [
        r'\bdataset',
        r'\bcorpus',
        r'\bsurvey\b',
        r'\bquestionnaire\b',
        r'\binterview',
        r'\bfocus\s+group',
        r'\bsensor',
        r'\bsatellite',
        r'\bregistry',
        r'\bbiobank',
    ]
    resource_count = sum(
        1 for p in resource_terms if re.search(p, text_lower)
    )

    # Count dependency markers
    dependency_terms = [
        r'\bdepends\s+on',
        r'\bcontingent\s+on',
        r'\bprerequisite',
        r'\brequires\s+(?:that|completion|results)',
        r'\binforms\s+(?:the|subsequent)',
        r'\bfeeds?\s+into',
    ]
    dependency_count = sum(
        len(re.findall(p, text_lower)) for p in dependency_terms
    )

    return {
        "regex_stage_count": stage_count,
        "regex_analysis_count": analysis_count,
        "regex_resource_count": resource_count,
        "regex_dependency_count": dependency_count,
        "regex_total_structural": stage_count + analysis_count + resource_count + dependency_count,
    }


def add_regex_structural_counts(features_df: pd.DataFrame, transcripts_dir: Path) -> pd.DataFrame:
    """Add regex-based structural counts to the features DataFrame."""
    structural_rows = []
    for path in sorted(transcripts_dir.glob("*.jsonl")):
        try:
            with jsonlines.open(path) as reader:
                transcript = next(iter(reader))
        except Exception:
            continue

        final_text = extract_final_response(transcript)
        if not final_text:
            continue

        counts = extract_structural_counts_regex(final_text)
        counts["run_id"] = transcript["metadata"]["run_id"]
        structural_rows.append(counts)

    structural_df = pd.DataFrame(structural_rows)
    if structural_df.empty:
        return features_df

    return features_df.merge(structural_df, on="run_id", how="left")


def merge_features_with_scores(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transcript features with judge/auditor scores."""
    return features_df.merge(scores_df, on="run_id", suffixes=("_feat", "_score"))


def run_verbosity_regression(
    merged_df: pd.DataFrame,
) -> dict:
    """Run the verbosity-controlled complexity regressions.

    Primary: complexity ~ condition + model + log_tokens + brief (ordinal)
    Secondary: fidelity ~ condition + model + complexity + log_tokens + brief

    Returns dict with model summaries and key statistics.
    """
    df = merged_df.copy()

    # Use token_count from transcript features
    df["log_tokens"] = np.log1p(df["token_count"])

    # Complexity column — use judge score
    complexity_col = "j_complexity_inflation"
    if complexity_col + "_score" in df.columns:
        complexity_col = complexity_col + "_score"
    elif complexity_col not in df.columns:
        return {"error": "no complexity column found"}

    fidelity_col = "j_objective_fidelity"
    if fidelity_col + "_score" in df.columns:
        fidelity_col = fidelity_col + "_score"

    # Condition and model columns
    cond_col = "condition_score" if "condition_score" in df.columns else "condition"
    model_col = "model_id_score" if "model_id_score" in df.columns else "model_id"
    brief_col = "brief_id_score" if "brief_id_score" in df.columns else "brief_id"

    # Drop rows with missing values
    cols_needed = [complexity_col, fidelity_col, "log_tokens", cond_col, model_col, brief_col]
    df = df.dropna(subset=[c for c in cols_needed if c in df.columns])

    if len(df) < 50:
        return {"error": f"insufficient data after merge: {len(df)} rows"}

    results = {}

    # --- Primary: OLS as a robustness check ---
    # complexity ~ condition + model + log_tokens
    # (Brief as fixed effect would consume too many df; use it as a cluster for robust SEs)
    try:
        df["_condition"] = pd.Categorical(
            df[cond_col],
            categories=["single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"],
        )
        df["_model"] = pd.Categorical(df[model_col])
        df["_brief"] = pd.Categorical(df[brief_col])
        df["_complexity"] = df[complexity_col].astype(float)
        df["_fidelity"] = df[fidelity_col].astype(float)

        # OLS with condition + model + log_tokens
        formula_primary = "_complexity ~ C(_condition) + C(_model) + log_tokens"
        ols_primary = ols(formula_primary, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["_brief"]}
        )
        results["ols_primary"] = {
            "summary": str(ols_primary.summary()),
            "r_squared": ols_primary.rsquared,
            "condition_pvalues": {
                k: v for k, v in ols_primary.pvalues.items()
                if "_condition" in str(k)
            },
            "log_tokens_coef": ols_primary.params.get("log_tokens", None),
            "log_tokens_pval": ols_primary.pvalues.get("log_tokens", None),
            "n": len(df),
        }
    except Exception as e:
        results["ols_primary"] = {"error": str(e)}

    # --- Primary: Ordinal regression ---
    try:
        # Prepare design matrix manually for OrderedModel
        # Must use float arrays to avoid dtype issues
        cond_dummies = pd.get_dummies(df["_condition"], prefix="cond", drop_first=True).astype(float)
        model_dummies = pd.get_dummies(df["_model"], prefix="model", drop_first=True).astype(float)
        log_tok = df[["log_tokens"]].astype(float).reset_index(drop=True)

        X = pd.concat([
            cond_dummies.reset_index(drop=True),
            model_dummies.reset_index(drop=True),
            log_tok,
        ], axis=1)
        y = df["_complexity"].astype(float).reset_index(drop=True)

        ordinal_model = OrderedModel(y, X, distr="logit")
        ordinal_result = ordinal_model.fit(method="bfgs", disp=False)
        results["ordinal_primary"] = {
            "summary": str(ordinal_result.summary()),
            "params": ordinal_result.params.to_dict(),
            "pvalues": ordinal_result.pvalues.to_dict(),
            "n": len(df),
        }
    except Exception as e:
        results["ordinal_primary"] = {"error": str(e)}

    # --- Secondary: fidelity ~ condition + model + complexity + log_tokens ---
    try:
        formula_secondary = "_fidelity ~ C(_condition) + C(_model) + _complexity + log_tokens"
        ols_secondary = ols(formula_secondary, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["_brief"]}
        )
        results["ols_secondary"] = {
            "summary": str(ols_secondary.summary()),
            "r_squared": ols_secondary.rsquared,
            "complexity_coef": ols_secondary.params.get("_complexity", None),
            "complexity_pval": ols_secondary.pvalues.get("_complexity", None),
            "n": len(df),
        }
    except Exception as e:
        results["ols_secondary"] = {"error": str(e)}

    # --- Partial correlation: complexity ~ condition, controlling for tokens ---
    try:
        partial_corrs = {}
        for cond_name in df[cond_col].unique():
            if cond_name == "single_shot":
                continue
            subset = df[df[cond_col].isin(["single_shot", cond_name])].copy()
            subset["is_treatment"] = (subset[cond_col] == cond_name).astype(float)
            # Partial Spearman: residualize both variables on log_tokens
            from scipy.stats import spearmanr
            # Residualize complexity on tokens
            resid_c = sm.OLS(subset["_complexity"], sm.add_constant(subset["log_tokens"])).fit().resid
            # Residualize treatment on tokens
            resid_t = sm.OLS(subset["is_treatment"], sm.add_constant(subset["log_tokens"])).fit().resid
            rho, pval = spearmanr(resid_c, resid_t)
            partial_corrs[f"single_shot_vs_{cond_name}"] = {
                "spearman_rho": rho,
                "p_value": pval,
                "n": len(subset),
            }
        results["partial_correlations"] = partial_corrs
    except Exception as e:
        results["partial_correlations"] = {"error": str(e)}

    return results


def complexity_vs_tokens_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the complexity vs tokens scatter plot."""
    df = merged_df.copy()
    complexity_col = "j_complexity_inflation"
    if complexity_col + "_score" in df.columns:
        complexity_col = complexity_col + "_score"

    cond_col = "condition_score" if "condition_score" in df.columns else "condition"
    model_col = "model_id_score" if "model_id_score" in df.columns else "model_id"

    return df[["run_id", cond_col, model_col, "token_count", complexity_col]].rename(
        columns={
            cond_col: "condition",
            model_col: "model_id",
            complexity_col: "complexity",
        }
    )


def run_verbosity_analysis(
    transcripts_dir: Path,
    scores_df: pd.DataFrame,
) -> dict:
    """Run the full verbosity analysis pipeline.

    1. Extract length features from all transcripts
    2. Add regex structural counts
    3. Merge with scores
    4. Run regressions

    Returns dict with all results.
    """
    # Step 1: Extract features
    features_df = extract_transcript_features(transcripts_dir)
    if features_df.empty:
        return {"error": "no transcript features extracted"}

    # Step 2: Add regex structural counts
    features_df = add_regex_structural_counts(features_df, transcripts_dir)

    # Step 3: Merge with scores
    merged = merge_features_with_scores(features_df, scores_df)
    if merged.empty:
        return {"error": "merge produced empty DataFrame"}

    # Step 4: Run regressions
    regression_results = run_verbosity_regression(merged)

    # Step 5: Prepare scatter plot data
    scatter_data = complexity_vs_tokens_data(merged)

    return {
        "features": features_df,
        "merged": merged,
        "regression": regression_results,
        "scatter_data": scatter_data,
    }
