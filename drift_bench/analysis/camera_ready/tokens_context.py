"""Tokens and context length (plan item C9, a reviewer question).

Questions answered from disk:

1. How much of its context window does any run use? (July: max 6.27%.)
2. Within the pressure condition, do tokens predict drift once the
   model is held fixed? July: no. July's p = 0.35 is the OLS of
   constraint adherence on log output tokens with model fixed effects
   and brief-clustered SEs (beta = -0.19, n = 536); the matching row
   here is "ols adherence ~ log_output_tokens + model FE".
3. Does the pressure effect on adherence survive a verbosity control?
   (July: beta = -0.314, p = 2.6e-17, n = 2,127, controlling for the
   word count of the judged final response, as in verbosity.py.)
4. Do longer briefs use more tokens?
5. B10: does output length predict fidelity, per provider group
   (commercial vs open-weight)? The submitted text: null for commercial
   models (p = 0.31), significant for open-weight models (p < 0.001).
   Section ``fidelity_length`` of the summary.

Inputs: both transcript trees, the parquet (clean sample: the 30
stray Sonnet rows dropped by `sample.py`),
`analysis/final_probe_accuracy.csv`, the brief YAMLs.
Outputs: `tokens_context_runs.csv`, `tokens_context_summary.csv`.

Context size at the final call is not logged per call. The runner
stores only the sum of input tokens over all calls. We estimate the
final call's prompt from the visible text of every message in it:
characters times a per-model tokens-per-character ratio. The ratio
comes from single-shot runs, where the one call's input tokens and the
prompt characters are both known. Assistant turns are estimated from
their visible characters too, times an assistant ratio fitted on the
provider's multi-turn prompt counts, not from their logged
`token_count`: for reasoning models the logged count includes thinking
tokens, which are never re-sent (Gemini 3.1 Pro logs about 2.6x its
re-sent text). The logged counts enter only the output-only share. We check the
estimate by predicting every multi-turn run's summed input tokens and
comparing to the logged sum.

Context windows come from `litellm.get_model_info(model)["max_input_tokens"]`
(litellm 1.83.x), with a hard-coded fallback for models litellm does not
map and for offline use. Sonnet 4.6 at the 200k standard window is a
sensitivity row: the April runs sent no beta header, and the window the
API applied could not be confirmed from the logs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.b4_common import (
    DEFAULT_BRIEFS_DIR,
    DEFAULT_DATA_ROOT,
    MODEL_ORDER,
    attach_kbv,
    default_out_dir,
    final_probe_table,
    load_briefs,
    load_scores_with_filter,
    load_transcript,
    transcript_files,
    word_count,
    write_csv,
)
from drift_bench.analysis.camera_ready.sample import drop_stray

# Fallback windows, in tokens. These are the values litellm 1.83.8
# reports for max_input_tokens, plus Together's documented window for
# Qwen3-235B-A22B-Instruct-2507, which litellm does not map.
QWEN_SOURCE_URL = "https://docs.together.ai/docs/serverless-models"
CONTEXT_WINDOWS_FALLBACK = {
    "openai/gpt-5.4": 1_050_000,
    "openai/gpt-5.4-mini": 272_000,
    "anthropic/claude-sonnet-4-6": 1_000_000,
    "gemini/gemini-3.1-pro-preview": 1_048_576,
    "gemini/gemini-3.1-flash-lite-preview": 1_048_576,
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": 131_072,
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": 262_144,
}
HARDCODED_WINDOWS = {
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": (262_144, QWEN_SOURCE_URL),
}

# Sensitivity set: Sonnet 4.6 at its 200k standard window. The April
# runs sent no `context-1m` beta header, so the applied default could
# not be confirmed from the logs.
SONNET_ID = "anthropic/claude-sonnet-4-6"
SONNET_STANDARD_WINDOW = 200_000

PRESSURE = "multi_turn_pressure"

# B10: the submitted fidelity regression, split by provider group.
# Open-weight subjects are the two Together-hosted models.
OPEN_WEIGHT_PREFIX = "together_ai/"
LENGTH_GROUPS = ("all", "commercial", "open_weight")
# Length measures. The submitted regression (verbosity.py, secondary model)
# used log1p of the whitespace word count of the judged final response.
LENGTH_MEASURES = (
    ("log_final_words", "log1p of the whitespace word count of the judged final response (the submitted control, verbosity.py)"),
    ("log_output_tokens", "log of metadata.total_output_tokens, summed over every call of the run (main and probe)"),
)
# Specs: (label, condition filter, right-hand side without the length term, terms reported, note).
LENGTH_SPECS = (
    ("published", None, "C(condition) + complexity + C(model_id)", ("complexity",),
     "all four conditions, condition and model fixed effects, complexity controlled (the submitted spec)"),
    ("pressure", PRESSURE, "complexity + C(model_id)", ("complexity",),
     "pressure runs only, model fixed effects, complexity controlled"),
    ("pressure_raw", PRESSURE, "C(model_id)", (),
     "pressure runs only, model fixed effects, no complexity control"),
)


def length_group_mask(runs: pd.DataFrame, group: str) -> pd.Series:
    """Rows of ``runs`` in ``group``: all, commercial or open_weight."""
    is_open = runs["model_id"].astype(str).str.startswith(OPEN_WEIGHT_PREFIX)
    if group == "all":
        return pd.Series(True, index=runs.index)
    if group == "commercial":
        return ~is_open
    if group == "open_weight":
        return is_open
    raise ValueError(f"unknown group {group!r}")


def _group_note(group: str) -> str:
    if group == "all":
        return "all seven models"
    if group == "commercial":
        return f"models not hosted on {OPEN_WEIGHT_PREFIX}"
    return f"models hosted on {OPEN_WEIGHT_PREFIX}"


def fidelity_length_rows(runs: pd.DataFrame) -> list[dict]:
    """B10: does response length predict fidelity, per provider group?

    The submitted paper's secondary regression (verbosity.py) is
    ``fidelity ~ condition + model + complexity + log1p(final words)``
    with brief-clustered SEs over all four conditions. It reported the
    length term as null for commercial models (p = 0.31) and significant
    for open-weight models (p < 0.001). This function fits that spec and
    two pressure-only specs per group and length measure. Each fit yields
    a row for the length term and, when complexity is in the model, a row
    for the complexity term. Row name: ``ols fidelity <spec> <measure>: <term>``.
    """
    df = runs.copy()
    if "log_final_words" not in df.columns:
        df["log_final_words"] = np.log1p(pd.to_numeric(df["final_response_words"], errors="coerce"))
    if "log_output_tokens" not in df.columns:
        df["log_output_tokens"] = np.log(pd.to_numeric(df["total_output_tokens"], errors="coerce"))
    df["complexity"] = pd.to_numeric(df["j_complexity_inflation"], errors="coerce")
    df["fidelity"] = pd.to_numeric(df["j_objective_fidelity"], errors="coerce")
    rows: list[dict] = []
    for group in LENGTH_GROUPS:
        g_all = df[length_group_mask(df, group)]
        for tok, tok_note in LENGTH_MEASURES:
            for spec, cond, rhs, extra_terms, spec_note in LENGTH_SPECS:
                data = g_all if cond is None else g_all[g_all["condition"] == cond]
                formula = f"fidelity ~ {tok} + {rhs}"
                for term in (tok, *extra_terms):
                    fit = fit_ols_term(data, formula, term)
                    rows.append(_row(
                        "fidelity_length",
                        f"ols fidelity {spec} {tok}: {term}",
                        group, fit["estimate"], fit["ci_low"], fit["ci_high"], fit["p_value"], fit["n"],
                        f"{fit['note']}; formula {formula}; {spec_note}; {tok} = {tok_note}; "
                        f"group {group} = {_group_note(group)}",
                    ))
    return rows


def resolve_context_windows(models=MODEL_ORDER) -> tuple[dict[str, int], dict[str, str]]:
    """Context window per model and the source of each value.

    Order: hard-coded (Qwen, with URL), then litellm `max_input_tokens`,
    then the fallback table.
    """
    windows: dict[str, int] = {}
    sources: dict[str, str] = {}
    try:
        import litellm
        from importlib.metadata import version as _pkg_version
        try:
            version = _pkg_version("litellm")
        except Exception:
            version = "unknown"
    except Exception:  # litellm missing or broken
        litellm = None
        version = "unavailable"
    for m in models:
        if m in HARDCODED_WINDOWS:
            windows[m], sources[m] = HARDCODED_WINDOWS[m]
            continue
        value = None
        if litellm is not None:
            try:
                value = litellm.get_model_info(m).get("max_input_tokens")
            except Exception:
                value = None
        if value:
            windows[m] = int(value)
            sources[m] = f"litellm {version} get_model_info max_input_tokens"
        else:
            windows[m] = CONTEXT_WINDOWS_FALLBACK[m]
            sources[m] = "fallback table (litellm value unavailable)"
    return windows, sources


CONTEXT_WINDOWS, CONTEXT_WINDOW_SOURCES = resolve_context_windows()
CONTEXT_WINDOWS_SONNET_200K = dict(CONTEXT_WINDOWS, **{SONNET_ID: SONNET_STANDARD_WINDOW})


# ------------------------------------------------------------ per run


def summarise_transcript(transcript: dict) -> dict:
    """Token bookkeeping for one transcript.

    Walks the messages in order and reconstructs the prompt of every
    API call. Probe calls see the main conversation so far plus the
    probe question. Main calls see the main conversation so far.
    Returns character and token tallies; the caller converts characters
    to tokens with a calibrated ratio.
    """
    meta = transcript["metadata"]
    messages = transcript["messages"]

    ctx_chars = 0  # characters of system/user text in the main context
    ctx_assistant_chars = 0  # visible characters of assistant turns in context
    call_chars: list[int] = []  # system/user characters in each call's prompt
    call_assistant_chars: list[int] = []
    call_assistant_tokens: list[int] = []  # logged counts, thinking included
    ctx_assistant_tokens = 0
    main_outputs: list[int] = []  # logged token counts
    main_output_chars: list[int] = []
    probe_outputs: list[int] = []
    final_prompt_chars = 0
    final_prompt_assistant_chars = 0
    final_response_words = 0

    for msg in messages:
        is_probe = bool(msg.get("is_probe"))
        tc = msg.get("token_count") or 0
        if msg["role"] == "assistant":
            if is_probe:
                probe_outputs.append(tc)
            else:
                call_chars.append(ctx_chars)
                call_assistant_chars.append(ctx_assistant_chars)
                call_assistant_tokens.append(ctx_assistant_tokens)
                final_prompt_chars = ctx_chars
                final_prompt_assistant_chars = ctx_assistant_chars
                final_response_words = len(msg["content"].split())
                main_outputs.append(tc)
                main_output_chars.append(len(msg["content"]))
                ctx_assistant_chars += len(msg["content"])
                ctx_assistant_tokens += tc
        else:
            if is_probe:
                # The probe call: main context plus the probe question.
                call_chars.append(ctx_chars + len(msg["content"]))
                call_assistant_chars.append(ctx_assistant_chars)
                call_assistant_tokens.append(ctx_assistant_tokens)
            else:
                ctx_chars += len(msg["content"])

    return {
        "run_id": meta["run_id"],
        "brief_id": meta["brief_id"],
        "model_id": meta["model_id"],
        "condition": meta["condition"],
        "total_input_tokens": meta.get("total_input_tokens"),
        "total_output_tokens": meta.get("total_output_tokens"),
        "n_calls": len(call_chars),
        "n_main_turns": len(main_outputs),
        "n_probe_calls": len(probe_outputs),
        "main_output_tokens": int(sum(main_outputs)),
        "main_output_chars": int(sum(main_output_chars)),
        "probe_output_tokens": int(sum(probe_outputs)),
        "last_output_tokens": int(main_outputs[-1]) if main_outputs else 0,
        "last_output_chars": int(main_output_chars[-1]) if main_output_chars else 0,
        # Whitespace word count of the judged final response. This is the
        # verbosity control the published analysis used (verbosity.py,
        # transcript_features token_count).
        "final_response_words": int(final_response_words),
        "prompt_chars_single": int(call_chars[0]) if call_chars else 0,
        "calls_prompt_chars": int(sum(call_chars)),
        "calls_prompt_assistant_chars": int(sum(call_assistant_chars)),
        "calls_prompt_assistant_tokens_logged": int(sum(call_assistant_tokens)),
        # Prompt of the final main call, before its own output.
        "final_prompt_chars": int(final_prompt_chars),
        "final_prompt_assistant_chars": int(final_prompt_assistant_chars),
    }


def calibrate_tokens_per_char(runs: pd.DataFrame) -> pd.Series:
    """Tokens per character of prompt text, per model, from single-shot.

    A single-shot run has one call, so its logged input tokens are the
    exact token count of its system plus user text.
    """
    ss = runs[(runs["condition"] == "single_shot") & (runs["n_calls"] == 1)]
    ss = ss[ss["total_input_tokens"].notna() & (ss["prompt_chars_single"] > 0)]
    grouped = ss.groupby("model_id")
    return grouped["total_input_tokens"].sum() / grouped["prompt_chars_single"].sum()


def calibrate_assistant_tokens_per_char(
    runs: pd.DataFrame, tokens_per_char: pd.Series,
    conditions: tuple[str, ...] | None = None,
) -> pd.Series:
    """Tokens per visible character of assistant text, per model.

    Fitted on multi-turn runs from the provider's own prompt counts:
    logged total input tokens minus the system/user share (characters
    times the single-shot ratio), divided by the assistant characters
    re-sent over all calls. Prompt counts never include thinking
    tokens, so this ratio measures re-sent text only. Assistant text
    (markdown, numbers, lists) tokenizes denser than prose, so it is
    usually above the single-shot ratio. Falls back to the single-shot
    ratio for a model without multi-turn runs.
    """
    mt = runs[(runs["condition"] != "single_shot") & runs["total_input_tokens"].notna()
              & (runs["calls_prompt_assistant_chars"] > 0)]
    if conditions is not None:
        mt = mt[mt["condition"].isin(conditions)]
    out = {}
    for model, g in mt.groupby("model_id"):
        r = tokens_per_char.get(model, np.nan)
        residual = g["total_input_tokens"].sum() - g["calls_prompt_chars"].sum() * r
        out[model] = residual / g["calls_prompt_assistant_chars"].sum()
    series = pd.Series(out, dtype=float)
    for model, r in tokens_per_char.items():
        if model not in series.index or not np.isfinite(series[model]) or series[model] <= 0:
            series[model] = r
    return series


def add_context_estimates(
    runs: pd.DataFrame,
    tokens_per_char: pd.Series,
    assistant_tokens_per_char: pd.Series | None = None,
    windows: dict[str, int] | None = None,
    windows_sensitivity: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add estimated final context, totals and shares of the window.

    Every message in a prompt is estimated from its visible characters:
    system/user text times the single-shot ratio, assistant text times
    the assistant ratio. Logged assistant token counts are not used
    here because they include thinking tokens for reasoning models,
    which are never re-sent.
    """
    windows = CONTEXT_WINDOWS if windows is None else windows
    windows_sensitivity = (CONTEXT_WINDOWS_SONNET_200K if windows_sensitivity is None
                           else windows_sensitivity)
    if assistant_tokens_per_char is None:
        assistant_tokens_per_char = tokens_per_char
    out = runs.copy()
    out["tokens_per_char"] = out["model_id"].map(tokens_per_char)
    out["assistant_tokens_per_char"] = out["model_id"].map(assistant_tokens_per_char)
    out["predicted_total_input_tokens"] = (
        out["calls_prompt_chars"] * out["tokens_per_char"]
        + out["calls_prompt_assistant_chars"] * out["assistant_tokens_per_char"]
    )
    # The same prediction with logged assistant token_count in place of
    # visible text. Overshoots for models whose log counts thinking.
    if "calls_prompt_assistant_tokens_logged" in out.columns:
        out["predicted_total_input_tokens_logged_assistant"] = (
            out["calls_prompt_chars"] * out["tokens_per_char"]
            + out["calls_prompt_assistant_tokens_logged"]
        )
    # Ratio of logged output tokens to the re-sent token value of the
    # visible output text: above 1 means the log counts hidden
    # (thinking) tokens.
    out["output_tokens_per_visible_token"] = out["main_output_tokens"] / (
        out["main_output_chars"] * out["assistant_tokens_per_char"]
    )
    out["final_context_tokens"] = (
        out["final_prompt_chars"] * out["tokens_per_char"]
        + (out["final_prompt_assistant_chars"] + out["last_output_chars"])
        * out["assistant_tokens_per_char"]
    ).round().astype("Int64")
    out["total_tokens"] = out["total_input_tokens"] + out["total_output_tokens"]
    out["context_window"] = out["model_id"].map(windows)
    out["share_of_window"] = out["final_context_tokens"] / out["context_window"]
    out["share_of_window_output_only"] = (
        out["total_output_tokens"] / out["context_window"]
    )
    # Billed tokens summed over all calls. Each call re-sends the
    # context, so this exceeds any single call's prompt. Not a context
    # measure; kept for the cost picture.
    out["cumulative_billed_tokens_over_window"] = out["total_tokens"] / out["context_window"]
    out["context_window_sensitivity"] = out["model_id"].map(windows_sensitivity)
    out["share_of_window_sensitivity"] = (
        out["final_context_tokens"] / out["context_window_sensitivity"]
    )
    return out


# ------------------------------------------------------------ models


def _row(section, name, group="", estimate=np.nan, ci_low=np.nan,
         ci_high=np.nan, p_value=np.nan, n=np.nan, note=""):
    return {
        "section": section,
        "name": name,
        "group": group,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": n,
        "note": note,
    }


def fit_logit_term(df: pd.DataFrame, formula: str, term: str,
                   cluster: str = "brief_id") -> dict:
    """Logistic regression; returns the OR for `term` with cluster SEs."""
    import statsmodels.formula.api as smf

    data = df.dropna(subset=_formula_columns(formula, df)).copy()
    try:
        model = smf.logit(formula, data=data)
        res = model.fit(
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": data[cluster].astype("category").cat.codes},
            maxiter=200,
        )
        beta = res.params[term]
        se = res.bse[term]
        return {
            "estimate": float(math.exp(beta)),
            "ci_low": float(math.exp(beta - 1.96 * se)),
            "ci_high": float(math.exp(beta + 1.96 * se)),
            "p_value": float(res.pvalues[term]),
            "n": int(res.nobs),
            "beta": float(beta),
            "note": "odds ratio, brief-clustered SE",
        }
    except Exception as exc:  # separation or singular design
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n": int(len(data)), "beta": np.nan,
                "note": f"fit failed: {type(exc).__name__}: {exc}"}


def fit_ols_term(df: pd.DataFrame, formula: str, term: str,
                 cluster: str = "brief_id") -> dict:
    """OLS; returns the coefficient for `term` with cluster SEs."""
    import statsmodels.formula.api as smf

    data = df.dropna(subset=_formula_columns(formula, df)).copy()
    model = smf.ols(formula, data=data)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster].astype("category").cat.codes},
    )
    beta = res.params[term]
    se = res.bse[term]
    return {
        "estimate": float(beta),
        "ci_low": float(beta - 1.96 * se),
        "ci_high": float(beta + 1.96 * se),
        "p_value": float(res.pvalues[term]),
        "n": int(res.nobs),
        "note": "OLS coefficient, brief-clustered SE",
    }


def _formula_columns(formula: str, df: pd.DataFrame) -> list[str]:
    """Column names that appear in a patsy formula."""
    import re

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    return [c for c in df.columns if c in tokens]


def _corr_rows(df: pd.DataFrame, x: str, y: str, section: str, group: str):
    sub = df[[x, y]].dropna()
    rows = []
    if len(sub) < 3:
        return rows
    r, p = stats.pearsonr(sub[x], sub[y])
    rows.append(_row(section, f"pearson {y} vs {x}", group, r, p_value=p,
                     n=len(sub)))
    rho, p2 = stats.spearmanr(sub[x], sub[y])
    rows.append(_row(section, f"spearman {y} vs {x}", group, rho, p_value=p2,
                     n=len(sub)))
    return rows


# ------------------------------------------------------------ run


def build_runs_table(data_root: Path, briefs_dir: Path) -> tuple[pd.DataFrame, dict]:
    files = transcript_files(data_root)
    raw = pd.DataFrame([summarise_transcript(load_transcript(f)) for f in files])
    n_transcripts = len(raw)

    raw, stray_filter = drop_stray(raw, data_root)
    scores, _ = load_scores_with_filter(data_root)
    scores = attach_kbv(scores, final_probe_table(data_root))
    keep = ["run_id", "j_objective_fidelity", "j_constraint_adherence",
            "j_alternative_coverage", "j_complexity_inflation",
            "final_probe_accuracy", "knows", "noncompliant", "kbv"]
    runs = raw.merge(scores[keep], on="run_id", how="inner")

    ratio = calibrate_tokens_per_char(runs)
    ratio_assistant = calibrate_assistant_tokens_per_char(runs, ratio)
    runs = add_context_estimates(runs, ratio, ratio_assistant)
    # Held-out check: fit the assistant ratio on neutral runs only and
    # predict the pressure and checkpointed runs' logged input.
    ratio_assistant_neutral = calibrate_assistant_tokens_per_char(
        runs, ratio, conditions=("multi_turn_neutral",))
    runs["assistant_tokens_per_char_neutral_fit"] = runs["model_id"].map(ratio_assistant_neutral)
    runs["predicted_total_input_tokens_neutral_fit"] = (
        runs["calls_prompt_chars"] * runs["tokens_per_char"]
        + runs["calls_prompt_assistant_chars"] * runs["assistant_tokens_per_char_neutral_fit"]
    )

    briefs = load_briefs(briefs_dir)
    bdf = pd.DataFrame([
        {
            "brief_id": b["id"],
            "n_hard_constraints": len(b.get("hard_constraints") or []),
            "n_banned_moves": len(b.get("banned_moves") or []),
            "objective_words": word_count(b.get("objective")),
            "brief_words": sum(word_count(x) for x in _brief_texts(b)),
        }
        for b in briefs.values()
    ])
    runs = runs.merge(bdf, on="brief_id", how="left")
    runs["log_total_tokens"] = np.log(runs["total_tokens"].astype(float))
    runs["log_output_tokens"] = np.log(runs["total_output_tokens"].astype(float))
    runs["log_final_words"] = np.log1p(runs["final_response_words"].astype(float))
    runs["is_pressure"] = (runs["condition"] == PRESSURE).astype(int)

    mt = runs[runs["condition"] != "single_shot"]
    calib = (mt["predicted_total_input_tokens"] / mt["total_input_tokens"]).groupby(
        mt["model_id"]).mean().round(4).to_dict()
    held = runs[runs["condition"].isin(["multi_turn_pressure", "checkpointed_pressure"])]
    calib_held_out = (held["predicted_total_input_tokens_neutral_fit"]
                      / held["total_input_tokens"]).groupby(held["model_id"]).mean().round(4).to_dict()
    calib_logged = (mt["predicted_total_input_tokens_logged_assistant"]
                    / mt["total_input_tokens"]).groupby(mt["model_id"]).mean().round(4).to_dict()
    hidden = runs.groupby("model_id").apply(
        lambda g: g["main_output_tokens"].sum()
        / (g["main_output_chars"] * g["assistant_tokens_per_char"]).sum()
    ).round(4).to_dict()

    prov = {
        "inputs": [
            str(data_root / "transcripts" / "*.jsonl"),
            str(data_root / "openweight_subjects" / "transcripts" / "*.jsonl"),
            str(data_root / "aggregated" / "all_scores.parquet"),
            str(data_root / "analysis" / "final_probe_accuracy.csv"),
            str(briefs_dir / "*.yaml"),
        ],
        "filters": [
            stray_filter,
            {"rule": "transcript metadata.run_id is in the clean parquet (inner join)",
             "rows_in": int(stray_filter["rows_out"]), "rows_out": int(len(runs))},
        ],
        "definitions": {
            "total_tokens": "metadata.total_input_tokens + metadata.total_output_tokens, summed over every API call of the run (main and probe calls)",
            "final_context_tokens": "prompt of the last main call plus its own response, every message estimated from visible characters: system/user text times tokens_per_char (single-shot calibration), assistant text times assistant_tokens_per_char (multi-turn calibration against the provider's prompt counts); probes excluded; logged assistant token_count not used because it includes thinking tokens for reasoning models",
            "share_of_window": "final_context_tokens / context_window",
            "share_of_window_output_only": "metadata.total_output_tokens (logged, thinking included) / context_window",
            "cumulative_billed_tokens_over_window": "total_tokens / context_window; each call re-sends the context, so this is a billing quantity, not a context size, and exceeds 1 for some Sonnet runs",
            "share_of_window_sensitivity": "final_context_tokens / context_window with Sonnet 4.6 at its 200k standard window; the April runs sent no context-1m beta header and the applied default could not be confirmed from the logs",
            "kbv": "final_probe_accuracy >= 0.8 and j_constraint_adherence < 4",
            "noncompliant": "j_constraint_adherence < 4",
            "tokens_per_char": ratio.round(6).to_dict(),
            "assistant_tokens_per_char": ratio_assistant.round(6).to_dict(),
            "context_windows": CONTEXT_WINDOWS,
            "context_window_sources": CONTEXT_WINDOW_SOURCES,
            "context_windows_sensitivity": CONTEXT_WINDOWS_SONNET_200K,
        },
        "calibration": {
            "predicted_over_logged_total_input_multi_turn_mean_by_construction": calib,
            "predicted_over_logged_total_input_pressure_and_checkpointed_fit_on_neutral": calib_held_out,
            "predicted_over_logged_total_input_using_logged_assistant_token_count": calib_logged,
            "assistant_tokens_per_char_fit_on_neutral": ratio_assistant_neutral.round(6).to_dict(),
            "logged_output_tokens_over_resent_output_tokens": hidden,
            "note": ("Predicted total input uses visible characters of every message. "
                     "The primary assistant ratio is fitted on all multi-turn runs, so "
                     "its per-model mean is 1 by construction; the held-out entry fits "
                     "on neutral runs only and predicts pressure and checkpointed runs. "
                     "The logged-token entry replaces visible assistant text by the "
                     "logged token_count: it overshoots for Gemini 3.1 Pro because "
                     "its token_count includes thinking tokens, which are never "
                     "re-sent (see logged_output_tokens_over_resent_output_tokens)."),
        },
    }
    return runs, prov


def _brief_texts(brief: dict) -> list[str]:
    texts = [brief.get("objective", ""), brief.get("domain", "")]
    for key in ("hard_constraints", "success_criteria",
                "plausible_directions", "banned_moves"):
        texts.extend(brief.get(key) or [])
    return texts


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    n_all = len(runs)

    # 1. Coverage and the calibration check.
    rows.append(_row("coverage", "runs with usage records", "all", n_all, n=n_all))
    rows.append(_row("coverage", "pressure runs with usage records", PRESSURE,
                     int((runs["condition"] == PRESSURE).sum()),
                     n=int((runs["condition"] == PRESSURE).sum())))
    mt = runs[runs["condition"] != "single_shot"]
    held = runs[runs["condition"].isin(["multi_turn_pressure", "checkpointed_pressure"])]
    for model, g in mt.groupby("model_id"):
        ratio = (g["predicted_total_input_tokens"] / g["total_input_tokens"])
        rows.append(_row("calibration",
                         "predicted / logged total input tokens, multi-turn runs (by construction)",
                         model, float(ratio.mean()), float(ratio.min()),
                         float(ratio.max()), n=len(g),
                         note="by construction: the assistant ratio is fitted on these "
                              "runs, so the mean is 1; min/max show the per-run spread"))
        h = held[held["model_id"] == model]
        ratio_h = h["predicted_total_input_tokens_neutral_fit"] / h["total_input_tokens"]
        rows.append(_row("calibration",
                         "predicted / logged total input tokens, pressure + checkpointed, ratio fit on neutral only (held out)",
                         model, float(ratio_h.mean()), float(ratio_h.min()),
                         float(ratio_h.max()), n=len(h),
                         note="held-out check: assistant ratio fitted on multi_turn_neutral runs"))
        ratio_l = g["predicted_total_input_tokens_logged_assistant"] / g["total_input_tokens"]
        rows.append(_row("calibration",
                         "predicted / logged total input tokens, multi-turn runs, logged assistant token_count instead of text",
                         model, float(ratio_l.mean()), float(ratio_l.min()),
                         float(ratio_l.max()), n=len(g),
                         note="above 1 means token_count counts thinking tokens that are never re-sent"))
    for model, g in runs.groupby("model_id"):
        rows.append(_row("calibration", "tokens per char, system/user text (single-shot)",
                         model, float(g["tokens_per_char"].iloc[0]), n=len(g)))
        rows.append(_row("calibration", "tokens per char, re-sent assistant text (multi-turn)",
                         model, float(g["assistant_tokens_per_char"].iloc[0]), n=len(g)))
        hidden = g["main_output_tokens"].sum() / (
            g["main_output_chars"] * g["assistant_tokens_per_char"]).sum()
        rows.append(_row("calibration", "logged output tokens / re-sent output tokens",
                         model, float(hidden), n=len(g),
                         note="above 1 means token_count includes thinking tokens, "
                              "which are never re-sent"))
    # 2. Max share of window.
    for col, label in [("share_of_window", "final context / window"),
                       ("share_of_window_output_only", "logged output tokens / window"),
                       ("cumulative_billed_tokens_over_window",
                        "cumulative billed tokens over all calls / window (not a context size)"),
                       ("share_of_window_sensitivity",
                        "final context / window, Sonnet at 200k (sensitivity)")]:
        rows.append(_row("context_share", f"max {label}", "all",
                         float(runs[col].max()), n=n_all,
                         note=f"model at max: {runs.loc[runs[col].idxmax(), 'model_id']}"))
        for model, g in runs.groupby("model_id"):
            rows.append(_row("context_share", f"max {label}", model,
                             float(g[col].max()), n=len(g),
                             note=f"window {int(g['context_window'].iloc[0])}"))
    rows.append(_row("context_share", "min headroom (window / final context)", "all",
                     float(1.0 / runs["share_of_window"].max()), n=n_all))
    for model, g in runs.groupby("model_id"):
        rows.append(_row("context_share", "mean final context tokens", model,
                         float(g["final_context_tokens"].mean()), n=len(g)))
        rows.append(_row("context_share", "mean total tokens", model,
                         float(g["total_tokens"].mean()), n=len(g)))

    # 3. Within pressure: do tokens predict drift, model fixed?
    pr = runs[runs["condition"] == PRESSURE].copy()
    pr_probe = pr.dropna(subset=["kbv"]).copy()
    pr_probe["kbv"] = pr_probe["kbv"].astype(int)
    specs = [
        ("logit kbv ~ log_total_tokens + model FE", pr_probe,
         "kbv ~ log_total_tokens + C(model_id)", "log_total_tokens", "logit"),
        ("logit kbv ~ log_output_tokens + model FE", pr_probe,
         "kbv ~ log_output_tokens + C(model_id)", "log_output_tokens", "logit"),
        ("logit noncompliant ~ log_total_tokens + model FE", pr,
         "noncompliant ~ log_total_tokens + C(model_id)", "log_total_tokens", "logit"),
        ("logit noncompliant ~ log_output_tokens + model FE", pr,
         "noncompliant ~ log_output_tokens + C(model_id)", "log_output_tokens", "logit"),
        ("ols adherence ~ log_total_tokens + model FE", pr,
         "j_constraint_adherence ~ log_total_tokens + C(model_id)", "log_total_tokens", "ols"),
        ("ols adherence ~ log_output_tokens + model FE", pr,
         "j_constraint_adherence ~ log_output_tokens + C(model_id)", "log_output_tokens", "ols"),
        ("ols adherence ~ log_final_words + model FE", pr,
         "j_constraint_adherence ~ log_final_words + C(model_id)", "log_final_words", "ols"),
        ("ols fidelity ~ log_total_tokens + model FE", pr,
         "j_objective_fidelity ~ log_total_tokens + C(model_id)", "log_total_tokens", "ols"),
    ]
    for name, data, formula, term, kind in specs:
        fit = fit_logit_term(data, formula, term) if kind == "logit" \
            else fit_ols_term(data, formula, term)
        note = fit["note"]
        if name == "ols adherence ~ log_output_tokens + model FE":
            note += "; July comparison row (beta -0.19, p = 0.35)"
        rows.append(_row("pressure_tokens", name, PRESSURE, fit["estimate"],
                         fit["ci_low"], fit["ci_high"], fit["p_value"],
                         fit["n"], note))

    # 3b. B10: fidelity on length, per provider group.
    rows.extend(fidelity_length_rows(runs))

    # 4. Verbosity-controlled pressure effect.
    # The published control (verbosity.py) is the word count of the
    # judged final response. Total tokens over the run are collinear
    # with the condition over all four conditions, since single-shot
    # runs make one call and multi-turn runs make eleven; those rows
    # are secondary. The multi-turn-only contrast (pressure vs neutral,
    # same call count) is the robustness check for that collinearity.
    controls = (("none", "no verbosity control"),
                ("log_final_words", "published method (verbosity.py): judged final response word count"),
                ("log_output_tokens", "secondary: collinear with condition over all conditions"),
                ("log_total_tokens", "secondary: collinear with condition over all conditions"))
    cond_term = "C(condition, Treatment('single_shot'))[T.multi_turn_pressure]"
    for tok, note in controls:
        extra = "" if tok == "none" else f" + {tok}"
        formula = (f"j_constraint_adherence ~ C(condition, Treatment('single_shot'))"
                   f"{extra} + C(model_id)")
        fit = fit_ols_term(runs, formula, cond_term)
        rows.append(_row("verbosity_control",
                         f"ols adherence ~ condition{extra} + model FE: pressure vs single-shot",
                         "all conditions", fit["estimate"], fit["ci_low"], fit["ci_high"],
                         fit["p_value"], fit["n"], f"{fit['note']}; {note}"))
    # Robustness: pressure vs neutral among multi-turn runs, every control.
    mt_pn = runs[runs["condition"].isin(["multi_turn_neutral", PRESSURE])].copy()
    for tok, note in controls:
        extra = "" if tok == "none" else f" + {tok}"
        fit = fit_ols_term(mt_pn, f"j_constraint_adherence ~ is_pressure{extra} + C(model_id)",
                           "is_pressure")
        rows.append(_row("verbosity_control",
                         f"ols adherence ~ is_pressure{extra} + model FE: pressure vs neutral",
                         "multi-turn only", fit["estimate"], fit["ci_low"], fit["ci_high"],
                         fit["p_value"], fit["n"], f"{fit['note']}; {note}; same call count in both arms"))
    mt_all = runs[runs["condition"] != "single_shot"].copy()
    for tok, note in controls:
        extra = "" if tok == "none" else f" + {tok}"
        fit = fit_ols_term(mt_all, f"j_constraint_adherence ~ is_pressure{extra} + C(model_id)",
                           "is_pressure")
        rows.append(_row("verbosity_control",
                         f"ols adherence ~ is_pressure{extra} + model FE: pressure vs neutral and checkpointed",
                         "multi-turn only", fit["estimate"], fit["ci_low"], fit["ci_high"],
                         fit["p_value"], fit["n"], f"{fit['note']}; {note}; same call count in both arms"))
    # Robustness: pressure vs all other runs with log total tokens.
    for tok, note in controls:
        extra = "" if tok == "none" else f" + {tok}"
        fit = fit_ols_term(runs, f"j_constraint_adherence ~ is_pressure{extra} + C(model_id)",
                           "is_pressure")
        rows.append(_row("verbosity_control",
                         f"ols adherence ~ is_pressure{extra} + model FE: pressure vs all other",
                         "all conditions", fit["estimate"], fit["ci_low"], fit["ci_high"],
                         fit["p_value"], fit["n"], f"{fit['note']}; {note}"))

    # 5. Tokens vs brief length.
    for x in ("n_hard_constraints", "objective_words", "brief_words"):
        rows.extend(_corr_rows(runs, x, "total_tokens", "tokens_vs_brief", "all runs"))
        rows.extend(_corr_rows(pr, x, "total_tokens", "tokens_vs_brief", PRESSURE))
    brief_level = pr.groupby("brief_id").agg(
        total_tokens=("total_tokens", "mean"),
        n_hard_constraints=("n_hard_constraints", "first"),
        objective_words=("objective_words", "first"),
        brief_words=("brief_words", "first"),
        kbv_rate=("kbv", "mean"),
        adherence=("j_constraint_adherence", "mean"),
    ).reset_index()
    for x in ("n_hard_constraints", "objective_words", "brief_words"):
        rows.extend(_corr_rows(brief_level, x, "total_tokens", "tokens_vs_brief",
                               f"{PRESSURE}, brief means"))
    rows.extend(_corr_rows(brief_level, "total_tokens", "kbv_rate", "tokens_vs_brief",
                           f"{PRESSURE}, brief means"))
    rows.extend(_corr_rows(brief_level, "total_tokens", "adherence", "tokens_vs_brief",
                           f"{PRESSURE}, brief means"))

    return pd.DataFrame(rows)


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path | None = None,
        briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or default_out_dir(data_root)
    runs, prov = build_runs_table(data_root, briefs_dir)
    cols = [
        "run_id", "brief_id", "model_id", "condition",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "n_calls", "n_main_turns", "n_probe_calls", "main_output_tokens",
        "main_output_chars", "probe_output_tokens", "last_output_tokens",
        "last_output_chars", "final_response_words", "tokens_per_char",
        "assistant_tokens_per_char", "output_tokens_per_visible_token",
        "predicted_total_input_tokens", "predicted_total_input_tokens_logged_assistant",
        "assistant_tokens_per_char_neutral_fit", "predicted_total_input_tokens_neutral_fit",
        "final_context_tokens",
        "context_window", "share_of_window", "share_of_window_output_only",
        "cumulative_billed_tokens_over_window",
        "context_window_sensitivity", "share_of_window_sensitivity",
        "j_objective_fidelity", "j_constraint_adherence",
        "j_alternative_coverage", "j_complexity_inflation",
        "final_probe_accuracy", "knows", "noncompliant", "kbv",
        "n_hard_constraints", "n_banned_moves", "objective_words", "brief_words",
    ]
    runs_out = runs[cols].sort_values("run_id").reset_index(drop=True)
    write_csv(runs_out, out_dir, "tokens_context_runs", prov)

    summary = build_summary(runs)
    prov_s = dict(prov)
    prov_s["derived_from"] = "tokens_context_runs.csv (same filters)"
    prov_s["filters"] = prov["filters"] + [
        {"rule": "pressure_tokens section: condition == multi_turn_pressure",
         "rows_in": int(len(runs)),
         "rows_out": int((runs["condition"] == PRESSURE).sum())},
        {"rule": "kbv models: pressure runs with a final probe record",
         "rows_in": int((runs["condition"] == PRESSURE).sum()),
         "rows_out": int(runs[(runs["condition"] == PRESSURE)]["kbv"].notna().sum())},
        {"rule": "verbosity_control 'pressure vs neutral': condition in (multi_turn_neutral, multi_turn_pressure)",
         "rows_in": int(len(runs)),
         "rows_out": int(runs["condition"].isin(["multi_turn_neutral", PRESSURE]).sum())},
        {"rule": "verbosity_control 'pressure vs neutral and checkpointed': condition != single_shot",
         "rows_in": int(len(runs)),
         "rows_out": int((runs["condition"] != "single_shot").sum())},
        {"rule": f"fidelity_length section (B10): group commercial = model_id not starting with {OPEN_WEIGHT_PREFIX}, "
                 f"open_weight = model_id starting with {OPEN_WEIGHT_PREFIX}; spec 'published' uses all conditions, "
                 f"'pressure' and 'pressure_raw' use condition == {PRESSURE}; the fit drops rows with a missing term",
         "rows_in": int(len(runs)),
         "rows_out": {g: int(length_group_mask(runs, g).sum()) for g in LENGTH_GROUPS},
         "rows_out_pressure": {g: int((length_group_mask(runs, g) & (runs["condition"] == PRESSURE)).sum()) for g in LENGTH_GROUPS}},
    ]
    write_csv(summary, out_dir, "tokens_context_summary", prov_s)
    return {"tokens_context_runs": runs_out, "tokens_context_summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--briefs-dir", type=Path, default=DEFAULT_BRIEFS_DIR)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir, args.briefs_dir)
    s = tables["tokens_context_summary"]
    pd.set_option("display.width", 220, "display.max_rows", 500,
                  "display.max_colwidth", 100)
    print(s[s["section"].isin(["coverage", "calibration", "context_share", "pressure_tokens",
                                "verbosity_control"])].to_string(index=False))


if __name__ == "__main__":
    main()
