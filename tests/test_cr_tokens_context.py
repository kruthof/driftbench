"""Tests for drift_bench.analysis.camera_ready.tokens_context.

Pure functions on synthetic transcripts. No network, no API keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_bench.analysis.camera_ready import tokens_context as tc


def _msg(role, content, token_count=None, is_probe=False):
    return {"role": role, "content": content, "timestamp": 0.0,
            "token_count": token_count, "latency_ms": None, "is_probe": is_probe}


def _transcript(run_id="r1", model="openai/gpt-5.4", condition="multi_turn_pressure",
                total_input=None, total_output=None, messages=None):
    return {
        "metadata": {
            "run_id": run_id, "brief_id": "b1", "model_id": model,
            "condition": condition, "repetition": 0, "start_time": 0.0,
            "end_time": 1.0, "total_input_tokens": total_input,
            "total_output_tokens": total_output, "total_cost_usd": 0.0,
            "error": None,
        },
        "messages": messages or [],
    }


def test_summarise_single_shot_counts_prompt_chars_and_one_call():
    t = _transcript(condition="single_shot", total_input=50, total_output=20,
                    messages=[_msg("system", "a" * 100), _msg("user", "b" * 100),
                              _msg("assistant", "x y z", token_count=20)])
    s = tc.summarise_transcript(t)
    assert s["n_calls"] == 1
    assert s["n_main_turns"] == 1
    assert s["n_probe_calls"] == 0
    assert s["prompt_chars_single"] == 200
    assert s["final_prompt_chars"] == 200
    assert s["final_prompt_assistant_chars"] == 0
    assert s["last_output_tokens"] == 20
    assert s["last_output_chars"] == 5
    assert s["final_response_words"] == 3


def test_summarise_multi_turn_excludes_probes_from_main_context():
    msgs = [
        _msg("system", "s" * 40),
        _msg("user", "u" * 60),
        _msg("assistant", "turn one", token_count=100),  # 8 visible chars
        _msg("user", "p" * 30, is_probe=True),
        _msg("assistant", "probe answer", token_count=7, is_probe=True),
        _msg("user", "n" * 20),
        _msg("assistant", "turn two words here", token_count=200),  # 19 chars
    ]
    s = tc.summarise_transcript(_transcript(total_input=500, total_output=307,
                                            messages=msgs))
    # Three calls: main 1, probe 1, main 2.
    assert s["n_calls"] == 3
    assert s["n_main_turns"] == 2
    assert s["n_probe_calls"] == 1
    assert s["main_output_tokens"] == 300
    assert s["main_output_chars"] == 8 + 19
    assert s["probe_output_tokens"] == 7
    # Final main call sees system + user + the next user turn, and the
    # visible text of turn one. The probe exchange is not in the context.
    assert s["final_prompt_chars"] == 40 + 60 + 20
    assert s["final_prompt_assistant_chars"] == 8
    assert s["last_output_tokens"] == 200
    assert s["last_output_chars"] == 19
    assert s["final_response_words"] == 4
    # Sum over calls: main1 (100 chars, 0 assistant), probe (100 + 30
    # chars, 8 assistant), main2 (120 chars, 8 assistant).
    assert s["calls_prompt_chars"] == 100 + 130 + 120
    assert s["calls_prompt_assistant_chars"] == 0 + 8 + 8


def test_calibration_uses_single_shot_runs_only():
    runs = pd.DataFrame([
        {"model_id": "m", "condition": "single_shot", "n_calls": 1,
         "total_input_tokens": 50, "prompt_chars_single": 200},
        {"model_id": "m", "condition": "single_shot", "n_calls": 1,
         "total_input_tokens": 150, "prompt_chars_single": 600},
        # Multi-turn rows must not enter the ratio.
        {"model_id": "m", "condition": "multi_turn_pressure", "n_calls": 11,
         "total_input_tokens": 9999, "prompt_chars_single": 200},
    ])
    ratio = tc.calibrate_tokens_per_char(runs)
    assert ratio["m"] == pytest.approx(200 / 800)


def test_context_estimates_and_shares():
    runs = pd.DataFrame([{
        "model_id": "openai/gpt-5.4", "calls_prompt_chars": 1000,
        "calls_prompt_assistant_chars": 200, "final_prompt_chars": 400,
        "final_prompt_assistant_chars": 400, "last_output_chars": 200,
        "main_output_tokens": 300, "main_output_chars": 600,
        "total_input_tokens": 600, "total_output_tokens": 400,
    }])
    ratio = pd.Series({"openai/gpt-5.4": 0.25})
    out = tc.add_context_estimates(runs, ratio, pd.Series({"openai/gpt-5.4": 0.5}),
                                   windows={"openai/gpt-5.4": 1000},
                                   windows_sensitivity={"openai/gpt-5.4": 500})
    r = out.iloc[0]
    assert r["predicted_total_input_tokens"] == pytest.approx(1000 * 0.25 + 200 * 0.5)
    # Assistant text enters by visible characters times the assistant
    # ratio, not by logged tokens.
    assert r["final_context_tokens"] == 400 * 0.25 + (400 + 200) * 0.5  # 400
    assert r["output_tokens_per_visible_token"] == pytest.approx(300 / (600 * 0.5))
    assert r["total_tokens"] == 1000
    assert r["share_of_window"] == pytest.approx(0.4)
    assert r["share_of_window_output_only"] == pytest.approx(0.4)
    assert r["cumulative_billed_tokens_over_window"] == pytest.approx(1.0)
    assert r["share_of_window_sensitivity"] == pytest.approx(0.8)


def test_assistant_ratio_fitted_from_logged_prompt_counts():
    # Two multi-turn runs: logged input = prompt chars * 0.25 + assistant chars * r_a.
    runs = pd.DataFrame([
        {"model_id": "m", "condition": "multi_turn_pressure", "total_input_tokens": 100 + 60,
         "calls_prompt_chars": 400, "calls_prompt_assistant_chars": 200},
        {"model_id": "m", "condition": "multi_turn_pressure", "total_input_tokens": 50 + 30,
         "calls_prompt_chars": 200, "calls_prompt_assistant_chars": 100},
        # A model without multi-turn runs keeps the single-shot ratio.
        {"model_id": "z", "condition": "single_shot", "total_input_tokens": 10,
         "calls_prompt_chars": 40, "calls_prompt_assistant_chars": 0},
    ])
    r = pd.Series({"m": 0.25, "z": 0.25})
    ra = tc.calibrate_assistant_tokens_per_char(runs, r)
    assert ra["m"] == pytest.approx(0.3)
    assert ra["z"] == 0.25


def test_context_windows_cover_every_subject_model():
    from drift_bench.analysis.camera_ready.b4_common import MODEL_ORDER
    for m in MODEL_ORDER:
        assert m in tc.CONTEXT_WINDOWS
        assert m in tc.CONTEXT_WINDOW_SOURCES
        assert tc.CONTEXT_WINDOWS[m] > 0
    assert tc.CONTEXT_WINDOWS_SONNET_200K[tc.SONNET_ID] == 200_000
    assert tc.CONTEXT_WINDOWS["together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"] == 262_144
    assert tc.QWEN_SOURCE_URL in tc.CONTEXT_WINDOW_SOURCES[
        "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"]


def test_resolve_context_windows_falls_back_when_litellm_unavailable(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "litellm", None)  # import raises
    windows, sources = tc.resolve_context_windows(["openai/gpt-5.4"])
    assert windows["openai/gpt-5.4"] == tc.CONTEXT_WINDOWS_FALLBACK["openai/gpt-5.4"]
    assert "fallback" in sources["openai/gpt-5.4"]


def test_fit_ols_term_recovers_slope_with_cluster_se():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    model = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = 2.0 * x + np.where(model == "a", 0.0, 1.0) + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y": y, "x": x, "model_id": model,
                       "brief_id": np.arange(n) % 20})
    fit = tc.fit_ols_term(df, "y ~ x + C(model_id)", "x")
    assert fit["estimate"] == pytest.approx(2.0, abs=0.05)
    assert fit["ci_low"] < 2.0 < fit["ci_high"]
    assert fit["n"] == n


def test_fit_logit_term_returns_odds_ratio_above_one_for_positive_effect():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.0 * x)))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": y, "x": x, "model_id": np.where(np.arange(n) % 2 == 0, "a", "b"),
                       "brief_id": np.arange(n) % 30})
    fit = tc.fit_logit_term(df, "y ~ x + C(model_id)", "x")
    assert fit["estimate"] > 1.5
    assert fit["ci_low"] > 1.0
    assert fit["p_value"] < 0.01


def test_fit_logit_term_reports_failure_instead_of_raising():
    # Perfect separation: every x > 0 is a 1.
    df = pd.DataFrame({"y": [0, 0, 0, 1, 1, 1], "x": [-3, -2, -1, 1, 2, 3],
                       "model_id": ["a"] * 6, "brief_id": [0, 1, 2, 0, 1, 2]})
    fit = tc.fit_logit_term(df, "y ~ x", "x")
    assert "n" in fit
    assert fit["n"] == 6


# ---------------------------------------------------------------- B10: fidelity on length per provider group

LLAMA_ID = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
QWEN_ID = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
CONDITIONS4 = ("single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure")


def test_length_group_mask_splits_on_the_together_prefix():
    runs = pd.DataFrame({"model_id": ["openai/gpt-5.4", LLAMA_ID, "anthropic/claude-sonnet-4-6", QWEN_ID]})
    assert tc.length_group_mask(runs, "all").tolist() == [True] * 4
    assert tc.length_group_mask(runs, "open_weight").tolist() == [False, True, False, True]
    assert tc.length_group_mask(runs, "commercial").tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        tc.length_group_mask(runs, "other")


def _length_runs(seed: int = 7) -> pd.DataFrame:
    """Fidelity rises with log1p(words) for the open-weight models only; complexity lowers it for all."""
    rng = np.random.default_rng(seed)
    slopes = {"openai/gpt-5.4": 0.0, "anthropic/claude-sonnet-4-6": 0.0, LLAMA_ID: 0.6, QWEN_ID: 0.6}
    rows = []
    k = 0
    for model, slope in slopes.items():
        for cond in CONDITIONS4:
            for b in range(30):
                k += 1
                words = int(rng.integers(100, 3000))
                cmp_ = int(rng.integers(0, 4))
                fid = 1.0 + slope * np.log1p(words) - 0.3 * cmp_ + rng.normal(0, 0.2)
                rows.append({"run_id": f"r{k}", "brief_id": f"b{b % 15}", "model_id": model, "condition": cond,
                             "final_response_words": words, "total_output_tokens": int(rng.integers(500, 20000)),
                             "j_objective_fidelity": fid, "j_complexity_inflation": cmp_})
    return pd.DataFrame(rows)


def test_fidelity_length_rows_cover_every_group_measure_spec_and_term():
    rows = pd.DataFrame(tc.fidelity_length_rows(_length_runs()))
    assert (rows["section"] == "fidelity_length").all()
    # 3 groups x 2 measures x (2 + 2 + 1) terms.
    assert len(rows) == 3 * 2 * 5
    names = set(rows["name"])
    assert "ols fidelity published log_final_words: log_final_words" in names
    assert "ols fidelity pressure log_output_tokens: complexity" in names
    assert "ols fidelity pressure_raw log_output_tokens: log_output_tokens" in names
    assert not any(n.startswith("ols fidelity pressure_raw") and n.endswith(": complexity") for n in names)
    n = rows.set_index(["name", "group"])["n"]
    # The submitted spec uses all four conditions; the pressure specs one.
    assert n[("ols fidelity published log_final_words: log_final_words", "open_weight")] == 2 * 4 * 30
    assert n[("ols fidelity pressure log_final_words: log_final_words", "open_weight")] == 2 * 30
    assert n[("ols fidelity pressure log_final_words: log_final_words", "all")] == 4 * 30


def test_fidelity_length_rows_recover_the_planted_slopes():
    rows = pd.DataFrame(tc.fidelity_length_rows(_length_runs())).set_index(["name", "group"])
    pub = "ols fidelity published log_final_words: log_final_words"
    ow, com = rows.loc[(pub, "open_weight")], rows.loc[(pub, "commercial")]
    assert ow["estimate"] == pytest.approx(0.6, abs=0.1) and ow["p_value"] < 0.001
    assert com["ci_low"] < 0 < com["ci_high"]
    cmp_row = rows.loc[("ols fidelity published log_final_words: complexity", "all")]
    assert cmp_row["estimate"] == pytest.approx(-0.3, abs=0.05)
    # Total output tokens carry no planted signal, so the slope stays small.
    tok = rows.loc[("ols fidelity pressure log_output_tokens: log_output_tokens", "open_weight")]
    assert abs(tok["estimate"]) < 0.15
