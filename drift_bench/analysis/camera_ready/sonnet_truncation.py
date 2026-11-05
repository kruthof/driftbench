"""Sonnet 4.6 output-cap prevalence per sample, condition and rule (plan item B16; writing gap WB-5).

Sonnet ran with a 4,096-token output cap. Two rules mark a run as capped:

* ``any_turn``: any non-probe assistant message has ``token_count == 4096``
  (the paper's prevalence rule; July: 226 of 315 as published).
* ``final_turn``: the judged final assistant message has ``token_count == 4096``
  (the rule of the paper's adherence test; 127 of 315 as published).

``corrections_ledger.csv`` holds the as-published counts only (its B16
rows put the any-turn count in ``recomputed_value`` and the final-turn
count in ``numerator``). This module writes both rules on the as-published
and the clean sample (``kbv_table.build_samples``), per condition and
pooled over the four main conditions, with Wilson 95% CIs.

Gemini 3.1 Pro is the only other subject that reaches the same 4,096-token
cap. Gemini's capped outputs log a few tokens fewer than the cap (4,092 here,
as 1,020 at the 1,024-token probe cap, ``kbv_table.GEMINI_PRO_PROBE_CAP``; no
main-run Pro message lies between 4,085 and 4,092 or above it), so its rows use
``token_count >= 4092`` in both rules. They follow the Sonnet rows.

Disk only, no API calls.
Run: ``python -m drift_bench.analysis.camera_ready.sonnet_truncation``
"""

from __future__ import annotations

import argparse

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready.kbv_table import (
    GEMINI_PRO,
    MAIN_CONDITIONS,
    SONNET,
    SONNET_OUTPUT_CAP,
    build_samples,
    filter_rules,
    load_judge_timestamps,
    load_scores,
    load_transcript_index,
    wilson_ci,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"

RULES = {"any_turn": "any_assistant_at_cap", "final_turn": "final_assistant_at_cap"}
GEMINI_PRO_LOGGED_OUTPUT_CAP = 4092
SAMPLES = ["as_published", "clean"]
ALL_MAIN = "all_main"


def truncation_table(sample_df: pd.DataFrame, sample: str, model_id: str = SONNET,
                     conditions: list[str] = MAIN_CONDITIONS) -> pd.DataFrame:
    """Capped-run counts per condition and pooled, both rules, one sample.

    ``sample_df`` must carry ``model_id, condition, any_assistant_at_cap,
    final_assistant_at_cap`` (the ``build_samples`` frame). Rows without a
    transcript (NA flags) are dropped from both numerator and denominator.
    """
    sub = sample_df[(sample_df["model_id"] == model_id) & sample_df["condition"].isin(conditions)]
    rows = []
    for rule, column in RULES.items():
        flags = sub[sub[column].notna()]
        groups = [(c, flags[flags["condition"] == c]) for c in conditions] + [(ALL_MAIN, flags)]
        for cond, g in groups:
            n = int(len(g))
            k = int(g[column].astype(bool).sum())
            lo, hi = wilson_ci(k, n)
            rows.append({
                "sample": sample,
                "model_id": model_id,
                "rule": rule,
                "condition": cond,
                "n_runs": n,
                "n_truncated": k,
                "n_untruncated": n - k,
                "share_truncated": (k / n) if n else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def with_gemini_pro_cap_flags(sample_df: pd.DataFrame, transcripts: pd.DataFrame,
                              logged_cap: int = GEMINI_PRO_LOGGED_OUTPUT_CAP) -> pd.DataFrame:
    """Gemini Pro rows of ``sample_df`` with both cap flags at Gemini's logged cap.

    ``transcripts`` is the ``load_transcript_index`` frame. Rows without a
    transcript keep NA flags, as for Sonnet.
    """
    sub = sample_df[sample_df["model_id"] == GEMINI_PRO].copy()
    idx = transcripts.drop_duplicates("run_id").set_index("run_id")
    max_tok = sub["run_id"].map(idx["max_assistant_token_count"])
    final_tok = sub["run_id"].map(idx["final_assistant_token_count"])
    has = sub["run_id"].isin(idx.index)
    sub["any_assistant_at_cap"] = (max_tok >= logged_cap).astype(object).where(has, pd.NA)
    sub["final_assistant_at_cap"] = (final_tok >= logged_cap).astype(object).where(has, pd.NA)
    return sub


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    scores = load_scores(data_root)
    judge_ts = load_judge_timestamps(data_root)
    transcripts = load_transcript_index(data_root)
    samples, log = build_samples(scores, judge_ts, transcripts)
    table = pd.concat(
        [truncation_table(samples[s], s) for s in SAMPLES]
        + [truncation_table(with_gemini_pro_cap_flags(samples[s], transcripts), s, model_id=GEMINI_PRO)
           for s in SAMPLES],
        ignore_index=True,
    )
    sonnet_rows = {}
    for s in SAMPLES:
        sub = samples[s][(samples[s]["model_id"] == SONNET) & samples[s]["condition"].isin(MAIN_CONDITIONS)]
        sonnet_rows[s] = {
            "sonnet_rows_main_conditions": int(len(sub)),
            "rows_without_transcript_dropped": int(sub["any_assistant_at_cap"].isna().sum()),
            "rows_out": int(sub["any_assistant_at_cap"].notna().sum()),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sonnet_truncation.csv"
    table.to_csv(path, index=False)
    rules = filter_rules()
    prov = {
        "csv": path.name,
        "date": dt.date.today().isoformat(),
        "inputs": [
            "aggregated/all_scores.parquet",
            "scores/judge_*.jsonl",
            "openweight_subjects/scores/judge_*.jsonl",
            "transcripts/*.jsonl",
            "openweight_subjects/transcripts/*.jsonl",
        ],
        "filters": {
            "samples": {k: rules[k] for k in SAMPLES},
            "model": f"model_id == {SONNET}, then model_id == {GEMINI_PRO}",
            "conditions": f"condition in {MAIN_CONDITIONS}; all_main pools the four",
            "transcript": "rows whose run has no transcript (NA cap flags) are dropped from numerator and denominator",
            "rules": {
                "any_turn": f"any non-probe assistant message of the run has token_count == {SONNET_OUTPUT_CAP}",
                "final_turn": f"the last non-probe assistant message has token_count == {SONNET_OUTPUT_CAP}",
                "gemini_pro": (f"both rules with token_count >= {GEMINI_PRO_LOGGED_OUTPUT_CAP}: Gemini's capped outputs "
                               f"log a few tokens fewer than the cap (4,092 at the 4,096-token output cap, 1,020 at "
                               f"the 1,024-token probe cap); no Gemini Pro message exceeds 4,092"),
            },
            "ci": "Wilson score interval, z = 1.959964",
        },
        "row_counts": {**log, "sonnet_rows_by_sample": sonnet_rows},
        "july_values": {
            "any_turn as_published all_main": "226 of 315",
            "final_turn as_published all_main": "127 of 315 (the ledger's B16 numerator column)",
            "any_turn clean all_main (plan)": "221 of 304",
        },
        "ledger_note": ("corrections_ledger.csv B16 'Sonnet runs truncated (any assistant message ...)' rows hold the any-turn "
                        "count in recomputed_value and the final-turn count in numerator, so the ledger's Frac macro for "
                        "those rows prints final-turn/denominator (127/315 for all_main), not the any-turn fraction"),
        "rows_out": int(len(table)),
    }
    with open(out_dir / "sonnet_truncation.csv.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    return {"sonnet_truncation": table}


def main() -> None:
    parser = argparse.ArgumentParser(description="sonnet_truncation (camera-ready). No API calls.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    t = run(args.data_root, args.out_dir)["sonnet_truncation"]
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print(t)


if __name__ == "__main__":
    main()
