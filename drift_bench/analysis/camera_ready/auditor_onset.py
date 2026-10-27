"""Auditor drift-event onset (plan items C15 and B1).

The auditor prompt asks for "up to 3 concrete drift events" with turn
numbers. Each event in `auditor_<run_id>.jsonl` has exactly two keys,
`turn` and `description`. There is no event type field, so this module
cannot report an event-type mix. It reports instead the mix of the
auditor's run-level `drift_classification` by condition, and the
distribution of event turns.

Computed:
- the number of events per run, by condition (July: exactly 3 in 100%
  of pressure runs);
- time to first flag over multi-turn runs with at least one event, by
  condition and by condition x model: share at turn 1, by turn 2, by
  turn 3 (single-shot is excluded as vacuous; checkpointed runs have 8
  turns, the other multi-turn conditions 6, so the shares are not
  comparable across conditions);
- the same over the July "drift cases": pressure runs the auditor
  classified trajectory_drift or trajectory_lock_in (July: 73% =
  239/329 by turn 2, 95% by turn 3);
- the drift classification mix and the event-turn distribution by
  condition.

Outputs: `auditor_onset_runs.csv`, `auditor_onset_summary.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready.b4_common import (
    CONDITIONS,
    DEFAULT_DATA_ROOT,
    default_out_dir,
    load_scores_with_filter,
    read_json_lines,
    score_files,
    write_csv,
)
from drift_bench.analysis.camera_ready.sample import drop_stray

DRIFT_CASE_CLASSES = ("trajectory_drift", "trajectory_lock_in")
PRESSURE = "multi_turn_pressure"


# ------------------------------------------------------------ per run


def event_turns(record: dict) -> list[int]:
    """Turn numbers of the auditor's drift events, in file order.

    Events without an integer turn are dropped.
    """
    turns = []
    for event in record.get("drift_events") or []:
        turn = event.get("turn") if isinstance(event, dict) else None
        if isinstance(turn, bool):
            continue
        if isinstance(turn, (int, float)) and not np.isnan(turn):
            turns.append(int(turn))
        elif isinstance(turn, str) and turn.strip().isdigit():
            turns.append(int(turn.strip()))
    return turns


def summarise_record(record: dict) -> dict:
    """One row per auditor record."""
    turns = event_turns(record)
    n_raw = len(record.get("drift_events") or [])
    first = min(turns) if turns else None
    return {
        "run_id": record["run_id"],
        "brief_id": record.get("brief_id"),
        "model_id": record.get("model_id"),
        "condition": record.get("condition"),
        "drift_classification": record.get("drift_classification"),
        "n_events": n_raw,
        "n_events_with_turn": len(turns),
        "event_turns": ";".join(str(t) for t in turns),
        "first_flag_turn": first,
        "last_flag_turn": max(turns) if turns else None,
        "flag_at_turn_1": None if first is None else int(first == 1),
        "flag_by_turn_2": None if first is None else int(first <= 2),
        "flag_by_turn_3": None if first is None else int(first <= 3),
        "is_drift_case": int(record.get("drift_classification") in DRIFT_CASE_CLASSES),
    }


# ------------------------------------------------------------ summary


def _onset_block(df: pd.DataFrame, section: str, group_cols: list[str]) -> list[dict]:
    """Time-to-first-flag shares for each group."""
    rows = []
    flagged = df[df["first_flag_turn"].notna()]
    if flagged.empty:
        return rows
    for key, g in flagged.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        rows.append({
            "section": section,
            **dict(zip(group_cols, key)),
            "n": n,
            "n_at_turn_1": int(g["flag_at_turn_1"].sum()),
            "share_at_turn_1": float(g["flag_at_turn_1"].mean()),
            "n_by_turn_2": int(g["flag_by_turn_2"].sum()),
            "share_by_turn_2": float(g["flag_by_turn_2"].mean()),
            "n_by_turn_3": int(g["flag_by_turn_3"].sum()),
            "share_by_turn_3": float(g["flag_by_turn_3"].mean()),
            "mean_first_flag_turn": float(g["first_flag_turn"].mean()),
            "median_first_flag_turn": float(g["first_flag_turn"].median()),
        })
    return rows


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    cond_order = [c for c in CONDITIONS if c in set(runs["condition"])]
    cond_order += sorted(set(runs["condition"]) - set(cond_order))

    # 1. Number of events per run, by condition.
    for cond in cond_order:
        g = runs[runs["condition"] == cond]
        counts = g["n_events"].value_counts().sort_index()
        for k, c in counts.items():
            rows.append({"section": "events_per_run", "condition": cond,
                         "statistic": f"n_events == {k}", "n": int(c),
                         "share": float(c / len(g)), "denominator": len(g)})
        rows.append({"section": "events_per_run", "condition": cond,
                     "statistic": "n_events == 3 (share)",
                     "n": int((g["n_events"] == 3).sum()),
                     "share": float((g["n_events"] == 3).mean()),
                     "denominator": len(g)})
        rows.append({"section": "events_per_run", "condition": cond,
                     "statistic": "mean n_events",
                     "share": float(g["n_events"].mean()), "denominator": len(g)})

    # 2. Time to first flag. Single-shot runs have one turn, so every
    # flag sits at turn 1 and the onset is vacuous; they are excluded.
    # Checkpointed runs have 8 turns against 6 for the other multi-turn
    # conditions, so "by turn 2" is a different fraction of the run.
    onset = runs[runs["condition"] != "single_shot"]
    rows.extend(_onset_block(onset, "first_flag_by_condition", ["condition"]))
    rows.extend(_onset_block(onset, "first_flag_by_condition_model",
                             ["condition", "model_id"]))
    drift_cases = runs[(runs["condition"] == PRESSURE) & (runs["is_drift_case"] == 1)]
    rows.extend(_onset_block(drift_cases.assign(subset="pressure drift cases"),
                             "first_flag_pressure_drift_cases", ["subset"]))
    rows.extend(_onset_block(drift_cases, "first_flag_pressure_drift_cases_by_model",
                             ["model_id"]))
    td_only = runs[(runs["condition"] == PRESSURE)
                   & (runs["drift_classification"] == "trajectory_drift")]
    rows.extend(_onset_block(td_only.assign(subset="pressure trajectory_drift only"),
                             "first_flag_pressure_trajectory_drift", ["subset"]))

    # 3. Drift classification mix by condition (no event type field exists).
    for cond in cond_order:
        g = runs[runs["condition"] == cond]
        for cls, c in g["drift_classification"].value_counts().items():
            rows.append({"section": "classification_mix", "condition": cond,
                         "statistic": cls, "n": int(c),
                         "share": float(c / len(g)), "denominator": len(g)})

    # 4. Event turn distribution by condition (all events pooled).
    for cond in cond_order:
        g = runs[runs["condition"] == cond]
        turns = [int(t) for s in g["event_turns"] if s for t in s.split(";")]
        if not turns:
            continue
        vc = pd.Series(turns).value_counts().sort_index()
        for t, c in vc.items():
            rows.append({"section": "event_turn_mix", "condition": cond,
                         "statistic": f"event at turn {t}", "n": int(c),
                         "share": float(c / len(turns)), "denominator": len(turns)})

    out = pd.DataFrame(rows)
    front = ["section", "condition", "model_id", "subset", "statistic"]
    cols = [c for c in front if c in out.columns] + \
        [c for c in out.columns if c not in front]
    return out[cols]


# ------------------------------------------------------------ run


def load_auditor_runs(data_root: Path) -> tuple[pd.DataFrame, dict]:
    files = score_files(data_root, "auditor")
    records = []
    for f in files:
        records.extend(read_json_lines(f))
    raw = pd.DataFrame([summarise_record(r) for r in records])
    n_records = len(raw)
    raw = raw.drop_duplicates(subset="run_id", keep="last")
    n_unique = len(raw)

    raw, stray_filter = drop_stray(raw, data_root)
    scores, _ = load_scores_with_filter(data_root)
    runs = raw[raw["run_id"].isin(scores["run_id"])].copy()
    runs = runs.sort_values(["condition", "model_id", "run_id"]).reset_index(drop=True)
    prov = {
        "inputs": [
            str(data_root / "scores" / "auditor_*.jsonl"),
            str(data_root / "openweight_subjects" / "scores" / "auditor_*.jsonl"),
            str(data_root / "aggregated" / "all_scores.parquet"),
        ],
        "filters": [
            {"rule": "one record per run_id (keep last line if duplicated)",
             "rows_in": n_records, "rows_out": n_unique},
            stray_filter,
            {"rule": "run_id is in the clean parquet",
             "rows_in": int(stray_filter["rows_out"]), "rows_out": int(len(runs))},
        ],
        "definitions": {
            "first_flag_turn": "min of drift_events[*].turn; NaN when the record has no event",
            "drift case": f"condition == {PRESSURE} and drift_classification in {list(DRIFT_CASE_CLASSES)} (the July 329-run denominator)",
            "event type": "not available: events carry only `turn` and `description`",
            "onset sections": "single_shot excluded (one turn, every flag at turn 1, vacuous); checkpointed_pressure runs have 8 turns and multi_turn_* 6, so by-turn-k shares are not comparable across conditions",
            "turns per condition": {"single_shot": 1, "multi_turn_neutral": 6, "multi_turn_pressure": 6, "checkpointed_pressure": 8},
        },
    }
    return runs, prov


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path | None = None
        ) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or default_out_dir(data_root)
    runs, prov = load_auditor_runs(data_root)
    write_csv(runs, out_dir, "auditor_onset_runs", prov)
    summary = build_summary(runs)
    prov_s = dict(prov)
    prov_s["derived_from"] = "auditor_onset_runs.csv (same filters)"
    prov_s["filters"] = prov["filters"] + [
        {"rule": "first_flag sections: condition != single_shot",
         "rows_in": int(len(runs)),
         "rows_out": int((runs["condition"] != "single_shot").sum())},
        {"rule": "first_flag sections: runs with at least one event",
         "rows_in": int((runs["condition"] != "single_shot").sum()),
         "rows_out": int(runs[runs["condition"] != "single_shot"]["first_flag_turn"].notna().sum())},
    ]
    write_csv(summary, out_dir, "auditor_onset_summary", prov_s)
    return {"auditor_onset_runs": runs, "auditor_onset_summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    s = tables["auditor_onset_summary"]
    print(s[s["section"].isin(["events_per_run", "first_flag_by_condition",
                                "first_flag_pressure_drift_cases",
                                "first_flag_pressure_trajectory_drift"])].to_string(index=False))


if __name__ == "__main__":
    main()
