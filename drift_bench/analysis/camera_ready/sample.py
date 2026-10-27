"""The camera-ready sample rule, shared by every camera_ready module.

Plan items B5, F3 and Q5: the paper drops the 30 stray Sonnet pilot
rows. The exact rule: a run is stray when its subject is Claude Sonnet
4.6 and the UTC date of the ``timestamp`` in its ``judge_<run_id>.jsonl``
file is 2026-04-16. Stray runs may carry transcripts, structured scores,
auditor files and probe rows, so every module that reads any of those
must drop them too. Use ``stray_run_ids`` or ``drop_stray``.

No API calls. Pure disk reads.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

SONNET = "anthropic/claude-sonnet-4-6"
STRAY_JUDGE_DATE = dt.date(2026, 4, 16)
STRAY_RULE = (
    f"drop runs with model_id == {SONNET} whose judge_<run_id>.jsonl "
    f"timestamp falls on {STRAY_JUDGE_DATE} (UTC); 30 runs on the April data"
)
SCORE_SUBDIRS = ("scores", "openweight_subjects/scores")


def _judge_date_utc(timestamp: object) -> dt.date | None:
    try:
        return dt.datetime.fromtimestamp(float(timestamp), dt.timezone.utc).date()
    except (TypeError, ValueError):
        return None


def stray_run_ids(
    data_root: Path,
    stray_date: dt.date = STRAY_JUDGE_DATE,
    subdirs: tuple[str, ...] = SCORE_SUBDIRS,
) -> set[str]:
    """Return the run ids of the stray Sonnet rows under ``data_root``."""
    stray: set[str] = set()
    for sub in subdirs:
        for path in sorted((data_root / sub).glob("judge_*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                line = fh.readline()
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("model_id") != SONNET:
                continue
            if _judge_date_utc(rec.get("timestamp")) == stray_date:
                stray.add(rec.get("run_id") or path.name[len("judge_") : -len(".jsonl")])
    return stray


def drop_stray(
    df: pd.DataFrame,
    data_root: Path,
    run_col: str = "run_id",
    stray: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop stray runs from ``df``. Returns the frame and a provenance filter."""
    if stray is None:
        stray = stray_run_ids(data_root)
    mask = df[run_col].isin(stray)
    out = df.loc[~mask].reset_index(drop=True)
    info = {
        "name": "drop_stray_sonnet_runs",
        "rule": STRAY_RULE,
        "n_stray_run_ids": len(stray),
        "rows_in": int(len(df)),
        "rows_dropped": int(mask.sum()),
        "rows_out": int(len(out)),
    }
    return out, info
