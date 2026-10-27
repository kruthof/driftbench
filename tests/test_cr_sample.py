"""Tests for the shared camera-ready sample rule (stray Sonnet runs)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready.sample import (
    SONNET,
    drop_stray,
    stray_run_ids,
)


def _ts(day: int, hour: int) -> float:
    return dt.datetime(2026, 4, day, hour, 0, tzinfo=dt.timezone.utc).timestamp()


def _write_judge(root: Path, sub: str, run_id: str, model_id: str, ts: float) -> None:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": run_id, "model_id": model_id, "timestamp": ts, "constraint_adherence": 3}
    (d / f"judge_{run_id}.jsonl").write_text(json.dumps(rec) + "\n")


def test_stray_rule_is_sonnet_and_utc_date(tmp_path: Path) -> None:
    _write_judge(tmp_path, "scores", "aaaaaaaaaaaa", SONNET, _ts(16, 19))  # stray
    _write_judge(tmp_path, "scores", "bbbbbbbbbbbb", SONNET, _ts(16, 23))  # stray, late UTC
    _write_judge(tmp_path, "scores", "cccccccccccc", SONNET, _ts(17, 2))  # main pass
    _write_judge(tmp_path, "scores", "dddddddddddd", "openai/gpt-5.4", _ts(16, 19))  # not Sonnet
    _write_judge(tmp_path, "openweight_subjects/scores", "eeeeeeeeeeee", SONNET, _ts(16, 19))
    assert stray_run_ids(tmp_path) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb", "eeeeeeeeeeee"}


def test_missing_timestamp_is_not_stray(tmp_path: Path) -> None:
    d = tmp_path / "scores"
    d.mkdir()
    (d / "judge_ffffffffffff.jsonl").write_text(json.dumps({"run_id": "ffffffffffff", "model_id": SONNET}) + "\n")
    assert stray_run_ids(tmp_path) == set()


def test_drop_stray_reports_counts(tmp_path: Path) -> None:
    _write_judge(tmp_path, "scores", "aaaaaaaaaaaa", SONNET, _ts(16, 19))
    df = pd.DataFrame({"run_id": ["aaaaaaaaaaaa", "aaaaaaaaaaaa", "zzzzzzzzzzzz"], "x": [1, 2, 3]})
    out, info = drop_stray(df, tmp_path)
    assert list(out["run_id"]) == ["zzzzzzzzzzzz"]
    assert info["rows_in"] == 3 and info["rows_dropped"] == 2 and info["rows_out"] == 1
    assert info["n_stray_run_ids"] == 1
    assert "2026-04-16" in info["rule"]
