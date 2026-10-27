"""Shared helpers for the B4 camera-ready modules.

Used by tokens_context, auditor_onset, per_model_tables and
constraint_count. Pure disk reads, no API calls.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "drift_bench" / "data"
DEFAULT_BRIEFS_DIR = REPO_ROOT / "drift_bench" / "briefs"


def default_out_dir(data_root: Path) -> Path:
    return data_root / "camera_ready" / "analysis"


CONDITIONS = [
    "single_shot",
    "multi_turn_neutral",
    "multi_turn_pressure",
    "checkpointed_pressure",
]

CONDITION_LABELS = {
    "single_shot": "Single-shot",
    "multi_turn_neutral": "Multi-turn neutral",
    "multi_turn_pressure": "Multi-turn pressure",
    "checkpointed_pressure": "Checkpointed pressure",
}

MODEL_ORDER = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
]

MODEL_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4-mini",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "Qwen3-235B",
}

# KBV, as defined in the rules file: the final restatement probe scores
# at least 0.8 and the published-arm judge gives adherence below 4.
KBV_PROBE_THRESHOLD = 0.8
KBV_ADHERENCE_THRESHOLD = 4


# ---------------------------------------------------------------- loading


def load_scores_with_filter(data_root: Path, clean: bool = True
                            ) -> tuple[pd.DataFrame, dict | None]:
    """The score aggregate and the sample filter that was applied.

    `clean=True` (the paper's sample) drops the 30 stray Sonnet pilot
    rows by the shared rule in `sample.py` and returns its provenance
    filter dict. `clean=False` returns the as-published parquet and
    None.
    """
    from drift_bench.analysis.camera_ready.sample import drop_stray

    df = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
    if not clean:
        return df, None
    return drop_stray(df, data_root)


def load_scores(data_root: Path, clean: bool = True) -> pd.DataFrame:
    """One row per scored run; the clean sample unless `clean=False`."""
    return load_scores_with_filter(data_root, clean)[0]


def transcript_files(data_root: Path) -> list[Path]:
    """Transcript files from the commercial and the open-weight trees."""
    dirs = [
        data_root / "transcripts",
        data_root / "openweight_subjects" / "transcripts",
    ]
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.jsonl")))
    return files


def score_files(data_root: Path, prefix: str) -> list[Path]:
    """Per-run score files with the given prefix, both trees."""
    dirs = [
        data_root / "scores",
        data_root / "openweight_subjects" / "scores",
    ]
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob(f"{prefix}_*.jsonl")))
    return files


def read_json_lines(path: Path) -> list[dict]:
    """Read every non-empty line of a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_transcript(path: Path) -> dict:
    """A transcript file holds one JSON object on its first line."""
    return read_json_lines(path)[0]


def load_briefs(briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> dict[str, dict]:
    """All briefs, keyed by brief id."""
    briefs = {}
    for path in sorted(briefs_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            brief = yaml.safe_load(fh)
        briefs[brief["id"]] = brief
    return briefs


def final_probe_table(data_root: Path, clean: bool = True) -> pd.DataFrame:
    """Final-turn probe accuracy per run (2026-07-26 probe outputs).

    Stray Sonnet runs are dropped by the shared rule unless `clean=False`.
    """
    from drift_bench.analysis.camera_ready.sample import drop_stray

    path = data_root / "analysis" / "final_probe_accuracy.csv"
    df = pd.read_csv(path)
    if clean:
        df, _ = drop_stray(df, data_root)
    return df


def attach_kbv(scores: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """Join final probe accuracy to the scores and compute KBV.

    Adds `final_probe_accuracy`, `knows`, `noncompliant`, `kbv`.
    Runs without a probe record get NaN for `knows` and `kbv`.
    """
    keep = probes[["run_id", "overall_accuracy"]].rename(
        columns={"overall_accuracy": "final_probe_accuracy"}
    )
    out = scores.merge(keep, on="run_id", how="left")
    out["noncompliant"] = (
        out["j_constraint_adherence"] < KBV_ADHERENCE_THRESHOLD
    ).astype(int)
    has = out["final_probe_accuracy"].notna()
    out["knows"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[has, "knows"] = (
        out.loc[has, "final_probe_accuracy"] >= KBV_PROBE_THRESHOLD
    ).astype(int)
    out["kbv"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[has, "kbv"] = (
        (out.loc[has, "knows"] == 1) & (out.loc[has, "noncompliant"] == 1)
    ).astype(int)
    return out


# ---------------------------------------------------------------- writing


def write_csv(
    df: pd.DataFrame, out_dir: Path, name: str, provenance: dict
) -> Path:
    """Write `<name>.csv` and its `<name>.csv.provenance.json` sibling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    prov = dict(provenance)
    prov.setdefault("date", date.today().isoformat())
    prov["output"] = str(csv_path)
    prov["rows_written"] = int(len(df))
    with open(out_dir / f"{name}.csv.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=_json_default)
    return csv_path


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(str(text).split())
