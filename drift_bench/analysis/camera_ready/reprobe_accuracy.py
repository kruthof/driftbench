"""Score replayed restatement probes (E3) and compare with the April final-probe accuracy.

Inputs:
- ``<data_root>/camera_ready/e3_reprobe_gemini_pro/reprobe_*.jsonl`` written by
  ``drift_bench.runners.reprobe`` (one line per replayed probe).
- ``<data_root>/analysis/final_probe_accuracy.csv`` (April, all 7 models).
- ``drift_bench/briefs/*.yaml``.

Outputs (in ``out_dir``, each with a ``.provenance.json`` sibling):
- ``reprobe_accuracy.csv``: one row per replayed probe with the original and the
  new ``overall_accuracy``, ``constraint_recall``, ``banned_recall`` and the
  original truncation flag.
- ``reprobe_summary.csv``: per model and condition: n, mean original and new
  accuracy, share with the original truncated, share >= 0.8 before and after.

Scoring uses ``drift_bench.analysis.probes.score_probe_accuracy``, the same
keyword matcher that produced the April CSV. When the reprobe directory does
not exist yet, both CSVs are written with headers and no rows.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from drift_bench.analysis.probes import load_brief, score_probe_accuracy

PACKAGE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_DIR / "data"
OUT_DIR = DATA_ROOT / "camera_ready" / "analysis"
REPROBE_SUBDIR = Path("camera_ready") / "e3_reprobe_gemini_pro"
FINAL_PROBE_CSV = Path("analysis") / "final_probe_accuracy.csv"

# The paper's "knows" threshold on probe accuracy (KBV definition).
ACCURACY_THRESHOLD = 0.8

# File prefix written by drift_bench.runners.reprobe (kept here so this module
# does not import the runner, which imports litellm).
OUTPUT_PREFIX = "reprobe_"
# The runner's per-invocation summary log. It matches the glob but holds no probes.
RUN_LOG_NAME = "reprobe_run_log.jsonl"

ACCURACY_COLUMNS = [
    "run_id", "brief_id", "model_id", "condition", "turn_index",
    "original_probe_tokens", "original_truncated",
    "original_overall_accuracy", "original_constraint_recall", "original_banned_recall",
    "new_probe_tokens", "new_finish_reason",
    "new_overall_accuracy", "new_constraint_recall", "new_banned_recall",
    "accuracy_delta", "probe_max_tokens", "temperature", "cost_usd",
]

SUMMARY_COLUMNS = [
    "model_id", "condition", "n", "n_with_original",
    "mean_original_accuracy", "mean_new_accuracy", "mean_accuracy_delta",
    "share_original_truncated", "share_new_finish_length",
    "share_original_ge_0_8", "share_new_ge_0_8",
    "mean_original_tokens", "mean_new_tokens",
]


def _read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line. A damaged line is skipped."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_reprobe_records(reprobe_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read every reprobe_*.jsonl line. Returns the frame and the list of files read."""
    if not reprobe_dir.is_dir():
        raise FileNotFoundError(
            f"reprobe directory not found: {reprobe_dir}; pass --reprobe-dir or run runners.reprobe first"
        )
    files = sorted(reprobe_dir.glob(f"{OUTPUT_PREFIX}*.jsonl"))
    files = [p for p in files if p.name != RUN_LOG_NAME]
    records = []
    for path in files:
        records.extend(_read_jsonl(path))
    return pd.DataFrame(records), [str(p) for p in files]


def score_records(records: pd.DataFrame, briefs_dir: Path) -> tuple[pd.DataFrame, int]:
    """Score each replayed probe against its brief. Returns the frame and the count without a brief."""
    if records.empty:
        return pd.DataFrame(), 0
    briefs: dict[str, dict | None] = {}
    rows = []
    missing_brief = 0
    for rec in records.to_dict("records"):
        brief_id = rec["brief_id"]
        if brief_id not in briefs:
            briefs[brief_id] = load_brief(brief_id, briefs_dir)
        brief = briefs[brief_id]
        if brief is None:
            missing_brief += 1
            continue
        acc = score_probe_accuracy(rec.get("new_probe_content") or "", brief)
        rows.append({
            "run_id": rec["run_id"],
            "brief_id": brief_id,
            "model_id": rec["model_id"],
            "condition": rec["condition"],
            "turn_index": rec["turn_index"],
            "original_probe_tokens": rec.get("original_probe_tokens"),
            "original_truncated": bool(rec.get("original_probe_truncated", False)),
            "new_probe_tokens": rec.get("new_probe_tokens"),
            "new_finish_reason": rec.get("new_finish_reason"),
            "new_overall_accuracy": acc["overall_accuracy"],
            "new_constraint_recall": acc["constraint_recall"],
            "new_banned_recall": acc["banned_recall"],
            "probe_max_tokens": rec.get("probe_max_tokens"),
            "temperature": rec.get("temperature"),
            "cost_usd": rec.get("cost_usd"),
        })
    return pd.DataFrame(rows), missing_brief


def join_original(scored: pd.DataFrame, final_probe_csv: Path) -> tuple[pd.DataFrame, int]:
    """Left-join the April accuracy on (run_id, turn_index). Returns the frame and the join count."""
    if scored.empty:
        return pd.DataFrame(columns=ACCURACY_COLUMNS), 0
    if final_probe_csv.exists():
        orig = pd.read_csv(final_probe_csv)[
            ["run_id", "turn_index", "overall_accuracy", "constraint_recall", "banned_recall"]
        ].rename(columns={
            "overall_accuracy": "original_overall_accuracy",
            "constraint_recall": "original_constraint_recall",
            "banned_recall": "original_banned_recall",
        })
        orig = orig.drop_duplicates(subset=["run_id", "turn_index"])
    else:
        orig = pd.DataFrame(columns=[
            "run_id", "turn_index", "original_overall_accuracy",
            "original_constraint_recall", "original_banned_recall",
        ])
    merged = scored.merge(orig, on=["run_id", "turn_index"], how="left")
    merged["accuracy_delta"] = merged["new_overall_accuracy"] - merged["original_overall_accuracy"]
    n_joined = int(merged["original_overall_accuracy"].notna().sum())
    return merged[ACCURACY_COLUMNS], n_joined


def summarize(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Per model and condition summary.

    ``share_original_ge_0_8`` and ``share_new_ge_0_8`` are fractions of
    ``n_with_original`` (rows with an April accuracy), so before and after use
    the same denominator. The other ``share_*`` columns are fractions of n.
    """
    if accuracy.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = accuracy.copy()
    has_orig = df["original_overall_accuracy"].notna()
    # NaN where there is no original, so the mean runs over n_with_original.
    df["orig_ge"] = (df["original_overall_accuracy"] >= ACCURACY_THRESHOLD).where(has_orig)
    df["new_ge"] = (df["new_overall_accuracy"] >= ACCURACY_THRESHOLD).where(has_orig)
    df["new_length"] = df["new_finish_reason"].astype(str).str.lower().eq("length")
    df["has_original"] = df["original_overall_accuracy"].notna()
    out = (
        df.groupby(["model_id", "condition"])
        .agg(
            n=("run_id", "size"),
            n_with_original=("has_original", "sum"),
            mean_original_accuracy=("original_overall_accuracy", "mean"),
            mean_new_accuracy=("new_overall_accuracy", "mean"),
            mean_accuracy_delta=("accuracy_delta", "mean"),
            share_original_truncated=("original_truncated", "mean"),
            share_new_finish_length=("new_length", "mean"),
            share_original_ge_0_8=("orig_ge", "mean"),
            share_new_ge_0_8=("new_ge", "mean"),
            mean_original_tokens=("original_probe_tokens", "mean"),
            mean_new_tokens=("new_probe_tokens", "mean"),
        )
        .reset_index()
    )
    return out[SUMMARY_COLUMNS]


def _write_csv(df: pd.DataFrame, path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    provenance = {**provenance, "rows_out": int(len(df)), "date": date.today().isoformat()}
    with open(path.with_name(path.name + ".provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


def run(
    data_root: Path = DATA_ROOT,
    out_dir: Path = OUT_DIR,
    reprobe_dir: Path | None = None,
    briefs_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    reprobe_dir = Path(reprobe_dir) if reprobe_dir else data_root / REPROBE_SUBDIR
    briefs_dir = Path(briefs_dir) if briefs_dir else data_root.parent / "briefs"
    final_probe_csv = data_root / FINAL_PROBE_CSV

    records, files = load_reprobe_records(reprobe_dir)
    scored, missing_brief = score_records(records, briefs_dir)
    accuracy, n_joined = join_original(scored, final_probe_csv)
    summary = summarize(accuracy)

    inputs = {
        "reprobe_glob": str(reprobe_dir / f"{OUTPUT_PREFIX}*.jsonl"),
        "reprobe_files": len(files),
        "final_probe_accuracy_csv": str(final_probe_csv),
        "final_probe_accuracy_exists": final_probe_csv.exists(),
        "briefs_dir": str(briefs_dir),
    }
    _write_csv(accuracy, out_dir / "reprobe_accuracy.csv", {
        "inputs": inputs,
        "filters": [
            "one row per replayed probe line; damaged JSON lines skipped",
            "rows whose brief_id has no YAML in briefs_dir are dropped",
            "original_* columns: left join on (run_id, turn_index) to final_probe_accuracy.csv; "
            "non-final probes get NaN",
            "original_truncated: original_probe_tokens >= 1020 (flag set by reprobe.py)",
            "scoring: drift_bench.analysis.probes.score_probe_accuracy on new_probe_content",
        ],
        "rows_in": int(len(records)),
        "rows_without_brief": int(missing_brief),
        "rows_joined_to_original": int(n_joined),
    })
    _write_csv(summary, out_dir / "reprobe_summary.csv", {
        "inputs": {"reprobe_accuracy_csv": str(out_dir / "reprobe_accuracy.csv")},
        "filters": [
            "group by model_id, condition",
            f"share_*_ge_0_8: overall_accuracy >= {ACCURACY_THRESHOLD} (KBV 'knows' threshold), "
            "fraction of n_with_original (rows with an April accuracy); NaN when that is 0",
            "share_original_truncated: mean of original_truncated",
            "share_new_finish_length: new_finish_reason == 'length'",
            "share_original_truncated and share_new_finish_length: fraction of n",
            "mean_original_accuracy and mean_accuracy_delta ignore NaN (rows with no original accuracy)",
        ],
        "rows_in": int(len(accuracy)),
    })
    return {"reprobe_accuracy": accuracy, "reprobe_summary": summary}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--reprobe-dir", type=Path, default=None)
    p.add_argument("--briefs-dir", type=Path, default=None)
    args = p.parse_args()
    out = run(args.data_root, args.out_dir, args.reprobe_dir, args.briefs_dir)
    for name, df in out.items():
        print(f"{name}: {len(df)} rows -> {Path(args.out_dir) / (name + '.csv')}")
    if not out["reprobe_summary"].empty:
        print(out["reprobe_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
