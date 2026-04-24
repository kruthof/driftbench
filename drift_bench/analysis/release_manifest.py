"""Generate a machine-readable release manifest from the current repository snapshot.

This script exists to keep manuscript counts, artifact inventories, and
human-validation status tied to the files that are actually present on disk.

Usage:
    python -m drift_bench.analysis.release_manifest
    python -m drift_bench.analysis.release_manifest --output drift_bench/data/analysis/release_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "drift_bench"
DATA_DIR = BENCH_DIR / "data"
HUMAN_VALIDATION_SET_1 = DATA_DIR / "human_validation" / "scoring_items.json"
HUMAN_VALIDATION_SET_2 = ROOT / "scoring_app_standalone_set2" / "scoring_items_set2.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path) -> Any:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_json_array_len(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else None


def _redacted_rater_summary(raters_dir: Path) -> dict[str, Any]:
    rater_files = sorted(raters_dir.glob("*.json"))
    run_to_ratings: dict[str, int] = {}
    per_file_counts = []

    for idx, path in enumerate(rater_files, start=1):
        with path.open() as f:
            data = json.load(f)
        per_file_counts.append({
            "rater_file_index": idx,
            "scores_submitted": len(data),
        })
        for run_id in data:
            run_to_ratings[run_id] = run_to_ratings.get(run_id, 0) + 1

    coverage = Counter(run_to_ratings.values())
    return {
        "completed_rater_files": len(rater_files),
        "scores_per_rater_file": per_file_counts,
        "unique_runs_with_completed_ratings": len(run_to_ratings),
        "ratings_per_run_histogram": {str(k): v for k, v in sorted(coverage.items())},
        "raw_rater_ids_redacted": True,
    }


def _summarize_jsonl_runs(scores_dir: Path, pattern: str, description: str) -> dict[str, Any]:
    rows = []
    for path in sorted(scores_dir.glob(pattern)):
        for rec in _iter_jsonl(path):
            rows.append(rec)

    if not rows:
        return {
            "description": description,
            "status": "missing",
            "files": 0,
            "rows": 0,
            "unique_runs": 0,
        }

    run_ids = [row["run_id"] for row in rows if "run_id" in row]
    return {
        "description": description,
        "status": "present",
        "files": len(list(scores_dir.glob(pattern))),
        "rows": len(rows),
        "unique_runs": len(set(run_ids)),
    }


def _load_scored_runs_from_jsonl(scores_dir: Path) -> pd.DataFrame:
    judge_rows = []
    auditor_rows = []

    for path in sorted(scores_dir.glob("judge_*.jsonl")):
        for rec in _iter_jsonl(path):
            judge_rows.append({
                "run_id": rec["run_id"],
                "brief_id": rec["brief_id"],
                "model_id": rec["model_id"],
                "condition": rec["condition"],
            })

    for path in sorted(scores_dir.glob("auditor_*.jsonl")):
        for rec in _iter_jsonl(path):
            auditor_rows.append({"run_id": rec["run_id"]})

    judge_df = pd.DataFrame(judge_rows)
    auditor_df = pd.DataFrame(auditor_rows)

    if not judge_df.empty and not auditor_df.empty:
        return judge_df.merge(auditor_df, on="run_id", how="inner")
    return judge_df


def _load_scored_snapshot(
    aggregated_candidates: list[Path],
    fallback_scores_dir: Path | None = None,
) -> pd.DataFrame:
    for path in aggregated_candidates:
        if path.exists():
            return pd.read_parquet(path)
    if fallback_scores_dir and fallback_scores_dir.exists():
        return _load_scored_runs_from_jsonl(fallback_scores_dir)
    return pd.DataFrame()


def _summarize_scored_df(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"status": "missing"}

    summary: dict[str, Any] = {
        "status": "present",
        "rows": int(len(df)),
    }

    if "run_id" in df.columns:
        summary["unique_runs"] = int(df["run_id"].nunique())
        summary["duplicate_run_ids"] = int(df["run_id"].duplicated().sum())
    if "brief_id" in df.columns:
        summary["brief_count"] = int(df["brief_id"].nunique())
    if "model_id" in df.columns:
        models = sorted(df["model_id"].dropna().unique().tolist())
        summary["model_count"] = len(models)
        summary["models"] = models
        summary["rows_by_model"] = {
            str(k): int(v) for k, v in df.groupby("model_id").size().items()
        }
    if "condition" in df.columns:
        conditions = sorted(df["condition"].dropna().unique().tolist())
        summary["condition_count"] = len(conditions)
        summary["conditions"] = conditions
        summary["rows_by_condition"] = {
            str(k): int(v) for k, v in df.groupby("condition").size().items()
        }

    if {"brief_id", "model_id", "condition"}.issubset(df.columns):
        cell_counts = (
            df.groupby(["brief_id", "model_id", "condition"])
            .size()
            .rename("n")
            .reset_index()
        )
        expected = int(cell_counts["n"].mode().iloc[0]) if not cell_counts.empty else 0
        summary["cell_count_summary"] = {
            "expected_mode": expected,
            "min": int(cell_counts["n"].min()),
            "max": int(cell_counts["n"].max()),
            "irregular_cells": int((cell_counts["n"] != expected).sum()),
        }

    return summary


def _format_tex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _build_manuscript_macros(manifest: dict[str, Any]) -> dict[str, str]:
    benchmark = manifest["analysis_scopes"]["benchmark_release"]
    human_validation = manifest["human_validation"]
    blind_validation = manifest["judge_validation_artifacts"]["blind_judge_core_subset"]
    structured_validation = manifest["judge_validation_artifacts"]["structured_judge_core_subset"]

    ratings_histogram = human_validation["ratings_per_run_histogram"]
    if len(ratings_histogram) == 1:
        ratings_per_item = next(iter(ratings_histogram))
    else:
        ratings_per_item = "mixed"

    set_1_size = human_validation.get("set_1_items")
    set_2_size = human_validation.get("set_2_items")
    set_size = set_1_size if set_1_size == set_2_size else f"{set_1_size}/{set_2_size}"

    return {
        "BenchmarkRuns": _format_tex_int(benchmark["rows"]),
        "BlindJudgeRuns": _format_tex_int(blind_validation["unique_runs"]),
        "StructuredJudgeRuns": _format_tex_int(structured_validation["unique_runs"]),
        "HumanValidationRuns": _format_tex_int(human_validation["unique_runs_with_completed_ratings"]),
        "HumanValidationSetCount": str(human_validation["scoring_sets"]),
        "HumanValidationSetSize": str(set_size),
        "HumanValidationSetOneSize": str(set_1_size),
        "HumanValidationSetTwoSize": str(set_2_size),
        "HumanValidationCompletedRaterFiles": str(human_validation["completed_rater_files"]),
        "HumanValidationRatingsPerItem": str(ratings_per_item),
        "HumanValidationStatus": "current validation snapshot",
        "HumanValidationRefreshNote": (
            "Additional blind ratings are still being collected outside the current repository snapshot; "
            "final counts and agreement estimates will be refreshed at submission lock."
        ),
    }


def _write_tex_macros(manifest: dict[str, Any], output_path: Path) -> None:
    macros = _build_manuscript_macros(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Auto-generated by python -m drift_bench.analysis.release_manifest",
        "% Do not edit by hand; regenerate from the current repository snapshot.",
    ]
    for name, value in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{_latex_escape(value)}}}")

    with output_path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def build_manifest() -> dict[str, Any]:
    core = _load_scored_snapshot(
        [DATA_DIR / "aggregated" / "all_scores.parquet"],
        fallback_scores_dir=DATA_DIR / "scores",
    )
    openweight = _load_scored_snapshot(
        [
            DATA_DIR / "openweight_subjects" / "aggregated" / "scores.parquet",
            DATA_DIR / "openweight_subjects" / "aggregated" / "all_scores.parquet",
        ],
        fallback_scores_dir=DATA_DIR / "openweight_subjects" / "scores",
    )
    monitored = _load_scored_snapshot(
        [DATA_DIR / "monitored" / "aggregated" / "scores.parquet"],
        fallback_scores_dir=DATA_DIR / "monitored" / "scores",
    )
    followup_a = _load_scored_snapshot(
        [DATA_DIR / "followup_a" / "aggregated" / "all_scores.parquet"],
        fallback_scores_dir=DATA_DIR / "followup_a" / "scores",
    )
    followup_b = _load_scored_snapshot(
        [DATA_DIR / "followup_b" / "aggregated" / "all_scores.parquet"],
        fallback_scores_dir=DATA_DIR / "followup_b" / "scores",
    )

    benchmark = pd.concat([core, openweight], ignore_index=True) if not openweight.empty else core
    human_validation_summary = _redacted_rater_summary(DATA_DIR / "human_validation" / "raters")
    human_validation_summary["scoring_sets"] = 2
    human_validation_summary["set_1_items"] = _load_json_array_len(HUMAN_VALIDATION_SET_1)
    human_validation_summary["set_2_items"] = _load_json_array_len(HUMAN_VALIDATION_SET_2)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_inventory": {
            "brief_yaml_files": len(list((BENCH_DIR / "briefs").glob("*.yaml"))),
            "prompt_templates": len(list((BENCH_DIR / "prompts").glob("*.j2"))),
            "judge_modules": len(list((BENCH_DIR / "judges").glob("*.py"))),
            "schema_files": len(list((BENCH_DIR / "schema").glob("*"))),
            "core_transcripts": len(list((DATA_DIR / "transcripts").glob("*.jsonl"))),
            "openweight_transcripts": len(list((DATA_DIR / "openweight_subjects" / "transcripts").glob("*.jsonl"))),
            "monitored_transcripts": len(list((DATA_DIR / "monitored" / "transcripts").glob("*.jsonl"))),
            "followup_a_transcripts": len(list((DATA_DIR / "followup_a" / "transcripts").glob("*.jsonl"))),
            "followup_b_transcripts": len(list((DATA_DIR / "followup_b" / "transcripts").glob("*.jsonl"))),
        },
        "analysis_scopes": {
            "benchmark_release": {
                "description": "Main seven-model benchmark release used for the paper's aggregate tables: five-model core benchmark plus the open-weight subject extension.",
                **_summarize_scored_df(benchmark),
            },
            "core_five_model_subset": {
                "description": "Five-model commercial/core subset used by the main aggregated parquet and the blind/structured judge comparisons.",
                **_summarize_scored_df(core),
            },
            "openweight_subject_extension": {
                "description": "Open-weight subject-model extension scored separately and combined with the core subset for benchmark totals.",
                **_summarize_scored_df(openweight),
            },
            "monitored_followup": {
                "description": "Constraint-monitoring intervention runs reported separately from the main benchmark totals.",
                **_summarize_scored_df(monitored),
            },
            "gemini_temperature_followup": {
                "description": "Gemini Flash rerun at temperature 0.7; ancillary sensitivity analysis, not part of benchmark totals.",
                **_summarize_scored_df(followup_a),
            },
            "rigor_pressure_followup": {
                "description": "Rigor-pressure follow-up; ancillary sensitivity analysis, not part of benchmark totals.",
                **_summarize_scored_df(followup_b),
            },
        },
        "judge_validation_artifacts": {
            "blind_judge_core_subset": _summarize_jsonl_runs(
                DATA_DIR / "scores",
                "blind_*.jsonl",
                "Blind-judge comparison files for the core five-model subset.",
            ),
            "structured_judge_core_subset": _summarize_jsonl_runs(
                DATA_DIR / "scores",
                "structured_*.jsonl",
                "Structured-judge comparison files for the core five-model subset.",
            ),
        },
        "human_validation": {
            "status": "interim_snapshot",
            "notes": [
                "The current repository snapshot contains completed blinded ratings only.",
                "Additional ratings can be incorporated later without changing the scoring protocol or artifact layout.",
                "Rater identities are intentionally redacted in this manifest; only aggregate completion counts are reported.",
            ],
            "set_1_items_path": str(HUMAN_VALIDATION_SET_1.relative_to(ROOT)),
            "set_2_items_path": str(HUMAN_VALIDATION_SET_2.relative_to(ROOT)),
            **human_validation_summary,
        },
        "key_files": {
            str(path.relative_to(ROOT)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in [
                ROOT / "README.md",
                ROOT / "neurips_2026.tex",
                BENCH_DIR / "config.yaml",
                BENCH_DIR / "config_followup_A.yaml",
                BENCH_DIR / "config_followup_B.yaml",
                BENCH_DIR / "config_monitored.yaml",
                BENCH_DIR / "config_openweight_subjects.yaml",
                DATA_DIR / "aggregated" / "all_scores.parquet",
                DATA_DIR / "openweight_subjects" / "aggregated" / "scores.parquet",
                DATA_DIR / "monitored" / "aggregated" / "scores.parquet",
            ]
            if path.exists()
        },
        "reproduction_commands": [
            "python -m drift_bench.pipeline --max-budget 400",
            "python -m drift_bench.run_openweight --experiment subjects",
            "python -m drift_bench.run_followup --experiment both",
            "python -m drift_bench.run_monitored",
            "python -m drift_bench.analysis.release_manifest --output drift_bench/data/analysis/release_manifest.json",
        ],
    }
    manifest["manuscript_counts"] = _build_manuscript_macros(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a DriftBench release manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "analysis" / "release_manifest.json",
        help="Where to write the manifest JSON.",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=DATA_DIR / "analysis" / "release_macros.tex",
        help="Where to write manuscript macros derived from the same manifest.",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    _write_tex_macros(manifest, args.tex_output)
    print(f"Wrote release manifest to {args.output}")
    print(f"Wrote manuscript macros to {args.tex_output}")


if __name__ == "__main__":
    main()
