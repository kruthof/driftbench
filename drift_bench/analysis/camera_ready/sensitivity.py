"""Sensitivity cells of the appendix figure ``fig_sensitivity.pdf`` (plan item X7).

Disk only. No API calls.

The figure compares KBV in two follow-up experiments with the April cells:

* ``temperature``: Gemini Flash-Lite at temperature 0.7 (``followup_a``,
  two repetitions, conditions neutral, pressure, checkpointed) beside the
  April temperature-1.0 cells.
* ``pressure_type``: rigor pressure (``followup_b``, condition
  ``multi_turn_pressure_rigor``, one repetition) for GPT-5.4, Sonnet 4.6 and
  Flash-Lite beside the April novelty-pressure cells.

Until now ``figures_cr.py`` recomputed the follow-up cells on the fly, so no
CSV of record held them and no ``\\cr`` macro could cite them. This module
writes that CSV.

Follow-up cells: the follow-up parquet gives the published-arm judge scores
(Opus for GPT-5.4 and Flash-Lite, GPT-5.4 for Sonnet, as in April). Probe
recall is recomputed with the same scorer that produced
``analysis/final_probe_accuracy.csv`` for the April cells
(``drift_bench.analysis.probes.analyze_probes_for_run`` on every probe, then
``compute_final_probe_accuracy`` for the last one). The stray rule
(``sample.STRAY_RULE``) is applied to the follow-up's own judge files; no
follow-up run is stray.

April cells: the clean rows of ``camera_ready/analysis/kbv_published_arm.csv``
(the CSV of record). They are also recomputed here from the main parquet and
``analysis/final_probe_accuracy.csv`` as a check; the provenance file records
whether the check passed.

KBV: final-probe ``overall_accuracy >= 0.8`` and ``j_constraint_adherence < 4``;
denominator = scored runs with a final probe. Wilson 95% CI. A final probe
counts as truncated when its ``token_count >= 1020`` (the April probe cap
rule); no waiver is applied, the count is reported.

Run counts (X11): each follow-up's ``run_summary.json`` gives the completed
and failed runs. The planned count is briefs x conditions x models x
repetitions from the same file. The transcript files and parquet rows on
disk are counted as a cross-check. A second rigor pass whose summary
overwrote the main ``run_summary.json`` is described in the runs provenance
(``other_rigor_pass``), with counts taken from the transcript and score files.

Outputs (in ``out_dir``), each with a ``.provenance.json`` sibling:

``sensitivity_cells.csv``  one row per (section, model, condition, variant)
``sensitivity_runs.csv``   one row per follow-up: planned, completed, failed runs

Run: ``python -m drift_bench.analysis.camera_ready.sensitivity``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

from drift_bench.analysis.camera_ready.kbv_table import (
    ADHERENCE_THRESHOLD,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    GEMINI_PRO_PROBE_CAP,
    PROBE_ACCURACY_THRESHOLD,
    summarize_transcript,
    wilson_ci,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, stray_run_ids
from drift_bench.analysis.probes import analyze_probes_for_run, compute_final_probe_accuracy, load_transcript

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIGURE_PROVENANCE = REPO_ROOT / "figures" / "fig_sensitivity.provenance.json"

GPT = "openai/gpt-5.4"
SONNET = "anthropic/claude-sonnet-4-6"
FLASH_LITE = "gemini/gemini-3.1-flash-lite-preview"

MTN = "multi_turn_neutral"
MTP = "multi_turn_pressure"
CKP = "checkpointed_pressure"
MTP_RIGOR = "multi_turn_pressure_rigor"

SECTION_TEMPERATURE = "temperature"
SECTION_PRESSURE_TYPE = "pressure_type"
VARIANT_APRIL = "april"
VARIANT_FOLLOWUP = "followup"
JUDGE_ARM = "published"

KBV_RULE = (
    f"final-probe overall_accuracy >= {PROBE_ACCURACY_THRESHOLD} (knows) and "
    f"j_constraint_adherence < {ADHERENCE_THRESHOLD} (violates); denominator = scored runs with a final probe"
)
CI_RULE = "Wilson score interval, z = 1.959964"
PROBE_SCORER = (
    "drift_bench.analysis.probes.analyze_probes_for_run (keyword scorer) on every restatement probe, "
    "then compute_final_probe_accuracy for the last probe per run; the same path as analysis/final_probe_accuracy.csv"
)
TRUNCATION_RULE = f"final probe assistant message token_count >= {GEMINI_PRO_PROBE_CAP}; counted, never waived"

# Section design. ``april_condition`` maps a follow-up condition to the April cell it is compared with.
SECTIONS: dict[str, dict] = {
    SECTION_TEMPERATURE: {
        "followup_dir": "followup_a",
        "models": [FLASH_LITE],
        "conditions": [MTN, MTP, CKP],
        "april_condition": {MTN: MTN, MTP: MTP, CKP: CKP},
        "setting": {VARIANT_APRIL: "temperature 1.0", VARIANT_FOLLOWUP: "temperature 0.7"},
    },
    SECTION_PRESSURE_TYPE: {
        "followup_dir": "followup_b",
        "models": [GPT, SONNET, FLASH_LITE],
        "conditions": [MTP_RIGOR],
        "april_condition": {MTP_RIGOR: MTP},
        "setting": {VARIANT_APRIL: "novelty pressure", VARIANT_FOLLOWUP: "rigor pressure"},
    },
}

COLUMNS = [
    "section", "model_id", "condition", "variant", "setting", "source", "judge_arm", "judge_model",
    "n_runs", "n_briefs", "n_with_final_probe", "knows", "violates", "kbv_count", "kbv_rate",
    "ci_low", "ci_high", "n_truncated_probes", "stray_rows_dropped",
]

RUNS_CSV = "sensitivity_runs.csv"
RUNS_COLUMNS = [
    "followup", "section", "setting", "planned", "completed", "failed", "unaccounted", "briefs", "n_conditions",
    "n_models", "repetitions", "conditions", "models", "transcripts_on_disk", "parquet_rows",
]
RUNS_RULES = {
    "planned": "briefs x len(conditions) x len(models) x repetitions, all four read from run_summary.json",
    "completed": "run_summary.json total_runs (pipeline.py writes the number of runs that returned a transcript)",
    "failed": "run_summary.json failed_runs (pipeline.py writes dispatched runs minus completed runs)",
    "unaccounted": "planned - completed - failed; 0 means the summary accounts for every planned run",
    "transcripts_on_disk": "number of <followup>/transcripts/*.jsonl files (cross-check of completed)",
    "parquet_rows": "rows of <followup>/aggregated/all_scores.parquet (scored runs)",
}


MAIN_RIGOR_RULES = {
    "completed": "main run_summary.json total_runs; pipeline.py counts transcripts already on disk as completed",
    "rigor_completed": "files transcripts/*_multi_turn_pressure_rigor_*.jsonl in the main data root",
    "rigor_failed": "briefs x len(models) x repetitions (main run_summary.json) - rigor_completed",
    "reused_existing_transcripts": "completed - rigor_completed",
    "single_shot_cells_on_disk": "distinct (brief, model, repetition) single-shot cells in the main transcripts dir for the summary's models and repetitions < repetitions",
    "n_rigor_scored": "rigor run ids with a scores/judge_<run_id>.jsonl in the main data root",
    "n_rigor_skeptical_scored": "rigor run ids with a scores_skeptical/skeptical_<run_id>.jsonl in the main data root",
    "overlap_with_followup_b": "rigor run ids that also appear as metadata.run_id in followup_b/transcripts/*.jsonl",
    "followup_b_rigor_cells": "distinct (brief, model, repetition) cells of followup_b transcripts with condition multi_turn_pressure_rigor",
    "followup_b_rigor_cells_rerun": "cells of the pass's rigor transcripts that are also followup_b rigor cells",
    "cells_not_in_followup_b": "cells of the pass's rigor transcripts that followup_b lacks",
    "start_gap_seconds": "earliest followup_b transcript metadata.start_time minus earliest start_time of the pass's rigor transcripts",
    "start_windows_overlap": "the [first, last] start_time ranges of the two sets of transcripts intersect",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_briefs(briefs_dir: Path) -> dict[str, dict]:
    """Every brief under ``briefs_dir`` keyed by its ``id``."""
    briefs: dict[str, dict] = {}
    for path in sorted(Path(briefs_dir).glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            brief = yaml.safe_load(fh)
        if isinstance(brief, dict) and brief.get("id"):
            briefs[brief["id"]] = brief
    return briefs


def final_probes(transcripts_dir: Path, briefs: dict[str, dict], run_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Final-probe accuracy and token count for the runs in ``run_ids``.

    Reads every transcript under ``transcripts_dir`` whose metadata run_id is
    wanted. Returns ``run_id, final_probe_accuracy, final_probe_turn,
    final_probe_token_count`` (one row per run with at least one probe) and a
    count block.
    """
    records: list[dict] = []
    tokens: dict[str, int | None] = {}
    n_matched = 0
    n_unreadable = 0
    missing_briefs: set[str] = set()
    for path in sorted(Path(transcripts_dir).glob("*.jsonl")):
        try:
            transcript = load_transcript(path)
        except Exception:  # noqa: BLE001  unreadable file: counted, skipped
            n_unreadable += 1
            continue
        meta = transcript.get("metadata", {})
        run_id = meta.get("run_id")
        if run_id not in run_ids:
            continue
        n_matched += 1
        brief = briefs.get(meta.get("brief_id"))
        if brief is None:
            missing_briefs.add(str(meta.get("brief_id")))
            continue
        records.extend(analyze_probes_for_run(transcript, brief))
        tokens[run_id] = summarize_transcript(transcript)["final_probe_token_count"]
    probe_df = pd.DataFrame(records)
    final = compute_final_probe_accuracy(probe_df) if not probe_df.empty else pd.DataFrame(
        columns=["run_id", "turn_index", "overall_accuracy"])
    out = final[["run_id", "turn_index", "overall_accuracy"]].rename(
        columns={"turn_index": "final_probe_turn", "overall_accuracy": "final_probe_accuracy"}).reset_index(drop=True)
    out["final_probe_token_count"] = out["run_id"].map(tokens)
    info = {
        "transcripts_dir": str(transcripts_dir),
        "runs_wanted": int(len(run_ids)),
        "transcripts_matched": n_matched,
        "transcripts_unreadable": n_unreadable,
        "runs_with_brief_missing": sorted(missing_briefs),
        "runs_with_final_probe": int(len(out)),
        "probe_scorer": PROBE_SCORER,
    }
    return out, info


# ---------------------------------------------------------------------------
# KBV
# ---------------------------------------------------------------------------


def kbv_flags(scores: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """Attach ``has_final_probe, knows, violates, kbv, probe_truncated`` to every scored row.

    Rows without a final probe keep ``knows`` and ``kbv`` as NA and stay out
    of the KBV denominator.
    """
    out = scores.merge(probes, on="run_id", how="left")
    out["has_final_probe"] = out["final_probe_accuracy"].notna()
    knows = out["final_probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD
    out["violates"] = out["j_constraint_adherence"] < ADHERENCE_THRESHOLD
    out["knows"] = knows.where(out["has_final_probe"])
    out["kbv"] = (knows & out["violates"]).where(out["has_final_probe"])
    tok = pd.to_numeric(out.get("final_probe_token_count"), errors="coerce")
    out["probe_truncated"] = out["has_final_probe"] & (tok >= GEMINI_PRO_PROBE_CAP)
    return out


def cell_row(flags: pd.DataFrame, section: str, model_id: str, condition: str, variant: str, setting: str,
             source: str, stray_rows_dropped: int) -> dict:
    """One CSV row from the flagged runs of a single cell."""
    with_probe = flags[flags["has_final_probe"]]
    n = int(len(with_probe))
    k = int(with_probe["kbv"].astype(bool).sum())
    lo, hi = wilson_ci(k, n)
    judges = sorted(set(flags["judge_model"].dropna())) if "judge_model" in flags.columns else []
    return {
        "section": section,
        "model_id": model_id,
        "condition": condition,
        "variant": variant,
        "setting": setting,
        "source": source,
        "judge_arm": JUDGE_ARM,
        "judge_model": ";".join(judges),
        "n_runs": int(len(flags)),
        "n_briefs": int(flags["brief_id"].nunique()),
        "n_with_final_probe": n,
        "knows": int(with_probe["knows"].astype(bool).sum()),
        "violates": int(flags["violates"].astype(bool).sum()),
        "kbv_count": k,
        "kbv_rate": (k / n) if n else math.nan,
        "ci_low": lo,
        "ci_high": hi,
        "n_truncated_probes": int(with_probe["probe_truncated"].astype(bool).sum()),
        "stray_rows_dropped": int(stray_rows_dropped),
    }


# ---------------------------------------------------------------------------
# Follow-up cells
# ---------------------------------------------------------------------------


def followup_cells(followup_root: Path, briefs: dict[str, dict], section: str,
                   main_stray: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """KBV cells of one follow-up experiment (``SECTIONS[section]``)."""
    spec = SECTIONS[section]
    followup_root = Path(followup_root)
    scores = pd.read_parquet(followup_root / "aggregated" / "all_scores.parquet")
    rows_in = int(len(scores))
    stray_local = stray_run_ids(followup_root, subdirs=("scores",))
    stray_main_hits = sorted(set(main_stray or set()) & set(scores["run_id"]))
    stray = stray_local | set(stray_main_hits)
    removed = scores[scores["run_id"].isin(stray)]
    scores = scores[~scores["run_id"].isin(stray)]
    rows_after_stray = int(len(scores))
    scores = scores[scores["condition"].isin(spec["conditions"]) & scores["model_id"].isin(spec["models"])]
    rows_in_cells = int(len(scores))
    scores = scores.dropna(subset=["j_constraint_adherence"])
    rows_scored = int(len(scores))

    probes, probe_info = final_probes(followup_root / "transcripts", briefs, set(scores["run_id"]))
    flags = kbv_flags(scores, probes)

    rows = []
    for model in spec["models"]:
        for cond in spec["conditions"]:
            cell = flags[(flags["model_id"] == model) & (flags["condition"] == cond)]
            if cell.empty:
                continue
            dropped = int(((removed["model_id"] == model) & (removed["condition"] == cond)).sum())
            rows.append(cell_row(cell, section, model, cond, VARIANT_FOLLOWUP, spec["setting"][VARIANT_FOLLOWUP],
                                 followup_root.name, dropped))
    cells = pd.DataFrame(rows, columns=COLUMNS)
    info = {
        "parquet": f"{followup_root.name}/aggregated/all_scores.parquet",
        "rows_in": rows_in,
        "filters": [
            {"name": "drop_stray_sonnet_runs", "rule": STRAY_RULE, "scope": f"{followup_root.name}/scores/judge_*.jsonl plus the main data root's stray ids",
             "stray_local_ids": sorted(stray_local), "stray_main_ids_matched": stray_main_hits, "rows_out": rows_after_stray},
            {"name": "cells", "rule": f"model_id in {spec['models']} and condition in {spec['conditions']}", "rows_out": rows_in_cells},
            {"name": "judge_score_present", "rule": "j_constraint_adherence is not null", "rows_out": rows_scored},
            {"name": "final_probe_present", "rule": "run has at least one restatement probe in its transcript (KBV denominator)",
             "rows_out": int(flags["has_final_probe"].sum())},
        ],
        "probes": probe_info,
        "truncation": TRUNCATION_RULE,
        "n_truncated_probes_total": int(flags["probe_truncated"].sum()),
    }
    return cells, info


# ---------------------------------------------------------------------------
# April cells
# ---------------------------------------------------------------------------


def april_cells(data_root: Path, analysis_dir: Path, section: str, briefs: dict[str, dict],
                stray: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """April comparator cells from ``kbv_published_arm.csv`` (clean), with a recomputation check.

    ``judge_model``, ``n_briefs``, ``stray_rows_dropped`` and
    ``n_truncated_probes`` come from the main parquet and transcripts. The KBV
    numbers come from the CSV of record; the recomputed values are compared
    and the result is reported, never substituted.
    """
    spec = SECTIONS[section]
    data_root, analysis_dir = Path(data_root), Path(analysis_dir)
    pub = pd.read_csv(analysis_dir / "kbv_published_arm.csv")
    clean = pub[pub["sample"] == "clean"].set_index(["model_id", "condition"])
    parquet = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
    if stray is None:
        stray = stray_run_ids(data_root) if (data_root / "scores").exists() else set()
    april_conditions = sorted(set(spec["april_condition"].values()))
    sub = parquet[parquet["model_id"].isin(spec["models"]) & parquet["condition"].isin(april_conditions)]
    is_stray = sub["run_id"].isin(stray)
    kept = sub[~is_stray]
    fpa_path = data_root / "analysis" / "final_probe_accuracy.csv"
    fpa = pd.read_csv(fpa_path)[["run_id", "turn_index", "overall_accuracy"]].rename(
        columns={"turn_index": "final_probe_turn", "overall_accuracy": "final_probe_accuracy"})
    tokens, tok_info = _april_probe_tokens(data_root / "transcripts", set(kept["run_id"]))
    probes = fpa.merge(tokens, on="run_id", how="left")
    flags = kbv_flags(kept, probes)

    rows = []
    checks = []
    for model in spec["models"]:
        for cond in spec["conditions"]:
            acond = spec["april_condition"][cond]
            if (model, acond) not in clean.index:
                warnings.warn(f"kbv_published_arm.csv has no clean row for {model} / {acond}", stacklevel=2)
                continue
            rec = clean.loc[(model, acond)]
            cell = flags[(flags["model_id"] == model) & (flags["condition"] == acond)]
            dropped = int((is_stray & (sub["model_id"] == model) & (sub["condition"] == acond)).sum())
            recomputed = cell_row(cell, section, model, acond, VARIANT_APRIL, spec["setting"][VARIANT_APRIL], "april_main", dropped)
            # The CSV of record supplies the published numbers; the recomputation is a check.
            row = dict(recomputed)
            row["n_runs"] = int(rec["n_scored"])
            for col in ("n_with_final_probe", "knows", "violates", "kbv_count"):
                row[col] = int(rec[col])
            for col in ("kbv_rate", "ci_low", "ci_high"):
                row[col] = float(rec[col])
            of_record = {"n_runs": int(rec["n_scored"])} | {c: int(rec[c]) for c in ("n_with_final_probe", "knows", "violates", "kbv_count")}
            redone = {c: int(recomputed[c]) for c in of_record}
            checks.append({"model_id": model, "condition": acond, "csv_of_record": of_record, "recomputed": redone,
                           "agree": of_record == redone})
            agree = of_record == redone
            if not agree:
                warnings.warn(f"April cell {model} / {acond}: kbv_published_arm.csv and the recomputation differ: {checks[-1]}", stacklevel=2)
            rows.append(row)
    cells = pd.DataFrame(rows, columns=COLUMNS)
    info = {
        "csv_of_record": "camera_ready/analysis/kbv_published_arm.csv (sample == clean)",
        "recomputed_from": ["aggregated/all_scores.parquet", "analysis/final_probe_accuracy.csv", "scores/judge_*.jsonl (stray rule)"],
        "stray_rule": STRAY_RULE,
        "rows_in_cells": int(len(sub)),
        "stray_rows_dropped": int(is_stray.sum()),
        "rows_after_stray": int(len(kept)),
        "probe_tokens": tok_info,
        "truncation": TRUNCATION_RULE,
        "checks": checks,
        "all_checks_agree": bool(all(c["agree"] for c in checks)),
    }
    return cells, info


def _april_probe_tokens(transcripts_dir: Path, run_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Final-probe token count of the April runs in ``run_ids`` (file name starts with the run id)."""
    rows = []
    n_read = 0
    for path in sorted(Path(transcripts_dir).glob("*.jsonl")):
        run_id = path.name.split("_", 1)[0]
        if run_id not in run_ids:
            continue
        try:
            transcript = load_transcript(path)
        except Exception:  # noqa: BLE001
            continue
        n_read += 1
        s = summarize_transcript(transcript)
        rows.append({"run_id": s["run_id"], "final_probe_token_count": s["final_probe_token_count"]})
    df = pd.DataFrame(rows, columns=["run_id", "final_probe_token_count"]).drop_duplicates("run_id")
    return df, {"transcripts_dir": str(transcripts_dir), "runs_wanted": int(len(run_ids)), "transcripts_read": n_read}


# ---------------------------------------------------------------------------
# Follow-up run counts (X11)
# ---------------------------------------------------------------------------


def planned_runs(summary: dict) -> int:
    """briefs x conditions x models x repetitions of a ``run_summary.json``."""
    return int(summary["briefs"]) * len(summary["conditions"]) * len(summary["models"]) * int(summary["repetitions"])


def run_count_row(followup_root: Path, section: str) -> tuple[dict, dict]:
    """One ``sensitivity_runs.csv`` row from ``<followup_root>/run_summary.json``, plus an info dict.

    Warns when the summary leaves runs unaccounted for, or when the transcript
    count on disk differs from the completed count.
    """
    followup_root = Path(followup_root)
    path = followup_root / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; the follow-up run counts come from it")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    planned = planned_runs(summary)
    completed = int(summary["total_runs"])
    failed = int(summary["failed_runs"])
    transcripts = len(list((followup_root / "transcripts").glob("*.jsonl")))
    parquet_path = followup_root / "aggregated" / "all_scores.parquet"
    parquet = pd.read_parquet(parquet_path) if parquet_path.exists() else None
    row = {
        "followup": followup_root.name,
        "section": section,
        "setting": SECTIONS[section]["setting"][VARIANT_FOLLOWUP],
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "unaccounted": planned - completed - failed,
        "briefs": int(summary["briefs"]),
        "n_conditions": len(summary["conditions"]),
        "n_models": len(summary["models"]),
        "repetitions": int(summary["repetitions"]),
        "conditions": ";".join(summary["conditions"]),
        "models": ";".join(summary["models"]),
        "transcripts_on_disk": transcripts,
        "parquet_rows": int(len(parquet)) if parquet is not None else math.nan,
    }
    if row["unaccounted"]:
        warnings.warn(f"{path}: planned {planned} != completed {completed} + failed {failed}", stacklevel=2)
    if transcripts != completed:
        warnings.warn(f"{followup_root.name}: {transcripts} transcripts on disk, run_summary.json says {completed} completed",
                      stacklevel=2)
    per_cell = []
    if parquet is not None:
        for (model, cond), n in parquet.groupby(["model_id", "condition"]).size().items():
            per_cell.append({"model_id": model, "condition": cond, "parquet_rows": int(n),
                             "planned": int(summary["briefs"]) * int(summary["repetitions"])})
    info = {
        "run_summary": f"{followup_root.name}/run_summary.json",
        "summary_fields_read": {k: summary.get(k) for k in ("total_runs", "failed_runs", "briefs", "conditions", "models",
                                                             "repetitions")},
        "completed_plus_failed_equals_planned": row["unaccounted"] == 0,
        "transcripts_equal_completed": transcripts == completed,
        "parquet_rows_by_cell": per_cell,
    }
    return row, info


def _transcript_metadata(path: Path) -> dict:
    """``metadata`` of a one-object transcript file."""
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.readline()).get("metadata") or {}


def _utc(ts: float | None) -> str | None:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat(timespec="seconds") if ts is not None else None


def main_rigor_pass(data_root: Path) -> dict | None:
    """The rigor-pressure pass whose summary overwrote the main ``run_summary.json``, if there is one.

    Its transcripts sit in the main transcript directory and are not part of
    ``followup_b``. Reported in the provenance for context only.

    ``completed`` in that summary is misleading. ``pipeline.py`` counts a cell
    whose transcript is already on disk as completed and does not re-run it.
    So the counts below come from the transcript and score files instead.
    """
    data_root = Path(data_root)
    path = data_root / "run_summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    if MTP_RIGOR not in (summary.get("conditions") or []):
        return None
    has_design = all(k in summary for k in ("briefs", "conditions", "models", "repetitions"))
    completed = summary.get("total_runs")

    # The pass's own work: rigor transcripts in the main transcript directory.
    rigor_meta = [_transcript_metadata(p) for p in sorted((data_root / "transcripts").glob(f"*_{MTP_RIGOR}_*.jsonl"))]
    rigor_completed = len(rigor_meta)
    rigor_planned = int(summary["briefs"]) * len(summary["models"]) * int(summary["repetitions"]) if has_design else None
    rigor_ids = {m.get("run_id") for m in rigor_meta}
    n_scored = sum((data_root / "scores" / f"judge_{rid}.jsonl").exists() for rid in rigor_ids)
    # The skeptical-prompt judge (GPT-5.4) did score some of these runs; the main judge did not.
    n_skeptical = sum((data_root / "scores_skeptical" / f"skeptical_{rid}.jsonl").exists() for rid in rigor_ids)

    # Single-shot cells of the pass's models and repetitions already on disk (reused, not re-run).
    ss_cells = set()
    if has_design:
        reps = set(range(int(summary["repetitions"])))
        for p in (data_root / "transcripts").glob("*_single_shot_*.jsonl"):
            m = _transcript_metadata(p)
            if m.get("condition") == "single_shot" and m.get("model_id") in summary["models"] and m.get("repetition") in reps:
                ss_cells.add((m.get("brief_id"), m.get("model_id"), m.get("repetition")))

    # Compare with followup_b: run ids, rigor cells, start times.
    fb_dir = SECTIONS[SECTION_PRESSURE_TYPE]["followup_dir"]
    fb_meta = [_transcript_metadata(p) for p in sorted((data_root / fb_dir / "transcripts").glob("*.jsonl"))]

    def cell(m: dict) -> tuple:
        return (m.get("brief_id"), m.get("model_id"), m.get("repetition"))

    pass_cells = {cell(m) for m in rigor_meta}
    fb_cells = {cell(m) for m in fb_meta if m.get("condition") == MTP_RIGOR}
    overlap = len(rigor_ids & {m.get("run_id") for m in fb_meta})
    pass_starts = [m["start_time"] for m in rigor_meta if m.get("start_time") is not None]
    fb_starts = [m["start_time"] for m in fb_meta if m.get("start_time") is not None]
    timed = bool(pass_starts and fb_starts)
    gap = round(min(fb_starts) - min(pass_starts), 1) if timed else None
    windows_overlap = (min(pass_starts) <= max(fb_starts) and min(fb_starts) <= max(pass_starts)) if timed else None

    rigor_failed = rigor_planned - rigor_completed if rigor_planned is not None else None
    reused = completed - rigor_completed if completed is not None else None
    n_rerun, n_extra = len(pass_cells & fb_cells), len(pass_cells - fb_cells)
    timing = (f"the first transcripts of the two started {abs(gap):.0f} s apart and their start windows "
              f"{'overlap' if windows_overlap else 'do not overlap'}" if timed else "start times are missing")
    note = (
        f"A separate rigor-pressure pass launched from the main data root; its run_summary.json overwrote the main one. "
        f"`completed` ({completed}) counts reused transcripts: pipeline.py counts a cell whose transcript is already on disk "
        f"as completed and does not re-run it. {reused} of the {completed} are such reused transcripts (the main transcript "
        f"directory holds {len(ss_cells)} single-shot cells of the pass's models and repetitions); {rigor_completed} are new "
        f"rigor transcripts, and {rigor_failed} of the {rigor_planned} planned rigor runs failed. "
        f"The pass ran at the same time as {fb_dir}: {timing}. "
        f"It re-ran {n_rerun} of {fb_dir}'s {len(fb_cells)} rigor cells plus {n_extra} cells {fb_dir} lacks; "
        f"{overlap} of its run ids appear in {fb_dir}. "
        f"It has no main-judge scores: {n_scored} of its {rigor_completed} rigor runs have a scores/judge_<run_id>.jsonl; "
        f"{n_skeptical} have a skeptical-prompt judge file (scores_skeptical/skeptical_<run_id>.jsonl). "
        f"Its runs are not in {fb_dir} and are not counted in sensitivity_runs.csv."
    )
    return {
        "run_summary": "run_summary.json (main data root)",
        "planned": planned_runs(summary) if has_design else None,
        "completed": completed,
        "failed": summary.get("failed_runs"),
        "conditions": summary.get("conditions"),
        "rigor_transcripts_in_main_transcripts_dir": rigor_completed,
        "rigor_completed": rigor_completed,
        "rigor_failed": rigor_failed,
        "reused_existing_transcripts": reused,
        "single_shot_cells_on_disk": len(ss_cells),
        "n_rigor_scored": int(n_scored),
        "n_rigor_skeptical_scored": int(n_skeptical),
        "overlap_with_followup_b": overlap,
        "followup_b_rigor_cells": len(fb_cells),
        "followup_b_rigor_cells_rerun": n_rerun,
        "cells_not_in_followup_b": n_extra,
        "rigor_first_start_utc": _utc(min(pass_starts)) if pass_starts else None,
        "followup_b_first_start_utc": _utc(min(fb_starts)) if fb_starts else None,
        "start_gap_seconds": gap,
        "start_windows_overlap": windows_overlap,
        "rules": MAIN_RIGOR_RULES,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Check against the figure's provenance sidecar
# ---------------------------------------------------------------------------


def check_against_figure(cells: pd.DataFrame, provenance_path: Path) -> dict:
    """Compare every cell with ``numbers_drawn`` of ``fig_sensitivity.provenance.json``."""
    provenance_path = Path(provenance_path)
    shown = str(provenance_path.relative_to(REPO_ROOT)) if provenance_path.is_relative_to(REPO_ROOT) else str(provenance_path)
    if not provenance_path.exists():
        return {"path": shown, "checked": False, "reason": "file not found"}
    with open(provenance_path, encoding="utf-8") as fh:
        drawn = json.load(fh).get("numbers_drawn", [])
    # The series label names the variant. Old labels: "April, ...", "Follow-up A, ...", "... (follow-up B)".
    # New labels: "Main benchmark, ...", "Temperature follow-up, ...", "... (rigor follow-up)".
    # Both sets mark the follow-up series with "follow-up"; every other series is the April run.
    index = {(d["model_id"], d["condition"], "a_temperature" if d["panel"] == "a_temperature" else "b_pressure_type",
              VARIANT_FOLLOWUP if "follow-up" in d["series"].lower() else "april"): d for d in drawn}
    mismatches = []
    n_compared = 0
    for r in cells.to_dict("records"):
        panel = "a_temperature" if r["section"] == SECTION_TEMPERATURE else "b_pressure_type"
        d = index.get((r["model_id"], r["condition"], panel, r["variant"]))
        if d is None:
            mismatches.append({"cell": (r["section"], r["model_id"], r["condition"], r["variant"]), "reason": "not drawn"})
            continue
        n_compared += 1
        if int(d["n"]) != int(r["n_with_final_probe"]) or int(d["kbv_count"]) != int(r["kbv_count"]):
            mismatches.append({"cell": (r["section"], r["model_id"], r["condition"], r["variant"]),
                               "figure": {"n": d["n"], "kbv_count": d["kbv_count"]},
                               "csv": {"n": int(r["n_with_final_probe"]), "kbv_count": int(r["kbv_count"])}})
    return {"path": shown, "checked": True, "cells_compared": n_compared, "mismatches": mismatches,
            "agree": not mismatches}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run(
    data_root: Path = DEFAULT_DATA_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    briefs_dir: Path | None = None,
    analysis_dir: Path | None = None,
    figure_provenance: Path | None = DEFAULT_FIGURE_PROVENANCE,
) -> dict[str, pd.DataFrame]:
    """Write ``sensitivity_cells.csv`` and ``sensitivity_runs.csv``.

    ``analysis_dir`` (default ``out_dir``) holds ``kbv_published_arm.csv``.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else out_dir
    briefs_dir = Path(briefs_dir) if briefs_dir is not None else data_root.parent / "briefs"
    briefs = load_briefs(briefs_dir)
    main_stray = stray_run_ids(data_root) if (data_root / "scores").exists() else set()

    frames: list[pd.DataFrame] = []
    section_info: dict[str, dict] = {}
    for section, spec in SECTIONS.items():
        april, a_info = april_cells(data_root, analysis_dir, section, briefs, main_stray)
        follow, f_info = followup_cells(data_root / spec["followup_dir"], briefs, section, main_stray)
        # Interleave: each April cell is followed by its follow-up cell.
        ordered = []
        for model in spec["models"]:
            for cond in spec["conditions"]:
                acond = spec["april_condition"][cond]
                ordered.append(april[(april["model_id"] == model) & (april["condition"] == acond)])
                ordered.append(follow[(follow["model_id"] == model) & (follow["condition"] == cond)])
        frames.append(pd.concat(ordered, ignore_index=True))
        section_info[section] = {"april": a_info, VARIANT_FOLLOWUP: f_info, "setting": spec["setting"]}
    cells = pd.concat(frames, ignore_index=True)[COLUMNS]
    figure_check = check_against_figure(cells, figure_provenance) if figure_provenance else {"checked": False}
    if figure_check.get("checked") and not figure_check.get("agree"):
        warnings.warn(f"sensitivity cells differ from {figure_provenance}: {figure_check['mismatches']}", stacklevel=2)

    provenance = {
        "inputs": [
            "camera_ready/analysis/kbv_published_arm.csv (April cells, sample == clean)",
            "aggregated/all_scores.parquet, analysis/final_probe_accuracy.csv, scores/judge_*.jsonl, transcripts/*.jsonl (April check, judge, stray count, probe token counts)",
            "followup_a/aggregated/all_scores.parquet, followup_a/transcripts/*.jsonl, followup_a/scores/judge_*.jsonl (temperature 0.7)",
            "followup_b/aggregated/all_scores.parquet, followup_b/transcripts/*.jsonl, followup_b/scores/judge_*.jsonl (rigor pressure)",
            f"{briefs_dir.name}/*.yaml (keyword probe scorer)",
        ],
        "sections": section_info,
        "judge_arm": "published: Opus 4.6 judges GPT-5.4 and Flash-Lite runs, GPT-5.4 judges Sonnet runs, in April and in the follow-ups alike; no common-judge arm exists for the follow-ups",
        "kbv": KBV_RULE,
        "ci": CI_RULE,
        "probe_scorer": PROBE_SCORER,
        "truncation": TRUNCATION_RULE,
        "columns": {
            "variant": "april (main run, temperature 1.0, novelty pressure) or followup (followup_a: temperature 0.7; followup_b: rigor pressure)",
            "n_runs": "scored rows in the cell after the stray rule",
            "n_with_final_probe": "runs with a final restatement probe (KBV denominator)",
            "stray_rows_dropped": "rows the stray rule removed from this cell",
        },
        "figure_provenance_check": figure_check,
        "rows_in": {s: {"april": section_info[s]["april"]["rows_in_cells"], "followup": section_info[s][VARIANT_FOLLOWUP]["rows_in"]} for s in SECTIONS},
        "date": dt.date.today().isoformat(),
    }
    write_csv_with_provenance(cells, out_dir / "sensitivity_cells.csv", provenance)

    run_rows, run_info = [], {}
    for section, spec in SECTIONS.items():
        row, info = run_count_row(data_root / spec["followup_dir"], section)
        run_rows.append(row)
        run_info[spec["followup_dir"]] = info
    runs = pd.DataFrame(run_rows, columns=RUNS_COLUMNS)
    write_csv_with_provenance(runs, out_dir / RUNS_CSV, {
        "inputs": [f"{spec['followup_dir']}/{name}" for spec in SECTIONS.values()
                   for name in ("run_summary.json", "transcripts/*.jsonl (count only)",
                                "aggregated/all_scores.parquet (row count only)")],
        "filters": RUNS_RULES,
        "rows_in": {"run_summaries_read": len(run_rows)},
        "followups": run_info,
        "other_rigor_pass": main_rigor_pass(data_root),
        "date": dt.date.today().isoformat(),
    })
    return {"sensitivity_cells": cells, "sensitivity_runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity cells (X7): KBV in the follow-up experiments. No API calls.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--briefs-dir", type=Path, default=None)
    parser.add_argument("--figure-provenance", type=Path, default=DEFAULT_FIGURE_PROVENANCE)
    args = parser.parse_args()
    results = run(args.data_root, args.out_dir, args.briefs_dir, figure_provenance=args.figure_provenance)
    cells = results["sensitivity_cells"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(cells[["section", "model_id", "condition", "variant", "n_with_final_probe", "kbv_count", "kbv_rate", "ci_low", "ci_high",
                     "n_truncated_probes", "stray_rows_dropped"]].to_string(index=False))
    print(f"-> {args.out_dir / 'sensitivity_cells.csv'} ({len(cells)} rows)")
    runs = results["sensitivity_runs"]
    print(runs[["followup", "planned", "completed", "failed", "unaccounted", "transcripts_on_disk", "parquet_rows"]]
          .to_string(index=False))
    print(f"-> {args.out_dir / RUNS_CSV} ({len(runs)} rows)")


if __name__ == "__main__":
    main()
