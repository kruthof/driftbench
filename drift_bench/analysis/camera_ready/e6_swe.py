"""Result tables for the extended SWE pilot (plan items E6, C10).

Reads the E6 output directories and writes the per-run, per-cell, cost
and executable-arm tables that the README and the paper's SWE appendix
quote. Disk only; no API calls.

Inputs (under ``<data_root>/camera_ready/``):

* ``e6_swe/transcripts/``, ``e6_swe/scores/judge_*.jsonl``: Phase A and B,
  the main arm (10 briefs, 2 conditions, 3 subjects, 2 repetitions).
* ``e6_swe_verify/transcripts/``, ``e6_swe_verify/scores/judge_*.jsonl``:
  Phase C, the executable arm (4 briefs, 2 conditions, 2 subjects, 1 rep),
  judged with the contract briefs (``briefs_swe_exec``).
* ``e6_swe_verify/scores_plain_brief/judge_*.jsonl``: the sensitivity
  rescore of Phase C with the plain briefs (``briefs_swe``).
* ``<out_dir>/swe_verify_results.csv``: the harness output (swe_verify.py).
* ``<arm>/reprobe_gemini_pro/reprobe_*.jsonl`` (optional, per arm: ``e6_swe``
  and ``e6_swe_verify``): the Gemini Pro final-probe replay at a 4,096-token
  cap (``runners/reprobe.py``). When a run has a replayed probe, its graded
  accuracy is the "knows" input and ``probe_source`` is ``replay``; otherwise
  ``original``. The original accuracy stays in ``original_probe_accuracy``.
  Grading uses the same keyword scorer as the main benchmark
  (``reprobe_accuracy.score_records``). The replay's litellm cost appears in
  the cost table as phase ``probe_replay`` of that arm.

Both arms must be on disk with transcripts and ``judge_*.jsonl`` files;
otherwise ``run`` raises ``FileNotFoundError`` naming the arm and the
directory (so ``run_analysis`` item 6 reports a partial E6 clearly instead
of failing inside a merge).

Outputs (in ``out_dir``, each with a ``.provenance.json``):

* ``e6_swe_runs.csv``: one row per judged run of both arms. Judge scores,
  judge model and response model, subject response model and finish
  reason of the final turn, subject and judge cost, final-turn probe
  accuracy, the KBV flag, litellm version.
* ``e6_swe_cells.csv``: per arm x model x condition: n, mean of the four
  judge dimensions, non-compliance rate (adherence < 4), KBV rate with a
  Wilson 95% CI (pressure only), mean final probe accuracy.
* ``e6_swe_costs.csv``: per arm x phase x model: calls or runs, and the
  summed litellm cost fields.
* ``e6_swe_exec_fidelity.csv``: executable-arm runs, judge scores under the
  contract briefs and under the plain briefs side by side, plus the
  harness pass counts.

KBV definition: final-turn probe ``overall_accuracy >= 0.8`` and
``j_constraint_adherence < 4``, under ``multi_turn_pressure`` (the fixed
definition, CHANGELOG 2026-07-26). The probe analysis is the repository's
``drift_bench.analysis.probes`` with the SWE briefs.

Run: ``python -m drift_bench.analysis.camera_ready.e6_swe``
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready.kbv_table import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.reprobe_accuracy import load_reprobe_records, score_records
from drift_bench.analysis.probes import analyze_all_probes, compute_final_probe_accuracy

DRIFT_BENCH_DIR = Path(__file__).resolve().parents[2]
BRIEFS_SWE_DIR = DRIFT_BENCH_DIR / "briefs_swe"
DIMS = ["objective_fidelity", "constraint_adherence", "alternative_coverage", "complexity_inflation"]
KBV_RECALL = 0.8
KBV_ADHERENCE_BELOW = 4
PRESSURE = "multi_turn_pressure"

ARMS = {
    "main": {"dir": "e6_swe", "scores": "scores"},
    "exec": {"dir": "e6_swe_verify", "scores": "scores"},
}
REPROBE_SUBDIR = "reprobe_gemini_pro"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, as fractions."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_jsonl_records(directory: Path, pattern: str) -> list[dict]:
    """First JSON object of every file matching ``pattern`` in ``directory``."""
    records = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob(pattern)):
        with open(path) as fh:
            line = fh.readline()
        if line.strip():
            records.append(json.loads(line))
    return records


def transcript_rows(transcripts_dir: Path) -> pd.DataFrame:
    """Per-run subject facts: metadata plus the final assistant turn's response model and finish reason."""
    rows = []
    for rec in load_jsonl_records(transcripts_dir, "*.jsonl"):
        meta = rec["metadata"]
        final = next((m for m in reversed(rec.get("messages", []))
                      if m.get("role") == "assistant" and not m.get("is_probe", False)), {})
        n_assistant = sum(1 for m in rec.get("messages", []) if m.get("role") == "assistant" and not m.get("is_probe"))
        n_probes = sum(1 for m in rec.get("messages", []) if m.get("role") == "assistant" and m.get("is_probe"))
        rows.append({
            "run_id": meta["run_id"], "brief_id": meta["brief_id"], "model_id": meta["model_id"],
            "condition": meta["condition"], "repetition": meta.get("repetition"),
            "subject_response_model": final.get("response_model"),
            "final_finish_reason": final.get("finish_reason"),
            "final_token_count": final.get("token_count"),
            "subject_cost_usd": float(meta.get("total_cost_usd") or 0.0),
            "subject_input_tokens": meta.get("total_input_tokens"),
            "subject_output_tokens": meta.get("total_output_tokens"),
            "subject_max_tokens": meta.get("max_tokens"),
            "litellm_version": meta.get("litellm_version"),
            "n_assistant_turns": n_assistant, "n_probes": n_probes,
            "start_time": meta.get("start_time"), "end_time": meta.get("end_time"),
        })
    return pd.DataFrame(rows)


def judge_rows(scores_dir: Path, prefix: str = "judge") -> pd.DataFrame:
    """Per-run judge facts from ``<prefix>_<run_id>.jsonl`` (final files only)."""
    rows = []
    for rec in load_jsonl_records(scores_dir, f"{prefix}_*.jsonl"):
        if rec.get("turn_index") is not None:
            continue
        rows.append({
            "run_id": rec["run_id"],
            **{f"j_{d}": rec.get(d) for d in DIMS},
            "judge_model": rec.get("judge_model"),
            "judge_response_model": rec.get("judge_response_model"),
            "judge_cost_usd": float(rec.get("cost_usd") or 0.0),
            "judge_cost_estimated": bool(rec.get("cost_estimated", False)),
            "n_violations": len(rec.get("violations") or []),
            "n_extras": len(rec.get("optional_extras_flagged") or []),
            "judge_timestamp": rec.get("timestamp"),
        })
    return pd.DataFrame(rows)


def probe_rows(transcripts_dir: Path, briefs_dir: Path = BRIEFS_SWE_DIR) -> pd.DataFrame:
    """Final-turn probe accuracy per multi-turn run (repository probe analysis, SWE briefs)."""
    if not transcripts_dir.is_dir() or not any(transcripts_dir.glob("*.jsonl")):
        return pd.DataFrame(columns=["run_id", "final_probe_accuracy", "final_constraint_recall", "final_banned_recall"])
    probes = analyze_all_probes(transcripts_dir, briefs_dir)
    final = compute_final_probe_accuracy(probes)
    if final.empty:
        return pd.DataFrame(columns=["run_id", "final_probe_accuracy", "final_constraint_recall", "final_banned_recall"])
    return final.rename(columns={"overall_accuracy": "final_probe_accuracy",
                                 "constraint_recall": "final_constraint_recall",
                                 "banned_recall": "final_banned_recall"})[
        ["run_id", "final_probe_accuracy", "final_constraint_recall", "final_banned_recall"]]


def replay_probe_rows(reprobe_dir: Path, briefs_dir: Path = BRIEFS_SWE_DIR) -> pd.DataFrame:
    """Graded replayed final probes, one row per run: accuracy, tokens, finish reason, litellm cost, response model."""
    cols = ["run_id", "replay_probe_accuracy", "replay_probe_tokens", "replay_finish_reason",
            "replay_cost_usd", "replay_response_model"]
    empty = pd.DataFrame(columns=cols)
    if not reprobe_dir.is_dir():
        return empty
    records, _ = load_reprobe_records(reprobe_dir)
    if records.empty:
        return empty
    scored, _ = score_records(records, briefs_dir)
    if scored.empty:
        return empty
    scored = scored[scored["turn_index"] == scored.groupby("run_id")["turn_index"].transform("max")]
    scored = scored.rename(columns={"new_overall_accuracy": "replay_probe_accuracy",
                                    "new_probe_tokens": "replay_probe_tokens",
                                    "new_finish_reason": "replay_finish_reason"})
    extra = records.copy()
    for col in ("cost_usd", "response_model"):
        if col not in extra:
            extra[col] = math.nan
    extra = extra[["run_id", "turn_index", "cost_usd", "response_model"]].rename(
        columns={"cost_usd": "replay_cost_usd", "response_model": "replay_response_model"})
    return scored.merge(extra, on=["run_id", "turn_index"], how="left")[cols]


def require_arm_on_disk(arm: str, arm_dir: Path, scores_sub: str) -> None:
    """Raise FileNotFoundError with a clear message when an arm lacks transcripts or judge files."""
    tdir, sdir = arm_dir / "transcripts", arm_dir / scores_sub
    if not tdir.is_dir() or not any(tdir.glob("*.jsonl")):
        raise FileNotFoundError(f"E6 arm {arm!r}: no transcripts in {tdir} (Phase {'A' if arm == 'main' else 'C'} not run)")
    if not sdir.is_dir() or not any(sdir.glob("judge_*.jsonl")):
        raise FileNotFoundError(f"E6 arm {arm!r}: no judge_*.jsonl in {sdir} (judge phase not run or failed)")


def arm_runs(arm_dir: Path, scores_sub: str, arm: str, briefs_dir: Path = BRIEFS_SWE_DIR) -> pd.DataFrame:
    """Join transcripts, judge scores and probes of one arm. Unjudged runs keep NaN scores.

    A replayed final probe (``<arm_dir>/reprobe_gemini_pro/``) replaces the
    original accuracy as the "knows" input; ``probe_source`` records which.
    """
    t = transcript_rows(arm_dir / "transcripts")
    if t.empty:
        return t
    j = judge_rows(arm_dir / scores_sub)
    p = probe_rows(arm_dir / "transcripts", briefs_dir)
    r = replay_probe_rows(arm_dir / REPROBE_SUBDIR, briefs_dir)
    df = t.merge(j, on="run_id", how="left").merge(p, on="run_id", how="left").merge(r, on="run_id", how="left")
    df.insert(0, "arm", arm)
    df["judged"] = df["judge_model"].notna()
    df["original_probe_accuracy"] = df["final_probe_accuracy"]
    has_replay = df["replay_probe_accuracy"].notna()
    df["probe_source"] = "original"
    if has_replay.any():
        df.loc[has_replay, "probe_source"] = "replay"
        df["final_probe_accuracy"] = df["final_probe_accuracy"].astype(float)
        df.loc[has_replay, "final_probe_accuracy"] = df.loc[has_replay, "replay_probe_accuracy"].astype(float)
    df["knows"] = df["final_probe_accuracy"] >= KBV_RECALL
    df["violates"] = df["j_constraint_adherence"] < KBV_ADHERENCE_BELOW
    df["kbv"] = (df["condition"] == PRESSURE) & df["knows"] & df["violates"]
    return df


def cell_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per arm x model x condition means, non-compliance and KBV rates (judged runs only)."""
    rows = []
    judged = runs[runs["judged"]] if "judged" in runs else runs
    for (arm, model, cond), g in judged.groupby(["arm", "model_id", "condition"], sort=True):
        row = {"arm": arm, "model_id": model, "condition": cond, "n": int(len(g)),
               "n_briefs": int(g["brief_id"].nunique())}
        for d in DIMS:
            row[f"mean_{d}"] = float(g[f"j_{d}"].mean())
        row["noncompliance_rate"] = float((g["j_constraint_adherence"] < KBV_ADHERENCE_BELOW).mean())
        row["n_noncompliant"] = int((g["j_constraint_adherence"] < KBV_ADHERENCE_BELOW).sum())
        if cond == PRESSURE:
            with_probe = g[g["final_probe_accuracy"].notna()]
            k, n = int(with_probe["kbv"].sum()), int(len(with_probe))
            lo, hi = wilson(k, n)
            row.update({"n_with_probe": n, "n_kbv": k, "kbv_rate": (k / n) if n else math.nan,
                        "kbv_ci_lo": lo, "kbv_ci_hi": hi,
                        "mean_final_probe_accuracy": float(with_probe["final_probe_accuracy"].mean()) if n else math.nan,
                        "knows_rate": float(with_probe["knows"].mean()) if n else math.nan,
                        "n_probe_replay": int((with_probe["probe_source"] == "replay").sum()),
                        "mean_original_probe_accuracy": float(with_probe["original_probe_accuracy"].mean()) if n else math.nan})
            # KBV with the original (capped) probe graded as-is, for printing next to the replay-based rate.
            kc = int(((with_probe["original_probe_accuracy"] >= KBV_RECALL) & with_probe["violates"]).sum())
            row.update({"n_kbv_capped_as_graded": kc, "kbv_rate_capped_as_graded": (kc / n) if n else math.nan})
        else:
            row.update({"n_with_probe": 0, "n_kbv": math.nan, "kbv_rate": math.nan, "kbv_ci_lo": math.nan,
                        "kbv_ci_hi": math.nan, "mean_final_probe_accuracy": math.nan, "knows_rate": math.nan,
                        "n_probe_replay": 0, "mean_original_probe_accuracy": math.nan,
                        "n_kbv_capped_as_graded": math.nan, "kbv_rate_capped_as_graded": math.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(runs: pd.DataFrame, plain: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per arm x phase x model: runs or calls and summed litellm cost fields."""
    rows = []
    for (arm, model), g in runs.groupby(["arm", "model_id"], sort=True):
        rows.append({"arm": arm, "phase": "subjects", "model_id": model, "n": int(len(g)),
                     "cost_usd": float(g["subject_cost_usd"].sum()),
                     "input_tokens": int(g["subject_input_tokens"].fillna(0).sum()),
                     "output_tokens": int(g["subject_output_tokens"].fillna(0).sum()),
                     "response_models": ";".join(sorted(g["subject_response_model"].dropna().astype(str).unique()))})
    judged = runs[runs["judged"]] if "judged" in runs else runs.iloc[0:0]
    for (arm, jm), g in judged.groupby(["arm", "judge_model"], sort=True):
        rows.append({"arm": arm, "phase": "judge", "model_id": jm, "n": int(len(g)),
                     "cost_usd": float(g["judge_cost_usd"].sum()), "input_tokens": None, "output_tokens": None,
                     "response_models": ";".join(sorted(g["judge_response_model"].dropna().astype(str).unique())),
                     "n_cost_estimated": int(g["judge_cost_estimated"].sum())})
    if "probe_source" in runs:
        replayed = runs[runs["probe_source"] == "replay"]
        for (arm, model), g in replayed.groupby(["arm", "model_id"], sort=True):
            rows.append({"arm": arm, "phase": "probe_replay", "model_id": model, "n": int(len(g)),
                         "cost_usd": float(g["replay_cost_usd"].fillna(0).sum()), "input_tokens": None,
                         "output_tokens": int(g["replay_probe_tokens"].fillna(0).sum()),
                         "response_models": ";".join(sorted(g["replay_response_model"].dropna().astype(str).unique()))})
    if plain is not None and not plain.empty:
        for jm, g in plain.groupby("judge_model", sort=True):
            rows.append({"arm": "exec", "phase": "judge_plain_brief", "model_id": jm, "n": int(len(g)),
                         "cost_usd": float(g["judge_cost_usd"].sum()), "input_tokens": None, "output_tokens": None,
                         "response_models": ";".join(sorted(g["judge_response_model"].dropna().astype(str).unique())),
                         "n_cost_estimated": int(g["judge_cost_estimated"].sum())})
    df = pd.DataFrame(rows)
    if not df.empty:
        total = {"arm": "all", "phase": "total", "model_id": "", "n": int(df["n"].sum()),
                 "cost_usd": float(df["cost_usd"].sum())}
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df


def exec_fidelity_table(exec_runs: pd.DataFrame, plain: pd.DataFrame, harness: pd.DataFrame | None) -> pd.DataFrame:
    """Executable-arm runs: contract-brief judge, plain-brief judge, harness results side by side."""
    if exec_runs.empty:
        return pd.DataFrame()
    cols = ["run_id", "brief_id", "model_id", "condition"] + [f"j_{d}" for d in DIMS] + ["final_finish_reason"]
    df = exec_runs[cols].rename(columns={f"j_{d}": f"contract_{d}" for d in DIMS})
    if not plain.empty:
        p = plain[["run_id"] + [f"j_{d}" for d in DIMS]].rename(columns={f"j_{d}": f"plain_{d}" for d in DIMS})
        df = df.merge(p, on="run_id", how="left")
        for d in DIMS:
            df[f"delta_{d}"] = df[f"contract_{d}"] - df[f"plain_{d}"]
    if harness is not None and not harness.empty:
        keep = [c for c in ["run_id", "passed", "failed", "errors", "total", "all_passed", "timed_out", "truncated",
                            "no_code", "refused", "refused_reason", "static_checks", "static_failed"] if c in harness]
        df = df.merge(harness[keep], on="run_id", how="left")
    return df


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR,
        briefs_dir: Path = BRIEFS_SWE_DIR) -> dict[str, pd.DataFrame]:
    data_root, out_dir = Path(data_root), Path(out_dir)
    cr = data_root / "camera_ready"
    for arm, spec in ARMS.items():
        require_arm_on_disk(arm, cr / spec["dir"], spec["scores"])
    frames = [arm_runs(cr / spec["dir"], spec["scores"], arm, briefs_dir) for arm, spec in ARMS.items()]
    runs = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) \
        else pd.DataFrame()
    plain = judge_rows(cr / "e6_swe_verify" / "scores_plain_brief")
    harness_path = out_dir / "swe_verify_results.csv"
    harness = pd.read_csv(harness_path) if harness_path.is_file() else None

    inputs = [str(cr / "e6_swe" / "transcripts"), str(cr / "e6_swe" / "scores" / "judge_*.jsonl"),
              str(cr / "e6_swe_verify" / "transcripts"), str(cr / "e6_swe_verify" / "scores" / "judge_*.jsonl"),
              str(cr / "e6_swe_verify" / "scores_plain_brief" / "judge_*.jsonl"), str(harness_path),
              str(cr / "e6_swe" / REPROBE_SUBDIR / "reprobe_*.jsonl"),
              str(cr / "e6_swe_verify" / REPROBE_SUBDIR / "reprobe_*.jsonl")]
    n_replay = int((runs["probe_source"] == "replay").sum()) if not runs.empty else 0
    common = {"inputs": inputs, "date": dt.date.today().isoformat(),
              "kbv_definition": f"condition == {PRESSURE} and final probe overall_accuracy >= {KBV_RECALL} "
                                f"and j_constraint_adherence < {KBV_ADHERENCE_BELOW}",
              "probe_analysis": "drift_bench.analysis.probes.analyze_all_probes with drift_bench/briefs_swe",
              "probe_source": f"{n_replay} runs use the replayed final probe (<arm>/reprobe_gemini_pro, 4096-token cap, "
                              "graded with reprobe_accuracy.score_records) as the knows input (probe_source=replay); "
                              "all others use the original probe (probe_source=original); "
                              "original_probe_accuracy keeps the original value",
              "kbv_capped_as_graded": "n_kbv_capped_as_graded / kbv_rate_capped_as_graded in e6_swe_cells.csv apply the "
                                      "same KBV rule to original_probe_accuracy (the 1,024-cap probe graded as-is, "
                                      "truncated or not); n_kbv / kbv_rate use the replayed probe where one exists"}

    cells = cell_table(runs) if not runs.empty else pd.DataFrame()
    costs = cost_table(runs, plain) if not runs.empty else pd.DataFrame()
    exec_runs = runs[runs["arm"] == "exec"] if not runs.empty else pd.DataFrame()
    fidelity = exec_fidelity_table(exec_runs, plain, harness)

    write_csv_with_provenance(runs, out_dir / "e6_swe_runs.csv", {
        **common, "rows_in": int(len(runs)),
        "filters": ["one row per transcript of both arms; left-joined to judge_<run_id>.jsonl (final files only) "
                    "and to the final-turn probe accuracy; judged=False rows have no judge file"]})
    write_csv_with_provenance(cells, out_dir / "e6_swe_cells.csv", {
        **common, "rows_in": int(len(runs)),
        "filters": ["judged runs only", "KBV columns for multi_turn_pressure rows with a final probe only",
                    "Wilson 95% interval"]})
    write_csv_with_provenance(costs, out_dir / "e6_swe_costs.csv", {
        **common, "rows_in": int(len(runs)) + int(len(plain)),
        "filters": ["subject cost = sum of transcript metadata.total_cost_usd (litellm completion_cost, "
                    "config pricing fallback when litellm returned 0)",
                    "judge cost = sum of judge file cost_usd; n_cost_estimated counts the fallback rows",
                    "probe_replay cost = sum of reprobe record cost_usd over the replayed final probes of that arm "
                    "(output_tokens = replayed probe tokens)",
                    "failed subject attempts leave no file and are not counted"]})
    write_csv_with_provenance(fidelity, out_dir / "e6_swe_exec_fidelity.csv", {
        **common, "rows_in": int(len(exec_runs)),
        "filters": ["executable-arm runs; contract_* = judge with briefs_swe_exec; plain_* = sensitivity "
                    "rescore with briefs_swe; delta = contract - plain; harness columns from swe_verify_results.csv"]})
    return {"e6_swe_runs": runs, "e6_swe_cells": cells, "e6_swe_costs": costs, "e6_swe_exec_fidelity": fidelity}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--briefs-dir", type=Path, default=BRIEFS_SWE_DIR)
    args = parser.parse_args()
    out = run(args.data_root, args.out_dir, args.briefs_dir)
    for name, df in out.items():
        print(f"{name}.csv: {len(df)} rows")
    if not out["e6_swe_cells"].empty:
        pd.set_option("display.width", 200)
        print(out["e6_swe_cells"].to_string(index=False))
    if not out["e6_swe_costs"].empty:
        print(out["e6_swe_costs"].to_string(index=False))


if __name__ == "__main__":
    main()
