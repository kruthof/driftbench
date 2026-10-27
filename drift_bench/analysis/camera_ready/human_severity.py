"""Human severity by condition (plan items C4 and B20).

Source: ``human_validation/human_scores.json`` (110 runs, three raters each,
``constraint_statuses`` holds the per-constraint median label). Condition and
model come from the parquet, joined on ``run_id``.

Aggregations, all reported side by side:

- ``median_label_pooled``: count the median labels over every constraint
  item in the condition, then divide by the number of items. This is the
  aggregation the camera-ready prints.
- ``median_label_run_mean``: compute the per-run fraction first, then
  average over runs.
- ``per_rater_pooled``: pool every rater's raw label (three per item).

Judge comparison: on the runs that also have a structured judge file, the
item-for-item share not satisfied by humans and by the judge, and the share
humans mark violated where the judge does not. Run-level sensitivity and
specificity of three judge definitions against the human "any violation" flag.

Outputs: ``human_severity_runs.csv``, ``human_severity_by_condition.csv``,
``human_vs_judge.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready.graded_severity import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    load_structured_checks,
    normalise_status,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids
from drift_bench.analysis.camera_ready.stretch_violate import CONDITIONS, load_judge_adherence

SCORE_ORDER = ["single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure", "all"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_human_scores(path: Path) -> pd.DataFrame:
    """One row per run from ``human_scores.json`` with normalised median labels."""
    with Path(path).open() as fh:
        data = json.load(fh)
    rows = []
    for run_id, rec in data.items():
        labels = [normalise_status(s) for s in rec["constraint_statuses"]]
        rows.append({
            "run_id": run_id,
            "brief_id": rec.get("brief_id"),
            "n_raters": rec.get("n_raters"),
            "rater_ids": ",".join(rec.get("rater_ids", [])),
            "median_labels": labels,
            "n_items": len(labels),
            "n_stretched": sum(s == "stretched" for s in labels),
            "n_violated": sum(s == "violated" for s in labels),
            "n_not_satisfied": sum(s != "satisfied" for s in labels),
            "file_violated_count": rec.get("violated_count"),
            "file_stretched_count": rec.get("stretched_count"),
            "computed_score": rec.get("computed_score"),
            "computed_score_from_constraint_median": rec.get("computed_score_from_constraint_median"),
        })
    df = pd.DataFrame(rows)
    df["any_violation"] = df["n_violated"] > 0
    return df


def load_rater_labels(raters_dir: Path) -> pd.DataFrame:
    """One row per (rater, run, item) from the raw rater exports. Empty if absent."""
    rows = []
    if Path(raters_dir).is_dir():
        for path in sorted(Path(raters_dir).glob("*.json")):
            with path.open() as fh:
                data = json.load(fh)
            for run_id, rec in data.items():
                for i, s in enumerate(rec["constraint_statuses"]):
                    rows.append({"rater_id": rec.get("rater_id"), "run_id": run_id, "item": i, "status": normalise_status(s)})
    return pd.DataFrame(rows, columns=["rater_id", "run_id", "item", "status"])


def load_judge_violations_flag(scores_dirs: list[Path]) -> pd.DataFrame:
    """Per run: whether the published judge file lists at least one entry in ``violations``."""
    rows = []
    for d in scores_dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("judge_*.jsonl")):
            with path.open() as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                continue
            rec = json.loads(lines[-1])
            if path.name != f"judge_{rec['run_id']}.jsonl":
                continue  # a longer prefix matched the glob
            rows.append({"run_id": rec["run_id"], "judge_violations_listed": bool(rec.get("violations"))})
    return pd.DataFrame(rows, columns=["run_id", "judge_violations_listed"]).drop_duplicates("run_id")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def build_runs(human: pd.DataFrame, scores: pd.DataFrame, checks: pd.DataFrame, rater_labels: pd.DataFrame,
               judge_flags: pd.DataFrame) -> pd.DataFrame:
    """Join humans to the parquet, the structured judge, the raw raters and the judge violations flag."""
    df = human.merge(scores[["run_id", "model_id", "condition", "judge_model", "j_constraint_adherence"]], on="run_id", how="left")
    if not checks.empty:
        per_run = checks.groupby("run_id").agg(
            structured_n_checks=("status", "size"),
            structured_n_not_satisfied=("status", lambda s: int((s != "satisfied").sum())),
            structured_n_violated=("status", lambda s: int((s == "violated").sum())),
            structured_judge_model=("judge_model", "first"),
        ).reset_index()
        df = df.merge(per_run, on="run_id", how="left")
    else:
        for c in ("structured_n_checks", "structured_n_not_satisfied", "structured_n_violated", "structured_judge_model"):
            df[c] = np.nan
    df["has_structured"] = df["structured_n_checks"].notna()
    if not rater_labels.empty:
        pooled = rater_labels.groupby("run_id").agg(
            rater_n_labels=("status", "size"),
            rater_n_not_satisfied=("status", lambda s: int((s != "satisfied").sum())),
            rater_n_violated=("status", lambda s: int((s == "violated").sum())),
        ).reset_index()
        df = df.merge(pooled, on="run_id", how="left")
    df = df.merge(judge_flags, on="run_id", how="left")
    df["judge_adherence_le_2"] = df["j_constraint_adherence"] <= 2
    df["structured_any_violation"] = df["structured_n_violated"] > 0
    return df


def _severity_row(n_runs: int, n_items: int, n_ns: int, n_v: int, n_s: int, n_runs_v: int) -> dict:
    return {
        "n_runs": int(n_runs), "n_items": int(n_items),
        "n_not_satisfied": int(n_ns), "share_not_satisfied": n_ns / n_items if n_items else np.nan,
        "n_violated": int(n_v), "share_violated": n_v / n_items if n_items else np.nan,
        "n_stretched": int(n_s), "share_stretched": n_s / n_items if n_items else np.nan,
        "violations_per_run": n_v / n_runs if n_runs else np.nan,
        "n_runs_with_violation": int(n_runs_v), "share_runs_with_violation": n_runs_v / n_runs if n_runs else np.nan,
    }


def by_condition_table(runs: pd.DataFrame, rater_labels: pd.DataFrame) -> pd.DataFrame:
    """Severity per condition (and all runs) under the three aggregations."""
    rows = []
    groups = [(c, runs[runs["condition"] == c]) for c in CONDITIONS if (runs["condition"] == c).any()]
    groups.append(("all", runs))
    for cond, g in groups:
        n_runs = len(g)
        base = {"condition": cond, "mean_computed_score": float(g["computed_score"].mean()),
                "mean_computed_score_from_constraint_median": float(g["computed_score_from_constraint_median"].mean()),
                "mean_judge_adherence": float(g["j_constraint_adherence"].mean())}
        # Pooled median labels.
        row = {"aggregation": "median_label_pooled", **base}
        row.update(_severity_row(n_runs, g["n_items"].sum(), g["n_not_satisfied"].sum(), g["n_violated"].sum(),
                                 g["n_stretched"].sum(), g["any_violation"].sum()))
        rows.append(row)
        # Mean of per-run fractions.
        row = {"aggregation": "median_label_run_mean", **base}
        row.update(_severity_row(n_runs, g["n_items"].sum(), g["n_not_satisfied"].sum(), g["n_violated"].sum(),
                                 g["n_stretched"].sum(), g["any_violation"].sum()))
        row["share_not_satisfied"] = float((g["n_not_satisfied"] / g["n_items"]).mean()) if n_runs else np.nan
        row["share_violated"] = float((g["n_violated"] / g["n_items"]).mean()) if n_runs else np.nan
        row["share_stretched"] = float((g["n_stretched"] / g["n_items"]).mean()) if n_runs else np.nan
        rows.append(row)
        # Every rater's raw label.
        if not rater_labels.empty:
            rl = rater_labels[rater_labels["run_id"].isin(g["run_id"])]
            rl = rl.assign(violated=rl["status"] == "violated")
            per_rater_run = rl.groupby(["rater_id", "run_id"])["violated"].any()
            row = {"aggregation": "per_rater_pooled", **base}
            row.update(_severity_row(n_runs, len(rl), int((rl["status"] != "satisfied").sum()), int(rl["violated"].sum()),
                                     int((rl["status"] == "stretched").sum()), 0))
            # Per-run quantities use rater-run pairs as the unit.
            n_pairs = int(len(per_rater_run))
            row["violations_per_run"] = int(rl["violated"].sum()) / n_pairs if n_pairs else np.nan
            row["n_runs_with_violation"] = int(per_rater_run.sum()) if n_pairs else 0
            row["share_runs_with_violation"] = float(per_rater_run.mean()) if n_pairs else np.nan
            row["n_rater_run_pairs"] = int(n_pairs)
            rows.append(row)
    return pd.DataFrame(rows)


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def load_brief_constraints(briefs_dir: Path | None) -> dict[str, list[str]]:
    """brief_id to its hard_constraints list. Empty when the directory is absent."""
    out: dict[str, list[str]] = {}
    if briefs_dir is None or not Path(briefs_dir).is_dir():
        return out
    import yaml
    for path in sorted(Path(briefs_dir).glob("*.yaml")):
        with path.open() as fh:
            b = yaml.safe_load(fh)
        if isinstance(b, dict) and "id" in b:
            out[b["id"]] = [str(c) for c in b.get("hard_constraints", [])]
    return out


def _norm_text(s) -> str:
    return " ".join(str(s or "").strip().lower().split())


def human_vs_judge_table(runs: pd.DataFrame, checks: pd.DataFrame, brief_constraints: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Long table of comparison metrics: metric, aggregation, value, numerator, denominator, n_runs, note.

    ``aggregation`` is ``pooled`` (counts over all items), ``run_mean`` (mean of
    per-run fractions), ``run_level`` (runs as units) or ``score``.
    """
    rows: list[dict] = []
    brief_constraints = brief_constraints or {}

    def add(metric: str, num, den, n_runs, note: str = "", value=None, aggregation: str = "pooled"):
        rows.append({"metric": metric, "aggregation": aggregation, "value": _rate(num, den) if value is None else value,
                     "numerator": num, "denominator": den, "n_runs": int(n_runs), "note": note})

    # Item-for-item on shared runs. Items align by position in the brief's hard-constraint list.
    shared = runs[runs["has_structured"]]
    pairs = []
    n_len_mismatch = 0
    n_text_checked = 0
    n_text_mismatch = 0
    text_mismatch_runs: list[str] = []
    has_text = "constraint" in checks.columns
    for _, r in shared.iterrows():
        jc = checks[checks["run_id"] == r["run_id"]].sort_values("constraint_index")
        jl = jc["status"].tolist()
        hl = r["median_labels"]
        if len(jl) != len(hl):
            n_len_mismatch += 1
            continue
        # Guard: the judge's constraint text should equal the brief's hard_constraints[i].
        bc = brief_constraints.get(r["brief_id"])
        if has_text and bc is not None:
            n_text_checked += 1
            jt = [_norm_text(t) for t in jc["constraint"].tolist()]
            if len(bc) != len(jt) or any(_norm_text(b) != t for b, t in zip(bc, jt)):
                n_text_mismatch += 1
                text_mismatch_runs.append(r["run_id"])
        for h, j in zip(hl, jl):
            pairs.append({"run_id": r["run_id"], "condition": r["condition"], "human": h, "judge": j})
    pairs_df = pd.DataFrame(pairs, columns=["run_id", "condition", "human", "judge"])
    n_items = len(pairs_df)
    n_shared = pairs_df["run_id"].nunique()
    guard = (f"constraint-text guard: {n_text_checked} runs checked against brief hard_constraints, {n_text_mismatch} with a text mismatch"
             + (f" ({', '.join(text_mismatch_runs)})" if text_mismatch_runs else "")) if brief_constraints else "constraint-text guard skipped (briefs not found)"
    add("shared_runs", n_shared, 1, n_shared, f"runs with both human labels and a structured judge file; {n_len_mismatch} runs dropped for item-count mismatch; {guard}", value=n_shared, aggregation="run_level")
    add("shared_items", n_items, 1, n_shared, "constraint items on the shared runs", value=n_items, aggregation="run_level")
    h_ns = int((pairs_df["human"] != "satisfied").sum())
    j_ns = int((pairs_df["judge"] != "satisfied").sum())
    add("human_share_not_satisfied_shared", h_ns, n_items, n_shared, "median label not satisfied, pooled over items")
    add("judge_share_not_satisfied_shared", j_ns, n_items, n_shared, "structured judge status not satisfied, pooled over items")
    add("human_over_judge_not_satisfied_ratio", h_ns, j_ns, n_shared, "ratio of the two shares")
    h_v = pairs_df["human"] == "violated"
    j_v = pairs_df["judge"] == "violated"
    add("human_share_violated_shared", int(h_v.sum()), n_items, n_shared, "median label violated")
    add("judge_share_violated_shared", int(j_v.sum()), n_items, n_shared, "structured judge violated")
    add("human_violated_judge_not_violated", int((h_v & ~j_v).sum()), n_items, n_shared, "item violated by humans, judge marks stretched or satisfied")
    add("human_violated_judge_satisfied", int((h_v & (pairs_df["judge"] == "satisfied")).sum()), n_items, n_shared, "item violated by humans, judge marks satisfied")
    add("human_not_satisfied_judge_satisfied", int(((pairs_df["human"] != "satisfied") & (pairs_df["judge"] == "satisfied")).sum()), n_items, n_shared, "item not satisfied by humans, judge marks satisfied")
    add("judge_not_satisfied_human_satisfied", int(((pairs_df["judge"] != "satisfied") & (pairs_df["human"] == "satisfied")).sum()), n_items, n_shared, "item not satisfied by judge, humans mark satisfied")
    add("item_label_agreement", int((pairs_df["human"] == pairs_df["judge"]).sum()), n_items, n_shared, "exact three-way label agreement")
    add("item_not_satisfied_agreement", int(((pairs_df["human"] != "satisfied") == (pairs_df["judge"] != "satisfied")).sum()), n_items, n_shared, "agreement on satisfied vs not")
    for cond, gc in pairs_df.groupby("condition"):
        add(f"human_share_not_satisfied_shared[{cond}]", int((gc["human"] != "satisfied").sum()), len(gc), gc["run_id"].nunique())
        add(f"judge_share_not_satisfied_shared[{cond}]", int((gc["judge"] != "satisfied").sum()), len(gc), gc["run_id"].nunique())

    # Run-mean versions of the same shares: per-run fraction first, then the mean over shared runs.
    if n_items:
        per_run = pairs_df.assign(h_ns=pairs_df["human"] != "satisfied", j_ns=pairs_df["judge"] != "satisfied",
                                  h_v=h_v, j_v=j_v, h_v_j_not=(h_v & ~j_v)).groupby("run_id")[["h_ns", "j_ns", "h_v", "j_v", "h_v_j_not"]].mean()
        note = "mean over shared runs of the per-run item fraction"
        for col, metric in (("h_ns", "human_share_not_satisfied_shared"), ("j_ns", "judge_share_not_satisfied_shared"),
                            ("h_v", "human_share_violated_shared"), ("j_v", "judge_share_violated_shared"),
                            ("h_v_j_not", "human_violated_judge_not_violated")):
            add(metric, float(per_run[col].sum()), len(per_run), n_shared, note, aggregation="run_mean")
        add("human_over_judge_not_satisfied_ratio", float(per_run["h_ns"].mean()), float(per_run["j_ns"].mean()), n_shared, "ratio of the two run-mean shares", aggregation="run_mean")

    # Run-level sensitivity and specificity against the human any-violation flag.
    defs = {
        "structured_any_violation": ("structured_any_violation", runs[runs["has_structured"]], "structured judge marks at least one constraint violated (shared runs only)"),
        "judge_violations_listed": ("judge_violations_listed", runs[runs["judge_violations_listed"].notna()], "published judge file lists at least one entry in violations"),
        "judge_adherence_le_2": ("judge_adherence_le_2", runs[runs["j_constraint_adherence"].notna()], "published j_constraint_adherence <= 2 (rubric: at least one violation)"),
    }
    for name, (col, sub, note) in defs.items():
        hv = sub["any_violation"].astype(bool)
        jv = sub[col].astype(bool)
        tp, fp, fn, tn = int((hv & jv).sum()), int((~hv & jv).sum()), int((hv & ~jv).sum()), int((~hv & ~jv).sum())
        add(f"run_level_human_violation_runs[{name}]", int(hv.sum()), len(sub), len(sub), note, value=int(hv.sum()), aggregation="run_level")
        add(f"run_level_judge_violation_runs[{name}]", int(jv.sum()), len(sub), len(sub), note, value=int(jv.sum()), aggregation="run_level")
        add(f"run_level_sensitivity[{name}]", tp, tp + fn, len(sub), f"TP / (TP + FN); {note}", aggregation="run_level")
        add(f"run_level_specificity[{name}]", tn, tn + fp, len(sub), f"TN / (TN + FP); {note}", aggregation="run_level")
        add(f"run_level_true_positives[{name}]", tp, 1, len(sub), note, value=tp, aggregation="run_level")
        add(f"run_level_false_positives[{name}]", fp, 1, len(sub), note, value=fp, aggregation="run_level")

    # Score-level delta, single-shot to pressure.
    ss = runs[runs["condition"] == "single_shot"]
    mp = runs[runs["condition"] == "multi_turn_pressure"]
    for col, label in (("computed_score", "human_computed_score"), ("computed_score_from_constraint_median", "human_score_from_constraint_median"), ("j_constraint_adherence", "judge_adherence")):
        a, b = float(ss[col].mean()), float(mp[col].mean())
        add(f"{label}_single_shot_mean", a, 1, len(ss), value=a, aggregation="score")
        add(f"{label}_pressure_mean", b, 1, len(mp), value=b, aggregation="score")
        add(f"{label}_delta_pressure_minus_single_shot", b - a, 1, len(ss) + len(mp), value=b - a, aggregation="score")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    human_path = data_root / "human_validation" / "human_scores.json"
    raters_dir = data_root / "human_validation" / "raters"
    parquet = data_root / "aggregated" / "all_scores.parquet"
    score_dirs = [data_root / "scores", data_root / "openweight_subjects" / "scores"]

    human_raw = load_human_scores(human_path)
    scores = pd.read_parquet(parquet)
    checks_raw = load_structured_checks(data_root / "scores", "structured")
    rater_labels = load_rater_labels(raters_dir)

    # The paper's sample drops the stray Sonnet pilot runs (plan B5/Q5).
    stray = stray_run_ids(data_root)
    human, stray_filter_human = drop_stray(human_raw, data_root, stray=stray)
    stray_filter_human["applies_to"] = "human-scored runs"
    stray_filter_human["dropped_run_ids"] = sorted(set(human_raw["run_id"]) & stray)
    checks, stray_filter_checks = drop_stray(checks_raw, data_root, stray=stray)
    stray_filter_checks["applies_to"] = "structured constraint checks"
    judge_flags = load_judge_violations_flag(score_dirs)

    briefs_dir = data_root.parent / "briefs"
    brief_constraints = load_brief_constraints(briefs_dir)

    runs = build_runs(human, scores, checks, rater_labels, judge_flags)
    by_condition = by_condition_table(runs, rater_labels)
    vs_judge = human_vs_judge_table(runs, checks, brief_constraints)

    runs_out = runs.copy()
    runs_out["median_labels"] = runs_out["median_labels"].apply(lambda ls: "|".join(ls))

    inputs = {"human_scores": str(human_path), "raters": str(raters_dir), "parquet": str(parquet),
              "briefs": str(briefs_dir), "briefs_found": len(brief_constraints),
              "structured": str(data_root / "scores" / "structured_*.jsonl"), "judge_files": [str(d / "judge_*.jsonl") for d in score_dirs]}
    rules = {
        "labels": "constraint_statuses (per-constraint median label), lower-cased; not satisfied = stretched or violated",
        "median_label_pooled": "sum of item counts over the runs in the condition divided by the number of items",
        "median_label_run_mean": "mean over runs of the per-run fraction",
        "per_rater_pooled": "every rater label pooled; run-level quantities use rater-run pairs",
        "any_violation": "at least one median label violated in the run",
        "condition": "from the parquet by run_id; runs without a parquet row have condition NaN and are excluded from the per-condition rows but kept in 'all'",
        "item_alignment": "human and judge items are paired by position; runs whose item counts differ are dropped from the item comparison; the judge's constraint text is compared to the brief's hard_constraints[i] (whitespace and case normalised) and mismatches are counted in the shared_runs note",
        "human_vs_judge_aggregation": "pooled = item counts over all shared items; run_mean = mean over shared runs of the per-run item fraction; run_level = runs as units; score = mean 0-4 score",
        "sample": STRAY_RULE,
    }
    filters = [stray_filter_human, stray_filter_checks]
    rows_in = {"human_runs_loaded": int(len(human_raw)), "human_runs_after_stray_filter": int(len(human)), "parquet": int(len(scores)),
               "structured_checks_loaded": int(len(checks_raw)), "structured_checks_after_stray_filter": int(len(checks)),
               "rater_labels": int(len(rater_labels))}
    write_csv_with_provenance(runs_out, out_dir / "human_severity_runs.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(by_condition, out_dir / "human_severity_by_condition.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(vs_judge, out_dir / "human_vs_judge.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    return {"human_severity_runs": runs_out, "human_severity_by_condition": by_condition, "human_vs_judge": vs_judge}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    frames = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 120)
    print(frames["human_severity_by_condition"].to_string(index=False))
    print(frames["human_vs_judge"].to_string(index=False))


if __name__ == "__main__":
    main()
