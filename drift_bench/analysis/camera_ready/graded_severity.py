"""Graded severity (gKBV) for the camera-ready (plan item C2).

gKBV for one run is the fraction of hard-constraint checks whose status is
not ``satisfied`` in the structured per-constraint judge file. The structured
files exist for the five commercial models. Sonnet is judged by GPT-5.4, the
other four by Claude Opus 4.6. A later Opus re-scoring (E4) can be read from
``drift_bench/data/camera_ready/e4_structured_opus/`` with prefix
``structured_opus``. That directory may be absent.

Outputs (in ``out_dir``):

- ``graded_severity_runs.csv``: one row per (source, run) with a structured file.
- ``graded_severity_summary.csv``: per source, model and condition.
- ``graded_severity_tests.csv``: per source and model, the single-shot to
  pressure Wilcoxon signed-rank test paired by brief, Holm-corrected.
- ``graded_severity_opus_arm_tests.csv``: the same tests for the seven
  Opus-judged subjects (E4 rows for Sonnet, Llama and Qwen; the published
  structured rows for the other four), Holm-corrected over that one
  seven-test family.

Run with ``python -m drift_bench.analysis.camera_ready.graded_severity``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.sample import SONNET as SAMPLE_SONNET
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids
from drift_bench.schema.validate import load_all_briefs

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_BRIEFS_DIR = Path(__file__).resolve().parents[2] / "briefs"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"

STATUSES = ("satisfied", "stretched", "violated")
KNOWS_THRESHOLD = 0.8
PROBE_TRUNCATION_TOKENS = 1020
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

# Structured-file sources: (name, directory relative to data_root, file prefix).
PUBLISHED_SOURCE = ("structured", Path("scores"), "structured")
OPUS_SOURCE = ("structured_opus", Path("camera_ready") / "e4_structured_opus", "structured_opus")

# The Opus arm: one Opus-judged test per subject. When a subject has Opus
# rows in both sources, the E4 re-scoring wins.
OPUS_JUDGE = "anthropic/claude-opus-4-6"
OPUS_ARM_SOURCE_ORDER = (OPUS_SOURCE[0], PUBLISHED_SOURCE[0])
OPUS_ARM_RULE = (
    f"one row per model_id from graded_severity_tests.csv with judge_model == {OPUS_JUDGE}; when a model has such rows in "
    f"both sources, source {OPUS_SOURCE[0]} (E4) is kept over {PUBLISHED_SOURCE[0]}; the raw Wilcoxon p is the per-source "
    "test of test_table, unchanged; holm_p_opus_arm is Holm step-down over the wilcoxon_p of these rows (one family)"
)

CHECK_COLUMNS = [
    "source", "run_id", "brief_id", "model_id", "condition", "judge_model",
    "constraint_index", "constraint", "status", "constraint_adherence",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def normalise_status(status: str | None) -> str:
    """Map a judge status label to satisfied / stretched / violated.

    Labels are case-insensitive. Unknown labels raise, so a new label in a
    future file cannot be counted silently.
    """
    s = (status or "").strip().lower()
    if s not in STATUSES:
        raise ValueError(f"unknown constraint status {status!r}")
    return s


def load_brief_constraints(briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> dict[str, list[str]]:
    """Map brief id to its ``hard_constraints`` list. An absent directory gives an empty map."""
    briefs_dir = Path(briefs_dir)
    if not briefs_dir.is_dir():
        return {}
    return {b["id"]: list(b.get("hard_constraints") or []) for b in load_all_briefs(briefs_dir)}


def match_checks_to_brief(
    checks: list[dict], hard_constraints: list[str] | None,
) -> tuple[list[tuple[int, dict]], list[dict], str]:
    """Keep the checks that belong to the brief's hard constraints.

    The structured judge sometimes appends extra entries (banned moves,
    success criteria) after the brief's constraints. Returns
    ``(kept, dropped, method)``: ``kept`` is a list of
    ``(constraint_index, check)``, ``dropped`` the extra checks, ``method``
    one of ``"none"`` (no brief or no extras), ``"exact_text"`` (every
    constraint matched one check by exact text) or ``"truncated"`` (fallback:
    the first ``len(hard_constraints)`` checks kept, with a warning).
    """
    if not hard_constraints or len(checks) <= len(hard_constraints):
        return list(enumerate(checks)), [], "none"
    texts = [(c.get("constraint") or "").strip() for c in checks]
    kept: list[tuple[int, dict]] = []
    used: set[int] = set()
    for j, hc in enumerate(hard_constraints):
        hits = [i for i, txt in enumerate(texts) if txt == hc.strip() and i not in used]
        if len(hits) != 1:
            kept = []
            break
        kept.append((j, checks[hits[0]]))
        used.add(hits[0])
    if kept:
        dropped = [c for i, c in enumerate(checks) if i not in used]
        return kept, dropped, "exact_text"
    n = len(hard_constraints)
    return list(enumerate(checks[:n])), checks[n:], "truncated"


def load_structured_checks(
    scores_dir: Path,
    prefix: str = "structured",
    source: str | None = None,
    briefs: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Load every ``<prefix>_<run_id>.jsonl`` file into one row per constraint check.

    Returns an empty frame with the standard columns when the directory does
    not exist or holds no matching files. Files with an empty
    ``constraint_checks`` list yield no rows, and the loader records them in
    ``df.attrs["files_without_checks"]``.

    ``briefs`` maps brief id to its hard constraints. When a file holds more
    checks than its brief has constraints, the extras are dropped by
    :func:`match_checks_to_brief` and recorded in
    ``df.attrs["dropped_extra_checks"]`` (run_id, source, brief_id, method,
    n_dropped, texts). Without ``briefs``, or for a brief not in the map,
    every check is kept.
    """
    source = source or prefix
    briefs = briefs or {}
    rows: list[dict] = []
    n_files = 0
    files_without_checks: list[str] = []
    dropped_extra_checks: list[dict] = []
    if scores_dir.is_dir():
        for path in sorted(scores_dir.glob(f"{prefix}_*.jsonl")):
            n_files += 1
            with path.open() as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                files_without_checks.append(path.name)
                continue
            # One record per file. If a file ever holds several, the last wins.
            rec = json.loads(lines[-1])
            if path.name != f"{prefix}_{rec['run_id']}.jsonl":
                # A longer prefix (structured_opus_*) also matches structured_*. Skip it.
                n_files -= 1
                continue
            checks = rec.get("constraint_checks") or []
            if not checks:
                files_without_checks.append(path.name)
                continue
            kept, dropped, method = match_checks_to_brief(checks, briefs.get(rec.get("brief_id")))
            if dropped:
                entry = {"run_id": rec["run_id"], "source": source, "brief_id": rec.get("brief_id"),
                         "method": method, "n_dropped": len(dropped),
                         "texts": [c.get("constraint") for c in dropped]}
                dropped_extra_checks.append(entry)
                if method == "truncated":
                    logger.warning("%s: %d extra constraint checks did not match the brief by text; "
                                   "kept the first %d by position", path.name, len(dropped), len(kept))
            for i, chk in kept:
                rows.append({
                    "source": source,
                    "run_id": rec["run_id"],
                    "brief_id": rec.get("brief_id"),
                    "model_id": rec.get("model_id"),
                    "condition": rec.get("condition"),
                    "judge_model": rec.get("judge_model"),
                    "constraint_index": i,
                    "constraint": chk.get("constraint"),
                    "status": normalise_status(chk.get("status")),
                    "constraint_adherence": rec.get("constraint_adherence"),
                })
    df = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    df.attrs["n_files"] = n_files
    df.attrs["files_without_checks"] = files_without_checks
    df.attrs["dropped_extra_checks"] = dropped_extra_checks
    return df


def summarise_runs(checks: pd.DataFrame) -> pd.DataFrame:
    """Collapse constraint checks to one row per (source, run) with gKBV."""
    cols = [
        "source", "run_id", "brief_id", "model_id", "condition", "judge_model",
        "constraint_adherence", "n_checks", "n_satisfied", "n_stretched",
        "n_violated", "n_not_satisfied", "g_kbv",
    ]
    if checks.empty:
        return pd.DataFrame(columns=cols)
    keys = ["source", "run_id", "brief_id", "model_id", "condition", "judge_model"]
    g = checks.groupby(keys, dropna=False, sort=False)
    out = g.agg(
        constraint_adherence=("constraint_adherence", "first"),
        n_checks=("status", "size"),
        n_satisfied=("status", lambda s: int((s == "satisfied").sum())),
        n_stretched=("status", lambda s: int((s == "stretched").sum())),
        n_violated=("status", lambda s: int((s == "violated").sum())),
    ).reset_index()
    out["n_not_satisfied"] = out["n_stretched"] + out["n_violated"]
    out["g_kbv"] = out["n_not_satisfied"] / out["n_checks"]
    return out[cols]


def load_structured_runs(
    scores_dir: Path, prefix: str = "structured", source: str | None = None,
    briefs: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Per-run gKBV table from one structured-file directory. Absent dir gives an empty frame."""
    checks = load_structured_checks(scores_dir, prefix, source, briefs)
    runs = summarise_runs(checks)
    runs.attrs.update(checks.attrs)
    return runs


def load_all_structured_runs(data_root: Path, briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> pd.DataFrame:
    """Published-arm structured files plus the Opus re-scoring when it exists."""
    briefs = load_brief_constraints(briefs_dir)
    frames = []
    for name, rel, prefix in (PUBLISHED_SOURCE, OPUS_SOURCE):
        frames.append(load_structured_runs(data_root / rel, prefix, name, briefs))
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else frames[0]
    out.attrs["n_files"] = {name: f.attrs.get("n_files", 0) for (name, _, _), f in zip((PUBLISHED_SOURCE, OPUS_SOURCE), frames)}
    out.attrs["files_without_checks"] = {name: f.attrs.get("files_without_checks", []) for (name, _, _), f in zip((PUBLISHED_SOURCE, OPUS_SOURCE), frames)}
    out.attrs["dropped_extra_checks"] = {name: f.attrs.get("dropped_extra_checks", []) for (name, _, _), f in zip((PUBLISHED_SOURCE, OPUS_SOURCE), frames)}
    return out


def load_final_probe_accuracy(path: Path) -> pd.DataFrame:
    """``final_probe_accuracy.csv`` reduced to run_id, turn_index, overall_accuracy."""
    df = pd.read_csv(path)
    df = df[["run_id", "turn_index", "overall_accuracy"]].drop_duplicates("run_id")
    return df.rename(columns={"turn_index": "probe_turn_index"})


def final_probe_truncation(transcript_dirs: list[Path], threshold: int = PROBE_TRUNCATION_TOKENS,
                           any_probe: bool = False, stats_out: dict | None = None) -> dict[str, bool]:
    """Map run_id to whether the final probe reply hit the token cap.

    A probe reply is truncated when its ``token_count`` is at least
    ``threshold`` (the cap was 1,024; replies stop at 1,020). With
    ``any_probe`` the flag is set when any probe reply in the run is
    truncated. Runs without a probe are absent from the map. ``stats_out``
    receives the number of transcript files scanned per directory.
    """
    out: dict[str, bool] = {}
    scanned: dict[str, int] = {}
    for d in transcript_dirs:
        scanned[str(d)] = 0
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl")):
            with path.open() as fh:
                line = fh.readline()
            if not line.strip():
                continue
            scanned[str(d)] += 1
            rec = json.loads(line)
            probes = [m for m in rec.get("messages", []) if m.get("is_probe") and m.get("role") == "assistant"]
            if not probes:
                continue
            counts = [(m.get("token_count") or 0) for m in probes]
            hit = [c >= threshold for c in counts]
            out[rec["metadata"]["run_id"]] = bool(any(hit) if any_probe else hit[-1])
    if stats_out is not None:
        stats_out["transcript_files_scanned"] = scanned
        stats_out["runs_with_probe"] = len(out)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(values: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean. NaN for fewer than two values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values. NaN inputs stay NaN and do not count."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    adjusted = np.full(len(p), np.nan)
    if m == 0:
        return adjusted.tolist()
    order = np.argsort(np.where(valid, p, np.inf))
    running = 0.0
    rank = 0
    for i in order:
        if not valid[i]:
            continue
        adj = (m - rank) * p[i]
        running = max(running, adj)
        adjusted[i] = min(1.0, running)
        rank += 1
    return adjusted.tolist()


WILCOXON_METHOD = "asymptotic"
WILCOXON_ROUND_DECIMALS = 9
WILCOXON_RULE = (
    f"scipy.stats.wilcoxon on brief-mean differences (pressure minus single-shot), briefs present in both conditions; "
    f"differences rounded to {WILCOXON_ROUND_DECIMALS} decimals so equal deltas tie exactly; "
    f"zero_method='wilcox' (zero differences dropped); method='{WILCOXON_METHOD}' (normal approximation with tie "
    f"correction, correction=False) is the primary p (wilcoxon_p); method='exact' is reported beside it (wilcoxon_p_exact)"
)


def brief_paired_test(runs: pd.DataFrame, value: str = "g_kbv", cond_a: str = "single_shot", cond_b: str = "multi_turn_pressure") -> dict:
    """Wilcoxon signed-rank test of brief means, cond_b minus cond_a.

    Repetitions within a brief and condition are averaged first. Only briefs
    present in both conditions enter. Differences are rounded to
    ``WILCOXON_ROUND_DECIMALS`` decimals so that equal deltas tie exactly.
    The primary p uses ``method=WILCOXON_METHOD``; the exact p is reported
    beside it. A brief-paired t-test is reported too.
    """
    empty = {"n_briefs_paired": 0, "mean_a": np.nan, "mean_b": np.nan, "mean_brief_delta": np.nan,
             "n_zero_deltas": 0, "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "wilcoxon_p_exact": np.nan, "paired_t_p": np.nan}
    means = runs[runs["condition"].isin([cond_a, cond_b])].groupby(["brief_id", "condition"])[value].mean().unstack("condition")
    if cond_a not in means or cond_b not in means:
        return empty
    means = means.dropna(subset=[cond_a, cond_b])
    a = means[cond_a].to_numpy(dtype=float)
    b = means[cond_b].to_numpy(dtype=float)
    diff = np.round(b - a, WILCOXON_ROUND_DECIMALS)
    out = dict(empty)
    out.update({"n_briefs_paired": int(len(means)), "mean_a": float(a.mean()) if len(a) else np.nan,
                "mean_b": float(b.mean()) if len(b) else np.nan,
                "mean_brief_delta": float(diff.mean()) if len(a) else np.nan,
                "n_zero_deltas": int((diff == 0).sum())})
    if len(diff) >= 2 and np.any(diff != 0):
        w = stats.wilcoxon(diff, zero_method="wilcox", method=WILCOXON_METHOD)
        out["wilcoxon_stat"] = float(w.statistic)
        out["wilcoxon_p"] = float(w.pvalue)
        try:
            out["wilcoxon_p_exact"] = float(stats.wilcoxon(diff, zero_method="wilcox", method="exact").pvalue)
        except ValueError:
            out["wilcoxon_p_exact"] = np.nan
        out["paired_t_p"] = float(stats.ttest_rel(b, a).pvalue)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def attach_recall(runs: pd.DataFrame, probes: pd.DataFrame, truncation: dict[str, bool] | None = None) -> pd.DataFrame:
    """Join final-probe accuracy and derive the recall flags."""
    df = runs.merge(probes, on="run_id", how="left")
    df["has_probe"] = df["overall_accuracy"].notna()
    df["perfect_recall"] = df["overall_accuracy"] == 1.0
    df["knows"] = df["overall_accuracy"] >= KNOWS_THRESHOLD
    trunc = truncation or {}
    df["final_probe_truncated"] = df["run_id"].map(trunc)
    df["final_probe_truncated"] = df["final_probe_truncated"].astype("boolean")
    return df


def _share_of_count(n_ns: pd.Series, k: int, at_least: bool = False) -> float:
    if len(n_ns) == 0:
        return np.nan
    hit = (n_ns >= k) if at_least else (n_ns == k)
    return float(hit.mean())


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Per source, model and condition summary of gKBV."""
    rows = []
    keys = ["source", "model_id", "condition"]
    for (source, model, cond), g in runs.groupby(keys, sort=True):
        knows = g[g["knows"]]
        perfect = g[g["perfect_recall"]]
        untrunc = perfect[perfect["final_probe_truncated"].fillna(False).astype(bool) == False]  # noqa: E712
        lo, hi = bootstrap_mean_ci(g["g_kbv"].to_numpy())
        plo, phi = bootstrap_mean_ci(perfect["g_kbv"].to_numpy())
        n_ns = g["n_not_satisfied"]
        rows.append({
            "source": source,
            "model_id": model,
            "judge_model": ",".join(sorted(g["judge_model"].dropna().unique())),
            "condition": cond,
            "n_runs": int(len(g)),
            "mean_g_kbv": float(g["g_kbv"].mean()),
            "se_g_kbv": float(g["g_kbv"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,
            "g_kbv_ci_lo": lo,
            "g_kbv_ci_hi": hi,
            "n_with_probe": int(g["has_probe"].sum()),
            "n_knows": int(len(knows)),
            "mean_g_kbv_knows": float(knows["g_kbv"].mean()) if len(knows) else np.nan,
            "n_perfect_recall": int(len(perfect)),
            "mean_g_kbv_perfect_recall": float(perfect["g_kbv"].mean()) if len(perfect) else np.nan,
            "g_kbv_perfect_ci_lo": plo,
            "g_kbv_perfect_ci_hi": phi,
            "n_final_probe_truncated": int(g["final_probe_truncated"].fillna(False).astype(bool).sum()),
            "n_perfect_recall_untruncated": int(len(untrunc)),
            "mean_g_kbv_perfect_recall_untruncated": float(untrunc["g_kbv"].mean()) if len(untrunc) else np.nan,
            "mean_n_checks": float(g["n_checks"].mean()),
            "mean_n_not_satisfied": float(n_ns.mean()),
            "modal_n_not_satisfied": int(n_ns.mode().iloc[0]),
            "share_0_not_satisfied": _share_of_count(n_ns, 0),
            "share_1_not_satisfied": _share_of_count(n_ns, 1),
            "share_2_not_satisfied": _share_of_count(n_ns, 2),
            "share_3plus_not_satisfied": _share_of_count(n_ns, 3, at_least=True),
            "total_checks": int(g["n_checks"].sum()),
            "total_stretched": int(g["n_stretched"].sum()),
            "total_violated": int(g["n_violated"].sum()),
        })
    return pd.DataFrame(rows)


def test_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Single-shot to pressure Wilcoxon per model, Holm-corrected within each source."""
    rows = []
    for (source, model), g in runs.groupby(["source", "model_id"], sort=True):
        res = brief_paired_test(g)
        rows.append({
            "source": source,
            "model_id": model,
            "judge_model": ",".join(sorted(g["judge_model"].dropna().unique())),
            "n_runs_single_shot": int((g["condition"] == "single_shot").sum()),
            "n_runs_pressure": int((g["condition"] == "multi_turn_pressure").sum()),
            "n_briefs_paired": res["n_briefs_paired"],
            "mean_g_kbv_single_shot_brief": res["mean_a"],
            "mean_g_kbv_pressure_brief": res["mean_b"],
            "mean_brief_delta": res["mean_brief_delta"],
            "n_zero_deltas": res["n_zero_deltas"],
            "wilcoxon_method": WILCOXON_METHOD,
            "wilcoxon_stat": res["wilcoxon_stat"],
            "wilcoxon_p": res["wilcoxon_p"],
            "wilcoxon_p_exact": res["wilcoxon_p_exact"],
            "paired_t_p": res["paired_t_p"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # holm_p: family = every model in the source (mixed judges until E4).
    # holm_p_within_judge: family = models sharing one judge within the source.
    df["holm_p"] = np.nan
    df["holm_p_paired_t"] = np.nan
    df["holm_p_within_judge"] = np.nan
    for source, idx in df.groupby("source").groups.items():
        df.loc[idx, "holm_p"] = holm_correction(df.loc[idx, "wilcoxon_p"].tolist())
        df.loc[idx, "holm_p_paired_t"] = holm_correction(df.loc[idx, "paired_t_p"].tolist())
    for (source, judge), idx in df.groupby(["source", "judge_model"]).groups.items():
        df.loc[idx, "holm_p_within_judge"] = holm_correction(df.loc[idx, "wilcoxon_p"].tolist())
    df["significant_holm_05"] = df["holm_p"] < 0.05
    df["significant_holm_05_paired_t"] = df["holm_p_paired_t"] < 0.05
    df["n_models_in_holm_family"] = df.groupby("source")["wilcoxon_p"].transform(lambda s: int(s.notna().sum()))
    df["n_models_in_within_judge_family"] = df.groupby(["source", "judge_model"])["wilcoxon_p"].transform(lambda s: int(s.notna().sum()))
    return df


OPUS_ARM_COLUMNS = [
    "source", "model_id", "judge_model", "n_runs_single_shot", "n_runs_pressure", "n_briefs_paired",
    "mean_g_kbv_single_shot_brief", "mean_g_kbv_pressure_brief", "mean_brief_delta", "n_zero_deltas",
    "wilcoxon_method", "wilcoxon_stat", "wilcoxon_p", "wilcoxon_p_exact", "paired_t_p",
]


def opus_arm_test_table(tests: pd.DataFrame, judge: str = OPUS_JUDGE) -> pd.DataFrame:
    """The single-shot to pressure tests under one judge, Holm over one family.

    Keeps one ``tests`` row per model whose ``judge_model`` is ``judge``,
    preferring the E4 source. The raw p values are those of ``test_table``;
    only the Holm family changes: every kept model, one family.
    """
    t = tests[tests["judge_model"] == judge].copy()
    if t.empty:
        return pd.DataFrame(columns=[*OPUS_ARM_COLUMNS, "holm_p_opus_arm", "holm_p_paired_t_opus_arm",
                                     "significant_holm_05_opus_arm", "n_models_in_opus_arm_family"])
    rank = {src: i for i, src in enumerate(OPUS_ARM_SOURCE_ORDER)}
    t["_rank"] = t["source"].map(rank).fillna(len(rank))
    t = t.sort_values(["model_id", "_rank"], kind="stable").drop_duplicates("model_id")
    out = t[OPUS_ARM_COLUMNS].reset_index(drop=True)
    out["holm_p_opus_arm"] = holm_correction(out["wilcoxon_p"].tolist())
    out["holm_p_paired_t_opus_arm"] = holm_correction(out["paired_t_p"].tolist())
    out["significant_holm_05_opus_arm"] = (out["holm_p_opus_arm"] < 0.05).astype(int)
    out["n_models_in_opus_arm_family"] = int(out["wilcoxon_p"].notna().sum())
    return out


# ---------------------------------------------------------------------------
# Output helpers (shared by the other camera-ready modules)
# ---------------------------------------------------------------------------

def write_csv_with_provenance(df: pd.DataFrame, path: Path, provenance: dict) -> None:
    """Write ``path`` and its sibling ``<name>.csv.provenance.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    prov = {"csv": path.name, "date": date.today().isoformat(), "rows_out": int(len(df)), **provenance}
    with path.with_name(path.name + ".provenance.json").open("w") as fh:
        json.dump(prov, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    probe_path = data_root / "analysis" / "final_probe_accuracy.csv"
    parquet = data_root / "aggregated" / "all_scores.parquet"
    transcript_dirs = [data_root / "transcripts", data_root / "openweight_subjects" / "transcripts"]

    loaded = load_all_structured_runs(data_root)
    # attrs do not survive merges; capture them now.
    files_found = dict(loaded.attrs.get("n_files", {}))
    files_without_checks = dict(loaded.attrs.get("files_without_checks", {}))
    dropped_extra_checks = dict(loaded.attrs.get("dropped_extra_checks", {}))
    n_loaded = int(len(loaded))

    # Keep only runs that are in the published parquet.
    scores = pd.read_parquet(parquet, columns=["run_id", "model_id", "condition"]) if parquet.exists() else None
    if scores is None:
        dropped_not_in_parquet: list[str] = []
        runs = loaded
    else:
        keep = loaded["run_id"].isin(set(scores["run_id"]))
        dropped_not_in_parquet = sorted(loaded.loc[~keep, "run_id"].tolist())
        runs = loaded[keep].reset_index(drop=True)
    parquet_filter = {"name": "parquet_filter", "rule": "drop structured runs whose run_id is not in all_scores.parquet",
                      "rows_in": n_loaded, "rows_dropped": len(dropped_not_in_parquet), "rows_out": int(len(runs)),
                      "dropped_not_in_parquet": dropped_not_in_parquet}

    # The paper's sample drops the stray Sonnet pilot runs (plan B5/Q5).
    stray = stray_run_ids(data_root)
    runs, stray_filter = drop_stray(runs, data_root, stray=stray)
    sonnet_pressure_structured = int(((runs["model_id"] == SAMPLE_SONNET) & (runs["condition"] == "multi_turn_pressure")).sum())
    if scores is not None:
        clean_scores, _ = drop_stray(scores, data_root, stray=stray)
        sonnet_pressure_parquet = int(((clean_scores["model_id"] == SAMPLE_SONNET) & (clean_scores["condition"] == "multi_turn_pressure")).sum())
    else:
        sonnet_pressure_parquet = None

    probes = load_final_probe_accuracy(probe_path)
    scan_stats: dict = {}
    truncation = final_probe_truncation(transcript_dirs, stats_out=scan_stats)
    any_truncation = final_probe_truncation(transcript_dirs, any_probe=True)
    runs = attach_recall(runs, probes, truncation)
    summary = summarise(runs)
    tests = test_table(runs)
    opus_tests = opus_arm_test_table(tests)

    # Side note for the Gemini Pro truncation variants (recorded, not tabulated).
    gp = runs[(runs["model_id"] == "gemini/gemini-3.1-pro-preview") & (runs["condition"] == "multi_turn_pressure") & runs["perfect_recall"]]
    gp_any_untrunc = gp[~gp["run_id"].map(any_truncation).fillna(False).astype(bool)]
    truncation_note = {
        "gemini_pro_pressure_perfect_recall_n": int(len(gp)),
        "final_probe_rule_untruncated_n": int((~gp["final_probe_truncated"].fillna(False).astype(bool)).sum()),
        "final_probe_rule_untruncated_mean_g_kbv": float(gp.loc[~gp["final_probe_truncated"].fillna(False).astype(bool), "g_kbv"].mean()) if len(gp) else None,
        "any_probe_rule_untruncated_n": int(len(gp_any_untrunc)),
        "any_probe_rule_untruncated_mean_g_kbv": float(gp_any_untrunc["g_kbv"].mean()) if len(gp_any_untrunc) else None,
    }

    inputs = {
        "structured_files": {
            PUBLISHED_SOURCE[0]: str(data_root / PUBLISHED_SOURCE[1] / f"{PUBLISHED_SOURCE[2]}_*.jsonl"),
            OPUS_SOURCE[0]: str(data_root / OPUS_SOURCE[1] / f"{OPUS_SOURCE[2]}_*.jsonl"),
        },
        "structured_files_found": files_found,
        "structured_files_without_checks": files_without_checks,
        "dropped_extra_checks": dropped_extra_checks,
        "opus_rescoring_present": (data_root / OPUS_SOURCE[1]).is_dir(),
        "parquet": str(parquet),
        "final_probe_accuracy": str(probe_path),
        "transcripts": [str(d) for d in transcript_dirs],
        "transcript_files_scanned": scan_stats.get("transcript_files_scanned", {}),
        "runs_with_probe_in_transcripts": scan_stats.get("runs_with_probe", 0),
    }
    rules = {
        "g_kbv": "n_not_satisfied / n_checks per run; not satisfied = status in {stretched, violated}; status labels lower-cased",
        "constraint_checks_matching": ("when a file holds more checks than its brief has hard_constraints, each constraint is matched "
                                       "to one check by exact text and the extras are dropped; if exact matching fails, the first "
                                       "len(hard_constraints) checks are kept by position with a warning; dropped extras are listed in "
                                       "inputs.dropped_extra_checks"),
        "parquet_filter": "structured runs whose run_id is not in all_scores.parquet are dropped (see filters[parquet_filter])",
        "sample": STRAY_RULE,
        "denominators": (f"clean sample (stray Sonnet runs dropped): this module counts runs with a structured file "
                         f"(Sonnet pressure n = {sonnet_pressure_structured}); stretch_violate.py counts parquet rows "
                         f"(Sonnet pressure n = {sonnet_pressure_parquet}). Sonnet pressure runs without a structured file are the runs without a usable final probe."),
        "perfect_recall": "final-turn overall_accuracy == 1.0 from final_probe_accuracy.csv; runs without a probe row (single-shot) are False",
        "knows": f"overall_accuracy >= {KNOWS_THRESHOLD}",
        "final_probe_truncated": f"last assistant probe message token_count >= {PROBE_TRUNCATION_TOKENS}; the summary's untruncated columns use this final-probe rule",
        "truncation_variants_note": truncation_note,
        "bootstrap_ci": f"percentile bootstrap of the mean, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
        "wilcoxon": WILCOXON_RULE,
        "holm": "holm_p: Holm step-down over all models within one source (a mixed-judge family until E4: Sonnet is GPT-5.4-judged, the other four Opus-judged); "
                "holm_p_within_judge: Holm over the models sharing one judge_model within the source; holm_p_paired_t: Holm over the brief-paired t-test p. "
                "Recommended paper family: brief-paired Wilcoxon (ties rounded, method stated) with Holm.",
    }
    filters = [parquet_filter, stray_filter]
    rows_in = {"structured_runs_loaded": n_loaded, "structured_runs_after_parquet_filter": parquet_filter["rows_out"],
               "structured_runs_after_stray_filter": int(len(runs)), "probe_rows": int(len(probes))}
    write_csv_with_provenance(runs, out_dir / "graded_severity_runs.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(summary, out_dir / "graded_severity_summary.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(tests, out_dir / "graded_severity_tests.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    opus_filters = filters + [{"name": "opus_arm_family", "rule": OPUS_ARM_RULE, "rows_in": int(len(tests)),
                               "rows_out": int(len(opus_tests)),
                               "kept": [f"{r.model_id} <- {r.source}" for r in opus_tests.itertuples()]}]
    write_csv_with_provenance(opus_tests, out_dir / "graded_severity_opus_arm_tests.csv",
                              {"inputs": inputs, "rules": rules, "filters": opus_filters, "rows_in": rows_in,
                               "derived_from": "graded_severity_tests.csv rows of the same run (no new test is computed)"})
    return {"graded_severity_runs": runs, "graded_severity_summary": summary, "graded_severity_tests": tests,
            "graded_severity_opus_arm_tests": opus_tests}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    frames = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s = frames["graded_severity_summary"]
    cols = ["source", "model_id", "condition", "n_runs", "mean_g_kbv", "n_perfect_recall",
            "mean_g_kbv_perfect_recall", "n_perfect_recall_untruncated", "mean_g_kbv_perfect_recall_untruncated",
            "mean_n_checks", "mean_n_not_satisfied", "modal_n_not_satisfied"]
    print(s[cols].to_string(index=False))
    t = frames["graded_severity_tests"]
    print(t[["source", "model_id", "n_briefs_paired", "n_zero_deltas", "wilcoxon_method", "wilcoxon_p", "wilcoxon_p_exact",
             "holm_p", "holm_p_within_judge", "paired_t_p", "holm_p_paired_t"]].to_string(index=False))


if __name__ == "__main__":
    main()
