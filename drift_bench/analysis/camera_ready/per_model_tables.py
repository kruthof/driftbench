"""Per-model, per-condition appendix tables (plan item D6).

For the clean sample (the parquet with the 30 stray Sonnet rows
dropped by `sample.py`, 2,116 runs): the mean and standard error of
the four judge dimensions and the five auditor dimensions, the auditor
drift-classification shares, and the KBV rate recomputed per cell from
`analysis/final_probe_accuracy.csv`, for every model x condition cell.
An as-published variant (2,146 rows, KBV from `analysis/kbv_summary.csv`)
is written beside it as `per_model_condition_means_as_published.csv`.

Outputs: `per_model_condition_means.csv` and a booktabs LaTeX file
`per_model_condition_means.tex` with one block per model and four
condition rows per block. The .tex holds two tabulars: judge dimensions
with KBV and drift shares, then auditor dimensions. It needs the
packages booktabs, multirow and graphicx, and scales each tabular to
the NeurIPS textwidth with resizebox.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready.b4_common import (
    CONDITION_LABELS,
    CONDITIONS,
    DEFAULT_DATA_ROOT,
    MODEL_LABELS,
    MODEL_ORDER,
    attach_kbv,
    default_out_dir,
    final_probe_table,
    load_scores_with_filter,
    write_csv,
)

JUDGE_DIMS = {
    "j_objective_fidelity": "OF",
    "j_constraint_adherence": "CA",
    "j_alternative_coverage": "AC",
    "j_complexity_inflation": "CI",
}
AUDITOR_DIMS = {
    "a_objective_fidelity": "OF",
    "a_constraint_adherence": "CA",
    "a_alternative_coverage": "AC",
    "a_complexity_inflation": "CI",
    "a_recoverability": "Rec",
}
DRIFT_CLASSES = ["no_drift", "mild_drift", "trajectory_drift", "trajectory_lock_in"]


def cell_means(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, SE and n per model x condition for every dimension."""
    dims = list(JUDGE_DIMS) + list(AUDITOR_DIMS)
    rows = []
    for (model, cond), g in scores.groupby(["model_id", "condition"]):
        row = {"model_id": model, "condition": cond, "n": int(len(g))}
        for d in dims:
            vals = g[d].astype(float).dropna()
            row[f"{d}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{d}_se"] = (float(vals.std(ddof=1) / np.sqrt(len(vals)))
                              if len(vals) > 1 else np.nan)
            row[f"{d}_n"] = int(len(vals))
        cls = g["a_drift_classification"]
        for c in DRIFT_CLASSES:
            row[f"share_{c}"] = float((cls == c).mean())
        row["n_classified"] = int(cls.notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    return _order(out)


def kbv_summary_from_probes(scores: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """Per-cell KBV in the kbv_summary.csv layout, from the probe table.

    Same definition as analysis/kbv_summary.csv: knows = final probe
    overall_accuracy >= 0.8; violates = judge adherence < 4.
    """
    df = attach_kbv(scores, probes)
    df = df[df["kbv"].notna()]
    g = df.groupby(["condition", "model_id"])
    out = pd.DataFrame({
        "total": g.size(),
        "knows_count": g["knows"].sum().astype(int),
        "violates_count": g["noncompliant"].sum().astype(int),
        "kbv_count": g["kbv"].sum().astype(int),
    }).reset_index()
    out["kbv_rate"] = out["kbv_count"] / out["total"]
    out["knows_rate"] = out["knows_count"] / out["total"]
    out["violates_rate"] = out["violates_count"] / out["total"]
    return out.rename(columns={"condition": "condition_score", "model_id": "model_id_score"})


def attach_kbv_rates(cells: pd.DataFrame, kbv_summary: pd.DataFrame) -> pd.DataFrame:
    """Join kbv_summary.csv (condition_score, model_id_score) onto the cells."""
    k = kbv_summary.rename(columns={
        "condition_score": "condition", "model_id_score": "model_id",
        "total": "kbv_n", "kbv_count": "kbv_count", "kbv_rate": "kbv_rate",
        "knows_rate": "knows_rate",
    })[["condition", "model_id", "kbv_n", "kbv_count", "kbv_rate", "knows_rate"]]
    return _order(cells.merge(k, on=["condition", "model_id"], how="left"))


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = df["model_id"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["_c"] = df["condition"].map({c: i for i, c in enumerate(CONDITIONS)}).fillna(99)
    df = df.sort_values(["_m", "_c"]).drop(columns=["_m", "_c"])
    return df.reset_index(drop=True)


# ------------------------------------------------------------ LaTeX


def _fmt(mean, se) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.2f} ({se:.2f})"


def _pct(x) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


SHORT_CONDITION = {
    "single_shot": "SS",
    "multi_turn_neutral": "MT-N",
    "multi_turn_pressure": "MT-P",
    "checkpointed_pressure": "CK-P",
}


def _block_rows(cells, dims, extra_cols):
    """Body lines: one block per model, one line per condition row."""
    lines = []
    present = list(dict.fromkeys(cells["model_id"]))
    ordered = [m for m in MODEL_ORDER if m in present] + \
        [m for m in present if m not in MODEL_ORDER]
    for model in ordered:
        block = cells[cells["model_id"] == model]
        label = _tex_escape(MODEL_LABELS.get(model, model))
        first = True
        for _, r in block.iterrows():
            lead = rf"\multirow{{{len(block)}}}{{*}}{{{label}}}" if first else ""
            first = False
            cond = SHORT_CONDITION.get(r["condition"], _tex_escape(str(r["condition"])))
            vals = [_fmt(r[f"{d}_mean"], r[f"{d}_se"]) for d in dims]
            tail = " & ".join(fn(r) for fn in extra_cols)
            lines.append(f"{lead} & {cond} & {int(r['n'])} & " + " & ".join(vals)
                         + f" & {tail} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return lines


def _table(caption, label, header, body):
    return [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrcccccrr}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\end{tabular}}",
        r"\end{table}",
    ]


def kbv_denominator_sentence(cells: pd.DataFrame) -> str:
    """Caption text for the KBV denominator, computed from the cells."""
    base = ("KBV is defined only for multi-turn conditions, and its denominator "
            "is the runs with a final restatement probe")
    has = cells.dropna(subset=["kbv_n"])
    smaller = has[has["kbv_n"] < has["n"]]
    if smaller.empty:
        return base + ", which equals $n$ in every cell shown."
    r = smaller.sort_values("n", ascending=False).iloc[0]
    label = MODEL_LABELS.get(r["model_id"], r["model_id"])
    cond = SHORT_CONDITION.get(r["condition"], r["condition"])
    return (base + f", which can be smaller than $n$ (for example {_tex_escape(label)} "
            f"{cond}: {int(r['kbv_n'])} of {int(r['n'])}).")


def render_latex(cells: pd.DataFrame) -> str:
    """Two booktabs tabulars, one block per model, one row per condition.

    Requires the LaTeX packages booktabs, multirow and graphicx
    (for resizebox). NeurIPS textwidth is 397pt; the tabulars are
    scaled to it.
    """
    lines = [
        "% Generated by drift_bench.analysis.camera_ready.per_model_tables. Do not edit by hand.",
        "% Requires: \\usepackage{booktabs,multirow,graphicx}",
        "% Cells are mean (standard error). KBV and drift shares are percentages.",
        "% Conditions: SS single-shot, MT-N multi-turn neutral, MT-P multi-turn pressure, CK-P checkpointed pressure.",
    ]
    judge_caption = (
        "Per-model, per-condition judge scores (mean, SE), KBV rate and auditor "
        "drift-classification shares. Judge: Opus 4.6 for non-Anthropic subjects, "
        "GPT-5.4 for Sonnet 4.6. $n$ counts scored runs. "
        + kbv_denominator_sentence(cells)
        + " SS: single-shot; MT-N: multi-turn neutral; "
        "MT-P: multi-turn pressure; CK-P: checkpointed pressure."
    )
    lines += _table(
        judge_caption, "tab:permodel_judge",
        r"Model & Cond. & $n$ & OF & CA & AC & CI & KBV \% & Traj.\ drift \% & Lock-in \% \\",
        _block_rows(cells, list(JUDGE_DIMS), [
            lambda r: _pct(r.get("kbv_rate")),
            lambda r: _pct(r["share_trajectory_drift"]),
            lambda r: _pct(r["share_trajectory_lock_in"]),
        ]),
    )
    lines.append("")
    auditor_caption = (
        "Per-model, per-condition auditor scores (GPT-5.4 auditor; mean, SE) and the "
        "share of runs the auditor classified as no drift or mild drift. Condition "
        "abbreviations as in Table~\\ref{tab:permodel_judge}."
    )
    lines += _table(
        auditor_caption, "tab:permodel_auditor",
        r"Model & Cond. & $n$ & OF & CA & AC & CI & Rec & No drift \% & Mild \% \\",
        _block_rows(cells, list(AUDITOR_DIMS), [
            lambda r: _pct(r["share_no_drift"]),
            lambda r: _pct(r["share_mild_drift"]),
        ]),
    )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ run


def _cells_for(scores: pd.DataFrame, kbv: pd.DataFrame) -> pd.DataFrame:
    return attach_kbv_rates(cell_means(scores), kbv)


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path | None = None
        ) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or default_out_dir(data_root)
    kbv_path = data_root / "analysis" / "kbv_summary.csv"
    kbv_published = pd.read_csv(kbv_path)

    # Clean sample (the paper's): stray rows dropped, KBV recomputed
    # per cell from the final probe so its denominator matches.
    scores, stray_filter = load_scores_with_filter(data_root)
    probes = final_probe_table(data_root)
    kbv_clean = kbv_summary_from_probes(scores, probes)
    cells = _cells_for(scores, kbv_clean)
    prov = {
        "inputs": [str(data_root / "aggregated" / "all_scores.parquet"),
                   str(data_root / "analysis" / "final_probe_accuracy.csv")],
        "filters": [
            stray_filter,
            {"rule": "kbv per cell recomputed from final_probe_accuracy.csv on the clean sample: "
                     "kbv = final overall_accuracy >= 0.8 and j_constraint_adherence < 4; "
                     "denominator = runs with a final probe; single_shot has no probe and gets NaN",
             "rows_in": int(len(scores)), "rows_out": int(cells["kbv_rate"].notna().sum())},
        ],
        "definitions": {
            "se": "sample standard deviation (ddof=1) / sqrt(n), over non-missing values",
            "share_<class>": "share of the cell's runs with that a_drift_classification",
        },
    }
    write_csv(cells, out_dir, "per_model_condition_means", prov)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_model_condition_means.tex").write_text(render_latex(cells), encoding="utf-8")

    # As-published variant: the 2,146-row parquet and kbv_summary.csv.
    scores_pub, _ = load_scores_with_filter(data_root, clean=False)
    cells_pub = _cells_for(scores_pub, kbv_published)
    prov_pub = {
        "inputs": [str(data_root / "aggregated" / "all_scores.parquet"), str(kbv_path)],
        "filters": [
            {"rule": "no filter: the as-published parquet, every row",
             "rows_in": int(len(scores_pub)), "rows_out": int(len(scores_pub))},
            {"rule": "kbv rates joined from kbv_summary.csv on (condition, model_id); single_shot has no probe and gets NaN",
             "rows_in": int(len(kbv_published)), "rows_out": int(cells_pub["kbv_rate"].notna().sum())},
        ],
        "definitions": prov["definitions"],
    }
    write_csv(cells_pub, out_dir, "per_model_condition_means_as_published", prov_pub)
    return {"per_model_condition_means": cells,
            "per_model_condition_means_as_published": cells_pub}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 250, "display.max_rows", 100)
    c = tables["per_model_condition_means"]
    print(c[["model_id", "condition", "n", "j_constraint_adherence_mean",
             "j_constraint_adherence_se", "a_constraint_adherence_mean",
             "kbv_rate", "share_trajectory_drift"]].to_string(index=False))


if __name__ == "__main__":
    main()
