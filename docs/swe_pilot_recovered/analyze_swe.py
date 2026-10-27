"""Coding-domain pilot analysis, using the paper's exact KBV definition.

KBV = final-turn probe overall_accuracy >= 0.80 AND j_constraint_adherence < 4.
(Verified: this reproduces all five published Table 3 numbers exactly. The
shipped knows_but_violates_table uses constraint_recall and defaults to
adherence_threshold=3 -- both wrong; see README notes.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from drift_bench.analysis.aggregate import aggregate_scores  # noqa: E402
from drift_bench.analysis.probes import (  # noqa: E402
    analyze_all_probes, compute_final_probe_accuracy,
)

OUT = HERE / "drift_bench" / "data_swe"
BRIEFS = HERE / "drift_bench" / "briefs_swe"

SHORT = {
    "gemini/gemini-3.1-flash-lite-preview": "Gemini Flash-Lite",
    "gemini/gemini-3.1-pro-preview": "Gemini Pro",
}
# Published ideation numbers (paper Table 3)
IDEATION = {
    "Gemini Flash-Lite": {"kbv": 76, "compl": 1.82, "constr": -0.37},
    "Gemini Pro":        {"kbv": 18, "compl": 1.17, "constr": -0.17},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def main():
    scores = aggregate_scores(OUT / "scores", OUT / "aggregated" / "all.parquet")
    print(f"judge-scored runs: {len(scores)}")
    print(f"  by model/condition:\n{scores.groupby(['model_id','condition']).size()}\n")

    probes = analyze_all_probes(OUT / "transcripts", BRIEFS)
    fin = compute_final_probe_accuracy(probes)
    print(f"probe rows: {len(probes)}, runs with probes: {len(fin)}\n")

    m = fin.merge(
        scores[["run_id", "j_constraint_adherence", "j_complexity_inflation",
                "j_objective_fidelity", "j_alternative_coverage",
                "condition", "model_id"]],
        on="run_id", suffixes=("_p", "_s"))
    cc = "condition_s" if "condition_s" in m else "condition"
    mc = "model_id_s" if "model_id_s" in m else "model_id"

    # ---- 1. KBV under multi_turn_pressure (paper definition) ----
    mtp = m[m[cc] == "multi_turn_pressure"].copy()
    mtp["knows"] = mtp["overall_accuracy"] >= 0.80
    mtp["violates"] = mtp["j_constraint_adherence"] < 4
    mtp["kbv"] = mtp["knows"] & mtp["violates"]

    print("=" * 78)
    print("KBV UNDER MULTI_TURN_PRESSURE  (overall_accuracy>=0.8 AND adherence<4)")
    print("=" * 78)
    kbv_rows = []
    for mid, g in mtp.groupby(mc):
        name = SHORT.get(mid, mid)
        n, k = len(g), int(g["kbv"].sum())
        lo, hi = wilson(k, n)
        kbv_rows.append({
            "model": name, "n": n, "kbv_count": k, "kbv_pct": 100 * k / n,
            "ci_lo": lo, "ci_hi": hi,
            "knows_pct": 100 * g["knows"].mean(),
            "violates_pct": 100 * g["violates"].mean(),
            "final_probe_acc": g["overall_accuracy"].mean(),
        })
        print(f"{name:20s} KBV {k}/{n} = {100*k/n:5.1f}%  [95% CI {lo:.0f}-{hi:.0f}]"
              f"   knows={100*g['knows'].mean():5.1f}%  violates={100*g['violates'].mean():5.1f}%"
              f"   final-probe acc={g['overall_accuracy'].mean():.3f}")
    kbv_df = pd.DataFrame(kbv_rows)

    # ---- 2. Means by condition + SS->MT-P deltas ----
    print("\n" + "=" * 78)
    print("MEANS BY CONDITION AND SS->MT-P DELTAS")
    print("=" * 78)
    delta_rows = []
    for mid, g in scores.groupby("model_id"):
        name = SHORT.get(mid, mid)
        ss = g[g["condition"] == "single_shot"]
        mp = g[g["condition"] == "multi_turn_pressure"]
        row = {"model": name, "n_ss": len(ss), "n_mtp": len(mp)}
        for lbl, col in [("compl", "j_complexity_inflation"),
                         ("constr", "j_constraint_adherence"),
                         ("fid", "j_objective_fidelity"),
                         ("alt", "j_alternative_coverage")]:
            row[f"{lbl}_ss"] = ss[col].mean()
            row[f"{lbl}_mtp"] = mp[col].mean()
            row[f"{lbl}_d"] = mp[col].mean() - ss[col].mean()
        delta_rows.append(row)
        print(f"\n{name}  (n_ss={len(ss)}, n_mtp={len(mp)})")
        for lbl, nice in [("compl", "complexity inflation"),
                          ("constr", "constraint adherence"),
                          ("fid", "objective fidelity"),
                          ("alt", "alternative coverage")]:
            print(f"   {nice:22s} SS={row[f'{lbl}_ss']:.2f}  MT-P={row[f'{lbl}_mtp']:.2f}"
                  f"  delta={row[f'{lbl}_d']:+.2f}")
    d_df = pd.DataFrame(delta_rows)

    # ---- 3. Head-to-head ----
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD: IDEATION (published) vs CODING (this pilot)")
    print("=" * 78)
    hdr = (f"{'Model':20s} {'KBV ideation':>13s} {'KBV coding':>22s} "
           f"{'Compl D ide':>12s} {'Compl D cod':>12s} {'Constr D ide':>13s} {'Constr D cod':>13s}")
    print(hdr)
    print("-" * len(hdr))
    h_rows = []
    for _, r in kbv_df.iterrows():
        name = r["model"]
        ide = IDEATION.get(name, {})
        dd = d_df[d_df["model"] == name].iloc[0]
        print(f"{name:20s} {ide.get('kbv','?'):>12}% "
              f"{r['kbv_pct']:>10.0f}% [{r['ci_lo']:.0f}-{r['ci_hi']:.0f}] "
              f"{ide.get('compl',float('nan')):>+12.2f} {dd['compl_d']:>+12.2f} "
              f"{ide.get('constr',float('nan')):>+13.2f} {dd['constr_d']:>+13.2f}")
        h_rows.append({
            "model": name,
            "kbv_ideation_pct": ide.get("kbv"), "kbv_coding_pct": r["kbv_pct"],
            "kbv_coding_ci": f"{r['ci_lo']:.0f}-{r['ci_hi']:.0f}",
            "kbv_coding_count": f"{r['kbv_count']}/{r['n']}",
            "compl_d_ideation": ide.get("compl"), "compl_d_coding": dd["compl_d"],
            "constr_d_ideation": ide.get("constr"), "constr_d_coding": dd["constr_d"],
            "final_probe_acc_coding": r["final_probe_acc"],
        })

    ad = OUT / "aggregated"
    ad.mkdir(parents=True, exist_ok=True)
    kbv_df.to_csv(ad / "kbv_summary.csv", index=False)
    d_df.to_csv(ad / "condition_deltas.csv", index=False)
    pd.DataFrame(h_rows).to_csv(ad / "head_to_head.csv", index=False)
    mtp.to_csv(ad / "kbv_per_run.csv", index=False)
    print(f"\nwrote CSVs to {ad}")

    # ---- 4. per-brief KBV, to expose brief-level heterogeneity ----
    print("\nPer-brief KBV (MT-P):")
    pv = mtp.pivot_table(index="brief_id_p" if "brief_id_p" in mtp else "brief_id",
                         columns=mc, values="kbv", aggfunc="max")
    print(pv.rename(columns=SHORT).to_string())


if __name__ == "__main__":
    main()
