"""Entry point for running all priority list analyses.

Usage:
    python -m drift_bench.run_analysis [--items 1 2 3] [--output-dir drift_bench/data/analysis]
    python -m drift_bench.run_analysis --items reliability camera_ready

Items (number or name):
    1 reliability:   Inter-rater reliability (run FIRST)
    2 verbosity:     Verbosity-controlled complexity regression
    3 probes:        Restatement probe analysis + surface fidelity gap
    4 validation:    LLM-judge validation
    5 debiasing:     Score debiasing and averaging
    6 camera_ready:  Every module in drift_bench.analysis.camera_ready with run()
    all: Run all items (default)
"""

import argparse
import importlib
import json
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from drift_bench.analysis.stats import load_aggregated

# Default paths
BASE_DIR = Path(__file__).parent
DEFAULT_AGGREGATED = BASE_DIR / "data" / "aggregated" / "all_scores.parquet"
DEFAULT_TRANSCRIPTS = BASE_DIR / "data" / "transcripts"
DEFAULT_BRIEFS = BASE_DIR / "briefs"
DEFAULT_OUTPUT = BASE_DIR / "data" / "analysis"
DEFAULT_FIGURES = BASE_DIR / "data" / "figures"
DEFAULT_DATA_ROOT = BASE_DIR / "data"
DEFAULT_CAMERA_READY_OUTPUT = BASE_DIR / "data" / "camera_ready" / "analysis"

ITEM_NAMES: dict[str, int] = {
    "reliability": 1,
    "verbosity": 2,
    "probes": 3,
    "validation": 4,
    "debiasing": 5,
    "camera_ready": 6,
}
ALL_ITEMS: list[int] = sorted(ITEM_NAMES.values())


def parse_items(tokens: list[str] | str) -> list[int]:
    """Turn ``--items`` tokens into item numbers.

    Accepts numbers ("1"), names ("camera_ready"), comma-separated lists
    ("1,2,3") and the word "all". Order is kept; duplicates are dropped.
    Raises ValueError on an unknown token.
    """
    if isinstance(tokens, str):
        tokens = [tokens]
    items: list[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            key = part.lower().replace("-", "_")
            if key == "all":
                chosen = ALL_ITEMS
            elif key in ITEM_NAMES:
                chosen = [ITEM_NAMES[key]]
            elif key.isdigit() and int(key) in ALL_ITEMS:
                chosen = [int(key)]
            else:
                raise ValueError(
                    f"Unknown item {part!r}. Use 1-{ALL_ITEMS[-1]}, "
                    f"{', '.join(ITEM_NAMES)} or 'all'."
                )
            for item in chosen:
                if item not in items:
                    items.append(item)
    return items


# Modules that read the CSVs the other camera_ready modules write; they
# run after everything else, in this order (macros is always the last).
# rebuttal_reproduction reads e7_strict_probe_runs.csv, which e7_summary writes.
ITEM_6_RUN_LAST = ("e7_summary", "rebuttal_reproduction", "macros")


def discover_camera_ready_modules(
    package_name: str = "drift_bench.analysis.camera_ready",
) -> list[tuple[str, ModuleType]]:
    """Import every module in the camera_ready package that defines ``run``.

    Returns (module_name, module) pairs in alphabetical order. Modules
    without a callable ``run`` are skipped. Private modules (``_name``)
    are skipped.
    """
    package = importlib.import_module(package_name)
    found: list[tuple[str, ModuleType]] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{info.name}")
        if not callable(getattr(module, "run", None)):
            continue
        if getattr(module, "RUN_IN_ITEM_6", True) is False:
            # Opt-out for modules that must be run on purpose (e.g. the
            # SWE harness, which executes model-written code).
            continue
        found.append((info.name, module))
    # Modules that consume the other modules' CSVs run last, in this order.
    tail = [n for n in ITEM_6_RUN_LAST if n in dict(found)]
    found.sort(key=lambda pair: (tail.index(pair[0]) + 1 if pair[0] in tail else 0, pair[0]))
    return found


def run_item_1(df: pd.DataFrame, output_dir: Path) -> None:
    """Item 1: Inter-rater reliability."""
    from drift_bench.analysis.reliability import run_reliability_analysis

    print("\n=== Item 1: Inter-Rater Reliability ===")
    results = run_reliability_analysis(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save kappa results
    kappa_df = results["cohens_kappa"]
    kappa_df.to_csv(output_dir / "reliability_kappa.csv", index=False)
    print("\nCohen's Kappa (overall):")
    overall = kappa_df[kappa_df["condition"] == "all"]
    for _, row in overall.iterrows():
        status = "OK" if row["kappa"] >= 0.6 else "WARNING"
        print(f"  {row['metric']:30s} kappa={row['kappa']:.3f}  n={row['n']}  [{status}]")

    # Save alpha results
    alpha_df = results["krippendorff_alpha"]
    alpha_df.to_csv(output_dir / "reliability_alpha.csv", index=False)
    print("\nKrippendorff's Alpha (overall):")
    overall_alpha = alpha_df[alpha_df["condition"] == "all"]
    for _, row in overall_alpha.iterrows():
        status = "OK" if row["alpha"] >= 0.667 else "WARNING"
        print(f"  {row['metric']:30s} alpha={row['alpha']:.3f}  n={row['n']}  [{status}]")

    # Save bootstrap CIs
    bootstrap_df = results["bootstrap_ci"]
    bootstrap_df.to_csv(output_dir / "reliability_bootstrap.csv", index=False)
    print("\nBootstrap 95% CIs:")
    for _, row in bootstrap_df.iterrows():
        print(
            f"  {row['metric']:30s} "
            f"kappa={row['kappa']:.3f} [{row['kappa_ci_lower']:.3f}, {row['kappa_ci_upper']:.3f}]  "
            f"alpha={row['alpha']:.3f} [{row['alpha_ci_lower']:.3f}, {row['alpha_ci_upper']:.3f}]"
        )

    # Save drift agreement
    drift_agr = results["drift_agreement"]
    with open(output_dir / "reliability_drift_agreement.json", "w") as f:
        json.dump(drift_agr, f, indent=2, default=float)
    print(f"\nDrift classification agreement: {drift_agr}")

    # Generate reliability figure
    from drift_bench.analysis.figures import fig_reliability
    fig_reliability(bootstrap_df, output_dir.parent / "figures" / "fig_reliability.pdf")
    print("Reliability figure saved.")


def run_item_2(
    df: pd.DataFrame,
    transcripts_dir: Path,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Item 2: Verbosity-controlled complexity regression."""
    from drift_bench.analysis.verbosity import run_verbosity_analysis
    from drift_bench.analysis.figures import fig_complexity_vs_tokens

    print("\n=== Item 2: Verbosity-Controlled Complexity Analysis ===")
    results = run_verbosity_analysis(transcripts_dir, df)

    output_dir.mkdir(parents=True, exist_ok=True)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return None

    # Save features
    features_df = results["features"]
    features_df.to_csv(output_dir / "transcript_features.csv", index=False)
    print(f"  Extracted features for {len(features_df)} transcripts")

    # Save merged data
    merged_df = results["merged"]
    merged_df.to_parquet(output_dir / "features_scores_merged.parquet", index=False)
    print(f"  Merged features+scores: {len(merged_df)} rows")

    # Print regression results
    regression = results["regression"]
    for model_name in ["ols_primary", "ordinal_primary", "ols_secondary"]:
        model_result = regression.get(model_name, {})
        if "error" in model_result:
            print(f"\n  {model_name}: ERROR — {model_result['error']}")
        elif "summary" in model_result:
            print(f"\n  {model_name}:")
            # Print compact summary
            lines = model_result["summary"].split("\n")
            for line in lines:
                if any(k in line.lower() for k in ["condition", "model", "log_tokens", "_complexity", "r-squared"]):
                    print(f"    {line}")

    # Save full regression results
    with open(output_dir / "regression_results.json", "w") as f:
        # Convert non-serializable items
        serializable = {}
        for k, v in regression.items():
            if isinstance(v, dict):
                serializable[k] = {
                    kk: (float(vv) if isinstance(vv, (int, float)) else str(vv))
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2, default=str)

    # Print partial correlations
    partial = regression.get("partial_correlations", {})
    if isinstance(partial, dict) and "error" not in partial:
        print("\n  Partial correlations (complexity ~ condition | tokens):")
        for comparison, vals in partial.items():
            if isinstance(vals, dict):
                print(f"    {comparison}: rho={vals.get('spearman_rho', 'N/A'):.3f}, p={vals.get('p_value', 'N/A'):.4f}")

    # Generate scatter figure
    scatter_data = results.get("scatter_data")
    if scatter_data is not None and not scatter_data.empty:
        figures_dir = output_dir.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_complexity_vs_tokens(scatter_data, figures_dir / "fig_complexity_vs_tokens.pdf")
        print("\n  Complexity vs tokens figure saved.")

    return merged_df


def run_item_3(
    df: pd.DataFrame,
    transcripts_dir: Path,
    briefs_dir: Path,
    output_dir: Path,
) -> None:
    """Item 3: Restatement probe analysis + surface fidelity gap."""
    from drift_bench.analysis.probes import run_probe_analysis
    from drift_bench.analysis.figures import fig_surface_fidelity_gap, fig_knows_but_violates

    print("\n=== Item 3: Restatement Probe Analysis + Surface Fidelity Gap ===")
    # Open-weight subject runs live outside the main transcripts tree; include them
    # so probe coverage spans all seven subject models.
    extra_dirs = [transcripts_dir.parent / "openweight_subjects" / "transcripts"]
    results = run_probe_analysis(transcripts_dir, briefs_dir, df, extra_transcripts_dirs=extra_dirs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save probe accuracy
    probe_df = results["probe_accuracy"]
    if not probe_df.empty:
        probe_df.to_csv(output_dir / "probe_accuracy.csv", index=False)
        print(f"  Analyzed probes for {probe_df['run_id'].nunique()} runs")

        # Summary stats
        final_probes = results["final_probe_accuracy"]
        if not final_probes.empty:
            final_probes.to_csv(output_dir / "final_probe_accuracy.csv", index=False)
            mean_acc = final_probes["overall_accuracy"].mean()
            print(f"  Mean final-turn probe accuracy: {mean_acc:.3f}")

            by_model = final_probes.groupby("model_id")["overall_accuracy"].mean()
            print("  By model:")
            for model, acc in by_model.items():
                print(f"    {model}: {acc:.3f}")
    else:
        print("  No probe data found in transcripts")

    # Save surface fidelity gap summary
    gap_summary = results["surface_gap_summary"]
    if not gap_summary.empty:
        gap_summary.to_csv(output_dir / "surface_gap_summary.csv", index=False)
        print("\n  Surface fidelity gap by condition:")
        cond_gap = gap_summary.groupby("condition")["mean_surface_gap"].mean()
        for cond in CONDITION_ORDER:
            if cond in cond_gap.index:
                print(f"    {cond:30s} gap={cond_gap[cond]:.3f}")

    # Save knows-but-violates table
    kbv_df = results["knows_but_violates"]
    if not kbv_df.empty:
        kbv_df.to_csv(output_dir / "knows_but_violates.csv", index=False)
        kbv_count = kbv_df["knows_but_violates"].sum()
        kbv_total = len(kbv_df)
        print(f"\n  Knows-but-violates: {kbv_count}/{kbv_total} ({kbv_count/kbv_total*100:.1f}%)")

    # Save KBV summary
    kbv_summary = results["kbv_summary"]
    if not kbv_summary.empty:
        kbv_summary.to_csv(output_dir / "kbv_summary.csv", index=False)

    # Generate figures
    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_surface_fidelity_gap(df, figures_dir / "fig_surface_gap.pdf")
    print("  Surface fidelity gap figure saved.")

    if not kbv_summary.empty:
        fig_knows_but_violates(kbv_summary, figures_dir / "fig_knows_but_violates.pdf")
        print("  Knows-but-violates figure saved.")


def run_item_4(
    df: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Item 4: LLM-judge validation."""
    from drift_bench.analysis.validation import run_validation_analysis

    print("\n=== Item 4: LLM-Judge Validation ===")
    results = run_validation_analysis(df, merged_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save human spot-check sample
    sample = results["spotcheck_sample"]
    if not sample.empty:
        cols = ["run_id", "brief_id", "model_id", "condition"]
        if "a_drift_classification" in sample.columns:
            cols.append("a_drift_classification")
        sample[cols].to_csv(output_dir / "spotcheck_sample.csv", index=False)
        print(f"  Selected {len(sample)} runs for human spot-checking")
        print(f"  Sample breakdown:")
        for cond in sample["condition"].unique():
            n = len(sample[sample["condition"] == cond])
            print(f"    {cond}: {n}")
    else:
        print("  Could not select spot-check sample")

    # Save positional bias results
    bias_df = results["positional_bias"]
    if isinstance(bias_df, pd.DataFrame) and not bias_df.empty:
        bias_df.to_csv(output_dir / "positional_bias.csv", index=False)
        print("\n  Positional bias check:")
        for _, row in bias_df.iterrows():
            sig = "*" if row["partial_p_value"] < 0.05 else ""
            print(
                f"    {row['metric']:25s} "
                f"raw_rho={row['raw_spearman_rho']:+.3f} "
                f"partial_rho={row['partial_spearman_rho']:+.3f}{sig} "
                f"(p={row['partial_p_value']:.4f})"
            )
    else:
        print("\n  Positional bias check: requires merged features+scores data (run item 2 first)")

    # Save structural validation
    struct_val = results["structural_validation"]
    with open(output_dir / "structural_validation.json", "w") as f:
        json.dump(struct_val, f, indent=2, default=float)
    print(f"\n  Structural count validation: {struct_val.get('status', struct_val)}")


CONDITION_ORDER = [
    "single_shot",
    "multi_turn_neutral",
    "multi_turn_pressure",
    "checkpointed_pressure",
]


def run_item_5(
    df: pd.DataFrame,
    transcripts_dir: Path,
    output_dir: Path,
) -> None:
    """Item 5: Debiasing — averaged scores + best-available-per-dimension."""
    from drift_bench.analysis.debiasing import run_debiasing_analysis
    from drift_bench.analysis.verbosity import extract_transcript_features

    print("\n=== Item 5: Score Debiasing & Averaging ===")

    features_df = extract_transcript_features(transcripts_dir)
    results = run_debiasing_analysis(df, features_df)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effectiveness
    eff = results["effectiveness"]
    eff.to_csv(output_dir / "debiasing_effectiveness.csv", index=False)

    print("\n  Positional bias comparison (partial rho with token count):")
    for metric in ["objective_fidelity", "constraint_adherence",
                    "alternative_coverage", "complexity_inflation"]:
        metric_rows = eff[eff["metric"] == metric]
        print(f"\n  {metric}:")
        for _, row in metric_rows.iterrows():
            sig = "*" if row["p_value"] < 0.05 else " "
            print(f"    {row['variant']:20s} rho={row['partial_rho']:+.3f}{sig}")

    # Save reliability comparison
    rel = results["reliability_comparison"]
    rel.to_csv(output_dir / "reliability_comparison.csv", index=False)

    print("\n  Criterion validity (correlation with drift classification):")
    for _, row in rel.iterrows():
        print(
            f"    {row['metric']:25s} "
            f"judge={row['judge_drift_corr']:+.3f} "
            f"auditor={row['auditor_drift_corr']:+.3f} "
            f"averaged={row['averaged_drift_corr']:+.3f}"
        )

    # Save debiased merged data
    merged = results["merged_debiased"]
    merged.to_parquet(output_dir / "debiased_scores.parquet", index=False)
    print(f"\n  Saved debiased scores: {len(merged)} rows")

    # Best-available recommendation
    print("\n  Best-available score per dimension:")
    print("    Fidelity:    AVERAGED (bias: -0.03 NS, best criterion validity)")
    print("    Constraints: AVERAGED (bias halved: -0.32 -> -0.16)")
    print("    Alternatives: JUDGE   (auditor bias worse: +0.46 vs +0.25)")
    print("    Complexity:  JUDGE    (best reliability kappa=0.71)")


def run_item_6(data_root: Path, out_dir: Path) -> dict[str, dict]:
    """Item 6: every camera-ready analysis module.

    Each module exposes ``run(data_root, out_dir) -> dict[str, DataFrame]``.
    Returns {module_name: module result}. A failing module is reported
    and does not stop the others.
    """
    print("\n=== Item 6: Camera-Ready Analyses ===")
    modules = discover_camera_ready_modules()
    if not modules:
        print("  No camera_ready modules with run() found.")
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name, module in modules:
        print(f"\n  --- {name} ---")
        try:
            result = module.run(data_root, out_dir)
            if not isinstance(result, dict):
                raise TypeError(
                    f"run() must return dict[str, DataFrame], got {type(result).__name__}"
                )
            results[name] = result
            for key, value in result.items():
                n = len(value) if hasattr(value, "__len__") else "?"
                print(f"    {key}: {n} rows")
        except Exception as e:  # keep going; report at the end
            print(f"  ERROR in {name}: {e}")
            results[name] = {"error": str(e)}
    failed = [n for n, r in results.items() if isinstance(r, dict) and "error" in r]
    print(f"\n  Ran {len(modules)} module(s); {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run priority list analyses for drift_bench."
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=["all"],
        help="Items to run: numbers or names, space- or comma-separated "
             f"(e.g. '1 2 3', '1,2,3', 'camera_ready'), or 'all'. Names: {', '.join(ITEM_NAMES)}",
    )
    parser.add_argument(
        "--aggregated",
        type=Path,
        default=DEFAULT_AGGREGATED,
        help="Path to aggregated parquet file",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=DEFAULT_TRANSCRIPTS,
        help="Path to transcripts directory",
    )
    parser.add_argument(
        "--briefs",
        type=Path,
        default=DEFAULT_BRIEFS,
        help="Path to briefs directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Data root passed to camera_ready modules (item 6)",
    )
    parser.add_argument(
        "--camera-ready-dir",
        type=Path,
        default=DEFAULT_CAMERA_READY_OUTPUT,
        help="Output directory for camera_ready modules (item 6)",
    )
    args = parser.parse_args()

    # Parse items
    try:
        items = parse_items(args.items)
    except ValueError as e:
        parser.error(str(e))

    output_dir = args.output_dir
    merged_df = None
    df = None

    # Items 1 to 5 need the aggregated scores. Item 6 reads data_root itself.
    if any(i in items for i in (1, 2, 3, 4, 5)):
        print(f"Loading aggregated scores from {args.aggregated}")
        df = load_aggregated(args.aggregated)
        print(f"  Loaded {len(df)} scored runs")

    # Run items in priority order
    if 1 in items:
        run_item_1(df, output_dir)

    if 2 in items:
        merged_df = run_item_2(df, args.transcripts, output_dir)

    if 3 in items:
        run_item_3(df, args.transcripts, args.briefs, output_dir)

    if 4 in items:
        run_item_4(df, merged_df, output_dir)

    if 5 in items:
        run_item_5(df, args.transcripts, output_dir)

    if 6 in items:
        run_item_6(args.data_root, args.camera_ready_dir)

    print(f"\n=== All requested items complete. Results in {output_dir} ===")


if __name__ == "__main__":
    main()
