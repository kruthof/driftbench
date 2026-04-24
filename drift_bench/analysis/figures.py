"""Publication-ready figures for drift_bench results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from drift_bench.analysis.stats import (
    load_aggregated,
    compute_condition_effects,
    compute_drift_classification_summary,
)

# Paper-quality defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})

CONDITION_ORDER = [
    "single_shot",
    "multi_turn_neutral",
    "multi_turn_pressure",
    "checkpointed_pressure",
]

CONDITION_LABELS = {
    "single_shot": "Single-shot",
    "multi_turn_neutral": "Multi-turn\n(neutral)",
    "multi_turn_pressure": "Multi-turn\n(pressure)",
    "checkpointed_pressure": "Checkpointed\n(pressure)",
}

METRIC_LABELS = {
    "j_objective_fidelity": "Goal Fidelity",
    "j_constraint_adherence": "Constraint\nAdherence",
    "j_alternative_coverage": "Alternative\nCoverage",
    "j_complexity_inflation": "Complexity\nInflation",
    "a_recoverability": "Recoverability",
}

MODEL_SHORT_NAMES = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4-mini",
    "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
    "gemini/gemini-3.1-pro-preview": "Gemini Pro",
    "gemini/gemini-3.1-flash-lite-preview": "Gemini Flash",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "Qwen3-235B",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-70B",
}


def _short_model(model_id: str) -> str:
    return MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])


def fig_main_scores(df: pd.DataFrame, output_path: Path) -> None:
    """Main result figure: grouped bar chart of scores by condition."""
    metrics = list(METRIC_LABELS.keys())
    effects = compute_condition_effects(df)
    effects = effects[effects["metric"].isin(metrics)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = effects[effects["metric"] == metric]
        metric_data = metric_data.set_index("condition").reindex(CONDITION_ORDER).reset_index()
        metric_data = metric_data.dropna(subset=["mean"])

        x = range(len(metric_data))
        ax.bar(
            x,
            metric_data["mean"],
            yerr=[
                metric_data["mean"] - metric_data["ci_lower"],
                metric_data["ci_upper"] - metric_data["mean"],
            ],
            capsize=3,
            color=sns.color_palette("muted", len(CONDITION_ORDER)),
            edgecolor="black",
            linewidth=0.5,
        )
        labels = [CONDITION_LABELS.get(c, c) for c in metric_data["condition"]]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        if idx == 0:
            ax.set_ylabel("Score")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fig_drift_classification(df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of drift classification proportions by condition."""
    summary = compute_drift_classification_summary(df)
    if summary.empty:
        return

    agg = summary.groupby(["condition", "a_drift_classification"])["count"].sum().reset_index()
    totals = agg.groupby("condition")["count"].sum().reset_index(name="total")
    agg = agg.merge(totals, on="condition")
    agg["proportion"] = agg["count"] / agg["total"]

    drift_classes = ["no_drift", "mild_drift", "trajectory_drift", "trajectory_lock_in"]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    present_conditions = [c for c in CONDITION_ORDER if c in agg["condition"].values]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(present_conditions))
    bottom = [0.0] * len(present_conditions)

    for cls, color in zip(drift_classes, colors):
        heights = []
        for cond in present_conditions:
            row = agg[(agg["condition"] == cond) & (agg["a_drift_classification"] == cls)]
            heights.append(row["proportion"].values[0] if len(row) > 0 else 0.0)
        ax.bar(
            x, heights, bottom=bottom,
            label=cls.replace("_", " ").title(),
            color=color, edgecolor="black", linewidth=0.5,
        )
        bottom = [b + h for b, h in zip(bottom, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in present_conditions], fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title("Drift Classification by Condition")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of mean constraint_adherence by model x condition."""
    if "j_constraint_adherence" not in df.columns:
        return

    df = df.copy()
    df["model_short"] = df["model_id"].map(_short_model)

    # Order models by pressure constraint adherence (worst at top)
    pressure = df[df["condition"] == "multi_turn_pressure"]
    model_order = (
        pressure.groupby("model_short")["j_constraint_adherence"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    pivot = df.pivot_table(
        values="j_constraint_adherence",
        index="model_short",
        columns="condition",
        aggfunc="mean",
    )
    pivot = pivot[[c for c in CONDITION_ORDER if c in pivot.columns]]
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    # Clean condition labels (single-line for heatmap)
    col_labels = {
        "single_shot": "Single-shot",
        "multi_turn_neutral": "Neutral",
        "multi_turn_pressure": "Pressure",
        "checkpointed_pressure": "Checkpointed",
    }
    pivot.columns = [col_labels.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=4, linewidths=0.5, ax=ax,
    )
    ax.set_title("Mean Constraint Adherence by Model and Condition")
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# --- New figures for priority list additions ---


def fig_surface_fidelity_gap(df: pd.DataFrame, output_path: Path) -> None:
    """Surface fidelity gap by condition and model.

    Shows the growing divergence between surface alignment (fidelity)
    and actual adherence (constraints + alternatives) under pressure.
    """
    from drift_bench.analysis.probes import compute_surface_fidelity_gap

    gap_df = compute_surface_fidelity_gap(df)
    if gap_df.empty:
        return

    gap_df["model_short"] = gap_df["model_id"].map(_short_model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Surface gap by condition (aggregated)
    ax = axes[0]
    cond_gap = (
        gap_df.groupby("condition")["surface_gap"]
        .agg(["mean", "std", "count"])
        .reindex(CONDITION_ORDER)
        .dropna()
    )
    x = range(len(cond_gap))
    bars = ax.bar(
        x, cond_gap["mean"],
        yerr=cond_gap["std"] / np.sqrt(cond_gap["count"]),
        capsize=4, color=sns.color_palette("muted", len(cond_gap)),
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in cond_gap.index], fontsize=8)
    ax.set_ylabel("Surface Fidelity Gap")
    ax.set_title("(a) Surface Gap by Condition")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Panel B: Surface gap by model under pressure
    ax = axes[1]
    pressure = gap_df[gap_df["condition"] == "multi_turn_pressure"]
    if not pressure.empty:
        model_gap = (
            pressure.groupby("model_short")["surface_gap"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
        x = range(len(model_gap))
        ax.bar(
            x, model_gap["mean"],
            yerr=model_gap["std"] / np.sqrt(model_gap["count"]),
            capsize=4, color=sns.color_palette("Set2", len(model_gap)),
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_gap.index, fontsize=8, rotation=15)
    ax.set_ylabel("Surface Fidelity Gap")
    ax.set_title("(b) Surface Gap by Model (Pressure)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fig_complexity_vs_tokens(scatter_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot: complexity vs token count, colored by condition.

    Shows that complexity inflation is not just more words.
    """
    if scatter_df.empty or "token_count" not in scatter_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    palette = {
        "single_shot": "#2ecc71",
        "multi_turn_neutral": "#3498db",
        "multi_turn_pressure": "#e74c3c",
        "checkpointed_pressure": "#f39c12",
    }

    for cond in CONDITION_ORDER:
        subset = scatter_df[scatter_df["condition"] == cond]
        if subset.empty:
            continue
        ax.scatter(
            subset["token_count"],
            subset["complexity"],
            label=CONDITION_LABELS.get(cond, cond).replace("\n", " "),
            color=palette.get(cond, "gray"),
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    ax.set_xlabel("Token Count (final response)")
    ax.set_ylabel("Complexity Score (0-4)")
    ax.set_title("Complexity vs. Verbosity by Condition")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(bootstrap_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of inter-rater reliability (kappa and alpha) with CIs."""
    if bootstrap_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    metrics = bootstrap_df["metric"].values
    x = np.arange(len(metrics))
    width = 0.35

    # Panel A: Cohen's kappa
    ax = axes[0]
    ax.bar(
        x, bootstrap_df["kappa"],
        yerr=[
            bootstrap_df["kappa"] - bootstrap_df["kappa_ci_lower"],
            bootstrap_df["kappa_ci_upper"] - bootstrap_df["kappa"],
        ],
        width=width, capsize=4, color="#3498db", edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylabel("Quadratic-weighted Kappa")
    ax.set_title("(a) Cohen's Kappa")
    ax.axhline(y=0.6, color="red", linestyle="--", linewidth=0.5, label="Substantial (0.6)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    # Panel B: Krippendorff's alpha
    ax = axes[1]
    ax.bar(
        x, bootstrap_df["alpha"],
        yerr=[
            bootstrap_df["alpha"] - bootstrap_df["alpha_ci_lower"],
            bootstrap_df["alpha_ci_upper"] - bootstrap_df["alpha"],
        ],
        width=width, capsize=4, color="#e67e22", edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylabel("Krippendorff's Alpha (interval)")
    ax.set_title("(b) Krippendorff's Alpha")
    ax.axhline(y=0.667, color="red", linestyle="--", linewidth=0.5, label="Acceptable (0.667)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fig_knows_but_violates(kbv_summary: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart showing knows-but-violates rates by model and condition."""
    if kbv_summary.empty:
        return

    # Resolve column names
    cond_col = "condition_score" if "condition_score" in kbv_summary.columns else "condition"
    model_col = "model_id_score" if "model_id_score" in kbv_summary.columns else "model_id"

    kbv_summary = kbv_summary.copy()
    kbv_summary["model_short"] = kbv_summary[model_col].map(_short_model)

    # Filter to multi-turn conditions only
    mt_conditions = ["multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"]
    plot_df = kbv_summary[kbv_summary[cond_col].isin(mt_conditions)]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    models = sorted(plot_df["model_short"].unique())
    x = np.arange(len(models))
    width = 0.25

    for i, cond in enumerate(mt_conditions):
        cond_data = plot_df[plot_df[cond_col] == cond]
        rates = []
        for m in models:
            row = cond_data[cond_data["model_short"] == m]
            rates.append(row["kbv_rate"].values[0] if len(row) > 0 else 0)
        ax.bar(
            x + i * width, rates, width,
            label=CONDITION_LABELS.get(cond, cond).replace("\n", " "),
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title("Knows-But-Violates Rate by Model and Condition")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    aggregated_path: Path,
    output_dir: Path,
) -> None:
    """Generate all publication figures."""
    df = load_aggregated(aggregated_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_main_scores(df, output_dir / "fig_main_scores.pdf")
    fig_drift_classification(df, output_dir / "fig_drift_classification.pdf")
    fig_model_comparison(df, output_dir / "fig_model_comparison.pdf")
    fig_surface_fidelity_gap(df, output_dir / "fig_surface_gap.pdf")

    print(f"Figures saved to {output_dir}")
