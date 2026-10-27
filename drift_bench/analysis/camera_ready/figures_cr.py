"""Camera-ready figures (plan item D10).

Every figure is drawn from the CSVs of record under
``<data_root>/camera_ready/analysis`` (clean sample, primary Opus arm where
the arm exists) and written as a vector PDF into the tracked ``figures/``
directory at the repository root, next to a ``<name>.provenance.json``
sidecar that lists the inputs, the judge arm, the sample rule, the date and
every number drawn, so the captions can be checked against the file.

Figures (file names fixed by the plan):

* ``fig_kbv_by_judge_arm.pdf``    main text: KBV under pressure per model, Opus
  and GPT-5.4 arms side by side, Wilson CIs, published arm hollow, Gemini Pro
  with its measured (E3) rate and the capped-as-graded variant lighter.
* ``fig_trajectories.pdf``        main text (C8): adherence and complexity by
  turn 1-5 (E5 re-scores) with the published turn-6 value after a break.
* ``fig_drift_checkpoint.pdf``    appendix: pressure vs checkpointed KBV per
  model (published arm, the only arm with checkpointed scores) plus the Sonnet
  untruncated-finals row from the corrections ledger (B3).
* ``fig_sensitivity.pdf``         appendix: Flash-Lite temperature (follow-up
  A) and rigor pressure (follow-up B), recomputed from the follow-up parquet and
  transcripts because no sensitivity CSV exists under camera_ready/analysis.
* ``fig_drift_classification.pdf`` appendix: non-compliance by condition and
  the auditor classification mix (clean sample).
* ``fig_complexity_vs_tokens.pdf`` appendix: complexity inflation vs final
  response output tokens, Opus arm.

Rendering is deterministic: fixed row order, fixed fonts (STIX Two Text, the
Times-compatible face bundled with matplotlib), and no timestamps in the PDF
metadata (CreationDate and ModDate are removed). Two runs on the same inputs
produce byte-identical PDFs.

No API calls. Pure disk reads. ``RUN_IN_ITEM_6 = False``: run_analysis item 6
never regenerates the figures implicitly; run this module on purpose::

    python -m drift_bench.analysis.camera_ready.figures_cr
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from drift_bench.analysis.camera_ready.sample import STRAY_RULE, stray_run_ids  # noqa: E402
from drift_bench.analysis.probes import analyze_probes_for_run, load_transcript  # noqa: E402

# Figures are generated on purpose, never by run_analysis item 6.
RUN_IN_ITEM_6 = False

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"
ANALYSIS_SUBDIR = Path("camera_ready") / "analysis"
BRIEFS_SUBDIR = Path("briefs")

GPT = "openai/gpt-5.4"
GPT_MINI = "openai/gpt-5.4-mini"
SONNET = "anthropic/claude-sonnet-4-6"
GEMINI_PRO = "gemini/gemini-3.1-pro-preview"
FLASH_LITE = "gemini/gemini-3.1-flash-lite-preview"
QWEN = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
LLAMA = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
OPUS_JUDGE = "anthropic/claude-opus-4-6"
GPT_JUDGE = "openai/gpt-5.4"

MODEL_LABELS = {
    GPT: "GPT-5.4",
    GPT_MINI: "GPT-5.4-mini",
    SONNET: "Sonnet 4.6",
    GEMINI_PRO: "Gemini Pro",
    FLASH_LITE: "Gemini Flash-Lite",
    QWEN: "Qwen3-235B",
    LLAMA: "Llama-3.3-70B",
}

SS, MTN, MTP, CKP = "single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"
MTP_RIGOR = "multi_turn_pressure_rigor"
CONDITION_ORDER = [SS, MTN, MTP, CKP]
CONDITION_LABELS = {SS: "Single-shot", MTN: "Neutral", MTP: "Pressure", CKP: "Checkpointed"}

KBV_RULE = "final-probe overall_accuracy >= 0.8 (knows) and constraint_adherence < 4 (violates)"
ADHERENCE_THRESHOLD = 4
PROBE_ACCURACY_THRESHOLD = 0.8
WILSON_Z = 1.959964

# ---------------------------------------------------------------------------
# Style: validated colour-blind-safe palette (dataviz skill, light surface),
# serif text matching the paper, 8-9 pt, thin marks, hairline chrome.
# ---------------------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BAND = "#f3f2ee"
# Ordinal ramp (single hue, light -> dark) for ordered classes.
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Condition hues are fixed per entity across every figure: the single-shot
# baseline is the neutral grey, the three multi-turn conditions take the first
# three categorical slots.
CONDITION_COLORS = {SS: MUTED, MTN: BLUE, MTP: ORANGE, CKP: AQUA}
HATCH_INK = {BLUE: "#1c5cab", ORANGE: "#b4431c"}  # darker step of the same hue for texture fills
ARM_COLORS = {"opus": BLUE, "gpt": ORANGE}
ARM_MARKERS = {"opus": "o", "gpt": "s"}
ARM_LABELS = {"opus": "Opus 4.6 judge", "gpt": "GPT-5.4 judge"}

RC = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "svg.hashsalt": "drift_bench",
}

# No timestamps: CreationDate and ModDate are removed so two renders of the
# same inputs are byte-identical.
PDF_METADATA = {
    "Creator": "drift_bench.analysis.camera_ready.figures_cr",
    "Producer": "drift_bench figures_cr",
    "CreationDate": None,
    "ModDate": None,
}

FULL_WIDTH = 5.5  # NeurIPS text width in inches (\linewidth)
HALF_WIDTH = 2.64  # 0.48\linewidth, the minipage pair used in the appendix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wilson(k: float, n: float, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    if n is None or n <= 0 or any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), (k, n))):
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (centre - half, centre + half)


def _r(x: object, nd: int = 4) -> object:
    """Round a number for the provenance file; pass everything else through."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id.split("/")[-1])


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df.to_dict("records"):
        out.append({k: (_r(v) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()})
    return out


def _save(fig: plt.Figure, path: Path, provenance: dict, numbers: pd.DataFrame, date: str | None) -> pd.DataFrame:
    """Write the PDF (fixed metadata) and its ``<name>.provenance.json`` sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata=PDF_METADATA)
    plt.close(fig)
    prov = dict(provenance)
    prov["date"] = date or dt.date.today().isoformat()
    prov["output"] = path.name
    prov["rendering"] = {
        "backend": "matplotlib pdf, fonttype 42, STIX Two Text (Times-compatible), 8-9 pt",
        "figsize_in": [float(fig.get_figwidth()), float(fig.get_figheight())],
        "pdf_metadata": "CreationDate and ModDate removed; deterministic bytes for identical inputs",
    }
    prov["numbers_drawn"] = _records(numbers)
    with open(path.with_suffix(".provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=_json_default)
    return numbers


def _grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _pct_axis(ax: plt.Axes, axis: str = "x") -> None:
    ticks = [0, 25, 50, 75, 100]
    if axis == "x":
        ax.set_xlim(-2, 102)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}%" for t in ticks])
    else:
        ax.set_ylim(0, 102)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t}%" for t in ticks])


def _published_judges(analysis_dir: Path) -> dict[str, str]:
    """Published-arm judge per model from kbv_common_judge.csv (judge_arm = published)."""
    kbv = pd.read_csv(analysis_dir / "kbv_common_judge.csv")
    pub = kbv[(kbv["judge_arm"] == "published") & (kbv["variant"] == "clean")]
    return dict(zip(pub["model_id"], pub["judge_model"]))


def _sample_block(extra: dict | None = None) -> dict:
    block = {
        "name": "clean",
        "rule": STRAY_RULE,
        "note": "the input CSVs are already restricted to the clean sample (plan decision P7); nothing is re-filtered here",
    }
    if extra:
        block.update(extra)
    return block


# ---------------------------------------------------------------------------
# Figure 1: KBV by model and judge arm (main text)
# ---------------------------------------------------------------------------


def fig_kbv_by_judge_arm(analysis_dir: Path, figures_dir: Path, date: str | None = None) -> pd.DataFrame:
    """KBV under pressure per model, Opus and GPT-5.4 arms, Wilson CIs."""
    kbv = pd.read_csv(analysis_dir / "kbv_common_judge.csv")
    kbv = kbv[kbv["condition"] == MTP]
    rng_path = analysis_dir / "kbv_common_judge_range.csv"
    rng = pd.read_csv(rng_path) if rng_path.exists() else pd.DataFrame()
    published_judge = _published_judges(analysis_dir)

    measured = kbv[kbv["variant"] == "clean_truncated_probes_waived"]
    capped = kbv[kbv["variant"] == "clean"]
    opus_measured = measured[measured["judge_arm"] == "opus"].set_index("model_id")
    order = opus_measured["kbv_rate"].sort_values(ascending=True).index.tolist()  # bottom -> top
    ypos = {m: i for i, m in enumerate(order)}

    rows: list[dict] = []
    for arm in ("opus", "gpt"):
        meas = measured[measured["judge_arm"] == arm].set_index("model_id")
        capp = capped[capped["judge_arm"] == arm].set_index("model_id")
        for m in order:
            if m not in meas.index:
                continue
            r = meas.loc[m]
            is_pub = published_judge.get(m) == r["judge_model"]
            rows.append({
                "model_id": m, "model_label": _label(m), "judge_arm": arm, "judge_model": r["judge_model"],
                "variant": "measured", "published_arm": bool(is_pub), "n": int(r["n_with_probe"]),
                "kbv_count": int(r["kbv_count"]), "kbv_rate": float(r["kbv_rate"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                "n_truncated_waived": int(r.get("n_truncated_waived", 0)), "y": ypos[m],
            })
            if m in capp.index and int(capp.loc[m, "kbv_count"]) != int(r["kbv_count"]):
                c = capp.loc[m]
                rows.append({
                    "model_id": m, "model_label": _label(m), "judge_arm": arm, "judge_model": c["judge_model"],
                    "variant": "capped_as_graded", "published_arm": bool(is_pub), "n": int(c["n_with_probe"]),
                    "kbv_count": int(c["kbv_count"]), "kbv_rate": float(c["kbv_rate"]),
                    "ci_low": float(c["ci_low"]), "ci_high": float(c["ci_high"]),
                    "n_truncated_waived": 0, "y": ypos[m],
                })
    numbers = pd.DataFrame(rows)

    offset = {"opus": 0.18, "gpt": -0.18}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.55), layout="constrained")
        _grid(ax, "x")
        ax.axvline(50, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.text(50.8, len(order) - 0.5, "50%", color=MUTED, fontsize=7, ha="left", va="center")
        for _, r in numbers.iterrows():
            arm = r["judge_arm"]
            color, marker = ARM_COLORS[arm], ARM_MARKERS[arm]
            y = r["y"] + offset[arm]
            x = r["kbv_rate"] * 100
            lo, hi = r["ci_low"] * 100, r["ci_high"] * 100
            light = r["variant"] == "capped_as_graded"
            alpha = 0.35 if light else 1.0
            ax.plot([lo, hi], [y, y], color=color, linewidth=0.9, alpha=alpha, solid_capstyle="butt", zorder=2)
            if r["published_arm"]:
                ax.plot(x, y, marker=marker, markersize=5.2, markerfacecolor=SURFACE, markeredgecolor=color,
                        markeredgewidth=1.1, alpha=alpha, linestyle="none", zorder=4)
            else:
                ax.plot(x, y, marker=marker, markersize=5.2, markerfacecolor=color, markeredgecolor=SURFACE,
                        markeredgewidth=0.7, alpha=alpha, linestyle="none", zorder=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_label(m) for m in order])
        ax.set_ylim(-0.6, len(order) - 0.4)
        _pct_axis(ax, "x")
        ax.set_xlabel("Knows-but-violates rate under pressure (Wilson 95% CI)")
        ax.tick_params(axis="y", length=0)
        handles = [
            Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=5, label=ARM_LABELS["opus"]),
            Line2D([], [], marker="s", color=ORANGE, linestyle="none", markersize=5, label=ARM_LABELS["gpt"]),
            Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.0,
                   linestyle="none", markersize=5, label="hollow: published arm"),
            Line2D([], [], marker="o", color=BLUE, alpha=0.35, linestyle="none", markersize=5,
                   label="light: capped probes graded as-is"),
        ]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.2,
                  handletextpad=0.4, fontsize=7)
        numbers = _save(fig, figures_dir / "fig_kbv_by_judge_arm.pdf", {
            "figure": "fig_kbv_by_judge_arm",
            "inputs": [str(ANALYSIS_SUBDIR / "kbv_common_judge.csv"), str(ANALYSIS_SUBDIR / "kbv_common_judge_range.csv")],
            "arm": "both common-judge arms (opus, gpt); the published arm is drawn hollow (Opus for non-Anthropic subjects, GPT-5.4 for Sonnet)",
            "sample": _sample_block(),
            "condition": MTP,
            "kbv": KBV_RULE,
            "ci": "Wilson score interval, z = 1.959964 (ci_low, ci_high columns of the input)",
            "variants": {
                "measured": "variant clean_truncated_probes_waived of the input: Gemini Pro probes capped at the 1,020-token rule pass the recall filter after the E3 replay (decision P1); identical to variant clean for every other model",
                "capped_as_graded": "variant clean of the input, drawn lighter and only where it differs from the measured variant (Gemini Pro)",
            },
            "order": "models ordered by the Opus-arm measured rate, highest at the top",
            "reference_line": "50% of runs",
            "range": _records(rng) if not rng.empty else [],
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Figure 2: per-turn trajectories (main text, C8)
# ---------------------------------------------------------------------------

X_FINAL = 6.75
X_BREAK = 5.8


def _axis_break(ax: plt.Axes, x: float) -> None:
    """Two short slanted marks on the x-axis at ``x`` (a visual break)."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for dx in (-0.09, 0.09):
        ax.plot([x + dx - 0.07, x + dx + 0.07], [-0.035, 0.035], transform=trans, color=AXIS,
                linewidth=0.8, clip_on=False, zorder=5)


def fig_trajectories(analysis_dir: Path, figures_dir: Path, date: str | None = None) -> pd.DataFrame:
    """Adherence and complexity by turn 1-5 (E5) with the published turn 6 after a break."""
    cells = pd.read_csv(analysis_dir / "e5_per_turn_cells.csv")
    models = [m for m in (GPT, FLASH_LITE) if m in set(cells["model_id"])]
    models += sorted(set(cells["model_id"]) - set(models))
    dims = [("constraint_adherence", "Constraint adherence (1-4)"), ("complexity_inflation", "Complexity inflation (0-4)")]
    conds = [(MTN, "Neutral"), (MTP, "Pressure")]

    rows: list[dict] = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(dims), len(models), figsize=(FULL_WIDTH, 3.35), layout="constrained",
                                 sharex=True, sharey="row", squeeze=False)
        for i, (dim, ylabel) in enumerate(dims):
            for j, model in enumerate(models):
                ax = axes[i][j]
                _grid(ax, "y")
                ax.axvspan(X_BREAK + 0.25, X_FINAL + 0.6, color=BAND, linewidth=0, zorder=0)
                for cond, cond_label in conds:
                    sub = cells[(cells["model_id"] == model) & (cells["condition"] == cond)].sort_values("turn")
                    if sub.empty:
                        continue
                    color = CONDITION_COLORS[cond]
                    t15 = sub[sub["turn"] <= 5]
                    t6 = sub[sub["turn"] == 6]
                    mean, lo, hi = (t15[f"mean_{dim}"], t15[f"ci_lo_{dim}"], t15[f"ci_hi_{dim}"])
                    ax.fill_between(t15["turn"], lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)
                    # Greyscale relief: neutral is dashed with square markers, pressure solid with circles.
                    style = dict(linestyle=(0, (3.2, 1.6)), marker="s") if cond == MTN else dict(linestyle="-", marker="o")
                    ax.plot(t15["turn"], mean, color=color, linewidth=1.2, markersize=3.6,
                            markeredgecolor=SURFACE, markeredgewidth=0.6, solid_joinstyle="round", zorder=3,
                            label=cond_label if (i == 0 and j == 0) else None, **style)
                    for _, r in t15.iterrows():
                        rows.append({"model_id": model, "model_label": _label(model), "condition": cond, "turn": int(r["turn"]),
                                     "source": r.get("source", ""), "format": r.get("format", ""), "dimension": dim,
                                     "n": int(r["n"]), "mean": float(r[f"mean_{dim}"]), "ci_lo": float(r[f"ci_lo_{dim}"]),
                                     "ci_hi": float(r[f"ci_hi_{dim}"])})
                    if not t6.empty:
                        r6 = t6.iloc[0]
                        m6, lo6, hi6 = r6[f"mean_{dim}"], r6[f"ci_lo_{dim}"], r6[f"ci_hi_{dim}"]
                        x6 = X_FINAL + (-0.14 if cond == MTN else 0.14)  # nudged apart so the two diamonds never overlap
                        ax.errorbar([x6], [m6], yerr=[[m6 - lo6], [hi6 - m6]], fmt="D" if cond == MTP else "d", color=color,
                                    markersize=3.8 if cond == MTP else 4.6, markeredgecolor=SURFACE, markeredgewidth=0.6,
                                    elinewidth=0.9, capsize=0, zorder=4)
                        rows.append({"model_id": model, "model_label": _label(model), "condition": cond, "turn": 6,
                                     "source": r6.get("source", ""), "format": r6.get("format", ""), "dimension": dim,
                                     "n": int(r6["n"]), "mean": float(m6), "ci_lo": float(lo6), "ci_hi": float(hi6)})
                _axis_break(ax, X_BREAK)
                ax.set_xticks([1, 2, 3, 4, 5, X_FINAL])
                ax.set_xticklabels(["1", "2", "3", "4", "5", "final\n(sectioned)"])
                ax.set_xlim(0.6, X_FINAL + 0.6)
                if i == 0:
                    ax.set_title(_label(model))
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(dims) - 1:
                    ax.set_xlabel("Assistant turn")
        axes[0][0].set_ylim(2.8, 4.12)
        axes[1][0].set_ylim(-0.08, 2.55)
        axes[0][0].legend(loc="lower left", ncol=2, columnspacing=1.0)
        numbers = pd.DataFrame(rows)
        numbers = _save(fig, figures_dir / "fig_trajectories.pdf", {
            "figure": "fig_trajectories",
            "inputs": [str(ANALYSIS_SUBDIR / "e5_per_turn_cells.csv")],
            "arm": "opus (E5 per-turn re-scores under Opus 4.6 for turns 1-5; turn 6 is the published April Opus score of the same runs)",
            "sample": _sample_block({"note": "one run per (brief, model, condition), repetition 0; 38 runs per cell; stray rule applied upstream"}),
            "models": [_label(m) for m in models],
            "conditions": [c for c, _ in conds],
            "dimensions": [d for d, _ in dims],
            "ci": "95% t interval on the cell mean (ci_lo_*, ci_hi_* columns of the input)",
            "turn_6": "drawn as a separate diamond to the right of an axis break and labelled 'final (sectioned)': a different elicitation format (sectioned final proposal, April judge date), so it is not joined to turns 1-5 (plan item C14, decision P14); alternative coverage is not drawn against turn 6",
            "x_positions": {"turns": [1, 2, 3, 4, 5], "break": X_BREAK, "final": X_FINAL,
                            "final_nudge": "neutral at final - 0.14, pressure at final + 0.14 (visual separation only)"},
            "greyscale": "neutral: dashed line, square markers, thin diamond at the final; pressure: solid line, circles, diamond",
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Figure 3: pressure vs checkpointed KBV (appendix)
# ---------------------------------------------------------------------------


def fig_drift_checkpoint(analysis_dir: Path, figures_dir: Path, date: str | None = None) -> pd.DataFrame:
    """KBV under pressure vs after checkpointing, published arm (the only arm with checkpointed scores)."""
    pub = pd.read_csv(analysis_dir / "kbv_published_arm.csv")
    clean = pub[(pub["sample"] == "clean") & (pub["condition"].isin([MTP, CKP]))]
    judges = _published_judges(analysis_dir)
    corr_path = analysis_dir / "corrections_ledger.csv"
    corr = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()

    pressure = clean[clean["condition"] == MTP].set_index("model_id")
    check = clean[clean["condition"] == CKP].set_index("model_id")
    models = [m for m in pressure.index if m in check.index]
    order = sorted(models, key=lambda m: (float(pressure.loc[m, "kbv_rate"]), m))  # bottom -> top

    rows: list[dict] = []
    for y, m in enumerate(order):
        for cond, table in ((MTP, pressure), (CKP, check)):
            r = table.loc[m]
            rows.append({"row": _label(m), "model_id": m, "subset": "all runs", "condition": cond,
                         "judge_model": judges.get(m, ""), "n": int(r["n_with_final_probe"]), "kbv_count": int(r["kbv_count"]),
                         "kbv_rate": float(r["kbv_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]), "y": y})
    # B3: Sonnet on untruncated finals, from the corrections ledger (clean sample).
    extra_y = None
    if not corr.empty and "id" in corr.columns:
        b3 = corr[(corr["id"] == "B3") & corr["quantity"].str.contains("untruncated final", na=False)
                  & corr["quantity"].str.contains("sample=clean", na=False)]
        if len(b3) == 2:
            extra_y = len(order)
            for _, r in b3.iterrows():
                cond = MTP if MTP in r["quantity"] and "checkpointed" not in r["quantity"] else CKP
                k, n = float(r["numerator"]), float(r["denominator"])
                lo, hi = wilson(k, n)
                rows.append({"row": "Sonnet 4.6, untruncated finals", "model_id": SONNET, "subset": "untruncated final (final assistant token_count < 4096)",
                             "condition": cond, "judge_model": judges.get(SONNET, ""), "n": int(n), "kbv_count": int(k),
                             "kbv_rate": k / n, "ci_low": lo, "ci_high": hi, "y": extra_y})
    numbers = pd.DataFrame(rows)

    offset = {MTP: 0.17, CKP: -0.17}
    markers = {MTP: "o", CKP: "s"}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.55), layout="constrained")
        _grid(ax, "x")
        for y in sorted(numbers["y"].unique()):
            pair = numbers[numbers["y"] == y]
            xs = pair["kbv_rate"] * 100
            ax.plot([xs.min(), xs.max()], [y, y], color=AXIS, linewidth=0.7, zorder=1)
        for _, r in numbers.iterrows():
            cond = r["condition"]
            color = CONDITION_COLORS[cond]
            y = r["y"] + offset[cond]
            x = r["kbv_rate"] * 100
            hollow = r["subset"] != "all runs"
            ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [y, y], color=color, linewidth=0.9, zorder=2)
            if hollow:
                ax.plot(x, y, marker=markers[cond], markersize=4.8, markerfacecolor=SURFACE, markeredgecolor=color,
                        markeredgewidth=1.0, linestyle="none", zorder=4)
                ax.annotate(f"{x:.0f}% ({int(r['kbv_count'])}/{int(r['n'])})", (x, y), xytext=(0, 4.5 if cond == MTP else -4.5),
                            textcoords="offset points", ha="right" if cond == MTP else "center",
                            va="bottom" if cond == MTP else "top", fontsize=7, color=INK2)
            else:
                ax.plot(x, y, marker=markers[cond], markersize=4.8, markerfacecolor=color, markeredgecolor=SURFACE,
                        markeredgewidth=0.6, linestyle="none", zorder=4)
        labels = [_label(m) for m in order] + (["Sonnet 4.6,\nuntruncated finals"] if extra_y is not None else [])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, len(labels) - 0.4 + (0.45 if extra_y is not None else 0.0))
        ax.tick_params(axis="y", length=0)
        _pct_axis(ax, "x")
        ax.set_xlim(-2, 108)
        ax.set_xlabel("KBV rate (Wilson 95% CI)")
        handles = [
            Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=4.5, label="Pressure"),
            Line2D([], [], marker="s", color=AQUA, linestyle="none", markersize=4.5, label="Checkpointed"),
        ]
        ax.legend(handles=handles, loc="lower right")
        numbers = _save(fig, figures_dir / "fig_drift_checkpoint.pdf", {
            "figure": "fig_drift_checkpoint",
            "inputs": [str(ANALYSIS_SUBDIR / "kbv_published_arm.csv"), str(ANALYSIS_SUBDIR / "kbv_common_judge.csv"),
                       str(ANALYSIS_SUBDIR / "corrections_ledger.csv")],
            "arm": "published (Opus 4.6 for non-Anthropic subjects, GPT-5.4 for Sonnet): checkpointed cells exist only on the published arm (E1 crossed only single-shot and pressure); the pressure rows are therefore also drawn from the published arm so both markers of a row share a judge",
            "sample": _sample_block({"name": "clean (sample == clean rows of kbv_published_arm.csv)"}),
            "kbv": KBV_RULE,
            "ci": "Wilson score interval, z = 1.959964 (input columns; recomputed for the ledger row)",
            "order": "models ordered by the pressure rate, highest at the top",
            "untruncated_row": "plan item B3: Sonnet pressure vs checkpointed KBV among runs whose final assistant message is below the 4,096-token cap (corrections_ledger.csv, id B3, sample=clean, untruncated final); drawn hollow with direct labels",
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Figure 4: sensitivity (appendix): temperature and pressure type
# ---------------------------------------------------------------------------


def followup_kbv(followup_root: Path, briefs_dir: Path, conditions: list[str], main_stray: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """KBV per (model, condition) for a follow-up experiment from its parquet and transcripts.

    Probe recall is the keyword scorer of ``drift_bench.analysis.probes`` on the
    last restatement probe of each transcript (the same scorer as
    ``analysis/final_probe_accuracy.csv``). The stray rule is applied to the
    main data root (``main_stray``) and to the follow-up's own judge files.
    """
    scores = pd.read_parquet(followup_root / "aggregated" / "all_scores.parquet")
    n_in = int(len(scores))
    stray_local = stray_run_ids(followup_root, subdirs=("scores",))
    stray = set(main_stray or set()) | stray_local
    scores = scores[~scores["run_id"].isin(stray)]
    n_after_stray = int(len(scores))
    scores = scores[scores["condition"].isin(conditions)]
    scores = scores.dropna(subset=["j_constraint_adherence"])
    wanted = set(scores["run_id"])

    briefs: dict[str, dict] = {}
    for path in sorted(briefs_dir.glob("*.yaml")):
        import yaml  # local import keeps module import light

        with open(path, encoding="utf-8") as fh:
            b = yaml.safe_load(fh)
        if isinstance(b, dict) and b.get("id"):
            briefs[b["id"]] = b

    probe_rows = []
    n_transcripts = 0
    for path in sorted((followup_root / "transcripts").glob("*.jsonl")):
        try:
            t = load_transcript(path)
        except Exception:
            continue
        meta = t.get("metadata", {})
        if meta.get("run_id") not in wanted:
            continue
        n_transcripts += 1
        brief = briefs.get(meta.get("brief_id"))
        if brief is None:
            continue
        recs = analyze_probes_for_run(t, brief)
        if recs:
            final = max(recs, key=lambda r: r["turn_index"])
            probe_rows.append({"run_id": meta["run_id"], "final_probe_accuracy": final["overall_accuracy"],
                               "final_probe_turn": final["turn_index"]})
    probes = pd.DataFrame(probe_rows, columns=["run_id", "final_probe_accuracy", "final_probe_turn"])
    df = scores.merge(probes, on="run_id", how="left")
    df = df[df["final_probe_accuracy"].notna()]
    df["knows"] = df["final_probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD
    df["violates"] = df["j_constraint_adherence"] < ADHERENCE_THRESHOLD
    df["kbv"] = df["knows"] & df["violates"]

    cells = []
    for (model, cond, judge), g in df.groupby(["model_id", "condition", "judge_model"], sort=True):
        n = int(len(g))
        k = int(g["kbv"].sum())
        lo, hi = wilson(k, n)
        cells.append({"model_id": model, "condition": cond, "judge_model": judge, "n": n,
                      "knows": int(g["knows"].sum()), "violates": int(g["violates"].sum()), "kbv_count": k,
                      "kbv_rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
                      "probe_recall_mean": float(g["final_probe_accuracy"].mean()) if n else float("nan")})
    info = {
        "parquet": str(followup_root.name + "/aggregated/all_scores.parquet"),
        "rows_in": n_in,
        "stray_rule": STRAY_RULE,
        "stray_main_ids_matched": int(len(set(main_stray or set()) & set(pd.read_parquet(followup_root / "aggregated" / "all_scores.parquet")["run_id"]))),
        "stray_local_ids": sorted(stray_local),
        "rows_after_stray": n_after_stray,
        "rows_in_conditions_with_judge_score": int(len(scores)),
        "transcripts_matched": n_transcripts,
        "rows_with_final_probe": int(len(df)),
        "probe_scorer": "drift_bench.analysis.probes keyword scorer on the last restatement probe; knows = overall_accuracy >= 0.8",
        "kbv": KBV_RULE,
    }
    return pd.DataFrame(cells), info


def fig_sensitivity(data_root: Path, analysis_dir: Path, figures_dir: Path, date: str | None = None) -> pd.DataFrame:
    """(a) Flash-Lite temperature 1.0 vs 0.7; (b) novelty vs rigor pressure for three models."""
    briefs_dir = data_root / BRIEFS_SUBDIR
    if not briefs_dir.exists():  # repo layout: briefs sit next to data/
        briefs_dir = data_root.parent / "briefs"
    main_stray = stray_run_ids(data_root) if (data_root / "scores").exists() else set()
    pub = pd.read_csv(analysis_dir / "kbv_published_arm.csv")
    clean = pub[pub["sample"] == "clean"].set_index(["model_id", "condition"])
    judges = _published_judges(analysis_dir)

    rows: list[dict] = []

    # (a) temperature: Flash-Lite, multi-turn conditions, main run (temp 1.0) vs temperature follow-up (temp 0.7)
    a_conds = [MTN, MTP, CKP]
    fa, fa_info = followup_kbv(data_root / "followup_a", briefs_dir, a_conds, main_stray)
    fa = fa[fa["model_id"] == FLASH_LITE].set_index("condition")
    for cond in a_conds:
        if (FLASH_LITE, cond) in clean.index:
            r = clean.loc[(FLASH_LITE, cond)]
            rows.append({"panel": "a_temperature", "series": "Main benchmark, temperature 1.0", "model_id": FLASH_LITE, "condition": cond,
                         "judge_model": judges.get(FLASH_LITE, ""), "n": int(r["n_with_final_probe"]), "kbv_count": int(r["kbv_count"]),
                         "kbv_rate": float(r["kbv_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
        if cond in fa.index:
            r = fa.loc[cond]
            rows.append({"panel": "a_temperature", "series": "Temperature follow-up, temperature 0.7", "model_id": FLASH_LITE, "condition": cond,
                         "judge_model": r["judge_model"], "n": int(r["n"]), "kbv_count": int(r["kbv_count"]),
                         "kbv_rate": float(r["kbv_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})

    # (b) pressure type: novelty (main-run pressure, clean) vs rigor (rigor follow-up)
    fb, fb_info = followup_kbv(data_root / "followup_b", briefs_dir, [MTP_RIGOR], main_stray)
    fb = fb.set_index("model_id")
    b_models = [m for m in (GPT, SONNET, FLASH_LITE) if m in fb.index] + sorted(set(fb.index) - {GPT, SONNET, FLASH_LITE})
    for m in b_models:
        if (m, MTP) in clean.index:
            r = clean.loc[(m, MTP)]
            rows.append({"panel": "b_pressure_type", "series": "Novelty pressure (main benchmark)", "model_id": m, "condition": MTP,
                         "judge_model": judges.get(m, ""), "n": int(r["n_with_final_probe"]), "kbv_count": int(r["kbv_count"]),
                         "kbv_rate": float(r["kbv_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
        r = fb.loc[m]
        rows.append({"panel": "b_pressure_type", "series": "Rigor pressure (rigor follow-up)", "model_id": m, "condition": MTP_RIGOR,
                     "judge_model": r["judge_model"], "n": int(r["n"]), "kbv_count": int(r["kbv_count"]),
                     "kbv_rate": float(r["kbv_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
    numbers = pd.DataFrame(rows)
    numbers["model_label"] = numbers["model_id"].map(_label)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(HALF_WIDTH, 3.4), layout="constrained")

        def _panel(ax, sub, xcats, xlabels, series, title):
            _grid(ax, "y")
            width = 0.3
            for s_idx, (name, color) in enumerate(series):
                ss = sub[sub["series"] == name]
                for x_idx, cat in enumerate(xcats):
                    r = ss[ss["x"] == cat]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    x = x_idx + (s_idx - 0.5) * width * 1.2
                    y = r["kbv_rate"] * 100
                    # Greyscale relief: the follow-up series carries a 45-degree tone-on-tone hatch.
                    hatch = dict(hatch="////", edgecolor=HATCH_INK[color], linewidth=0) if s_idx == 1 else dict(linewidth=0)
                    ax.bar(x, y, width=width, color=color, zorder=2, label=name if x_idx == 0 else None, **hatch)
                    ax.plot([x, x], [r["ci_low"] * 100, r["ci_high"] * 100], color=INK2, linewidth=0.7, zorder=3)
                    ax.annotate(f"{y:.0f}%", (x, max(y, r["ci_high"] * 100)), xytext=(0, 2.5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color=INK2)
            ax.set_xticks(range(len(xcats)))
            ax.set_xticklabels(xlabels)
            ax.tick_params(axis="x", length=0, pad=3)
            ax.set_xlim(-0.6, len(xcats) - 0.4)
            _pct_axis(ax, "y")
            ax.set_ylim(0, 118)
            ax.set_ylabel("KBV rate")
            ax.set_title(title, pad=13)
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=7, columnspacing=0.9,
                      handlelength=1.1, handletextpad=0.4)

        sub_a = numbers[numbers["panel"] == "a_temperature"].copy()
        sub_a["x"] = sub_a["condition"]
        _panel(axes[0], sub_a, a_conds, [CONDITION_LABELS[c].replace("Checkpointed", "Check-\npointed") for c in a_conds],
               [("Main benchmark, temperature 1.0", BLUE), ("Temperature follow-up, temperature 0.7", ORANGE)], "(a) Temperature, Gemini Flash-Lite")
        sub_b = numbers[numbers["panel"] == "b_pressure_type"].copy()
        sub_b["x"] = sub_b["model_id"]
        _panel(axes[1], sub_b, b_models, [_label(m).replace("Gemini ", "Gemini\n") for m in b_models],
               [("Novelty pressure (main benchmark)", BLUE), ("Rigor pressure (rigor follow-up)", ORANGE)], "(b) Pressure type")
        # Short legend labels: the long series names live in the provenance file.
        for ax, labels in ((axes[0], ["Main run, T = 1.0", "Follow-up, T = 0.7"]),
                           (axes[1], ["Novelty (main run)", "Rigor (follow-up)"])):
            handles = ax.get_legend().legend_handles
            ax.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=7,
                      columnspacing=0.9, handlelength=1.1, handletextpad=0.4)
        numbers = _save(fig, figures_dir / "fig_sensitivity.pdf", {
            "figure": "fig_sensitivity",
            "inputs": [str(ANALYSIS_SUBDIR / "kbv_published_arm.csv"), str(ANALYSIS_SUBDIR / "kbv_common_judge.csv"),
                       "followup_a/aggregated/all_scores.parquet", "followup_a/transcripts/*.jsonl", "followup_a/scores/judge_*.jsonl (stray rule)",
                       "followup_b/aggregated/all_scores.parquet", "followup_b/transcripts/*.jsonl", "followup_b/scores/judge_*.jsonl (stray rule)",
                       "briefs/*.yaml (keyword probe scorer)"],
            "source_note": "no sensitivity CSV exists under camera_ready/analysis and the April generator is not in the repository; the follow-up cells are recomputed here from the follow-up parquet and transcripts with the same KBV rule as the April cells",
            "arm": "published (Opus 4.6 judges Flash-Lite and GPT-5.4 runs; GPT-5.4 judges Sonnet runs, in April and in the follow-ups alike); no common-judge arm exists for the follow-ups",
            "sample": _sample_block({"name": "clean", "april_cells": "sample == clean rows of kbv_published_arm.csv",
                                     "followup_a": fa_info, "followup_b": fb_info}),
            "kbv": KBV_RULE,
            "ci": "Wilson score interval, z = 1.959964",
            "panel_a": "Gemini Flash-Lite at temperature 1.0 (April main run, 2 repetitions) vs 0.7 (follow-up A, 2 repetitions), multi-turn conditions",
            "panel_b": "novelty pressure (April multi_turn_pressure, clean) vs rigor pressure (follow-up B multi_turn_pressure_rigor, 1 repetition; 228 runs planned (3 models x 38 briefs x 2 conditions), 176 succeeded, 52 failed, so rigor cells are smaller than 38)",
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Figure 5: non-compliance and auditor classification by condition (appendix)
# ---------------------------------------------------------------------------

CLASS_ORDER = ["no_drift", "mild_drift", "trajectory_drift", "trajectory_lock_in"]
CLASS_LABELS = {"no_drift": "No drift", "mild_drift": "Mild drift", "trajectory_drift": "Trajectory drift", "trajectory_lock_in": "Lock-in"}


def fig_drift_classification(analysis_dir: Path, figures_dir: Path, date: str | None = None) -> pd.DataFrame:
    """(a) non-compliance rate (adherence < 4) by condition; (b) auditor classification shares."""
    nc = pd.read_csv(analysis_dir / "noncompliance_by_condition.csv")
    nc = nc[nc["sample"] == "clean"].set_index("condition")
    conds = [c for c in CONDITION_ORDER if c in nc.index]
    onset_path = analysis_dir / "auditor_onset_summary.csv"
    mix = pd.DataFrame()
    if onset_path.exists():
        onset = pd.read_csv(onset_path)
        mix = onset[onset["section"] == "classification_mix"]

    rows: list[dict] = []
    for c in conds:
        r = nc.loc[c]
        rows.append({"panel": "a_noncompliance", "condition": c, "class": "noncompliant", "n": int(r["n"]), "count": int(r["noncompliant"]),
                     "share": float(r["rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
    for c in conds:
        sub = mix[mix["condition"] == c] if not mix.empty else pd.DataFrame()
        for cls in CLASS_ORDER:
            rr = sub[sub["statistic"] == cls] if not sub.empty else pd.DataFrame()
            if rr.empty:
                if not sub.empty:
                    rows.append({"panel": "b_classification", "condition": c, "class": cls, "n": int(sub["denominator"].iloc[0]),
                                 "count": 0, "share": 0.0, "ci_low": None, "ci_high": None})
                continue
            rr = rr.iloc[0]
            rows.append({"panel": "b_classification", "condition": c, "class": cls, "n": int(rr["denominator"]), "count": int(rr["n"]),
                         "share": float(rr["share"]), "ci_low": None, "ci_high": None})
    numbers = pd.DataFrame(rows)
    has_mix = (numbers["panel"] == "b_classification").any()

    with plt.rc_context(RC):
        ncols = 2 if has_mix else 1
        fig, axes = plt.subplots(1, ncols, figsize=(FULL_WIDTH if has_mix else HALF_WIDTH, 2.15), layout="constrained",
                                 squeeze=False, width_ratios=[1.2, 1.1] if has_mix else [1])
        ax = axes[0][0]
        ax.tick_params(axis="x", labelsize=7)
        _grid(ax, "y")
        sub = numbers[numbers["panel"] == "a_noncompliance"].set_index("condition")
        for i, c in enumerate(conds):
            r = sub.loc[c]
            y = r["share"] * 100
            ax.bar(i, y, width=0.55, color=CONDITION_COLORS[c], linewidth=0, zorder=2)
            ax.plot([i, i], [r["ci_low"] * 100, r["ci_high"] * 100], color=INK2, linewidth=0.7, zorder=3)
            ax.annotate(f"{y:.1f}%", (i, r["ci_high"] * 100), xytext=(0, 2.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 70)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_yticklabels([f"{t}%" for t in (0, 20, 40, 60)])
        ax.set_ylabel("Runs with adherence < 4")
        ax.set_title("(a) Constraint non-compliance" if has_mix else "Constraint non-compliance")

        if has_mix:
            ax = axes[0][1]
            _grid(ax, "x")
            subm = numbers[numbers["panel"] == "b_classification"]
            for i, c in enumerate(conds):
                y = len(conds) - 1 - i
                left = 0.0
                for k, cls in enumerate(CLASS_ORDER):
                    rr = subm[(subm["condition"] == c) & (subm["class"] == cls)]
                    if rr.empty:
                        continue
                    share = float(rr["share"].iloc[0]) * 100
                    if share <= 0:
                        continue
                    ax.barh(y, share, left=left, height=0.6, color=ORDINAL[k], edgecolor=SURFACE, linewidth=1.0, zorder=2,
                            label=CLASS_LABELS[cls] if i == 0 else None)
                    if share >= 8:
                        ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center", fontsize=7,
                                color=INK if k == 0 else SURFACE, zorder=3)
                    left += share
            ax.set_yticks(range(len(conds)))
            ax.set_yticklabels([CONDITION_LABELS[c] for c in reversed(conds)])
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
            ax.set_xlabel("Share of runs")
            ax.set_title("(b) Auditor drift classification")
            handles = [Line2D([], [], marker="s", color=ORDINAL[k], linestyle="none", markersize=6, label=CLASS_LABELS[cls])
                       for k, cls in enumerate(CLASS_ORDER)]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, columnspacing=0.8,
                      handletextpad=0.3, fontsize=7)
        numbers = _save(fig, figures_dir / "fig_drift_classification.pdf", {
            "figure": "fig_drift_classification",
            "inputs": [str(ANALYSIS_SUBDIR / "noncompliance_by_condition.csv"), str(ANALYSIS_SUBDIR / "auditor_onset_summary.csv")],
            "arm": "published (judge adherence: Opus 4.6 for non-Anthropic subjects, GPT-5.4 for Sonnet; the auditor is GPT-5.4 for every run); pooled over all seven models",
            "sample": _sample_block({"name": "clean (sample == clean rows; auditor rows filtered to the clean parquet upstream)"}),
            "panel_a": "share of scored runs with judge constraint_adherence < 4, Wilson 95% CI",
            "panel_b": "auditor drift_classification shares per condition (classification_mix section of auditor_onset_summary.csv); ordered classes drawn with a single-hue ordinal ramp",
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Figure 6: complexity inflation vs output tokens (appendix)
# ---------------------------------------------------------------------------


def fig_complexity_vs_tokens(analysis_dir: Path, figures_dir: Path, date: str | None = None, n_bins: int = 6) -> pd.DataFrame:
    """Complexity inflation vs final-response output tokens, Opus arm, clean sample."""
    runs = pd.read_csv(analysis_dir / "tokens_context_runs.csv")
    crossed_path = analysis_dir / "crossed_scores_runs.csv"
    crossed = pd.read_csv(crossed_path) if crossed_path.exists() else pd.DataFrame()
    n_in = int(len(runs))
    runs = runs.copy()
    runs["complexity_opus"] = runs["j_complexity_inflation"].astype(float)
    runs["judge_source"] = "published Opus score"
    n_sonnet_replaced = 0
    n_sonnet_dropped = 0
    if SONNET in set(runs["model_id"]):
        son = runs["model_id"] == SONNET
        runs.loc[son, "complexity_opus"] = np.nan
        if not crossed.empty and "opus_complexity_inflation" in crossed.columns:
            cx = crossed[crossed["model_id"] == SONNET][["run_id", "opus_complexity_inflation"]].dropna()
            cx = cx.drop_duplicates("run_id").set_index("run_id")["opus_complexity_inflation"]
            mapped = runs.loc[son, "run_id"].map(cx)
            runs.loc[son, "complexity_opus"] = mapped.values
            n_sonnet_replaced = int(mapped.notna().sum())
            runs.loc[son & runs["complexity_opus"].notna(), "judge_source"] = "E1 crossed Opus score"
        n_sonnet_dropped = int((son & runs["complexity_opus"].isna()).sum())
    runs = runs[runs["complexity_opus"].notna() & runs["last_output_tokens"].notna()]
    runs = runs.sort_values(["condition", "run_id"]).reset_index(drop=True)
    x_all = runs["last_output_tokens"].astype(float)
    edges = np.unique(np.quantile(x_all, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array([x_all.min(), x_all.max()])
    edges[-1] = edges[-1] + 1e-9

    conds = [c for c in CONDITION_ORDER if c in set(runs["condition"])]
    rows: list[dict] = []
    for c in conds:
        sub = runs[runs["condition"] == c]
        bins = np.digitize(sub["last_output_tokens"].astype(float), edges[1:-1], right=False)
        for b in range(len(edges) - 1):
            g = sub[bins == b]
            n = int(len(g))
            if n == 0:
                continue
            m = float(g["complexity_opus"].mean())
            if n > 1:
                from scipy import stats

                h = float(stats.t.ppf(0.975, n - 1) * g["complexity_opus"].std(ddof=1) / math.sqrt(n))
            else:
                h = float("nan")
            rows.append({"condition": c, "bin": b, "tokens_low": float(edges[b]), "tokens_high": float(edges[b + 1]),
                         "tokens_median": float(g["last_output_tokens"].median()), "n": n, "mean_complexity": m,
                         "ci_low": m - h if n > 1 else None, "ci_high": m + h if n > 1 else None})
    numbers = pd.DataFrame(rows)
    counts = runs.groupby("condition").size().to_dict()

    rng = np.random.RandomState(20261026)  # fixed seed: deterministic jitter
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.5), layout="constrained")
        _grid(ax, "y")
        jitter = rng.uniform(-0.12, 0.12, size=len(runs))
        ax.scatter(runs["last_output_tokens"], runs["complexity_opus"] + jitter, s=3.5, color=MUTED, alpha=0.16,
                   linewidths=0, zorder=1, rasterized=False)
        for c in conds:
            sub = numbers[numbers["condition"] == c]
            color = CONDITION_COLORS[c]
            ax.plot(sub["tokens_median"], sub["mean_complexity"], color=color, linewidth=1.2, marker="o", markersize=3.6,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3, label=CONDITION_LABELS[c])
            ok = sub["ci_low"].notna()
            ax.vlines(sub.loc[ok, "tokens_median"], sub.loc[ok, "ci_low"], sub.loc[ok, "ci_high"], color=color, linewidth=0.8, zorder=2)
        ax.set_ylim(-0.3, 4.3)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_xlabel("Final-response output tokens")
        ax.set_ylabel("Complexity inflation (0-4)")
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=0.8, handlelength=1.4,
                  handletextpad=0.4, fontsize=7)
        numbers = _save(fig, figures_dir / "fig_complexity_vs_tokens.pdf", {
            "figure": "fig_complexity_vs_tokens",
            "inputs": [str(ANALYSIS_SUBDIR / "tokens_context_runs.csv"), str(ANALYSIS_SUBDIR / "crossed_scores_runs.csv")],
            "arm": "opus: published Opus 4.6 complexity for non-Anthropic subjects; Sonnet single-shot and pressure rows take the E1 crossed Opus score; Sonnet neutral and checkpointed rows have no Opus score and are dropped",
            "sample": _sample_block({"name": "clean (tokens_context_runs.csv is restricted to the clean parquet)",
                                     "rows_in": n_in, "sonnet_rows_replaced_by_crossed_opus": n_sonnet_replaced,
                                     "sonnet_rows_dropped_no_opus_score": n_sonnet_dropped, "rows_drawn": int(len(runs)),
                                     "rows_drawn_by_condition": {k: int(v) for k, v in counts.items()}}),
            "x": "last_output_tokens: logged output tokens of the final main assistant turn (the judged response); the 4,096-token cap bounds the axis",
            "y": "judge complexity_inflation (0-4) under the Opus arm; the grey points carry a fixed-seed vertical jitter of +-0.12 for visibility, the binned means do not",
            "bins": f"{len(edges) - 1} quantile bins of final-response tokens over all drawn runs (edges: {[round(float(e), 1) for e in edges]}); per condition and bin: mean complexity with a 95% t interval, drawn at the bin median of tokens",
            "scatter_points": int(len(runs)),
        }, numbers, date)
    return numbers


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

FIGURE_FUNCTIONS = {
    "kbv_by_judge_arm": fig_kbv_by_judge_arm,
    "trajectories": fig_trajectories,
    "drift_checkpoint": fig_drift_checkpoint,
    "sensitivity": fig_sensitivity,
    "drift_classification": fig_drift_classification,
    "complexity_vs_tokens": fig_complexity_vs_tokens,
}


def run(data_root: Path, out_dir: Path | None = None, figures_dir: Path | None = None, date: str | None = None,
        only: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Draw every camera-ready figure into ``figures_dir`` (default: ``<repo>/figures``).

    ``data_root`` is ``drift_bench/data``; the CSVs of record are read from
    ``<data_root>/camera_ready/analysis``. ``out_dir`` is accepted for the
    run_analysis item-6 signature and is not written to. Returns the numbers
    drawn per figure.
    """
    data_root = Path(data_root)
    analysis_dir = data_root / ANALYSIS_SUBDIR
    figures_dir = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for name, fn in FIGURE_FUNCTIONS.items():
        if only and name not in only:
            continue
        if name == "sensitivity":
            results[name] = fn(data_root, analysis_dir, figures_dir, date)
        else:
            results[name] = fn(analysis_dir, figures_dir, date)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Camera-ready figures (D10). No API calls.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "drift_bench" / "data")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--date", type=str, default=None, help="date written to the provenance files (default: today)")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURE_FUNCTIONS), default=None)
    args = parser.parse_args(argv)
    results = run(args.data_root, None, args.figures_dir, args.date, args.only)
    for name, df in results.items():
        print(f"{name}: {len(df)} numbers drawn -> {args.figures_dir / ('fig_' + name + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
