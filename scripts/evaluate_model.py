"""Evaluate a new subject model on DriftBench.

Wraps `drift_bench.pipeline` so you can score any litellm-compatible model
without hand-editing `config.yaml`. The script:

1. Clones `drift_bench/config.yaml` and swaps the subjects list to just your
   model.
2. Adds a default rate-limit entry for your model's provider if it isn't
   already configured.
3. Runs the full pipeline (subjects + cross-family judge + auditor).
4. Aggregates per-run JSONLs into a parquet at the model's output dir.
5. Prints a comparison table against the published 5-model + open-weight
   benchmark baseline (loaded from the HF dataset's `aggregated/all_scores.parquet`).

Example:

    export OPENAI_API_KEY=sk-...
    python scripts/evaluate_model.py --model openai/gpt-4o --max-budget 50

    # Open-weight via Together.ai:
    python scripts/evaluate_model.py \\
        --model "together_ai/deepseek-ai/DeepSeek-V3" \\
        --provider together_ai \\
        --max-budget 30

Outputs land at `drift_bench/data/external/<model-slug>/`. Re-running the
same command resumes (existing transcripts/scores are detected and skipped).
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift_bench.pipeline import run_experiment  # noqa: E402
from drift_bench.analysis.aggregate import aggregate_scores  # noqa: E402

CONFIG_PATH = ROOT / "drift_bench" / "config.yaml"
BRIEFS_DIR = ROOT / "drift_bench" / "briefs"
DEFAULT_OUTPUT = ROOT / "drift_bench" / "data" / "external"

# Conservative defaults — bump for self-hosted endpoints if you control the rate.
DEFAULT_RATE_LIMITS = {
    "requests_per_minute": 60,
    "tokens_per_minute": 800_000,
}

# Litellm prefix -> provider name in config rate_limits. The provider name is
# whatever appears as the first path segment of the litellm model id.
PROVIDER_FROM_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "google": "google",
    "together_ai": "together_ai",
    "groq": "groq",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "fireworks_ai": "fireworks_ai",
    "azure": "openai",
    "bedrock": "bedrock",
}


def _slugify(model_id: str) -> str:
    """Make a filesystem-safe slug from a model id like 'openai/gpt-4o'."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("_")


def _infer_provider(model_id: str) -> str:
    prefix = model_id.split("/", 1)[0]
    return PROVIDER_FROM_PREFIX.get(prefix, prefix)


def make_eval_config(
    model_id: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    conditions: list[str] | None,
    repetitions: int | None,
) -> Path:
    """Write a config.yaml variant scoping the run to a single subject model."""
    base = yaml.safe_load(CONFIG_PATH.read_text())

    base["models"]["subjects"] = [
        {
            "id": model_id,
            "provider": provider,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ]
    if conditions:
        base["experiment"]["conditions"] = conditions
    if repetitions is not None:
        base["experiment"]["repetitions"] = repetitions

    base.setdefault("rate_limits", {})
    if provider not in base["rate_limits"]:
        base["rate_limits"][provider] = dict(DEFAULT_RATE_LIMITS)

    out = ROOT / "drift_bench" / f"config_eval_{_slugify(model_id)}.yaml"
    out.write_text(yaml.dump(base, default_flow_style=False, sort_keys=False))
    return out


def _baseline_summary(parquet_path: Path):
    """Load the published benchmark parquet and compute per-condition means."""
    if not parquet_path.exists():
        return None
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "j_constraint_adherence" not in df.columns:
        return None
    return (
        df.groupby("condition")
        .agg(
            n=("run_id", "count"),
            obj_fid=("j_objective_fidelity", "mean"),
            constr_adh=("j_constraint_adherence", "mean"),
            alt_cov=("j_alternative_coverage", "mean"),
            cplx_inf=("j_complexity_inflation", "mean"),
            drift_rate=("j_constraint_adherence", lambda s: float((s < 3).mean())),
        )
        .round(3)
    )


def _model_summary(parquet_path: Path):
    if not parquet_path.exists():
        return None
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None
    return (
        df.groupby("condition")
        .agg(
            n=("run_id", "count"),
            obj_fid=("j_objective_fidelity", "mean"),
            constr_adh=("j_constraint_adherence", "mean"),
            alt_cov=("j_alternative_coverage", "mean"),
            cplx_inf=("j_complexity_inflation", "mean"),
            drift_rate=("j_constraint_adherence", lambda s: float((s < 3).mean())),
        )
        .round(3)
    )


def print_comparison(model_id: str, model_summary, baseline_summary) -> None:
    print(f"\n=== {model_id} per-condition mean scores ===")
    if model_summary is None or model_summary.empty:
        print("  no scored runs — check the run logs above")
        return
    print(model_summary.to_string())
    if baseline_summary is not None:
        print("\n=== Published benchmark baseline (mean over all 7 models) ===")
        print(baseline_summary.to_string())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        required=True,
        help="Litellm-compatible model id, e.g. openai/gpt-4o, "
             "anthropic/claude-3-5-sonnet-20241022, "
             "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    )
    p.add_argument(
        "--provider",
        default=None,
        help="Provider name for rate limits (default: inferred from --model prefix)",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Subset of conditions (default: all four)",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repetitions per (brief, condition) cell (default: 1)",
    )
    p.add_argument(
        "--max-budget",
        type=float,
        default=None,
        help="Hard cap on API spend in USD (default: from config.yaml)",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output dir (default: {DEFAULT_OUTPUT.relative_to(ROOT)}/<model-slug>)",
    )
    p.add_argument(
        "--baseline-parquet",
        type=Path,
        default=ROOT / "drift_bench" / "data" / "aggregated" / "all_scores.parquet",
        help="Published-benchmark parquet to compare against in the summary table",
    )
    p.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the baseline-vs-new model comparison table",
    )
    args = p.parse_args()

    provider = args.provider or _infer_provider(args.model)
    slug = _slugify(args.model)
    output = args.output or (DEFAULT_OUTPUT / slug)
    output.mkdir(parents=True, exist_ok=True)

    print(f"=== DriftBench external-model evaluation ===")
    print(f"  model:        {args.model}")
    print(f"  provider:     {provider}")
    print(f"  conditions:   {args.conditions or 'all 4 (from config)'}")
    print(f"  repetitions:  {args.repetitions}")
    print(f"  output:       {output.relative_to(ROOT)}")
    print()

    eval_config = make_eval_config(
        model_id=args.model,
        provider=provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        conditions=args.conditions,
        repetitions=args.repetitions,
    )
    print(f"  scratch config: {eval_config.relative_to(ROOT)}")

    summary = asyncio.run(
        run_experiment(
            config_path=eval_config,
            briefs_dir=BRIEFS_DIR,
            output_base=output,
            max_budget=args.max_budget,
            concurrency=args.concurrency,
        )
    )
    print(f"\nrun summary: {summary}")

    parquet_out = output / "aggregated" / "all_scores.parquet"
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    df = aggregate_scores(output / "scores", parquet_out)
    print(f"\naggregated {len(df)} rows -> {parquet_out.relative_to(ROOT)}")

    if not args.no_comparison:
        print_comparison(
            args.model,
            _model_summary(parquet_out),
            _baseline_summary(args.baseline_parquet),
        )

    print(
        f"\nDone. To re-score with a different judge, point JudgePipeline at "
        f"a different judge model id and re-run the relevant block."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
