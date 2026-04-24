"""Follow-up experiments for reviewer responses.

Experiment A: Re-run Gemini Flash at temperature 0.7 (kills temperature confound)
Experiment B: Run rigor-pressure condition on subset (kills prompt-specificity concern)

Usage:
    python -m drift_bench.run_followup --experiment A
    python -m drift_bench.run_followup --experiment B
    python -m drift_bench.run_followup --experiment both
"""

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from drift_bench.pipeline import run_experiment

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

BRIEF_SUBSET_B = [
    "nlp_01", "econ_01", "neuro_01", "eco_01", "robot_01",
    "pubhealth_01", "matsci_01", "genomics_01", "climate_01", "edu_01",
]


def save_config(config: dict, name: str) -> Path:
    """Save a temporary config file."""
    path = BASE_DIR / f"config_followup_{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


def make_config_a() -> Path:
    """Experiment A: Gemini Flash at temperature 0.7."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["models"]["subjects"] = [
        {
            "id": "gemini/gemini-3.1-flash-lite-preview",
            "provider": "google",
            "max_tokens": 4096,
            "temperature": 0.7,
        }
    ]
    config["experiment"]["conditions"] = [
        "single_shot",
        "multi_turn_neutral",
        "multi_turn_pressure",
        "checkpointed_pressure",
    ]
    config["experiment"]["repetitions"] = 2
    config["paths"]["transcripts"] = "data/transcripts_followup_a/"
    config["paths"]["scores"] = "data/scores_followup_a/"
    return save_config(config, "A")


def make_config_b() -> Path:
    """Experiment B: Rigor pressure on 10-brief subset, 3 models, temp 0.7 for all."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["models"]["subjects"] = [
        {
            "id": "openai/gpt-5.4",
            "provider": "openai",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        {
            "id": "anthropic/claude-sonnet-4-6",
            "provider": "anthropic",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        {
            "id": "gemini/gemini-3.1-flash-lite-preview",
            "provider": "google",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
    ]
    config["experiment"]["conditions"] = [
        "single_shot",
        "multi_turn_pressure_rigor",
    ]
    config["experiment"]["repetitions"] = 1
    config["paths"]["transcripts"] = "data/transcripts_followup_b/"
    config["paths"]["scores"] = "data/scores_followup_b/"
    return save_config(config, "B")


async def run_a(max_budget: float):
    """Run Experiment A."""
    config_path = make_config_a()
    output = BASE_DIR / "data" / "followup_a"
    output.mkdir(parents=True, exist_ok=True)

    print("Experiment A: Gemini Flash at temperature 0.7")
    print("  Models: gemini-3.1-flash-lite-preview (temp=0.7)")
    print("  Conditions: SS, MT-N, MT-P, CK-P")
    print("  Briefs: all 38")
    print("  Reps: 2")
    print(f"  Budget: ${max_budget}")
    print(f"  Output: {output}")
    print()

    summary = await run_experiment(
        config_path=config_path,
        briefs_dir=BASE_DIR / "briefs",
        output_base=output,
        conditions=["single_shot", "multi_turn_neutral",
                     "multi_turn_pressure", "checkpointed_pressure"],
        max_budget=max_budget,
    )
    return summary


async def run_b(max_budget: float):
    """Run Experiment B."""
    config_path = make_config_b()
    output = BASE_DIR / "data" / "followup_b"
    output.mkdir(parents=True, exist_ok=True)

    # Load only the subset of briefs
    from drift_bench.schema.validate import load_all_briefs
    all_briefs = load_all_briefs(BASE_DIR / "briefs")
    subset_briefs = [b for b in all_briefs if b["id"] in BRIEF_SUBSET_B]
    print(f"Experiment B: Rigor pressure on {len(subset_briefs)}-brief subset")
    print("  Models: GPT-5.4, Sonnet 4.6, Gemini Flash (all at temp=0.7)")
    print("  Conditions: SS, multi_turn_pressure_rigor")
    print(f"  Briefs: {[b['id'] for b in subset_briefs]}")
    print("  Reps: 1")
    print(f"  Budget: ${max_budget}")
    print(f"  Output: {output}")
    print()

    summary = await run_experiment(
        config_path=config_path,
        briefs_dir=BASE_DIR / "briefs",
        output_base=output,
        conditions=["single_shot", "multi_turn_pressure_rigor"],
        models=["openai/gpt-5.4", "anthropic/claude-sonnet-4-6",
                "gemini/gemini-3.1-flash-lite-preview"],
        repetitions=1,
        max_budget=max_budget,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run follow-up experiments")
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "both"],
        required=True,
    )
    parser.add_argument("--max-budget", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.experiment in ("A", "both"):
        print("\n" + "=" * 60)
        print("EXPERIMENT A: Gemini Flash at temperature 0.7")
        print("=" * 60)
        asyncio.run(run_a(args.max_budget))

    if args.experiment in ("B", "both"):
        print("\n" + "=" * 60)
        print("EXPERIMENT B: Rigor pressure subset")
        print("=" * 60)
        asyncio.run(run_b(args.max_budget))


if __name__ == "__main__":
    main()
