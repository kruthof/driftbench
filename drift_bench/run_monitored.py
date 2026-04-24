"""Run the monitored condition experiment.

Tests the two-model architecture: subject generates under pressure,
GPT-5.4-mini monitors constraints after each turn and intervenes
when violations are detected.

Runs on 4 high-drift models x 20 briefs x 1 rep = 80 runs.

Usage:
    python -m drift_bench.run_monitored --max-budget 100
"""

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from drift_bench.pipeline import run_experiment

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# 20 briefs spanning diverse domains
BRIEF_SUBSET = [
    "nlp_01", "nlp_02", "econ_01", "neuro_01", "eco_01",
    "pubhealth_01", "matsci_01", "genomics_01", "climate_01", "edu_01",
    "robot_01", "finance_01", "psych_01", "hci_01", "epi_01",
    "legal_01", "soccomp_01", "cogsci_02", "energy_01", "urban_01",
]


def make_config() -> Path:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Only high-drift models
    config["models"]["subjects"] = [
        {"id": "anthropic/claude-sonnet-4-6", "provider": "anthropic", "max_tokens": 4096, "temperature": 0.7},
        {"id": "gemini/gemini-3.1-flash-lite-preview", "provider": "google", "max_tokens": 4096, "temperature": 1.0},
        {"id": "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "provider": "together_ai", "max_tokens": 4096, "temperature": 0.7},
        {"id": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "provider": "together_ai", "max_tokens": 4096, "temperature": 0.7},
    ]

    config["experiment"]["conditions"] = ["multi_turn_pressure_monitored"]
    config["experiment"]["repetitions"] = 1

    config["rate_limits"]["together_ai"] = {
        "requests_per_minute": 60,
        "tokens_per_minute": 800000,
    }

    path = BASE_DIR / "config_monitored.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


async def main_async(max_budget: float):
    config_path = make_config()
    output = BASE_DIR / "data" / "monitored"
    output.mkdir(parents=True, exist_ok=True)

    print("Monitored Experiment")
    print(f"  Models: Sonnet, Flash, Qwen, Llama (high-drift)")
    print(f"  Condition: multi_turn_pressure_monitored")
    print(f"  Monitor: GPT-5.4-mini")
    print(f"  Briefs: {len(BRIEF_SUBSET)}")
    print(f"  Reps: 1")
    print(f"  Budget: ${max_budget}")
    print()

    summary = await run_experiment(
        config_path=config_path,
        briefs_dir=BASE_DIR / "briefs",
        output_base=output,
        conditions=["multi_turn_pressure_monitored"],
        repetitions=1,
        max_budget=max_budget,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-budget", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(args.max_budget))


if __name__ == "__main__":
    main()
