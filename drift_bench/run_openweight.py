"""Open-weight experiments: Qwen3-235B and Llama-3.3-70B as subjects and judges.

Experiment 2a: Qwen as subject (38 briefs x 4 conditions x 1 rep = 152 runs)
Experiment 2b: Llama as subject (38 briefs x 4 conditions x 1 rep = 152 runs)
Experiment 4a: Qwen as third judge (re-score 400 existing transcripts)
Experiment 4b: Llama as fourth judge (re-score 400 existing transcripts)

Usage:
    python -m drift_bench.run_openweight --experiment subjects
    python -m drift_bench.run_openweight --experiment judges
    python -m drift_bench.run_openweight --experiment all
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import jsonlines
import yaml

from drift_bench.pipeline import run_experiment
from drift_bench.judges.judge import JudgePipeline, JudgeScore
from drift_bench.runners.base import BaseRunner, Transcript, RunMetadata, Message

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

QWEN_MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
LLAMA_MODEL = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"


def make_subject_config() -> Path:
    """Config for running Qwen and Llama as subject models."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["models"]["subjects"] = [
        {
            "id": QWEN_MODEL,
            "provider": "together_ai",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        {
            "id": LLAMA_MODEL,
            "provider": "together_ai",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
    ]

    config["experiment"]["conditions"] = [
        "single_shot",
        "multi_turn_neutral",
        "multi_turn_pressure",
        "checkpointed_pressure",
    ]
    config["experiment"]["repetitions"] = 1

    # Add together_ai rate limits
    config["rate_limits"]["together_ai"] = {
        "requests_per_minute": 60,
        "tokens_per_minute": 800000,
    }

    path = BASE_DIR / "config_openweight_subjects.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


async def run_subjects(max_budget: float):
    """Run Qwen and Llama as subject models."""
    config_path = make_subject_config()
    output = BASE_DIR / "data" / "openweight_subjects"
    output.mkdir(parents=True, exist_ok=True)

    print("Open-weight subjects experiment")
    print(f"  Models: Qwen3-235B, Llama-3.3-70B (both at temp=0.7)")
    print(f"  Conditions: SS, MT-N, MT-P, CK-P")
    print(f"  Briefs: all 38")
    print(f"  Reps: 1")
    print(f"  Budget: ${max_budget}")
    print(f"  Output: {output}")
    print()

    summary = await run_experiment(
        config_path=config_path,
        briefs_dir=BASE_DIR / "briefs",
        output_base=output,
        max_budget=max_budget,
        concurrency=8,
    )
    return summary


async def run_judges(max_budget: float):
    """Run Qwen and Llama as judges on existing transcripts.

    Uses the same rubrics and calibration examples as the main judge.
    Scores a stratified subset of 400 existing transcripts.
    """
    import numpy as np

    # Load existing transcripts
    transcripts_dir = BASE_DIR / "data" / "transcripts"
    transcript_paths = sorted(transcripts_dir.glob("*.jsonl"))

    transcripts = []
    for path in transcript_paths:
        try:
            with jsonlines.open(path) as reader:
                transcripts.append(next(iter(reader)))
        except Exception:
            continue

    print(f"Loaded {len(transcripts)} transcripts")

    # Stratified sample: 100 per condition
    rng = np.random.RandomState(42)
    selected = []
    for cond in ["single_shot", "multi_turn_neutral",
                 "multi_turn_pressure", "checkpointed_pressure"]:
        cond_trans = [t for t in transcripts if t["metadata"]["condition"] == cond]
        if len(cond_trans) > 100:
            idx = rng.choice(len(cond_trans), size=100, replace=False)
            selected.extend([cond_trans[i] for i in idx])
        else:
            selected.extend(cond_trans)

    print(f"Selected {len(selected)} transcripts for judging")

    # Load briefs
    from drift_bench.schema.validate import load_all_briefs
    briefs = load_all_briefs(BASE_DIR / "briefs")
    briefs_by_id = {b["id"]: b for b in briefs}

    # Load rubrics and calibration
    rubrics_path = BASE_DIR / "judges" / "rubrics.yaml"
    with open(rubrics_path) as f:
        rubrics = yaml.safe_load(f)
    calibration_path = BASE_DIR / "judges" / "calibration.yaml"
    with open(calibration_path) as f:
        calibration = yaml.safe_load(f)

    from drift_bench.prompts.loader import render_template, render_brief_for_prompt
    import re

    output_dir = BASE_DIR / "data" / "openweight_judges"
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(8)
    logger = logging.getLogger(__name__)
    budget_spent = 0.0

    # Format calibration examples
    cal_examples = []
    for ex in calibration.get("examples", []):
        cal_examples.append(
            f"Example ({ex['label']}):\n"
            f"Brief: {ex['brief_summary'].strip()}\n"
            f"Proposal: {ex['proposal_summary'].strip()}\n"
            f"Scores: objective_fidelity={ex['scores']['objective_fidelity']}, "
            f"constraint_adherence={ex['scores']['constraint_adherence']}, "
            f"alternative_coverage={ex['scores']['alternative_coverage']}, "
            f"complexity_inflation={ex['scores']['complexity_inflation']}\n"
            f"Rationale: {ex['rationale'].strip()}"
        )
    cal_text = "\n\n".join(cal_examples)

    async def score_one(transcript: dict, judge_model: str, prefix: str):
        nonlocal budget_spent
        run_id = transcript["metadata"]["run_id"]
        out_path = output_dir / f"{prefix}_{run_id}.jsonl"

        # Resumability
        if out_path.exists():
            return

        brief_id = transcript["metadata"]["brief_id"]
        brief = briefs_by_id.get(brief_id)
        if brief is None:
            return

        # Extract final proposal
        final_proposal = ""
        for msg in reversed(transcript.get("messages", [])):
            if msg.get("role") == "assistant" and not msg.get("is_probe", False):
                final_proposal = msg.get("content", "")
                break
        if not final_proposal:
            return

        brief_text = render_brief_for_prompt(brief)
        prompt = render_template(
            "judge_blind.j2",
            brief=brief_text,
            final_proposal=final_proposal,
            calibration_examples=cal_text,
            rubrics=yaml.dump(rubrics, default_flow_style=False),
        )

        async with semaphore:
            if budget_spent >= max_budget:
                return

            try:
                import litellm
                start = time.monotonic()
                response = await litellm.acompletion(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert research evaluation judge. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                elapsed = (time.monotonic() - start) * 1000
                content = response.choices[0].message.content or ""

                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    cost = 0.0
                budget_spent += cost

                try:
                    scores = json.loads(content)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        scores = json.loads(match.group())
                    else:
                        logger.error(f"Non-JSON from {prefix}: {content[:100]}")
                        return

                result = {
                    "run_id": run_id,
                    "brief_id": brief_id,
                    "model_id": transcript["metadata"]["model_id"],
                    "condition": transcript["metadata"]["condition"],
                    "judge_model": judge_model,
                    "objective_fidelity": int(scores.get("objective_fidelity", 0)),
                    "constraint_adherence": int(scores.get("constraint_adherence", 0)),
                    "alternative_coverage": int(scores.get("alternative_coverage", 0)),
                    "complexity_inflation": int(scores.get("complexity_inflation", 0)),
                    "summary": scores.get("summary", ""),
                    "latency_ms": elapsed,
                    "cost": cost,
                }

                with jsonlines.open(out_path, mode="w") as writer:
                    writer.write(result)

                logger.info(f"{prefix} scored {run_id} (${budget_spent:.2f})")

            except Exception as e:
                logger.error(f"{prefix} error on {run_id}: {e}")

    # Score with both models
    tasks = []
    for t in selected:
        tasks.append(score_one(t, QWEN_MODEL, "qwen_judge"))
        tasks.append(score_one(t, LLAMA_MODEL, "llama_judge"))

    print(f"Launching {len(tasks)} judge calls...")
    await asyncio.gather(*tasks)
    print(f"Done. Budget spent: ${budget_spent:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["subjects", "judges", "all"], required=True)
    parser.add_argument("--max-budget", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.experiment in ("subjects", "all"):
        print("\n" + "=" * 60)
        print("OPEN-WEIGHT SUBJECTS: Qwen3-235B + Llama-3.3-70B")
        print("=" * 60)
        asyncio.run(run_subjects(args.max_budget))

    if args.experiment in ("judges", "all"):
        print("\n" + "=" * 60)
        print("OPEN-WEIGHT JUDGES: Qwen3-235B + Llama-3.3-70B on 400 transcripts")
        print("=" * 60)
        asyncio.run(run_judges(args.max_budget))


if __name__ == "__main__":
    main()
