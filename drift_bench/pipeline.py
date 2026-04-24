"""Pipeline orchestrator: single entry point for the full drift_bench experiment."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from drift_bench.schema.validate import load_all_briefs
from drift_bench.runners.single_shot import SingleShotRunner
from drift_bench.runners.multi_turn import MultiTurnRunner
from drift_bench.runners.checkpointed import CheckpointedRunner
from drift_bench.runners.monitored import MonitoredRunner
import jsonlines
from drift_bench.runners.base import Transcript, RunMetadata, Message
from drift_bench.judges.judge import JudgePipeline
from drift_bench.judges.auditor import AuditorPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONDITION_RUNNERS = {
    "single_shot": SingleShotRunner,
    "multi_turn_neutral": MultiTurnRunner,
    "multi_turn_pressure": MultiTurnRunner,
    "multi_turn_pressure_rigor": MultiTurnRunner,
    "multi_turn_pressure_monitored": MonitoredRunner,
    "checkpointed_pressure": CheckpointedRunner,
}


async def run_experiment(
    config_path: Path,
    briefs_dir: Path,
    output_base: Path,
    conditions: list[str] | None = None,
    models: list[str] | None = None,
    repetitions: int | None = None,
    enable_restatement_probe: bool = True,
    brief_reinjection_ids: list[str] | None = None,
    concurrency: int = 5,
    skip_scoring: bool = False,
    max_budget: float | None = None,
) -> dict[str, Any]:
    """Run the full experimental pipeline.

    Args:
        config_path: Path to config.yaml
        briefs_dir: Directory containing YAML briefs
        output_base: Base directory for all output
        conditions: Which conditions to run (default: all from config)
        models: Which subject models to run (default: all from config)
        repetitions: How many repetitions per cell (default: from config)
        enable_restatement_probe: Whether to run restatement probes
        brief_reinjection_ids: List of brief IDs for brief-reinjection condition
        concurrency: Max concurrent API calls
        skip_scoring: If True, only run subjects (no judge/auditor)

    Returns:
        Summary dict with counts and timing
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if conditions is None:
        conditions = config["experiment"]["conditions"]
    if models is None:
        models = [m["id"] for m in config["models"]["subjects"]]
    if repetitions is None:
        repetitions = config["experiment"]["repetitions"]
    if brief_reinjection_ids is None:
        brief_reinjection_ids = []
    if max_budget is None:
        max_budget = config.get("budget", {}).get("max_usd", float("inf"))
    budget_spent = 0.0

    transcripts_dir = output_base / "transcripts"
    scores_dir = output_base / "scores"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    briefs = load_all_briefs(briefs_dir)
    logger.info(f"Loaded {len(briefs)} briefs")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Models: {models}")
    logger.info(f"Repetitions: {repetitions}")

    # Phase A: Run all subject conditions
    semaphore = asyncio.Semaphore(concurrency)
    all_transcripts: list[Transcript] = []
    start_time = time.time()

    # Create shared runners (one per condition) so rate limiters accumulate state
    shared_runners = {
        cond: CONDITION_RUNNERS[cond](config_path) for cond in conditions
    }

    def _find_existing_transcript(
        brief_id: str, condition: str, model_id: str, rep: int
    ) -> Transcript | None:
        """Check if a transcript for this cell already exists on disk."""
        model_safe = model_id.replace("/", "_")
        pattern = f"*_{brief_id}_{condition}_{model_safe}.jsonl"
        for path in transcripts_dir.glob(pattern):
            try:
                with jsonlines.open(path) as reader:
                    for entry in reader:
                        meta = entry.get("metadata", {})
                        # Validate all fields match exactly (not just filename pattern)
                        if (meta.get("brief_id") == brief_id
                                and meta.get("condition") == condition
                                and meta.get("model_id") == model_id
                                and meta.get("repetition") == rep):
                            return Transcript(
                                metadata=RunMetadata(**meta),
                                messages=[Message(**m) for m in entry["messages"]],
                            )
            except Exception:
                continue
        return None

    async def run_one(
        brief: dict, model_id: str, condition: str, rep: int
    ) -> Transcript | None:
        nonlocal budget_spent
        # Check for existing result (resumability) — before semaphore, no API cost
        existing = _find_existing_transcript(brief["id"], condition, model_id, rep)
        if existing:
            logger.info(f"Skipping {brief['id']} / {condition} / {model_id} / rep={rep} (exists)")
            return existing

        async with semaphore:
            # Budget check INSIDE semaphore to prevent race condition
            if budget_spent >= max_budget:
                logger.warning(f"Budget limit ${max_budget:.2f} reached — skipping remaining runs")
                return None
            runner = shared_runners[condition]
            enable_reinjection = brief["id"] in brief_reinjection_ids
            try:
                logger.info(f"Running {brief['id']} / {condition} / {model_id} / rep={rep}")
                kwargs: dict[str, Any] = {
                    "brief": brief,
                    "model_id": model_id,
                    "condition": condition,
                    "repetition": rep,
                    "output_dir": transcripts_dir,
                }
                if condition != "single_shot":
                    kwargs["enable_restatement_probe"] = enable_restatement_probe
                    kwargs["enable_brief_reinjection"] = enable_reinjection
                transcript = await runner.run(**kwargs)
                budget_spent += transcript.metadata.total_cost_usd
                logger.info(
                    f"Completed {brief['id']} / {condition} / {model_id} / rep={rep} "
                    f"(${transcript.metadata.total_cost_usd:.4f}, total: ${budget_spent:.2f}/{max_budget:.2f})"
                )
                return transcript
            except Exception as e:
                logger.error(
                    f"Failed {brief['id']} / {condition} / {model_id} / rep={rep}: {e}"
                )
                return None

    tasks = []
    for brief in briefs:
        for condition in conditions:
            for model_id in models:
                for rep in range(repetitions):
                    tasks.append(run_one(brief, model_id, condition, rep))

    logger.info(f"Starting {len(tasks)} subject runs")
    results = await asyncio.gather(*tasks)
    all_transcripts = [t for t in results if t is not None]
    subject_time = time.time() - start_time
    logger.info(f"Completed {len(all_transcripts)}/{len(tasks)} runs in {subject_time:.0f}s")

    total_cost = sum(t.metadata.total_cost_usd for t in all_transcripts)

    if skip_scoring:
        summary = {
            "total_runs": len(all_transcripts),
            "failed_runs": len(tasks) - len(all_transcripts),
            "total_time_seconds": subject_time,
            "total_cost_usd": total_cost,
        }
        _save_summary(summary, output_base)
        return summary

    # Phase B: Judge scoring
    logger.info("Starting judge scoring")
    judge = JudgePipeline(config_path)
    judge_scores = await judge.score_all(all_transcripts, briefs, scores_dir, concurrency)
    logger.info(f"Judge scored {len(judge_scores)} transcripts")

    # Phase C: Fresh-auditor evaluation
    logger.info("Starting auditor evaluation")
    auditor = AuditorPipeline(config_path)
    auditor_scores = await auditor.audit_all(all_transcripts, briefs, scores_dir, concurrency)
    logger.info(f"Auditor scored {len(auditor_scores)} transcripts")

    total_time = time.time() - start_time
    subject_cost = sum(t.metadata.total_cost_usd for t in all_transcripts)

    summary = {
        "total_runs": len(all_transcripts),
        "failed_runs": len(tasks) - len(all_transcripts),
        "judge_scores": len(judge_scores),
        "auditor_scores": len(auditor_scores),
        "total_time_seconds": total_time,
        "subject_cost_usd": subject_cost,
        "briefs": len(briefs),
        "conditions": conditions,
        "models": models,
        "repetitions": repetitions,
    }
    _save_summary(summary, output_base)
    return summary


def _save_summary(summary: dict, output_base: Path) -> None:
    summary_path = output_base / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="drift_bench: trajectory drift benchmark pipeline")
    parser.add_argument(
        "--config", type=Path, default=Path("drift_bench/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--briefs-dir", type=Path, default=Path("drift_bench/briefs"),
        help="Directory containing YAML briefs",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("drift_bench/data"),
        help="Base output directory",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Conditions to run (default: all from config)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model IDs to run (default: all from config)",
    )
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="Repetitions per cell (default: from config)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--no-probe", action="store_true",
        help="Disable restatement probes",
    )
    parser.add_argument(
        "--reinjection-ids", nargs="+", default=None,
        help="Brief IDs to use brief re-injection on",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Only run subjects, skip judge and auditor scoring",
    )
    parser.add_argument(
        "--max-budget", type=float, default=None,
        help="Maximum budget in USD (default: from config)",
    )

    args = parser.parse_args()

    asyncio.run(run_experiment(
        config_path=args.config,
        briefs_dir=args.briefs_dir,
        output_base=args.output,
        conditions=args.conditions,
        models=args.models,
        repetitions=args.repetitions,
        enable_restatement_probe=not args.no_probe,
        brief_reinjection_ids=args.reinjection_ids,
        concurrency=args.concurrency,
        skip_scoring=args.skip_scoring,
        max_budget=args.max_budget,
    ))


if __name__ == "__main__":
    main()
