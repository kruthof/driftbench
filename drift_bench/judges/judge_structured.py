"""Structured judge: scores via explicit constraint and alternative extraction.

Addresses the calibration gap between judge and auditor on constraint
adherence and alternative coverage by forcing mechanical extraction
before scoring. The judge must check each constraint individually and
each alternative direction individually, then map counts to scores.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import jsonlines
import yaml

from drift_bench.runners.base import BaseRunner
from drift_bench.prompts.loader import render_template, render_brief_for_prompt
from drift_bench.judges.judge import (
    first_json_object,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RUBRICS_PATH,
    load_calibration,
    load_rubrics,
    score_from_record,
    score_output_path,
    truncate_transcript_dict_at_turn,
)


@dataclass
class StructuredJudgeScore:
    """Score from the structured (extraction-first) judge."""
    run_id: str
    brief_id: str
    model_id: str
    condition: str
    judge_model: str
    objective_fidelity: int
    constraint_adherence: int
    alternative_coverage: int
    complexity_inflation: int
    constraint_checks: list[dict] = field(default_factory=list)
    direction_checks: list[dict] = field(default_factory=list)
    summary: str = ""
    violations: list[str] = field(default_factory=list)
    optional_extras_flagged: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    judge_latency_ms: float = 0.0
    raw_response: str = ""
    # Added for the camera-ready (X1, X2). Old records lack these keys.
    turn_index: int | None = None
    judge_response_model: str | None = None
    cost_usd: float = 0.0
    # True when cost_usd came from the config `pricing` fallback, not litellm.
    cost_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredJudgePipeline(BaseRunner):
    """Score proposals via structured constraint/alternative extraction."""

    def __init__(
        self,
        config_path: Path | None = None,
        judge_override: str | None = None,
        output_prefix: str = "structured",
        prompts_dir: Path | None = None,
        rubrics_path: Path | None = None,
        calibration_path: Path | None = None,
    ):
        super().__init__(config_path, prompts_dir=prompts_dir)
        self.judge_override = judge_override
        self.output_prefix = output_prefix
        self.rubrics_path = Path(rubrics_path) if rubrics_path else DEFAULT_RUBRICS_PATH
        self.calibration_path = (
            Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
        )
        self.rubrics = load_rubrics(self.rubrics_path)
        self.calibration = load_calibration(self.calibration_path)

    def _select_judge_model(self, subject_model: str) -> str:
        if self.judge_override:
            return self.judge_override
        judges_config = self.config["models"]["judges"]
        if "anthropic" in subject_model:
            return judges_config["for_anthropic_runs"]
        return judges_config["default"]

    def _output_path(self, output_dir: Path, run_id: str, turn_index: int | None = None) -> Path:
        return score_output_path(output_dir, self.output_prefix, run_id, turn_index)

    def _extract_final_proposal(self, transcript: dict) -> str:
        for msg in reversed(transcript.get("messages", [])):
            if msg.get("role") == "assistant" and not msg.get("is_probe", False):
                return msg.get("content", "")
        return ""

    def _format_calibration_examples(self) -> str:
        examples = []
        for ex in self.calibration.get("examples", []):
            examples.append(
                f"Example ({ex['label']}):\n"
                f"Brief: {ex['brief_summary'].strip()}\n"
                f"Proposal: {ex['proposal_summary'].strip()}\n"
                f"Scores: objective_fidelity={ex['scores']['objective_fidelity']}, "
                f"constraint_adherence={ex['scores']['constraint_adherence']}, "
                f"alternative_coverage={ex['scores']['alternative_coverage']}, "
                f"complexity_inflation={ex['scores']['complexity_inflation']}\n"
                f"Rationale: {ex['rationale'].strip()}"
            )
        return "\n\n".join(examples)

    def _format_constraints_list(self, brief: dict) -> str:
        constraints = brief.get("hard_constraints", [])
        return "\n".join(f"- {c}" for c in constraints)

    def _format_directions_list(self, brief: dict) -> str:
        directions = brief.get("plausible_directions", [])
        return "\n".join(f"- {d}" for d in directions)

    def build_prompt(self, transcript: dict, brief: dict[str, Any]) -> str:
        """Render the structured judge prompt. No API call."""
        meta = transcript["metadata"]
        final_proposal = self._extract_final_proposal(transcript)
        if not final_proposal:
            raise ValueError(f"No final proposal in {meta['run_id']}")
        return render_template(
            "judge_structured.j2",
            prompts_dir=self.prompts_dir,
            brief=render_brief_for_prompt(brief),
            final_proposal=final_proposal,
            calibration_examples=self._format_calibration_examples(),
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
            constraints_list=self._format_constraints_list(brief),
            directions_list=self._format_directions_list(brief),
        )

    async def score_transcript(
        self,
        transcript: dict,
        brief: dict[str, Any],
        output_dir: Path | None = None,
        turn_index: int | None = None,
    ) -> StructuredJudgeScore:
        if turn_index is not None:
            transcript = truncate_transcript_dict_at_turn(transcript, turn_index)

        meta = transcript["metadata"]
        judge_model = self._select_judge_model(meta["model_id"])
        prompt = self.build_prompt(transcript, brief)

        result = await self._call_llm(
            model_id=judge_model,
            messages=[
                {"role": "system", "content": "You are an expert research evaluation judge. Follow the structured extraction process exactly. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        try:
            scores_dict = json.loads(result["content"])
        except json.JSONDecodeError:
            scores_dict = first_json_object(result["content"])

        score = StructuredJudgeScore(
            run_id=meta["run_id"],
            brief_id=meta["brief_id"],
            model_id=meta["model_id"],
            condition=meta["condition"],
            judge_model=judge_model,
            objective_fidelity=int(scores_dict.get("objective_fidelity", 0)),
            constraint_adherence=int(scores_dict.get("constraint_adherence", 0)),
            alternative_coverage=int(scores_dict.get("alternative_coverage", 0)),
            complexity_inflation=int(scores_dict.get("complexity_inflation", 0)),
            constraint_checks=scores_dict.get("constraint_checks", []),
            direction_checks=scores_dict.get("direction_checks", []),
            summary=scores_dict.get("summary", ""),
            violations=scores_dict.get("violations", []),
            optional_extras_flagged=scores_dict.get("optional_extras_flagged", []),
            timestamp=time.time(),
            judge_latency_ms=result["latency_ms"],
            raw_response=result["content"],
            turn_index=turn_index,
            judge_response_model=result.get("response_model"),
            cost_usd=float(result.get("cost", 0.0) or 0.0),
            cost_estimated=bool(result.get("cost_estimated", False)),
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_path(output_dir, score.run_id, turn_index)
            with jsonlines.open(path, mode="w") as writer:
                writer.write(score.to_dict())

        return score

    async def score_all(
        self,
        transcripts_dir: Path,
        briefs_dir: Path,
        output_dir: Path,
        concurrency: int = 10,
        turn_index: int | None = None,
    ) -> list[StructuredJudgeScore]:
        import asyncio

        briefs_by_id = {}
        for path in sorted(briefs_dir.glob("*.yaml")):
            with open(path) as f:
                brief = yaml.safe_load(f)
            briefs_by_id[brief["id"]] = brief

        transcripts = []
        for path in sorted(transcripts_dir.glob("*.jsonl")):
            try:
                with jsonlines.open(path) as reader:
                    transcripts.append(next(iter(reader)))
            except Exception:
                continue

        semaphore = asyncio.Semaphore(concurrency)
        logger = logging.getLogger(__name__)
        scores: list[StructuredJudgeScore] = []

        async def score_one(t: dict) -> StructuredJudgeScore | Exception:
            existing = self._output_path(output_dir, t["metadata"]["run_id"], turn_index)
            if existing.exists():
                try:
                    with jsonlines.open(existing) as reader:
                        record = next(iter(reader))
                        return score_from_record(StructuredJudgeScore, record)
                except Exception:
                    pass
            async with semaphore:
                brief_id = t["metadata"]["brief_id"]
                brief = briefs_by_id.get(brief_id)
                if brief is None:
                    return ValueError(f"Brief not found: {brief_id}")
                try:
                    return await self.score_transcript(t, brief, output_dir, turn_index=turn_index)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[score_one(t) for t in transcripts])

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Structured judge error: {r}")
            else:
                scores.append(r)

        return scores
