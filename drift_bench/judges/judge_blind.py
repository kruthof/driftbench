"""Blind judge: scores final proposal against brief WITHOUT seeing the transcript.

This is a robustness check to eliminate positional bias. The blind judge
sees only the original brief and the final proposal, not the conversation
that produced it. If scores from the blind judge agree with transcript-aware
scores, the positional bias does not drive the substantive findings.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import jsonlines
import yaml

from drift_bench.runners.base import BaseRunner
from drift_bench.prompts.loader import render_template, render_brief_for_prompt
from drift_bench.judges.judge import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RUBRICS_PATH,
    load_calibration,
    load_rubrics,
    score_from_record,
    score_output_path,
    truncate_transcript_dict_at_turn,
)


@dataclass
class BlindJudgeScore:
    """Score from the blind (transcript-free) judge."""
    run_id: str
    brief_id: str
    model_id: str
    condition: str
    judge_model: str
    objective_fidelity: int
    constraint_adherence: int
    alternative_coverage: int
    complexity_inflation: int
    summary: str
    violations: list[str]
    optional_extras_flagged: list[str]
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


class BlindJudgePipeline(BaseRunner):
    """Score proposals using brief + final proposal only (no transcript)."""

    def __init__(
        self,
        config_path: Path | None = None,
        judge_override: str | None = None,
        output_prefix: str = "blind",
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
        """Select judge model: the override if set, else the cross-family rule."""
        if self.judge_override:
            return self.judge_override
        judges_config = self.config["models"]["judges"]
        if "anthropic" in subject_model:
            return judges_config["for_anthropic_runs"]
        return judges_config["default"]

    def _output_path(self, output_dir: Path, run_id: str, turn_index: int | None = None) -> Path:
        return score_output_path(output_dir, self.output_prefix, run_id, turn_index)

    def _extract_final_proposal(self, transcript: dict) -> str:
        """Extract the last assistant message (non-probe)."""
        for msg in reversed(transcript.get("messages", [])):
            if msg.get("role") == "assistant" and not msg.get("is_probe", False):
                return msg.get("content", "")
        return ""

    def _format_calibration_examples(self) -> str:
        """Format gold calibration examples."""
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

    def build_prompt(self, transcript: dict, brief: dict[str, Any]) -> str:
        """Render the blind judge prompt. No API call."""
        meta = transcript["metadata"]
        final_proposal = self._extract_final_proposal(transcript)
        if not final_proposal:
            raise ValueError(f"No final proposal in {meta['run_id']}")
        return render_template(
            "judge_blind.j2",
            prompts_dir=self.prompts_dir,
            brief=render_brief_for_prompt(brief),
            final_proposal=final_proposal,
            calibration_examples=self._format_calibration_examples(),
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
        )

    async def score_transcript(
        self,
        transcript: dict,
        brief: dict[str, Any],
        output_dir: Path | None = None,
        turn_index: int | None = None,
    ) -> BlindJudgeScore:
        """Score a single transcript using only brief + final proposal."""
        if turn_index is not None:
            transcript = truncate_transcript_dict_at_turn(transcript, turn_index)

        meta = transcript["metadata"]
        judge_model = self._select_judge_model(meta["model_id"])
        prompt = self.build_prompt(transcript, brief)

        result = await self._call_llm(
            model_id=judge_model,
            messages=[
                {"role": "system", "content": "You are an expert research evaluation judge. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        try:
            scores_dict = json.loads(result["content"])
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result["content"], re.DOTALL)
            if match:
                scores_dict = json.loads(match.group())
            else:
                raise ValueError(f"Non-JSON: {result['content'][:200]}")

        score = BlindJudgeScore(
            run_id=meta["run_id"],
            brief_id=meta["brief_id"],
            model_id=meta["model_id"],
            condition=meta["condition"],
            judge_model=judge_model,
            objective_fidelity=int(scores_dict.get("objective_fidelity", 0)),
            constraint_adherence=int(scores_dict.get("constraint_adherence", 0)),
            alternative_coverage=int(scores_dict.get("alternative_coverage", 0)),
            complexity_inflation=int(scores_dict.get("complexity_inflation", 0)),
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
    ) -> list[BlindJudgeScore]:
        """Score all transcripts with the blind judge."""
        import asyncio
        import yaml as _yaml

        # Load briefs
        briefs_by_id = {}
        for path in sorted(briefs_dir.glob("*.yaml")):
            with open(path) as f:
                brief = _yaml.safe_load(f)
            briefs_by_id[brief["id"]] = brief

        # Load transcripts
        transcripts = []
        for path in sorted(transcripts_dir.glob("*.jsonl")):
            try:
                with jsonlines.open(path) as reader:
                    transcripts.append(next(iter(reader)))
            except Exception:
                continue

        semaphore = asyncio.Semaphore(concurrency)
        logger = logging.getLogger(__name__)
        scores: list[BlindJudgeScore] = []

        async def score_one(t: dict) -> BlindJudgeScore | Exception:
            # Resumability
            existing = self._output_path(output_dir, t["metadata"]["run_id"], turn_index)
            if existing.exists():
                try:
                    with jsonlines.open(existing) as reader:
                        record = next(iter(reader))
                        return score_from_record(BlindJudgeScore, record)
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
                logger.error(f"Blind judge error: {r}")
            else:
                scores.append(r)

        return scores
