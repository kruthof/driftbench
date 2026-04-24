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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredJudgePipeline(BaseRunner):
    """Score proposals via structured constraint/alternative extraction."""

    def __init__(self, config_path: Path | None = None):
        super().__init__(config_path)
        rubrics_path = Path(__file__).parent / "rubrics.yaml"
        with open(rubrics_path) as f:
            self.rubrics = yaml.safe_load(f)
        calibration_path = Path(__file__).parent / "calibration.yaml"
        with open(calibration_path) as f:
            self.calibration = yaml.safe_load(f)

    def _select_judge_model(self, subject_model: str) -> str:
        judges_config = self.config["models"]["judges"]
        if "anthropic" in subject_model:
            return judges_config["for_anthropic_runs"]
        return judges_config["default"]

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

    async def score_transcript(
        self,
        transcript: dict,
        brief: dict[str, Any],
        output_dir: Path | None = None,
    ) -> StructuredJudgeScore:
        meta = transcript["metadata"]
        judge_model = self._select_judge_model(meta["model_id"])
        brief_text = render_brief_for_prompt(brief)
        final_proposal = self._extract_final_proposal(transcript)
        calibration_text = self._format_calibration_examples()

        if not final_proposal:
            raise ValueError(f"No final proposal in {meta['run_id']}")

        prompt = render_template(
            "judge_structured.j2",
            brief=brief_text,
            final_proposal=final_proposal,
            calibration_examples=calibration_text,
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
            constraints_list=self._format_constraints_list(brief),
            directions_list=self._format_directions_list(brief),
        )

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
            match = re.search(r"\{.*\}", result["content"], re.DOTALL)
            if match:
                scores_dict = json.loads(match.group())
            else:
                raise ValueError(f"Non-JSON: {result['content'][:200]}")

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
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"structured_{score.run_id}.jsonl"
            with jsonlines.open(path, mode="w") as writer:
                writer.write(score.to_dict())

        return score

    async def score_all(
        self,
        transcripts_dir: Path,
        briefs_dir: Path,
        output_dir: Path,
        concurrency: int = 10,
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
            existing = output_dir / f"structured_{t['metadata']['run_id']}.jsonl"
            if existing.exists():
                try:
                    with jsonlines.open(existing) as reader:
                        record = next(iter(reader))
                        return StructuredJudgeScore(**{
                            k: record[k]
                            for k in StructuredJudgeScore.__dataclass_fields__
                        })
                except Exception:
                    pass
            async with semaphore:
                brief_id = t["metadata"]["brief_id"]
                brief = briefs_by_id.get(brief_id)
                if brief is None:
                    return ValueError(f"Brief not found: {brief_id}")
                try:
                    return await self.score_transcript(t, brief, output_dir)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[score_one(t) for t in transcripts])

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Structured judge error: {r}")
            else:
                scores.append(r)

        return scores
