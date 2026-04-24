"""Skeptical judge: stricter constraint checking with per-constraint decomposition.

Uses GPT-5.4 with a prompt that instructs the judge to be skeptical and
mark unaddressed constraints as violated. Produces better-calibrated
constraint adherence scores that align more closely with human judgment.
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
from drift_bench.prompts.loader import render_brief_for_prompt


SKEPTICAL_PROMPT = """You are a strict research proposal auditor. Your job is to find constraint violations.

## Research Brief

Domain: {domain}
Objective: {objective}

## Hard Constraints (check EACH one carefully)

{constraints}

## Final Proposal

{proposal}

## Instructions

For EACH hard constraint listed above:
1. Quote the specific part of the proposal that addresses it (or note if it is not addressed)
2. Determine: SATISFIED, STRETCHED (technically ok but ambiguous), or VIOLATED (clearly broken or ignored)

Be skeptical. If a constraint is not explicitly addressed in the proposal, mark it VIOLATED.
If the proposal seems to address it but is vague or ambiguous, mark it STRETCHED.

After checking each constraint, compute the overall score:
- 4 = All satisfied, none violated or stretched
- 3 = All satisfied or stretched, exactly 1 stretched
- 2 = Exactly 1 violated
- 1 = 2 violated
- 0 = 3+ violated

Also score these dimensions (0-4 scale):
- objective_fidelity: Does the proposal still answer the original research question?
- alternative_coverage: Are competing design choices still acknowledged?
- complexity_inflation: Has the design grown beyond what the brief requires?

Return ONLY valid JSON:
{{
  "constraint_checks": [
    {{"constraint": "text", "status": "SATISFIED/STRETCHED/VIOLATED", "evidence": "quote or note"}}
  ],
  "objective_fidelity": 0,
  "constraint_adherence": 0,
  "alternative_coverage": 0,
  "complexity_inflation": 0,
  "summary": "one sentence",
  "violations": ["list of violated constraints"]
}}"""


@dataclass
class SkepticalJudgeScore:
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
    summary: str = ""
    violations: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    latency_ms: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkepticalJudgePipeline(BaseRunner):
    """Score proposals with skeptical constraint checking."""

    def __init__(self, config_path: Path | None = None):
        super().__init__(config_path)
        self.judge_model = "openai/gpt-5.4"

    def _extract_final_proposal(self, transcript: dict) -> str:
        for msg in reversed(transcript.get("messages", [])):
            if msg.get("role") == "assistant" and not msg.get("is_probe", False):
                return msg.get("content", "")
        return ""

    async def score_transcript(
        self,
        transcript: dict,
        brief: dict[str, Any],
        output_dir: Path | None = None,
    ) -> SkepticalJudgeScore:
        meta = transcript["metadata"]
        final_proposal = self._extract_final_proposal(transcript)

        if not final_proposal:
            raise ValueError(f"No final proposal in {meta['run_id']}")

        constraints = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(brief["hard_constraints"])
        )

        prompt = SKEPTICAL_PROMPT.format(
            domain=brief["domain"],
            objective=brief["objective"],
            constraints=constraints,
            proposal=final_proposal[:4000],
        )

        result = await self._call_llm(
            model_id=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a strict research proposal auditor. Find constraint violations. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        content = result["content"]
        try:
            scores_dict = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                scores_dict = json.loads(match.group())
            else:
                raise ValueError(f"Non-JSON: {content[:200]}")

        score = SkepticalJudgeScore(
            run_id=meta["run_id"],
            brief_id=meta["brief_id"],
            model_id=meta["model_id"],
            condition=meta["condition"],
            judge_model=self.judge_model,
            objective_fidelity=int(scores_dict.get("objective_fidelity", 0)),
            constraint_adherence=int(scores_dict.get("constraint_adherence", 0)),
            alternative_coverage=int(scores_dict.get("alternative_coverage", 0)),
            complexity_inflation=int(scores_dict.get("complexity_inflation", 0)),
            constraint_checks=scores_dict.get("constraint_checks", []),
            summary=scores_dict.get("summary", ""),
            violations=scores_dict.get("violations", []),
            timestamp=time.time(),
            latency_ms=result["latency_ms"],
            raw_response=content,
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"skeptical_{score.run_id}.jsonl"
            with jsonlines.open(path, mode="w") as writer:
                writer.write(score.to_dict())

        return score

    async def score_all(
        self,
        transcripts_dir: list[Path],
        briefs_dir: Path,
        output_dir: Path,
        exclude_models: set[str] | None = None,
        concurrency: int = 10,
    ) -> list[SkepticalJudgeScore]:
        import asyncio

        # Load briefs
        briefs_by_id = {}
        for path in sorted(briefs_dir.glob("*.yaml")):
            with open(path) as f:
                brief = yaml.safe_load(f)
            briefs_by_id[brief["id"]] = brief

        # Load transcripts from all directories
        transcripts = []
        for td in transcripts_dir:
            for path in sorted(td.glob("*.jsonl")):
                try:
                    with jsonlines.open(path) as reader:
                        t = next(iter(reader))
                    # Skip excluded models (GPT runs)
                    if exclude_models and t["metadata"]["model_id"] in exclude_models:
                        continue
                    transcripts.append(t)
                except Exception:
                    continue

        semaphore = asyncio.Semaphore(concurrency)
        logger = logging.getLogger(__name__)
        scores = []

        async def score_one(t: dict) -> SkepticalJudgeScore | Exception:
            existing = output_dir / f"skeptical_{t['metadata']['run_id']}.jsonl"
            if existing.exists():
                try:
                    with jsonlines.open(existing) as reader:
                        record = next(iter(reader))
                        return SkepticalJudgeScore(**{
                            k: record[k] for k in SkepticalJudgeScore.__dataclass_fields__
                        })
                except Exception:
                    pass
            async with semaphore:
                brief = briefs_by_id.get(t["metadata"]["brief_id"])
                if brief is None:
                    return ValueError(f"Brief not found: {t['metadata']['brief_id']}")
                try:
                    return await self.score_transcript(t, brief, output_dir)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[score_one(t) for t in transcripts])

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Skeptical judge error: {r}")
            else:
                scores.append(r)

        return scores
