"""LLM-as-judge scoring pipeline."""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import jsonlines
import yaml

from drift_bench.runners.base import BaseRunner, Message, RunMetadata, Transcript
from drift_bench.prompts.loader import render_template, render_brief_for_prompt


@dataclass
class JudgeScore:
    """Structured score from the judge."""
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgePipeline(BaseRunner):
    """Score transcripts using LLM-as-judge with cross-family assignment."""

    def __init__(self, config_path: Path | None = None):
        super().__init__(config_path)
        rubrics_path = Path(__file__).parent / "rubrics.yaml"
        with open(rubrics_path) as f:
            self.rubrics = yaml.safe_load(f)
        calibration_path = Path(__file__).parent / "calibration.yaml"
        with open(calibration_path) as f:
            self.calibration = yaml.safe_load(f)

    def _select_judge_model(self, subject_model: str) -> str:
        """Select judge model using cross-family rule."""
        judges_config = self.config["models"]["judges"]
        if "anthropic" in subject_model:
            return judges_config["for_anthropic_runs"]
        return judges_config["default"]

    def _extract_final_proposal(self, transcript: Transcript) -> str:
        """Extract the last assistant message as the final proposal."""
        for msg in reversed(transcript.messages):
            if msg.role == "assistant" and not msg.is_probe:
                return msg.content
        raise ValueError(f"No assistant message in transcript {transcript.metadata.run_id}")

    def _format_transcript_for_judge(self, transcript: Transcript) -> str:
        """Format transcript as readable text, excluding probe messages."""
        lines = []
        turn = 0
        for msg in transcript.messages:
            if msg.is_probe:
                continue
            if msg.role == "system":
                continue
            if msg.role == "user":
                turn += 1
                lines.append(f"--- Turn {turn} (User) ---")
                lines.append(msg.content)
            elif msg.role == "assistant":
                lines.append(f"--- Turn {turn} (Assistant) ---")
                lines.append(msg.content)
            lines.append("")
        return "\n".join(lines)

    def _format_calibration_examples(self) -> str:
        """Format gold calibration examples for inclusion in judge prompt."""
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

    async def score_transcript(
        self,
        transcript: Transcript,
        brief: dict[str, Any],
        output_dir: Path | None = None,
    ) -> JudgeScore:
        """Score a single transcript."""
        judge_model = self._select_judge_model(transcript.metadata.model_id)
        brief_text = render_brief_for_prompt(brief)
        transcript_text = self._format_transcript_for_judge(transcript)
        final_proposal = self._extract_final_proposal(transcript)
        calibration_text = self._format_calibration_examples()

        prompt = render_template(
            "judge.j2",
            brief=brief_text,
            transcript=transcript_text,
            final_proposal=final_proposal,
            calibration_examples=calibration_text,
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
        )

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
                raise ValueError(f"Judge returned non-JSON: {result['content'][:200]}")

        score = JudgeScore(
            run_id=transcript.metadata.run_id,
            brief_id=transcript.metadata.brief_id,
            model_id=transcript.metadata.model_id,
            condition=transcript.metadata.condition,
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
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"judge_{score.run_id}.jsonl"
            with jsonlines.open(path, mode="w") as writer:
                writer.write(score.to_dict())

        return score

    async def score_all(
        self,
        transcripts: list[Transcript],
        briefs: list[dict[str, Any]],
        output_dir: Path,
        concurrency: int = 5,
    ) -> list[JudgeScore]:
        """Score a list of transcripts."""
        import asyncio

        briefs_by_id = {b["id"]: b for b in briefs}
        semaphore = asyncio.Semaphore(concurrency)
        scores: list[JudgeScore] = []

        async def score_one(t: Transcript) -> JudgeScore | Exception:
            # Resumability: skip if score file already exists
            if output_dir:
                existing = output_dir / f"judge_{t.metadata.run_id}.jsonl"
                if existing.exists():
                    try:
                        with jsonlines.open(existing) as reader:
                            record = next(iter(reader))
                            return JudgeScore(**{k: record[k] for k in JudgeScore.__dataclass_fields__})
                    except Exception:
                        pass  # re-score if file is corrupt
            async with semaphore:
                brief = briefs_by_id[t.metadata.brief_id]
                try:
                    return await self.score_transcript(t, brief, output_dir)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[score_one(t) for t in transcripts])

        for r in results:
            if isinstance(r, Exception):
                logging.getLogger(__name__).error(f"Judge error: {r}")
            else:
                scores.append(r)

        return scores
