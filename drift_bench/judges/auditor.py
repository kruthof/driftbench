"""Fresh-auditor pipeline: independent evaluation of drift and recoverability."""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import jsonlines
import yaml

from drift_bench.runners.base import BaseRunner, Message, RunMetadata, Transcript
from drift_bench.prompts.loader import render_template, render_brief_for_prompt
from drift_bench.judges.judge import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RUBRICS_PATH,
    load_calibration,
    load_rubrics,
    score_from_record,
    score_output_path,
    truncate_transcript_at_turn,
)


DriftClassification = Literal[
    "no_drift", "mild_drift", "trajectory_drift", "trajectory_lock_in"
]


@dataclass
class AuditorScore:
    """Structured score from the fresh auditor."""
    run_id: str
    brief_id: str
    model_id: str
    condition: str
    auditor_model: str
    objective_fidelity: int
    constraint_adherence: int
    alternative_coverage: int
    complexity_inflation: int
    recoverability: int
    drift_classification: DriftClassification
    drift_events: list[dict[str, Any]]
    corrected_proposal: str
    timestamp: float = 0.0
    auditor_latency_ms: float = 0.0
    raw_response: str = ""
    # Added for the camera-ready (X1, X2). Old records lack these keys.
    turn_index: int | None = None
    auditor_response_model: str | None = None
    cost_usd: float = 0.0
    # True when cost_usd came from the config `pricing` fallback, not litellm.
    cost_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditorPipeline(BaseRunner):
    """Fresh-auditor evaluation: independent drift assessment and correction."""

    def __init__(
        self,
        config_path: Path | None = None,
        auditor_override: str | None = None,
        output_prefix: str = "auditor",
        prompts_dir: Path | None = None,
        rubrics_path: Path | None = None,
        calibration_path: Path | None = None,
    ):
        """Build the auditor.

        auditor_override: model id used for every run, regardless of the
            subject family. None keeps the config-based routing.
        output_prefix: file prefix for score files (default ``auditor``).
        prompts_dir, rubrics_path, calibration_path: override the built-in
            prompt directory and rubric/calibration YAML files.
        """
        super().__init__(config_path, prompts_dir=prompts_dir)
        self.auditor_override = auditor_override
        self.output_prefix = output_prefix
        self.default_auditor_model = self.config["models"]["auditor"]["id"]
        self.auditor_temp = self.config["models"]["auditor"]["temperature"]
        self.auditor_max_tokens = self.config["models"]["auditor"]["max_tokens"]
        self.rubrics_path = Path(rubrics_path) if rubrics_path else DEFAULT_RUBRICS_PATH
        self.calibration_path = (
            Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
        )
        self.rubrics = load_rubrics(self.rubrics_path)
        self.calibration = load_calibration(self.calibration_path)

    def _output_path(self, output_dir: Path, run_id: str, turn_index: int | None = None) -> Path:
        return score_output_path(output_dir, self.output_prefix, run_id, turn_index)

    def _extract_final_proposal(self, transcript: Transcript) -> str:
        """Extract final assistant response."""
        for msg in reversed(transcript.messages):
            if msg.role == "assistant" and not msg.is_probe:
                return msg.content
        raise ValueError(f"No assistant message in {transcript.metadata.run_id}")

    def _format_transcript(self, transcript: Transcript) -> str:
        """Format transcript for auditor, excluding probe messages."""
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
        """Format gold calibration examples for inclusion in auditor prompt."""
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

    def _select_auditor_model(self, subject_model: str) -> str:
        """Select auditor model using cross-family rule.

        Note: existing scores (generated before this fix) used GPT-5.4
        for all runs including OpenAI subjects. This is documented as a
        limitation. Future runs use cross-family routing.
        """
        if self.auditor_override:
            return self.auditor_override
        if "openai" in subject_model:
            # OpenAI subjects get a non-OpenAI auditor
            judges_config = self.config["models"]["judges"]
            return judges_config["default"]  # Claude Opus
        return self.default_auditor_model  # GPT-5.4 for non-OpenAI

    async def audit_transcript(
        self,
        transcript: Transcript,
        brief: dict[str, Any],
        output_dir: Path | None = None,
        turn_index: int | None = None,
    ) -> AuditorScore:
        """Run fresh-auditor evaluation on a single transcript.

        turn_index: when set, audit the transcript truncated after the
            k-th non-probe assistant message.
        """
        if turn_index is not None:
            transcript = truncate_transcript_at_turn(transcript, turn_index)

        auditor_model = self._select_auditor_model(transcript.metadata.model_id)
        brief_text = render_brief_for_prompt(brief)
        transcript_text = self._format_transcript(transcript)
        final_proposal = self._extract_final_proposal(transcript)
        calibration_text = self._format_calibration_examples()

        prompt = render_template(
            "auditor.j2",
            prompts_dir=self.prompts_dir,
            brief=brief_text,
            transcript=transcript_text,
            final_proposal=final_proposal,
            calibration_examples=calibration_text,
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
        )

        result = await self._call_llm(
            model_id=auditor_model,
            messages=[
                {"role": "system", "content": "You are a fresh research auditor. You have not seen this conversation before. Return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.auditor_temp,
            max_tokens=self.auditor_max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            scores_dict = json.loads(result["content"])
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result["content"], re.DOTALL)
            if match:
                scores_dict = json.loads(match.group())
            else:
                raise ValueError(f"Auditor returned non-JSON: {result['content'][:200]}")

        score = AuditorScore(
            run_id=transcript.metadata.run_id,
            brief_id=transcript.metadata.brief_id,
            model_id=transcript.metadata.model_id,
            condition=transcript.metadata.condition,
            auditor_model=auditor_model,
            objective_fidelity=int(scores_dict.get("objective_fidelity", 0)),
            constraint_adherence=int(scores_dict.get("constraint_adherence", 0)),
            alternative_coverage=int(scores_dict.get("alternative_coverage", 0)),
            complexity_inflation=int(scores_dict.get("complexity_inflation", 0)),
            recoverability=int(scores_dict.get("recoverability", 0)),
            drift_classification=scores_dict.get("drift_classification", "no_drift"),
            drift_events=scores_dict.get("drift_events", []),
            corrected_proposal=scores_dict.get("corrected_proposal", ""),
            timestamp=time.time(),
            auditor_latency_ms=result["latency_ms"],
            raw_response=result["content"],
            turn_index=turn_index,
            auditor_response_model=result.get("response_model"),
            cost_usd=float(result.get("cost", 0.0) or 0.0),
            cost_estimated=bool(result.get("cost_estimated", False)),
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_path(output_dir, score.run_id, turn_index)
            with jsonlines.open(path, mode="w") as writer:
                writer.write(score.to_dict())

        return score

    async def audit_all(
        self,
        transcripts: list[Transcript],
        briefs: list[dict[str, Any]],
        output_dir: Path,
        concurrency: int = 5,
        turn_index: int | None = None,
    ) -> list[AuditorScore]:
        """Audit a list of transcripts."""
        import asyncio

        briefs_by_id = {b["id"]: b for b in briefs}
        semaphore = asyncio.Semaphore(concurrency)

        async def audit_one(t: Transcript) -> AuditorScore | Exception:
            # Resumability: skip if score file already exists
            if output_dir:
                existing = self._output_path(output_dir, t.metadata.run_id, turn_index)
                if existing.exists():
                    try:
                        with jsonlines.open(existing) as reader:
                            record = next(iter(reader))
                            return score_from_record(AuditorScore, record)
                    except Exception:
                        pass  # re-score if file is corrupt
            async with semaphore:
                brief = briefs_by_id[t.metadata.brief_id]
                try:
                    return await self.audit_transcript(t, brief, output_dir, turn_index=turn_index)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[audit_one(t) for t in transcripts])

        scores = []
        for r in results:
            if isinstance(r, Exception):
                logging.getLogger(__name__).error(f"Auditor error: {r}")
            else:
                scores.append(r)

        return scores
