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


DEFAULT_RUBRICS_PATH = Path(__file__).parent / "rubrics.yaml"
DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration.yaml"


def load_rubrics(rubrics_path: Path | None = None) -> dict[str, Any]:
    """Load the rubric YAML. Defaults to drift_bench/judges/rubrics.yaml."""
    with open(rubrics_path or DEFAULT_RUBRICS_PATH) as f:
        return yaml.safe_load(f)


def load_calibration(calibration_path: Path | None = None) -> dict[str, Any]:
    """Load the calibration YAML. Defaults to drift_bench/judges/calibration.yaml."""
    with open(calibration_path or DEFAULT_CALIBRATION_PATH) as f:
        return yaml.safe_load(f)


def score_from_record(cls, record: dict[str, Any]):
    """Rebuild a score dataclass from a stored record.

    Keys the dataclass does not know are ignored. Optional fields that
    the record lacks (written by an older version) take their defaults.
    A missing required field raises TypeError.
    """
    return cls(**{k: v for k, v in record.items() if k in cls.__dataclass_fields__})


def _is_probe(msg: Any) -> bool:
    if isinstance(msg, dict):
        return bool(msg.get("is_probe", False))
    return bool(getattr(msg, "is_probe", False))


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def truncate_messages_at_turn(messages: list, turn_index: int) -> list:
    """Keep messages up to and including the k-th non-probe assistant message.

    Turn k is the k-th assistant message that is not a probe (1-based).
    k counts ALL non-probe assistant messages. In checkpointed_pressure the
    two checkpoint replies count too, so k runs 1..8 there and the probes
    sit after assistant messages 1, 2, 4, 5 and 7. In the monitored
    condition, replies to monitor interventions count as well.
    Probe messages that follow the kept assistant message are dropped;
    probe messages before it are kept and filtered later as usual.
    Works on Message objects and on raw transcript dicts.
    Raises ValueError if the transcript has fewer than k assistant turns.
    """
    if turn_index < 1:
        raise ValueError(f"turn_index must be >= 1, got {turn_index}")
    seen = 0
    for i, msg in enumerate(messages):
        if _role(msg) == "assistant" and not _is_probe(msg):
            seen += 1
            if seen == turn_index:
                return list(messages[: i + 1])
    raise ValueError(
        f"turn_index {turn_index} out of range: transcript has {seen} assistant turns"
    )


def truncate_transcript_at_turn(transcript: Transcript, turn_index: int) -> Transcript:
    """Return a copy of a Transcript truncated after assistant turn k."""
    return Transcript(
        metadata=transcript.metadata,
        messages=truncate_messages_at_turn(transcript.messages, turn_index),
    )


def truncate_transcript_dict_at_turn(transcript: dict, turn_index: int) -> dict:
    """Return a copy of a raw transcript dict truncated after assistant turn k."""
    out = dict(transcript)
    out["messages"] = truncate_messages_at_turn(transcript.get("messages", []), turn_index)
    return out


def score_output_path(output_dir: Path, prefix: str, run_id: str, turn_index: int | None) -> Path:
    """Build `<prefix>_<run_id>.jsonl` or `<prefix>_t<K>_<run_id>.jsonl`."""
    if turn_index is None:
        return output_dir / f"{prefix}_{run_id}.jsonl"
    return output_dir / f"{prefix}_t{turn_index}_{run_id}.jsonl"



def first_json_object(text: str) -> dict:
    r"""Return the first complete JSON object in ``text``.

    Judges sometimes append prose after a valid object. A greedy
    ``\{.*\}`` regex keeps that prose and ``json.loads`` then fails
    with "Extra data". This scans from each ``{`` and decodes the first
    object that parses. Raises ``ValueError`` when none does.

    A judge that revises itself and emits a second object after the
    first keeps the FIRST object, which may be the retracted draft.
    This happened once in E4 (run e301452993b4, Opus structured judge:
    draft, a "Wait, let me reconsider" paragraph, then a final object).
    That record was overridden by hand to the final object and carries a
    ``manual_override`` field; see the E4 README.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError(f"Judge returned non-JSON: {text[:200]}")

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
    # Added for the camera-ready (X1, X2). Old records lack these keys.
    turn_index: int | None = None
    judge_response_model: str | None = None
    cost_usd: float = 0.0
    # True when cost_usd came from the config `pricing` fallback, not litellm.
    cost_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgePipeline(BaseRunner):
    """Score transcripts using LLM-as-judge with cross-family assignment."""

    def __init__(
        self,
        config_path: Path | None = None,
        judge_override: str | None = None,
        output_prefix: str = "judge",
        prompts_dir: Path | None = None,
        rubrics_path: Path | None = None,
        calibration_path: Path | None = None,
    ):
        """Build the judge.

        judge_override: model id used for every run, regardless of the
            subject family. None keeps the config-based cross-family routing.
        output_prefix: file prefix for score files (default ``judge``).
        prompts_dir, rubrics_path, calibration_path: override the built-in
            prompt directory and rubric/calibration YAML files.
        """
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

    def build_prompt(self, transcript: Transcript, brief: dict[str, Any]) -> str:
        """Render the judge prompt for a transcript. No API call."""
        brief_text = render_brief_for_prompt(brief)
        transcript_text = self._format_transcript_for_judge(transcript)
        final_proposal = self._extract_final_proposal(transcript)
        calibration_text = self._format_calibration_examples()
        return render_template(
            "judge.j2",
            prompts_dir=self.prompts_dir,
            brief=brief_text,
            transcript=transcript_text,
            final_proposal=final_proposal,
            calibration_examples=calibration_text,
            rubrics=yaml.dump(self.rubrics, default_flow_style=False),
        )

    async def score_transcript(
        self,
        transcript: Transcript,
        brief: dict[str, Any],
        output_dir: Path | None = None,
        turn_index: int | None = None,
    ) -> JudgeScore:
        """Score a single transcript.

        turn_index: when set, score the transcript truncated after the
            k-th non-probe assistant message. The record carries
            ``turn_index`` and the file name gets a ``_t<K>`` infix.
        """
        if turn_index is not None:
            transcript = truncate_transcript_at_turn(transcript, turn_index)

        judge_model = self._select_judge_model(transcript.metadata.model_id)
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
            scores_dict = first_json_object(result["content"])

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
        transcripts: list[Transcript],
        briefs: list[dict[str, Any]],
        output_dir: Path,
        concurrency: int = 5,
        turn_index: int | None = None,
    ) -> list[JudgeScore]:
        """Score a list of transcripts."""
        import asyncio

        briefs_by_id = {b["id"]: b for b in briefs}
        semaphore = asyncio.Semaphore(concurrency)
        scores: list[JudgeScore] = []

        async def score_one(t: Transcript) -> JudgeScore | Exception:
            # Resumability: skip if score file already exists
            if output_dir:
                existing = self._output_path(output_dir, t.metadata.run_id, turn_index)
                if existing.exists():
                    try:
                        with jsonlines.open(existing) as reader:
                            record = next(iter(reader))
                            return score_from_record(JudgeScore, record)
                    except Exception:
                        pass  # re-score if file is corrupt
            async with semaphore:
                brief = briefs_by_id[t.metadata.brief_id]
                try:
                    return await self.score_transcript(t, brief, output_dir, turn_index=turn_index)
                except Exception as e:
                    return e

        results = await asyncio.gather(*[score_one(t) for t in transcripts])

        for r in results:
            if isinstance(r, Exception):
                logging.getLogger(__name__).error(f"Judge error: {r}")
            else:
                scores.append(r)

        return scores
