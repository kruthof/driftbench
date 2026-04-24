"""LLM-based structural complexity extraction pipeline.

Extracts counts of stages, components, datasets, sub-experiments,
and dependencies from research proposals using cross-family LLM judges.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import jsonlines

from drift_bench.runners.base import BaseRunner, Transcript
from drift_bench.prompts.loader import render_template


@dataclass
class StructuralCounts:
    """Structural complexity counts for a single run."""
    run_id: str
    brief_id: str
    model_id: str
    condition: str
    judge_model: str
    stages: int
    components: int
    datasets_resources: int
    sub_experiments: int
    dependencies: int
    total_structural: int = 0
    timestamp: float = 0.0
    latency_ms: float = 0.0
    raw_response: str = ""

    def __post_init__(self):
        self.total_structural = (
            self.stages + self.components + self.datasets_resources +
            self.sub_experiments + self.dependencies
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructureExtractor(BaseRunner):
    """Extract structural complexity counts from research proposals via LLM."""

    def _select_judge_model(self, subject_model: str) -> str:
        """Select judge model using cross-family rule."""
        judges_config = self.config["models"]["judges"]
        if "anthropic" in subject_model:
            return judges_config["for_anthropic_runs"]
        return judges_config["default"]

    def _extract_final_proposal(self, transcript: dict) -> str:
        """Extract the last assistant message (non-probe) from transcript dict."""
        for msg in reversed(transcript.get("messages", [])):
            if msg.get("role") == "assistant" and not msg.get("is_probe", False):
                return msg.get("content", "")
        return ""

    async def extract_counts(
        self,
        transcript: dict,
        output_dir: Path | None = None,
    ) -> StructuralCounts:
        """Extract structural counts from a single transcript."""
        meta = transcript["metadata"]
        judge_model = self._select_judge_model(meta["model_id"])
        final_proposal = self._extract_final_proposal(transcript)

        if not final_proposal:
            raise ValueError(f"No final proposal in transcript {meta['run_id']}")

        prompt = render_template(
            "structure_extraction.j2",
            proposal=final_proposal,
        )

        result = await self._call_llm(
            model_id=judge_model,
            messages=[
                {"role": "system", "content": "You are a research proposal analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        try:
            counts_dict = json.loads(result["content"])
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result["content"], re.DOTALL)
            if match:
                counts_dict = json.loads(match.group())
            else:
                raise ValueError(f"Non-JSON response: {result['content'][:200]}")

        counts = StructuralCounts(
            run_id=meta["run_id"],
            brief_id=meta["brief_id"],
            model_id=meta["model_id"],
            condition=meta["condition"],
            judge_model=judge_model,
            stages=int(counts_dict.get("stages", 0)),
            components=int(counts_dict.get("components", 0)),
            datasets_resources=int(counts_dict.get("datasets_resources", 0)),
            sub_experiments=int(counts_dict.get("sub_experiments", 0)),
            dependencies=int(counts_dict.get("dependencies", 0)),
            timestamp=time.time(),
            latency_ms=result["latency_ms"],
            raw_response=result["content"],
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"structure_{counts.run_id}.jsonl"
            with jsonlines.open(path, mode="w") as writer:
                writer.write(counts.to_dict())

        return counts

    async def extract_all(
        self,
        transcripts_dir: Path,
        output_dir: Path,
        concurrency: int = 5,
    ) -> list[StructuralCounts]:
        """Extract structural counts from all transcripts."""
        import asyncio

        # Load all transcripts
        transcript_paths = sorted(transcripts_dir.glob("*.jsonl"))
        transcripts = []
        for path in transcript_paths:
            try:
                with jsonlines.open(path) as reader:
                    transcripts.append(next(iter(reader)))
            except Exception:
                continue

        semaphore = asyncio.Semaphore(concurrency)
        results: list[StructuralCounts] = []
        logger = logging.getLogger(__name__)

        async def extract_one(t: dict) -> StructuralCounts | Exception:
            # Resumability: skip if output already exists
            existing = output_dir / f"structure_{t['metadata']['run_id']}.jsonl"
            if existing.exists():
                try:
                    with jsonlines.open(existing) as reader:
                        record = next(iter(reader))
                        return StructuralCounts(**{
                            k: record[k]
                            for k in StructuralCounts.__dataclass_fields__
                        })
                except Exception:
                    pass
            async with semaphore:
                try:
                    return await self.extract_counts(t, output_dir)
                except Exception as e:
                    return e

        tasks = [extract_one(t) for t in transcripts]
        completed = await asyncio.gather(*tasks)

        for r in completed:
            if isinstance(r, Exception):
                logger.error(f"Structure extraction error: {r}")
            else:
                results.append(r)

        return results
