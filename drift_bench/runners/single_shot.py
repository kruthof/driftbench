"""Single-shot runner: one prompt, one response."""

import time
from pathlib import Path
from typing import Any

from drift_bench.runners.base import BaseRunner, Message, RunMetadata, Transcript
from drift_bench.prompts.loader import get_turn_prompts


class SingleShotRunner(BaseRunner):
    """Execute a single-shot condition: one user prompt, one assistant response."""

    async def run(
        self,
        brief: dict[str, Any],
        model_id: str,
        condition: str = "single_shot",
        repetition: int = 0,
        output_dir: Path | None = None,
    ) -> Transcript:
        metadata = RunMetadata(
            brief_id=brief["id"],
            model_id=model_id,
            condition=condition,
            repetition=repetition,
            start_time=time.time(),
        )

        turn_prompts = get_turn_prompts("single_shot", brief)

        messages: list[Message] = [
            Message(role="system", content=self.system_prompt, timestamp=time.time()),
            Message(role="user", content=turn_prompts[0], timestamp=time.time()),
        ]

        result = await self._call_llm(
            model_id=model_id,
            messages=self._build_messages_list(messages),
            temperature=self._get_temperature(model_id),
            max_tokens=self._get_max_tokens(model_id),
        )

        messages.append(Message(
            role="assistant",
            content=result["content"],
            timestamp=time.time(),
            token_count=result["output_tokens"],
            latency_ms=result["latency_ms"],
        ))

        metadata.end_time = time.time()
        metadata.total_input_tokens = result["input_tokens"]
        metadata.total_output_tokens = result["output_tokens"]
        metadata.total_cost_usd = result["cost"]

        transcript = Transcript(metadata=metadata, messages=messages)
        if output_dir:
            transcript.save(output_dir)
        return transcript
