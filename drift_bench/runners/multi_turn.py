"""Multi-turn runners: neutral and pressure conditions."""

import time
from pathlib import Path
from typing import Any

from drift_bench.runners.base import (
    LITELLM_VERSION, BaseRunner, Message, RunMetadata, Transcript,
)
from drift_bench.prompts.loader import (
    get_turn_prompts, get_restatement_probe, render_brief_for_prompt,
    get_checkpoint_indices,
)


class MultiTurnRunner(BaseRunner):
    """Execute multi-turn conditions (neutral, pressure, or checkpointed)."""

    async def run(
        self,
        brief: dict[str, Any],
        model_id: str,
        condition: str = "multi_turn_neutral",
        repetition: int = 0,
        output_dir: Path | None = None,
        enable_restatement_probe: bool = True,
        enable_brief_reinjection: bool = False,
    ) -> Transcript:
        metadata = RunMetadata(
            brief_id=brief["id"],
            model_id=model_id,
            condition=condition,
            repetition=repetition,
            start_time=time.time(),
            temperature=self._get_temperature(model_id),
            max_tokens=self._get_max_tokens(model_id),
            litellm_version=LITELLM_VERSION,
        )

        turn_prompts = get_turn_prompts(condition, brief, prompts_dir=self.prompts_dir)
        checkpoint_indices = get_checkpoint_indices(condition)
        conversation: list[Message] = [
            Message(role="system", content=self.system_prompt, timestamp=time.time()),
        ]

        total_input = 0
        total_output = 0
        total_cost = 0.0

        for turn_idx, user_prompt in enumerate(turn_prompts):
            # Optionally re-inject brief
            if enable_brief_reinjection and turn_idx > 0:
                brief_text = render_brief_for_prompt(brief)
                user_prompt = f"[BRIEF REMINDER]\n{brief_text}\n\n{user_prompt}"

            conversation.append(Message(
                role="user",
                content=user_prompt,
                timestamp=time.time(),
            ))

            # Main generation call
            result = await self._call_llm(
                model_id=model_id,
                messages=self._build_messages_list(conversation),
                temperature=self._get_temperature(model_id),
                max_tokens=self._get_max_tokens(model_id),
            )

            conversation.append(Message(
                role="assistant",
                content=result["content"],
                timestamp=time.time(),
                token_count=result["output_tokens"],
                latency_ms=result["latency_ms"],
                finish_reason=result.get("finish_reason"),
                response_model=result.get("response_model"),
            ))

            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_cost += result["cost"]

            # Restatement probe (separate call, does NOT enter main conversation)
            is_checkpoint = turn_idx in checkpoint_indices
            is_last = turn_idx == len(turn_prompts) - 1
            if enable_restatement_probe and not is_checkpoint and not is_last:
                probe_result = await self._run_restatement_probe(
                    model_id=model_id,
                    conversation=conversation,
                )
                conversation.append(Message(
                    role="user",
                    content=get_restatement_probe(prompts_dir=self.prompts_dir),
                    timestamp=time.time(),
                    is_probe=True,
                ))
                conversation.append(Message(
                    role="assistant",
                    content=probe_result["content"],
                    timestamp=time.time(),
                    token_count=probe_result["output_tokens"],
                    latency_ms=probe_result["latency_ms"],
                    is_probe=True,
                    finish_reason=probe_result.get("finish_reason"),
                    response_model=probe_result.get("response_model"),
                ))
                total_input += probe_result["input_tokens"]
                total_output += probe_result["output_tokens"]
                total_cost += probe_result["cost"]

        metadata.end_time = time.time()
        metadata.total_input_tokens = total_input
        metadata.total_output_tokens = total_output
        metadata.total_cost_usd = total_cost

        transcript = Transcript(metadata=metadata, messages=conversation)
        if output_dir:
            transcript.save(output_dir)
        return transcript

    def _get_probe_max_tokens(self) -> int:
        """Probe cap from config (experiment.probe_max_tokens).

        Defaults to 1024, the value the April 2026 runs used.
        """
        return int(self.config.get("experiment", {}).get("probe_max_tokens", 1024))

    async def _run_restatement_probe(
        self,
        model_id: str,
        conversation: list[Message],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run the restatement probe as a separate API call.

        Uses the current conversation context but the probe exchange
        is NOT added to the main conversation the model sees on subsequent turns
        (filtered by is_probe flag in _build_messages_list).

        ``max_tokens`` defaults to ``experiment.probe_max_tokens`` in config.yaml.
        """
        if max_tokens is None:
            max_tokens = self._get_probe_max_tokens()
        probe_messages = self._build_messages_list(conversation) + [
            {"role": "user", "content": get_restatement_probe(prompts_dir=self.prompts_dir)}
        ]
        return await self._call_llm(
            model_id=model_id,
            messages=probe_messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
