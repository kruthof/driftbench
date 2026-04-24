"""Checkpointed runner: pressure condition with structured reflection checkpoints."""

from pathlib import Path
from typing import Any

from drift_bench.runners.multi_turn import MultiTurnRunner
from drift_bench.runners.base import Transcript


class CheckpointedRunner(MultiTurnRunner):
    """Extends MultiTurnRunner for checkpointed_pressure condition.

    The checkpoint prompts are already inserted into the turn list by
    get_turn_prompts("checkpointed_pressure", brief) in the prompt loader,
    so this runner just delegates to the parent with the right condition.
    """

    async def run(
        self,
        brief: dict[str, Any],
        model_id: str,
        condition: str = "checkpointed_pressure",
        repetition: int = 0,
        output_dir: Path | None = None,
        enable_restatement_probe: bool = True,
        enable_brief_reinjection: bool = False,
    ) -> Transcript:
        return await super().run(
            brief=brief,
            model_id=model_id,
            condition=condition,
            repetition=repetition,
            output_dir=output_dir,
            enable_restatement_probe=enable_restatement_probe,
            enable_brief_reinjection=enable_brief_reinjection,
        )
