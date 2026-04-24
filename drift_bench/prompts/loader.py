"""Load and render Jinja2 prompt templates."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent

# Module-level singleton to avoid re-creating on every render
_env: Environment | None = None


def get_env() -> Environment:
    """Get or create Jinja2 environment (singleton)."""
    global _env
    if _env is None:
        _env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _env


def render_template(template_name: str, **kwargs: Any) -> str:
    """Render a named template with given variables."""
    env = get_env()
    template = env.get_template(template_name)
    return template.render(**kwargs)


def render_system_prompt() -> str:
    """Render the system prompt (no variables needed)."""
    return render_template("system.j2")


def render_brief_for_prompt(brief: dict[str, Any]) -> str:
    """Format a brief dict into a readable string for insertion into prompts."""
    lines = [
        f"Domain: {brief['domain']}",
        f"Objective: {brief['objective']}",
        "",
        "Hard constraints:",
    ]
    for c in brief["hard_constraints"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Success criteria:")
    for s in brief["success_criteria"]:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("Plausible directions:")
    for d in brief["plausible_directions"]:
        lines.append(f"  - {d}")
    lines.append("")
    lines.append("Banned moves:")
    for b in brief["banned_moves"]:
        lines.append(f"  - {b}")
    return "\n".join(lines)


def get_turn_prompts(condition: str, brief: dict[str, Any]) -> list[str]:
    """Return the ordered list of user prompts for a given condition.

    For single_shot: returns a 1-element list.
    For multi_turn_neutral: returns a 6-element list.
    For multi_turn_pressure: returns a 6-element list.
    For checkpointed_pressure: returns an 8-element list (6 turns + 2 checkpoints).
    """
    brief_text = render_brief_for_prompt(brief)

    if condition == "single_shot":
        return [render_template("single_shot.j2", brief=brief_text)]

    elif condition == "multi_turn_neutral":
        full = render_template("multi_turn_neutral.j2", brief=brief_text)
        return _split_turns(full)

    elif condition == "multi_turn_pressure":
        full = render_template("multi_turn_pressure.j2", brief=brief_text)
        return _split_turns(full)

    elif condition == "multi_turn_pressure_rigor":
        full = render_template("multi_turn_pressure_rigor.j2", brief=brief_text)
        return _split_turns(full)

    elif condition == "checkpointed_pressure":
        pressure_turns = get_turn_prompts("multi_turn_pressure", brief)
        checkpoint_text = render_template("checkpoint.j2")
        result = []
        for i, turn in enumerate(pressure_turns):
            result.append(turn)
            if (i + 1) in (2, 4):  # after turn 2 and turn 4
                result.append(checkpoint_text)
        return result

    else:
        raise ValueError(f"Unknown condition: {condition}")


def get_checkpoint_indices(condition: str) -> set[int]:
    """Return the set of turn indices that are checkpoint turns.

    For checkpointed_pressure: indices 2 and 5 (after inserting checkpoints
    after pressure turns 2 and 4). For all other conditions: empty set.
    """
    if condition == "checkpointed_pressure":
        # Pressure turns: [0,1,2,3,4,5] -> after inserting checkpoints after
        # positions 1 and 3: [p0, p1, CKPT, p2, p3, CKPT, p4, p5]
        # Checkpoint indices are 2 and 5
        return {2, 5}
    return set()


def get_restatement_probe() -> str:
    """Return the restatement probe prompt text."""
    return render_template("restatement_probe.j2")


def _split_turns(text: str) -> list[str]:
    """Split a rendered multi-turn template on '---TURN---' markers."""
    parts = text.split("---TURN---")
    return [p.strip() for p in parts if p.strip()]
