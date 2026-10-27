"""Load and render Jinja2 prompt templates."""

from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent

# Process-wide default prompts directory. ``None`` means PROMPTS_DIR.
# ``set_default_prompts_dir`` changes it (pipeline ``--prompts-dir``).
_default_prompts_dir: Path | None = None

# One Jinja2 environment per resolved prompts directory.
_envs: dict[Path, Environment] = {}


def set_default_prompts_dir(prompts_dir: Path | str | None) -> None:
    """Set the prompts directory used when a caller passes none.

    ``None`` restores the built-in ``drift_bench/prompts`` directory.
    Callers that cannot pass ``prompts_dir`` explicitly (the condition
    runners) pick this up through ``get_turn_prompts`` and friends.
    """
    global _default_prompts_dir
    _default_prompts_dir = None if prompts_dir is None else Path(prompts_dir)


def get_default_prompts_dir() -> Path:
    """Return the prompts directory used when a caller passes none."""
    return _default_prompts_dir if _default_prompts_dir is not None else PROMPTS_DIR


def resolve_prompts_dir(prompts_dir: Path | str | None = None) -> Path:
    """Resolve an optional prompts directory to an absolute path."""
    base = Path(prompts_dir) if prompts_dir is not None else get_default_prompts_dir()
    return base.resolve()


def get_env(prompts_dir: Path | str | None = None) -> Environment:
    """Get or create the Jinja2 environment for a prompts directory.

    A custom directory falls back to the built-in ``drift_bench/prompts``
    for any template it does not contain. So a pilot directory only needs
    the templates it overrides.
    """
    resolved = resolve_prompts_dir(prompts_dir)
    env = _envs.get(resolved)
    if env is None:
        if resolved == PROMPTS_DIR.resolve():
            loader = FileSystemLoader(str(resolved))
        else:
            if not resolved.is_dir():
                raise FileNotFoundError(f"Prompts directory not found: {resolved}")
            loader = ChoiceLoader([
                FileSystemLoader(str(resolved)),
                FileSystemLoader(str(PROMPTS_DIR)),
            ])
        env = Environment(
            loader=loader,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _envs[resolved] = env
    return env


def render_template(
    template_name: str,
    prompts_dir: Path | str | None = None,
    **kwargs: Any,
) -> str:
    """Render a named template with given variables.

    ``prompts_dir`` selects the template directory. No template uses a
    variable of that name, so the keyword cannot clash with template
    variables.
    """
    env = get_env(prompts_dir)
    template = env.get_template(template_name)
    return template.render(**kwargs)


def render_system_prompt(prompts_dir: Path | str | None = None) -> str:
    """Render the system prompt (no variables needed)."""
    return render_template("system.j2", prompts_dir=prompts_dir)


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


def get_turn_prompts(
    condition: str,
    brief: dict[str, Any],
    prompts_dir: Path | str | None = None,
) -> list[str]:
    """Return the ordered list of user prompts for a given condition.

    For single_shot: returns a 1-element list.
    For multi_turn_neutral: returns a 6-element list.
    For multi_turn_pressure: returns a 6-element list.
    For checkpointed_pressure: returns an 8-element list (6 turns + 2 checkpoints).
    """
    brief_text = render_brief_for_prompt(brief)

    if condition == "single_shot":
        return [render_template("single_shot.j2", prompts_dir=prompts_dir, brief=brief_text)]

    elif condition == "multi_turn_neutral":
        full = render_template("multi_turn_neutral.j2", prompts_dir=prompts_dir, brief=brief_text)
        return _split_turns(full)

    elif condition == "multi_turn_pressure":
        full = render_template("multi_turn_pressure.j2", prompts_dir=prompts_dir, brief=brief_text)
        return _split_turns(full)

    elif condition == "multi_turn_pressure_rigor":
        full = render_template("multi_turn_pressure_rigor.j2", prompts_dir=prompts_dir, brief=brief_text)
        return _split_turns(full)

    elif condition == "checkpointed_pressure":
        pressure_turns = get_turn_prompts("multi_turn_pressure", brief, prompts_dir=prompts_dir)
        checkpoint_text = render_template("checkpoint.j2", prompts_dir=prompts_dir)
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


def get_restatement_probe(prompts_dir: Path | str | None = None) -> str:
    """Return the restatement probe prompt text."""
    return render_template("restatement_probe.j2", prompts_dir=prompts_dir)


def _split_turns(text: str) -> list[str]:
    """Split a rendered multi-turn template on '---TURN---' markers."""
    parts = text.split("---TURN---")
    return [p.strip() for p in parts if p.strip()]
