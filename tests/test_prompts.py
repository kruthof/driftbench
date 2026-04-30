"""Prompt loader: turn counts, checkpoint indices, template rendering."""

import yaml

from drift_bench.prompts.loader import (
    get_checkpoint_indices,
    get_restatement_probe,
    get_turn_prompts,
    render_brief_for_prompt,
    render_system_prompt,
    render_template,
)


EXPECTED_TURN_COUNTS = {
    "single_shot": 1,
    "multi_turn_neutral": 6,
    "multi_turn_pressure": 6,
    "multi_turn_pressure_rigor": 6,
    "checkpointed_pressure": 8,
}


def test_system_prompt_renders() -> None:
    text = render_system_prompt()
    assert text.strip()
    # System prompt teaches the model what's measured — flag if it's been gutted.
    assert "fidelity" in text.lower()


def test_restatement_probe_renders() -> None:
    text = get_restatement_probe()
    assert text.strip()


def test_render_brief_includes_all_sections(sample_brief: dict) -> None:
    text = render_brief_for_prompt(sample_brief)
    for keyword in (
        "Domain:",
        "Objective:",
        "Hard constraints:",
        "Success criteria:",
        "Plausible directions:",
        "Banned moves:",
    ):
        assert keyword in text, f"missing section: {keyword}"


def test_turn_counts_per_condition(sample_brief: dict) -> None:
    """Pin the documented turn count for every supported condition."""
    for condition, expected in EXPECTED_TURN_COUNTS.items():
        prompts = get_turn_prompts(condition, sample_brief)
        assert len(prompts) == expected, (
            f"{condition}: expected {expected} turns, got {len(prompts)}"
        )
        # No empty turns.
        assert all(p.strip() for p in prompts)


def test_checkpoint_indices_match_inserted_positions(sample_brief: dict) -> None:
    """checkpointed_pressure has the documented checkpoint slots at indices 2 and 5."""
    prompts = get_turn_prompts("checkpointed_pressure", sample_brief)
    indices = get_checkpoint_indices("checkpointed_pressure")
    assert indices == {2, 5}
    # Sanity-check by grepping the rendered checkpoint marker into those slots.
    checkpoint_text = render_template("checkpoint.j2")
    marker = checkpoint_text.split()[0]  # e.g. "CHECKPOINT."
    for idx in indices:
        assert marker in prompts[idx], (
            f"slot {idx} should contain checkpoint marker {marker!r}"
        )


def test_non_checkpointed_conditions_have_no_checkpoint_indices() -> None:
    for condition in ("single_shot", "multi_turn_neutral", "multi_turn_pressure"):
        assert get_checkpoint_indices(condition) == set()


def test_judge_templates_render_without_strict_undefined(sample_brief: dict) -> None:
    """All four judge variants render with their full kwarg set under StrictUndefined."""
    brief_text = render_brief_for_prompt(sample_brief)
    transcript = "--- Turn 1 (User) ---\nfoo\n\n--- Turn 1 (Assistant) ---\nbar"
    proposal = "final proposal text"
    rubrics = "rubric placeholder"
    calibration = "calibration placeholder"

    common = dict(
        brief=brief_text,
        final_proposal=proposal,
        rubrics=rubrics,
        calibration_examples=calibration,
    )

    assert render_template("judge.j2", transcript=transcript, **common).strip()
    assert render_template("auditor.j2", transcript=transcript, **common).strip()
    assert render_template("judge_blind.j2", **common).strip()
    assert render_template(
        "judge_structured.j2",
        constraints_list="- c1\n- c2\n- c3",
        directions_list="- d1\n- d2",
        **common,
    ).strip()


def test_real_brief_renders_in_every_condition(briefs_dir) -> None:
    """Pick the first real brief and confirm every supported condition renders."""
    paths = sorted(briefs_dir.glob("*.yaml"))
    assert paths, "no briefs found"
    with open(paths[0]) as f:
        brief = yaml.safe_load(f)
    for condition in EXPECTED_TURN_COUNTS:
        prompts = get_turn_prompts(condition, brief)
        assert len(prompts) == EXPECTED_TURN_COUNTS[condition]
