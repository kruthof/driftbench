"""Schema validation: all shipped briefs validate; malformed briefs are rejected."""

from pathlib import Path

import pytest

from drift_bench.schema.validate import (
    load_all_briefs,
    load_and_validate_brief,
    validate_brief,
)


def test_all_shipped_briefs_validate(briefs_dir: Path) -> None:
    """Every YAML brief in briefs/ must validate against the JSON schema."""
    briefs = load_all_briefs(briefs_dir)
    assert len(briefs) >= 1
    ids = [b["id"] for b in briefs]
    assert len(ids) == len(set(ids)), f"duplicate brief IDs: {ids}"


def test_brief_count_matches_release_claim(briefs_dir: Path) -> None:
    """README and CLAUDE.md claim 38 briefs across 24 domains."""
    briefs = load_all_briefs(briefs_dir)
    assert len(briefs) == 38
    domains = {b["domain"] for b in briefs}
    assert len(domains) >= 20  # tolerate small reorganisation but flag big drift


def test_minimal_brief_validates(sample_brief: dict) -> None:
    assert validate_brief(sample_brief) == []


def test_missing_required_field_rejected(sample_brief: dict) -> None:
    sample_brief.pop("hard_constraints")
    errors = validate_brief(sample_brief)
    assert any("hard_constraints" in e for e in errors)


def test_invalid_id_format_rejected(sample_brief: dict) -> None:
    sample_brief["id"] = "BAD-ID-FORMAT"
    errors = validate_brief(sample_brief)
    assert errors


def test_too_few_constraints_rejected(sample_brief: dict) -> None:
    sample_brief["hard_constraints"] = ["only one"]
    errors = validate_brief(sample_brief)
    assert errors


def test_additional_property_rejected(sample_brief: dict) -> None:
    sample_brief["unexpected_field"] = "value"
    errors = validate_brief(sample_brief)
    assert errors


def test_load_and_validate_raises_on_invalid(tmp_path: Path, sample_brief: dict) -> None:
    import yaml
    sample_brief.pop("objective")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(sample_brief))
    with pytest.raises(ValueError, match="invalid"):
        load_and_validate_brief(bad)
