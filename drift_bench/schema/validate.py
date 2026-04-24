"""Validate research briefs against the JSON schema."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

SCHEMA_PATH = Path(__file__).parent / "brief_schema.json"


def load_schema() -> dict:
    """Load the brief JSON schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_brief(brief: dict[str, Any]) -> list[str]:
    """Validate a single brief dict. Returns list of error messages (empty = valid)."""
    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    return [e.message for e in validator.iter_errors(brief)]


def load_and_validate_brief(path: Path) -> dict[str, Any]:
    """Load a YAML brief file and validate it. Raises ValueError on invalid."""
    with open(path) as f:
        brief = yaml.safe_load(f)
    errors = validate_brief(brief)
    if errors:
        raise ValueError(f"Brief {path} invalid:\n" + "\n".join(f"  - {e}" for e in errors))
    return brief


def load_all_briefs(briefs_dir: Path) -> list[dict[str, Any]]:
    """Load and validate all YAML briefs in a directory."""
    briefs = []
    for p in sorted(briefs_dir.glob("*.yaml")):
        briefs.append(load_and_validate_brief(p))
    return briefs
