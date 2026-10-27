"""F4 / E6: the SWE pilot briefs live in drift_bench/briefs_swe and leave briefs/ at 38.

Pure file checks: schema validation, counts, agentic share, and the
executable-arm test files. No network, no API keys.
"""

from pathlib import Path

import yaml

from drift_bench.schema.validate import load_all_briefs, validate_brief

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIEFS_SWE = PROJECT_ROOT / "drift_bench" / "briefs_swe"
BRIEFS_SWE_EXEC = PROJECT_ROOT / "drift_bench" / "briefs_swe_exec"
RECOVERED = PROJECT_ROOT / "docs" / "swe_pilot_recovered" / "briefs"
EXPECTED_IDS = [f"swe_{i:02d}" for i in range(1, 11)]
# The pilot report (section 1.4, section 3) calls swe_07 and swe_08 agentic.
# swe_09 and swe_10 are the two new agentic briefs. E6 asks for 3 of 10.
AGENTIC_IDS = {"swe_07", "swe_08", "swe_09", "swe_10"}
EXEC_ARM_IDS = {"swe_02", "swe_06", "swe_08", "swe_10"}


def test_ten_swe_briefs_validate() -> None:
    briefs = load_all_briefs(BRIEFS_SWE)
    assert [b["id"] for b in briefs] == EXPECTED_IDS
    for b in briefs:
        assert validate_brief(b) == [], b["id"]


def test_main_briefs_dir_still_holds_38(briefs_dir: Path) -> None:
    briefs = load_all_briefs(briefs_dir)
    assert len(briefs) == 38
    assert not any(b["id"].startswith("swe_") for b in briefs)
    assert not list(briefs_dir.glob("swe_*.yaml"))


def test_swe_brief_shape_matches_pilot() -> None:
    """Pilot report section 3: each brief has 6 to 7 hard constraints and 4 banned moves."""
    for b in load_all_briefs(BRIEFS_SWE):
        assert 6 <= len(b["hard_constraints"]) <= 7, b["id"]
        assert len(b["banned_moves"]) == 4, b["id"]
        assert len(b["success_criteria"]) >= 3, b["id"]
        assert len(b["plausible_directions"]) == 3, b["id"]
        assert b["domain"].startswith("software engineering"), b["id"]


def test_at_least_three_agentic_briefs() -> None:
    briefs = {b["id"]: b for b in load_all_briefs(BRIEFS_SWE)}
    agentic = {bid for bid, b in briefs.items() if "agent" in b["domain"]}
    assert agentic == AGENTIC_IDS
    assert len(agentic) >= 3


def test_recovered_briefs_are_unchanged() -> None:
    """The 8 recovered briefs validated without edits; the copies are byte-identical."""
    for i in range(1, 9):
        name = f"swe_{i:02d}.yaml"
        assert (BRIEFS_SWE / name).read_bytes() == (RECOVERED / name).read_bytes(), name


def test_new_briefs_have_checkable_constraints() -> None:
    """swe_09 and swe_10: at least 3 constraints carry a number or a forbidden-import rule."""
    import re

    for bid in ("swe_09", "swe_10"):
        b = yaml.safe_load((BRIEFS_SWE / f"{bid}.yaml").read_text())
        checkable = [c for c in b["hard_constraints"]
                     if re.search(r"\b\d+\b", c) or "standard library" in c or "eval" in c]
        assert len(checkable) >= 3, (bid, checkable)


def test_exec_arm_test_files_exist_with_contracts() -> None:
    import ast

    tests_dir = BRIEFS_SWE / "tests"
    found = {p.name[: -len("_test.py")] for p in tests_dir.glob("swe_*_test.py")}
    assert found == EXEC_ARM_IDS
    for bid in EXEC_ARM_IDS:
        src = (tests_dir / f"{bid}_test.py").read_text()
        assert ast.get_docstring(ast.parse(src)), f"{bid}_test.py needs an interface-contract docstring"
    # The in-place test files must not be collected by a bare `pytest`.
    assert "collect_ignore_glob" in (tests_dir / "conftest.py").read_text()


def test_tests_subdir_does_not_leak_into_brief_loading() -> None:
    """load_all_briefs globs *.yaml non-recursively, so briefs_swe/tests/ is invisible."""
    assert len(load_all_briefs(BRIEFS_SWE)) == 10


def test_exec_briefs_dir_holds_the_four_contract_copies() -> None:
    """drift_bench/briefs_swe_exec/: the 4 executable-arm briefs with the interface contract appended."""
    briefs = load_all_briefs(BRIEFS_SWE_EXEC)
    assert [b["id"] for b in briefs] == sorted(EXEC_ARM_IDS)
    originals = {b["id"]: b for b in load_all_briefs(BRIEFS_SWE)}
    for b in briefs:
        assert validate_brief(b) == []
        assert "Interface contract for the implementation (solution.py):" in b["objective"]
        for key in ("domain", "hard_constraints", "success_criteria", "plausible_directions", "banned_moves"):
            assert b[key] == originals[b["id"]][key], (b["id"], key)


def test_review_wordings_in_new_briefs() -> None:
    """F4 review items 5 and 6: registry-size rejection; swe_09 budget, file naming, per-path reads."""
    swe_10 = yaml.safe_load((BRIEFS_SWE / "swe_10.yaml").read_text())
    assert any("reject a registry that does not" in c for c in swe_10["hard_constraints"])
    swe_09 = yaml.safe_load((BRIEFS_SWE / "swe_09.yaml").read_text())
    assert "names the candidate files" in swe_09["objective"]
    assert any("same call as the last file content" in c for c in swe_09["hard_constraints"])
    assert any("read_file on each named file" in d for d in swe_09["plausible_directions"])
    assert not any("one batch" in d for d in swe_09["plausible_directions"])
