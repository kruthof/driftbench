"""Release build: local paths are stripped from staged copies only.

`scripts/build_dataset_release.py` copies files into a staging dir and then
rewrites the staged text files without the absolute repository root and
without Claude Code scratchpad prefixes. These tests use synthetic trees in
tmp_path; they never touch the real data or the network.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_dataset_release", ROOT / "scripts" / "build_dataset_release.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRATCH = (
    "/private/tmp/claude-501/-Users-someone-repo/"
    "00000000-0000-0000-0000-000000000000/scratchpad/"
)


def test_strip_text_makes_root_paths_relative() -> None:
    b = _load()
    root = Path("/home/someone/repo")
    text = (
        '{"inputs": ["/home/someone/repo/drift_bench/data/x.csv"], '
        '"cwd": "/home/someone/repo", "other": "/home/someone/repo_old/y"}'
    )
    out = b.strip_local_paths_text(text, root)
    data = json.loads(out)
    assert data["inputs"] == ["drift_bench/data/x.csv"]
    assert data["cwd"] == "."
    # A sibling directory that only shares the prefix is left alone.
    assert data["other"] == "/home/someone/repo_old/y"


def test_strip_text_replaces_scratchpad_prefix() -> None:
    b = _load()
    text = f"--briefs-dir {SCRATCH}e2_smoke_briefs"
    out = b.strip_local_paths_text(text, Path("/home/someone/repo"))
    assert out == "--briefs-dir <scratchpad>/e2_smoke_briefs"


def test_strip_rewrites_staged_copies_and_never_the_source(tmp_path: Path) -> None:
    b = _load()
    root = tmp_path / "repo"
    src = root / "drift_bench" / "data" / "camera_ready"
    src.mkdir(parents=True)
    prov = src / "x.csv.provenance.json"
    prov_text = json.dumps({"inputs": [f"{root}/drift_bench/data/scores"], "rows_in": 3})
    prov.write_text(prov_text)
    log = src / "run_log.jsonl"
    log.write_text(json.dumps({"command": f"{root}/.venv/bin/python -m x"}) + "\n")
    clean = src / "clean.csv"
    clean.write_text("a,b\n1,2\n")
    binary = src / "t.parquet"
    binary_bytes = b"PAR1" + str(root).encode() + b"/x"
    binary.write_bytes(binary_bytes)

    out = tmp_path / "stage"
    b.copy_tree_filtered(src, out / "camera_ready")
    changed = b.strip_local_paths(out, root)

    names = sorted(p.name for p in changed)
    assert names == ["run_log.jsonl", "x.csv.provenance.json"]
    staged_prov = json.loads((out / "camera_ready" / prov.name).read_text())
    assert staged_prov["inputs"] == ["drift_bench/data/scores"]
    staged_log = json.loads((out / "camera_ready" / log.name).read_text())
    assert staged_log["command"] == ".venv/bin/python -m x"
    # Untouched: a file without the root, and a binary file.
    assert (out / "camera_ready" / clean.name).read_text() == "a,b\n1,2\n"
    assert (out / "camera_ready" / binary.name).read_bytes() == binary_bytes
    # The sources are unchanged.
    assert prov.read_text() == prov_text
    assert str(root) in log.read_text()


def test_gate_fails_when_root_path_remains(tmp_path: Path) -> None:
    b = _load()
    root = tmp_path / "repo"
    out = tmp_path / "stage"
    out.mkdir()
    (out / "ok.json").write_text('{"path": "drift_bench/data/x.csv"}')
    assert b.verify_gates(out, (), root) == 0
    (out / "bad.md").write_text(f"see {root}/drift_bench/data/x.csv")
    assert b.verify_gates(out, (), root) == 2


def test_gate_fails_on_a_home_path_outside_the_root(tmp_path: Path) -> None:
    b = _load()
    out = tmp_path / "stage"
    out.mkdir()
    home = tmp_path / "home" / "someone"
    (out / "ok.json").write_text('{"path": "drift_bench/data/x.csv"}')
    assert b.verify_gates(out, (), tmp_path / "repo", home=home) == 0
    (out / "log.txt").write_text(f"cache at {home}/.cache/x")
    assert b.verify_gates(out, (), tmp_path / "repo", home=home) == 2
    # a root-level home is ignored, or every absolute path would fail
    assert b.verify_gates(out, (), tmp_path / "repo", home=Path("/")) == 0


def test_gate_fails_on_scratchpad_prefix(tmp_path: Path) -> None:
    b = _load()
    out = tmp_path / "stage"
    out.mkdir()
    (out / "README.md").write_text(f"--briefs-dir {SCRATCH}e2")
    assert b.verify_gates(out, (), tmp_path / "repo") == 2


def test_structure_files_are_not_excluded() -> None:
    b = _load()
    assert not b._is_excluded(Path("drift_bench/data/scores/structure_00086087f846.jsonl"))
    assert not b._is_excluded(Path("drift_bench/prompts/structure_extraction.j2"))
    assert b._is_excluded(Path("drift_bench/data/x.pdf"))
    assert b._is_excluded(Path("drift_bench/prompts/__pycache__/loader.cpython-313.pyc"))
    assert b._is_excluded(Path("drift_bench/data/.DS_Store"))


def test_rater_instructions_ship_with_the_human_validation_files() -> None:
    # The paper quotes the rater app's instructions and says they are released.
    b = _load()
    entry = ("drift_bench/data/human_validation/rater_instructions.md", "human_validation/rater_instructions.md")
    assert entry in b.COPY_PLAN
    assert not b._is_excluded(Path(entry[0]))


def test_out_dir_must_not_overlap_a_source() -> None:
    b = _load()
    with pytest.raises(SystemExit):
        b._check_out_dir(b.ROOT)
    with pytest.raises(SystemExit):
        b._check_out_dir(b.ROOT / "drift_bench" / "data")
    with pytest.raises(SystemExit):
        b._check_out_dir(b.ROOT / "drift_bench" / "data" / "scores" / "stage")
    b._check_out_dir(b.ROOT / "Dataset")


def test_out_dir_must_not_hold_the_card_the_anon_config_or_foreign_files(tmp_path: Path) -> None:
    b = _load()
    # docs/ holds docs/dataset_card.md, a source the build reads.
    with pytest.raises(SystemExit):
        b._check_out_dir(b.ROOT / "docs")
    with pytest.raises(SystemExit):
        b._check_out_dir(b.ROOT / ".anonymize.json")
    # .git is not empty and holds no MANIFEST.txt from an earlier build
    # (in a worktree it is a file, which is refused too).
    if (b.ROOT / ".git").exists():
        with pytest.raises(SystemExit):
            b._check_out_dir(b.ROOT / ".git")
    # Synthetic root: a foreign non-empty dir is refused; an earlier build or an empty dir is not.
    root = tmp_path / "repo"
    root.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="no MANIFEST.txt"):
        b._check_out_dir(foreign, root)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "MANIFEST.txt").write_text("")
    (earlier / "LICENSE").write_text("x")
    b._check_out_dir(earlier, root)
    (tmp_path / "a_file").write_text("x")
    with pytest.raises(SystemExit, match="is a file"):
        b._check_out_dir(tmp_path / "a_file", root)
    (tmp_path / "empty").mkdir()
    b._check_out_dir(tmp_path / "empty", root)
    b._check_out_dir(tmp_path / "new", root)
