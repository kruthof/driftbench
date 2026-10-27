"""The paper's macro guard and the macros it loads (plan B19, X6, X8).

Three checks on real files, no network, no API keys:

(a) every macro defined in both the tex fallback block and the generated
    ``release_macros.tex`` has the same value (skipped when the generated
    file is absent, as on a clean checkout);
(b) every ``\\IfFileExists{A}{...}`` guard in the preamble ``\\input``s the
    same file A it tests for (the July typo tested ``release_macros2.tex``
    and so always used the fallback);
(c) no macro value in the fallback block, the generated file, or
    ``cr_macros.tex`` is ``None`` or ``nan``.

Plus unit tests for the generator's item-count fallback on synthetic files.

Checks (a) and (b), and the fallback-block part of (c)
(``test_fallback_has_no_none``), read the paper source ``neurips_2026.tex``.
The code-only public repository does not ship it, so they skip there. The
other two (c) checks read only the generated ``release_macros.tex`` and
``cr_macros.tex``; each skips when its file is absent (it exists after
``reproduce.sh --hf`` or a local analysis run). The unit tests always run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from drift_bench.analysis import release_manifest as rm

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "neurips_2026.tex"
RELEASE_MACROS = ROOT / "drift_bench" / "data" / "analysis" / "release_macros.tex"
CR_MACROS = ROOT / "drift_bench" / "data" / "camera_ready" / "analysis" / "cr_macros.tex"

BAD_VALUE_RE = re.compile(r"\b(None|nan|NaN)\b")


# ---------------------------------------------------------------------------
# Small TeX parsers (brace matching; values may nest braces like 2{,}146)
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    """Drop ``%`` comments. ``\\%`` is an escaped percent and stays."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _braced_group(text: str, open_idx: int) -> tuple[str, int]:
    """Return the content of the brace group opening at ``open_idx`` and the
    index just past its closing brace."""
    assert text[open_idx] == "{", text[open_idx : open_idx + 20]
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def _next_group(text: str, idx: int) -> tuple[str, int]:
    """Skip whitespace from ``idx`` and read the next brace group."""
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    return _braced_group(text, idx)


def parse_newcommands(text: str) -> dict[str, str]:
    """``\\newcommand{\\Name}{value}`` pairs, in order. Comments are ignored."""
    text = _strip_comments(text)
    out: dict[str, str] = {}
    for m in re.finditer(r"\\newcommand\s*(?=\{)", text):
        name_group, nxt = _braced_group(text, m.end())
        value, _ = _next_group(text, nxt)
        name = name_group.strip().lstrip("\\")
        assert name not in out, f"{name} defined twice"
        out[name] = value.strip()
    return out


def parse_guards(text: str) -> list[dict[str, object]]:
    """Every ``\\IfFileExists{file}{true}{false}`` with both branches."""
    text = _strip_comments(text)
    guards = []
    for m in re.finditer(r"\\IfFileExists\s*(?=\{)", text):
        file_arg, nxt = _braced_group(text, m.end())
        true_branch, nxt = _next_group(text, nxt)
        false_branch, _ = _next_group(text, nxt)
        guards.append({"file": file_arg.strip(), "true": true_branch, "false": false_branch})
    return guards


def _inputs(branch: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\\input\s*\{([^}]*)\}", branch)]


def _preamble(text: str) -> str:
    return text.split(r"\begin{document}", 1)[0]


@pytest.fixture(scope="module")
def tex_text() -> str:
    if not TEX.exists():
        pytest.skip("neurips_2026.tex absent (code-only release ships no paper source)")
    return TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def release_guard(tex_text: str) -> dict[str, object]:
    guards = [g for g in parse_guards(_preamble(tex_text)) if g["file"].endswith("release_macros.tex")]
    assert len(guards) == 1, f"expected one release_macros guard, found {len(guards)}"
    return guards[0]


@pytest.fixture(scope="module")
def fallback_macros(release_guard: dict[str, object]) -> dict[str, str]:
    macros = parse_newcommands(str(release_guard["false"]))
    assert macros, "fallback block defines no macros"
    return macros


# ---------------------------------------------------------------------------
# (b) the guard tests the file it inputs
# ---------------------------------------------------------------------------


def test_guard_inputs_the_file_it_tests(tex_text: str) -> None:
    guards = parse_guards(_preamble(tex_text))
    assert guards, "no \\IfFileExists guard in the preamble"
    for g in guards:
        inputs = _inputs(str(g["true"]))
        assert inputs == [g["file"]], f"guard tests {g['file']!r} but inputs {inputs!r}"
        assert "release_macros2" not in g["file"]


def test_guard_paths_are_repo_relative(tex_text: str) -> None:
    for g in parse_guards(_preamble(tex_text)):
        assert not str(g["file"]).startswith("/"), g["file"]
        assert str(g["file"]).startswith("drift_bench/data/"), g["file"]


def test_cr_macros_not_loaded_twice(tex_text: str, release_guard: dict[str, object]) -> None:
    """The true branch loads release_macros.tex, which itself inputs
    cr_macros.tex. Loading cr_macros.tex again there would redefine 1,500
    macros with \\newcommand and fail. The false branch may load it directly."""
    assert not any("cr_macros" in p for p in _inputs(str(release_guard["true"])))
    fallback_guards = [g for g in parse_guards(str(release_guard["false"])) if "cr_macros" in str(g["file"])]
    assert len(fallback_guards) == 1
    assert _inputs(str(fallback_guards[0]["true"])) == [fallback_guards[0]["file"]]
    if RELEASE_MACROS.exists():
        text = _strip_comments(RELEASE_MACROS.read_text(encoding="utf-8"))
        assert text.count("\\input{") == 1, "release_macros.tex should input cr_macros.tex exactly once"
        assert "cr_macros.tex" in text


# ---------------------------------------------------------------------------
# (a) fallback block and generated file agree
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RELEASE_MACROS.exists(), reason="generated release_macros.tex absent (clean checkout)")
def test_fallback_matches_generated(fallback_macros: dict[str, str]) -> None:
    generated = parse_newcommands(RELEASE_MACROS.read_text(encoding="utf-8"))
    assert generated, "generated file defines no macros"
    shared = sorted(set(fallback_macros) & set(generated))
    assert shared, "fallback and generated file share no macro names"
    diffs = [
        f"  \\{name}: fallback={fallback_macros[name]!r} generated={generated[name]!r}"
        for name in shared
        if " ".join(fallback_macros[name].split()) != " ".join(generated[name].split())
    ]
    assert not diffs, "fallback block is stale; differences:\n" + "\n".join(diffs)


@pytest.mark.skipif(not RELEASE_MACROS.exists(), reason="generated release_macros.tex absent (clean checkout)")
def test_fallback_and_generated_define_same_names(fallback_macros: dict[str, str]) -> None:
    generated = parse_newcommands(RELEASE_MACROS.read_text(encoding="utf-8"))
    assert set(fallback_macros) == set(generated), (
        f"only in fallback: {sorted(set(fallback_macros) - set(generated))}; "
        f"only in generated: {sorted(set(generated) - set(fallback_macros))}"
    )


# ---------------------------------------------------------------------------
# (c) no None / nan anywhere
# ---------------------------------------------------------------------------


def _assert_no_bad_values(macros: dict[str, str], label: str) -> None:
    bad = {name: value for name, value in macros.items() if BAD_VALUE_RE.search(value)}
    assert not bad, f"{label}: macros with None/nan values: {bad}"


def test_fallback_has_no_none(fallback_macros: dict[str, str]) -> None:
    _assert_no_bad_values(fallback_macros, "fallback block")


@pytest.mark.skipif(not RELEASE_MACROS.exists(), reason="generated release_macros.tex absent")
def test_generated_has_no_none() -> None:
    text = RELEASE_MACROS.read_text(encoding="utf-8")
    _assert_no_bad_values(parse_newcommands(text), "release_macros.tex")
    assert not BAD_VALUE_RE.search(_strip_comments(text))


@pytest.mark.skipif(not CR_MACROS.exists(), reason="cr_macros.tex absent")
def test_cr_macros_have_no_none() -> None:
    text = CR_MACROS.read_text(encoding="utf-8")
    macros = parse_newcommands(text)
    assert macros
    _assert_no_bad_values(macros, "cr_macros.tex")
    for name in macros:
        assert re.fullmatch(r"cr[A-Z][A-Za-z]*", name), f"bad cr macro name {name}"


# ---------------------------------------------------------------------------
# Generator: the item-count fallback and the None refusal (synthetic files)
# ---------------------------------------------------------------------------


def test_scoring_set_size_prefers_item_list(tmp_path: Path) -> None:
    items = tmp_path / "items.json"
    items.write_text(json.dumps([{"run_id": f"r{i}"} for i in range(7)]))
    raters = tmp_path / "raters"
    raters.mkdir()
    (raters / "human_rater_set2_rater_A.json").write_text(json.dumps({"r0": {}, "r1": {}}))
    n, source = rm._scoring_set_size("set2", items, None, raters_dir=raters)
    assert n == 7
    assert source.endswith("items.json")


def test_scoring_set_size_falls_back_to_csv_then_raters(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    csv = tmp_path / "items.csv"
    csv.write_text("run_id,item_number\nr1,1\nr2,2\nr1,3\n")
    raters = tmp_path / "raters"
    raters.mkdir()
    (raters / "human_rater_set2_rater_A.json").write_text(json.dumps({"a": {}, "b": {}, "c": {}}))
    (raters / "human_rater_set2_rater_C.json").write_text(json.dumps({"b": {}, "c": {}, "d": {}}))
    (raters / "human_rater_set1_rater_A.json").write_text(json.dumps({"x": {}}))

    n, source = rm._scoring_set_size("set2", missing, csv, raters_dir=raters)
    assert (n, source.endswith("items.csv")) == (2, True)

    n, source = rm._scoring_set_size("set2", missing, tmp_path / "missing.csv", raters_dir=raters)
    assert n == 4  # distinct run_ids across the set-2 rater files only
    assert "human_rater_set2_" in source

    with pytest.raises(FileNotFoundError, match="set2"):
        rm._scoring_set_size("set2", missing, None, raters_dir=tmp_path / "none")


@pytest.mark.parametrize("value", ["None", "nan", "NaN", "55/None", "55/nan", "set None"])
def test_write_tex_macros_refuses_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(rm, "_build_manuscript_macros", lambda manifest: {"X": value})
    with pytest.raises(ValueError, match="X"):
        rm._write_tex_macros({}, tmp_path / "m.tex")
    assert not (tmp_path / "m.tex").exists()


def test_write_tex_macros_accepts_words_containing_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the whole words None / nan are refused; 'nonetheless' and 'nano' pass."""
    monkeypatch.setattr(rm, "_build_manuscript_macros", lambda manifest: {"X": "nonetheless 55", "Y": "nano"})
    rm._write_tex_macros({}, tmp_path / "m.tex")
    assert parse_newcommands((tmp_path / "m.tex").read_text()) == {"X": "nonetheless 55", "Y": "nano"}


# ---------------------------------------------------------------------------
# Generator: the five-model core subset comes from config.yaml
# ---------------------------------------------------------------------------

CORE_FIVE = [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3.1-pro-preview",
    "openai/gpt-5.4-mini",
    "gemini/gemini-3.1-flash-lite-preview",
]


def test_core_model_ids_from_real_config() -> None:
    assert rm._core_model_ids() == CORE_FIVE


def test_core_subset_ignores_openweight_rows_without_openweight_parquet(tmp_path: Path) -> None:
    """all_scores.parquet holds all seven models. With an empty open-weight
    frame the subset must still be the five config models, not all seven."""
    core = pd.DataFrame({
        "run_id": [f"r{i}" for i in range(7)],
        "model_id": CORE_FIVE + ["together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"],
    })
    openweight = pd.DataFrame()
    assert openweight.empty
    benchmark = pd.concat([core, openweight], ignore_index=True) if not openweight.empty else core
    benchmark = benchmark.drop_duplicates("run_id", ignore_index=True)
    assert len(benchmark) == 7

    config = tmp_path / "config.yaml"
    config.write_text("models:\n  subjects:\n" + "".join(f"    - id: \"{m}\"\n" for m in CORE_FIVE))
    core_five = rm._core_subset(core, rm._core_model_ids(config))
    assert sorted(core_five["model_id"]) == sorted(CORE_FIVE)
    assert len(core_five) == 5


def test_core_subset_without_model_column_is_unchanged() -> None:
    core = pd.DataFrame({"run_id": ["a", "b"]})
    assert rm._core_subset(core, CORE_FIVE).equals(core)
