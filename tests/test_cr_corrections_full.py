"""Tests for drift_bench.analysis.camera_ready.corrections_full.

The synthetic tests check the LaTeX-to-Markdown rules and the failure modes.
The data tests re-render the released list and compare it with the committed
Markdown; they skip when ``drift_bench/data`` is absent (code-only clone).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drift_bench.analysis.camera_ready import corrections_full as cf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "drift_bench" / "data"
OUT = DATA / "camera_ready" / "analysis"

SYNTH = r"""% a comment line
Intro with \emph{Told} and \texttt{a\_b.csv}.

\begin{itemize}\small
\item \textbf{Run count} (told). $n = \crPublishedN$ became $\crCleanN$; ``8\% to 99\%'' became \crMin--\crMax\%.
\item \textbf{Tests} (regenerated). $p = \crMant \times 10^{\crExp}$ and \crP\ (run-clustered), $\kappa = 0.73$--$0.89$, ${\leq}\,2$.
\item \textbf{Counts} (later). \BlindJudgeRuns\ runs, judge~$\times$~condition, Opus~4.6.
\end{itemize}
"""

VALUES = {
    "crPublishedN": "2{,}146",
    "crCleanN": "2{,}116",
    "crMin": "7.9",
    "crMax": "-1.0",
    "crMant": "2.0",
    "crExp": "-8",
    "crP": "$p = 0.441$",
    "BlindJudgeRuns": "2{,}123",
}


def test_render_synthetic():
    md, used = cf.render(SYNTH, VALUES)
    assert md.startswith("# " + cf.TITLE + "\n\n" + cf.HEADER + "\n\n")
    assert "Intro with *Told* and `a_b.csv`." in md
    assert "- **Run count** (told). n = 2,146 became 2,116; \u201c8% to 99%\u201d became 7.9\u2013-1.0%." in md
    assert "p = 2.0 \u00d7 10\u207b\u2078 and p = 0.441 (run-clustered), \u03ba = 0.73\u20130.89, \u2264 2." in md
    assert "- **Counts** (later). 2,123 runs, judge \u00d7 condition, Opus 4.6." in md
    assert "\\" not in md and "{" not in md
    assert len(used) == 8


def test_undefined_macro_fails():
    with pytest.raises(cf.RenderError, match="undefined macro"):
        cf.render(SYNTH.replace(r"\crCleanN", r"\crCleanNx"), VALUES)


def test_missing_rebut_macro_fails():
    # A "Reproduced" claim prints a \crRebut macro; when its row stops matching the
    # macro is not defined, and the render must fail.
    tex = "x\n\n\\begin{itemize}\n\\item \\textbf{A} (told). Reproduced: \\crRebutGone.\n\\end{itemize}\n"
    with pytest.raises(cf.RenderError, match=r"\\crRebutGone"):
        cf.render(tex, VALUES)


def test_unknown_command_fails():
    with pytest.raises(cf.RenderError, match="left after the render"):
        cf.render(SYNTH.replace("Intro with", r"Intro \footnote with"), VALUES)


def test_math_valued_macro_inside_math_fails():
    with pytest.raises(cf.RenderError, match="inside a math span"):
        cf.render(SYNTH.replace(r"$n = \crPublishedN$", r"$q = \crP$"), VALUES)


def test_number_tokens_reads_superscripts_and_separators():
    assert cf.number_tokens("2{,}116 and 10\u207b\u2078") == ["2,116", "10", "8"]
    assert cf.number_tokens("p = 0.441") == ["0.441"]


def _data_inputs() -> tuple[Path, Path, Path]:
    tex = DATA / cf.TEX_REL
    macros = DATA / cf.MACROS_REL
    release = DATA / cf.RELEASE_MACROS_REL
    if not (tex.is_file() and macros.is_file() and release.is_file() and (OUT / cf.MD_NAME).is_file()):
        pytest.skip("released data absent (code-only clone)")
    return tex, macros, release


def test_committed_markdown_is_current(tmp_path):
    _data_inputs()
    cf.run(DATA, tmp_path)
    assert (tmp_path / cf.MD_NAME).read_bytes() == (OUT / cf.MD_NAME).read_bytes()
    assert (tmp_path / (cf.MD_NAME + ".provenance.json")).is_file()


def test_committed_markdown_has_no_latex():
    _data_inputs()
    md = (OUT / cf.MD_NAME).read_text(encoding="utf-8")
    assert "\\" not in md
    assert "{" not in md and "}" not in md


def test_macro_numbers_equal_cr_macros_json():
    tex_path, macros_path, release_path = _data_inputs()
    values = cf.load_release_macros(release_path)
    values.update(cf.load_cr_macros(macros_path))
    tex = tex_path.read_text(encoding="utf-8")
    marked, used = cf.render(tex, values, marked=True)
    committed = (OUT / cf.MD_NAME).read_text(encoding="utf-8")
    assert marked.replace(cf.OPEN, "").replace(cf.CLOSE, "") == committed
    # Every macro value, as it stands in the Markdown, carries the numbers of cr_macros.json.
    segments = []
    rest = marked
    while cf.OPEN in rest:
        start = rest.index(cf.OPEN) + 1
        end = rest.index(cf.CLOSE, start)
        segments.append(rest[start:end])
        rest = rest[end + 1:]
    assert len(segments) == len(used) > 0
    for seg, u in zip(segments, used):
        assert cf.number_tokens(seg) == cf.number_tokens(u["raw"]), (u["name"], seg, u["raw"])
        assert cf.number_tokens(u["raw"]), u["name"]


def test_not_run_by_item_6():
    # A static render of a committed file: run_analysis item 6 must skip it.
    assert cf.RUN_IN_ITEM_6 is False
