"""Render the full corrections list to Markdown.

The camera-ready paper prints a short corrections appendix. The complete list
(every changed number and every regenerated rebuttal value, with the
"Reproduced" checks) sits in the release as ``corrections_full.tex``. This
module renders that file to ``corrections_full.md`` in plain Markdown with
Unicode math, so it can be read without LaTeX.

Inputs (under ``data_root``):

* ``camera_ready/analysis/corrections_full.tex``: the list, with ``\\cr...``
  macros and literal section numbers.
* ``camera_ready/analysis/cr_macros.json``: the value of every ``\\cr...``
  macro, as the paper prints it.
* ``analysis/release_macros.tex``: the two release counts the list uses
  (``\\BlindJudgeRuns``, ``\\HumanValidationCompletedRaterFiles``).

Outputs (in ``out_dir``): ``corrections_full.md`` and
``corrections_full.md.provenance.json``.

The render fails on any macro that is not defined. ``rebuttal_reproduction``
defines a ``\\crRebut...`` macro only for a posted value that the released
files reproduce, so a "Reproduced" claim whose row stops matching makes the
render fail. The render also fails if any LaTeX command or brace is left.

Run: ``python -m drift_bench.analysis.camera_ready.corrections_full``
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"

TEX_REL = "camera_ready/analysis/corrections_full.tex"
MACROS_REL = "camera_ready/analysis/cr_macros.json"
RELEASE_MACROS_REL = "analysis/release_macros.tex"
MD_NAME = "corrections_full.md"

# A static render of a committed file, not an analysis: run_analysis item 6 skips it.
# tests/test_cr_corrections_full.py re-renders it and fails when the committed
# Markdown no longer matches corrections_full.tex and cr_macros.json.
RUN_IN_ITEM_6 = False

TITLE = "DriftBench camera-ready: full list of corrections"
HEADER = (
    "This is the complete list of corrections that the camera-ready paper's appendix "
    "\u201cCorrections to the submitted version\u201d refers to. That appendix keeps the "
    "corrections that change a claim of the submitted version, the corrections to the "
    "human-subject reporting, and the rebuttal statements that were withdrawn, changed or not "
    "upheld; this list is the complete set of entries that appendix was shortened from. "
    "It is generated from `corrections_full.tex` and "
    "`cr_macros.json` (plus `release_macros.tex` for two release counts) by "
    "`python -m drift_bench.analysis.camera_ready.corrections_full`, so every number is the "
    "value the paper's macros print. The tags told, later and regenerated are the paper's; "
    "they and the reproduced mark are defined in the first paragraph below. Section, appendix, table and "
    "figure numbers refer to the camera-ready paper."
)

# Private-use characters that wrap each expanded macro value during the render.
# They keep a value's text apart from its neighbours (so "--" never merges with a
# negative value) and let the tests find every macro value in the output.
OPEN, CLOSE = "\ue000", "\ue001"

MACRO_RE = re.compile(r"\\([A-Za-z]+)")
NEWCOMMAND_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Commands the converter knows; any other command name left after macro
# expansion is an error.
TEXT_COMMANDS = {"textbf", "emph", "texttt", "item", "begin", "end", "small"}
MATH_SYMBOLS = {
    "times": "\u00d7",
    "rho": "\u03c1",
    "kappa": "\u03ba",
    "chi": "\u03c7",
    "to": "\u2192",
    "le": "\u2264",
    "leq": "\u2264",
    "ge": "\u2265",
    "geq": "\u2265",
}
SUPERSCRIPT = str.maketrans("0123456789-+", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a")
FROM_SUPERSCRIPT = {v: k for k, v in zip("0123456789-+", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a")}


class RenderError(ValueError):
    """The tex cannot be rendered: an undefined macro or a leftover command."""


# ---------------------------------------------------------------------------
# Macro values
# ---------------------------------------------------------------------------


def load_cr_macros(path: Path) -> dict[str, str]:
    """``{name without backslash: value}`` from ``cr_macros.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {m["name"].lstrip("\\"): m["value"] for m in data["macros"]}


def load_release_macros(path: Path) -> dict[str, str]:
    """``{name: value}`` for every one-line ``\\newcommand`` in ``release_macros.tex``."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = NEWCOMMAND_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def is_value_macro(name: str) -> bool:
    """A paper macro: ``\\crX...`` or a capitalised release macro."""
    return bool(re.match(r"cr[A-Z]", name)) or name[:1].isupper()


# ---------------------------------------------------------------------------
# LaTeX to Markdown
# ---------------------------------------------------------------------------


def _strip_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(COMMENT_RE.sub("", line))
    return "\n".join(lines)


def _braced(text: str, start: int) -> tuple[str, int]:
    """The argument of the ``{`` at ``start`` and the index after its ``}``."""
    if start >= len(text) or text[start] != "{":
        raise RenderError(f"expected '{{' at {start}: {text[max(0, start - 30):start + 30]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise RenderError(f"unbalanced braces after {text[max(0, start - 30):start + 30]!r}")


def _superscript(arg: str) -> str:
    core = arg.replace(OPEN, "").replace(CLOSE, "")
    if core and all(ch in "0123456789-+" for ch in core):
        return arg.translate(SUPERSCRIPT)
    return "^(" + arg + ")"


def math_to_text(math: str) -> str:
    """Convert the inside of a ``$...$`` span to plain Unicode text."""
    s = math.replace("{,}", ",").replace("\\%", "%").replace("\\,", " ")

    def sup(m: re.Match) -> str:
        if m.group(1) is not None:
            return _superscript(m.group(1))
        return _superscript(m.group(2))

    s = re.sub(r"\^(?:\{([^{}]*)\}|([0-9A-Za-z]))", sup, s)

    def sym(m: re.Match) -> str:
        name = m.group(1)
        if name not in MATH_SYMBOLS:
            raise RenderError(f"unknown math command \\{name} in ${math}$")
        return MATH_SYMBOLS[name]

    s = re.sub(r"\\([A-Za-z]+)", sym, s)
    s = s.replace("{", "").replace("}", "")
    return s


def _replace_command(text: str, name: str, fmt) -> str:
    """Replace ``\\name{arg}`` by ``fmt(arg)``, innermost first, until none is left."""
    token = "\\" + name + "{"
    while True:
        i = text.find(token)
        if i < 0:
            return text
        arg, end = _braced(text, i + len(token) - 1)
        text = text[:i] + fmt(arg) + text[end:]


def text_to_markdown(text: str) -> str:
    """Convert a LaTeX text fragment (macros already expanded) to Markdown."""
    s = re.sub(r"(?<!\\)\$([^$]*)\$", lambda m: math_to_text(m.group(1)), text)
    s = _replace_command(s, "texttt", lambda a: "`" + a.replace("\\_", "_") + "`")
    s = _replace_command(s, "textbf", lambda a: "**" + a + "**")
    s = _replace_command(s, "emph", lambda a: "*" + a + "*")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("{,}", ",").replace("{}", "")
    s = s.replace("\\%", "%").replace("\\$", "$").replace("\\_", "_").replace("\\&", "&")
    s = s.replace("\\ ", " ").replace("\\,", " ").replace("~", " ")
    s = re.sub(r"(?<!-)--(?!-)", "\u2013", s)
    s = re.sub(r" {2,}", " ", s)
    return s


def value_to_markdown(value: str) -> str:
    """A macro's printed LaTeX value, as Markdown text."""
    return text_to_markdown(value)


def expand_macros(tex: str, values: dict[str, str]) -> tuple[str, list[dict]]:
    """Replace every value macro by its Markdown value, wrapped in OPEN/CLOSE.

    Returns the text and one record per occurrence. Raises on an undefined
    value macro and on a ``$``-valued macro used inside math.
    """
    out: list[str] = []
    used: list[dict] = []
    pos = 0
    for m in MACRO_RE.finditer(tex):
        name = m.group(1)
        if not is_value_macro(name):
            continue
        if name not in values:
            raise RenderError(f"undefined macro \\{name}")
        raw = values[name]
        in_math = len(re.findall(r"(?<!\\)\$", tex[:m.start()])) % 2 == 1
        if in_math and "$" in raw:
            raise RenderError(f"\\{name} prints math ({raw!r}) inside a math span")
        end = m.end()
        if tex.startswith("{}", end):
            end += 2
        md = math_to_text(raw) if in_math else value_to_markdown(raw)
        out.append(tex[pos:m.start()])
        out.append(OPEN + md + CLOSE)
        used.append({"name": "\\" + name, "raw": raw, "markdown": md})
        pos = end
    out.append(tex[pos:])
    return "".join(out), used


def _check_clean(md: str) -> None:
    left = sorted(set(re.findall(r"\\[A-Za-z]+|\\.", md)))
    if left:
        raise RenderError(f"LaTeX commands left after the render: {left}")
    if "{" in md or "}" in md:
        i = md.find("{") if "{" in md else md.find("}")
        raise RenderError(f"brace left after the render: {md[max(0, i - 40):i + 40]!r}")


def split_list(tex: str) -> tuple[list[str], list[str]]:
    """Intro paragraphs and ``\\item`` bodies of the corrections list."""
    tex = _strip_comments(tex)
    m = re.search(r"\\begin\{itemize\}(?:\\small)?(.*?)\\end\{itemize\}", tex, re.S)
    if not m:
        raise RenderError("no itemize environment in the corrections list")
    intro = [p.strip() for p in re.split(r"\n\s*\n", tex[:m.start()]) if p.strip()]
    items = [" ".join(x.split()) for x in re.split(r"\\item\s+", m.group(1)) if x.strip()]
    if tex[m.end():].strip():
        raise RenderError("text after the itemize environment")
    return intro, items


def render(tex: str, values: dict[str, str], marked: bool = False) -> tuple[str, list[dict]]:
    """Render the corrections list to Markdown.

    With ``marked=True`` every macro value stays wrapped in OPEN/CLOSE (for tests).
    """
    intro, items = split_list(tex)
    blocks = [f"# {TITLE}", HEADER]
    used: list[dict] = []
    for para in intro:
        body, u = expand_macros(" ".join(para.split()), values)
        used += u
        blocks.append(text_to_markdown(body))
    lines = []
    for item in items:
        body, u = expand_macros(item, values)
        used += u
        lines.append("- " + text_to_markdown(body))
    blocks.append("\n".join(lines))
    md = "\n\n".join(blocks) + "\n"
    _check_clean(md)
    if not marked:
        md = md.replace(OPEN, "").replace(CLOSE, "")
    return md, used


def number_tokens(text: str) -> list[str]:
    """Number tokens of a value or rendered text, with superscripts read as digits."""
    plain = "".join(FROM_SUPERSCRIPT.get(ch, ch) for ch in text)
    plain = plain.replace("{,}", ",")
    return NUMBER_RE.findall(plain)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Render ``corrections_full.md`` and its provenance into ``out_dir``."""
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir is not None else data_root / "camera_ready" / "analysis"
    tex_path = data_root / TEX_REL
    macros_path = data_root / MACROS_REL
    release_path = data_root / RELEASE_MACROS_REL
    for p in (tex_path, macros_path, release_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing input {p.name} under the data root")
    values = load_release_macros(release_path)
    values.update(load_cr_macros(macros_path))
    tex = tex_path.read_text(encoding="utf-8")
    md, used = render(tex, values)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / MD_NAME
    md_path.write_text(md, encoding="utf-8")
    _, items = split_list(tex)
    prov = {
        "description": (
            "Markdown render of the full corrections list that the camera-ready paper's "
            "corrections appendix refers to. Every \\cr... macro is replaced by its value in "
            "cr_macros.json and the two release macros by their values in release_macros.tex; "
            "the LaTeX markup is converted to Markdown and Unicode. The render fails on an "
            "undefined macro or on any LaTeX command left over."
        ),
        "inputs": [
            {"path": TEX_REL, "sha256": _sha256(tex_path)},
            {"path": MACROS_REL, "sha256": _sha256(macros_path)},
            {"path": RELEASE_MACROS_REL, "sha256": _sha256(release_path)},
        ],
        "entries": len(items),
        "macro_occurrences": len(used),
        "distinct_macros": len({u["name"] for u in used}),
        "command": "python -m drift_bench.analysis.camera_ready.corrections_full",
        "output": MD_NAME,
        "date": dt.date.today().isoformat(),
    }
    with open(out_dir / (MD_NAME + ".provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    titles = [re.match(r"\\textbf\{(.*?)\}\s*\(([^)]*)\)", it) for it in items]
    entries = pd.DataFrame(
        {
            "entry": [m.group(1) if m else "" for m in titles],
            "tag": [m.group(2) if m else "" for m in titles],
        }
    )
    return {"corrections_full_entries": entries}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = run(args.data_root, args.out_dir)["corrections_full_entries"]
    out_dir = args.out_dir if args.out_dir is not None else Path(args.data_root) / "camera_ready" / "analysis"
    print(f"wrote {Path(out_dir) / MD_NAME}: {len(out)} entries")


if __name__ == "__main__":
    main()
