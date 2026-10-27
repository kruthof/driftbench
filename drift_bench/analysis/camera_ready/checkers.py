"""Programmatic checkers for the deterministically checkable hard constraints (plan item C11).

Three parts.

1. A classifier (`classify_constraint`) that labels a constraint string with a
   `checker_type` and a `checkable` flag. The hand-verified labels and the rule
   parameters live in `checker_rules.yaml` next to this file.
2. A checker runtime (`check_constraint`) that applies a rule to the final
   non-probe assistant message of a run. It returns `satisfied`, `violated` or
   `undetermined` with the evidence it used.
3. Agreement of the checker with the structured judge's `constraint_checks`
   and with the human per-constraint median labels.

A constraint counts as checkable only when a regex plus arithmetic over the
proposal text decides it. The runtime never guesses: no stated quantity means
`undetermined`, and conflicting stated quantities mean `undetermined`.

Usage: python -m drift_bench.analysis.camera_ready.checkers
The human-agreement rows read the text the raters saw from
camera_ready/human_validation_set{1,2}_items.csv. Write them once with
`--export-set1 scoring_app_standalone/scoring_items.json` and
`--export-set2 scoring_app_standalone_set2/scoring_items_set2.json`.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE / "checker_rules.yaml"
PKG_ROOT = HERE.parents[1]  # drift_bench/
DEFAULT_DATA_ROOT = PKG_ROOT / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
DEFAULT_BRIEFS_DIR = PKG_ROOT / "briefs"

CHECKABLE_TYPES = (
    "numeric_cap_count",
    "numeric_cap_budget",
    "numeric_cap_duration",
    "numeric_cap_quantity",
    "numeric_range",
    "exact_count",
    "threshold_restatement",
)
# threshold_restatement: the stated number is a definition echo ("below 80% of AMI");
# the checker tests restatement, not compliance, so agreement tables exclude it.
AGREEMENT_TYPES = tuple(t for t in CHECKABLE_TYPES if t != "threshold_restatement")
NOT_CHECKABLE_TYPES = ("enumerated_choice", "forbidden_term", "required_term", "none")
ALL_TYPES = CHECKABLE_TYPES + NOT_CHECKABLE_TYPES

SATISFIED, VIOLATED, UNDETERMINED = "satisfied", "violated", "undetermined"

# ---------------------------------------------------------------------------
# Briefs and rules
# ---------------------------------------------------------------------------


def load_briefs(briefs_dir: Path = DEFAULT_BRIEFS_DIR) -> dict[str, list[str]]:
    """Return {brief_id: [hard constraints]} in file-name order."""
    out: dict[str, list[str]] = {}
    for path in sorted(Path(briefs_dir).glob("*.yaml")):
        brief = yaml.safe_load(path.read_text())
        out[brief["id"]] = list(brief["hard_constraints"])
    return out


def load_rules(path: Path = RULES_PATH) -> list[dict]:
    text = Path(path).read_text()
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
# A number token: not glued to a letter, digit or dot before it ("F1", "0.20"), not an identifier
# hyphenation ("top-5", "SIB-200"; a digit-hyphen range like "60-100" is allowed), and no ordinal suffix.
NUM = r"(?:(?<![\w.])(?<![A-Za-z]-)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!(?:st|nd|rd|th)\b)|\b(?:" + "|".join(_WORD_NUM) + r")\b)"
DURATION_UNIT_WORDS = r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|term"
PHYS_UNIT_WORDS = (r"ppm|km|cm|mA|mm\s?Hg|°C|\+/-\s*\d+\s*C|\bC\b|%|percent|percentage points|-?point|parameters?|billion|million|[MB]-parameter|\dB\b|\d+M\b|"
                   r"micrometers?|x\b|\bz\b|redshift|\d{2}:\d{2}|\s?[MGK]B\b|lines of")
CMP_WORDS = (r"at least|minimum|no more than|not exceed|capped|fewer than|maximum|under|below|between|or better|or more|up to|at most|"
             r"within \d|exactly|fixed at|\bper\b|bounded at|never exceed|stay under|remain under")
DEFINITION_WORDS = (r"defined as|is defined|metric is|outcome is|unit of analysis is|operationali[sz]ed|measured as|measured by|measured from|"
                    r"measured using|measured exclusively|derived from|must be a (?:nationally|single dependent)|is the single|must be \w+ comments")
DESIGN_START = (r"^(?:retrospective|prospective|observational|quasi-experimental|randomi[sz]ed|cluster-randomi[sz]ed|individually randomi[sz]ed|"
                r"between-subjects|within-subjects|ecologic|field trial|cell culture|human participants|event study|paired design|single platform|"
                r"use only|analysis restricted to|sample restricted to|ancestry restricted|patients limited to|participants must be|participants are)")


def classify_constraint(text: str) -> dict:
    """Heuristic label for one constraint string.

    Returns {"checker_type", "checkable", "reason"}. The YAML keeps the hand
    label; `label_source` there records where the two differ. Order of the
    tests: numeric forms first, then definitions and design types, then the
    lexical forms (choice, required, forbidden).
    """
    t = text.strip()
    low = t.lower()
    has_num = re.search(NUM, low) is not None
    has_cmp = re.search(CMP_WORDS, low) is not None
    has_dur = re.search(NUM + r"(?:\s*(?:-|to)\s*" + NUM + r")?[\s-]*(?:(?:full|academic|consecutive|calendar)\s+){0,2}(?:" + DURATION_UNIT_WORDS + r")\b", low) is not None
    has_year_range = re.search(r"(?:19|20)\d\d\s*-\s*(?:19|20)\d\d", low) is not None
    has_phys = re.search(NUM + r"\s*(?:" + PHYS_UNIT_WORDS + r")", t) is not None
    has_exact = re.search(r"\bexactly\b", low) is not None
    has_range = re.search(r"between\s+" + NUM + r"\s+and\s+" + NUM + r"|" + NUM + r"\s*(?:-|to)\s*" + NUM + r"|\+/-", low) is not None
    has_count_noun = re.search(NUM + r"\s+(?:[\w-]+\s+){0,2}?(?:sites?|clusters?|cities|conditions?|participants?|trials?|objects?|schools?|households?|"
                               r"utilities|children|counties|firms|patients|arms?|languages?|pairs?|specimens?|vessels?|attempts?|demonstrations?|"
                               r"layouts?|obstacles?|examples?|jurisdictions?|areas?|students?|observations?|classes|ratios?|accounts?|users?|residents|"
                               r"functions?|lines|calls|packages?|endpoints?|tables?|files?|retries|requests?|workers?|attempts?|tasks?|entries|tokens?|"
                               r"operations?|commands?|formats?|dependenc(?:y|ies))", low) is not None

    if "budget" in low and re.search(r"gbp|usd|£|\$", low):
        return _lab("numeric_cap_budget", True, "budget cap with a currency amount")
    if has_dur and (has_exact or has_cmp or has_range or re.search(r"\bfixed\b|\bfor\b\s*" + NUM + r"|\bover\b\s*" + NUM + r"|lasting|span|duration", low)):
        return _lab("numeric_cap_duration", True, "duration with a limit, an exact value or a range")
    if has_year_range:
        return _lab("numeric_cap_duration", True, "calendar window")
    if has_exact and has_num and not has_phys:
        return _lab("exact_count", True, "exact count")
    if has_phys and has_range:
        return _lab("numeric_range", True, "physical quantity with a range")
    if has_range and (has_count_noun or re.search(r"aged?\b|sample size", low)):
        return _lab("numeric_range", True, "count or age range")
    if has_phys and (has_cmp or has_exact or re.search(r"\bat\b\s*" + NUM + r"|\bis\b\s*" + NUM + r"|\bof\b\s*" + NUM + r"|held at|above|below", low)):
        return _lab("numeric_cap_quantity", True, "physical quantity with a limit")
    if has_count_noun and has_cmp:
        return _lab("numeric_cap_count", True, "count with a comparator")
    if re.search(DEFINITION_WORDS, low):
        return _lab("none", False, "definition or operationalisation; semantic equivalence needs judgement")
    if re.search(DESIGN_START, low):
        return _lab("none", False, "design type or sample scope; a property of the whole design")
    if re.search(r"\bor\b", low) and re.search(r"\bonly\b|not both|or equivalent|single|not a mixture|not multiple", low):
        return _lab("enumerated_choice", False, "choice among named options needs reading of the design")
    if re.match(r"^(?:no|not|never)\b", low) or re.search(r"\bmust not\b|must exclude|exclude\b|permitted|allowed|, no\b|; no\b", low):
        return _lab("forbidden_term", False, "bans a term; presence cannot be separated from negation and restatement")
    if re.search(r"^must\b|\bmust\b", low):
        return _lab("required_term", False, "requires an element; presence of its words equals a restatement")
    return _lab("none", False, "design type, definition or procedure; no regex decides it")


def _lab(ctype: str, checkable: bool, reason: str) -> dict:
    return {"checker_type": ctype, "checkable": checkable, "reason": reason}


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

_UNICODE_MAP = {
    "–": "-", "—": "-", "‒": "-", "−": "-", "×": "x", " ": " ",
    " ": " ", " ": " ", "≈": "~", "≥": ">=", "≤": "<=", "⩾": ">=",
    "⩽": "<=", "±": "+/-", "’": "'", "“": '"', "”": '"', "₂": "2",
}


_LATEX_MAP = [
    (r"\\geq?\b", ">="), (r"\\leq?\b", "<="), (r"\\gt\b", ">"), (r"\\lt\b", "<"), (r"\\approx\b", "~"),
    (r"\\sim\b", "~"), (r"\\pm\b", "+/-"), (r"\\times\b", "x"), (r"\\%", "%"), (r"\\,|\\;|\\ ", " "),
    (r"\\(?:text|mathrm|textbf|mathit)\{([^}]*)\}", r"\1"),
]
_MATH_SPAN = re.compile(r"\$([^$\n]{1,80})\$")


def _strip_math_span(m: re.Match) -> str:
    """Drop the `$` delimiters of an inline-math span; keep currency amounts."""
    body = m.group(1)
    if re.search(r"[\\=<>^_~]", body) or re.match(r"\s*[A-Za-z]", body):
        return body
    return m.group(0)


def normalise_text(text: str) -> str:
    """Map typographic variants and LaTeX comparators to ASCII; strip markdown emphasis."""
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    for rx, dst in _LATEX_MAP:
        text = re.sub(rx, dst, text)
    text = _MATH_SPAN.sub(_strip_math_span, text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?<!\w)\*(?!\s)", "", text)
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    return text


_ALT_HEAD = re.compile(
    r"^\s*(?:#{1,6}\s*|\d+[.)]\s*|[ivx]+[.)]\s*)\s*(?:\d+[.)]\s*)?(?:two\s+|2\s+)?(?:plausible\s+)?alternatives?\b[^.]{0,80}$"
    r"|^\s*(?:two\s+|2\s+)?plausible\s+alternatives?\b[^.]{0,60}$"
    r"|^\s*alternatives?\s+not\s+chosen\b[^.]{0,60}$", re.I)
_SECTION_HEAD = re.compile(r"^\s*(?:#{1,6}\s*|\d+[.)]\s*|[ivx]+[.)]\s*|-\s*)?\s*(?:\d+[.)]\s*)?(?:final\s+)?constraints?\s+checklist|^\s*(?:#{1,6}\s*|\d+[.)]\s*)?\s*(?:\d+[.)]\s*)?final\s+constraint|^\s*#{1,6}\s.*(?:checklist|constraint)", re.I)


def strip_alternatives(text: str) -> tuple[str, bool]:
    """Remove the 'plausible alternatives not chosen' section.

    The final-turn prompt fixes the section order: alternatives, then the
    constraint checklist. Numbers inside the alternatives section describe
    designs the proposal rejects, so the checker ignores them. The cut runs
    from the alternatives heading to the checklist heading, or to the end of
    the text when no checklist heading follows.
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if len(line) <= 120 and _ALT_HEAD.match(line):
            start = i
            break
    if start is None:
        return text, False
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if len(lines[j]) <= 120 and _SECTION_HEAD.match(lines[j]):
            end = j
            break
    removed = sum(len(l) for l in lines[start:end])
    if removed > 0.5 * max(1, len(text)):
        # A cut this large means the heading was not the alternatives section.
        return text, False
    return "\n".join(lines[:start] + lines[end:]), True


def split_units(text: str) -> list[str]:
    """Split into lines, then into sentences, for context matching."""
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", line)
        out.extend(p for p in parts if p)
    return out


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------

MIN_CMP = r"at least|a minimum of|minimum of|minimum|min\.|\bmin\b|no fewer than|not fewer than|not less than|no less than|>=|more than|greater than|>"
MAX_CMP = (r"no more than|not more than|at most|a maximum of|maximum of|maximum|max\.|\bmax\b|up to|under|below|fewer than|less than|"
           r"not exceed(?:ing)?|not to exceed|does not exceed|capped at|cap of|within|<=|<")
EXACT_CMP = r"exactly|precisely"
APPROX_CMP = r"approximately|approx\.|approx|about|~|roughly|around|circa"
BETWEEN_CMP = r"between"
CMP = rf"(?P<cmp>{MIN_CMP}|{MAX_CMP}|{EXACT_CMP}|{APPROX_CMP}|{BETWEEN_CMP})"
CUR = r"(?:£|\$|GBP|USD|US\$|EUR|€)"
MULT = r"(?:k|m|bn|thousand|million|billion)\b"
ADJ = r"(?:[A-Za-z][\w-]*\s+){0,2}?"
GENERIC_GROUP = (r"arms?|conditions?|groups?|cells?|sections?|sites?|schools?|languages?|objects?|methods?|temperatures?|"
                 r"levels?|class(?:es)?|clusters?|blends?|ratios?|items?|participants?|subjects?|layouts?|domains?|pairs?|"
                 r"utilit(?:y|ies)|treatments?|settings?|strata|stratum|vessels?|jurisdictions?|polic(?:y|ies)")

DURATION_UNITS = {
    "seconds": (r"seconds?|secs?\b|s\b", 1 / 86400),
    "minutes": (r"minutes?|mins?\b", 1 / 1440),
    "hours": (r"hours?|hrs?\b|h\b", 1 / 24),
    "days": (r"days?\b", 1.0),
    "weeks": (r"weeks?|wks?\b", 7.0),
    "months": (r"months?|mos?\b", 30.4375),
    "years": (r"years?|yrs?\b", 365.25),
    "academic_years": (r"academic\s+years?|school\s+years?", 365.25),
}
FREQ_WORDS = [
    (r"every\s+other\s+day|alternate\s+days", 2.0),
    (r"twice\s+(?:weekly|a\s+week|per\s+week)|two\s+times\s+(?:a|per)\s+week", 3.5),
    (r"three\s+times\s+(?:a|per)\s+week|thrice\s+weekly", 7 / 3),
    (r"daily|every\s+day|each\s+day|once\s+a\s+day", 1.0),
    (r"weekly|every\s+week|each\s+week|once\s+a\s+week|once\s+per\s+week", 7.0),
    (r"fortnightly|every\s+two\s+weeks|every\s+2\s+weeks|every\s+other\s+week", 14.0),
    (r"monthly|every\s+month|once\s+a\s+month", 30.4375),
]


@dataclass
class Mention:
    cmp: str  # value | min | max | range | dims
    value: float
    value2: float | None
    snippet: str
    start: int = 0
    end: int = 0
    verdict: str = UNDETERMINED


def parse_number(tok: str) -> float:
    tok = tok.strip().lower().replace(",", "")
    if tok in _WORD_NUM:
        return float(_WORD_NUM[tok])
    return float(tok)


def apply_mult(value: float, mult: str | None, kind: str) -> float:
    if not mult:
        return value
    m = mult.lower()
    if m in ("k", "thousand"):
        return value * 1e3
    if kind == "budget" and m in ("m", "million"):
        return value * 1e6
    if kind == "budget" and m in ("bn", "billion"):
        return value * 1e9
    return value


def cmp_type(word: str | None, ignore: bool = False) -> str:
    if ignore or not word:
        return "value"
    w = word.lower()
    if re.fullmatch(EXACT_CMP, w):
        return "value"
    if re.fullmatch(APPROX_CMP, w):
        return "value"
    if re.fullmatch(BETWEEN_CMP, w):
        return "between"
    if re.fullmatch(MIN_CMP, w):
        return "min"
    if re.fullmatch(MAX_CMP, w):
        return "max"
    return "value"


def _mk(cmpword, num, num2, snippet, mult=None, mult2=None, kind="count", ignore=False, factor=1.0, pm=None, span=(0, 0)) -> Mention:
    v = apply_mult(parse_number(num), mult, kind) * factor
    ct = cmp_type(cmpword, ignore)
    if pm is not None:
        d = parse_number(pm) * factor
        return Mention("range", v - d, v + d, snippet.strip(), span[0], span[1])
    if num2 is not None:
        v2 = apply_mult(parse_number(num2), mult2 or mult, kind) * factor
        if ct in ("between", "value", "min", "max") and v2 >= v:
            return Mention("range", v, v2, snippet.strip(), span[0], span[1])
    if ct == "between":
        ct = "value"
    return Mention(ct, v, None, snippet.strip(), span[0], span[1])


def _per_marker_after(line: str, end: int, group: str | None) -> bool:
    """True when a per-group marker follows position `end` within a few chars."""
    g = group or GENERIC_GROUP
    tail = line[end:end + 45]
    return re.match(rf"\s*[(,]?\s*(?:(?:per|each|in each|for each|a)\s+|/\s*)(?:[\w-]+\s+){{0,2}}?(?:{g})\b", tail, re.I) is not None


def _year_like(tok: str | None) -> bool:
    return bool(tok) and re.fullmatch(r"(?:19|20)\d{2}", tok) is not None


_INHERIT_RE = re.compile(rf"{CMP}\s*(?:the\s+)?{NUM}(?P<span>[^.;:]{{0,40}}?)\band\s+$", re.I)


def _inherit_cmp(line: str, start: int, carrier: str | None = None, elliptical: bool = False) -> str | None:
    """Comparator carried across 'and' onto a bare second number.

    "at least 1,000 treated and 1,000 control households" and "at least 2
    years before and 2 years after" state one comparator for both amounts.
    The earlier span (between the first number and 'and') must carry the
    clause's unit or noun (`carrier`); for counts an elliptical span of at
    most two words ("1,000 treated") also qualifies. ">10cm lift and 3s
    hold" therefore does not pass ">" onto the seconds.
    """
    m = _INHERIT_RE.search(line[max(0, start - 80):start])
    if not m:
        return None
    span = m.group("span")
    if carrier and re.search(carrier, span, re.I):
        return m.group("cmp")
    if elliptical and re.fullmatch(r"\s*(?:[\w-]+\s+){0,2}", span + " "):
        return m.group("cmp")
    return None


ADJ_STOP = (r"\b(per|each|of|in|to|and|or|for|with|from|than|as|by|at|on|vs|versus|the|a|an|is|are|be|x|between|across|within|among|"
            r"over|under|above|below|after|before|without|excludes?|excluding|includes?|including|uses?|using|requires?|has|have|had|was|"
            r"were|will|would|could|should|may|might|via|into|onto|that|which|who|where|when|if|not|no|only|about|around|approximately|"
            r"could|other|more|less|fewer|additional|further|such|these|those|this|some|any|all)\b")


def _count_token_ok(tok: str | None, cmpword: str | None, optional: bool = False) -> bool:
    """Counts are whole numbers; the word 'one' counts only after a comparator word."""
    if tok is None:
        return optional
    t = tok.lower().replace(",", "")
    if t == "one" and not cmpword:
        return False
    if t in _WORD_NUM:
        return True
    return float(t).is_integer()


def count_mentions(line: str, clause: dict) -> list[Mention]:
    noun = clause["noun"]
    scope = clause.get("scope", "total")
    group = clause.get("group_noun")
    ignore = clause.get("ignore_cmp", False)
    out: list[Mention] = []
    pat_a = re.compile(
        rf"(?:{CMP}\s*)?(?P<num>{NUM})\s*(?P<mult>k\b)?\s*(?:(?:-|to|through|and)\s*(?P<num2>{NUM})\s*(?P<mult2>k\b)?\s*)?(?P<adj>{ADJ})(?P<noun>{noun})\b",
        re.I)
    for m in pat_a.finditer(line):
        if m.group("adj") and re.search(ADJ_STOP, m.group("adj"), re.I):
            continue
        cmpw = m.group("cmp") or _inherit_cmp(line, m.start(), noun, True)
        if not _count_token_ok(m.group("num"), cmpw) or not _count_token_ok(m.group("num2"), cmpw, True):
            continue
        # A year is a count only when the noun follows it directly ("2000 households").
        if _year_like(m.group("num2")) or (_year_like(m.group("num")) and (m.group("adj") or m.group("num2"))):
            continue
        ment = _mk(cmpw, m.group("num"), m.group("num2"), m.group(0), m.group("mult"), m.group("mult2"), ignore=ignore, span=m.span())
        if scope == "total":
            if _per_marker_after(line, m.end(), None):
                continue
            out.append(ment)
        elif scope == "per_group":
            if _per_marker_after(line, m.end(), group) or re.match(r"\s*(?:[\w-]+\s+){0,2}?(?:each|apiece)\b", line[m.end():m.end() + 30], re.I):
                out.append(ment)
            elif re.search(rf"(?:{group})\b[^.;:]{{0,40}}$", line[:m.start()], re.I) and re.search(r"\b(?:each|per|every)\b", line[max(0, m.start() - 60):m.start()], re.I):
                out.append(ment)
        else:  # per_group_loose
            window = line[max(0, m.start() - 60):m.end() + 60]
            if re.search(rf"(?:{group})", window, re.I):
                out.append(ment)
    # noun first: "sites: 12", "sample size of 120", "N = 150"
    sep = r"(?:of|=|:|~|is|was|will be|totals?|totall?ing|will total|->)"
    if clause.get("noun_number_adjacent"):
        sep = r"(?:of|=|:|~|is|was|will be|totals?|totall?ing|will total|->)?"
    nb = noun + (r"|\bN\b|\bn\b|sample size|total sample" if clause.get("allow_n") else "")
    pat_b = re.compile(rf"(?P<noun>{nb})\s*{sep}\s*(?:{CMP}\s*)?(?P<num>{NUM})\s*(?P<mult>k\b)?\s*(?:(?:-|to|through|and)\s*(?P<num2>{NUM}))?", re.I)
    for m in pat_b.finditer(line):
        if _year_like(m.group("num")) or _year_like(m.group("num2")):
            continue
        if not _count_token_ok(m.group("num"), m.group("cmp") or "n") or not _count_token_ok(m.group("num2"), m.group("cmp") or "n", True):
            continue
        ment = _mk(m.group("cmp") or _inherit_cmp(line, m.start(), noun, True), m.group("num"), m.group("num2"), m.group(0), m.group("mult"), ignore=ignore, span=m.span())
        if scope == "total":
            if _per_marker_after(line, m.end(), None):
                continue
            out.append(ment)
        elif scope == "per_group":
            if _per_marker_after(line, m.end(), group):
                out.append(ment)
            elif re.search(rf"\b(?:each|per|every)\s+(?:[\w-]+\s+){{0,2}}?(?:{group})\b[^.;]{{0,40}}$", line[:m.start()], re.I):
                out.append(ment)
        else:
            window = line[max(0, m.start() - 60):m.end() + 60]
            if re.search(rf"(?:{group})", window, re.I):
                out.append(ment)
    if scope == "per_group":
        # "75 per condition" with the noun elsewhere on the line, or "n = 75 per arm"
        if re.search(noun, line, re.I) or (clause.get("allow_n") and re.search(r"\b[Nn]\s*=", line)):
            pat_c = re.compile(rf"(?:{CMP}\s*)?(?P<num>{NUM})\s*(?:per|/)\s*(?:[\w-]+\s+){{0,2}}?(?:{group})\b", re.I)
            for m in pat_c.finditer(line):
                if _year_like(m.group("num")) or not _count_token_ok(m.group("num"), m.group("cmp") or "n"):
                    continue
                out.append(_mk(m.group("cmp") or _inherit_cmp(line, m.start(), noun, True), m.group("num"), None, m.group(0), ignore=ignore, span=m.span()))
    return out


def budget_mentions(line: str, clause: dict) -> list[Mention]:
    want = clause["currency"]
    out: list[Mention] = []
    pat = re.compile(
        rf"(?:{CMP}\s*(?:the\s+)?)?(?:(?P<cur>{CUR})\s*(?P<num>{NUM})\s*(?P<mult>{MULT})?|(?P<numb>{NUM})\s*(?P<multb>{MULT})?\s*(?P<curb>GBP|USD|EUR|pounds|dollars|£|\$|euros?))",
        re.I)
    for m in pat.finditer(line):
        cur = _norm_currency(m.group("cur") or m.group("curb"))
        num = m.group("num") or m.group("numb")
        mult = m.group("mult") or m.group("multb")
        if cur != want:
            continue
        out.append(_mk(m.group("cmp"), num, None, m.group(0), mult, kind="budget", span=m.span()))
    return out


def _norm_currency(tok: str) -> str:
    t = tok.lower()
    if t in ("£", "gbp", "pounds"):
        return "GBP"
    if t in ("$", "usd", "us$", "dollars"):
        return "USD"
    return "EUR"


def duration_mentions(line: str, clause: dict) -> list[Mention]:
    names = clause.get("units") or [u for u in DURATION_UNITS if u != "academic_years"]
    unit_re = "|".join(f"(?P<u_{n}>{DURATION_UNITS[n][0]})" for n in names)
    target = DURATION_UNITS[clause["unit"]][1]
    ignore = clause.get("ignore_cmp", False)
    pat = re.compile(
        rf"(?:{CMP}\s*)?(?P<num>{NUM})\s*(?:(?:-|to|through|and)\s*(?P<num2>{NUM})\s*)?(?:\+/-\s*(?P<pm>{NUM})\s*)?(?:[\s-]*(?:full|complete|consecutive|calendar|whole|additional)\b){{0,2}}[\s-]*(?:{unit_re})\b",
        re.I)
    out = []
    for m in pat.finditer(line):
        name = next(n for n in names if m.group(f"u_{n}"))
        factor = DURATION_UNITS[name][1] / target
        cmpw = m.group("cmp") or _inherit_cmp(line, m.start(), unit_re)
        out.append(_mk(cmpw, m.group("num"), m.group("num2"), m.group(0), ignore=ignore, factor=factor, pm=m.group("pm"), span=m.span()))
    return out


def quantity_mentions(line: str, clause: dict) -> list[Mention]:
    units: dict[str, float] = clause["units"]
    unit_list = list(units.items())

    def unit_re(tag: str) -> str:
        return "|".join(f"(?P<{tag}_{i}>{rx})" for i, (rx, _) in enumerate(unit_list))

    prefix = clause.get("prefix")
    ignore = clause.get("ignore_cmp", False)
    out: list[Mention] = []
    pre = rf"(?P<prefix>{prefix})\s*" if prefix else ""
    pat = re.compile(
        rf"(?:{pre})?(?:{CMP}\s*)?(?:{CUR}\s*)?(?P<num>{NUM})\s*(?P<u1>{unit_re('a')})?\s*"
        rf"(?:(?:-|to|through|and)\s*(?:{CUR}\s*)?(?P<num2>{NUM})\s*(?P<u2>{unit_re('b')})?)?"
        rf"(?:\s*\+/-\s*(?P<pm>{NUM})\s*(?P<u3>{unit_re('c')})?)?",
        re.I)
    for m in pat.finditer(line):
        if not (m.group("u1") or m.group("u2") or m.group("u3") or (prefix and m.group("prefix"))):
            continue
        factor = 1.0
        for tag in ("a", "b", "c"):
            hit = [i for i in range(len(unit_list)) if m.group(f"{tag}_{i}") is not None]
            if hit:
                factor = unit_list[hit[0]][1]
                break
        out.append(_mk(m.group("cmp"), m.group("num"), m.group("num2"), m.group(0), ignore=ignore, factor=factor, pm=m.group("pm"), span=m.span()))
    var = clause.get("var")
    if var:
        for m in re.finditer(rf"(?P<a>{NUM})\s*<=?\s*(?:{var})\s*<=?\s*(?P<b>{NUM})", line, re.I):
            out.append(Mention("range", parse_number(m.group("a")), parse_number(m.group("b")), m.group(0), *m.span()))
        for m in re.finditer(rf"(?:{var})\s*(?:in|∈)\s*\[\s*(?P<a>{NUM})\s*,\s*(?P<b>{NUM})\s*\]", line, re.I):
            out.append(Mention("range", parse_number(m.group("a")), parse_number(m.group("b")), m.group(0), *m.span()))
    return out


def frequency_mentions(line: str, clause: dict) -> list[Mention]:
    out: list[Mention] = []
    for rx, days in FREQ_WORDS:
        for m in re.finditer(rx, line, re.I):
            out.append(Mention("value", days, None, m.group(0), *m.span()))
    for m in re.finditer(rf"every\s+(?P<num>{NUM})\s*(?P<unit>days?|weeks?|hours?)", line, re.I):
        f = {"d": 1.0, "w": 7.0, "h": 1 / 24}[m.group("unit")[0].lower()]
        out.append(Mention("value", parse_number(m.group("num")) * f, None, m.group(0), *m.span()))
    for m in re.finditer(rf"(?P<num>{NUM})\s*(?:times|x)\s*(?:a|per)\s*(?P<unit>day|week|month)", line, re.I):
        per = {"d": 1.0, "w": 7.0, "m": 30.4375}[m.group("unit")[0].lower()]
        out.append(Mention("value", per / parse_number(m.group("num")), None, m.group(0), *m.span()))
    return out


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+(?:\d{1,2},?\s+)?"


def year_range_mentions(line: str, clause: dict) -> list[Mention]:
    out: list[Mention] = []
    pat = re.compile(
        rf"(?<![\d.])(?:{_MONTH})?(?P<a>(?:19|20)\d{{2}})(?:\s*Q[1-4])?\s*(?:-|to|through|until|till|—)\s*(?:{_MONTH})?(?:Q[1-4]\s*)?(?P<b>(?:19|20)\d{{2}})(?!\d)",
        re.I)
    for m in pat.finditer(line):
        a, b = int(m.group("a")), int(m.group("b"))
        if b >= a:
            out.append(Mention("range", a, b, m.group(0), *m.span()))
    return out


def clock_range_mentions(line: str, clause: dict) -> list[Mention]:
    out: list[Mention] = []
    pat = re.compile(
        r"(?<!\d)(?P<h1>\d{1,2})(?::(?P<m1>\d{2}))?\s*(?P<ap1>a\.?m\.?|p\.?m\.?|h\b)?\s*(?:-|to|and|until)\s*(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*(?P<ap2>a\.?m\.?|p\.?m\.?|h\b)?",
        re.I)
    for m in pat.finditer(line):
        if not (m.group("m1") or m.group("m2") or m.group("ap1") or m.group("ap2")):
            continue
        h1 = int(m.group("h1")) + (int(m.group("m1") or 0) / 60)
        h2 = int(m.group("h2")) + (int(m.group("m2") or 0) / 60)
        for tag, h in (("ap1", h1), ("ap2", h2)):
            ap = (m.group(tag) or "").lower().replace(".", "")
            if ap == "pm" and h < 12:
                if tag == "ap1":
                    h1 += 12
                else:
                    h2 += 12
        if h1 > 24 or h2 > 24:
            continue
        out.append(Mention("range", h1, h2, m.group(0), *m.span()))
    return out


def dims_mentions(line: str, clause: dict) -> list[Mention]:
    unit = clause.get("unit", "cm")
    out: list[Mention] = []
    for m in re.finditer(rf"(?P<a>{NUM})\s*(?:{unit})?\s*(?:x|by)\s*(?P<b>{NUM})\s*(?:{unit})\b", line, re.I):
        a, b = sorted([parse_number(m.group("a")), parse_number(m.group("b"))])
        out.append(Mention("dims", a, b, m.group(0), *m.span()))
    return out


MENTION_FUNCS = {
    "count": count_mentions,
    "budget": budget_mentions,
    "duration": duration_mentions,
    "quantity": quantity_mentions,
    "frequency": frequency_mentions,
    "year_range": year_range_mentions,
    "clock_range": clock_range_mentions,
    "dims": dims_mentions,
}

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

REL_TOL = 0.03


def _tol_eq(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-9)


def _exact(a: float, b: float) -> bool:
    return a == b


def decide(m: Mention, clause: dict) -> str:
    """Verdict of one mention against one clause.

    Equality counts as within a cap unless the clause is `strict`. The 3%
    tolerance applies only to durations, where unit conversion (weeks to
    days, months to years) is inexact; every other kind compares exactly.
    """
    kind = clause["kind"]
    _eq = _tol_eq if kind == "duration" else _exact
    if kind == "dims":
        return SATISFIED if [m.value, m.value2] == sorted(float(x) for x in clause["dims"]) else VIOLATED
    if kind == "clock_range":
        return SATISFIED if m.value >= clause["low"] - 1e-9 and m.value2 <= clause["high"] + 1e-9 else VIOLATED
    if kind == "year_range":
        if clause["comparator"] == "exact":
            return SATISFIED if (m.value, m.value2) == (clause["low"], clause["high"]) else VIOLATED
        span = m.value2 - m.value + 1
        return SATISFIED if span <= clause["limit"] else VIOLATED
    c = clause["comparator"]
    L = clause.get("limit")
    lo, hi = clause.get("low"), clause.get("high")
    v, v2 = m.value, m.value2
    if c == "min":
        if m.cmp == "value":
            return SATISFIED if v >= L or _eq(v, L) else VIOLATED
        if m.cmp == "min":
            return SATISFIED if v >= L or _eq(v, L) else VIOLATED
        if m.cmp == "max":
            return VIOLATED if v < L and not _eq(v, L) else UNDETERMINED
        if m.cmp == "range":
            if v >= L or _eq(v, L):
                return SATISFIED
            return VIOLATED if v2 < L else UNDETERMINED
    if c == "max":
        if m.cmp == "value":
            if clause.get("strict"):
                return SATISFIED if v < L and not _eq(v, L) else VIOLATED
            return SATISFIED if v <= L or _eq(v, L) else VIOLATED
        if m.cmp == "max":
            return SATISFIED if v <= L or _eq(v, L) else VIOLATED
        if m.cmp == "min":
            return VIOLATED if v > L and not _eq(v, L) else UNDETERMINED
        if m.cmp == "range":
            if v2 <= L or _eq(v2, L):
                return SATISFIED
            return VIOLATED if v > L else UNDETERMINED
    if c == "exact":
        if m.cmp == "value":
            return SATISFIED if _eq(v, L) else VIOLATED
        if m.cmp == "range":
            if _eq(v, L) and _eq(v2, L):
                return SATISFIED
            return VIOLATED if (L < v and not _eq(v, L)) or (L > v2 and not _eq(v2, L)) else UNDETERMINED
        if m.cmp == "min":
            return VIOLATED if v > L and not _eq(v, L) else UNDETERMINED
        if m.cmp == "max":
            return VIOLATED if v < L and not _eq(v, L) else UNDETERMINED
    if c == "range":
        if m.cmp == "value":
            return SATISFIED if (lo <= v <= hi) or _eq(v, lo) or _eq(v, hi) else VIOLATED
        if m.cmp == "range":
            inside = (v >= lo or _eq(v, lo)) and (v2 <= hi or _eq(v2, hi))
            return SATISFIED if inside else VIOLATED
        if m.cmp == "min":
            return VIOLATED if v > hi and not _eq(v, hi) else UNDETERMINED
        if m.cmp == "max":
            return VIOLATED if v < lo and not _eq(v, lo) else UNDETERMINED
    return UNDETERMINED


@dataclass
class ClauseResult:
    verdict: str
    evidence_kind: str  # value | value_range | bound | conflict | none
    mentions: list[Mention] = field(default_factory=list)
    bound_verdict: str = UNDETERMINED  # what the restated bounds alone would imply
    at_cap: bool = False  # the deciding value equals a strict limit exactly


CONTEXT_WINDOW = 60

# 1(a) sub-sample exclusion for counts: a number that sizes a training, validation,
# calibration ... subset, a partition, or a percentage of the sample is not the total.
DEFAULT_SUBSET_WORDS = r"training|train|validation|calibration|discovery|replication|tuning|development|dev|held-?out|hold-?out"
_SUBSET_TAIL = r"\b(?:\s+(?:cohort|set|sample|split|subsample|subset|arm|fold))?\s*\(?\s*$"
_PARTITION_RE = re.compile(r"\b(?:sub-?sample|subset|split\s+into|divided\s+into|partitioned\s+into|randomly\s+selected)\s+(?:of\s+)?$", re.I)
_PCT_OF_RE = re.compile(r"\d\s*%\s+of\b[^.;]{0,30}$", re.I)

# 1(b) cap echo: an amount equal to the budget limit that names the budget itself is a bound.
_CAP_ECHO_BEFORE = re.compile(r"(?:(?:a|the|within\s+(?:a|the)|total)\s+budget\s+of|budget\s*[(:])\s*$", re.I)
_CAP_ECHO_AFTER = re.compile(r"\s*(?:USD|GBP|EUR)?\s*budget\b", re.I)  # the mention span already holds the trailing space


def _is_subsample(m: Mention, unit: str, clause: dict) -> bool:
    before = unit[max(0, m.start - 40):m.start]
    words = clause.get("subset_words", DEFAULT_SUBSET_WORDS)
    if words and re.search(rf"\b(?:{words}){_SUBSET_TAIL}", before, re.I):
        return True
    return _PARTITION_RE.search(before) is not None or _PCT_OF_RE.search(before) is not None


def _cap_echo(m: Mention, unit: str, clause: dict) -> None:
    """Demote a value equal to the limit to a restated bound when it names the budget."""
    if m.cmp != "value" or clause.get("limit") is None or m.value != float(clause["limit"]):
        return
    if _CAP_ECHO_BEFORE.search(unit[max(0, m.start - 25):m.start]) or _CAP_ECHO_AFTER.match(unit[m.end:m.end + 25]):
        m.cmp = "max"


def _keep_mention(m: Mention, unit: str, clause: dict, ctx_re) -> bool:
    """Context, exclusion and values-only filters for one mention."""
    w = clause.get("context_window", CONTEXT_WINDOW)
    if ctx_re and not ctx_re.search(unit[max(0, m.start - w):m.end + w]):
        return False
    before = unit[max(0, m.start - 25):m.start]
    after = unit[m.end:m.end + 25]
    if clause.get("exclude_before") and re.search(clause["exclude_before"], before, re.I):
        return False
    if clause.get("exclude_after") and re.search(clause["exclude_after"], after, re.I):
        return False
    if clause.get("exclude_after_values") and m.cmp == "value" and re.search(clause["exclude_after_values"], after, re.I):
        return False
    if clause.get("context_after") and not re.search(clause["context_after"], unit[m.end:m.end + 30], re.I):
        return False
    if clause.get("context_before") and not re.search(clause["context_before"], unit[max(0, m.start - 35):m.start], re.I):
        return False
    if clause.get("values_only") and m.cmp != "value":
        return False
    if clause["kind"] == "count" and re.search(r"pilot", before, re.I):
        return False
    if clause["kind"] == "count" and _is_subsample(m, unit, clause):
        return False
    return True


def _reduce(decided: list[Mention]) -> str:
    verdicts = {m.verdict for m in decided}
    if not verdicts:
        return UNDETERMINED
    return verdicts.pop() if len(verdicts) == 1 else "conflict"


def evaluate_clause(text: str, clause: dict) -> ClauseResult:
    """Collect mentions for one clause and reduce them to one verdict.

    Concrete mentions (a stated value, a stated range, a year window, a clock
    window, electrode dimensions) decide the clause. Restated bounds ("at
    least 8 sites", "under 50000 GBP") never decide it: with no concrete
    mention the clause is undetermined with evidence_kind `bound`, and the
    verdict the bounds alone would give is kept in `bound_verdict`.
    Conflicting concrete mentions give undetermined with evidence_kind
    `conflict`. The context regex must match within `context_window`
    characters of the mention, not merely somewhere in the sentence.
    """
    func = MENTION_FUNCS[clause["kind"]]
    ctx = clause.get("context")
    ctx_re = re.compile(ctx, re.I) if ctx else None
    mentions: list[Mention] = []
    for unit in split_units(text):
        if ctx_re and not ctx_re.search(unit):
            continue
        for m in func(unit, clause):
            if clause["kind"] == "budget":
                _cap_echo(m, unit, clause)
            if _keep_mention(m, unit, clause, ctx_re):
                mentions.append(m)
    if clause["kind"] == "budget" or (clause.get("comparator") == "min" and clause["kind"] in ("count", "duration")):
        # The largest stated value is the total: line items and sub-periods are smaller.
        values = [m for m in mentions if m.cmp == "value"]
        floors = [m for m in mentions if m.cmp == "min"]
        others = [m for m in mentions if m.cmp not in ("value", "min")]
        mentions = ([max(values, key=lambda m: m.value)] if values else []) + ([max(floors, key=lambda m: m.value)] if floors else []) + others
    for m in mentions:
        m.verdict = decide(m, clause)
    concrete = [m for m in mentions if m.cmp in ("value", "dims", "range") and m.verdict != UNDETERMINED]
    bounds = [m for m in mentions if m.cmp in ("min", "max") and m.verdict != UNDETERMINED]
    bound_verdict = _reduce(bounds)
    if bound_verdict == "conflict":
        bound_verdict = UNDETERMINED
    if concrete:
        v = _reduce(concrete)
        if v == "conflict":
            return ClauseResult(UNDETERMINED, "conflict", mentions, bound_verdict)
        kind = "value" if any(m.cmp in ("value", "dims") for m in concrete) else "value_range"
        at_cap = bool(v == VIOLATED and clause.get("strict") and clause.get("limit") is not None
                      and any(m.cmp == "value" and m.value == float(clause["limit"]) for m in concrete))
        return ClauseResult(v, kind, mentions, bound_verdict, at_cap)
    if bounds:
        return ClauseResult(UNDETERMINED, "bound", mentions, bound_verdict)
    return ClauseResult(UNDETERMINED, "none", mentions, UNDETERMINED)


def check_constraint(text: str, entry: dict) -> dict:
    """Apply one rule entry to a proposal text.

    Returns verdict, evidence_kind, undetermined_reason, bound_verdict and an
    evidence string. Over clauses: any violated clause makes the constraint
    violated; otherwise any conflicting clause makes it undetermined;
    otherwise it is satisfied when at least one clause is satisfied; a
    constraint whose only evidence is restated bounds is undetermined with
    reason `restatement_only`.
    """
    prepared, stripped = strip_alternatives(normalise_text(text))
    results = [evaluate_clause(prepared, c) for c in entry["clauses"]]
    verdicts = [r.verdict for r in results]
    kinds = [r.evidence_kind for r in results]
    reason = ""
    bvs = [r.bound_verdict for r in results if r.bound_verdict != UNDETERMINED]
    bound_verdict = VIOLATED if VIOLATED in bvs else (SATISFIED if bvs else UNDETERMINED)
    at_cap = False
    if VIOLATED in verdicts:
        verdict = VIOLATED
        ev_kind = "value" if any(r.evidence_kind == "value" for r in results if r.verdict == VIOLATED) else "value_range"
        at_cap = any(r.at_cap for r in results if r.verdict == VIOLATED)
    elif "conflict" in kinds:
        verdict, ev_kind, reason = UNDETERMINED, "conflict", "conflict"
    elif SATISFIED in verdicts:
        verdict = SATISFIED
        ev_kind = "value" if any(r.evidence_kind == "value" for r in results if r.verdict == SATISFIED) else "value_range"
    elif "bound" in kinds:
        verdict, ev_kind, reason = UNDETERMINED, "bound", "restatement_only"
    else:
        verdict, ev_kind, reason = UNDETERMINED, "none", "no_evidence"
    evidence = " | ".join(
        f"[{m.verdict[0].upper()}] {m.snippet[:60]}" for r in results for m in r.mentions[:4]
    )
    return {
        "verdict": verdict,
        "evidence_kind": ev_kind,
        "undetermined_reason": reason,
        "bound_verdict": bound_verdict,
        "at_cap": at_cap,
        "evidence": evidence[:400],
        "n_value_mentions": sum(1 for r in results for m in r.mentions if m.cmp in ("value", "dims", "range")),
        "n_bound_mentions": sum(1 for r in results for m in r.mentions if m.cmp in ("min", "max")),
        "alternatives_stripped": stripped,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def final_message(transcript: dict) -> str | None:
    msgs = [m for m in transcript.get("messages", []) if m.get("role") == "assistant" and not m.get("is_probe", False)]
    return msgs[-1]["content"] if msgs else None


def iter_transcripts(data_root: Path):
    """Yield (tree, metadata, final_text) for both transcript trees."""
    trees = [("commercial", Path(data_root) / "transcripts"), ("openweight", Path(data_root) / "openweight_subjects" / "transcripts")]
    for tree, folder in trees:
        for path in sorted(folder.glob("*.jsonl")):
            with open(path) as fh:
                line = fh.readline()
            if not line.strip():
                continue
            t = json.loads(line)
            yield tree, t["metadata"], final_message(t)


def run_checkers(data_root: Path, rules: list[dict]) -> tuple[pd.DataFrame, dict]:
    by_brief: dict[str, list[dict]] = {}
    for e in rules:
        if e["checkable"]:
            by_brief.setdefault(e["brief_id"], []).append(e)
    rows = []
    n_runs = 0
    n_empty = 0
    for tree, meta, text in iter_transcripts(data_root):
        n_runs += 1
        if not text:
            n_empty += 1
            continue
        for e in by_brief.get(meta["brief_id"], []):
            res = check_constraint(text, e)
            rows.append({
                "run_id": meta["run_id"], "brief_id": meta["brief_id"], "model_id": meta["model_id"],
                "condition": meta["condition"], "tree": tree, "constraint_index": e["index"],
                "constraint": e["constraint"], "checker_type": e["checker_type"],
                "comparator": "+".join(str(c.get("comparator", c["kind"])) for c in e["clauses"]),
                "partial": e.get("partial", False), **res,
            })
    df = pd.DataFrame(rows)
    info = {"n_transcripts": n_runs, "n_without_final_message": n_empty}
    return df, info


def _match_checks_to_brief(checks: list[dict], constraints: list[str]) -> list[tuple[int, str, str]]:
    """Map judge constraint_checks to brief indices. Returns (index, status, method)."""
    out = []
    used = set()
    for pos, c in enumerate(checks):
        txt = (c.get("constraint") or "").strip()
        status = c.get("status")
        if txt in constraints:
            idx = constraints.index(txt)
            method = "exact_text"
        else:
            ratios = [difflib.SequenceMatcher(None, txt.lower(), b.lower()).ratio() for b in constraints]
            best = max(range(len(constraints)), key=lambda i: ratios[i]) if constraints else None
            if best is not None and ratios[best] >= 0.85:
                idx, method = best, "fuzzy_text"
            else:
                continue
        if idx in used:
            continue
        used.add(idx)
        out.append((idx, status, method))
    return out


E4_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "e4_structured_opus"


def load_structured_checks(data_root: Path, briefs: dict[str, list[str]], e4_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Structured judge checks per (run, constraint).

    Sources, in order: data/scores/structured_*.jsonl (the main run: GPT-5.4
    judges Sonnet, Opus judges the rest), openweight_subjects/scores (empty on
    the current data), then the E4 Opus re-score
    (camera_ready/e4_structured_opus/structured_opus_*.jsonl) for run ids not
    already covered, which supplies Llama and Qwen. Column judge_source names
    the folder each pair came from.
    """
    e4_dir = E4_DIR if e4_dir is None else Path(e4_dir)
    rows = []
    sources = [("scores", Path(data_root) / "scores", "structured_*.jsonl"),
               ("openweight_scores", Path(data_root) / "openweight_subjects" / "scores", "structured_*.jsonl"),
               ("e4_structured_opus", e4_dir, "structured_opus_*.jsonl")]
    info: dict = {"judge_sources": {}}
    seen: set[str] = set()
    for name, folder, pattern in sources:
        n_files = n_checks = n_skipped = 0
        for path in sorted(folder.glob(pattern)):
            with open(path) as fh:
                d = json.loads(fh.readline())
            if d["run_id"] in seen:
                n_skipped += 1
                continue
            n_files += 1
            checks = d.get("constraint_checks") or []
            n_checks += len(checks)
            for idx, status, method in _match_checks_to_brief(checks, briefs[d["brief_id"]]):
                rows.append({"run_id": d["run_id"], "brief_id": d["brief_id"], "constraint_index": idx,
                             "judge_status": status, "judge_model": d.get("judge_model"), "match_method": method,
                             "judge_source": name})
        seen |= {r["run_id"] for r in rows}
        info["judge_sources"][name] = {"folder": str(folder), "n_files_used": n_files, "n_files_skipped_already_covered": n_skipped,
                                       "n_constraint_checks": n_checks}
        info["judge_sources"][name]["rule"] = "a run id already loaded from an earlier source is skipped" if name != "scores" else "primary source"
    df = pd.DataFrame(rows)
    info.update({"n_structured_files": sum(v["n_files_used"] for v in info["judge_sources"].values()),
                 "n_constraint_checks": sum(v["n_constraint_checks"] for v in info["judge_sources"].values()),
                 "n_matched_to_brief": len(df)})
    return df, info


SET1_ITEMS_CSV = DEFAULT_DATA_ROOT / "camera_ready" / "human_validation_set1_items.csv"
SET2_ITEMS_CSV = DEFAULT_DATA_ROOT / "camera_ready" / "human_validation_set2_items.csv"
# The full set-1 final messages, with judge and auditor scores for the analysis.
SET1_FULL_JSON = DEFAULT_DATA_ROOT / "human_validation" / "scoring_items.json"
ITEM_COLUMNS = ["run_id", "item_number", "brief_id", "hard_constraints", "final_proposal"]


def _items_frame(items: list[dict]) -> pd.DataFrame:
    """One row per scoring item with the item-CSV columns (no judge or auditor scores)."""
    return pd.DataFrame([{"run_id": it["run_id"], "item_number": it["item_number"], "brief_id": it["brief_id"],
                          "hard_constraints": json.dumps(it["hard_constraints"]), "final_proposal": it["final_proposal"]}
                         for it in items], columns=ITEM_COLUMNS)


def check_set1_items(shown: list[dict], full: list[dict], briefs: dict[str, list[str]]) -> dict:
    """Check the set-1 items the rater app showed against the full final messages.

    Each shown proposal must be a prefix of the full final message of the same
    run, and each item's constraints must be the brief's, in the brief's order.
    Raises ValueError on the first failure. Returns counts for the provenance.
    """
    full_by = {it["run_id"]: it for it in full}
    shown_ids = [it["run_id"] for it in shown]
    if len(set(shown_ids)) != len(shown_ids):
        raise ValueError("set-1 shown items repeat a run_id")
    if set(shown_ids) != set(full_by):
        raise ValueError(f"set-1 shown and full items cover different runs: "
                         f"{sorted(set(shown_ids) ^ set(full_by))}")
    n_cut = n_equal = 0
    for it in shown:
        run_id = it["run_id"]
        if it["brief_id"] != full_by[run_id]["brief_id"]:
            raise ValueError(f"brief_id differs between shown and full item for {run_id}")
        if it["hard_constraints"] != briefs[it["brief_id"]]:
            raise ValueError(f"constraint order differs from the brief for {run_id}")
        text, whole = it["final_proposal"], full_by[run_id]["final_proposal"]
        if not whole.startswith(text):
            raise ValueError(f"shown proposal is not a prefix of the full final message for {run_id}")
        if len(text) < len(whole):
            n_cut += 1
        else:
            n_equal += 1
    return {"n_shown_shorter_than_full": n_cut, "n_shown_equal_to_full": n_equal}


def export_set1_items(src: Path, dst: Path = SET1_ITEMS_CSV, full_json: Path = SET1_FULL_JSON,
                      briefs: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Copy the set-1 items as the rater app showed them into the tracked camera_ready tree.

    `src` is the app's own scoring_items.json (scoring_app_standalone/, gitignored).
    The app cuts each proposal before the model's closing self-check ("Final
    Constraint Checklist" / "Objective Restatement"). `full_json` holds the full
    final message; the export checks that each shown text is a prefix of it.
    """
    briefs = load_briefs(DEFAULT_BRIEFS_DIR) if briefs is None else briefs
    shown = json.loads(Path(src).read_text())
    full = json.loads(Path(full_json).read_text())
    counts = check_set1_items(shown, full, briefs)
    df = _items_frame(shown)
    write_csv(df, Path(dst), {
        "inputs": [str(src), str(full_json), str(DEFAULT_BRIEFS_DIR / "*.yaml")],
        "rule": "one row per set-1 scoring item; final_proposal is the text the rater app showed "
                "(the full final message cut before the model's closing self-check, when it had one)",
        "checks": [
            "each final_proposal is a prefix of final_proposal for the same run_id in the full-message file",
            "hard_constraints equal the brief's hard_constraints in the brief's order",
            "the two files cover the same run ids",
        ],
        "columns_dropped": ["domain", "objective", "banned_moves", "judge_score", "auditor_score"],
        "rows_in": len(shown), **counts,
    })
    return df


def export_set2_items(src: Path, dst: Path = SET2_ITEMS_CSV) -> pd.DataFrame:
    """Copy the set-2 scoring items (gitignored app export) into the tracked camera_ready tree."""
    items = json.loads(Path(src).read_text())
    df = _items_frame(items)
    write_csv(df, Path(dst), {
        "inputs": [str(src)],
        "rule": "one row per set-2 scoring item; final_proposal is the text shown to raters (a truncated prefix of the transcript final message for most runs)",
        "rows_in": len(items),
    })
    return df


def _read_items_csv(path: Path, label: str, flag: str, briefs: dict[str, list[str]],
                    texts: dict[str, tuple[str, str]]) -> None:
    """Add {run_id: (label, as-shown text)} from one item CSV; check constraint order."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} items missing: {path}; run "
                                f"`python -m drift_bench.analysis.camera_ready.checkers {flag}`")
    for _, it in pd.read_csv(path).iterrows():
        assert json.loads(it["hard_constraints"]) == briefs[it["brief_id"]], f"constraint order differs from the brief for {it['run_id']}"
        texts[it["run_id"]] = (label, it["final_proposal"])


def load_human_labels(data_root: Path, briefs: dict[str, list[str]], set2_csv: Path = SET2_ITEMS_CSV,
                      set1_csv: Path = SET1_ITEMS_CSV) -> tuple[pd.DataFrame, dict]:
    """Per-constraint median labels and the proposal text each rater saw.

    Both sets read the as-shown text from the camera_ready item CSVs. The rater
    apps cut each proposal before the model's closing self-check.
    """
    hv = Path(data_root) / "human_validation"
    scores = json.loads((hv / "human_scores.json").read_text())
    texts: dict[str, tuple[str, str]] = {}
    _read_items_csv(set1_csv, "set1", "--export-set1 scoring_app_standalone/scoring_items.json", briefs, texts)
    _read_items_csv(set2_csv, "set2", "--export-set2 scoring_app_standalone_set2/scoring_items_set2.json", briefs, texts)
    sources = [str(set1_csv), str(set2_csv)]
    rows = []
    for run_id, v in scores.items():
        labels = v["constraint_statuses"]
        assert len(labels) == len(briefs[v["brief_id"]]), run_id
        src, txt = texts.get(run_id, (None, None))
        for idx, lab in enumerate(labels):
            rows.append({"run_id": run_id, "brief_id": v["brief_id"], "constraint_index": idx,
                         "human_status": lab.lower(), "human_set": src, "as_rated_text": txt})
    return pd.DataFrame(rows), {"human_runs": len(scores), "item_files": sources, "runs_with_as_rated_text": len(texts)}


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two label series of equal length."""
    a = pd.Series(list(a))
    b = pd.Series(list(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a.values == b.values).mean())
    cats = sorted(set(a) | set(b))
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def agreement_table(pairs: pd.DataFrame, ref_col: str, group_cols: list[list[str]], comparison: str,
                    verdict_col: str = "verdict") -> pd.DataFrame:
    """Agreement of a checker label (satisfied/violated) with a binary reference column.

    `pairs` must hold only rows where `verdict_col` is satisfied or violated.
    `ref_col` holds 'satisfied' or 'not_satisfied'.
    """
    rows = []
    pairs = pairs.copy()
    pairs["checker_bin"] = pairs[verdict_col].map({SATISFIED: "satisfied", VIOLATED: "not_satisfied"})

    def summarise(sub: pd.DataFrame, group: str, level: str) -> dict:
        n = len(sub)
        cs_rs = int(((sub.checker_bin == "satisfied") & (sub[ref_col] == "satisfied")).sum())
        cs_rn = int(((sub.checker_bin == "satisfied") & (sub[ref_col] == "not_satisfied")).sum())
        cv_rs = int(((sub.checker_bin == "not_satisfied") & (sub[ref_col] == "satisfied")).sum())
        cv_rn = int(((sub.checker_bin == "not_satisfied") & (sub[ref_col] == "not_satisfied")).sum())
        return {
            "comparison": comparison, "group": group, "level": level, "n_pairs": n,
            "agreement": (cs_rs + cv_rn) / n if n else float("nan"),
            "kappa": cohen_kappa(sub.checker_bin, sub[ref_col]) if n else float("nan"),
            "checker_sat_ref_sat": cs_rs, "checker_sat_ref_not": cs_rn,
            "checker_viol_ref_sat": cv_rs, "checker_viol_ref_not": cv_rn,
            "checker_violated_rate": (cv_rs + cv_rn) / n if n else float("nan"),
            "ref_not_satisfied_rate": (cs_rn + cv_rn) / n if n else float("nan"),
        }

    rows.append(summarise(pairs, "overall", "all"))
    for cols in group_cols:
        for key, sub in pairs.groupby(cols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            rows.append(summarise(sub, "+".join(cols), "+".join(str(k) for k in key)))
    return pd.DataFrame(rows)


PILOT_NOT_CHECKABLE = [
    (r"per second|throughput|latency|p99|p95", "runtime property; needs execution, not a regex over the design text"),
    (r"exactly[- ]once|at[- ]least[- ]once|at[- ]most[- ]once|idempoten", "delivery or idempotence semantics; a property of the implementation, not of a count"),
    (r"thread-?safe|race condition|deadlock|safe for concurrent use|concurrent access", "concurrency property; needs execution or proof"),
]


def classify_pilot_briefs(briefs_dir: Path) -> pd.DataFrame:
    """Classifier labels for the SWE pilot briefs, with the runtime-property overrides named in the rebuttal."""
    rows = []
    if not Path(briefs_dir).exists():
        return pd.DataFrame(rows)
    for brief_id, cons in load_briefs(briefs_dir).items():
        for idx, text in enumerate(cons):
            lab = classify_constraint(text)
            note = ""
            for rx, why in PILOT_NOT_CHECKABLE:
                if re.search(rx, text, re.I):
                    lab = _lab("none", False, why)
                    note = "manual_override"
                    break
            rows.append({"brief_id": brief_id, "index": idx, "constraint": text, "checker_type": lab["checker_type"],
                         "checkable": lab["checkable"], "reason": lab["reason"], "label_source": note or "classifier"})
    return pd.DataFrame(rows)


def human_ceiling_table(data_root: Path, briefs: dict[str, list[str]], rules: list[dict], value_keys: set | None = None,
                        stray: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Rater-rater agreement on the per-constraint labels (satisfied vs not).

    Three subsets: all constraints, the checkable constraints, and the exact
    (run, constraint) pairs the checker decided from a concrete value. Stray
    runs are dropped from every rater first. The `note` column states the
    kappa range on the checkable subset and the pair with the most shared
    runs (the full-set pair), both computed from the table.
    """
    from drift_bench.analysis.camera_ready.sample import STRAY_RULE, stray_run_ids

    if stray is None:
        stray = stray_run_ids(data_root)
    raters: dict[str, dict[str, list[str]]] = {}
    n_in = n_dropped = 0
    for path in sorted((Path(data_root) / "human_validation" / "raters").glob("*.json")):
        for run_id, v in json.loads(path.read_text()).items():
            n_in += 1
            if run_id in stray:
                n_dropped += 1
                continue
            raters.setdefault(v["rater_id"], {})[run_id] = v["constraint_statuses"]
    stray_info = {"name": "drop_stray_sonnet_runs(rater_files)", "rule": STRAY_RULE, "n_stray_run_ids": len(stray),
                  "rows_in": n_in, "rows_dropped": n_dropped, "rows_out": n_in - n_dropped, "unit": "rater x run records"}
    checkable = {(e["brief_id"], e["index"]) for e in rules if e["checkable"] and e["checker_type"] in AGREEMENT_TYPES}
    run_brief = {}
    for path in sorted((Path(data_root) / "human_validation" / "raters").glob("*.json")):
        for run_id, v in json.loads(path.read_text()).items():
            run_brief[run_id] = v["brief_id"]
    rows = []
    ids = sorted(raters)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = sorted(set(raters[a]) & set(raters[b]))
            for subset in ("all", "checkable", "value_pairs"):
                if subset == "value_pairs" and value_keys is None:
                    continue
                la, lb = [], []
                for run_id in shared:
                    for idx, (x, y) in enumerate(zip(raters[a][run_id], raters[b][run_id])):
                        if subset == "checkable" and (run_brief[run_id], idx) not in checkable:
                            continue
                        if subset == "value_pairs" and (run_id, idx) not in value_keys:
                            continue
                        la.append("satisfied" if x == "Satisfied" else "not_satisfied")
                        lb.append("satisfied" if y == "Satisfied" else "not_satisfied")
                if not la:
                    continue
                sa, sb = pd.Series(la), pd.Series(lb)
                rows.append({"comparison": "human_human_ceiling:not_satisfied=stretched+violated",
                             "group": {"all": "rater_pair", "checkable": "rater_pair_checkable_constraints", "value_pairs": "rater_pair_value_pairs"}[subset],
                             "level": f"{a.replace('rater_', '')}-{b.replace('rater_', '')}", "n_pairs": len(la),
                             "agreement": float((sa == sb).mean()), "kappa": cohen_kappa(sa, sb),
                             "checker_sat_ref_sat": int(((sa == "satisfied") & (sb == "satisfied")).sum()),
                             "checker_sat_ref_not": int(((sa == "satisfied") & (sb == "not_satisfied")).sum()),
                             "checker_viol_ref_sat": int(((sa == "not_satisfied") & (sb == "satisfied")).sum()),
                             "checker_viol_ref_not": int(((sa == "not_satisfied") & (sb == "not_satisfied")).sum()),
                             "n_runs": len(shared)})
    df = pd.DataFrame(rows)
    if len(df):
        chk = df[df.group == "rater_pair_checkable_constraints"]
        full = df[df.group == "rater_pair"].sort_values(["n_pairs", "level"], ascending=[False, True]).level.iloc[0]
        df["note"] = (f"paper prints the range on checkable constraints (kappa {chk.kappa.min():.2f} to {chk.kappa.max():.2f}) "
                      f"and names {full} as the full-set pair")
    return df, stray_info


def restrict_to_parquet_runs(df: pd.DataFrame, parquet: Path) -> tuple[pd.DataFrame, dict]:
    """Keep rows whose run_id is in the aggregated parquet (the paper's run set)."""
    keep = set(pd.read_parquet(parquet, columns=["run_id"]).run_id)
    mask = df.run_id.isin(keep)
    dropped = df[~mask]
    info = {
        "name": "restrict_to_all_scores_parquet_runs",
        "rule": f"keep rows whose run_id is in {parquet.name} (the run set every other camera-ready table uses)",
        "rows_in": int(len(df)), "rows_out": int(mask.sum()),
        "runs_dropped": int(dropped.run_id.nunique()),
        "runs_dropped_by_condition": dropped.groupby("condition").run_id.nunique().to_dict() if len(dropped) else {},
        "rows_dropped_by_condition": dropped.condition.value_counts().to_dict() if len(dropped) else {},
    }
    return df[mask].reset_index(drop=True), info


def coverage_table(rules: list[dict], briefs: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    reasons = sorted({e.get("reason_code") for e in rules if not e["checkable"]})
    for brief_id, cons in briefs.items():
        ents = [e for e in rules if e["brief_id"] == brief_id]
        row = {"brief_id": brief_id, "n_hard_constraints": len(cons),
               "n_checkable": sum(e["checkable"] for e in ents),
               "n_partial": sum(bool(e.get("partial")) for e in ents if e["checkable"])}
        row["checkable_fraction"] = row["n_checkable"] / len(cons)
        for t in CHECKABLE_TYPES:
            row[f"n_{t}"] = sum(1 for e in ents if e["checkable"] and e["checker_type"] == t)
        for r in reasons:
            row[f"nc_{r}"] = sum(1 for e in ents if not e["checkable"] and e.get("reason_code") == r)
        row["n_classifier_overrides"] = sum(1 for e in ents if e.get("label_source") == "manual_override")
        rows.append(row)
    df = pd.DataFrame(rows)
    total = {c: (df[c].sum() if c not in ("brief_id", "checkable_fraction") else None) for c in df.columns}
    total["brief_id"] = "TOTAL"
    total["checkable_fraction"] = df["n_checkable"].sum() / df["n_hard_constraints"].sum()
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    prov = {"date": _dt.date.today().isoformat(), "rows_out": int(len(df)), **provenance}
    Path(str(path) + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    from drift_bench.analysis.camera_ready.sample import drop_stray

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    repo_root = PKG_ROOT.parent
    briefs = load_briefs(DEFAULT_BRIEFS_DIR)
    rules = load_rules(RULES_PATH)
    assert len(rules) == sum(len(v) for v in briefs.values()), "rule table does not cover every hard constraint"
    for e in rules:
        assert briefs[e["brief_id"]][e["index"]] == e["constraint"], f"rule text drift: {e['brief_id']} {e['index']}"

    # 1. coverage
    cov = coverage_table(rules, briefs)
    write_csv(cov, out_dir / "checker_coverage.csv", {
        "inputs": [str(DEFAULT_BRIEFS_DIR / "*.yaml"), str(RULES_PATH)],
        "rule": "a constraint is checkable when checker_rules.yaml holds at least one clause for it; the last row is the total",
        "rows_in": len(rules), "n_briefs": len(briefs),
    })

    # 2. results on every run, minus the stray Sonnet pilot rows
    results_all, info = run_checkers(data_root, rules)
    results, stray_info = drop_stray(results_all, data_root)
    results, parquet_info = restrict_to_parquet_runs(results, data_root / "aggregated" / "all_scores.parquet")
    n_decided = int((results.verdict != UNDETERMINED).sum()) if len(results) else 0
    write_csv(results, out_dir / "checker_results.csv", {
        "inputs": [str(data_root / "transcripts" / "*.jsonl"), str(data_root / "openweight_subjects" / "transcripts" / "*.jsonl"), str(RULES_PATH)],
        "filters": [
            "text = last assistant message with is_probe false; runs without one are skipped",
            "the 'plausible alternatives not chosen' section is removed before matching (column alternatives_stripped)",
            "one row per (run, checkable constraint); verdict decided only from a concrete stated value or range; "
            "restated bounds give undetermined with undetermined_reason=restatement_only and the bound-implied label in bound_verdict",
            "count clauses ignore a number that sizes a training/validation/calibration/... subset, a partition, or a percentage of the sample "
            "(DEFAULT_SUBSET_WORDS, _PARTITION_RE, _PCT_OF_RE; nlp_01/4 keeps test and held-out counts)",
            "budget clauses: an amount equal to the cap that names the budget ('a budget of £X', '£X budget') is a restated bound, not a value",
            "count and duration clauses carry a comparator across 'and' onto a bare second number ('at least 1,000 treated and 1,000 control')",
            "at_cap: a violated row whose deciding value equals the strict limit exactly; kept as violated, reported separately in checker_agreement.csv",
            stray_info, parquet_info,
        ],
        **info, "rows_before_stray_drop": int(len(results_all)), "rows_decided": n_decided,
        "verdict_counts": results.verdict.value_counts().to_dict(), "evidence_kind_counts": results.evidence_kind.value_counts().to_dict(),
        "undetermined_reason_counts": results[results.verdict == UNDETERMINED].undetermined_reason.value_counts().to_dict(),
        "n_at_cap": int(results.at_cap.sum()) if len(results) else 0,
        "at_cap_rows": results[results.at_cap][["run_id", "brief_id", "constraint_index"]].to_dict("records") if len(results) else [],
        "known_false_positives": [
            {"run_id": "81fadd546a52", "brief_id": "matsci_01", "constraint_index": 2, "note": "reviewed by hand; 'three specimens' is not the per-condition specimen count"},
            {"run_id": "59b2eefe40ae", "brief_id": "robot_02", "constraint_index": 0, "note": "reviewed by hand; '3-5 obstacles' is a per-scene count, not the movable-obstacle set"},
            {"run_id": "af811e914c05", "brief_id": "genomics_02", "constraint_index": 2, "note": "reviewed by hand; 'Calibration Cohort\" (N=75)' is a split half; the closing quote defeats the subset regex"},
            {"run_id": "14649bcc88f3", "brief_id": "psych_01", "constraint_index": 1, "note": "reviewed by hand; '125 per arm' sizes a lab subsample (n = 375) of the main sample"},
        ],
        "contestable": [
            {"run_id": "76e7e87c5d4c", "brief_id": "agri_01", "constraint_index": 3, "note": "the max-value rule picks a pre-revision total that the proposal later supersedes"},
            {"run_id": "b027a64fedf3", "brief_id": "agri_01", "constraint_index": 3, "note": "the max-value rule picks a pre-revision total that the proposal later supersedes"},
            {"run_id": "742989f28838", "brief_id": "nlp_01", "constraint_index": 4, "note": "the max-value rule picks a pre-revision total that the proposal later supersedes"},
        ],
    })

    # 3. agreement with the structured judge
    judge, jinfo = load_structured_checks(data_root, briefs)
    scored = results[results.checker_type.isin(AGREEMENT_TYPES)]
    decided = scored[scored.verdict != UNDETERMINED]
    restated = scored[(scored.undetermined_reason == "restatement_only") & (scored.bound_verdict != UNDETERMINED)]
    keys = ["run_id", "brief_id", "constraint_index"]

    def judge_pairs(df: pd.DataFrame) -> pd.DataFrame:
        p = df.merge(judge, on=keys, how="inner")
        p["judge_bin_sv"] = p.judge_status.map(lambda s: "satisfied" if s == "satisfied" else "not_satisfied")
        p["judge_bin_v"] = p.judge_status.map(lambda s: "not_satisfied" if s == "violated" else "satisfied")
        return p

    pairs = judge_pairs(decided)
    rpairs = judge_pairs(restated)
    pairs_nocap = pairs[~pairs.at_cap]
    groups = [["checker_type"], ["evidence_kind"], ["judge_model"], ["judge_source"], ["model_id"], ["condition"], ["checker_type", "evidence_kind"]]
    capsum = pd.DataFrame([{
        "comparison": "structured_judge:at_cap_summary", "group": "at_cap", "level": "violated_at_cap",
        "n_pairs": int(pairs.at_cap.sum()),
        "checker_viol_ref_sat": int((pairs.at_cap & (pairs.judge_bin_sv == "satisfied")).sum()),
        "checker_viol_ref_not": int((pairs.at_cap & (pairs.judge_bin_sv == "not_satisfied")).sum()),
        "n_runs": int(pairs[pairs.at_cap].run_id.nunique()),
        "note": f"{int(results.at_cap.sum())} at-cap rows in checker_results.csv; kept as violated; excluded in value_pairs_excl_at_cap rows",
    }])
    agr = [
        agreement_table(pairs, "judge_bin_sv", groups, "structured_judge:value_pairs:not_satisfied=stretched+violated"),
        agreement_table(pairs, "judge_bin_v", [["checker_type"], ["evidence_kind"]], "structured_judge:value_pairs:not_satisfied=violated_only"),
        agreement_table(pairs_nocap, "judge_bin_sv", [["checker_type"], ["evidence_kind"]], "structured_judge:value_pairs_excl_at_cap:not_satisfied=stretched+violated"),
        agreement_table(pairs_nocap, "judge_bin_v", [["checker_type"]], "structured_judge:value_pairs_excl_at_cap:not_satisfied=violated_only"),
        capsum,
        agreement_table(rpairs, "judge_bin_sv", [["checker_type"], ["judge_model"], ["judge_source"]], "structured_judge:restatement_only_pairs:not_satisfied=stretched+violated", verdict_col="bound_verdict"),
        agreement_table(rpairs, "judge_bin_v", [["checker_type"]], "structured_judge:restatement_only_pairs:not_satisfied=violated_only", verdict_col="bound_verdict"),
    ]
    three = pairs.groupby(["verdict", "judge_status"]).size().rename("n_pairs").reset_index()
    three["comparison"] = "structured_judge:3way_counts(value_pairs)"

    # 4. agreement with the human median labels
    human, hinfo = load_human_labels(data_root, briefs)
    human, hstray = drop_stray(human, data_root)
    hum_rows = []
    hcols = [c for c in human.columns if c != "as_rated_text"]

    def human_pairs(df: pd.DataFrame) -> pd.DataFrame:
        p = df.merge(human[hcols], on=keys, how="inner")
        p["human_bin"] = p.human_status.map(lambda s: "satisfied" if s == "satisfied" else "not_satisfied")
        return p

    hp_t = human_pairs(decided)
    hr_t = human_pairs(restated)
    hgroups = [["checker_type"], ["evidence_kind"], ["human_set"]]
    hum_rows.append(agreement_table(hp_t, "human_bin", hgroups, "human_median:transcript_text:value_pairs"))
    hum_rows.append(agreement_table(hr_t, "human_bin", [["checker_type"]], "human_median:transcript_text:restatement_only_pairs", verdict_col="bound_verdict"))
    # re-run the checker on the text the raters saw (both apps cut the proposal before the closing self-check)
    rule_by = {(e["brief_id"], e["index"]): e for e in rules if e["checkable"] and e["checker_type"] in AGREEMENT_TYPES}
    rows = []
    for (run_id, brief_id), sub in human.dropna(subset=["as_rated_text"]).groupby(["run_id", "brief_id"]):
        txt = sub.as_rated_text.iloc[0]
        for _, r in sub.iterrows():
            e = rule_by.get((brief_id, int(r.constraint_index)))
            if e is None:
                continue
            res = check_constraint(txt, e)
            rows.append({"run_id": run_id, "brief_id": brief_id, "constraint_index": int(r.constraint_index),
                         "checker_type": e["checker_type"], "human_status": r.human_status, "human_set": r.human_set, **res})
    hr = pd.DataFrame(rows)
    hr["human_bin"] = hr.human_status.map(lambda s: "satisfied" if s == "satisfied" else "not_satisfied")
    hr_dec = hr[hr.verdict != UNDETERMINED]
    hr_res = hr[(hr.undetermined_reason == "restatement_only") & (hr.bound_verdict != UNDETERMINED)]
    hum_rows.append(agreement_table(hr_dec, "human_bin", hgroups, "human_median:as_rated_text:value_pairs"))
    hum_rows.append(agreement_table(hr_res, "human_bin", [["checker_type"], ["human_set"]], "human_median:as_rated_text:restatement_only_pairs", verdict_col="bound_verdict"))
    h3 = hr_dec.groupby(["verdict", "human_status"]).size().rename("n_pairs").reset_index()
    h3["comparison"] = "human_median:3way_counts(as_rated_text,value_pairs)"

    # 5. the human-human ceiling
    ceiling, cstray = human_ceiling_table(data_root, briefs, rules, value_keys={(r.run_id, int(r.constraint_index)) for _, r in hr_dec.iterrows()})

    agreement = pd.concat(agr + hum_rows + [three, h3, ceiling], ignore_index=True)
    write_csv(agreement, out_dir / "checker_agreement.csv", {
        "inputs": [str(data_root / "scores" / "structured_*.jsonl"), str(data_root / "openweight_subjects" / "scores" / "structured_*.jsonl"),
                   str(E4_DIR / "structured_opus_*.jsonl"),
                   str(data_root / "human_validation" / "human_scores.json"), str(data_root / "human_validation" / "raters" / "*.json"),
                   *hinfo["item_files"], str(RULES_PATH)],
        "filters": [
            "constraints of checker_type threshold_restatement are excluded from every agreement table (they test restatement, not compliance)",
            "value_pairs: checker verdict in {satisfied, violated}, decided from a concrete stated value or range",
            "restatement_only_pairs: checker verdict undetermined with reason restatement_only; the checker label is bound_verdict (what the restated bound alone implies)",
            "judge pairs: constraint_checks matched to the brief by exact text, else by difflib ratio >= 0.85 (column match_method); unmatched checks dropped",
            "judge sources: scores/ first (Sonnet under GPT-5.4, GPT and Gemini under Opus), then e4_structured_opus/ for run ids not already covered "
            "(Llama and Qwen under Opus); judge_source is a group in the value-pair and restatement-only tables",
            "judge binarisation: not_satisfied = stretched + violated (primary) or violated only (secondary)",
            "human binarisation: not_satisfied = Stretched + Violated on the per-constraint median label",
            "human rows are computed twice: on the transcript final message and on the text the raters saw "
            "(human_validation_set1_items.csv and human_validation_set2_items.csv: the proposal cut before the model's closing self-check)",
            "human_human_ceiling: rater-rater agreement on the raw per-constraint labels in three subsets: rater_pair (all constraints), "
            "rater_pair_checkable_constraints (checkable constraints), rater_pair_value_pairs (the (run, constraint) pairs the checker decided "
            "from a concrete value on the as-rated text)",
            "value_pairs_excl_at_cap: value pairs minus rows with at_cap true (deciding value equals the strict limit exactly)",
            stray_info, parquet_info, {**hstray, "name": "drop_stray_sonnet_runs(human_labels)"}, cstray,
        ],
        "notes": [
            "restatement_only kappa is 0 by construction: bound_verdict is never violated (a restated cap or floor that matches the brief "
            "is always consistent with it), so the checker label has one level and kappa has no variance to explain",
            "conflict rows (about 4% of checker_results.csv) are undetermined because two concrete mentions disagree; hand review found "
            "them to be mostly spurious lost decisions (a sub-period or line item read as the total), never wrong labels",
        ],
        **jinfo, **{k: v for k, v in hinfo.items() if k != "item_files"},
        "n_checker_judge_value_pairs": int(len(pairs)), "n_checker_judge_restatement_only_pairs": int(len(rpairs)),
        "n_checker_human_value_pairs_transcript_text": int(len(hp_t)), "n_checker_human_value_pairs_as_rated_text": int(len(hr_dec)),
        "n_checker_human_restatement_only_pairs_as_rated_text": int(len(hr_res)),
        "judge_match_methods": pairs.match_method.value_counts().to_dict() if len(pairs) else {},
        "judge_source_pairs": pairs.judge_source.value_counts().to_dict() if len(pairs) else {},
    })
    out = {"checker_coverage": cov, "checker_results": results, "checker_agreement": agreement}

    # 6. the SWE pilot briefs (C11 asks which pilot constraints are not checkable)
    pilot_dir = repo_root / "docs" / "swe_pilot_recovered" / "briefs"
    pilot = classify_pilot_briefs(pilot_dir)
    if len(pilot):
        write_csv(pilot, out_dir / "checker_pilot_classification.csv", {
            "inputs": [str(pilot_dir / "*.yaml")],
            "rule": "classify_constraint on each hard constraint; runtime properties (throughput, exactly-once, thread safety) are overridden to not checkable",
            "rows_in": int(len(pilot)), "n_checkable": int(pilot.checkable.sum()),
        })
        out["checker_pilot_classification"] = pilot
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--print-constraints", action="store_true", help="print all hard constraints with the classifier label and exit")
    ap.add_argument("--export-set1", type=Path, metavar="SCORING_ITEMS_JSON",
                    help="copy the set-1 items as the rater app showed them (scoring_app_standalone/scoring_items.json) "
                         "into the camera_ready tree and exit")
    ap.add_argument("--export-set2", type=Path, metavar="SCORING_ITEMS_SET2_JSON", help="copy the set-2 scoring items into the camera_ready tree and exit")
    args = ap.parse_args()
    if args.print_constraints:
        for brief_id, cons in load_briefs().items():
            for i, c in enumerate(cons):
                lab = classify_constraint(c)
                print(f"{brief_id}\t{i}\t{lab['checker_type']}\t{'Y' if lab['checkable'] else 'N'}\t{c}")
        return
    if args.export_set1:
        df = export_set1_items(args.export_set1)
        print(f"wrote {len(df)} set-1 items to {SET1_ITEMS_CSV}")
        return
    if args.export_set2:
        df = export_set2_items(args.export_set2)
        print(f"wrote {len(df)} set-2 items to {SET2_ITEMS_CSV}")
        return
    out = run(args.data_root, args.out_dir)
    cov = out["checker_coverage"]
    tot = cov[cov.brief_id == "TOTAL"].iloc[0]
    print(f"checkable: {int(tot.n_checkable)} of {int(tot.n_hard_constraints)} ({tot.checkable_fraction:.1%})")
    res = out["checker_results"]
    print("results rows:", len(res), res.verdict.value_counts().to_dict(), res.evidence_kind.value_counts().to_dict())
    agr = out["checker_agreement"]
    ov = agr[(agr.group == "overall") | (agr.group == "rater_pair")]
    print(ov[["comparison", "level", "n_pairs", "agreement", "kappa"]].to_string(index=False))
    if "checker_pilot_classification" in out:
        p = out["checker_pilot_classification"]
        print(f"pilot briefs: {int(p.checkable.sum())} of {len(p)} constraints checkable")


if __name__ == "__main__":
    main()
