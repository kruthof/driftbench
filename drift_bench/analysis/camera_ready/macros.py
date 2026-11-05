"""Emit every number the paper prints as a LaTeX macro (plan items X6, B19, B5).

The module reads the camera-ready CSVs in ``drift_bench/data/camera_ready/analysis/``
and writes three files next to them:

* ``cr_macros.tex``: one ``\\newcommand{\\crName}{value}`` per macro.
* ``cr_macros.json``: name, value, source CSV, row selector, column, format.
* ``cr_macros.tex.provenance.json``: inputs, selection rule, counts, the
  ``missing`` list, and the date.

The registry is declarative. A ``MacroSpec`` names the CSV, a ``where``
dict that selects exactly one row, the column to read, and a format. A
missing CSV, an empty selection, a multi-row selection or a NaN value raises
a clear error when the spec is ``required``; otherwise the spec lands in the
``missing`` list of the provenance and no macro is emitted for it. The tex
file never contains ``None`` or ``nan``.

The ``corrections_ledger.csv`` rows are registered dynamically: one macro
per row (``\\crLedger<Slug>``), a ``Frac`` variant when the row has a
numerator and a denominator, and a ``Pub`` variant when the row has a
published value.

Run: ``python -m drift_bench.analysis.camera_ready.macros``
      ``python -m drift_bench.analysis.camera_ready.macros --list``
      ``python -m drift_bench.analysis.camera_ready.macros --check-tex neurips_2026.tex``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"

MACRO_PREFIX = "cr"
NAME_RE = re.compile(r"^\\cr[A-Z][A-Za-z]*$")
TEX_MACRO_RE = re.compile(r"\\cr[A-Z][A-Za-z]*")
BAD_WORD_RE = re.compile(r"\b(nan|none)\b", re.IGNORECASE)
MATH_RE = re.compile(r"\$[^$]*\$")

FORMATS = (
    "int",  # 2116 -> 2{,}116
    "round",  # 8048.3 -> 8{,}048
    "pct0",  # 0.25 -> 25
    "pct1",  # 0.25 -> 25.0
    "dec1",  # 15.998 -> 16.0
    "dec2",  # 2.447 -> 2.45
    "dec3",  # -1.1184 -> -1.118
    "pval",  # 0.0003 -> <0.001 ; 0.0510 -> 0.051
    "ci",  # (0.166, 0.358) -> [16.6, 35.8]   (percent, one decimal)
    "cidec2",  # (1.368, 4.376) -> [1.37, 4.38]
    "cidec3",  # (-0.405, -0.269) -> [-0.405, -0.269]
    "frac",  # (19, 76) -> 19/76
    "auto",  # ints with separators, floats to at most 3 decimals, text escaped
    "nproviders",  # "openai/a;gemini/b;gemini/c" -> 2 (distinct model developers, MODEL_PROVIDERS)
)

MODEL_LABELS = {
    "openai/gpt-5.4": "Gpt",
    "openai/gpt-5.4-mini": "Mini",
    "anthropic/claude-sonnet-4-6": "Sonnet",
    "gemini/gemini-3.1-pro-preview": "Pro",
    "gemini/gemini-3.1-flash-lite-preview": "FlashLite",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "Qwen",
}
# Model developer by model-id prefix ("provider" in the paper: seven models from five providers).
# The longest matching prefix wins; case is ignored. together_ai/ is the API host of the
# open-weight models, not their developer, so the next path segment decides. Gemini models are
# Google under any of the prefixes LiteLLM uses for them. An unlisted id raises.
MODEL_PROVIDERS = {
    "openai/": "OpenAI",
    "anthropic/": "Anthropic",
    "gemini/": "Google",
    "google/": "Google",
    "vertex_ai/gemini": "Google",
    "together_ai/meta-llama/": "Meta",
    "together_ai/Qwen/": "Alibaba",
}
STRUCTURED_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
]
CONDITION_LABELS = {
    "single_shot": "SingleShot",
    "multi_turn_neutral": "Neutral",
    "multi_turn_pressure": "Pressure",
    "checkpointed_pressure": "Checkpoint",
}
MULTI_TURN = ["multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"]
DIMENSIONS = {
    "objective_fidelity": "Fidelity",
    "constraint_adherence": "Adherence",
    "alternative_coverage": "Coverage",
    "complexity_inflation": "Complexity",
}
TYPOLOGY_TYPES = {
    "resource_limit": "ResourceLimit",
    "data_source": "DataSource",
    "count_scope_limit": "CountScopeLimit",
    "sample_floor": "SampleFloor",
    "method_requirement": "MethodRequirement",
    "design_requirement": "DesignRequirement",
    "duration_window": "DurationWindow",
    "prohibition": "Prohibition",
}

# Labels for the experiment tables (E1 crossed judge, E2, E3, E5, E6, E7).
JUDGE_ARMS = {"opus": "Opus", "gpt": "Gpt", "published": "Published"}
KBV_VARIANTS = {"clean": "", "clean_truncated_probes_waived": "Waived"}
E2_SOURCES = {"april_clean": "April", "e2_uncapped": "Uncapped"}
E2_CONDITIONS = {"single_shot": "SingleShot", "multi_turn_pressure": "Pressure"}
E2_DESIGNS = {"unpaired_runs": "Unpaired", "brief_paired": "Paired"}
# metric -> (label, format of the delta, format of the CI pair or None for CiLow/CiHigh)
E2_METRICS = {
    "kbv_rate": ("Kbv", "pct1", "ci"),
    "objective_fidelity": ("Fidelity", "dec2", "cidec2"),
    "constraint_adherence": ("Adherence", "dec2", "cidec2"),
    "alternative_coverage": ("Coverage", "dec2", "cidec2"),
    "complexity_inflation": ("Complexity", "dec2", "cidec2"),
    "output_tokens_per_run": ("OutputTokens", "round", None),
    "final_tokens": ("FinalTokens", "round", None),
    "cap_turn_share": ("CapTurnShare", "pct1", "ci"),
    "final_cap_share": ("FinalCapShare", "pct1", "ci"),
}
E5_MODELS = {"openai/gpt-5.4": "Gpt", "gemini/gemini-3.1-flash-lite-preview": "FlashLite", "pooled": "Pooled"}
E5_CONDITIONS = {"multi_turn_neutral": "Neutral", "multi_turn_pressure": "Pressure", "pooled": ""}
E5_CONTRASTS = {"t2-t1": "TwoOne", "t5-t1": "FiveOne", "t5-t2": "FiveTwo", "t6april-t5": "SixAprilFive", "t6check-t5": "SixCheckFive"}
E7_MODELS = {
    "openai/gpt-5.4": "Gpt",
    "anthropic/claude-sonnet-4-6": "Sonnet",
    "gemini/gemini-3.1-pro-preview": "Pro",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama",
    "all": "Pooled",
}
SWE_MODELS = {"gemini/gemini-3.1-flash-lite-preview": "FlashLite", "gemini/gemini-3.1-pro-preview": "Pro", "openai/gpt-5.4-mini": "Mini"}
SWE_EXEC_MODELS = ("gemini/gemini-3.1-pro-preview", "openai/gpt-5.4-mini")
SWE_ARMS = {"main": "", "exec": "Exec"}

DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
TEEN_WORDS = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
TENS_WORDS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}


class MacroError(ValueError):
    """A spec could not be resolved. The message names the spec."""


@dataclass(frozen=True)
class MacroSpec:
    """One macro: where to read it and how to print it.

    ``column`` is one column name, or a pair of names for ``ci*`` and
    ``frac`` formats. ``where`` maps column -> value; the match is exact on
    the stripped string, or numeric when both sides parse as numbers. A
    list or tuple value matches any of its items.
    ``agg`` (``sum``, ``mean``, ``count``, ``min``, ``max``) allows a
    multi-row selection and reduces it. ``min`` and ``max`` return the cell
    of ``column`` in the row where ``by`` (default: ``column``) is smallest
    or largest; with ``by`` set, a tie for that extreme raises, because the
    label it would print is ambiguous. ``required=False`` sends a failure to
    the ``missing`` list instead of raising. ``default`` is emitted verbatim
    when the selection is empty (for zero-count categories a summary CSV
    omits).
    """

    name: str
    csv: str
    where: dict[str, Any]
    column: str | tuple[str, str]
    fmt: str
    required: bool = True
    agg: str | None = None
    note: str = ""
    default: str | None = None
    by: str | None = None

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise MacroError(f"bad macro name {self.name!r}: must match {NAME_RE.pattern}")
        if self.fmt not in FORMATS:
            raise MacroError(f"{self.name}: unknown fmt {self.fmt!r}; use one of {FORMATS}")
        two = self.fmt in ("ci", "cidec2", "cidec3", "frac")
        if two and not (isinstance(self.column, tuple) and len(self.column) == 2):
            raise MacroError(f"{self.name}: fmt {self.fmt!r} needs a (low, high) or (num, den) column pair")
        if not two and not isinstance(self.column, str):
            raise MacroError(f"{self.name}: fmt {self.fmt!r} needs one column name")
        if self.agg is not None and self.agg not in ("sum", "mean", "count", "min", "max"):
            raise MacroError(f"{self.name}: unknown agg {self.agg!r}")
        if self.by is not None and self.agg not in ("min", "max"):
            raise MacroError(f"{self.name}: `by` needs agg min or max, not {self.agg!r}")

    def describe(self) -> str:
        return f"{self.name} <- {self.csv} where {self.where} column {self.column} fmt {self.fmt}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def _is_blank(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    s = str(raw).strip()
    return s == "" or s.lower() in ("nan", "none", "null", "na", "<na>")


def _to_float(raw: Any) -> float:
    if _is_blank(raw):
        raise MacroError("value is blank or NaN")
    try:
        x = float(str(raw).strip())
    except ValueError as exc:
        raise MacroError(f"value {raw!r} is not a number") from exc
    if math.isnan(x) or math.isinf(x):
        raise MacroError(f"value {raw!r} is not finite")
    return x


def _to_decimal(raw: Any) -> Decimal:
    """Exact decimal of the cell text (or repr of a float from an agg)."""
    _to_float(raw)  # validates: not blank, numeric, finite
    if isinstance(raw, (int, float)):
        return Decimal(repr(raw))
    return Decimal(str(raw).strip())


def _quant(d: Decimal, places: int) -> str:
    return str(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_int(raw: Any) -> str:
    d = _to_decimal(raw)
    whole = d.to_integral_value(rounding=ROUND_HALF_UP)
    if abs(d - whole) > Decimal("1e-9"):
        raise MacroError(f"value {raw!r} is not an integer")
    return _thousands(int(whole))


def fmt_round(raw: Any) -> str:
    return _thousands(int(_to_decimal(raw).to_integral_value(rounding=ROUND_HALF_UP)))


def fmt_pct(raw: Any, places: int) -> str:
    return _quant(_to_decimal(raw) * 100, places)


def fmt_dec(raw: Any, places: int) -> str:
    return _quant(_to_decimal(raw), places)


def fmt_pval(raw: Any) -> str:
    """Bare p value: '<0.001' below the cutoff, else three decimals."""
    d = _to_decimal(raw)
    if d < 0:
        raise MacroError(f"p value {raw!r} is negative")
    if d > 1:
        raise MacroError(f"p value {raw!r} is above 1")
    if d < Decimal("0.001"):
        return "<0.001"
    return _quant(d, 3)


def fmt_pval_tex(bare: str) -> str:
    """Self-contained prose form: '$p < 0.001$' or '$p = 0.086$'."""
    return "$p < 0.001$" if bare == "<0.001" else f"$p = {bare}$"


def fmt_pval_val(bare: str) -> str:
    """Table form: the number only; '<' is wrapped in math so it is never bare."""
    return "$<$0.001" if bare == "<0.001" else bare


def fmt_ci(low: Any, high: Any, scale: int, places: int) -> str:
    lo = _quant(_to_decimal(low) * scale, places)
    hi = _quant(_to_decimal(high) * scale, places)
    return f"[{lo}, {hi}]"


def fmt_frac(num: Any, den: Any) -> str:
    return f"{fmt_int(num)}/{fmt_int(den)}"


def fmt_auto(raw: Any) -> str:
    """Ints get separators, floats keep at most 3 decimals, text is escaped."""
    if _is_blank(raw):
        raise MacroError("value is blank or NaN")
    s = str(raw).strip()
    try:
        return fmt_int(int(s))
    except (ValueError, MacroError):
        pass
    try:
        d = Decimal(s)
    except InvalidOperation:
        return latex_escape(s)
    if not d.is_finite():
        raise MacroError(f"value {raw!r} is not finite")
    if d.as_tuple().exponent < -3:
        d = d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    if d == d.to_integral_value() and "." not in s and "e" not in s.lower():
        return fmt_int(int(d))
    return format(d, "f")


def model_provider(model_id: str) -> str:
    """Developer of ``model_id`` from ``MODEL_PROVIDERS`` (longest prefix, case ignored)."""
    mid = str(model_id).strip().lower()
    hits = [prefix for prefix in MODEL_PROVIDERS if mid.startswith(prefix.lower())]
    if not mid or not hits:
        raise MacroError(f"no provider for model id {model_id!r}; add its prefix to MODEL_PROVIDERS")
    return MODEL_PROVIDERS[max(hits, key=len)]


def fmt_nproviders(raw: Any) -> str:
    """Number of distinct providers among the semicolon-separated model ids of one cell."""
    if _is_blank(raw):
        raise MacroError("value is blank or NaN")
    ids = [part.strip() for part in str(raw).split(";") if part.strip()]
    return fmt_int(len({model_provider(mid) for mid in ids}))


def format_value(spec: MacroSpec, values: Any) -> str:
    """Format the value(s) read for ``spec``. ``values`` is a scalar or a pair."""
    f = spec.fmt
    if f == "int":
        return fmt_int(values)
    if f == "round":
        return fmt_round(values)
    if f == "pct0":
        return fmt_pct(values, 0)
    if f == "pct1":
        return fmt_pct(values, 1)
    if f == "dec1":
        return fmt_dec(values, 1)
    if f == "dec2":
        return fmt_dec(values, 2)
    if f == "dec3":
        return fmt_dec(values, 3)
    if f == "pval":
        return fmt_pval(values)
    if f == "ci":
        return fmt_ci(values[0], values[1], 100, 1)
    if f == "cidec2":
        return fmt_ci(values[0], values[1], 1, 2)
    if f == "cidec3":
        return fmt_ci(values[0], values[1], 1, 3)
    if f == "frac":
        return fmt_frac(values[0], values[1])
    if f == "auto":
        return fmt_auto(values)
    if f == "nproviders":
        return fmt_nproviders(values)
    raise MacroError(f"{spec.name}: unknown fmt {f!r}")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def read_csv_str(path: Path) -> pd.DataFrame:
    """Read a CSV with every cell as a string so formatting is exact."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _match_mask(series: pd.Series, value: Any) -> pd.Series:
    # A list or tuple value matches a row when the cell matches any of its items.
    if isinstance(value, (list, tuple)):
        mask = pd.Series(False, index=series.index)
        for item in value:
            mask |= _match_mask(series, item)
        return mask
    text = series.astype(str).str.strip()
    mask = text == str(value).strip()
    try:
        target = float(str(value).strip())
    except ValueError:
        return mask
    numeric = pd.to_numeric(text, errors="coerce")
    return mask | (numeric == target)


def select_rows(df: pd.DataFrame, spec: MacroSpec) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in spec.where.items():
        if col not in df.columns:
            raise MacroError(f"{spec.name}: column {col!r} not in {spec.csv} (columns: {list(df.columns)})")
        mask &= _match_mask(df[col], value)
    rows = df[mask]
    if rows.empty and spec.default is None:
        raise MacroError(f"{spec.name}: no row in {spec.csv} matches {spec.where}")
    if len(rows) > 1 and spec.agg is None:
        raise MacroError(f"{spec.name}: {len(rows)} rows in {spec.csv} match {spec.where}; expected one")
    return rows


def _extreme_row(rows: pd.DataFrame, spec: MacroSpec, key: str) -> Any:
    """Index label of the row where ``key`` is smallest (agg min) or largest (agg max)."""
    if key not in rows.columns:
        raise MacroError(f"{spec.name}: column {key!r} not in {spec.csv}")
    numeric = pd.to_numeric(rows[key], errors="coerce")
    if numeric.isna().any():
        raise MacroError(f"{spec.name}: non-numeric or blank cells in {spec.csv} column {key!r} under agg {spec.agg}")
    target = numeric.min() if spec.agg == "min" else numeric.max()
    hits = list(numeric.index[numeric == target])
    if len(hits) > 1 and spec.by is not None:
        raise MacroError(f"{spec.name}: {len(hits)} rows of {spec.csv} tie for the {spec.agg} of {key!r}; the label is ambiguous")
    return hits[0]


def _column_values(rows: pd.DataFrame, spec: MacroSpec, col: str) -> Any:
    if col not in rows.columns:
        raise MacroError(f"{spec.name}: column {col!r} not in {spec.csv}")
    if spec.agg is None:
        raw = rows[col].iloc[0]
        if _is_blank(raw):
            raise MacroError(f"{spec.name}: {spec.csv} column {col!r} is blank for {spec.where}")
        return raw
    if spec.agg in ("min", "max"):
        raw = rows.at[_extreme_row(rows, spec, spec.by or col), col]
        if _is_blank(raw):
            raise MacroError(f"{spec.name}: {spec.csv} column {col!r} is blank in the {spec.agg} row for {spec.where}")
        return raw
    numeric = pd.to_numeric(rows[col], errors="coerce")
    if spec.agg == "count":
        return float(len(rows))
    if numeric.isna().any():
        raise MacroError(f"{spec.name}: non-numeric or blank cells in {spec.csv} column {col!r} under agg")
    return float(numeric.sum()) if spec.agg == "sum" else float(numeric.mean())


def resolve(spec: MacroSpec, csv_dir: Path, cache: dict[str, pd.DataFrame]) -> tuple[str, Any]:
    """Return (formatted value, raw value(s)) for ``spec``."""
    path = csv_dir / spec.csv
    if spec.csv not in cache:
        if not path.exists():
            raise MacroError(f"{spec.name}: CSV not found: {path}")
        cache[spec.csv] = read_csv_str(path)
    rows = select_rows(cache[spec.csv], spec)
    if rows.empty:
        # Empty selection with a default: the summary omitted a zero-count row.
        value, raw = str(spec.default), str(spec.default)
    else:
        if isinstance(spec.column, tuple):
            raw = (_column_values(rows, spec, spec.column[0]), _column_values(rows, spec, spec.column[1]))
        else:
            raw = _column_values(rows, spec, spec.column)
        value = format_value(spec, raw)
    if BAD_WORD_RE.search(value):
        raise MacroError(f"{spec.name}: formatted value {value!r} contains None or nan")
    return value, raw


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def number_words(token: str) -> str:
    """'2' -> 'Two', '15' -> 'Fifteen', '42' -> 'FortyTwo', '4096' -> 'FourZeroNineSix'."""
    n = int(token)
    if len(token) <= 2 and not token.startswith("0"):
        if n < 10:
            return DIGIT_WORDS[n]
        if n < 20:
            return TEEN_WORDS[n - 10]
        return TENS_WORDS[n // 10] + (DIGIT_WORDS[n % 10] if n % 10 else "")
    return "".join(DIGIT_WORDS[int(ch)] for ch in token)


def slug(text: str) -> str:
    """CamelCase letters only. Digits are spelled out. 'turn 2' -> 'TurnTwo'."""
    parts: list[str] = []
    for token in re.findall(r"[A-Za-z]+|[0-9]+", str(text)):
        if token.isdigit():
            parts.append(number_words(token))
        else:
            parts.append(token[0].upper() + token[1:].lower())
    out = "".join(parts)
    if not out:
        raise MacroError(f"cannot build a macro name from {text!r}")
    return out


def _name(*parts: str) -> str:
    return "\\" + MACRO_PREFIX + "".join(parts)


# ---------------------------------------------------------------------------
# Static registry
# ---------------------------------------------------------------------------


def _kbv_specs() -> list[MacroSpec]:
    csv = "kbv_published_arm.csv"
    specs: list[MacroSpec] = []
    for model, label in MODEL_LABELS.items():
        for cond in MULTI_TURN:
            w = {"sample": "clean", "model_id": model, "condition": cond}
            base = "Kbv" + label + CONDITION_LABELS[cond]
            specs += [
                MacroSpec(_name(base), csv, w, "kbv_rate", "pct1"),
                MacroSpec(_name(base, "Frac"), csv, w, ("kbv_count", "n_with_final_probe"), "frac"),
                MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci"),
            ]
        w = {"sample": "as_published", "model_id": model, "condition": "multi_turn_pressure"}
        base = "KbvPub" + label + "Pressure"
        specs += [
            MacroSpec(_name(base), csv, w, "kbv_rate", "pct1"),
            MacroSpec(_name(base, "Frac"), csv, w, ("kbv_count", "n_with_final_probe"), "frac"),
            MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci"),
        ]
    pro = "gemini/gemini-3.1-pro-preview"
    for cond in MULTI_TURN:
        clabel = CONDITION_LABELS[cond]
        w = {"sample": "clean_truncated_probes_waived", "model_id": pro, "condition": cond}
        specs += [
            MacroSpec(_name("KbvPro", clabel, "Waived"), csv, w, "kbv_rate", "pct1"),
            MacroSpec(_name("KbvPro", clabel, "WaivedFrac"), csv, w, ("kbv_count", "n_with_final_probe"), "frac"),
            MacroSpec(_name("KbvPro", clabel, "WaivedCi"), csv, w, ("ci_low", "ci_high"), "ci"),
            MacroSpec(_name("KbvPro", clabel, "WaivedImputed"), csv, w, "knows_imputed", "int"),
        ]
        w = {"sample": "clean_untruncated_probes", "model_id": pro, "condition": cond}
        specs += [
            MacroSpec(_name("KbvPro", clabel, "Excluded"), csv, w, "kbv_rate", "pct1"),
            MacroSpec(_name("KbvPro", clabel, "ExcludedFrac"), csv, w, ("kbv_count", "n_with_final_probe"), "frac"),
            MacroSpec(_name("KbvPro", clabel, "ExcludedCi"), csv, w, ("ci_low", "ci_high"), "ci"),
        ]
    # Sonnet pressure denominators (B15).
    den = "kbv_probe_denominator.csv"
    w = {"sample": "as_published", "model_id": "anthropic/claude-sonnet-4-6", "condition": "multi_turn_pressure"}
    specs += [
        MacroSpec(_name("SonnetPubPressureScored"), den, w, "n_scored", "int"),
        MacroSpec(_name("SonnetPubPressureNoProbe"), den, w, "n_without_final_probe", "int"),
        MacroSpec(_name("SonnetPubPressureWithProbe"), den, w, "n_with_final_probe", "int"),
    ]
    w = {"sample": "clean", "model_id": "anthropic/claude-sonnet-4-6", "condition": "multi_turn_pressure"}
    specs += [
        MacroSpec(_name("SonnetPressureScored"), den, w, "n_scored", "int"),
        MacroSpec(_name("SonnetPressureWithProbe"), den, w, "n_with_final_probe", "int"),
    ]
    # Above-50% membership (A12).
    above = "kbv_above50_pressure.csv"
    for sample, label in (("clean", ""), ("as_published", "Pub")):
        w = {"sample": sample, "condition": "multi_turn_pressure", "cut": "0.5"}
        specs += [
            MacroSpec(_name("KbvAboveHalf", label, "Models"), above, w, "n_above", "int"),
            MacroSpec(_name("KbvAboveHalf", label, "Frac"), above, w, ("n_above", "n_models"), "frac"),
            MacroSpec(_name("KbvAboveHalf", label, "Total"), above, w, "n_models", "int"),
        ]
    return specs


def _noncompliance_specs() -> list[MacroSpec]:
    csv = "noncompliance_by_condition.csv"
    specs: list[MacroSpec] = []
    for cond, clabel in CONDITION_LABELS.items():
        for sample, slabel in (("clean", ""), ("as_published", "Pub")):
            w = {"sample": sample, "condition": cond}
            base = "Noncompliant" + slabel + clabel
            specs += [
                MacroSpec(_name(base), csv, w, "rate", "pct1"),
                MacroSpec(_name(base, "Frac"), csv, w, ("noncompliant", "n"), "frac"),
                MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci"),
                MacroSpec(_name(base, "N"), csv, w, "n", "int"),
            ]
    specs += [
        MacroSpec(_name("CleanN"), csv, {"sample": "clean"}, "n", "int", agg="sum",
                  note="rows of the clean sample over the four conditions"),
        MacroSpec(_name("PublishedN"), csv, {"sample": "as_published"}, "n", "int", agg="sum",
                  note="rows of all_scores.parquet over the four conditions"),
    ]
    return specs


def _failed_runs_specs() -> list[MacroSpec]:
    csv = "failed_runs_summary.csv"
    rows = {
        "FailedRuns": "failed_runs",
        "SubjectFailures": "subject_failures",
        "ScoringFailures": "scoring_failures",
        "DesignRows": "design_rows",
        "ParquetRows": "parquet_rows",
        "NonSonnetShortfall": "non_sonnet_design_shortfall",
        "SonnetNoTranscript": "sonnet_scored_runs_without_transcript",
        "RigorTranscripts": "transcripts_outside_main_conditions",
        "StrayN": "anomaly:stray_pilot_row",
        "TranscriptFiles": "rows_in:transcripts",
        "JudgeFiles": "rows_in:judge_files",
        "AuditorFiles": "rows_in:auditor_files",
    }
    return [MacroSpec(_name(n), csv, {"quantity": q}, "value", "int") for n, q in rows.items()]


def _per_model_specs() -> list[MacroSpec]:
    csv = "per_model_condition_means.csv"
    specs: list[MacroSpec] = []
    for model, mlabel in MODEL_LABELS.items():
        for cond, clabel in CONDITION_LABELS.items():
            w = {"model_id": model, "condition": cond}
            specs.append(MacroSpec(_name("N", mlabel, clabel), csv, w, "n", "int", required=False))
            for dim, dlabel in DIMENSIONS.items():
                specs.append(MacroSpec(_name("Mean", dlabel, mlabel, clabel), csv, w, f"j_{dim}_mean", "dec2", required=False))
                specs.append(MacroSpec(_name("Se", dlabel, mlabel, clabel), csv, w, f"j_{dim}_se", "dec2", required=False))
            specs.append(MacroSpec(_name("ShareTrajectoryDrift", mlabel, clabel), csv, w, "share_trajectory_drift", "pct1", required=False))
            if cond != "single_shot":
                specs.append(MacroSpec(_name("KbvTable", mlabel, clabel), csv, w, "kbv_rate", "pct1", required=False))
                specs.append(MacroSpec(_name("KnowsRate", mlabel, clabel), csv, w, "knows_rate", "pct1", required=False))
    w = {"model_id": "anthropic/claude-sonnet-4-6", "condition": "multi_turn_pressure"}
    specs.append(MacroSpec(_name("MeanPubAdherenceSonnetPressure"), "per_model_condition_means_as_published.csv", w,
                           "j_constraint_adherence_mean", "dec2", required=False))
    specs.append(MacroSpec(_name("NPubSonnetPressure"), "per_model_condition_means_as_published.csv", w, "n", "int", required=False))
    return specs


def _graded_severity_specs() -> list[MacroSpec]:
    summ = "graded_severity_summary.csv"
    tests = "graded_severity_tests.csv"
    specs: list[MacroSpec] = []
    for model in STRUCTURED_MODELS:
        mlabel = MODEL_LABELS[model]
        for cond, clabel in CONDITION_LABELS.items():
            w = {"source": "structured", "model_id": model, "condition": cond}
            specs += [
                MacroSpec(_name("Gkbv", mlabel, clabel), summ, w, "mean_g_kbv", "pct1", required=False),
                MacroSpec(_name("Gkbv", mlabel, clabel, "Ci"), summ, w, ("g_kbv_ci_lo", "g_kbv_ci_hi"), "ci", required=False),
            ]
            if cond == "multi_turn_pressure":
                # X12: the rebuttal's perfect-recall row (pressure only). Perfect recall is
                # final-probe overall_accuracy == 1.0 (graded_severity.attach_recall).
                note = "pressure; perfect recall = final-probe overall_accuracy == 1.0"
                specs += [
                    MacroSpec(_name("Gkbv", mlabel, clabel, "N"), summ, w, "n_runs", "int", required=False),
                    MacroSpec(_name("Gkbv", mlabel, clabel, "PerfectRecall"), summ, w, "mean_g_kbv_perfect_recall", "pct1",
                              required=False, note=note),
                    MacroSpec(_name("Gkbv", mlabel, clabel, "PerfectRecallN"), summ, w, "n_perfect_recall", "int",
                              required=False, note=note),
                ]
        w = {"source": "structured", "model_id": model}
        specs += [
            MacroSpec(_name("Gkbv", mlabel, "Delta"), tests, w, "mean_brief_delta", "dec3", required=False),
            MacroSpec(_name("Gkbv", mlabel, "WilcoxonP"), tests, w, "wilcoxon_p", "pval", required=False),
            MacroSpec(_name("Gkbv", mlabel, "HolmP"), tests, w, "holm_p_within_judge", "pval", required=False),
            MacroSpec(_name("Gkbv", mlabel, "HolmAllP"), tests, w, "holm_p", "pval", required=False),
            MacroSpec(_name("Gkbv", mlabel, "PairedBriefs"), tests, w, "n_briefs_paired", "int", required=False),
        ]
    return specs


def _stretch_violate_specs() -> list[MacroSpec]:
    summ = "stretch_violate_summary.csv"
    move = "stretch_violate_movement.csv"
    specs: list[MacroSpec] = []
    for group, glabel in (("all", "All"), ("all_excl_sonnet_llama", "ExclSonnetLlama")):
        for cond, clabel in CONDITION_LABELS.items():
            w = {"model_group": group, "condition": cond}
            specs += [
                MacroSpec(_name("BelowThree", glabel, clabel), summ, w, "share_below_3", "pct1", required=False),
                MacroSpec(_name("BelowThree", glabel, clabel, "Frac"), summ, w, ("n_below_3", "n_runs"), "frac", required=False),
                MacroSpec(_name("AtThree", glabel, clabel), summ, w, "share_at_3", "pct1", required=False),
                MacroSpec(_name("MeanAdherence", glabel, clabel), summ, w, "mean_adherence", "dec2", required=False),
                MacroSpec(_name("N", glabel, clabel), summ, w, "n_runs", "int", required=False),
            ]
            if cond != "single_shot":
                specs += [
                    MacroSpec(_name("KbvAtThree", glabel, clabel), summ, w, "share_kbv_at_3", "pct1", required=False),
                    MacroSpec(_name("KbvAtThree", glabel, clabel, "Frac"), summ, w, ("n_kbv_at_3", "n_kbv_positive"), "frac", required=False),
                    MacroSpec(_name("KbvBelowThree", glabel, clabel), summ, w, "share_kbv_below_3", "pct1", required=False),
                    MacroSpec(_name("KbvPositive", glabel, clabel), summ, w, "n_kbv_positive", "int", required=False),
                ]
        w = {"model_group": group}
        specs += [
            MacroSpec(_name("BelowThree", glabel, "FisherP"), move, w, "below_3_fisher_p", "pval", required=False),
            MacroSpec(_name("AdherenceDelta", glabel), move, w, "delta_mean_adherence", "dec2", required=False),
            MacroSpec(_name("AdherenceDelta", glabel, "WilcoxonP"), move, w, "brief_paired_wilcoxon_p", "pval", required=False),
            MacroSpec(_name("AdherenceDelta", glabel, "MannWhitneyP"), move, w, "mann_whitney_p", "pval", required=False),
        ]
    for model, mlabel in MODEL_LABELS.items():
        w = {"model_group": model, "condition": "multi_turn_pressure"}
        specs += [
            MacroSpec(_name("BelowThree", mlabel, "Pressure"), summ, w, "share_below_3", "pct1", required=False),
            MacroSpec(_name("KbvAtThree", mlabel, "Pressure"), summ, w, "share_kbv_at_3", "pct1", required=False),
        ]
        w = {"model_group": model}
        specs += [
            MacroSpec(_name("AdherenceDelta", mlabel), move, w, "delta_mean_adherence", "dec2", required=False),
            MacroSpec(_name("AdherenceDelta", mlabel, "WilcoxonP"), move, w, "brief_paired_wilcoxon_p", "pval", required=False),
        ]
    return specs


def _human_specs() -> list[MacroSpec]:
    hvj = "human_vs_judge.csv"
    sev = "human_severity_by_condition.csv"
    specs: list[MacroSpec] = [
        MacroSpec(_name("HumanJudgeSharedRuns"), hvj, {"metric": "shared_runs", "aggregation": "run_level"}, "value", "int", required=False),
        MacroSpec(_name("HumanJudgeSharedItems"), hvj, {"metric": "shared_items", "aggregation": "run_level"}, "value", "int", required=False),
    ]
    for agg, alabel in (("pooled", "Pooled"), ("run_mean", "RunMean")):
        for metric, mlabel in (
            ("human_share_not_satisfied_shared", "HumanNotSatisfied"),
            ("judge_share_not_satisfied_shared", "JudgeNotSatisfied"),
            ("human_share_violated_shared", "HumanViolated"),
            ("judge_share_violated_shared", "JudgeViolated"),
            ("human_violated_judge_not_violated", "HumanViolatedJudgeNot"),
        ):
            w = {"metric": metric, "aggregation": agg}
            specs.append(MacroSpec(_name(mlabel, alabel), hvj, w, "value", "pct1", required=False))
            if agg == "pooled":
                specs.append(MacroSpec(_name(mlabel, alabel, "Frac"), hvj, w, ("numerator", "denominator"), "frac", required=False))
        w = {"metric": "human_over_judge_not_satisfied_ratio", "aggregation": agg}
        specs.append(MacroSpec(_name("HumanOverJudgeRatio", alabel), hvj, w, "value", "dec2", required=False))
    for metric, mlabel in (
        ("human_violated_judge_satisfied", "HumanViolatedJudgeSatisfied"),
        ("human_not_satisfied_judge_satisfied", "HumanNotSatisfiedJudgeSatisfied"),
        ("judge_not_satisfied_human_satisfied", "JudgeNotSatisfiedHumanSatisfied"),
        ("item_label_agreement", "ItemLabelAgreement"),
        ("item_not_satisfied_agreement", "ItemNotSatisfiedAgreement"),
    ):
        w = {"metric": metric, "aggregation": "pooled"}
        specs.append(MacroSpec(_name(mlabel), hvj, w, "value", "pct1", required=False))
        specs.append(MacroSpec(_name(mlabel, "Frac"), hvj, w, ("numerator", "denominator"), "frac", required=False))
    for cond, clabel in CONDITION_LABELS.items():
        for who, wlabel in (("human", "Human"), ("judge", "Judge")):
            w = {"metric": f"{who}_share_not_satisfied_shared[{cond}]", "aggregation": "pooled"}
            specs.append(MacroSpec(_name(wlabel, "NotSatisfiedPooled", clabel), hvj, w, "value", "pct1", required=False))
    for metric, mlabel in (
        ("human_computed_score_single_shot_mean", "HumanScoreSingleShot"),
        ("human_computed_score_pressure_mean", "HumanScorePressure"),
        ("human_computed_score_delta_pressure_minus_single_shot", "HumanScoreDelta"),
        ("human_score_from_constraint_median_single_shot_mean", "HumanMedianScoreSingleShot"),
        ("human_score_from_constraint_median_pressure_mean", "HumanMedianScorePressure"),
        ("human_score_from_constraint_median_delta_pressure_minus_single_shot", "HumanMedianScoreDelta"),
        ("judge_adherence_single_shot_mean", "JudgeScoreSingleShot"),
        ("judge_adherence_pressure_mean", "JudgeScorePressure"),
        ("judge_adherence_delta_pressure_minus_single_shot", "JudgeScoreDelta"),
    ):
        w = {"metric": metric, "aggregation": "score"}
        specs.append(MacroSpec(_name(mlabel), hvj, w, "value", "dec2", required=False))
        specs.append(MacroSpec(_name(mlabel, "N"), hvj, w, "n_runs", "int", required=False))
    for rule, rlabel in (("judge_adherence_le_2", "AdherenceLeTwo"), ("judge_violations_listed", "ViolationsListed"), ("structured_any_violation", "Structured")):
        for metric, mlabel in (
            ("run_level_sensitivity", "Sensitivity"),
            ("run_level_specificity", "Specificity"),
        ):
            w = {"metric": f"{metric}[{rule}]", "aggregation": "run_level"}
            specs.append(MacroSpec(_name("Judge", mlabel, rlabel), hvj, w, "value", "pct1", required=False))
            specs.append(MacroSpec(_name("Judge", mlabel, rlabel, "Frac"), hvj, w, ("numerator", "denominator"), "frac", required=False))
        w = {"metric": f"run_level_human_violation_runs[{rule}]", "aggregation": "run_level"}
        specs.append(MacroSpec(_name("HumanViolationRuns", rlabel, "Frac"), hvj, w, ("numerator", "denominator"), "frac", required=False))
    # Human severity by condition (B20), median-label aggregation.
    for cond, clabel in list(CONDITION_LABELS.items()) + [("all", "All")]:
        for agg, alabel in (("median_label_pooled", "Pooled"), ("median_label_run_mean", "RunMean")):
            w = {"aggregation": agg, "condition": cond}
            specs += [
                MacroSpec(_name("HumanSevNotSatisfied", alabel, clabel), sev, w, "share_not_satisfied", "pct1", required=False),
                MacroSpec(_name("HumanSevViolated", alabel, clabel), sev, w, "share_violated", "pct1", required=False),
                MacroSpec(_name("HumanSevStretched", alabel, clabel), sev, w, "share_stretched", "pct1", required=False),
            ]
        w = {"aggregation": "median_label_pooled", "condition": cond}
        specs += [
            MacroSpec(_name("HumanSevNotSatisfiedPooled", clabel, "Frac"), sev, w, ("n_not_satisfied", "n_items"), "frac", required=False),
            MacroSpec(_name("HumanSevViolatedPooled", clabel, "Frac"), sev, w, ("n_violated", "n_items"), "frac", required=False),
            MacroSpec(_name("HumanSevRunsWithViolation", clabel), sev, w, "share_runs_with_violation", "pct1", required=False),
            MacroSpec(_name("HumanSevRunsWithViolation", clabel, "Frac"), sev, w, ("n_runs_with_violation", "n_runs"), "frac", required=False),
            MacroSpec(_name("HumanSevN", clabel), sev, w, "n_runs", "int", required=False),
            MacroSpec(_name("HumanSevMeanScore", clabel), sev, w, "mean_computed_score", "dec2", required=False),
        ]
    return specs


def _tokens_specs() -> list[MacroSpec]:
    csv = "tokens_context_summary.csv"
    specs: list[MacroSpec] = [
        MacroSpec(_name("TokensRuns"), csv, {"section": "coverage", "name": "runs with usage records", "group": "all"}, "estimate", "int", required=False),
        MacroSpec(_name("TokensPressureRuns"), csv, {"section": "coverage", "name": "pressure runs with usage records", "group": "multi_turn_pressure"}, "n", "int", required=False),
        MacroSpec(_name("MaxContextShare"), csv, {"section": "context_share", "name": "max final context / window", "group": "all"}, "estimate", "pct1", required=False),
        MacroSpec(_name("MaxOutputShare"), csv, {"section": "context_share", "name": "max logged output tokens / window", "group": "all"}, "estimate", "pct1", required=False),
        MacroSpec(_name("MaxBilledShare"), csv, {"section": "context_share", "name": "max cumulative billed tokens over all calls / window (not a context size)", "group": "all"}, "estimate", "pct1", required=False),
        MacroSpec(_name("MaxContextShareSonnetTwoHundredK"), csv, {"section": "context_share", "name": "max final context / window, Sonnet at 200k (sensitivity)", "group": "all"}, "estimate", "pct1", required=False),
        MacroSpec(_name("MinHeadroom"), csv, {"section": "context_share", "name": "min headroom (window / final context)", "group": "all"}, "estimate", "dec1", required=False),
    ]
    for model, mlabel in MODEL_LABELS.items():
        specs.append(MacroSpec(_name("MaxContextShare", mlabel), csv, {"section": "context_share", "name": "max final context / window", "group": model}, "estimate", "pct1", required=False))
        specs.append(MacroSpec(_name("MeanFinalContext", mlabel), csv, {"section": "context_share", "name": "mean final context tokens", "group": model}, "estimate", "round", required=False))
    controls = (("", "Raw"), (" + log_final_words", "Words"), (" + log_output_tokens", "Output"), (" + log_total_tokens", "Total"))
    contrasts = (
        ("is_pressure", "pressure vs neutral", "multi-turn only", "PressureVsNeutral"),
        ("condition", "pressure vs single-shot", "all conditions", "PressureVsSingleShot"),
        ("is_pressure", "pressure vs all other", "all conditions", "PressureVsOther"),
    )
    for term, contrast, group, clabel in contrasts:
        for ctrl, klabel in controls:
            name = f"ols adherence ~ {term}{ctrl} + model FE: {contrast}"
            w = {"section": "verbosity_control", "name": name, "group": group}
            specs += [
                MacroSpec(_name("Verbosity", clabel, klabel), csv, w, "estimate", "dec3", required=False),
                MacroSpec(_name("Verbosity", clabel, klabel, "Ci"), csv, w, ("ci_low", "ci_high"), "cidec3", required=False),
                MacroSpec(_name("Verbosity", clabel, klabel, "P"), csv, w, "p_value", "pval", required=False),
                MacroSpec(_name("Verbosity", clabel, klabel, "N"), csv, w, "n", "int", required=False),
            ]
    for name, label in (
        ("logit kbv ~ log_total_tokens + model FE", "KbvTotalTokens"),
        ("logit kbv ~ log_output_tokens + model FE", "KbvOutputTokens"),
        ("ols adherence ~ log_total_tokens + model FE", "AdherenceTotalTokens"),
    ):
        w = {"section": "pressure_tokens", "name": name, "group": "multi_turn_pressure"}
        specs += [
            MacroSpec(_name("Tokens", label), csv, w, "estimate", "dec2", required=False),
            MacroSpec(_name("Tokens", label, "Ci"), csv, w, ("ci_low", "ci_high"), "cidec2", required=False),
            MacroSpec(_name("Tokens", label, "P"), csv, w, "p_value", "pval", required=False),
        ]
    return specs


def _onset_specs() -> list[MacroSpec]:
    csv = "auditor_onset_summary.csv"
    w = {"section": "first_flag_pressure_drift_cases"}
    specs = [
        MacroSpec(_name("OnsetCases"), csv, w, "n", "int", required=False),
        MacroSpec(_name("OnsetByTurnTwo"), csv, w, "share_by_turn_2", "pct1", required=False),
        MacroSpec(_name("OnsetByTurnTwoFrac"), csv, w, ("n_by_turn_2", "n"), "frac", required=False),
        MacroSpec(_name("OnsetByTurnThree"), csv, w, "share_by_turn_3", "pct1", required=False),
        MacroSpec(_name("OnsetByTurnThreeFrac"), csv, w, ("n_by_turn_3", "n"), "frac", required=False),
    ]
    w = {"section": "first_flag_pressure_trajectory_drift"}
    specs += [
        MacroSpec(_name("OnsetTrajectoryByTurnTwo"), csv, w, "share_by_turn_2", "pct1", required=False),
        MacroSpec(_name("OnsetTrajectoryByTurnThree"), csv, w, "share_by_turn_3", "pct1", required=False),
    ]
    for cond, clabel in CONDITION_LABELS.items():
        if cond == "single_shot":
            continue
        w = {"section": "first_flag_by_condition", "condition": cond}
        specs += [
            MacroSpec(_name("OnsetByTurnTwo", clabel), csv, w, "share_by_turn_2", "pct1", required=False),
            MacroSpec(_name("OnsetByTurnThree", clabel), csv, w, "share_by_turn_3", "pct1", required=False),
        ]
    w = {"section": "events_per_run", "condition": "multi_turn_pressure", "statistic": "n_events == 3 (share)"}
    specs.append(MacroSpec(_name("ThreeEventsPressureShare"), csv, w, "share", "pct0", required=False))
    for cls, label in (("trajectory_drift", "TrajectoryDrift"), ("trajectory_lock_in", "LockIn"), ("mild_drift", "MildDrift"), ("no_drift", "NoDrift")):
        for cond, clabel in CONDITION_LABELS.items():
            w = {"section": "classification_mix", "condition": cond, "statistic": cls}
            # auditor_onset.py omits zero-count classes, so an empty selection means 0.0.
            specs.append(MacroSpec(_name("Class", label, clabel), csv, w, "share", "pct1", required=False, default="0.0",
                                   note="0.0 when the summary has no row for this class (zero count)"))
    return specs


def _constraint_count_specs() -> list[MacroSpec]:
    csv = "constraint_count_summary.csv"
    specs = [
        MacroSpec(_name("Briefs"), csv, {"section": "coverage", "name": "briefs"}, "estimate", "int", required=False),
        MacroSpec(_name("HardConstraints"), csv, {"section": "coverage", "name": "hard constraints in total"}, "estimate", "int", required=False),
        MacroSpec(_name("HardConstraintsMin"), csv, {"section": "coverage", "name": "hard constraints per brief: min"}, "estimate", "int", required=False),
        MacroSpec(_name("HardConstraintsMax"), csv, {"section": "coverage", "name": "hard constraints per brief: max"}, "estimate", "int", required=False),
        MacroSpec(_name("BannedMovesMin"), csv, {"section": "coverage", "name": "banned moves per brief: min"}, "estimate", "int", required=False),
        MacroSpec(_name("BannedMovesMax"), csv, {"section": "coverage", "name": "banned moves per brief: max"}, "estimate", "int", required=False),
        MacroSpec(_name("PressureKbvRate"), csv, {"section": "coverage", "name": "pressure KBV rate"}, "estimate", "pct1", required=False),
    ]
    for name, label in (
        ("kbv ~ n_hard_constraints + model FE", "KbvHardConstraints"),
        ("kbv ~ n_banned_moves + model FE", "KbvBannedMoves"),
        ("kbv ~ hard_constraint_words + model FE", "KbvConstraintWords"),
        ("noncompliant ~ n_hard_constraints + model FE", "NoncompliantHardConstraints"),
    ):
        w = {"section": "logit_pressure", "name": name}
        specs += [
            MacroSpec(_name("Or", label), csv, w, "estimate", "dec2", required=False),
            MacroSpec(_name("Or", label, "Ci"), csv, w, ("ci_low", "ci_high"), "cidec2", required=False),
            MacroSpec(_name("Or", label, "P"), csv, w, "p_value", "pval", required=False),
        ]
    for name, label in (
        ("spearman adherence vs n_hard_constraints (pressure brief means)", "AdherenceHardConstraints"),
        ("spearman kbv_rate vs n_hard_constraints (pressure brief means)", "KbvHardConstraints"),
        ("spearman fidelity vs n_hard_constraints (pressure brief means)", "FidelityHardConstraints"),
        ("spearman fidelity vs brief_words (pressure brief means)", "FidelityBriefWords"),
        ("spearman adherence_change vs brief_words (pressure brief means)", "AdherenceChangeBriefWords"),
    ):
        w = {"section": "brief_level", "name": name}
        specs += [
            MacroSpec(_name("Rho", label), csv, w, "estimate", "dec2", required=False),
            MacroSpec(_name("Rho", label, "P"), csv, w, "p_value", "pval", required=False),
        ]
    w = {"section": "ols_change", "name": "fidelity ~ pressure x log_brief_words + model FE (interaction)"}
    specs += [
        MacroSpec(_name("FidelityBriefWordsInteraction"), csv, w, "estimate", "dec2", required=False),
        MacroSpec(_name("FidelityBriefWordsInteractionP"), csv, w, "p_value", "pval", required=False),
    ]
    return specs


def _typology_specs() -> list[MacroSpec]:
    specs: list[MacroSpec] = [
        MacroSpec(_name("TypologyTopFourShare"), "typology_excess.csv", {"fragile_top4": "True"}, "excess_share", "pct1", required=False, agg="sum",
                  note="share of the single-shot to pressure excess in non-satisfied checks that the four most fragile types absorb"),
        MacroSpec(_name("TypologyTopFourCheckShare"), "typology_excess.csv", {"fragile_top4": "True"}, "check_share", "pct1", required=False, agg="sum"),
        MacroSpec(_name("TypologyChecks"), "typology_by_type.csv", {"condition": "all"}, "n_checks", "int", required=False, agg="sum"),
        MacroSpec(_name("TypologyNonSatisfied"), "typology_by_type.csv", {"condition": "all"}, "n_non_satisfied", "int", required=False, agg="sum"),
        MacroSpec(_name("TypologyTypes"), "typology_by_type.csv", {"condition": "all"}, "type", "int", required=False, agg="count"),
        MacroSpec(_name("TypologyOmnibusP"), "typology_interaction.csv", {"role": "primary"}, "p_value", "pval", required=False),
        MacroSpec(_name("TypologyOmnibusStat"), "typology_interaction.csv", {"role": "primary"}, "statistic", "dec2", required=False),
        MacroSpec(_name("TypologyOmnibusDf"), "typology_interaction.csv", {"role": "primary"}, "df", "int", required=False),
        MacroSpec(_name("TypologyOmnibusClusters"), "typology_interaction.csv", {"role": "primary"}, "n_clusters", "int", required=False),
        MacroSpec(_name("TypologyOmnibusN"), "typology_interaction.csv", {"role": "primary"}, "n", "int", required=False),
        MacroSpec(_name("TypologyBriefClusterP"), "typology_interaction.csv", {"test": "wald_cluster_brief"}, "p_value", "pval", required=False),
        MacroSpec(_name("TypologyLikelihoodRatioP"), "typology_interaction.csv", {"test": "likelihood_ratio"}, "p_value", "pval", required=False),
        MacroSpec(_name("TypologyQuantVsRestP"), "typology_interaction.csv", {"test": "quant_vs_rest_split"}, "p_value", "pval", required=False),
        MacroSpec(_name("TypologyBestSplitOr"), "typology_splits.csv", {"rank": "1"}, "interaction_or", "dec2", required=False),
        MacroSpec(_name("TypologyBestSplitCi"), "typology_splits.csv", {"rank": "1"}, ("ci_low_cluster_run", "ci_high_cluster_run"), "cidec2", required=False),
        MacroSpec(_name("TypologyBestSplitP"), "typology_splits.csv", {"rank": "1"}, "p_value_cluster_run", "pval", required=False),
        MacroSpec(_name("TypologySplitsSearched"), "typology_splits.csv", {"rank": "1"}, "n_splits_searched", "int", required=False),
        MacroSpec(_name("TypologyQuantVsRestOr"), "typology_splits.csv", {"is_quant_vs_rest": "True"}, "interaction_or", "dec2", required=False),
        MacroSpec(_name("TypologyAgreementRows"), "typology_agreement.csv", {}, "n_rows_both_labelled", "int", required=False),
        MacroSpec(_name("TypologyAgreement"), "typology_agreement.csv", {}, "percent_agreement", "pct1", required=False),
        MacroSpec(_name("TypologyKappa"), "typology_agreement.csv", {}, "cohens_kappa", "dec2", required=False),
    ]
    for t, tlabel in TYPOLOGY_TYPES.items():
        w = {"condition": "all", "type": t}
        specs += [
            MacroSpec(_name("TypeRate", tlabel), "typology_by_type.csv", w, "rate", "pct1", required=False),
            MacroSpec(_name("TypeRate", tlabel, "Ci"), "typology_by_type.csv", w, ("ci_low", "ci_high"), "ci", required=False),
            MacroSpec(_name("TypeConstraints", tlabel), "typology_by_type.csv", w, "n_constraints", "int", required=False),
        ]
        w = {"condition": "multi_turn_pressure", "type": t}
        specs.append(MacroSpec(_name("TypeRate", tlabel, "Pressure"), "typology_by_type.csv", w, "rate", "pct1", required=False))
    return specs


def _mechanism_specs() -> list[MacroSpec]:
    """Self-audit mechanism (mechanism_summary.csv, regenerated 2026-10-26 09:29 labels).

    Three departure sources share one label scheme: structured-judge
    departures (no prefix, ``Mech``), auditor drift events (``auditorevt_``,
    ``MechAud``) and human median labels (``human_``, ``MechHuman``).
    """
    csv = "mechanism_summary.csv"
    specs: list[MacroSpec] = []
    quantities = (
        ("departures_with_identified_constraint", "Departures", ("num", "pct", "frac")),
        ("departures_stretched", "DeparturesStretched", ("pct", "frac")),
        ("departures_violated", "DeparturesViolated", ("pct", "frac")),
        ("departed_checklist_found", "ChecklistFound", ("pct", "frac")),
        ("departed_appears_in_final_checklist", "AppearsInChecklist", ("pct", "frac", "ci")),
        ("departed_appears_in_final_checklist_strict", "AppearsInChecklistStrict", ("pct", "frac", "ci")),
        ("departed_appears_given_checklist_found", "AppearsGivenChecklist", ("pct", "frac")),
        ("departed_appears_given_checklist_found_strict", "AppearsGivenChecklistStrict", ("pct", "frac")),
        ("departed_explicit_or_hedged_compliance", "ComplianceLanguage", ("pct", "frac", "ci")),
        ("departed_explicit_compliance_only", "ComplianceExplicit", ("pct", "frac")),
        ("departed_hedged", "Hedged", ("pct", "frac")),
        ("departed_flagged", "DepartedFlagged", ("num", "pct", "frac")),
        ("departed_unflagged_no_verdict", "UnflaggedNoVerdict", ("pct", "frac")),
        ("departed_pattern_acknowledge_then_depart", "Acknowledge", ("pct", "frac")),
        ("departed_pattern_silent", "Silent", ("pct", "frac")),
        ("departed_pattern_renegotiated", "Renegotiated", ("pct", "frac")),
        ("departed_pattern_run_acknowledge_then_depart", "AcknowledgeRun", ("pct", "frac")),
        ("departed_pattern_run_silent", "SilentRun", ("pct", "frac")),
        ("departed_pattern_run_renegotiated", "RenegotiatedRun", ("pct", "frac")),
        ("departed_silent_truncated", "SilentTruncated", ("num", "pct")),
        ("departed_silent_true_drop", "SilentTrueDrop", ("num", "pct")),
        ("departed_absent_truncated", "AbsentTruncated", ("num", "pct")),
        ("departed_acknowledged_any_turn", "AcknowledgedAnyTurn", ("pct",)),
        ("departed_mentioned_final_text", "MentionedFinalText", ("pct",)),
        ("departed_renegotiation_loose_run", "RenegotiationLooseRun", ("pct",)),
        ("departed_renegotiation_strict_final", "RenegotiationStrictFinal", ("pct",)),
        ("satisfied_appears_in_final_checklist", "SatisfiedAppears", ("pct", "frac")),
        ("satisfied_flagged", "SatisfiedFlagged", ("num", "frac")),
        ("satisfied_explicit_or_hedged_compliance", "SatisfiedCompliance", ("pct",)),
        ("violated_only_appears_in_final_checklist", "ViolatedOnlyAppears", ("pct", "frac")),
        ("violated_only_pattern_acknowledge_then_depart", "ViolatedOnlyAcknowledge", ("pct", "frac")),
        ("violated_only_flagged", "ViolatedOnlyFlagged", ("num", "frac")),
    )
    for prefix, plabel in (("", "Mech"), ("auditorevt_", "MechAud"), ("human_", "MechHuman")):
        for q, qlabel, fmts in quantities:
            w = {"quantity": prefix + q}
            if "num" in fmts:
                specs.append(MacroSpec(_name(plabel, qlabel, "N"), csv, w, "numerator", "int", required=False))
            if "pct" in fmts:
                specs.append(MacroSpec(_name(plabel, qlabel), csv, w, "rate", "pct1", required=False))
            if "frac" in fmts:
                specs.append(MacroSpec(_name(plabel, qlabel, "Frac"), csv, w, ("numerator", "denominator"), "frac", required=False))
            if "ci" in fmts:
                specs.append(MacroSpec(_name(plabel, qlabel, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False))
        for model in STRUCTURED_MODELS:
            w = {"quantity": f"{prefix}departed_appears_in_final_checklist[{model}]"}
            specs.append(MacroSpec(_name(plabel, "AppearsInChecklist", MODEL_LABELS[model]), csv, w, "rate", "pct1", required=False))
            specs.append(MacroSpec(_name(plabel, "AppearsInChecklist", MODEL_LABELS[model], "Frac"), csv, w, ("numerator", "denominator"), "frac", required=False))
    # Monitored runs: does naming a constraint at one intervention predict a re-flag at the next?
    monitor = (
        ("monitor_reflag_given_named_previous_intervention", "MonitorReflagNamed"),
        ("monitor_flag_given_not_named_previous_intervention", "MonitorFlagNotNamed"),
        ("monitor_reflag_given_named_any_earlier_intervention", "MonitorReflagNamedAny"),
        ("monitor_flag_given_never_named_earlier", "MonitorFlagNeverNamed"),
        ("monitor_named_given_named_prev_pass_all_passes", "MonitorNamedGivenNamedAllPasses"),
        ("monitor_named_given_not_named_prev_pass_all_passes", "MonitorNamedGivenNotNamedAllPasses"),
    )
    for q, qlabel in monitor:
        w = {"quantity": q}
        specs += [
            MacroSpec(_name(qlabel), csv, w, "rate", "pct1", required=False),
            MacroSpec(_name(qlabel, "Frac"), csv, w, ("numerator", "denominator"), "frac", required=False),
            MacroSpec(_name(qlabel, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False),
        ]
    for model in ("anthropic/claude-sonnet-4-6", "gemini/gemini-3.1-flash-lite-preview",
                  "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        mlabel = MODEL_LABELS[model]
        for q, qlabel in monitor[4:]:
            w = {"quantity": f"{q}[{model}]"}
            specs.append(MacroSpec(_name(qlabel, mlabel), csv, w, "rate", "pct1", required=False))
            specs.append(MacroSpec(_name(qlabel, mlabel, "Frac"), csv, w, ("numerator", "denominator"), "frac", required=False))
    return specs


def _checker_specs() -> list[MacroSpec]:
    cov = "checker_coverage.csv"
    agr = "checker_agreement.csv"
    w = {"brief_id": "TOTAL"}
    specs = [
        MacroSpec(_name("CheckableConstraints"), cov, w, "n_checkable", "int", required=False),
        MacroSpec(_name("CheckablePartial"), cov, w, "n_partial", "int", required=False),
        MacroSpec(_name("CheckableTotal"), cov, w, "n_hard_constraints", "int", required=False),
        MacroSpec(_name("CheckableShare"), cov, w, "checkable_fraction", "pct1", required=False),
        MacroSpec(_name("CheckableFrac"), cov, w, ("n_checkable", "n_hard_constraints"), "frac", required=False),
    ]
    for comparison, label in (
        ("structured_judge:value_pairs:not_satisfied=stretched+violated", "CheckerJudge"),
        ("structured_judge:value_pairs:not_satisfied=violated_only", "CheckerJudgeViolatedOnly"),
        ("structured_judge:restatement_only_pairs:not_satisfied=stretched+violated", "CheckerJudgeRestatement"),
        ("human_median:as_rated_text:value_pairs", "CheckerHuman"),
        ("human_median:transcript_text:value_pairs", "CheckerHumanTranscript"),
    ):
        w = {"comparison": comparison, "group": "overall", "level": "all"}
        specs += [
            MacroSpec(_name(label, "Agreement"), agr, w, "agreement", "pct1", required=False),
            MacroSpec(_name(label, "Kappa"), agr, w, "kappa", "dec2", required=False),
            MacroSpec(_name(label, "Pairs"), agr, w, "n_pairs", "int", required=False),
            MacroSpec(_name(label, "CheckerViolatedRate"), agr, w, "checker_violated_rate", "pct1", required=False),
            MacroSpec(_name(label, "RefNotSatisfiedRate"), agr, w, "ref_not_satisfied_rate", "pct1", required=False),
        ]
    # Human-human ceiling: per rater pair only, so the headline is the mean over pairs.
    ceiling = "human_human_ceiling:not_satisfied=stretched+violated"
    w = {"comparison": ceiling, "group": "rater_pair"}
    specs += [
        MacroSpec(_name("HumanHumanCeilingAgreement"), agr, w, "agreement", "pct1", required=False, agg="mean",
                  note="mean over rater pairs of the pairwise not-satisfied agreement"),
        MacroSpec(_name("HumanHumanCeilingKappa"), agr, w, "kappa", "dec2", required=False, agg="mean"),
        MacroSpec(_name("HumanHumanCeilingPairs"), agr, w, "n_pairs", "int", required=False, agg="sum"),
        MacroSpec(_name("HumanHumanCeilingRaterPairs"), agr, w, "level", "int", required=False, agg="count"),
    ]
    for pair in ("A-B", "A-C", "A-D", "B-C", "C-D"):
        w = {"comparison": ceiling, "group": "rater_pair", "level": pair}
        plabel = pair.replace("-", "")
        specs += [
            MacroSpec(_name("HumanHumanCeilingAgreement", plabel), agr, w, "agreement", "pct1", required=False),
            MacroSpec(_name("HumanHumanCeilingKappa", plabel), agr, w, "kappa", "dec2", required=False),
        ]
    # X12 (P9, P11): the ceiling on checkable constraints, as a range over rater pairs.
    # The min and max pairs are found from the CSV, never named here.
    w = {"comparison": ceiling, "group": "rater_pair_checkable_constraints"}
    note = "rater pairs, checkable constraints both raters labelled"
    specs += [
        MacroSpec(_name("HumanHumanCeilingCheckableKappaMin"), agr, w, "kappa", "dec2", required=False, agg="min", note=note),
        MacroSpec(_name("HumanHumanCeilingCheckableKappaMinPair"), agr, w, "level", "auto", required=False, agg="min",
                  by="kappa", note=note + "; the rater pair with the lowest kappa"),
        MacroSpec(_name("HumanHumanCeilingCheckableKappaMax"), agr, w, "kappa", "dec2", required=False, agg="max", note=note),
        MacroSpec(_name("HumanHumanCeilingCheckableKappaMaxPair"), agr, w, "level", "auto", required=False, agg="max",
                  by="kappa", note=note + "; the rater pair with the highest kappa"),
        MacroSpec(_name("HumanHumanCeilingCheckableKappa"), agr, w, "kappa", "dec2", required=False, agg="mean",
                  note=note + "; mean over rater pairs"),
        MacroSpec(_name("HumanHumanCeilingCheckableRaterPairs"), agr, w, "level", "int", required=False, agg="count", note=note),
        MacroSpec(_name("HumanHumanCeilingCheckablePairs"), agr, w, "n_pairs", "int", required=False, agg="sum",
                  note=note + "; constraint comparisons summed over rater pairs"),
    ]
    return specs


# ---------------------------------------------------------------------------
# Experiment tables (E1 crossed judge, E2, E3, E5, E6, E7)
# ---------------------------------------------------------------------------


def _crossed_judge_specs() -> list[MacroSpec]:
    """E1 crossed-judge outputs (kbv_crossed.py).

    Arms: ``opus`` and ``gpt`` are the common-judge arms, ``published`` is the
    cross-family arm of the submission. Under the opus and gpt arms the
    neutral and checkpointed rows of the aggregate table keep the published
    judge (E1 re-scored single-shot and pressure only).
    """
    specs: list[MacroSpec] = []
    csv = "kbv_common_judge.csv"
    rng = "kbv_common_judge_range.csv"
    for arm, alabel in JUDGE_ARMS.items():
        for variant, vlabel in KBV_VARIANTS.items():
            for model, mlabel in MODEL_LABELS.items():
                w = {"judge_arm": arm, "variant": variant, "model_id": model, "condition": "multi_turn_pressure"}
                base = "Kbv" + alabel + "ArmPressure" + mlabel + vlabel
                specs += [
                    MacroSpec(_name(base), csv, w, "kbv_rate", "pct1", required=False),
                    MacroSpec(_name(base, "Frac"), csv, w, ("kbv_count", "n_with_probe"), "frac", required=False),
                    MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False),
                    MacroSpec(_name(base, "N"), csv, w, "n_with_probe", "int", required=False),
                ]
                if vlabel:
                    specs.append(MacroSpec(_name(base, "Imputed"), csv, w, "n_truncated_waived", "int", required=False,
                                           note="truncated final probes whose recall is waived (counted as knows)"))
            w = {"judge_arm": arm, "variant": variant}
            base = "Kbv" + alabel + "Arm" + vlabel
            specs += [
                MacroSpec(_name(base, "Min"), rng, w, "min_kbv_rate", "pct1", required=False),
                MacroSpec(_name(base, "MinModel"), rng, w, "argmin_model", "auto", required=False),
                MacroSpec(_name(base, "Max"), rng, w, "max_kbv_rate", "pct1", required=False),
                MacroSpec(_name(base, "MaxModel"), rng, w, "argmax_model", "auto", required=False),
                MacroSpec(_name(base, "AboveHalf"), rng, w, "n_above_cut", "int", required=False),
                MacroSpec(_name(base, "AboveHalfFrac"), rng, w, ("n_above_cut", "n_models"), "frac", required=False),
                MacroSpec(_name(base, "AboveHalfSet"), rng, w, "models_above_cut", "auto", required=False,
                          note="semicolon-separated model ids above the 50% cut"),
                MacroSpec(_name(base, "Models"), rng, w, "n_models", "int", required=False),
            ]
    # X14: providers among the Opus-arm models above 50% (same row as \crKbvOpusArmAboveHalfFrac).
    specs.append(MacroSpec(_name("KbvOpusArmAboveHalfProviders"), rng, {"judge_arm": "opus", "variant": "clean"},
                           "models_above_cut", "nproviders", required=False,
                           note="distinct providers (MODEL_PROVIDERS) among the models above the 50% cut"))
    # Judge offset (Opus minus GPT-5.4 on the same transcript) and its decomposition.
    jo = "judge_offset.csv"
    for dim, dlabel in DIMENSIONS.items():
        w = {"block": "offset_overall", "dimension": dim, "model_id": "all", "condition": "ss+mtp"}
        specs += [
            MacroSpec(_name("JudgeOffset", dlabel), jo, w, "estimate", "dec3", required=False),
            MacroSpec(_name("JudgeOffset", dlabel, "Ci"), jo, w, ("ci_low", "ci_high"), "cidec3", required=False),
            MacroSpec(_name("JudgeOffset", dlabel, "N"), jo, w, "n_runs", "int", required=False),
        ]
        for cond, clabel in E2_CONDITIONS.items():
            w = {"block": "offset_overall_by_condition", "dimension": dim, "condition": cond}
            specs += [
                MacroSpec(_name("JudgeOffset", dlabel, clabel), jo, w, "estimate", "dec3", required=False),
                MacroSpec(_name("JudgeOffset", dlabel, clabel, "Ci"), jo, w, ("ci_low", "ci_high"), "cidec3", required=False),
                MacroSpec(_name("JudgeOffset", dlabel, clabel, "N"), jo, w, "n_runs", "int", required=False),
            ]
        for model, mlabel in MODEL_LABELS.items():
            w = {"block": "offset_by_model", "dimension": dim, "model_id": model}
            specs += [
                MacroSpec(_name("JudgeOffset", dlabel, mlabel), jo, w, "estimate", "dec2", required=False),
                MacroSpec(_name("JudgeOffset", dlabel, mlabel, "Ci"), jo, w, ("ci_low", "ci_high"), "cidec2", required=False),
                MacroSpec(_name("JudgeOffset", dlabel, mlabel, "N"), jo, w, "n_runs", "int", required=False),
            ]
            w = {"block": "delta_judge_difference", "dimension": dim, "model_id": model}
            specs.append(MacroSpec(_name("DeltaJudgeDiff", dlabel, mlabel), jo, w, "estimate", "dec2", required=False,
                                   note="pressure-minus-single-shot delta under Opus minus the same delta under GPT-5.4"))
        w = {"block": "anthropic_excess", "dimension": dim}
        specs += [
            MacroSpec(_name("AnthropicExcess", dlabel), jo, w, "estimate", "dec2", required=False,
                      note="Sonnet's per-subject offset minus the mean of the other six"),
            MacroSpec(_name("AnthropicExcess", dlabel, "Ci"), jo, w, ("ci_low", "ci_high"), "cidec2", required=False),
        ]
        w = {"block": "interaction_judge_x_condition", "dimension": dim}
        specs += [
            MacroSpec(_name("JudgeInteraction", dlabel, "Coef"), jo, w, "coef", "dec3", required=False),
            MacroSpec(_name("JudgeInteraction", dlabel, "Se"), jo, w, "se", "dec3", required=False),
            MacroSpec(_name("JudgeInteraction", dlabel, "P"), jo, w, "estimate", "pval", required=False,
                      note="p of judge x condition in OLS with model FE, brief-clustered SE"),
            MacroSpec(_name("JudgeInteraction", dlabel, "N"), jo, w, "n_runs", "int", required=False),
        ]
        w = {"block": "delta_max_abs_judge_difference", "dimension": dim}
        specs += [
            MacroSpec(_name("DeltaJudgeDiffMax", dlabel), jo, w, "estimate", "dec2", required=False),
            MacroSpec(_name("DeltaJudgeDiffMax", dlabel, "Model"), jo, w, "model_id", "auto", required=False),
        ]
    for judge, jlabel in (("opus", "OpusArm"), ("gpt", "GptArm"), ("both_judges_pooled", "Pooled")):
        w = {"block": "pressure_share_in_cells_3_4", "judge": judge}
        specs.append(MacroSpec(_name("PressureCellsThreeFour", jlabel), jo, w, "estimate", "pct1", required=False,
                               note="share of pressure adherence scores in {3, 4}"))
    # Aggregate table per arm (Table 2 and Table 3 under each judge arm).
    ag = "common_judge_aggregate.csv"
    for arm, alabel in JUDGE_ARMS.items():
        models = [("all", "All")] + ([] if arm == "published" else list(MODEL_LABELS.items()))
        for model, mlabel in models:
            for cond, clabel in CONDITION_LABELS.items():
                w = {"table_arm": arm, "row_type": "condition_mean", "model_id": model, "condition": cond}
                specs.append(MacroSpec(_name("N", mlabel, clabel, alabel, "Arm"), ag, w, "n", "int", required=False))
                for dim, dlabel in DIMENSIONS.items():
                    specs += [
                        MacroSpec(_name("Mean", dlabel, mlabel, clabel, alabel, "Arm"), ag, w, f"{dim}_mean", "dec2", required=False),
                        MacroSpec(_name("Se", dlabel, mlabel, clabel, alabel, "Arm"), ag, w, f"{dim}_se", "dec2", required=False),
                    ]
            w = {"table_arm": arm, "row_type": "delta_pressure_minus_single_shot", "model_id": model}
            for dim, dlabel in DIMENSIONS.items():
                specs += [
                    MacroSpec(_name("Delta", dlabel, mlabel, alabel, "Arm"), ag, w, f"{dim}_mean", "dec2", required=False,
                              note="pressure mean minus single-shot mean under this arm"),
                    MacroSpec(_name("Delta", dlabel, mlabel, alabel, "Arm", "Se"), ag, w, f"{dim}_se", "dec2", required=False),
                ]
    # Auditor agreement per arm.
    aa = "auditor_agreement_by_arm.csv"
    for arm, alabel in JUDGE_ARMS.items():
        for definition, dlabel in (("kbv_vs_auditor", "Kbv"), ("noncompliant_vs_auditor", "Noncompliant")):
            w = {"judge_arm": arm, "definition": definition, "condition": "multi_turn_pressure"}
            base = "AuditorAgree" + alabel + "Arm" + dlabel
            specs += [
                MacroSpec(_name(base), aa, w, "percent_agreement", "pct1", required=False),
                MacroSpec(_name(base, "Rho"), aa, w, "spearman_rho", "dec2", required=False),
                MacroSpec(_name(base, "RhoP"), aa, w, "spearman_p", "pval", required=False),
                MacroSpec(_name(base, "N"), aa, w, "n", "int", required=False),
                MacroSpec(_name(base, "JudgeRate"), aa, w, "judge_flag_rate", "pct1", required=False),
                MacroSpec(_name(base, "AuditorRate"), aa, w, "auditor_drift_rate", "pct1", required=False),
                MacroSpec(_name(base, "BothPositive"), aa, w, "both_positive", "int", required=False),
                MacroSpec(_name(base, "BothNegative"), aa, w, "both_negative", "int", required=False),
                MacroSpec(_name(base, "JudgeOnly"), aa, w, "judge_only", "int", required=False),
                MacroSpec(_name(base, "AuditorOnly"), aa, w, "auditor_only", "int", required=False),
            ]
    # Below-3 shares and anchor-3 KBV shares per arm.
    sv = "stretch_violate_by_judge_arm.csv"
    for arm, alabel in JUDGE_ARMS.items():
        for model, mlabel in [("all", "All")] + list(MODEL_LABELS.items()):
            w = {"judge_arm": arm, "model_id": model}
            base = alabel + "Arm" + mlabel
            specs += [
                MacroSpec(_name("BelowThree", base, "SingleShot"), sv, w, "share_below3_ss", "pct1", required=False),
                MacroSpec(_name("BelowThree", base, "SingleShotFrac"), sv, w, ("below3_ss", "n_ss"), "frac", required=False),
                MacroSpec(_name("BelowThree", base, "Pressure"), sv, w, "share_below3_mtp", "pct1", required=False),
                MacroSpec(_name("BelowThree", base, "PressureFrac"), sv, w, ("below3_mtp", "n_mtp"), "frac", required=False),
                MacroSpec(_name("KbvAtThree", base), sv, w, "share_kbv_at3", "pct1", required=False),
                MacroSpec(_name("KbvAtThree", base, "Frac"), sv, w, ("kbv_at3", "n_kbv_positive_mtp"), "frac", required=False),
                MacroSpec(_name("KbvBelowThree", base), sv, w, "share_kbv_below3", "pct1", required=False),
                MacroSpec(_name("KbvPositive", base), sv, w, "n_kbv_positive_mtp", "int", required=False),
            ]
    # Judge drift control: April scores re-run today on the same transcripts.
    dc = "judge_drift_control.csv"
    for judge, jlabel in (("gpt", "Gpt"), ("opus", "Opus"), ("both", "Both")):
        for dim, dlabel in DIMENSIONS.items():
            w = {"control_judge": judge, "dimension": dim}
            base = "DriftControl" + jlabel + dlabel
            specs += [
                MacroSpec(_name(base, "Exact"), dc, w, "exact_agreement", "pct1", required=False),
                MacroSpec(_name(base, "Diff"), dc, w, "mean_diff_today_minus_april", "dec2", required=False),
                MacroSpec(_name(base, "WithinOne"), dc, w, "within_1", "pct1", required=False),
                MacroSpec(_name(base, "Rho"), dc, w, "spearman_rho", "dec2", required=False),
                MacroSpec(_name(base, "N"), dc, w, "n", "int", required=False),
                MacroSpec(_name(base, "MeanApril"), dc, w, "mean_april", "dec2", required=False),
                MacroSpec(_name(base, "MeanToday"), dc, w, "mean_today", "dec2", required=False),
            ]
    return specs


def _e7_strict_probe_specs() -> list[MacroSpec]:
    """E7 strict probe re-grading (e7_strict_probe.py): April, paraphrase-tolerant (equiv) and strict rules."""
    acc = "e7_strict_probe_accuracy.csv"
    runs = "e7_strict_probe_runs.csv"
    specs: list[MacroSpec] = []
    columns = (
        ("mean_april_accuracy", "AprilRecall", "pct1"),
        ("mean_april_equiv_regrade", "EquivRecall", "pct1"),
        ("mean_strict_regrade", "StrictRecall", "pct1"),
        ("mean_lenient_regrade", "LenientRecall", "pct1"),
        ("mean_strict_constraint_recall", "StrictConstraintRecall", "pct1"),
        ("mean_strict_banned_recall", "StrictBannedRecall", "pct1"),
        ("share_ge_0_8_april", "AprilAboveCut", "pct1"),
        ("share_ge_0_8_april_equiv", "EquivAboveCut", "pct1"),
        ("share_ge_0_8_strict", "StrictAboveCut", "pct1"),
        ("n_side_change_april_equiv_vs_april", "SideChangeEquiv", "int"),
        ("n_side_change_strict_vs_april", "SideChangeStrict", "int"),
        ("share_objective_exact", "ObjectiveExact", "pct1"),
        ("n_violating", "Violating", "int"),
        ("kbv_april", "KbvApril", "int"),
        ("kbv_april_equiv", "KbvEquiv", "int"),
        ("kbv_strict", "KbvStrict", "int"),
        ("kbv_rate_april", "KbvAprilRate", "pct1"),
        ("kbv_rate_april_equiv", "KbvEquivRate", "pct1"),
        ("kbv_rate_strict", "KbvStrictRate", "pct1"),
        ("n_selected", "N", "int"),
        ("n_truncated_probes", "Truncated", "int"),
        ("total_cost_usd", "Cost", "dec2"),
    )
    for model, mlabel in E7_MODELS.items():
        w = {"model_id": model}
        for col, label, fmt in columns:
            specs.append(MacroSpec(_name("ESeven", label, mlabel), acc, w, col, fmt, required=False))
        for col, label in (("kbv_april", "KbvAprilFrac"), ("kbv_april_equiv", "KbvEquivFrac"), ("kbv_strict", "KbvStrictFrac")):
            specs.append(MacroSpec(_name("ESeven", label, mlabel), acc, w, (col, "n_with_adherence"), "frac", required=False))
    specs.append(MacroSpec(_name("ESevenGrader"), acc, {"model_id": "all"}, "grader_models_reported", "auto", required=False))
    specs += [
        MacroSpec(_name("ESevenRuns"), runs, {}, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("ESevenRunsTruncated"), runs, {"probe_truncated": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESevenRunsKbvApril"), runs, {"kbv_april": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESevenRunsKbvStrict"), runs, {"kbv_strict": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESevenRunsParseFailed"), runs, {"parse_ok": "False"}, "run_id", "int", required=False, agg="count", default="0"),
    ]
    return specs


def _reprobe_specs() -> list[MacroSpec]:
    """E3 Gemini Pro probe replay at a 4,096-token cap (reprobe.py)."""
    summ = "reprobe_summary.csv"
    acc = "reprobe_accuracy.csv"
    pro = "gemini/gemini-3.1-pro-preview"
    specs: list[MacroSpec] = []
    columns = (
        ("n", "Replays", "int"),
        ("mean_original_accuracy", "OriginalRecall", "pct1"),
        ("mean_new_accuracy", "ReplayRecall", "pct1"),
        ("mean_accuracy_delta", "RecallGain", "pct1"),
        ("share_original_truncated", "OriginalCapHits", "pct1"),
        ("share_new_finish_length", "ReplayCapHits", "pct1"),
        ("share_original_ge_0_8", "OriginalAboveCut", "pct1"),
        ("share_new_ge_0_8", "ReplayAboveCut", "pct1"),
        ("mean_original_tokens", "OriginalTokens", "round"),
        ("mean_new_tokens", "ReplayTokens", "round"),
    )
    for cond in MULTI_TURN:
        clabel = CONDITION_LABELS[cond]
        w = {"model_id": pro, "condition": cond}
        for col, label, fmt in columns:
            specs.append(MacroSpec(_name("EThree", label, clabel), summ, w, col, fmt, required=False))
    # Pooled over the three multi-turn conditions, from the per-run file.
    base = {"model_id": pro}
    specs += [
        MacroSpec(_name("EThreeReplays"), acc, base, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("EThreeOriginalCapHits"), acc, {**base, "original_truncated": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("EThreeReplayCapHits"), acc, {**base, "new_finish_reason": "length"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("EThreeOriginalRecall"), acc, base, "original_overall_accuracy", "pct1", required=False, agg="mean"),
        MacroSpec(_name("EThreeReplayRecall"), acc, base, "new_overall_accuracy", "pct1", required=False, agg="mean"),
        MacroSpec(_name("EThreeReplayCap"), acc, base, "probe_max_tokens", "int", required=False, agg="mean",
                  note="probe max_tokens of the replay; one value for every row"),
        MacroSpec(_name("EThreeCost"), acc, base, "cost_usd", "dec2", required=False, agg="sum"),
    ]
    for cond in MULTI_TURN:
        clabel = CONDITION_LABELS[cond]
        specs += [
            MacroSpec(_name("EThreeOriginalCapHits", clabel, "N"), acc, {**base, "condition": cond, "original_truncated": "True"},
                      "run_id", "int", required=False, agg="count", default="0"),
            MacroSpec(_name("EThreeReplayCapHits", clabel, "N"), acc, {**base, "condition": cond, "new_finish_reason": "length"},
                      "run_id", "int", required=False, agg="count", default="0"),
        ]
    return specs


def _e2_uncapped_specs() -> list[MacroSpec]:
    """E2 Sonnet without the 4,096-token cap (e2_uncapped.py): cells, deltas, offset."""
    cells = "e2_uncapped_cells.csv"
    deltas = "e2_uncapped_deltas.csv"
    off = "e2_uncapped_offset.csv"
    specs: list[MacroSpec] = []
    token_columns = (
        ("n_runs", "N", "int"),
        ("n_briefs", "Briefs", "int"),
        ("max_tokens", "MaxTokens", "int"),
        ("cap_turns", "CapTurns", "int"),
        ("cap_turn_share", "CapTurnShare", "pct1"),
        ("runs_any_cap", "RunsAnyCap", "int"),
        ("runs_any_cap_share", "RunsAnyCapShare", "pct1"),
        ("final_cap", "FinalCap", "int"),
        ("final_cap_share", "FinalCapShare", "pct1"),
        ("final_tokens_mean", "FinalTokens", "round"),
        ("output_tokens_per_run_mean", "OutputTokens", "round"),
        ("input_tokens_per_run_mean", "InputTokens", "round"),
        ("turn_tokens_mean", "TurnTokens", "round"),
        ("turn_tokens_median", "TurnTokensMedian", "round"),
        ("turn_tokens_max", "TurnTokensMax", "round"),
        ("per_run_max_median", "RunMaxMedian", "round"),
        ("cost_usd_per_run_mean", "CostPerRun", "dec2"),
    )
    for source, slabel in E2_SOURCES.items():
        for cond, clabel in E2_CONDITIONS.items():
            # Token statistics do not depend on the judge arm; the opus row selects one row.
            w = {"source": source, "condition": cond, "judge_arm": "opus"}
            for col, label, fmt in token_columns:
                specs.append(MacroSpec(_name("ETwo", label, slabel, clabel), cells, w, col, fmt, required=False))
            specs += [
                MacroSpec(_name("ETwoFinalCapFrac", slabel, clabel), cells, w, ("final_cap", "n_runs"), "frac", required=False),
                MacroSpec(_name("ETwoRunsAnyCapFrac", slabel, clabel), cells, w, ("runs_any_cap", "n_runs"), "frac", required=False),
            ]
            for arm, alabel in (("opus", "Opus"), ("gpt", "Gpt")):
                w = {"source": source, "condition": cond, "judge_arm": arm}
                suffix = slabel + clabel + alabel + "Arm"
                for dim, dlabel in DIMENSIONS.items():
                    specs += [
                        MacroSpec(_name("ETwoMean", dlabel, suffix), cells, w, f"{dim}_mean", "dec2", required=False),
                        MacroSpec(_name("ETwoMean", dlabel, suffix, "Ci"), cells, w, (f"{dim}_ci_low", f"{dim}_ci_high"), "cidec2", required=False),
                        MacroSpec(_name("ETwoSd", dlabel, suffix), cells, w, f"{dim}_sd", "dec2", required=False),
                    ]
                specs += [
                    MacroSpec(_name("ETwoAdherenceLtFour", suffix), cells, w, "adherence_lt4", "int", required=False),
                    MacroSpec(_name("ETwoAdherenceLtThree", suffix), cells, w, "adherence_lt3", "int", required=False),
                    MacroSpec(_name("ETwoAdherenceEqFour", suffix), cells, w, "adherence_eq4", "int", required=False),
                    MacroSpec(_name("ETwoJudgeDate", suffix), cells, w, "judge_date_utc", "auto", required=False),
                ]
                if cond == "multi_turn_pressure":
                    base = "ETwoKbv" + slabel + alabel + "Arm"
                    specs += [
                        MacroSpec(_name(base), cells, w, "kbv_rate", "pct1", required=False),
                        MacroSpec(_name(base, "Frac"), cells, w, ("kbv_count", "n_with_final_probe"), "frac", required=False),
                        MacroSpec(_name(base, "Ci"), cells, w, ("kbv_ci_low", "kbv_ci_high"), "ci", required=False),
                        MacroSpec(_name(base, "Knows"), cells, w, "knows", "int", required=False),
                    ]
    for cond, clabel in E2_CONDITIONS.items():
        for arm, alabel in (("opus", "Opus"), ("gpt", "Gpt")):
            for metric, (mlabel, fmt, cifmt) in E2_METRICS.items():
                if metric == "kbv_rate" and cond != "multi_turn_pressure":
                    continue
                for design, glabel in E2_DESIGNS.items():
                    w = {"condition": cond, "judge_arm": arm, "metric": metric, "design": design}
                    base = "ETwoDelta" + mlabel + ("" if metric == "kbv_rate" else clabel) + alabel + "Arm" + glabel
                    specs.append(MacroSpec(_name(base), deltas, w, "delta", fmt, required=False,
                                           note="E2 uncapped minus April clean; design " + design))
                    if cifmt:
                        specs.append(MacroSpec(_name(base, "Ci"), deltas, w, ("ci_low", "ci_high"), cifmt, required=False))
                    else:
                        specs += [
                            MacroSpec(_name(base, "CiLow"), deltas, w, "ci_low", fmt, required=False),
                            MacroSpec(_name(base, "CiHigh"), deltas, w, "ci_high", fmt, required=False),
                        ]
                    specs += [
                        MacroSpec(_name(base, "P"), deltas, w, "p_value", "pval", required=False),
                        MacroSpec(_name(base, "Excludes"), deltas, w, "ci_excludes_zero", "auto", required=False,
                                  note="True when the 95% CI excludes zero"),
                        MacroSpec(_name(base, "NUncapped"), deltas, w, "n_e2", "int", required=False),
                        MacroSpec(_name(base, "NApril"), deltas, w, "n_april", "int", required=False),
                    ]
                    if design == "brief_paired":
                        specs += [
                            MacroSpec(_name(base, "BriefsUp"), deltas, w, "briefs_up", "int", required=False),
                            MacroSpec(_name(base, "BriefsDown"), deltas, w, "briefs_down", "int", required=False),
                            MacroSpec(_name(base, "BriefsTie"), deltas, w, "briefs_tie", "int", required=False),
                        ]
    for scope, slabel in (("overall", "Overall"), ("single_shot", "SingleShot"), ("multi_turn_pressure", "Pressure")):
        for dim, dlabel in DIMENSIONS.items():
            w = {"scope": scope, "dimension": dim}
            base = "ETwoOffset" + dlabel + slabel
            specs += [
                MacroSpec(_name(base), off, w, "e2_offset", "dec2", required=False, note="Opus minus GPT-5.4 on the E2 runs"),
                MacroSpec(_name(base, "Ci"), off, w, ("e2_ci_low", "e2_ci_high"), "cidec2", required=False),
                MacroSpec(_name(base, "N"), off, w, "n_runs", "int", required=False),
            ]
            if scope == "overall":
                specs += [
                    MacroSpec(_name(base, "EOne"), off, w, "e1_offset", "dec2", required=False, note="E1 comparator: Sonnet offset_by_model, ss+mtp"),
                    MacroSpec(_name(base, "EOneCi"), off, w, ("e1_ci_low", "e1_ci_high"), "cidec2", required=False),
                    MacroSpec(_name(base, "EOneN"), off, w, "e1_n_runs", "int", required=False),
                ]
    return specs


def _e5_per_turn_specs() -> list[MacroSpec]:
    """E5 per-turn Opus re-scoring (per_turn.py): cell means per turn and the turn contrasts.

    Delta macros carry no ``Delta`` word to stay under 60 characters: the
    contrast token (``TwoOne`` = turn 2 minus turn 1, ``SixAprilFive`` =
    published turn 6 minus re-scored turn 5) marks them.
    """
    cells = "e5_per_turn_cells.csv"
    deltas = "e5_per_turn_deltas.csv"
    specs: list[MacroSpec] = []
    for model, mlabel in E5_MODELS.items():
        if model == "pooled":
            continue
        for cond, clabel in E5_CONDITIONS.items():
            if cond == "pooled":
                continue
            for turn in range(1, 7):
                w = {"model_id": model, "condition": cond, "turn": str(turn)}
                suffix = mlabel + clabel + "Turn" + number_words(str(turn))
                specs.append(MacroSpec(_name("EFiveN", suffix), cells, w, "n", "int", required=False))
                for dim, dlabel in DIMENSIONS.items():
                    specs += [
                        MacroSpec(_name("EFiveMean", dlabel, suffix), cells, w, f"mean_{dim}", "dec2", required=False),
                        MacroSpec(_name("EFiveMean", dlabel, suffix, "Ci"), cells, w, (f"ci_lo_{dim}", f"ci_hi_{dim}"), "cidec2", required=False),
                    ]
                specs += [
                    MacroSpec(_name("EFiveBelowFour", suffix), cells, w, "share_ca_lt4", "pct1", required=False),
                    MacroSpec(_name("EFiveBelowFour", suffix, "Ci"), cells, w, ("share_ca_lt4_ci_lo", "share_ca_lt4_ci_hi"), "ci", required=False),
                    MacroSpec(_name("EFiveKbv", suffix), cells, w, "kbv_rate", "pct1", required=False),
                    MacroSpec(_name("EFiveKbv", suffix, "Ci"), cells, w, ("kbv_ci_lo", "kbv_ci_hi"), "ci", required=False),
                    MacroSpec(_name("EFiveRecall", suffix), cells, w, "probe_recall_mean", "pct1", required=False),
                ]
    for model, mlabel in E5_MODELS.items():
        for cond, clabel in E5_CONDITIONS.items():
            if cond == "pooled" and model != "pooled":
                continue
            for dim, dlabel in DIMENSIONS.items():
                for contrast, klabel in E5_CONTRASTS.items():
                    w = {"model_id": model, "condition": cond, "dimension": dim, "contrast": contrast}
                    base = "EFive" + dlabel + mlabel + clabel + klabel
                    specs += [
                        MacroSpec(_name(base), deltas, w, "mean_delta", "dec2", required=False, note="turn contrast " + contrast),
                        MacroSpec(_name(base, "Exact"), deltas, w, "mean_delta", "dec3", required=False),
                        MacroSpec(_name(base, "Ci"), deltas, w, ("ci_lo", "ci_hi"), "cidec2", required=False),
                        MacroSpec(_name(base, "P"), deltas, w, "p_paired_t", "pval", required=False),
                        MacroSpec(_name(base, "WilcoxonP"), deltas, w, "p_wilcoxon", "pval", required=False),
                        MacroSpec(_name(base, "N"), deltas, w, "n", "int", required=False),
                    ]
                    if contrast == "t2-t1":
                        specs += [
                            MacroSpec(_name(base, "StepShare"), deltas, w, "step_share", "dec2", required=False,
                                      note="(t2 - t1) / (t5 - t1); only meaningful when the t5 - t1 CI excludes zero"),
                            MacroSpec(_name(base, "StepExcludesZero"), deltas, w, "step_share_denominator_ci_excludes_zero", "auto", required=False),
                        ]
    return specs


def _e6_swe_specs() -> list[MacroSpec]:
    """E6 extended SWE pilot (e6_swe.py): cells, run counts, exec arm, harness, costs."""
    cells = "e6_swe_cells.csv"
    runs = "e6_swe_runs.csv"
    exec_fid = "e6_swe_exec_fidelity.csv"
    verify = "swe_verify_results.csv"
    costs = "e6_swe_costs.csv"
    pro = "gemini/gemini-3.1-pro-preview"
    specs: list[MacroSpec] = []
    for arm, alabel in SWE_ARMS.items():
        for model, mlabel in SWE_MODELS.items():
            if arm == "exec" and model not in SWE_EXEC_MODELS:
                continue
            prefix = "ESix" + alabel
            for cond, clabel in E2_CONDITIONS.items():
                w = {"arm": arm, "model_id": model, "condition": cond}
                suffix = mlabel + clabel
                specs += [
                    MacroSpec(_name(prefix, "N", suffix), cells, w, "n", "int", required=False),
                    MacroSpec(_name(prefix, "Briefs", suffix), cells, w, "n_briefs", "int", required=False),
                    MacroSpec(_name(prefix, "Noncompliant", suffix), cells, w, "noncompliance_rate", "pct1", required=False),
                    MacroSpec(_name(prefix, "Noncompliant", suffix, "Frac"), cells, w, ("n_noncompliant", "n"), "frac", required=False),
                ]
                for dim, dlabel in DIMENSIONS.items():
                    specs.append(MacroSpec(_name(prefix, "Mean", dlabel, suffix), cells, w, f"mean_{dim}", "dec2", required=False))
                if cond == "multi_turn_pressure":
                    specs += [
                        MacroSpec(_name(prefix, "Kbv", mlabel), cells, w, "kbv_rate", "pct1", required=False),
                        MacroSpec(_name(prefix, "Kbv", mlabel, "Frac"), cells, w, ("n_kbv", "n_with_probe"), "frac", required=False),
                        MacroSpec(_name(prefix, "Kbv", mlabel, "Ci"), cells, w, ("kbv_ci_lo", "kbv_ci_hi"), "ci", required=False),
                        MacroSpec(_name(prefix, "KbvCapped", mlabel), cells, w, "kbv_rate_capped_as_graded", "pct1", required=False,
                                  note="capped probe text graded as-is, no replay"),
                        MacroSpec(_name(prefix, "KbvCapped", mlabel, "Frac"), cells, w, ("n_kbv_capped_as_graded", "n_with_probe"), "frac", required=False),
                        MacroSpec(_name(prefix, "Recall", mlabel), cells, w, "mean_final_probe_accuracy", "pct1", required=False),
                        MacroSpec(_name(prefix, "OriginalRecall", mlabel), cells, w, "mean_original_probe_accuracy", "pct1", required=False),
                        MacroSpec(_name(prefix, "Knows", mlabel), cells, w, "knows_rate", "pct1", required=False),
                        MacroSpec(_name(prefix, "ProbeReplays", mlabel), cells, w, "n_probe_replay", "int", required=False),
                    ]
    # Run-level counts (cap hits and the adherence histogram).
    pro_ss = {"arm": "main", "model_id": pro, "condition": "single_shot"}
    specs += [
        MacroSpec(_name("ESixRuns"), runs, {}, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("ESixMainRuns"), runs, {"arm": "main"}, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("ESixExecRuns"), runs, {"arm": "exec"}, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("ESixCapHits"), runs, {"final_finish_reason": "length"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixMainCapHits"), runs, {"arm": "main", "final_finish_reason": "length"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixProSingleShotCapHits"), runs, {**pro_ss, "final_finish_reason": "length"}, "run_id", "int", required=False, agg="count", default="0",
                  note="main-arm Pro single-shot finals with finish_reason length"),
        MacroSpec(_name("ESixProSingleShotCapNoncompliant"), runs, {**pro_ss, "final_finish_reason": "length", "violates": "True"}, "run_id", "int",
                  required=False, agg="count", default="0", note="of those cap hits, runs the judge scored below 4"),
        MacroSpec(_name("ESixProSingleShotNoncompliant"), runs, {**pro_ss, "violates": "True"}, "run_id", "int", required=False, agg="count", default="0"),
    ]
    for score in ("0", "1", "2", "3", "4"):
        specs.append(MacroSpec(_name("ESixRunsAdherence", number_words(score)), runs, {"j_constraint_adherence": score}, "run_id", "int",
                               required=False, agg="count", default="0", note="runs over both arms with this adherence score"))
    # X13: a KBV that counts only adherence scores below 3 (pressure, both arms), and the highest
    # single-shot complexity inflation (both arms; 0 is the scale minimum, so a maximum of 0 means every run is 0).
    specs += [
        MacroSpec(_name("ESixKbvBelowThree"), runs,
                  {"condition": "multi_turn_pressure", "knows": "True", "j_constraint_adherence": ("0", "1", "2")},
                  "run_id", "int", required=False, agg="count", default="0",
                  note="pressure runs over both arms that pass the knows filter and score below 3 on adherence"),
        MacroSpec(_name("ESixMaxComplexitySingleShot"), runs, {"condition": "single_shot"}, "j_complexity_inflation", "int",
                  required=False, agg="max", note="highest complexity inflation of any single-shot run over both arms"),
    ]
    # Exec arm: contract-brief vs plain-brief scores on the same transcripts.
    specs.append(MacroSpec(_name("ESixExecCompared"), exec_fid, {}, "run_id", "int", required=False, agg="count"))
    for dim, dlabel in DIMENSIONS.items():
        specs.append(MacroSpec(_name("ESixExecAgree", dlabel), exec_fid, {f"delta_{dim}": "0"}, "run_id", "int", required=False, agg="count", default="0",
                               note="exec runs where the contract and plain briefs give the same score"))
        specs += [
            MacroSpec(_name("ESixExecContract", dlabel), exec_fid, {}, f"contract_{dim}", "dec2", required=False, agg="mean"),
            MacroSpec(_name("ESixExecPlain", dlabel), exec_fid, {}, f"plain_{dim}", "dec2", required=False, agg="mean"),
        ]
        for model in SWE_EXEC_MODELS:
            mlabel = SWE_MODELS[model]
            for cond, clabel in E2_CONDITIONS.items():
                w = {"model_id": model, "condition": cond}
                specs += [
                    MacroSpec(_name("ESixExecContract", dlabel, mlabel, clabel), exec_fid, w, f"contract_{dim}", "dec2", required=False, agg="mean"),
                    MacroSpec(_name("ESixExecPlain", dlabel, mlabel, clabel), exec_fid, w, f"plain_{dim}", "dec2", required=False, agg="mean"),
                ]
    # Harness results.
    specs += [
        MacroSpec(_name("ESixHarnessRuns"), verify, {}, "run_id", "int", required=False, agg="count"),
        MacroSpec(_name("ESixHarnessAllPassed"), verify, {"all_passed": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixHarnessTests"), verify, {}, "total", "int", required=False, agg="sum"),
        MacroSpec(_name("ESixHarnessPassed"), verify, {}, "passed", "int", required=False, agg="sum"),
        MacroSpec(_name("ESixHarnessFailed"), verify, {}, "failed", "int", required=False, agg="sum"),
        MacroSpec(_name("ESixHarnessErrors"), verify, {}, "errors", "int", required=False, agg="sum"),
        MacroSpec(_name("ESixHarnessTruncated"), verify, {"truncated": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixHarnessRefused"), verify, {"refused": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixHarnessNoCode"), verify, {"no_code": "True"}, "run_id", "int", required=False, agg="count", default="0"),
        MacroSpec(_name("ESixHarnessStaticChecks"), verify, {}, "static_checks", "int", required=False, agg="sum"),
        MacroSpec(_name("ESixHarnessStaticFailed"), verify, {}, "static_failed", "int", required=False, agg="sum"),
    ]
    # Costs.
    specs += [
        MacroSpec(_name("ESixCostTotal"), costs, {"arm": "all", "phase": "total"}, "cost_usd", "dec2", required=False),
        MacroSpec(_name("ESixCostCalls"), costs, {"arm": "all", "phase": "total"}, "n", "int", required=False),
    ]
    for arm, alabel in SWE_ARMS.items():
        for model, mlabel in SWE_MODELS.items():
            if arm == "exec" and model not in SWE_EXEC_MODELS:
                continue
            w = {"arm": arm, "phase": "subjects", "model_id": model}
            specs += [
                MacroSpec(_name("ESixCost", alabel, "Subjects", mlabel), costs, w, "cost_usd", "dec2", required=False),
                MacroSpec(_name("ESixCost", alabel, "Subjects", mlabel, "Calls"), costs, w, "n", "int", required=False),
            ]
        specs += [
            MacroSpec(_name("ESixCost", alabel, "Judge"), costs, {"arm": arm, "phase": "judge"}, "cost_usd", "dec2", required=False),
            MacroSpec(_name("ESixCost", alabel, "ProbeReplay"), costs, {"arm": arm, "phase": "probe_replay"}, "cost_usd", "dec2", required=False),
        ]
    specs.append(MacroSpec(_name("ESixCostExecJudgePlain"), costs, {"arm": "exec", "phase": "judge_plain_brief"}, "cost_usd", "dec2", required=False))
    return specs


def _sensitivity_specs() -> list[MacroSpec]:
    """X7 sensitivity cells (sensitivity.py): KBV in the follow-up experiments beside the April cells.

    Temperature: Flash-Lite at 0.7 (``\\crSensFlashLiteTemp<Cond>``) vs the April 1.0
    cells (``\\crSensFlashLiteTempApril<Cond>``). Pressure type: rigor pressure
    (``\\crSensRigor<Model>``) vs April novelty pressure (``\\crSensNovelty<Model>``).
    """
    csv = "sensitivity_cells.csv"
    specs: list[MacroSpec] = []

    def _cell(base: str, where: dict[str, str], with_n: bool) -> list[MacroSpec]:
        out = [
            MacroSpec(_name(base), csv, where, "kbv_rate", "pct1", required=False),
            MacroSpec(_name(base, "Frac"), csv, where, ("kbv_count", "n_with_final_probe"), "frac", required=False),
            MacroSpec(_name(base, "Ci"), csv, where, ("ci_low", "ci_high"), "ci", required=False),
        ]
        if with_n:
            out.append(MacroSpec(_name(base, "N"), csv, where, "n_with_final_probe", "int", required=False))
        return out

    flash = "gemini/gemini-3.1-flash-lite-preview"
    for cond in MULTI_TURN:
        clabel = CONDITION_LABELS[cond]
        specs += _cell("SensFlashLiteTemp" + clabel, {"section": "temperature", "model_id": flash, "condition": cond, "variant": "followup"}, True)
        specs += _cell("SensFlashLiteTempApril" + clabel, {"section": "temperature", "model_id": flash, "condition": cond, "variant": "april"}, False)
    for model in ("openai/gpt-5.4", "anthropic/claude-sonnet-4-6", flash):
        mlabel = MODEL_LABELS[model]
        specs += _cell("SensRigor" + mlabel, {"section": "pressure_type", "model_id": model, "condition": "multi_turn_pressure_rigor", "variant": "followup"}, True)
        specs += _cell("SensNovelty" + mlabel, {"section": "pressure_type", "model_id": model, "condition": "multi_turn_pressure", "variant": "april"}, False)
    return specs


def _followup_runs_specs() -> list[MacroSpec]:
    """X11 follow-up run counts (sensitivity.py, ``sensitivity_runs.csv``).

    ``\\crFollowup<A|B><Planned|Completed|Failed>Runs``. Planned is briefs x
    conditions x models x repetitions of the follow-up's ``run_summary.json``;
    completed and failed are its ``total_runs`` and ``failed_runs``.
    """
    csv = "sensitivity_runs.csv"
    specs: list[MacroSpec] = []
    for followup, flabel in (("followup_a", "A"), ("followup_b", "B")):
        w = {"followup": followup}
        for col, clabel in (("planned", "Planned"), ("completed", "Completed"), ("failed", "Failed")):
            specs.append(MacroSpec(_name("Followup", flabel, clabel, "Runs"), csv, w, col, "int", required=False))
    return specs


# ---------------------------------------------------------------------------
# X8 groups: E4 gKBV per arm, surface gap, probe accuracy, B10, B16, judge sensitivity
# ---------------------------------------------------------------------------

OPUS_JUDGE = "anthropic/claude-opus-4-6"
# E4: the subjects whose Opus structured scores live in source structured_opus.
E4_OPUS_MODELS = (
    "anthropic/claude-sonnet-4-6",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
)
ARM_CONDITIONS = {"single_shot": "SingleShot", "multi_turn_pressure": "Pressure"}


def _gkbv_opus_arm_specs() -> list[MacroSpec]:
    """E4 gKBV under one judge (Opus 4.6) for all seven subjects.

    ``\\crGkbvOpusArm<Model><Condition>`` (+ ``Ci``, ``N``) for single-shot and
    pressure. Sonnet, Llama and Qwen read ``source=structured_opus`` (E4); the
    four other models read ``source=structured``, which Opus already judged
    (same values as ``\\crGkbv<Model><Condition>``). The single-shot to
    pressure tests come from graded_severity_opus_arm_tests.csv, one Holm
    family of seven: ``\\crGkbvOpusArmDelta<Model>``, ``\\crGkbvOpusArmRawP<Model>``
    (brief-paired Wilcoxon), ``\\crGkbvOpusArmHolmP<Model>`` (Holm over the
    seven), ``\\crGkbvOpusArmPairedBriefs<Model>``, plus the family size
    ``\\crGkbvOpusArmHolmFamily`` and the count of Holm-significant models
    ``\\crGkbvOpusArmHolmSignificant``. No GPT-5.4 structured arm over all
    subjects exists on disk, so there is no GPT-arm family.
    """
    summ = "graded_severity_summary.csv"
    tests = "graded_severity_opus_arm_tests.csv"
    specs: list[MacroSpec] = []
    for model, mlabel in MODEL_LABELS.items():
        source = "structured_opus" if model in E4_OPUS_MODELS else "structured"
        note = f"Opus 4.6 structured judge, source {source}"
        for cond, clabel in ARM_CONDITIONS.items():
            w = {"source": source, "model_id": model, "condition": cond, "judge_model": OPUS_JUDGE}
            base = "GkbvOpusArm" + mlabel + clabel
            specs += [
                MacroSpec(_name(base), summ, w, "mean_g_kbv", "pct1", required=False, note=note),
                MacroSpec(_name(base, "Ci"), summ, w, ("g_kbv_ci_lo", "g_kbv_ci_hi"), "ci", required=False, note=note),
                MacroSpec(_name(base, "N"), summ, w, "n_runs", "int", required=False, note=note),
            ]
        w = {"source": source, "model_id": model, "judge_model": OPUS_JUDGE}
        specs += [
            MacroSpec(_name("GkbvOpusArmDelta", mlabel), tests, w, "mean_brief_delta", "dec3", required=False, note=note),
            MacroSpec(_name("GkbvOpusArmRawP", mlabel), tests, w, "wilcoxon_p", "pval", required=False,
                      note=note + "; brief-paired Wilcoxon, unadjusted"),
            MacroSpec(_name("GkbvOpusArmHolmP", mlabel), tests, w, "holm_p_opus_arm", "pval", required=False,
                      note=note + "; Holm over the seven Opus-judged models"),
            MacroSpec(_name("GkbvOpusArmPairedBriefs", mlabel), tests, w, "n_briefs_paired", "int", required=False, note=note),
        ]
    specs += [
        MacroSpec(_name("GkbvOpusArmHolmFamily"), tests, {"judge_model": OPUS_JUDGE}, "n_models_in_opus_arm_family", "int",
                  required=False, agg="mean"),
        MacroSpec(_name("GkbvOpusArmHolmSignificant"), tests, {"judge_model": OPUS_JUDGE}, "significant_holm_05_opus_arm", "int",
                  required=False, agg="sum", note="models with Holm-adjusted p < 0.05 in the seven-test family"),
    ]
    return specs


def _sv_constraint_opus_arm_specs() -> list[MacroSpec]:
    """X15 (C3): the constraint-level stretched/violated split under the one Opus judge.

    ``\\crSvConstraintOpusArm<Checks|Stretched|Violated><Condition>`` for single-shot
    and pressure, summed over the seven subjects. The rows with judge_model == Opus
    are the four commercial subjects of source structured and Sonnet, Llama and Qwen
    of source structured_opus (E4); Sonnet's GPT-5.4 structured row is excluded by
    the judge filter. No GPT-5.4 structured arm over all subjects exists on disk.
    """
    summ = "graded_severity_summary.csv"
    specs: list[MacroSpec] = []
    note = "Opus 4.6 structured judge, seven subjects pooled; per-constraint checks"
    for cond, clabel in ARM_CONDITIONS.items():
        w = {"judge_model": OPUS_JUDGE, "condition": cond, "source": ["structured", "structured_opus"]}
        for col, label in (("total_checks", "Checks"), ("total_stretched", "Stretched"), ("total_violated", "Violated")):
            specs.append(MacroSpec(_name("SvConstraintOpusArm", label, clabel), summ, w, col, "int",
                                   required=False, agg="sum", note=note))
    return specs


def _surface_gap_specs() -> list[MacroSpec]:
    """Surface fidelity gap per judge arm, model and condition (surface_gap.py).

    ``\\crSfg<Arm>Arm<Model><Condition>`` and ``Ci``; model ``All`` pools the
    seven subjects. Runs: the clean single-shot and pressure runs with both
    judge arms (crossed_scores_runs).
    """
    csv = "surface_gap.csv"
    specs: list[MacroSpec] = []
    for arm, alabel in JUDGE_ARMS.items():
        for model in ["all", *MODEL_LABELS]:
            mlabel = "All" if model == "all" else MODEL_LABELS[model]
            for cond, clabel in ARM_CONDITIONS.items():
                w = {"judge_arm": arm, "model_id": model, "condition": cond}
                base = "Sfg" + alabel + "Arm" + mlabel + clabel
                specs += [
                    MacroSpec(_name(base), csv, w, "sfg", "dec2", required=False),
                    MacroSpec(_name(base, "Ci"), csv, w, ("sfg_ci_low", "sfg_ci_high"), "cidec2", required=False),
                ]
    return specs


PROBE_SAMPLES = {"as_published": "AsPublished", "clean": "Clean", "clean_untruncated_probes": "CleanUntrunc"}
PROBE_SCOPES = {"multi_turn": "", "multi_turn_pressure": "Pressure"}


def _probe_accuracy_specs() -> list[MacroSpec]:
    """B4 pooled final-probe accuracy (probe_accuracy.py).

    Pooled over models: ``\\crProbeAccPooled[Pressure]<Sample>`` (run mean, +
    ``Ci``, ``N``), ``\\crProbeAccModelMean[Pressure]<Sample>`` (unweighted mean
    of the per-model means) and ``\\crProbeAccShareAboveCutPooled[Pressure]<Sample>``
    (+ ``Frac``). Per model: ``\\crProbeAcc<Model>[Pressure]<Sample>`` and
    ``\\crProbeAccShareAboveCut<Model>[Pressure]<Sample>`` (+ ``Frac``). No
    scope suffix means the three multi-turn conditions pooled. Samples:
    ``AsPublished`` (the submitted 97.3%), ``Clean``, ``CleanUntrunc``.
    ``\\crProbeAccPooledAllRowsFile`` is the mean over every row of
    final_probe_accuracy.csv, follow-up rigor runs included (97.4).
    """
    csv = "probe_accuracy_summary.csv"
    specs: list[MacroSpec] = []
    for sample, slabel in PROBE_SAMPLES.items():
        for scope, sclabel in PROBE_SCOPES.items():
            w = {"sample": sample, "scope": scope, "model_id": "all", "aggregation": "run_mean"}
            base = "ProbeAccPooled" + sclabel + slabel
            share = "ProbeAccShareAboveCutPooled" + sclabel + slabel
            specs += [
                MacroSpec(_name(base), csv, w, "mean_accuracy", "pct1", required=False),
                MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False),
                MacroSpec(_name(base, "N"), csv, w, "n", "int", required=False),
                MacroSpec(_name(share), csv, w, "share_at_or_above_cut", "pct1", required=False),
                MacroSpec(_name(share, "Frac"), csv, w, ("n_at_or_above_cut", "n"), "frac", required=False),
                MacroSpec(_name("ProbeAccModelMean" + sclabel + slabel), csv, {**w, "aggregation": "model_mean"},
                          "mean_accuracy", "pct1", required=False),
            ]
            for model, mlabel in MODEL_LABELS.items():
                wm = {"sample": sample, "scope": scope, "model_id": model, "aggregation": "run_mean"}
                mshare = "ProbeAccShareAboveCut" + mlabel + sclabel + slabel
                specs += [
                    MacroSpec(_name("ProbeAcc" + mlabel + sclabel + slabel), csv, wm, "mean_accuracy", "pct1", required=False),
                    MacroSpec(_name(mshare), csv, wm, "share_at_or_above_cut", "pct1", required=False),
                    MacroSpec(_name(mshare, "Frac"), csv, wm, ("n_at_or_above_cut", "n"), "frac", required=False),
                ]
    w = {"sample": "probe_file", "scope": "all_rows", "model_id": "all", "aggregation": "run_mean"}
    specs += [
        MacroSpec(_name("ProbeAccPooledAllRowsFile"), csv, w, "mean_accuracy", "pct1", required=False,
                  note="every row of final_probe_accuracy.csv, rigor follow-up runs included"),
        MacroSpec(_name("ProbeAccPooledAllRowsFileN"), csv, w, "n", "int", required=False),
    ]
    return specs


LENGTH_GROUP_LABELS = {"all": "All", "commercial": "Commercial", "open_weight": "OpenWeight"}


def _length_fidelity_specs() -> list[MacroSpec]:
    """B10 fidelity on output length per provider group (tokens_context.py, section fidelity_length).

    ``\\crLengthFidelity<Group>``: pressure runs, ``fidelity ~ log_output_tokens +
    complexity + model FE``, brief-clustered SEs. ``Raw``: the same without
    complexity. ``Pub``: the submitted spec (all conditions, condition and
    model FE, complexity, log1p final-response words). ``Complexity`` and
    ``PubComplexity``: the complexity coefficient of the pressure and the
    submitted spec. Each with ``Ci``, ``P`` and ``N`` (no ``N`` for the
    complexity rows; it equals the length row's).
    """
    csv = "tokens_context_summary.csv"
    rows = (
        ("", "pressure", "log_output_tokens", "log_output_tokens", True),
        ("Raw", "pressure_raw", "log_output_tokens", "log_output_tokens", True),
        ("Complexity", "pressure", "log_output_tokens", "complexity", False),
        ("Pub", "published", "log_final_words", "log_final_words", True),
        ("PubComplexity", "published", "log_final_words", "complexity", False),
    )
    specs: list[MacroSpec] = []
    for group, glabel in LENGTH_GROUP_LABELS.items():
        for label, spec, tok, term, with_n in rows:
            w = {"section": "fidelity_length", "name": f"ols fidelity {spec} {tok}: {term}", "group": group}
            base = "LengthFidelity" + label + glabel
            specs += [
                MacroSpec(_name(base), csv, w, "estimate", "dec2", required=False),
                MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "cidec2", required=False),
                MacroSpec(_name(base, "P"), csv, w, "p_value", "pval", required=False),
            ]
            if with_n:
                specs.append(MacroSpec(_name(base, "N"), csv, w, "n", "int", required=False))
    return specs


def _sonnet_truncation_specs() -> list[MacroSpec]:
    """B16 Sonnet output-cap prevalence (sonnet_truncation.py).

    ``\\crSonnetTruncatedClean`` (+ ``Frac``, ``Pct``, ``Den``, ``Ci``): runs with
    any assistant message at 4,096 tokens, clean sample, four main conditions.
    ``\\crSonnetTruncatedPubPct``: the as-published share (the count is
    ``\\crSonnetTruncatedFrac``). ``\\crSonnetFinalCap{Clean,Pub}`` (+ ``Frac``,
    ``Pct``): the final assistant message at the cap. Per condition:
    ``\\crSonnetTruncatedClean<Condition>Frac`` and ``\\crSonnetFinalCapClean<Condition>Frac``.
    Gemini Pro at its logged cap (4,092 tokens): ``\\crGeminiProCapClean`` (+ ``Frac``),
    ``\\crGeminiProFinalCapClean`` and ``\\crGeminiProFinalCapClean<Condition>``.
    """
    csv = "sonnet_truncation.csv"
    specs: list[MacroSpec] = []

    def _block(base: str, sample: str, rule: str, cond: str, parts: tuple[str, ...],
               model: str = "anthropic/claude-sonnet-4-6") -> list[MacroSpec]:
        w = {"sample": sample, "model_id": model, "rule": rule, "condition": cond}
        made = {
            "": MacroSpec(_name(base), csv, w, "n_truncated", "int", required=False),
            "Frac": MacroSpec(_name(base, "Frac"), csv, w, ("n_truncated", "n_runs"), "frac", required=False),
            "Pct": MacroSpec(_name(base, "Pct"), csv, w, "share_truncated", "pct1", required=False),
            "Den": MacroSpec(_name(base, "Den"), csv, w, "n_runs", "int", required=False),
            "Ci": MacroSpec(_name(base, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False),
        }
        return [made[p] for p in parts]

    specs += _block("SonnetTruncatedClean", "clean", "any_turn", "all_main", ("", "Frac", "Pct", "Den", "Ci"))
    specs += _block("SonnetTruncatedPub", "as_published", "any_turn", "all_main", ("Pct",))
    specs += _block("SonnetFinalCapClean", "clean", "final_turn", "all_main", ("", "Frac", "Pct"))
    specs += _block("SonnetFinalCapPub", "as_published", "final_turn", "all_main", ("", "Frac", "Pct"))
    for cond, clabel in CONDITION_LABELS.items():
        specs += _block("SonnetTruncatedClean" + clabel, "clean", "any_turn", cond, ("Frac",))
        specs += _block("SonnetFinalCapClean" + clabel, "clean", "final_turn", cond, ("Frac",))
    specs += _block("GeminiProCapClean", "clean", "any_turn", "all_main", ("", "Frac"), model="gemini/gemini-3.1-pro-preview")
    specs += _block("GeminiProFinalCapClean", "clean", "final_turn", "all_main", ("",), model="gemini/gemini-3.1-pro-preview")
    for cond, clabel in CONDITION_LABELS.items():
        specs += _block("GeminiProFinalCapClean" + clabel, "clean", "final_turn", cond, ("",), model="gemini/gemini-3.1-pro-preview")
    return specs


# judge_opus / judge_gpt are the E1 crossed judges: Opus 4.6 on the Sonnet runs only,
# GPT-5.4 on the six other subjects. The labels say "Crossed" so they are not read as a full arm.
SV_JUDGE_SOURCES = {"skeptical": "Skeptical", "structured": "Structured", "judge_opus": "CrossedOpus", "judge_gpt": "CrossedGpt"}
SV_JUDGE_CONDITIONS = {
    "skeptical": list(CONDITION_LABELS),
    "structured": list(CONDITION_LABELS),
    "judge_opus": ["single_shot", "multi_turn_pressure"],
    "judge_gpt": ["single_shot", "multi_turn_pressure"],
}


def _judge_sensitivity_specs() -> list[MacroSpec]:
    """C3 judge sensitivity of the below-3 split (stretch_violate_by_judge.csv, pooled rows).

    ``\\crSvJudge<Judge><Condition><Metric>`` for each re-scoring judge
    (Skeptical: GPT-5.4 skeptical prompt, five subjects; Structured: the
    structured judge, five commercial subjects; CrossedOpus: E1 Opus on the
    Sonnet runs; CrossedGpt: E1 GPT-5.4 on the six other subjects) and
    ``\\crSvJudgePublishedVs<Judge>...``
    for the published scores of the same runs. Metrics: ``BelowThree`` (share
    of runs below 3), ``MeanAdherence``, ``N``; on multi-turn conditions also
    ``KbvBelowThree`` (+ ``Frac``: published KBV-positive runs below 3 under the
    judge) and, for the re-scoring judges, ``KbvAtThreeMoved`` (+ ``Frac``:
    published KBV runs at 3 that the judge puts below 3).
    """
    csv = "stretch_violate_by_judge.csv"
    specs: list[MacroSpec] = []
    for source, jlabel in SV_JUDGE_SOURCES.items():
        for judge_source, label in ((source, jlabel), ("published", "PublishedVs" + jlabel)):
            for cond in SV_JUDGE_CONDITIONS[source]:
                w = {"judge_source": judge_source, "compared_with": source, "model_id": "all", "condition": cond}
                base = "SvJudge" + label + CONDITION_LABELS[cond]
                specs += [
                    MacroSpec(_name(base, "BelowThree"), csv, w, "share_below_3", "pct1", required=False),
                    MacroSpec(_name(base, "MeanAdherence"), csv, w, "mean_adherence", "dec2", required=False),
                    MacroSpec(_name(base, "N"), csv, w, "n_runs", "int", required=False),
                ]
                if cond == "single_shot":
                    continue
                specs += [
                    MacroSpec(_name(base, "KbvBelowThree"), csv, w, "share_published_kbv_below_3_under_judge", "pct1", required=False),
                    MacroSpec(_name(base, "KbvBelowThreeFrac"), csv, w,
                              ("n_published_kbv_below_3_under_judge", "n_published_kbv_positive"), "frac", required=False),
                ]
                if judge_source != "published":
                    specs += [
                        MacroSpec(_name(base, "KbvAtThreeMoved"), csv, w, "share_published_kbv_at_3_moved_below_3", "pct1", required=False),
                        MacroSpec(_name(base, "KbvAtThreeMovedFrac"), csv, w,
                                  ("n_published_kbv_at_3_moved_below_3", "n_published_kbv_at_3"), "frac", required=False),
                    ]
    return specs


# ---------------------------------------------------------------------------
# X10 groups: KBV under monitoring, checker coverage and agreement per checker type
# ---------------------------------------------------------------------------

MONITORED_MODELS = (
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3.1-flash-lite-preview",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
)


def _monitoring_specs() -> list[MacroSpec]:
    """X10 monitoring cells (monitoring.py): KBV, adherence and interventions under monitoring per subject.

    ``\\crMonKbv<Model>`` (+ ``Frac``, ``Ci``, ``N``) over the judged runs;
    ``\\crMonAdherence<Model>`` and ``\\crMonComplexity<Model>`` (+ ``Ci``, t interval)
    over the judged runs; ``\\crMonInterventions<Model>`` (mean per run) and
    ``\\crMonInterventionRuns<Model>Frac`` over the transcripts;
    ``\\crMonKbvUnmonitored<Model>`` (+ ``Frac``, ``Ci``) is the clean unmonitored
    pressure cell; ``\\crMonKbvDiff<Model>`` (+ ``Ci``, Newcombe) is monitored minus unmonitored.
    """
    csv = "monitoring_cells.csv"
    specs: list[MacroSpec] = []
    for model in MONITORED_MODELS:
        label = MODEL_LABELS[model]
        w = {"model_id": model}
        specs += [
            MacroSpec(_name("MonKbv", label), csv, w, "kbv_rate", "pct1", required=False),
            MacroSpec(_name("MonKbv", label, "Frac"), csv, w, ("kbv_count", "n_with_final_probe"), "frac", required=False),
            MacroSpec(_name("MonKbv", label, "Ci"), csv, w, ("ci_low", "ci_high"), "ci", required=False),
            MacroSpec(_name("MonKbv", label, "N"), csv, w, "n_with_final_probe", "int", required=False),
            MacroSpec(_name("MonTranscripts", label), csv, w, "n_transcripts", "int", required=False),
            MacroSpec(_name("MonAdherence", label), csv, w, "mean_adherence", "dec2", required=False),
            MacroSpec(_name("MonAdherence", label, "Ci"), csv, w, ("adherence_ci_low", "adherence_ci_high"), "cidec2", required=False),
            MacroSpec(_name("MonComplexity", label), csv, w, "mean_complexity", "dec2", required=False),
            MacroSpec(_name("MonComplexity", label, "Ci"), csv, w, ("complexity_ci_low", "complexity_ci_high"), "cidec2", required=False),
            MacroSpec(_name("MonInterventions", label), csv, w, "mean_interventions", "dec1", required=False,
                      note="mean monitor interventions per transcript (all transcripts, judged or not)"),
            MacroSpec(_name("MonInterventionRuns", label, "Frac"), csv, w, ("n_with_intervention", "n_transcripts"), "frac", required=False),
            MacroSpec(_name("MonKbvUnmonitored", label), csv, w, "unmonitored_kbv_rate", "pct1", required=False),
            MacroSpec(_name("MonKbvUnmonitored", label, "Frac"), csv, w,
                      ("unmonitored_kbv_count", "unmonitored_n_with_final_probe"), "frac", required=False),
            MacroSpec(_name("MonKbvUnmonitored", label, "Ci"), csv, w, ("unmonitored_ci_low", "unmonitored_ci_high"), "ci", required=False),
            MacroSpec(_name("MonKbvDiff", label), csv, w, "kbv_diff", "pct1", required=False,
                      note="monitored minus clean unmonitored pressure KBV, percentage points"),
            MacroSpec(_name("MonKbvDiff", label, "Ci"), csv, w, ("kbv_diff_ci_low", "kbv_diff_ci_high"), "ci", required=False),
        ]
    w = {"model_id": "anthropic/claude-sonnet-4-6"}
    specs += [
        MacroSpec(_name("MonOutputCappedSonnetFrac"), csv, w, ("n_output_capped", "n_transcripts"), "frac", required=False,
                  note="monitored Sonnet runs with any non-probe reply at the 4,096-token cap"),
        MacroSpec(_name("MonFinalOutputCappedSonnetFrac"), csv, w, ("n_final_output_capped", "n_transcripts"), "frac", required=False),
    ]
    return specs


# Checker types of checker_rules.yaml: per-brief counts in checker_coverage.csv (columns n_<type>),
# per-type agreement rows in checker_agreement.csv (group checker_type).
CHECKER_TYPES = {
    "numeric_cap_count": "NumericCapCount",
    "numeric_cap_budget": "NumericCapBudget",
    "numeric_cap_duration": "NumericCapDuration",
    "numeric_cap_quantity": "NumericCapQuantity",
    "numeric_range": "NumericRange",
    "exact_count": "ExactCount",
    "threshold_restatement": "ThresholdRestatement",
}
# threshold_restatement constraints yield no value-evidence pairs, so they have no agreement row.
CHECKER_AGREEMENT_TYPES = tuple(t for t in CHECKER_TYPES if t != "threshold_restatement")


def _checker_type_specs() -> list[MacroSpec]:
    """C11 per checker type: checkable constraints and checker-judge agreement.

    ``\\crCheckType<Type>Checkable`` and ``Frac`` (of all hard constraints) read the
    TOTAL row of ``checker_coverage.csv``; ``FracCheckable`` is the share of the
    checkable constraints. ``\\crCheckType<Type>Agree``, ``Pairs`` and ``Kappa`` read the
    value-pair agreement with the structured judge (not satisfied = stretched or
    violated). No ``Share`` percentage: neither CSV holds a per-type ratio column.
    """
    cov = "checker_coverage.csv"
    agr = "checker_agreement.csv"
    total = {"brief_id": "TOTAL"}
    specs: list[MacroSpec] = []
    for ctype, label in CHECKER_TYPES.items():
        base = "CheckType" + label
        specs += [
            MacroSpec(_name(base, "Checkable"), cov, total, f"n_{ctype}", "int", required=False),
            MacroSpec(_name(base, "Frac"), cov, total, (f"n_{ctype}", "n_hard_constraints"), "frac", required=False),
            MacroSpec(_name(base, "FracCheckable"), cov, total, (f"n_{ctype}", "n_checkable"), "frac", required=False),
        ]
        if ctype in CHECKER_AGREEMENT_TYPES:
            w = {"comparison": "structured_judge:value_pairs:not_satisfied=stretched+violated", "group": "checker_type", "level": ctype}
            specs += [
                MacroSpec(_name(base, "Agree"), agr, w, "agreement", "pct1", required=False),
                MacroSpec(_name(base, "Pairs"), agr, w, "n_pairs", "int", required=False),
                # Most pairs are satisfied/satisfied, so raw agreement is near ceiling; cite kappa beside it.
                MacroSpec(_name(base, "Kappa"), agr, w, "kappa", "dec2", required=False),
            ]
    return specs


def _mechanism_handcheck_specs() -> list[MacroSpec]:
    """X9 mechanism hand-check (mechanism_handcheck_agreement.csv): an LLM annotator (Opus 4.6) on the 50 sampled departures.

    ``\\crMechHandcheck<Comparison>`` is the agreement in percent, with ``Agree``
    (count), ``N``, ``Frac`` and ``Kappa`` (Cohen's kappa). Comparisons:
    ``Appears`` (detector vs LLM, yes / no / no_checklist), ``AppearsBinary``,
    ``Verdict`` (all rows), ``VerdictBothAppear``. The ``departs`` rows get no
    agreement macros: the judge says yes on every sampled row, so their
    ``n_agree`` is the LLM yes count, not agreement.
    ``\\crMechHandcheckLlmDeparts<Yes|No|Unclear>`` (+ ``Frac``) give the LLM's
    departs labels; ``\\crMechHandcheckDeparts<Yes|No|Unclear><GptArm|OpusArm>Frac``
    split them by structured judge (GPT-5.4 judges Sonnet; on the Opus arm the
    annotator is the judge's own model). ``\\crMechHandcheckLlmFlaggedN`` counts
    the departures the LLM finds flagged as unmet in the final checklist.
    ``\\crMechHandcheck<Det|Llm>Hedged`` count hedged verdicts among the rows
    where both find the constraint in the checklist.
    """
    csv = "mechanism_handcheck_agreement.csv"
    specs: list[MacroSpec] = []
    comparisons = (("appears", "Appears"), ("appears_binary", "AppearsBinary"), ("verdict", "Verdict"),
                   ("verdict_both_appear", "VerdictBothAppear"))
    for comparison, label in comparisons:
        w = {"comparison": comparison}
        base = "MechHandcheck" + label
        specs += [
            MacroSpec(_name(base), csv, w, "frac_agree", "pct1", required=False),
            MacroSpec(_name(base, "Agree"), csv, w, "n_agree", "int", required=False),
            MacroSpec(_name(base, "N"), csv, w, "n", "int", required=False),
            MacroSpec(_name(base, "Frac"), csv, w, ("n_agree", "n"), "frac", required=False),
            MacroSpec(_name(base, "Kappa"), csv, w, "kappa", "dec2", required=False),
        ]
    w = {"comparison": "departs"}
    for lab, llabel in (("yes", "Yes"), ("no", "No"), ("unclear", "Unclear")):
        specs += [
            MacroSpec(_name("MechHandcheckLlmDeparts", llabel), csv, w, f"b_n_{lab}", "int", required=False),
            MacroSpec(_name("MechHandcheckLlmDeparts", llabel, "Frac"), csv, w, (f"b_n_{lab}", "n"), "frac", required=False),
        ]
    for arm, alabel in (("departs_gpt_arm", "GptArm"), ("departs_opus_arm", "OpusArm")):
        w = {"comparison": arm}
        for lab, llabel in (("yes", "Yes"), ("no", "No"), ("unclear", "Unclear")):
            specs.append(MacroSpec(_name("MechHandcheckDeparts", llabel, alabel, "Frac"), csv, w, (f"b_n_{lab}", "n"),
                                   "frac", required=False))
    # Every sampled row is a judge departure, so the verdict row's flagged count is the departures flagged as unmet.
    specs.append(MacroSpec(_name("MechHandcheckLlmFlaggedN"), csv, {"comparison": "verdict"}, "b_n_flagged_noncompliant",
                           "int", required=False))
    w = {"comparison": "verdict_both_appear"}
    specs += [
        MacroSpec(_name("MechHandcheckDetHedged"), csv, w, "a_n_hedged", "int", required=False),
        MacroSpec(_name("MechHandcheckLlmHedged"), csv, w, "b_n_hedged", "int", required=False),
    ]
    # X13: the sample size is the row count of the seeded sample itself, not the n of one comparison.
    specs.append(MacroSpec(_name("MechHandcheckSampleN"), "mechanism_handcheck_sample.csv", {}, "run_id", "int",
                           required=False, agg="count", note="departures in the seeded hand-check sample"))
    return specs


def _rebuttal_reproduction_specs() -> list[MacroSpec]:
    """X16 (rebuttal_reproduction.py): posted rebuttal values recomputed under the rebuttal's own definition.

    ``\\crRebut<Key>`` prints the reproduced value, ``\\crRebut<Key>Frac`` its
    numerator/denominator and ``\\crRebut<Key>N`` its denominator. Every macro
    selects its row with ``matches_at_posted_precision == yes``: if a re-run no
    longer matches the posted value, the macro goes missing and the paper's
    "reproduced" claim fails to compile instead of printing a wrong claim.
    """
    csv = "rebuttal_reproduction.csv"
    # (key, fmt, add Frac, add N)
    table = [
        ("PublishedN", "int", False, False), ("CleanN", "int", False, False), ("StrayN", "int", False, False),
        ("FailedRuns", "int", False, False), ("CellSingleShot", "int", False, False),
        ("CellMultiTurnNeutral", "int", False, False), ("CellMultiTurnPressure", "int", False, False),
        ("CellCheckpointedPressure", "int", False, False),
        ("OpusArmMin", "dec1", True, False), ("OpusArmMax", "dec1", True, False),
        ("GptKbv", "dec1", True, False), ("GptAuditor", "dec1", True, False),
        ("ProKbvWaived", "dec1", True, False), ("ProKbvGraded", "dec1", True, False),
        ("ProProbesTruncated", "dec1", True, False), ("ProImperfect", "int", True, False),
        ("ProbePooled", "dec1", False, True), ("UntruncMin", "dec1", False, False),
        ("OnsetByTurnTwo", "dec1", True, False), ("ThreeEvents", "dec1", True, False),
        ("SonnetCapRuns", "dec1", True, False), ("OtherCapRuns", "int", True, True),
        ("SonnetAdhTrunc", "dec3", False, True), ("SonnetAdhUntrunc", "dec3", False, True),
        ("SonnetAdhP", "pval", False, False), ("SonnetAdhN", "int", False, False),
        ("SonnetUntruncPressure", "dec1", True, False), ("SonnetUntruncCkpt", "dec1", True, False),
        ("HumanNotSatShared", "dec1", False, True), ("JudgeNotSatShared", "dec1", False, True),
        ("HumanViolJudgeNot", "dec1", False, True), ("SevNotSatSingle", "dec1", False, True),
        ("SevViolSingle", "dec1", True, False), ("SevNotSatPressure", "dec1", False, True),
        ("SevViolPressure", "dec1", False, True), ("SevViolPerRunSingle", "dec2", False, False),
        ("SevViolPerRunPressure", "dec2", False, False), ("SevRunsViolSingle", "dec1", True, False),
        ("SevRunsViolPressure", "dec1", True, False), ("HumanDelta", "dec3", False, True),
        ("HumanMedianDelta", "dec3", False, True), ("HumanRuns", "int", False, False),
        ("HumanItems", "int", False, False), ("HumanLabels", "int", False, False),
        ("RaterFiles", "int", False, False), ("Raters", "int", False, False),
        ("RaterFlagMin", "int", False, False), ("RaterFlagMax", "int", False, False),
        ("GkbvGpt", "dec3", False, True), ("GkbvMini", "dec3", False, True), ("GkbvPro", "dec3", False, True),
        ("GkbvFlashLite", "dec3", False, True), ("GkbvSonnet", "dec3", False, True),
        ("GkbvProPerfect", "dec3", False, True), ("HardPerBrief", "dec2", False, False),
        ("SonnetNotSatPerRun", "dec2", False, False), ("SonnetWilcoxonP", "pval", False, False),
        ("HolmTGpt", "pval", False, False), ("HolmTFlashLite", "pval", False, False),
        ("StretchedOfNonSat", "int", True, False), ("TypologyChecks", "int", False, False),
        ("HardConstraints", "int", False, False),
        ("KbvAtThree", "dec1", True, False), ("BelowThreeSingle", "dec1", True, False),
        ("BelowThreePressure", "dec1", True, False), ("SonnetLlamaBelowThree", "int", True, False),
        ("CellsThreeFour", "dec1", True, False), ("SonnetGptArm", "dec1", True, False),
        ("ExclBelowThreeSingle", "int", True, False), ("ExclBelowThreePressure", "int", True, False),
        ("ExclAdhDelta", "dec3", False, False),
        ("SubstringTurnOne", "dec1", False, True), ("TokensRuns", "int", False, False),
        ("TokensPressureRuns", "int", False, False),
        ("MonitorReflagNamed", "dec1", True, False), ("MonitorNotNamedPrev", "dec1", True, False),
        ("MonitoredRuns", "int", False, False),
        ("MonFlashLiteUnmonitored", "dec1", True, False), ("MonFlashLiteMonitored", "dec1", True, False),
    ]
    specs: list[MacroSpec] = []
    for key, fmt, frac, n in table:
        w = {"key": key, "matches_at_posted_precision": "yes"}
        specs.append(MacroSpec(_name("Rebut", key), csv, w, "value", fmt, required=False,
                               note="posted rebuttal value reproduced under the rebuttal's definition"))
        if frac:
            specs.append(MacroSpec(_name("Rebut", key, "Frac"), csv, w, ("numerator", "denominator"), "frac", required=False))
        if n:
            specs.append(MacroSpec(_name("Rebut", key, "N"), csv, w, "denominator", "int", required=False))
    w = {"key": "ExclAdhDeltaP", "matches_at_posted_precision": "yes"}
    specs += [
        MacroSpec(_name("RebutExclAdhDeltaPMantissa"), csv, w, "value_mantissa", "dec1", required=False,
                  note="mantissa of the Mann-Whitney p; print as \\crRebutExclAdhDeltaPMantissa \\times 10^{\\crRebutExclAdhDeltaPExponent}"),
        MacroSpec(_name("RebutExclAdhDeltaPExponent"), csv, w, "value_exponent", "int", required=False),
    ]
    return specs


def static_specs() -> list[MacroSpec]:
    specs: list[MacroSpec] = []
    specs += _kbv_specs()
    specs += _noncompliance_specs()
    specs += _failed_runs_specs()
    specs += _per_model_specs()
    specs += _graded_severity_specs()
    specs += _stretch_violate_specs()
    specs += _human_specs()
    specs += _tokens_specs()
    specs += _onset_specs()
    specs += _constraint_count_specs()
    specs += _typology_specs()
    specs += _mechanism_specs()
    specs += _checker_specs()
    specs += _crossed_judge_specs()
    specs += _e7_strict_probe_specs()
    specs += _reprobe_specs()
    specs += _e2_uncapped_specs()
    specs += _e5_per_turn_specs()
    specs += _e6_swe_specs()
    specs += _sensitivity_specs()
    specs += _followup_runs_specs()
    specs += _gkbv_opus_arm_specs()
    specs += _sv_constraint_opus_arm_specs()
    specs += _surface_gap_specs()
    specs += _probe_accuracy_specs()
    specs += _length_fidelity_specs()
    specs += _sonnet_truncation_specs()
    specs += _judge_sensitivity_specs()
    specs += _monitoring_specs()
    specs += _checker_type_specs()
    specs += _mechanism_handcheck_specs()
    specs += _rebuttal_reproduction_specs()
    return specs


# ---------------------------------------------------------------------------
# Dynamic ledger registry
# ---------------------------------------------------------------------------


def ledger_specs(csv_dir: Path, csv_name: str = "corrections_ledger.csv") -> list[MacroSpec]:
    """One macro per ledger row, plus Frac and Pub variants when the row has them.

    The macro name is ``\\crLedger<Id><Quantity slug>``. Rows are selected by
    ``id`` and ``quantity``, so the macro survives a re-run that changes the
    recomputed value but keeps the row label.
    """
    path = csv_dir / csv_name
    if not path.exists():
        raise MacroError(f"ledger CSV not found: {path}")
    df = read_csv_str(path)
    for col in ("id", "quantity", "recomputed_value"):
        if col not in df.columns:
            raise MacroError(f"{csv_name}: column {col!r} missing")
    specs: list[MacroSpec] = []
    seen: dict[str, str] = {}
    for _, row in df.iterrows():
        rid, quantity = str(row["id"]).strip(), str(row["quantity"]).strip()
        base = "Ledger" + slug(rid) + slug(quantity)
        name = _name(base)
        if name in seen:
            raise MacroError(f"ledger rows {seen[name]!r} and {(rid, quantity)!r} give the same macro {name}")
        seen[name] = f"{rid}: {quantity}"
        w = {"id": rid, "quantity": quantity}
        specs.append(MacroSpec(name, csv_name, w, "recomputed_value", "auto", note=str(row.get("rule", "") or "")))
        if not _is_blank(row.get("numerator")) and not _is_blank(row.get("denominator")):
            specs.append(MacroSpec(_name(base, "Frac"), csv_name, w, ("numerator", "denominator"), "frac"))
        if not _is_blank(row.get("denominator")):
            specs.append(MacroSpec(_name(base, "Den"), csv_name, w, "denominator", "int"))
        if not _is_blank(row.get("published_value")):
            specs.append(MacroSpec(_name(base, "Pub"), csv_name, w, "published_value", "auto"))
    return specs


def _ledger_alias_specs() -> list[MacroSpec]:
    """Short names for ledger rows the paper quotes often. Same rows as the ledger macros."""
    csv = "corrections_ledger.csv"
    trunc = {"id": "B16", "quantity": "Sonnet runs truncated (any assistant message token_count == 4096), all_main"}
    onset2 = {"id": "B1", "quantity": "share of pressure drift cases with first auditor event by turn 2"}
    onset3 = {"id": "B1", "quantity": "share of pressure drift cases with first auditor event by turn 3"}
    return [
        MacroSpec(_name("SonnetTruncatedRuns"), csv, trunc, "recomputed_value", "int", required=False,
                  note="runs with any assistant message at the 4,096 cap, four main conditions"),
        MacroSpec(_name("SonnetTruncatedFrac"), csv, trunc, ("recomputed_value", "denominator"), "frac", required=False),
        MacroSpec(_name("SonnetTruncatedDen"), csv, trunc, "denominator", "int", required=False),
        MacroSpec(_name("OnsetPubByTurnTwo"), csv, onset2, "recomputed_value", "dec1", required=False,
                  note="as-published sample (329 drift cases); the clean-sample value is \\crOnsetByTurnTwo"),
        MacroSpec(_name("OnsetPubByTurnTwoFrac"), csv, onset2, ("numerator", "denominator"), "frac", required=False),
        MacroSpec(_name("OnsetPubByTurnThree"), csv, onset3, "recomputed_value", "dec1", required=False),
        MacroSpec(_name("OnsetPubByTurnThreeFrac"), csv, onset3, ("numerator", "denominator"), "frac", required=False),
    ]


def build_registry(csv_dir: Path) -> list[MacroSpec]:
    specs = static_specs() + _ledger_alias_specs() + ledger_specs(csv_dir)
    check_unique_names(specs)
    return specs


def check_unique_names(specs: list[MacroSpec]) -> None:
    """Raise on duplicate names, including the ``Val`` twins of p-value macros."""
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise MacroError(f"duplicate macro names in the registry: {dupes}")
    clash = sorted({s.name + "Val" for s in specs if s.fmt == "pval"} & set(names))
    if clash:
        raise MacroError(f"macro names collide with the Val twin of a p-value macro: {clash}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def resolve_all(specs: list[MacroSpec], csv_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Resolve every spec. Returns (emitted rows, missing rows)."""
    cache: dict[str, pd.DataFrame] = {}
    emitted: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for spec in specs:
        try:
            value, raw = resolve(spec, csv_dir, cache)
        except MacroError as exc:
            if spec.required:
                raise MacroError(f"required macro failed: {exc}") from exc
            missing.append({**spec_record(spec), "reason": str(exc)})
            continue
        row = {
            **spec_record(spec),
            "value": emitted_value(spec, value),
            "raw": [str(r) for r in raw] if isinstance(raw, tuple) else str(raw),
            "variant": None,
        }
        emitted.append(row)
        if spec.fmt == "pval":
            # Table twin: the number only, with '<' kept inside math.
            emitted.append({**row, "name": spec.name + "Val", "value": fmt_pval_val(value), "variant": "Val"})
    return emitted, missing


def spec_record(spec: MacroSpec) -> dict[str, Any]:
    """The JSON record of a spec. ``spec_from_record`` inverts it.

    ``by`` is written only when set, so records of specs without it keep
    their earlier shape.
    """
    record = {
        "name": spec.name,
        "csv": spec.csv,
        "where": dict(spec.where),
        "column": list(spec.column) if isinstance(spec.column, tuple) else spec.column,
        "fmt": spec.fmt,
        "agg": spec.agg,
        "required": spec.required,
        "default": spec.default,
        "note": spec.note,
    }
    if spec.by is not None:
        record["by"] = spec.by
    return record


def spec_from_record(row: dict[str, Any]) -> MacroSpec:
    column = tuple(row["column"]) if isinstance(row["column"], list) else row["column"]
    return MacroSpec(row["name"], row["csv"], dict(row["where"]), column, row["fmt"],
                     required=bool(row.get("required", True)), agg=row.get("agg"),
                     note=row.get("note", "") or "", default=row.get("default"), by=row.get("by"))


def emitted_value(spec: MacroSpec, value: str) -> str:
    """The text written to the tex for a resolved value (p values get the prose form)."""
    return fmt_pval_tex(value) if spec.fmt == "pval" else value


def verify_against_csvs(json_path: Path, csv_dir: Path) -> list[dict[str, str]]:
    """Re-resolve every macro in ``cr_macros.json`` and report differences.

    A difference is a changed raw cell, a changed formatted value, a macro
    that no longer resolves, or a ``missing`` entry that now resolves. An
    empty list means the artifact is current with the CSVs.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cache: dict[str, pd.DataFrame] = {}
    diffs: list[dict[str, str]] = []
    for row in payload.get("macros", []):
        if row.get("variant"):
            continue
        spec = spec_from_record(row)
        try:
            value, raw = resolve(spec, csv_dir, cache)
        except MacroError as exc:
            diffs.append({"name": spec.name, "kind": "no_longer_resolves", "detail": str(exc)})
            continue
        raw_s = [str(r) for r in raw] if isinstance(raw, tuple) else str(raw)
        if raw_s != row["raw"]:
            diffs.append({"name": spec.name, "kind": "raw_changed", "detail": f"{row['raw']} -> {raw_s}"})
        elif emitted_value(spec, value) != row["value"]:
            diffs.append({"name": spec.name, "kind": "value_changed", "detail": f"{row['value']} -> {emitted_value(spec, value)}"})
    for row in payload.get("missing", []):
        try:
            spec = spec_from_record(row)
            value, _ = resolve(spec, csv_dir, cache)
        except (MacroError, KeyError):
            continue
        diffs.append({"name": spec.name, "kind": "now_resolves", "detail": emitted_value(spec, value)})
    return diffs


def render_tex(emitted: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by python -m drift_bench.analysis.camera_ready.macros",
        "% Do not edit by hand. Every value comes from a CSV in drift_bench/data/camera_ready/analysis/.",
        "% In prose, end a macro with {} or a control space (\\crCleanN{} runs, \\crCleanN\\ runs); no xspace is appended.",
        "% \\cr...P macros print '$p = 0.086$' / '$p < 0.001$'; the ...PVal twin prints the number only, for tables.",
        f"% Generated {dt.date.today().isoformat()}; {len(emitted)} macros.",
    ]
    for row in emitted:
        lines.append(f"\\newcommand{{{row['name']}}}{{{row['value']}}}")
    return "\n".join(lines) + "\n"


def check_against_tex(tex_path: Path, macros_tex_path: Path | None = None,
                      defined: set[str] | None = None) -> dict[str, list[str]]:
    """Compare ``\\cr...`` macros used in ``tex_path`` with those defined.

    ``defined`` can be passed directly; otherwise it is read from
    ``macros_tex_path`` (default: the generated ``cr_macros.tex``).
    Returns ``{"used_not_defined": [...], "defined_not_used": [...]}``.
    This is a warning, never an error.
    """
    text = Path(tex_path).read_text(encoding="utf-8")
    used = set(TEX_MACRO_RE.findall(text))
    if defined is None:
        src = Path(macros_tex_path or DEFAULT_OUT_DIR / "cr_macros.tex")
        defined = set()
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\\newcommand\{(\\cr[A-Za-z]+)\}", line)
                if m:
                    defined.add(m.group(1))
    return {
        "used_not_defined": sorted(used - defined),
        "defined_not_used": sorted(defined - used),
    }


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR,
        csv_dir: Path | None = None, specs: list[MacroSpec] | None = None) -> dict[str, pd.DataFrame]:
    """Resolve the registry against the CSVs in ``csv_dir`` and write the outputs.

    ``csv_dir`` defaults to ``data_root/camera_ready/analysis``. ``specs``
    overrides the registry (for tests).
    """
    csv_dir = Path(csv_dir) if csv_dir is not None else Path(data_root) / "camera_ready" / "analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_registry(csv_dir) if specs is None else list(specs)
    emitted, missing = resolve_all(specs, csv_dir)

    bad = [r["name"] for r in emitted if BAD_WORD_RE.search(r["value"])]
    if bad:
        raise MacroError(f"macro values contain None or nan: {bad}")
    bare = [r["name"] for r in emitted if "<" in MATH_RE.sub("", r["value"]) or ">" in MATH_RE.sub("", r["value"])]
    if bare:
        raise MacroError(f"macro values contain a bare < or > outside math mode: {bare}")
    tex = render_tex(emitted)
    tex_path = out_dir / "cr_macros.tex"
    tex_path.write_text(tex, encoding="utf-8")
    with open(out_dir / "cr_macros.json", "w", encoding="utf-8") as fh:
        json.dump({"macros": emitted, "missing": missing}, fh, indent=2)

    inputs = sorted({s.csv for s in specs})
    provenance = {
        "inputs": [str(csv_dir / c) for c in inputs],
        "inputs_missing_on_disk": [c for c in inputs if not (csv_dir / c).exists()],
        "filters": [
            "each macro selects one row of its CSV by exact match on every (column, value) in `where`; numeric cells also match numerically; a list value matches any of its items",
            "specs with `agg` reduce the selected rows (sum, mean or count) instead of requiring one row",
            "a required spec that finds no CSV, no row, several rows or a blank cell raises; an optional spec is listed under `missing`",
            "formats: " + ", ".join(FORMATS),
            "nproviders: number of distinct providers among the semicolon-separated model ids of the cell; provider = MODEL_PROVIDERS entry of the longest matching id prefix, case ignored (openai/ OpenAI; anthropic/ Anthropic; gemini/, google/, vertex_ai/gemini Google; together_ai/meta-llama/ Meta; together_ai/Qwen/ Alibaba); an unlisted id raises",
            "ledger macros: one per corrections_ledger.csv row, selected by (id, quantity); Frac, Den and Pub variants when the row has those cells",
        ],
        "row_counts": {
            "macros_registered": len(specs),
            "macros_emitted": len(emitted),
            "macros_missing": len(missing),
            "macros_required": sum(s.required for s in specs),
        },
        "missing": missing,
        "outputs": ["cr_macros.tex", "cr_macros.json"],
        "date": dt.date.today().isoformat(),
    }
    with open(out_dir / "cr_macros.tex.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    table = pd.DataFrame(emitted) if emitted else pd.DataFrame(columns=["name", "value", "csv"])
    missing_df = pd.DataFrame(missing) if missing else pd.DataFrame(columns=["name", "csv", "reason"])
    return {"cr_macros": table, "cr_macros_missing": missing_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the camera-ready paper macros from the analysis CSVs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--csv-dir", type=Path, default=None, help="where the analysis CSVs live (default: <data-root>/camera_ready/analysis)")
    parser.add_argument("--list", action="store_true", help="print name, value and source for every macro")
    parser.add_argument("--check-tex", type=Path, default=None, help="report macros used in this tex but not defined, and defined but not used")
    parser.add_argument("--verify", action="store_true", help="do not regenerate; re-resolve cr_macros.json against the CSVs and exit 1 on any difference")
    args = parser.parse_args()

    csv_dir = args.csv_dir if args.csv_dir is not None else args.data_root / "camera_ready" / "analysis"
    if args.verify:
        diffs = verify_against_csvs(args.out_dir / "cr_macros.json", csv_dir)
        for d in diffs:
            print(f"  {d['kind']}: {d['name']}: {d['detail']}")
        print(f"verify {args.out_dir / 'cr_macros.json'}: {len(diffs)} differences")
        raise SystemExit(1 if diffs else 0)

    tables = run(args.data_root, args.out_dir, csv_dir=args.csv_dir)
    macros = tables["cr_macros"]
    missing = tables["cr_macros_missing"]
    if args.list:
        for _, row in macros.iterrows():
            where = ", ".join(f"{k}={v}" for k, v in row["where"].items())
            print(f"{row['name']:<60} {row['value']:<28} {row['csv']} [{where}] {row['column']} {row['fmt']}")
    print(f"registered {len(macros) + len(missing)}, emitted {len(macros)}, missing {len(missing)}")
    for _, row in missing.iterrows():
        print(f"  missing {row['name']}: {row['reason']}")
    print(f"wrote {args.out_dir / 'cr_macros.tex'} and cr_macros.json")
    if args.check_tex is not None:
        report = check_against_tex(args.check_tex, args.out_dir / "cr_macros.tex")
        print(f"check {args.check_tex}: {len(report['used_not_defined'])} used but not defined, "
              f"{len(report['defined_not_used'])} defined but not used")
        for name in report["used_not_defined"]:
            print(f"  WARNING used but not defined: {name}")


if __name__ == "__main__":
    main()
