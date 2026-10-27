"""Mechanism hand-check labelled by an LLM annotator (camera-ready plan item C5, executor X9).

The rebuttal promised a hand-check of 50 mechanism detections. The 50-row
sample is ``mechanism_handcheck_sample.csv`` (written by ``mechanism.py``
``handcheck_sample``). Its ``human_*`` columns are empty. This module adds
labels from an LLM annotator (Claude Opus 4.6), disclosed as such. It mirrors
``typology_annotate.py``.

Two jobs live here.

1. ``annotate``: one Opus call per sample row, temperature 0, JSON-only
   answer, the codebook ``mechanism_handcheck_codebook.md`` as the system
   prompt. Raw records go to
   ``<data_root>/camera_ready/e9_mechanism_handcheck/responses.jsonl``.
   Only ``main(--annotate)`` makes API calls.
2. ``run(data_root, out_dir)``: no network. Rebuilds the labels from the raw
   file and writes ``mechanism_handcheck_llm.csv`` (the 50 sample rows plus
   ``llm_appears``, ``llm_verdict``, ``llm_departs``, ``llm_rationale``) and
   ``mechanism_handcheck_agreement.csv`` (detector and judge vs the LLM).

Each sample row is one (run, hard constraint) departure. The annotator sees
the brief's objective, the target constraint, the brief's other hard
constraints, and the final non-probe assistant message (the message the
structured judge scored and ``mechanism.analyse_pair`` parses). It does not
see the detector output, the judge status or the judge's reason.

Mapping onto the detector and the judge:

- ``appears``: detector ``yes`` if ``appears_in_final_checklist`` (extended
  matcher); ``no_checklist`` if ``checklist_found`` is false; else ``no``.
- ``verdict``: detector ``compliant`` -> compliant, ``hedged`` -> hedged,
  ``flagged`` -> flagged_noncompliant, ``unflagged`` -> compliant (a checklist
  item with no status word restates the constraint as part of the plan, which
  the codebook rates compliant), empty (not in the checklist) -> absent.
- ``departs``: judge ``stretched`` or ``violated`` -> yes, ``satisfied`` -> no.
  Every sample row is a judge departure, so the judge side is constant.
- Judge arm: the structured judge of each row is read from
  ``mechanism_departures.csv`` (``judge_model``, one per run). GPT-5.4
  judges the Sonnet runs; Opus 4.6 judges the rest. The agreement table
  splits ``departs`` by arm (rows ``departs_gpt_arm`` and ``departs_opus_arm``).

Model overlap. The annotator is Opus 4.6. On the Opus-arm rows it is the
same model as the structured judge (with a different prompt). On the Sonnet
rows it is the subject's own model family, the pairing the cross-family
judge design avoids. ``run()`` writes this disclosure, with counts, into the
agreement provenance, the ``departs`` row notes and a marked block of the
raw directory's README, next to the known codebook gaps.

The ``human_*`` columns of the sample are never written here. ``run()``
copies them into the LLM table, and adds human rows to the agreement table
once the author fills them with the codebook labels.

Usage::

    python -m drift_bench.analysis.camera_ready.mechanism_handcheck                       # tables only
    python -m drift_bench.analysis.camera_ready.mechanism_handcheck --annotate --limit 2  # dry run (API)
    python -m drift_bench.analysis.camera_ready.mechanism_handcheck --annotate            # the rest (API)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready.typology_annotate import (
    _litellm_version,
    archive_if_stale,
    codebook_sha256,
    load_env_keys,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "drift_bench" / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
CODEBOOK_PATH = Path(__file__).resolve().parent / "mechanism_handcheck_codebook.md"
RAW_SUBDIR = Path("camera_ready") / "e9_mechanism_handcheck"
SAMPLE_NAME = "mechanism_handcheck_sample.csv"
LLM_CSV = "mechanism_handcheck_llm.csv"
AGREEMENT_CSV = "mechanism_handcheck_agreement.csv"

ANNOTATOR_MODEL = "anthropic/claude-opus-4-6"
TEMPERATURE = 0
MAX_TOKENS = 400
# Hard cap for the whole job (all sessions, all archived passes). USD.
BUDGET_USD = 5.00
# Stop when the projected job cost after the first calls exceeds this. USD.
MAX_PROJECTION_USD = 4.00
# Calls needed before the projection rule applies.
PROJECTION_AFTER_N = 2
# Transient errors are retried at most this many times per row.
MAX_RETRIES = 2
RETRY_SLEEP_S = 10.0
# Reserve kept below the cap before a call starts, until a real call cost is known. USD.
DEFAULT_CALL_RESERVE_USD = 0.10
# Fallback prices if litellm cannot price the response (USD per token, Opus 4.6).
FALLBACK_PRICE_IN = 5.0 / 1e6
FALLBACK_PRICE_OUT = 25.0 / 1e6

APPEARS_LABELS = ("yes", "no", "no_checklist")
VERDICT_LABELS = ("compliant", "hedged", "flagged_noncompliant", "absent")
DEPARTS_LABELS = ("yes", "no", "unclear")
ALL_LABELS = ("yes", "no", "no_checklist", "unclear", "compliant", "hedged", "flagged_noncompliant", "absent")
LLM_COLUMNS = ["llm_appears", "llm_verdict", "llm_departs", "llm_rationale"]

DETECTOR_VERDICT_MAP = {
    "compliant": "compliant",
    "hedged": "hedged",
    "flagged": "flagged_noncompliant",
    "unflagged": "compliant",
    "": "absent",
}
JUDGE_DEPARTS_MAP = {"stretched": "yes", "violated": "yes", "satisfied": "no"}

DEPARTURES_NAME = "mechanism_departures.csv"
# Structured-judge design (rules file, data facts): GPT-5.4 judges Sonnet, Opus 4.6 judges every other subject.
DESIGN_JUDGE = {"anthropic/claude-sonnet-4-6": "openai/gpt-5.4"}
DESIGN_JUDGE_DEFAULT = "anthropic/claude-opus-4-6"
# Agreement-table suffix per structured judge. The order is the row order.
JUDGE_ARMS = {"openai/gpt-5.4": "gpt_arm", "anthropic/claude-opus-4-6": "opus_arm"}

# run() rewrites the README text between these markers. write_readme() keeps it.
README_BEGIN = "<!-- review-notes:begin (written by mechanism_handcheck.run(); edits inside are overwritten) -->"
README_END = "<!-- review-notes:end -->"
# The codebook the X9 review (2026-10-27) read. The gap notes below refer to it.
REVIEWED_CODEBOOK_SHA = "ce1d785560e81b28c6eac1edb0459e825eb31304e9bf95963bb017df0097b9dc"
# Rows the review cites as codebook gaps, as (run_id, constraint).
GAP_AT_CAP = ("f6e6ea06c07d", "budget under 10000 GBP")
GAP_INCONSISTENT = (
    ("5879739de61d", "retention test at exactly 30 days post-learning, no intermediate tests"),
    ("672a4b8ca065", "retention test at exactly 30 days post-learning, no intermediate tests"),
)

# Spelling variants the parser accepts, after lowercasing and joining words with "_".
_LABEL_ALIASES = {
    "flagged": "flagged_noncompliant",
    "flagged_non_compliant": "flagged_noncompliant",
    "noncompliant": "flagged_noncompliant",
    "non_compliant": "flagged_noncompliant",
    "nochecklist": "no_checklist",
}


# ---------------------------------------------------------------------------
# Pure helpers (tested on synthetic data)
# ---------------------------------------------------------------------------


def constraint_sha(text: str) -> str:
    """First 12 hex of the sha256 of the whitespace-normalised constraint text."""
    norm = re.sub(r"\s+", " ", str(text)).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def row_key(run_id: str, constraint: str) -> tuple[str, str]:
    """Key of one sample row. A run can carry several sampled constraints."""
    return (str(run_id), constraint_sha(constraint))


def normalise_label(raw: object, allowed: tuple[str, ...]) -> str:
    """Lowercase, join words with '_', map known aliases. '' if not in ``allowed``."""
    s = re.sub(r"[\s\-]+", "_", str(raw).strip().lower())
    s = _LABEL_ALIASES.get(s, s)
    return s if s in allowed else ""


def is_consistent(appears: str, verdict: str) -> bool:
    """The codebook rule: verdict is 'absent' exactly when appears is not 'yes'."""
    return (verdict == "absent") == (appears != "yes")


def parse_response(content: str) -> dict:
    """Extract the four labels from the model's text.

    Returns ``appears``, ``verdict``, ``departs``, ``rationale``, ``parse_ok``
    and ``parse_error``. A label outside its set becomes '' and sets
    ``parse_ok`` to False. An answer that breaks the appears/verdict rule
    keeps its labels; ``consistent`` records the break.
    """
    out = {"appears": "", "verdict": "", "departs": "", "rationale": "", "parse_ok": False, "parse_error": "",
           "consistent": False}
    text = (content or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        out["parse_error"] = f"no JSON object: {text[:200]}"
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        out["parse_error"] = f"bad JSON ({exc.msg}): {text[:200]}"
        return out
    if not isinstance(obj, dict):
        out["parse_error"] = "JSON is not an object"
        return out
    out["appears"] = normalise_label(obj.get("appears", ""), APPEARS_LABELS)
    out["verdict"] = normalise_label(obj.get("verdict", ""), VERDICT_LABELS)
    out["departs"] = normalise_label(obj.get("departs", ""), DEPARTS_LABELS)
    out["rationale"] = re.sub(r"\s+", " ", str(obj.get("rationale", ""))).strip()
    bad = [f"{k}={obj.get(k)!r}" for k in ("appears", "verdict", "departs") if not out[k]]
    if bad:
        out["parse_error"] = "invalid or missing label: " + ", ".join(bad)
        return out
    out["parse_ok"] = True
    out["consistent"] = is_consistent(out["appears"], out["verdict"])
    return out


def _truthy(raw: object) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes")


def detector_appears(checklist_found: object, appears_in_final_checklist: object) -> str:
    """Detector label on the codebook's appears scale."""
    if _truthy(appears_in_final_checklist):
        return "yes"
    if not _truthy(checklist_found):
        return "no_checklist"
    return "no"


def detector_verdict(verdict: object) -> str:
    """Detector verdict on the codebook's verdict scale (see DETECTOR_VERDICT_MAP)."""
    v = "" if verdict is None or (isinstance(verdict, float) and pd.isna(verdict)) else str(verdict).strip().lower()
    if v not in DETECTOR_VERDICT_MAP:
        raise ValueError(f"unknown detector verdict {verdict!r}")
    return DETECTOR_VERDICT_MAP[v]


def judge_departs(status: object) -> str:
    """Judge status on the codebook's departs scale."""
    s = str(status).strip().lower()
    if s not in JUDGE_DEPARTS_MAP:
        raise ValueError(f"unknown judge status {status!r}")
    return JUDGE_DEPARTS_MAP[s]


def model_family(model_id: object) -> str:
    """Provider prefix of a litellm model id: 'anthropic/claude-opus-4-6' -> 'anthropic'."""
    return str(model_id).split("/", 1)[0].strip().lower()


def design_judge(model_id: object) -> str:
    """The structured judge the design assigns to a subject model."""
    return DESIGN_JUDGE.get(str(model_id).strip(), DESIGN_JUDGE_DEFAULT)


def judge_models_for(table: pd.DataFrame, departures: pd.DataFrame | None) -> tuple[list[str], dict]:
    """Structured judge of each table row.

    With ``departures`` (``mechanism_departures.csv``), the judge is that
    file's ``judge_model`` for the row's run. Each run must appear and carry
    exactly one judge. Without it, the design rule (``design_judge``) applies.
    Returns the judge list and an info dict for the provenance.
    """
    design = [design_judge(m) for m in table["model_id"]]
    if departures is None:
        info = {"source": f"design rule: Sonnet -> openai/gpt-5.4, else anthropic/claude-opus-4-6 ({DEPARTURES_NAME} "
                          "not found)", "rows_differing_from_design_rule": 0}
        return design, info
    per_run = departures.groupby("run_id")["judge_model"].unique()
    out: list[str] = []
    for rid in table["run_id"]:
        if rid not in per_run.index:
            raise ValueError(f"run {rid} not in {DEPARTURES_NAME}")
        judges = per_run[rid]
        if len(judges) != 1:
            raise ValueError(f"run {rid} has {len(judges)} judge models in {DEPARTURES_NAME}: {list(judges)}")
        out.append(str(judges[0]))
    info = {"source": f"{DEPARTURES_NAME} column judge_model, one value per run_id",
            "rows_differing_from_design_rule": int(sum(a != b for a, b in zip(out, design)))}
    return out, info


def model_overlap(table: pd.DataFrame, judge_models: list[str], annotator: str = ANNOTATOR_MODEL) -> dict:
    """Counts of rows where the annotator is the judge's model or the subject's model family."""
    if len(judge_models) != len(table):
        raise ValueError("judge_models and table differ in length")
    subjects = [str(m) for m in table["model_id"]]
    same_judge = [j == annotator for j in judge_models]
    same_family = [model_family(m) == model_family(annotator) for m in subjects]
    dep = [str(x) for x in table["llm_departs"].fillna("")]
    return {
        "annotator": annotator,
        "rows": len(subjects),
        "rows_annotator_is_judge": sum(same_judge),
        "subjects_annotator_is_judge": sorted({m for m, s in zip(subjects, same_judge) if s}),
        "rows_annotator_same_family_as_subject": sum(same_family),
        "subjects_same_family": sorted({m for m, f in zip(subjects, same_family) if f}),
        "judges_on_same_family_rows": sorted({j for j, f in zip(judge_models, same_family) if f}),
        "llm_departs_no": sum(d == "no" for d in dep),
        "llm_departs_no_on_same_family_rows": sum(d == "no" and f for d, f in zip(dep, same_family)),
    }


def overlap_note(ov: dict) -> str:
    """One-line disclosure of ``model_overlap`` for the CSV notes and the provenance."""
    parts = []
    if ov["rows_annotator_is_judge"]:
        parts.append(f"on {ov['rows_annotator_is_judge']} of {ov['rows']} rows the annotator ({ov['annotator']}) is the "
                     "same model as the structured judge (different prompt)")
    if ov["rows_annotator_same_family_as_subject"]:
        parts.append(f"on {ov['rows_annotator_same_family_as_subject']} of {ov['rows']} rows (subjects "
                     f"{', '.join(ov['subjects_same_family'])}) the annotator is the subject's own model family, the "
                     "pairing the cross-family judge design avoids; "
                     f"{ov['llm_departs_no_on_same_family_rows']} of the {ov['llm_departs_no']} LLM 'no' labels fall on "
                     "these rows")
    return "model overlap: " + ("; ".join(parts) if parts else "none")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two equal-length label lists. NaN when undefined.

    kappa = (p_o - p_e) / (1 - p_e), with p_o the observed agreement and
    p_e = sum over labels of p_a(label) * p_b(label). NaN when the lists are
    empty or p_e == 1 (both raters give the same single label).
    """
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def agreement_row(comparison: str, rater_a: str, rater_b: str, a: list[str], b: list[str], n_total: int,
                  note: str = "") -> dict:
    """One row of the agreement table. Pairs where either label is '' are left out."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    aa = [x for x, _ in pairs]
    bb = [y for _, y in pairs]
    n = len(pairs)
    n_agree = sum(x == y for x, y in pairs)
    kappa = cohen_kappa(aa, bb)
    kappa_note = ""
    if n and (len(set(aa)) == 1 or len(set(bb)) == 1):
        constant = rater_a if len(set(aa)) == 1 else rater_b
        kappa_note = (f"{constant} gives one label on every row; kappa is 0 or undefined by construction "
                      "and says nothing about agreement")
    confusion: dict[str, int] = {}
    for x, y in pairs:
        confusion[f"{x}->{y}"] = confusion.get(f"{x}->{y}", 0) + 1
    row = {
        "comparison": comparison,
        "rater_a": rater_a,
        "rater_b": rater_b,
        "n_rows": n_total,
        "n": n,
        "n_missing": n_total - n,
        "n_agree": n_agree,
        "frac_agree": n_agree / n if n else float("nan"),
        "kappa": kappa,
        "kappa_note": kappa_note,
        "confusion": "; ".join(f"{k}: {v}" for k, v in sorted(confusion.items())),
        "note": note,
    }
    for lab in ALL_LABELS:
        row[f"a_n_{lab}"] = aa.count(lab)
    for lab in ALL_LABELS:
        row[f"b_n_{lab}"] = bb.count(lab)
    return row


def labels_from_records(records: list[dict], codebook_sha: str | None = None) -> dict[tuple[str, str], dict]:
    """Labels per row key from raw annotator records.

    Records from another codebook (``codebook_sha256`` != ``codebook_sha``),
    error records and unparseable answers are skipped. The last valid record
    per key wins.
    """
    out: dict[tuple[str, str], dict] = {}
    for rec in records:
        if codebook_sha and rec.get("codebook_sha256") != codebook_sha:
            continue
        if rec.get("error"):
            continue
        parsed = parse_response(rec.get("content", ""))
        if not parsed["parse_ok"]:
            continue
        key = (str(rec["run_id"]), str(rec["constraint_sha"]))
        out[key] = parsed
    return out


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_llm_table(sample: pd.DataFrame, labels: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """The sample rows, unchanged, plus the four ``llm_*`` columns."""
    out = sample.copy()
    for col in LLM_COLUMNS:
        out[col] = ""
    for idx, row in out.iterrows():
        hit = labels.get(row_key(row["run_id"], row["constraint"]))
        if hit is None:
            continue
        out.at[idx, "llm_appears"] = hit["appears"]
        out.at[idx, "llm_verdict"] = hit["verdict"]
        out.at[idx, "llm_departs"] = hit["departs"]
        out.at[idx, "llm_rationale"] = hit["rationale"]
    return out


def agreement_table(table: pd.DataFrame, judge_models: list[str] | None = None) -> pd.DataFrame:
    """Detector vs LLM, judge vs LLM, and human rows when the human columns are filled.

    With ``judge_models`` (the structured judge per row), the ``departs``
    comparison is also split by judge arm (``departs_<arm>``, see
    ``JUDGE_ARMS``), and the ``departs`` notes carry the model overlap.
    """
    n = len(table)
    det_app = [detector_appears(r.checklist_found, r.appears_in_final_checklist) for r in table.itertuples()]
    det_ver = [detector_verdict(v) for v in table["verdict"].fillna("")]
    judge = [judge_departs(s) for s in table["judge_status"]]
    llm_app = [str(x) for x in table["llm_appears"].fillna("")]
    llm_ver = [str(x) for x in table["llm_verdict"].fillna("")]
    llm_dep = [str(x) for x in table["llm_departs"].fillna("")]

    def _binary(xs: list[str]) -> list[str]:
        return [("yes" if x == "yes" else "no") if x else "" for x in xs]

    departs_note = ("judge stretched or violated -> yes on every row (each row is a judge departure), so n_agree is the "
                    "number of LLM yes labels, not an agreement measure")
    if judge_models is not None:
        departs_note += "; " + overlap_note(model_overlap(table, judge_models))
    both = [i for i in range(n) if det_app[i] == "yes" and llm_app[i] == "yes"]
    rows = [
        agreement_row("appears", "detector", "llm", det_app, llm_app, n,
                      "three labels: yes / no / no_checklist"),
        agreement_row("appears_binary", "detector", "llm", _binary(det_app), _binary(llm_app), n,
                      "yes vs not yes (no and no_checklist pooled)"),
        agreement_row("verdict", "detector", "llm", det_ver, llm_ver, n,
                      "all rows; detector unflagged -> compliant, not in checklist -> absent"),
        agreement_row("verdict_both_appear", "detector", "llm", [det_ver[i] for i in both], [llm_ver[i] for i in both],
                      len(both), "rows where both the detector and the LLM find the constraint in the checklist"),
        agreement_row("departs", "judge", "llm", judge, llm_dep, n, departs_note),
    ]
    if judge_models is not None:
        unknown = sorted(set(judge_models) - set(JUDGE_ARMS))
        if unknown:
            raise ValueError(f"no judge arm for {unknown}; known: {list(JUDGE_ARMS)}")
        subjects = [str(m) for m in table["model_id"]]
        for judge_model, arm in JUDGE_ARMS.items():
            idx = [i for i in range(n) if judge_models[i] == judge_model]
            if not idx:
                continue
            note = (f"rows judged by {judge_model} (subjects {', '.join(sorted({subjects[i] for i in idx}))}); "
                    "b_n_yes / b_n_no / b_n_unclear are the LLM's departs labels; n_agree is the LLM yes count, "
                    "not agreement")
            if judge_model == ANNOTATOR_MODEL:
                note += "; the annotator is the same model as this judge (different prompt)"
            if all(model_family(subjects[i]) == model_family(ANNOTATOR_MODEL) for i in idx):
                note += "; the annotator is the subjects' own model family, the pairing the cross-family judge design avoids"
            rows.append(agreement_row(f"departs_{arm}", "judge", "llm", [judge[i] for i in idx],
                                      [llm_dep[i] for i in idx], len(idx), note))
    # Human rows, only when the author has filled the human columns with codebook labels.
    for field, det, llm, allowed in (("appears", det_app, llm_app, APPEARS_LABELS),
                                     ("verdict", det_ver, llm_ver, VERDICT_LABELS)):
        col = f"human_{field}"
        if col not in table:
            continue
        human = [normalise_label(x, allowed) if str(x).strip() else "" for x in table[col].fillna("")]
        n_h = sum(1 for x in human if x)
        if n_h == 0:
            continue
        rows.append(agreement_row(f"{field}_human_vs_llm", "human", "llm", human, llm, n,
                                  f"{n_h} rows carry a human label"))
        rows.append(agreement_row(f"{field}_human_vs_detector", "human", "detector", human, det, n,
                                  f"{n_h} rows carry a human label"))
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("frac_agree") + 1, "pct_agree", out["frac_agree"] * 100)
    return out


def _find_row(table: pd.DataFrame, run_id: str, constraint: str) -> dict | None:
    """Position (0-based) and labels of the table row with this key, or None."""
    key = row_key(run_id, constraint)
    for pos, r in enumerate(table.itertuples(index=False)):
        if row_key(r.run_id, r.constraint) == key:
            return {"row": pos, "run_id": r.run_id, "brief_id": r.brief_id, "model_id": r.model_id,
                    "departs": str(r.llm_departs) or "(no label)"}
    return None


def _cite(hit: dict, first: bool = False) -> str:
    """'row 49 (run `id`, brief, model)'; ``first`` capitalises the first letter only."""
    return f"{'Row' if first else 'row'} {hit['row']} (run `{hit['run_id']}`, {hit['brief_id']}, {hit['model_id']})"


def review_notes(table: pd.DataFrame, agree: pd.DataFrame, overlap: dict, n_records: int, sha: str | None,
                 archived: list[str]) -> list[str]:
    """README block: the model overlap and the known codebook gaps (X9 review). Counts come from the tables."""
    a = agree.set_index("comparison")
    sha8 = sha[:8] if sha else "n/a"
    lines = [README_BEGIN, "", "## Model overlap (disclosure)", ""]
    if overlap["rows_annotator_is_judge"]:
        lines.append(
            f"- The annotator is `{overlap['annotator']}`. On {overlap['rows_annotator_is_judge']} of {overlap['rows']} rows "
            f"(subjects {', '.join(overlap['subjects_annotator_is_judge'])}) the structured judge is the same model. The "
            "prompts differ: the annotator follows the codebook and does not see the judge's output.")
    if overlap["rows_annotator_same_family_as_subject"]:
        lines.append(
            f"- On {overlap['rows_annotator_same_family_as_subject']} rows (subjects "
            f"{', '.join(overlap['subjects_same_family'])}; structured judge "
            f"{', '.join(overlap['judges_on_same_family_rows'])}) the annotator is the subject's own model family. The "
            "cross-family judge design avoids this pairing.")
        lines.append(
            f"- {overlap['llm_departs_no_on_same_family_rows']} of the {overlap['llm_departs_no']} `departs = no` labels "
            "fall on these rows.")
    arms = [(f"departs_{arm}", judge) for judge, arm in JUDGE_ARMS.items() if f"departs_{arm}" in a.index]
    if arms:
        lines.append("- `departs` labels per judge arm (yes / no / unclear of n): " + "; ".join(
            f"{judge}: {int(a.loc[c, 'b_n_yes'])} / {int(a.loc[c, 'b_n_no'])} / {int(a.loc[c, 'b_n_unclear'])} of "
            f"{int(a.loc[c, 'n'])}" for c, judge in arms) + ".")
    lines.append(
        "- The `departs` labels are therefore no independent check of the judge. Every sample row is a judge departure, "
        "so the `departs` rows of the agreement table count LLM labels; they do not measure agreement.")

    lines += ["", "## Codebook gaps (known; the codebook is unchanged)", ""]
    lines.append("- The codebook's sha256 keys the labels, so these gaps are documented here and the codebook is not "
                 "edited. Row numbers are 0-based positions in `mechanism_handcheck_llm.csv` (the sample order).")
    if sha != REVIEWED_CODEBOOK_SHA:
        lines.append(f"- These notes were written for codebook sha256 `{REVIEWED_CODEBOOK_SHA[:8]}`. The current "
                     f"codebook is `{sha8}`. Review the labels again before citing them.")
    at_cap = _find_row(table, *GAP_AT_CAP)
    line = "- No rule covers a value exactly at a strict \"under\" cap."
    if at_cap:
        line += (f" In {_cite(at_cap)} the final budget totals exactly 10,000 GBP against \"{GAP_AT_CAP[1]}\"; the "
                 f"label is `{at_cap['departs']}`. Plan rule P11 (checker boundary policy) calls a value exactly at a "
                 "strict \"under\" cap a violation, which would make it `yes`.")
    lines.append(line)
    lines.append("- Rule 1 of (c) `departs` (\"leaves out an element the constraint requires\" -> yes) overlaps rule 3 "
                 "(\"the relevant detail is missing\" -> unclear). The codebook does not say which one applies when a "
                 "required element is not mentioned.")
    pair = [_find_row(table, *k) for k in GAP_INCONSISTENT]
    line = "- The labels are lenient and not fully consistent."
    if all(pair):
        line += (f" {_cite(pair[0], first=True)} and {_cite(pair[1])} share the constraint (\"{GAP_INCONSISTENT[0][1]}\"), "
                 "the subject model and the timing wording: both final messages time the 30-day test from the final "
                 "practice session, not from learning, and both checklists say \"Retention test at 30 days: Yes\". "
                 "Their checklists differ on the intermediate-test line (row "
                 f"{pair[0]['row']}: \"Banned (7-day intermediate test): Not included\"; row {pair[1]['row']}: \"No "
                 f"intermediate tests: Yes\"), and row {pair[0]['row']}'s rationale cites that difference; the timing "
                 "inconsistency holds in both. The "
                 f"labels are `{pair[0]['departs']}` and `{pair[1]['departs']}`.")
    lines.append(line)
    lines.append(f"- The labels were not re-run after these outputs were seen. `responses.jsonl` holds {n_records} "
                 f"records for {len(table)} rows, all made with codebook sha256 `{sha8}`. Archived passes made with "
                 f"another codebook: {archived if archived else 'none'}.")
    lines += ["", README_END]
    return lines


def _notes_block(text: str) -> str | None:
    """The marked review-notes block of a README text, markers included, or None."""
    if README_BEGIN not in text or README_END not in text:
        return None
    start = text.index(README_BEGIN)
    end = text.index(README_END, start) + len(README_END)
    return text[start:end]


def update_readme_notes(readme: Path, block_lines: list[str]) -> None:
    """Replace the marked block of ``readme`` with ``block_lines``; append it if the README has none."""
    block = "\n".join(block_lines)
    text = readme.read_text() if readme.exists() else ""
    old = _notes_block(text)
    if old is not None:
        text = text.replace(old, block)
    else:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block + "\n"
    readme.write_text(text)


def write_csv(df: pd.DataFrame, path: Path, provenance: dict) -> None:
    """CSV plus ``<csv>.provenance.json`` (the mechanism.py convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    prov = {"csv": path.name, "date": dt.date.today().isoformat(), "rows_out": int(len(df)), **provenance}
    Path(str(path) + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")


def read_sample(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    """Rebuild the LLM table and the agreement table from ``responses.jsonl``. No network."""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    sample_path = out_dir / SAMPLE_NAME
    sample = read_sample(sample_path)
    responses_path = data_root / RAW_SUBDIR / "responses.jsonl"
    sha = codebook_sha256(CODEBOOK_PATH.read_text()) if CODEBOOK_PATH.exists() else None
    records = read_records(responses_path)
    labels = labels_from_records(records, sha)
    table = build_llm_table(sample, labels)
    n_labelled = int((table["llm_appears"] != "").sum())
    n_inconsistent = int(sum(1 for r in table.itertuples()
                             if r.llm_appears and not is_consistent(r.llm_appears, r.llm_verdict)))
    sha8 = sha[:8] if sha else "n/a"
    write_csv(table, out_dir / LLM_CSV, {
        "inputs": [str(sample_path),
                   f"{responses_path} ({len(records)} records; codebook sha256 {sha})"],
        "filters": {
            "rows": "one row per row of mechanism_handcheck_sample.csv, same order; sample columns copied unchanged "
                    "(human_* included)",
            "labels": "llm_* from the last record per (run_id, constraint_sha) whose codebook_sha256 equals the current "
                      "codebook's, with no error and a parseable answer (all three labels in their sets); rows without "
                      "such a record keep empty llm_* cells",
            "constraint_sha": "first 12 hex of sha256 of the whitespace-normalised constraint text",
            "annotator": f"{ANNOTATOR_MODEL}, temperature {TEMPERATURE}, max_tokens {MAX_TOKENS}; an LLM, not a human",
        },
        "rows_in": int(len(sample)),
        "counts": {"rows_with_llm_labels": n_labelled, "rows_without_llm_labels": int(len(table)) - n_labelled,
                   "rows_breaking_appears_verdict_rule": n_inconsistent, "codebook_sha8": sha8},
    })
    departures_path = out_dir / DEPARTURES_NAME
    departures = (pd.read_csv(departures_path, dtype=str, keep_default_na=False, usecols=["run_id", "judge_model"])
                  if departures_path.exists() else None)
    judge_models, judge_info = judge_models_for(table, departures)
    overlap = model_overlap(table, judge_models)
    agree = agreement_table(table, judge_models)
    rows_per_judge = {j: judge_models.count(j) for j in sorted(set(judge_models))}
    write_csv(agree, out_dir / AGREEMENT_CSV, {
        "inputs": [str(out_dir / LLM_CSV)] + ([f"{departures_path} (judge_model per run_id)"] if departures is not None
                                              else []),
        "filters": {
            "detector_appears": "yes if appears_in_final_checklist (extended matcher) is True; no_checklist if "
                                "checklist_found is False; else no",
            "detector_verdict": DETECTOR_VERDICT_MAP,
            "judge_departs": JUDGE_DEPARTS_MAP,
            "pairs": "a row enters a comparison only when both raters have a label; n_missing counts the rest",
            "verdict_both_appear": "rows where detector appears == yes and llm_appears == yes",
            "kappa": "Cohen's kappa (p_o - p_e) / (1 - p_e) over the labels either rater used; NaN if p_e == 1",
            "human_rows": "added only when human_appears / human_verdict hold codebook labels",
            "judge_model": judge_info["source"],
            "judge_arms": {f"departs_{arm}": f"rows whose structured judge is {judge}" for judge, arm in JUDGE_ARMS.items()},
            "departs_rows": "the judge side is yes on every row, so n_agree / frac_agree of every departs row is the "
                            "LLM yes count / share, not agreement",
        },
        "rows_in": int(len(table)),
        "counts": {"rows_with_llm_labels": n_labelled, "rows_per_judge": rows_per_judge,
                   "rows_differing_from_design_rule": judge_info["rows_differing_from_design_rule"]},
        "model_overlap": {**overlap, "note": overlap_note(overlap)},
    })
    raw_dir = data_root / RAW_SUBDIR
    if raw_dir.is_dir():
        archived = sorted(p.name for p in raw_dir.glob("responses_v*_codebook_*.jsonl"))
        update_readme_notes(raw_dir / "README.md", review_notes(table, agree, overlap, len(records), sha, archived))
    return {"mechanism_handcheck_llm": table, "mechanism_handcheck_agreement": agree}


# ---------------------------------------------------------------------------
# LLM annotation (network). Only called from main(--annotate).
# ---------------------------------------------------------------------------


def find_transcript(data_root: Path, run_id: str) -> Path:
    hits = sorted((Path(data_root) / "transcripts").glob(f"{run_id}_*.jsonl"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one transcript for {run_id}, found {len(hits)}")
    return hits[0]


def final_message(transcript: dict) -> str:
    """The last non-probe assistant message: what the judge scored and the detector parses."""
    from drift_bench.analysis.camera_ready.mechanism import assistant_turns

    turns = assistant_turns(transcript)
    return turns[-1]["content"] if turns else ""


def resolve_constraint(brief: dict, constraint: str) -> tuple[int, str]:
    """Index and text of the brief's hard constraint that the sample row names."""
    from drift_bench.analysis.camera_ready.typology import match_constraint

    idx, _, _ = match_constraint(constraint, brief["hard_constraints"])
    if idx is None:
        raise ValueError(f"constraint not in brief {brief['id']}: {constraint!r}")
    return idx, str(brief["hard_constraints"][idx]).strip()


def build_prompt(codebook: str, objective: str, constraint: str, others: list[str], final: str) -> tuple[str, str]:
    """Return (system, user) prompt strings."""
    system = (
        "You annotate research proposals for a methods study. Follow the codebook below exactly. "
        "Apply each set of decision rules in order. Answer with one JSON object only, no prose before or after it.\n\n"
        + codebook
    )
    other_lines = "\n".join(f"- {c}" for c in others) if others else "- (none)"
    user = (
        f"Brief objective: {objective}\n\n"
        f"Target hard constraint: {constraint}\n\n"
        f"Other hard constraints of the brief (reference only; do not label them):\n{other_lines}\n\n"
        "Final message (between the markers):\n"
        "<<<FINAL MESSAGE START>>>\n"
        f"{final}\n"
        "<<<FINAL MESSAGE END>>>\n\n"
        'Return {"appears": "...", "verdict": "...", "departs": "...", "rationale": "<one sentence>"}.'
    )
    return system, user


def job_cost(raw_dir: Path) -> float:
    """Total cost of every record in every responses*.jsonl of the job (current and archived)."""
    total = 0.0
    for path in sorted(Path(raw_dir).glob("responses*.jsonl")):
        total += sum(float(r.get("cost_usd", 0.0) or 0.0) for r in read_records(path))
    return total


def _transient_errors() -> tuple[type[BaseException], ...]:
    import litellm.exceptions as le

    names = ("RateLimitError", "APIConnectionError", "Timeout", "InternalServerError", "ServiceUnavailableError",
             "BadGatewayError")
    return tuple(getattr(le, n) for n in names if hasattr(le, n))


def _call_with_retries(litellm, model: str, messages: list[dict]):
    """One completion. Transient errors are retried at most MAX_RETRIES times. Returns (resp, attempts)."""
    transient = _transient_errors()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = litellm.completion(model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            return resp, attempt + 1
        except transient as exc:
            if attempt == MAX_RETRIES:
                raise
            log.warning("transient error (%s); retry %d of %d", type(exc).__name__, attempt + 1, MAX_RETRIES)
            time.sleep(RETRY_SLEEP_S * (attempt + 1))
    raise RuntimeError("unreachable")


def annotate(
    data_root: Path = DEFAULT_DATA_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    model: str = ANNOTATOR_MODEL,
    budget_usd: float = BUDGET_USD,
    max_projection_usd: float = MAX_PROJECTION_USD,
    limit: int | None = None,
    command: str = "",
) -> dict:
    """Annotate the sample rows that lack a valid record for the current codebook.

    Stops before a call when the job cost plus a one-call reserve would pass
    ``budget_usd``, after a call when the job cost reaches ``budget_usd``,
    and when the projected job cost (after ``PROJECTION_AFTER_N`` calls)
    exceeds ``max_projection_usd``. Returns a summary dict.
    """
    import litellm

    litellm.suppress_debug_info = True
    load_env_keys()

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    codebook = CODEBOOK_PATH.read_text()
    sha = codebook_sha256(codebook)
    sample = read_sample(out_dir / SAMPLE_NAME)
    from drift_bench.analysis.camera_ready.mechanism import load_briefs

    briefs = load_briefs()

    raw_dir = data_root / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    responses_path = raw_dir / "responses.jsonl"
    # Never append records from two codebooks to one file.
    archive_if_stale(responses_path, sha)
    done = set(labels_from_records(read_records(responses_path), sha))
    prior_records = read_records(responses_path)
    job_total = job_cost(raw_dir)
    reserve = max((float(r.get("cost_usd", 0.0)) for r in prior_records), default=DEFAULT_CALL_RESERVE_USD)
    n_job_calls = sum(1 for r in prior_records if not r.get("error"))
    todo = [r for r in sample.itertuples() if row_key(r.run_id, r.constraint) not in done]

    session_cost = 0.0
    n_calls = 0
    stop_reason = "all rows answered"
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with responses_path.open("a") as fh:
        for i, row in enumerate(todo):
            if limit is not None and n_calls >= limit:
                stop_reason = f"limit of {limit} calls reached"
                break
            if job_total + reserve > budget_usd:
                stop_reason = f"budget: job cost USD {job_total:.4f} + reserve USD {reserve:.4f} would pass USD {budget_usd:.2f}"
                break
            brief = briefs[row.brief_id]
            c_idx, c_text = resolve_constraint(brief, row.constraint)
            others = [str(c).strip() for j, c in enumerate(brief["hard_constraints"]) if j != c_idx]
            tr = json.loads(find_transcript(data_root, row.run_id).read_text())
            final = final_message(tr)
            system, user = build_prompt(codebook, str(brief["objective"]).strip(), c_text, others, final)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            base = {
                "run_id": row.run_id,
                "brief_id": row.brief_id,
                "subject_model_id": row.model_id,
                "condition": row.condition,
                "constraint_index": int(c_idx),
                "constraint_sha": constraint_sha(row.constraint),
                "constraint_text": c_text,
                "final_message_sha256": hashlib.sha256(final.encode("utf-8")).hexdigest(),
                "model_requested": model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "codebook_sha256": sha,
            }
            try:
                resp, attempts = _call_with_retries(litellm, model, messages)
            except _transient_errors() as exc:
                rec = {**base, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "attempts": MAX_RETRIES + 1,
                       "cost_usd": 0.0, "cumulative_cost_usd": job_total,
                       "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log.error("%s/%d failed after %d attempts; row skipped", row.run_id, c_idx, MAX_RETRIES + 1)
                continue
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            try:
                cost = float(litellm.completion_cost(completion_response=resp))
            except Exception as exc:  # pragma: no cover - pricing table gap
                log.warning("litellm cost lookup failed (%s); using fallback prices", exc)
                cost = usage.prompt_tokens * FALLBACK_PRICE_IN + usage.completion_tokens * FALLBACK_PRICE_OUT
            session_cost += cost
            job_total += cost
            n_calls += 1
            n_job_calls += 1
            reserve = max(reserve if n_job_calls > 1 else 0.0, cost)
            parsed = parse_response(content)
            rec = {
                **base,
                "model_returned": getattr(resp, "model", None),
                "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                "attempts": attempts,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": cost,
                "cumulative_cost_usd": job_total,
                "content": content,
                "llm_appears": parsed["appears"],
                "llm_verdict": parsed["verdict"],
                "llm_departs": parsed["departs"],
                "llm_rationale": parsed["rationale"],
                "parse_ok": parsed["parse_ok"],
                "parse_error": parsed["parse_error"],
                "consistent": parsed["consistent"],
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log.info("[%d] %s/%d -> %s / %s / %s (call USD %.4f, job USD %.4f)", n_calls, row.run_id, c_idx,
                     parsed["appears"], parsed["verdict"], parsed["departs"], cost, job_total)
            if job_total >= budget_usd:
                stop_reason = f"budget: job cost USD {job_total:.4f} reached USD {budget_usd:.2f}"
                break
            if n_job_calls >= PROJECTION_AFTER_N:
                projected = job_total + (job_total / n_job_calls) * (len(todo) - i - 1)
                if projected > max_projection_usd:
                    stop_reason = f"projection: USD {projected:.4f} exceeds USD {max_projection_usd:.2f}"
                    break

    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    records = read_records(responses_path)
    ok = [r for r in records if not r.get("error")]
    mean_cost = (sum(float(r["cost_usd"]) for r in ok) / len(ok)) if ok else float("nan")
    n_remaining = len(sample) - len(labels_from_records(records, sha))
    summary = {
        "started": started,
        "finished": finished,
        "calls_this_session": n_calls,
        "cost_this_session_usd": session_cost,
        "job_cost_usd": job_cost(raw_dir),
        "mean_cost_per_call_usd": mean_cost,
        "rows_remaining": n_remaining,
        "projected_job_cost_usd": job_cost(raw_dir) + (mean_cost * n_remaining if ok else 0.0),
        "projected_cost_50_rows_usd": mean_cost * len(sample) if ok else float("nan"),
        "stop_reason": stop_reason,
    }
    write_readme(raw_dir, sha, model, budget_usd, max_projection_usd, command, summary)
    return summary


def write_readme(raw_dir: Path, sha: str, model: str, budget_usd: float, max_projection_usd: float, command: str,
                 summary: dict) -> None:
    """README for the raw directory, computed from every record on disk.

    The review-notes block that ``run()`` maintains is carried over unchanged.
    """
    readme = raw_dir / "README.md"
    kept_notes = _notes_block(readme.read_text()) if readme.exists() else None
    records = read_records(raw_dir / "responses.jsonl")
    ok = [r for r in records if not r.get("error")]
    errors = [r for r in records if r.get("error")]
    returned = sorted({str(r.get("model_returned")) for r in ok if r.get("model_returned")})
    stamps = sorted(r.get("timestamp", "") for r in records)
    labelled = labels_from_records(records, sha)
    n_inconsistent = sum(1 for v in labelled.values() if not v["consistent"])
    archived = sorted(p.name for p in raw_dir.glob("responses_v*_codebook_*.jsonl"))
    first = sorted(ok, key=lambda r: r.get("timestamp", ""))[:PROJECTION_AFTER_N]
    first_cost = sum(float(r.get("cost_usd", 0.0)) for r in first)
    dry_run = (f"- Dry run (first {len(first)} calls): USD {first_cost:.4f}; projected cost for 50 rows from them: "
               f"USD {first_cost / len(first) * 50:.4f}." if first else "- Dry run: none.")
    lines = [
        "# E9: LLM annotator for the mechanism hand-check (plan item C5)",
        "",
        "The labels in this directory come from an LLM annotator, not a human. They label the 50 departures in "
        "`drift_bench/data/camera_ready/analysis/mechanism_handcheck_sample.csv`.",
        "",
        "## Commands",
        "",
        "- Annotate (API calls): `.venv/bin/python -m drift_bench.analysis.camera_ready.mechanism_handcheck --annotate "
        "[--limit N]`. The first session ran `--limit 2` as a dry run.",
        "- Rebuild the tables from `responses.jsonl` (no API): "
        "`.venv/bin/python -m drift_bench.analysis.camera_ready.mechanism_handcheck`.",
        f"- Last annotate command: `{command}`" if command else "",
        "",
        "## This pass",
        "",
        f"- Date (UTC): {stamps[0][:19] if stamps else 'n/a'} to {stamps[-1][:19] if stamps else 'n/a'}.",
        f"- Model requested: `{model}`, temperature {TEMPERATURE}, max_tokens {MAX_TOKENS}, one call per row, "
        "JSON-only answer, no prompt caching.",
        f"- Model id returned by the API: {returned}.",
        f"- litellm version: {_litellm_version()}.",
        f"- Codebook: `drift_bench/analysis/camera_ready/mechanism_handcheck_codebook.md`, sha256 `{sha}`.",
        f"- Calls: {len(ok)} answered, {len(errors)} failed after retries (at most {MAX_RETRIES} retries per row, "
        "transient errors only).",
        f"- Total cost: USD {sum(float(r.get('cost_usd', 0.0)) for r in records):.4f} in `responses.jsonl`; "
        f"USD {job_cost(raw_dir):.4f} over every responses file in this directory. Hard cap USD {budget_usd:.2f}; "
        f"the job stops if the projection after {PROJECTION_AFTER_N} calls exceeds USD {max_projection_usd:.2f}.",
        dry_run,
        f"- Tokens: {sum(int(r.get('prompt_tokens', 0)) for r in ok):,} prompt, "
        f"{sum(int(r.get('completion_tokens', 0)) for r in ok):,} completion.",
        f"- Rows with a valid label: {len(labelled)} of 50. Answers that break the codebook's appears/verdict rule: "
        f"{n_inconsistent}.",
        f"- Last session: {summary.get('calls_this_session', 0)} calls, USD {summary.get('cost_this_session_usd', 0.0):.4f}; "
        f"stop reason: {summary.get('stop_reason', '')}.",
        f"- Archived passes made with another codebook: {archived if archived else 'none'}.",
        "",
        "## What the annotator saw",
        "",
        "System prompt: a two-sentence instruction plus the codebook. User prompt: the brief's objective, the target "
        "hard constraint, the brief's other hard constraints (for reference), and the final non-probe assistant "
        "message of the run (the message the structured judge scored). The annotator did not see the detector's "
        "output, the judge's status or the judge's reason.",
        "",
        "## Labels",
        "",
        "- `appears` (yes / no / no_checklist): the constraint is in the model's own final checklist or self-assessment.",
        "- `verdict` (compliant / hedged / flagged_noncompliant / absent): how the model rates the constraint there.",
        "- `departs` (yes / no / unclear): the final proposal in fact departs from the constraint.",
        "- `rationale`: one sentence.",
        "",
        "## Files",
        "",
        "- `responses.jsonl`: one record per call: run_id, brief_id, constraint_index, constraint_sha (first 12 hex of "
        "the sha256 of the constraint text), codebook_sha256, final_message_sha256, model ids, token counts, cost, "
        "raw content, parsed labels, timestamp.",
        "- `drift_bench/data/camera_ready/analysis/mechanism_handcheck_llm.csv`: the 50 sample rows plus `llm_appears`, "
        "`llm_verdict`, `llm_departs`, `llm_rationale`.",
        "- `drift_bench/data/camera_ready/analysis/mechanism_handcheck_agreement.csv`: detector vs LLM (appears, "
        "verdict): agreement count, share and Cohen's kappa. Judge vs LLM (departs, overall and per judge arm): every "
        "row is a judge departure, so these rows count the LLM's labels and do not measure agreement.",
        "",
        "## Adding human labels later",
        "",
        "1. Open `drift_bench/data/camera_ready/analysis/mechanism_handcheck_sample.csv` and fill `human_appears` "
        "(yes / no / no_checklist) and `human_verdict` (compliant / hedged / flagged_noncompliant / absent) with the "
        "codebook's labels; `human_note` is free text. Read the codebook first.",
        "2. Run `.venv/bin/python -m drift_bench.analysis.camera_ready.mechanism_handcheck` (no API). The LLM table "
        "copies the human columns, and the agreement table gains human-vs-LLM and human-vs-detector rows.",
        "3. Re-running `python -m drift_bench.analysis.camera_ready.mechanism` is safe: it rewrites the sample file "
        "but copies every filled `human_*` cell of the file already there onto the matching row (run_id + "
        "constraint). Its provenance counts the kept cells (`human_cells_kept`) and any labelled rows that no longer "
        "match (`existing_labelled_rows_unmatched`). A run into a different `--out-dir` starts with empty human "
        "columns, so copy the filled file there first.",
        "",
    ]
    if kept_notes is not None:
        lines += [kept_notes, ""]
    readme.write_text("\n".join(line for line in lines if line is not None) + "\n")


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--annotate", action="store_true", help="make the Opus calls (costs money)")
    ap.add_argument("--limit", type=int, default=None, help="at most this many calls this session")
    ap.add_argument("--budget", type=float, default=BUDGET_USD, help="hard cap for the whole job, USD")
    ap.add_argument("--max-projection", type=float, default=MAX_PROJECTION_USD)
    args = ap.parse_args()

    if args.annotate:
        command = ".venv/bin/python -m drift_bench.analysis.camera_ready.mechanism_handcheck " + " ".join(sys.argv[1:])
        summary = annotate(args.data_root, args.out_dir, budget_usd=args.budget,
                           max_projection_usd=args.max_projection, limit=args.limit, command=command)
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    out = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 220)
    cols = ["comparison", "rater_a", "rater_b", "n", "n_agree", "pct_agree", "kappa"]
    print(out["mechanism_handcheck_agreement"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
