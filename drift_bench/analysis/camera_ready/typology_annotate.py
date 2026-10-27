"""Constraint typology: build the annotation sheet and run the LLM annotator (E8).

Two jobs live here.

1. ``build_sheet``: one row per hard constraint from the briefs. No network.
   ``run(data_root, out_dir)`` does this and keeps any labels already in the sheet.
2. ``annotate``: one Opus call per constraint, temperature 0, with the codebook
   as the system prompt. Writes ``label_llm`` and ``llm_rationale`` into the sheet
   and the raw responses to ``camera_ready/e8_typology_llm/responses.jsonl``.
   Only ``main(--annotate)`` makes API calls.

Usage::

    python -m drift_bench.analysis.camera_ready.typology_annotate            # sheet only
    python -m drift_bench.analysis.camera_ready.typology_annotate --annotate  # API calls
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "drift_bench" / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
CODEBOOK_PATH = Path(__file__).resolve().parent / "typology_codebook.md"
BRIEFS_DIR = PROJECT_ROOT / "drift_bench" / "briefs"

ANNOTATOR_MODEL = "anthropic/claude-opus-4-6"
BUDGET_USD = 4.0
# Prompt caching is off. Opus 4.6 needs a cacheable prefix of at least 4096 tokens
# and the system prompt here is about 1.6k tokens, so a cache_control marker can
# never engage (the first pass showed 0 cache-read tokens on all 234 calls). When
# this is True the code sets the marker and records the cache-read tokens.
USE_PROMPT_CACHE = False
# Fallback prices if litellm cannot price the response (USD per token).
FALLBACK_PRICE_IN = 5.0 / 1e6
FALLBACK_PRICE_OUT = 25.0 / 1e6

CATEGORY_IDS = [
    "resource_limit",
    "sample_floor",
    "duration_window",
    "count_scope_limit",
    "prohibition",
    "data_source",
    "method_requirement",
    "design_requirement",
]

SHEET_COLUMNS = [
    "brief_id",
    "constraint_index",
    "constraint_text",
    "label_author",
    "label_llm",
    "llm_rationale",
]


def load_constraints(briefs_dir: Path = BRIEFS_DIR) -> pd.DataFrame:
    """One row per hard constraint, in brief-file order."""
    rows = []
    for path in sorted(briefs_dir.glob("*.yaml")):
        brief = yaml.safe_load(path.read_text())
        for i, text in enumerate(brief["hard_constraints"]):
            rows.append(
                {
                    "brief_id": brief["id"],
                    "constraint_index": i,
                    "constraint_text": str(text).strip(),
                    "objective": str(brief["objective"]).strip(),
                }
            )
    return pd.DataFrame(rows)


def build_sheet(constraints: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the sheet. Keep labels from ``existing`` where brief_id and index match."""
    sheet = constraints[["brief_id", "constraint_index", "constraint_text"]].copy()
    for col in ("label_author", "label_llm", "llm_rationale"):
        sheet[col] = ""
    if existing is not None and len(existing):
        key = ["brief_id", "constraint_index"]
        keep = existing[key + [c for c in ("label_author", "label_llm", "llm_rationale") if c in existing]]
        keep = keep.copy()
        keep["constraint_index"] = keep["constraint_index"].astype(int)
        merged = sheet.merge(keep, on=key, how="left", suffixes=("", "_old"))
        for col in ("label_author", "label_llm", "llm_rationale"):
            old = f"{col}_old"
            if old in merged:
                merged[col] = merged[old].fillna("").astype(str)
                merged = merged.drop(columns=[old])
        sheet = merged
    return sheet[SHEET_COLUMNS].reset_index(drop=True)


def read_sheet(path: Path) -> pd.DataFrame:
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet["constraint_index"] = sheet["constraint_index"].astype(int)
    return sheet


def write_sheet(sheet: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet[SHEET_COLUMNS].to_csv(path, index=False)


def write_provenance(csv_path: Path, inputs: list[str], filters: list[str], n_in: int, n_out: int) -> None:
    prov = {
        "csv": csv_path.name,
        "inputs": inputs,
        "filters": filters,
        "rows_in": n_in,
        "rows_out": n_out,
        "date": dt.date.today().isoformat(),
    }
    csv_path.with_name(csv_path.name + ".provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


def labels_from_responses(responses_path: Path, codebook_sha: str | None = None) -> dict[tuple[str, int], tuple[str, str]]:
    """Read ``label_llm`` and ``llm_rationale`` from the raw annotator records.

    ``responses.jsonl`` is the record of the LLM pass, so a fresh output
    directory gets the same labels as the published sheet. Records made
    with another codebook (``codebook_sha256`` != ``codebook_sha``) are
    ignored. The last record per constraint wins.
    """
    labels: dict[tuple[str, int], tuple[str, str]] = {}
    if not responses_path.exists():
        return labels
    for line in responses_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if codebook_sha and rec.get("codebook_sha256") != codebook_sha:
            continue
        cat = rec.get("label_llm")
        rat = rec.get("llm_rationale")
        if cat is None or rat is None:
            cat, rat = parse_response(rec.get("content", ""))
        labels[(rec["brief_id"], int(rec["constraint_index"]))] = (str(cat), str(rat))
    return labels


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    """Build or refresh the typology sheet. No network.

    ``label_author`` is kept from the sheet already in ``out_dir``.
    ``label_llm`` and ``llm_rationale`` come from
    ``<data_root>/camera_ready/e8_typology_llm/responses.jsonl`` (current
    codebook only); the sheet's own values fill any constraint the raw
    file does not cover.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "typology_sheet.csv"
    constraints = load_constraints()
    existing = read_sheet(sheet_path) if sheet_path.exists() else None
    sheet = build_sheet(constraints, existing)
    responses_path = Path(data_root) / "camera_ready" / "e8_typology_llm" / "responses.jsonl"
    sha = codebook_sha256(CODEBOOK_PATH.read_text()) if CODEBOOK_PATH.exists() else None
    raw_labels = labels_from_responses(responses_path, sha)
    n_from_raw = 0
    for idx, row in sheet.iterrows():
        hit = raw_labels.get((row.brief_id, int(row.constraint_index)))
        if hit is not None:
            sheet.at[idx, "label_llm"], sheet.at[idx, "llm_rationale"] = hit
            n_from_raw += 1
    write_sheet(sheet, sheet_path)
    write_provenance(
        sheet_path,
        inputs=[
            str(BRIEFS_DIR / "*.yaml"),
            f"{responses_path} (label_llm, llm_rationale; codebook {sha[:8] if sha else 'n/a'}; {n_from_raw} rows)",
            "typology_sheet.csv (label_author, and LLM labels for rows the raw file lacks)",
        ],
        filters=["one row per entry of hard_constraints, in brief-file order; no rows dropped"],
        n_in=len(constraints),
        n_out=len(sheet),
    )
    return {"typology_sheet": sheet}


# ---------------------------------------------------------------------------
# LLM annotation (network). Only called from main(--annotate).
# ---------------------------------------------------------------------------


def load_env_keys(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ. Never print the values."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_prompt(codebook: str, constraint_text: str, objective: str) -> tuple[str, str]:
    """Return (system, user) prompt strings."""
    system = (
        "You annotate hard constraints from scientific research briefs with one category "
        "from the codebook below. Apply the decision rules in order. Answer with one JSON "
        "object only, no prose before or after it.\n\n" + codebook
    )
    user = (
        f"Brief objective: {objective}\n\n"
        f"Hard constraint: {constraint_text}\n\n"
        'Return {"category": "<id>", "rationale": "<one sentence>"}.'
    )
    return system, user


def parse_response(content: str) -> tuple[str, str]:
    """Extract (category, rationale) from the model's text. Unknown ids map to ''."""
    text = content.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return "", f"unparseable: {text[:200]}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "", f"unparseable: {text[:200]}"
    cat = str(obj.get("category", "")).strip()
    rat = str(obj.get("rationale", "")).strip()
    if cat not in CATEGORY_IDS:
        return "", f"invalid category {cat!r}: {rat}"
    return cat, rat


def cache_read_tokens(usage) -> int:
    """Cache-read tokens from a litellm usage object. Reads both field spellings."""
    details = getattr(usage, "prompt_tokens_details", None)
    for value in (getattr(details, "cached_tokens", None), getattr(usage, "cache_read_input_tokens", None)):
        if value:
            return int(value)
    return 0


def codebook_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def archive_if_stale(responses_path: Path, sha: str) -> Path | None:
    """Archive ``responses.jsonl`` when its records were made with another codebook.

    A record with a missing or different ``codebook_sha256`` makes the file stale.
    The file is renamed to ``responses_v<N>_codebook_<sha8>.jsonl`` (N = next free
    pass number, sha8 = first 8 hex of the records' own sha, or ``unknown``) so the
    new pass never appends to it. Returns the archive path, or None when nothing
    was archived.
    """
    if not responses_path.exists():
        return None
    recs = [json.loads(line) for line in responses_path.read_text().splitlines() if line.strip()]
    if not recs:
        return None
    shas = {r.get("codebook_sha256") for r in recs}
    if shas == {sha}:
        return None
    existing = []
    for p in responses_path.parent.glob("responses_v*_codebook_*.jsonl"):
        m = re.match(r"responses_v(\d+)_codebook_", p.name)
        if m:
            existing.append(int(m.group(1)))
    n = max(existing, default=0) + 1
    old = sorted(s for s in shas if s)
    sha8 = old[0][:8] if len(old) == 1 else "unknown"
    target = responses_path.parent / f"responses_v{n}_codebook_{sha8}.jsonl"
    responses_path.rename(target)
    log.warning("responses.jsonl was made with another codebook; archived it as %s", target.name)
    return target


def archived_passes(raw_dir: Path, sheet: pd.DataFrame) -> list[dict]:
    """Summarise earlier passes kept as ``responses_v<k>_codebook_<sha8>.jsonl``.

    For each file: call count, cost, codebook sha prefix (from the file name), and the
    number of labels that differ from the sheet's current ``label_llm``.
    """
    current = {(r.brief_id, int(r.constraint_index)): r.label_llm for r in sheet.itertuples()}
    out = []
    for path in sorted(raw_dir.glob("responses_v*_codebook_*.jsonl")):
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        changed = [
            (r["brief_id"], int(r["constraint_index"]), r.get("label_llm", ""), current.get((r["brief_id"], int(r["constraint_index"])), ""))
            for r in recs
            if r.get("label_llm", "") != current.get((r["brief_id"], int(r["constraint_index"])), "")
        ]
        shas = sorted({r["codebook_sha256"] for r in recs if r.get("codebook_sha256")})
        out.append(
            {
                "file": path.name,
                "codebook_sha256_prefix": path.stem.split("_codebook_")[-1],
                "codebook_sha256_in_records": shas,
                "n_calls": len(recs),
                "cost_usd": sum(float(r.get("cost_usd", 0.0)) for r in recs),
                "first_timestamp": min((r.get("timestamp", "") for r in recs), default=""),
                "last_timestamp": max((r.get("timestamp", "") for r in recs), default=""),
                "n_labels_changed_vs_current": len(changed),
                "changed": changed,
            }
        )
    return out


def _litellm_version() -> str:
    try:
        from importlib.metadata import version

        return version("litellm")
    except Exception:  # pragma: no cover
        return "unknown"


def annotate(
    sheet_path: Path,
    raw_dir: Path,
    model: str = ANNOTATOR_MODEL,
    budget_usd: float = BUDGET_USD,
    only_missing: bool = True,
) -> float:
    """Annotate every constraint. Returns the total cost in USD."""
    import litellm

    litellm.suppress_debug_info = True
    load_env_keys()

    codebook = CODEBOOK_PATH.read_text()
    sha = codebook_sha256(codebook)
    constraints = load_constraints()
    sheet = read_sheet(sheet_path)
    objectives = {
        (r.brief_id, int(r.constraint_index)): r.objective for r in constraints.itertuples()
    }

    raw_dir.mkdir(parents=True, exist_ok=True)
    responses_path = raw_dir / "responses.jsonl"
    # Never append records from two codebooks to one file.
    archive_if_stale(responses_path, sha)
    done: set[tuple[str, int]] = set()
    if responses_path.exists():
        for line in responses_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["brief_id"], int(rec["constraint_index"])))

    total_cost = 0.0
    n_calls = 0
    model_ids_seen: set[str] = set()
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    with responses_path.open("a") as fh:
        for idx, row in sheet.iterrows():
            key = (row.brief_id, int(row.constraint_index))
            if only_missing and (key in done or row.label_llm):
                continue
            if total_cost >= budget_usd:
                log.warning("Budget of USD %.2f reached after %d calls; stopping.", budget_usd, n_calls)
                break
            system, user = build_prompt(codebook, row.constraint_text, objectives[key])
            if USE_PROMPT_CACHE:
                system_msg = {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}
            else:
                system_msg = {"role": "system", "content": system}
            messages = [system_msg, {"role": "user", "content": user}]
            resp = litellm.completion(model=model, messages=messages, temperature=0, max_tokens=300)
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            try:
                cost = float(litellm.completion_cost(completion_response=resp))
            except Exception as exc:  # pragma: no cover - pricing table gap
                log.warning("litellm cost lookup failed (%s); using fallback prices", exc)
                cost = usage.prompt_tokens * FALLBACK_PRICE_IN + usage.completion_tokens * FALLBACK_PRICE_OUT
            total_cost += cost
            n_calls += 1
            model_ids_seen.add(getattr(resp, "model", "") or "")
            cat, rat = parse_response(content)
            sheet.at[idx, "label_llm"] = cat
            sheet.at[idx, "llm_rationale"] = rat
            rec = {
                "brief_id": row.brief_id,
                "constraint_index": int(row.constraint_index),
                "constraint_text": row.constraint_text,
                "model_requested": model,
                "model_returned": getattr(resp, "model", None),
                "temperature": 0,
                "codebook_sha256": sha,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                **({"cache_read_tokens": cache_read_tokens(usage)} if USE_PROMPT_CACHE else {}),
                "cost_usd": cost,
                "cumulative_cost_usd": total_cost,
                "content": content,
                "label_llm": cat,
                "llm_rationale": rat,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log.info("[%d] %s/%d -> %s (call USD %.4f, total USD %.4f)", n_calls, *key, cat, cost, total_cost)
            # Persist the sheet as we go so a crash loses nothing.
            if n_calls % 10 == 0:
                write_sheet(sheet, sheet_path)

    write_sheet(sheet, sheet_path)
    n_labelled = int((sheet["label_llm"] != "").sum())
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    history = archived_passes(raw_dir, sheet)
    lines = [
        "# E8: LLM annotator for the constraint typology",
        "",
        "Command: `.venv/bin/python -m drift_bench.analysis.camera_ready.typology_annotate --annotate [--redo]`",
        "",
        "## Current pass",
        "",
        f"Started (UTC): {started}",
        "",
        f"Finished (UTC): {finished}",
        "",
        f"Model requested: `{model}`, temperature 0, max_tokens 300, one call per constraint.",
        "",
        f"Model id returned by the API: {sorted(m for m in model_ids_seen if m)}",
        "",
        f"litellm version: {_litellm_version()}",
        "",
        f"Codebook: `drift_bench/analysis/camera_ready/typology_codebook.md`, sha256 of the text sent in the system prompt: `{sha}`.",
        "",
        f"Calls this session: {n_calls}. Cost this session: USD {total_cost:.4f}. Budget: USD {budget_usd:.2f}.",
        "",
        f"Constraints with a label after this session: {n_labelled} of {len(sheet)}.",
        "",
        "Prompt caching: "
        + (
            "cache_control set on the system message; `cache_read_tokens` recorded per call."
            if USE_PROMPT_CACHE
            else "not attempted. Opus 4.6 caches only prefixes of at least 4096 tokens; the system prompt is about 1.6k tokens, "
            "so the first pass's cache_control marker never engaged (0 cache-read tokens on all calls). The marker and the field were removed."
        ),
        "",
        "Prompt: the system message is the codebook with a short instruction. The user message holds the brief objective and the constraint text. "
        "The annotator never saw the author labels.",
        "",
        "Files: `responses.jsonl` holds one record per call with `codebook_sha256`, token counts, cost, raw content, and the parsed label. "
        "The parsed labels are also in `drift_bench/data/camera_ready/analysis/typology_sheet.csv` (`label_llm`, `llm_rationale`).",
        "",
        "## Earlier passes",
        "",
    ]
    if not history:
        lines.append("None on disk.")
    for h in history:
        lines += [
            f"- `{h['file']}`: {h['n_calls']} calls, USD {h['cost_usd']:.4f}, codebook sha256 prefix `{h['codebook_sha256_prefix']}`"
            + (f" (records carry `{h['codebook_sha256_in_records']}`)" if h["codebook_sha256_in_records"] else " (records carry no sha; the prefix comes from the file name)")
            + f", {h['first_timestamp'][:19]} to {h['last_timestamp'][:19]} UTC. "
            f"Labels that differ from the current `label_llm`: {h['n_labels_changed_vs_current']}.",
        ]
        for b, i, old, new_ in h["changed"]:
            lines.append(f"    - {b}#{i}: {old} -> {new_}")
    lines += [
        "",
        f"Current pass: `responses.jsonl`, {n_calls} calls, USD {total_cost:.4f}, codebook sha256 `{sha}`, finished {finished[:19]} UTC.",
        "",
    ]
    (raw_dir / "README.md").write_text("\n".join(lines))
    return total_cost


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--annotate", action="store_true", help="make the Opus calls (costs money)")
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--redo", action="store_true", help="re-annotate rows that already have label_llm")
    args = ap.parse_args()

    run(args.data_root, args.out_dir)
    if args.annotate:
        raw_dir = Path(args.data_root) / "camera_ready" / "e8_typology_llm"
        cost = annotate(
            Path(args.out_dir) / "typology_sheet.csv",
            raw_dir,
            budget_usd=args.budget,
            only_missing=not args.redo,
        )
        print(f"Total annotation cost this session: USD {cost:.4f}")


if __name__ == "__main__":
    main()
