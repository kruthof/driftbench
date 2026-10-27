"""E7: strict LLM re-grading of the final restatement probes (plan items B4, C16).

The set. The final-turn probe of every clean ``multi_turn_pressure`` run of
four subjects: Claude Sonnet 4.6, GPT-5.4, Llama 3.3 70B and Gemini 3.1 Pro.
Each cell has 76 runs once the 30 stray Sonnet pilot runs are dropped
(``sample.stray_run_ids``), so the set has 304 probes. The final probe is the
last ``is_probe`` assistant message in the transcript. It follows turn 5; the
turn-6 final proposal has no probe. This matches the April rule
(``probes.compute_final_probe_accuracy`` takes the last probe per run).

The grader. ``anthropic/claude-opus-4-6`` at temperature 0 reads the brief's
objective, hard constraints and banned moves and the probe response, and
labels every item ``exact``, ``loose``, ``wrong`` or ``missing`` with a
quoted span as evidence. The rubric is ``prompts/probe_grader_strict.j2``.
The grader does not compute a score. Two rules are applied in code to the
same labels:

- strict: an item counts as recalled only when labelled ``exact``;
- paraphrase-tolerant (column prefix ``april_equiv``): an item counts when
  labelled ``exact`` or ``loose``. The April scorer
  (``probes.score_probe_accuracy``) counts an item when a key phrase of it
  appears in the response, which tolerates paraphrase around the phrase
  but not a missing item;
- lenient: ``exact``, ``loose`` or ``wrong``. On the 304 probes this is
  the closest analogue to the substring scorer (0.9407 against 0.9406).

``overall_accuracy`` under each rule is recalled items over all items
(hard constraints plus banned moves), the April formula.

Output: one JSON record per probe in ``<out_dir>/records.jsonl`` and one
line per invocation in ``<out_dir>/run_log.jsonl``. The script resumes: run
ids already on disk are skipped and a damaged last line is cut off first.

Usage::

    .venv/bin/python -m drift_bench.analysis.camera_ready.e7_regrade --dry-run
    .venv/bin/python -m drift_bench.analysis.camera_ready.e7_regrade --limit 3 --max-budget 1
    .venv/bin/python -m drift_bench.analysis.camera_ready.e7_regrade --max-budget 12
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from drift_bench.analysis.camera_ready.sample import stray_run_ids
from drift_bench.prompts.loader import render_template

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "drift_bench"
DEFAULT_DATA_ROOT = PACKAGE_DIR / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "e7_strict_probes"
BRIEFS_DIR = PACKAGE_DIR / "briefs"
CONFIG_PATH = PACKAGE_DIR / "config.yaml"
TRANSCRIPT_SUBDIRS = ("transcripts", "openweight_subjects/transcripts")

TEMPLATE_NAME = "probe_grader_strict.j2"
RECORDS_NAME = "records.jsonl"
RUN_LOG_NAME = "run_log.jsonl"

GRADER_MODEL = "anthropic/claude-opus-4-6"
GRADER_TEMPERATURE = 0.0
GRADER_MAX_TOKENS = 4096
MAX_CONCURRENCY = 4

CONDITION = "multi_turn_pressure"
MODELS = (
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "gemini/gemini-3.1-pro-preview",
)
EXPECTED_PER_MODEL = 76

# The April probe cap was 1,024 tokens; capped responses stop at 1,020.
PROBE_TRUNCATION_TOKENS = 1020

LABELS = ("exact", "loose", "wrong", "missing")
RULES = {
    "strict": ("exact",),
    "april_equiv": ("exact", "loose"),
    "lenient": ("exact", "loose", "wrong"),
}

# Dry-run assumptions. Input tokens are chars / 4. Output tokens per call are
# an assumption: about 45 tokens per graded item (label plus a quoted span)
# plus a fixed overhead for the JSON skeleton and the objective entry.
CHARS_PER_TOKEN = 4.0
ASSUMED_OUTPUT_TOKENS_PER_ITEM = 45
ASSUMED_OUTPUT_TOKENS_FIXED = 150


class GradeParseError(ValueError):
    """The grader's reply is not a valid grade."""


@dataclass
class ProbeItem:
    """One probe to grade."""

    run_id: str
    brief_id: str
    model_id: str
    condition: str
    turn_index: int
    message_index: int
    probe_token_count: int | None
    probe_response: str
    transcript_path: str

    @property
    def probe_truncated(self) -> bool:
        return (self.probe_token_count or 0) >= PROBE_TRUNCATION_TOKENS


# ---------------------------------------------------------------------------
# Environment and pricing
# ---------------------------------------------------------------------------


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ. Never print the values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_pricing(model_id: str = GRADER_MODEL, config_path: Path = CONFIG_PATH) -> dict[str, float]:
    """Per-token prices from config.yaml for the budget fallback and the dry run."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pricing = (cfg.get("pricing") or {}).get(model_id)
    if not pricing:
        raise KeyError(f"No pricing for {model_id} in {config_path}")
    return {
        "input_cost_per_token": float(pricing["input_cost_per_token"]),
        "output_cost_per_token": float(pricing["output_cost_per_token"]),
    }


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------


def load_transcript(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def load_briefs(briefs_dir: Path = BRIEFS_DIR) -> dict[str, dict]:
    briefs: dict[str, dict] = {}
    for path in sorted(briefs_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            brief = yaml.safe_load(f)
        briefs[brief["id"]] = brief
    return briefs


def select_final_probe(transcript: dict) -> dict | None:
    """Return the last probe of a transcript, or None when it has no probe.

    The result holds ``turn_index`` (non-probe user turns seen before the
    probe, as in ``probes.extract_probes``), ``message_index``,
    ``token_count`` and ``content``.
    """
    turn_idx = 0
    last: dict | None = None
    for i, msg in enumerate(transcript.get("messages", [])):
        if msg.get("role") == "user" and not msg.get("is_probe", False):
            turn_idx += 1
        if msg.get("is_probe", False) and msg.get("role") == "assistant":
            last = {
                "turn_index": turn_idx,
                "message_index": i,
                "token_count": msg.get("token_count"),
                "content": msg.get("content") or "",
            }
    return last


def select_probes(
    data_root: Path,
    models: tuple[str, ...] = MODELS,
    condition: str = CONDITION,
    stray: set[str] | None = None,
    transcript_subdirs: tuple[str, ...] = TRANSCRIPT_SUBDIRS,
) -> list[ProbeItem]:
    """Collect the final probe of every clean run in the chosen cells.

    Runs in ``stray`` are dropped. Runs without a probe are skipped and
    logged. The result is sorted by run_id.
    """
    if stray is None:
        stray = stray_run_ids(data_root)
    items: list[ProbeItem] = []
    for sub in transcript_subdirs:
        tdir = data_root / sub
        if not tdir.is_dir():
            continue
        for path in sorted(tdir.glob("*.jsonl")):
            try:
                t = load_transcript(path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Skipping unreadable transcript %s: %s", path.name, e)
                continue
            meta = t.get("metadata", {})
            if meta.get("condition") != condition or meta.get("model_id") not in models:
                continue
            run_id = meta.get("run_id")
            if run_id in stray:
                continue
            probe = select_final_probe(t)
            if probe is None:
                logger.warning("Run %s has no probe; skipped", run_id)
                continue
            items.append(ProbeItem(
                run_id=run_id,
                brief_id=meta.get("brief_id"),
                model_id=meta.get("model_id"),
                condition=meta.get("condition"),
                turn_index=probe["turn_index"],
                message_index=probe["message_index"],
                probe_token_count=probe["token_count"],
                probe_response=probe["content"],
                transcript_path=str(path),
            ))
    items.sort(key=lambda x: x.run_id)
    return items


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


def render_rubric(brief: dict, probe_response: str) -> str:
    """Render the strict grading prompt. Jinja runs with StrictUndefined."""
    return render_template(
        TEMPLATE_NAME,
        objective=brief["objective"],
        hard_constraints=list(brief["hard_constraints"]),
        banned_moves=list(brief["banned_moves"]),
        probe_response=probe_response,
    )


def rubric_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the grader's reply. Accepts a bare object or one inside a code fence."""
    if text is None:
        raise GradeParseError("empty reply")
    candidate = text.strip()
    m = _FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Last resort: the outermost braces.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise GradeParseError("no JSON object in reply")
    try:
        return json.loads(candidate[start:end + 1])
    except json.JSONDecodeError as e:
        raise GradeParseError(f"invalid JSON: {e}") from e


def _check_entry(entry: Any, where: str) -> dict:
    if not isinstance(entry, dict):
        raise GradeParseError(f"{where}: entry is not an object")
    label = entry.get("label")
    if label not in LABELS:
        raise GradeParseError(f"{where}: label {label!r} not in {LABELS}")
    evidence = entry.get("evidence", "")
    if evidence is None:
        evidence = ""
    if not isinstance(evidence, str):
        raise GradeParseError(f"{where}: evidence is not a string")
    return {"label": label, "evidence": evidence}


def _check_list(entries: Any, n: int, where: str) -> list[dict]:
    if not isinstance(entries, list):
        raise GradeParseError(f"{where}: not a list")
    if len(entries) != n:
        raise GradeParseError(f"{where}: expected {n} entries, got {len(entries)}")
    out = []
    for i, entry in enumerate(entries, start=1):
        checked = _check_entry(entry, f"{where}[{i}]")
        idx = entry.get("index")
        if idx != i:
            raise GradeParseError(f"{where}[{i}]: index {idx!r} is not {i}")
        checked["index"] = i
        out.append(checked)
    return out


def validate_grade(obj: Any, n_constraints: int, n_banned: int) -> dict:
    """Check the parsed reply against the schema. Returns a clean copy."""
    if not isinstance(obj, dict):
        raise GradeParseError("top level is not an object")
    for key in ("objective", "hard_constraints", "banned_moves"):
        if key not in obj:
            raise GradeParseError(f"missing key {key!r}")
    grade = {
        "objective": _check_entry(obj["objective"], "objective"),
        "hard_constraints": _check_list(obj["hard_constraints"], n_constraints, "hard_constraints"),
        "banned_moves": _check_list(obj["banned_moves"], n_banned, "banned_moves"),
    }
    extra = obj.get("extra_items", [])
    if extra is None:
        extra = []
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise GradeParseError("extra_items is not a list of strings")
    grade["extra_items"] = extra
    cut = obj.get("response_cut_off", False)
    if not isinstance(cut, bool):
        raise GradeParseError("response_cut_off is not a boolean")
    grade["response_cut_off"] = cut
    return grade


def parse_grade(text: str, n_constraints: int, n_banned: int) -> dict:
    return validate_grade(extract_json(text), n_constraints, n_banned)


def _recall(labels: list[str], counted: tuple[str, ...]) -> float:
    if not labels:
        return 1.0
    return sum(1 for x in labels if x in counted) / len(labels)


def score_grade(grade: dict) -> dict:
    """Apply the strict, paraphrase-tolerant (april_equiv) and lenient rules to one grade.

    Returns ``{rule}_overall_accuracy``, ``{rule}_constraint_recall`` and
    ``{rule}_banned_recall`` for each rule, plus label counts. The overall
    accuracy is recalled items over all items, as in April.
    """
    c_labels = [e["label"] for e in grade["hard_constraints"]]
    b_labels = [e["label"] for e in grade["banned_moves"]]
    out: dict[str, Any] = {
        "n_constraints": len(c_labels),
        "n_banned": len(b_labels),
        "objective_label": grade["objective"]["label"],
        "n_extra_items": len(grade.get("extra_items", [])),
        "response_cut_off": bool(grade.get("response_cut_off", False)),
    }
    for label in LABELS:
        out[f"n_constraint_{label}"] = c_labels.count(label)
        out[f"n_banned_{label}"] = b_labels.count(label)
    for rule, counted in RULES.items():
        out[f"{rule}_overall_accuracy"] = _recall(c_labels + b_labels, counted)
        out[f"{rule}_constraint_recall"] = _recall(c_labels, counted)
        out[f"{rule}_banned_recall"] = _recall(b_labels, counted)
    return out


# ---------------------------------------------------------------------------
# Records on disk
# ---------------------------------------------------------------------------


def read_records(path: Path, repair: bool = True) -> tuple[list[dict], int]:
    """Read records.jsonl. A damaged last line is cut off when ``repair`` is set.

    Returns the records and the number of bytes removed by the repair.
    """
    if not path.is_file():
        return [], 0
    raw = path.read_bytes()
    records: list[dict] = []
    good_end = 0
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl == -1:
            chunk, next_pos, complete = raw[pos:], len(raw), False
        else:
            chunk, next_pos, complete = raw[pos:nl], nl + 1, True
        text = chunk.decode("utf-8", errors="replace").strip()
        if text:
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                if complete:
                    logger.warning("Damaged line in %s at byte %d; stopping read", path, pos)
                break
        if not complete:
            break
        good_end = next_pos
        pos = next_pos
    removed = len(raw) - good_end
    if removed and repair:
        with open(path, "r+b") as f:
            f.truncate(good_end)
        logger.warning("Repaired %s: removed %d trailing bytes", path, removed)
    elif removed and not repair:
        # Nothing was changed on disk; report what a repair would remove.
        pass
    return records, removed


def done_run_ids(records: list[dict], retry_failed: bool = False) -> set[str]:
    """Run ids to skip. With ``retry_failed`` only parsed records count."""
    if retry_failed:
        return {r["run_id"] for r in records if r.get("parse_ok")}
    return {r["run_id"] for r in records}


def write_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def rewrite_records(path: Path, records: list[dict]) -> None:
    """Rewrite the file without the records being retried."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def estimate_item_cost(prompt: str, n_items: int, pricing: dict[str, float]) -> dict[str, float]:
    in_tok = len(prompt) / CHARS_PER_TOKEN
    out_tok = ASSUMED_OUTPUT_TOKENS_FIXED + ASSUMED_OUTPUT_TOKENS_PER_ITEM * (n_items + 1)
    cost = in_tok * pricing["input_cost_per_token"] + out_tok * pricing["output_cost_per_token"]
    return {"input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost}


def estimate_total(items: list[ProbeItem], briefs: dict[str, dict], pricing: dict[str, float]) -> dict:
    tot_in = tot_out = tot_cost = 0.0
    for it in items:
        brief = briefs[it.brief_id]
        prompt = render_rubric(brief, it.probe_response)
        n_items = len(brief["hard_constraints"]) + len(brief["banned_moves"])
        est = estimate_item_cost(prompt, n_items, pricing)
        tot_in += est["input_tokens"]
        tot_out += est["output_tokens"]
        tot_cost += est["cost_usd"]
    return {
        "n_calls": len(items),
        "est_input_tokens": int(tot_in),
        "est_output_tokens": int(tot_out),
        "est_cost_usd": tot_cost,
        "assumptions": {
            "chars_per_token": CHARS_PER_TOKEN,
            "output_tokens_per_item": ASSUMED_OUTPUT_TOKENS_PER_ITEM,
            "output_tokens_fixed": ASSUMED_OUTPUT_TOKENS_FIXED,
            "pricing": pricing,
        },
    }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class Budget:
    """Running spend with a hard cap. Checked inside the semaphore."""

    def __init__(self, max_usd: float, spent: float = 0.0):
        self.max_usd = max_usd
        self.spent = spent
        self.exhausted = False

    def allows(self, next_cost: float) -> bool:
        if self.spent + next_cost > self.max_usd:
            self.exhausted = True
            return False
        return True

    def add(self, cost: float) -> None:
        self.spent += cost


async def call_grader(
    prompt: str,
    model_id: str = GRADER_MODEL,
    temperature: float = GRADER_TEMPERATURE,
    max_tokens: int = GRADER_MAX_TOKENS,
    pricing: dict[str, float] | None = None,
    attempts: int = 3,
) -> dict:
    """One grader call with simple retry. Returns content, usage, cost, model."""
    import litellm  # imported here so tests can patch litellm.acompletion

    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            start = time.monotonic()
            response = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - start) * 1000
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Grader call failed (attempt %d/%d): %s", attempt, attempts, e)
            if attempt < attempts:
                await asyncio.sleep(2.0 * attempt)
    else:
        raise RuntimeError(f"Grader call failed after {attempts} attempts: {last_err}")

    usage = response.usage
    cost = 0.0
    cost_estimated = False
    try:
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("Cost lookup failed: %s", e)
    if not cost and pricing:
        cost = (
            (usage.prompt_tokens or 0) * pricing["input_cost_per_token"]
            + (usage.completion_tokens or 0) * pricing["output_cost_per_token"]
        )
        cost_estimated = True
    choice = response.choices[0]
    return {
        "content": choice.message.content or "",
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost,
        "cost_estimated": cost_estimated,
        "latency_ms": latency_ms,
        "finish_reason": getattr(choice, "finish_reason", None),
        "response_model": getattr(response, "model", None),
    }


def build_record(item: ProbeItem, brief: dict, prompt: str, reply: dict, grader_model: str) -> dict:
    n_c = len(brief["hard_constraints"])
    n_b = len(brief["banned_moves"])
    record: dict[str, Any] = {
        **asdict(item),
        "probe_truncated": item.probe_truncated,
        "grader_model_requested": grader_model,
        "grader_model_reported": reply.get("response_model"),
        "grader_temperature": GRADER_TEMPERATURE,
        "grader_max_tokens": GRADER_MAX_TOKENS,
        "template": TEMPLATE_NAME,
        "rubric_sha256": rubric_sha256(prompt),
        "input_tokens": reply.get("input_tokens"),
        "output_tokens": reply.get("output_tokens"),
        "cost_usd": reply.get("cost_usd"),
        "cost_estimated": reply.get("cost_estimated", False),
        "latency_ms": reply.get("latency_ms"),
        "finish_reason": reply.get("finish_reason"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_text": reply.get("content", ""),
    }
    try:
        grade = parse_grade(record["raw_text"], n_c, n_b)
        record["parse_ok"] = True
        record["parse_error"] = None
        record["grade"] = grade
        record["scores"] = score_grade(grade)
    except GradeParseError as e:
        record["parse_ok"] = False
        record["parse_error"] = str(e)
        record["grade"] = None
        record["scores"] = None
    return record


async def grade_items(
    items: list[ProbeItem],
    briefs: dict[str, dict],
    records_path: Path,
    budget: Budget,
    pricing: dict[str, float],
    grader_model: str = GRADER_MODEL,
    concurrency: int = MAX_CONCURRENCY,
) -> dict:
    """Grade every item, at most ``concurrency`` at a time, under the budget."""
    sem = asyncio.Semaphore(min(concurrency, MAX_CONCURRENCY))
    lock = asyncio.Lock()
    stats = {"graded": 0, "parse_failed": 0, "call_failed": 0, "skipped_budget": 0, "cost_usd": 0.0}

    async def one(item: ProbeItem) -> None:
        brief = briefs[item.brief_id]
        prompt = render_rubric(brief, item.probe_response)
        n_items = len(brief["hard_constraints"]) + len(brief["banned_moves"])
        est = estimate_item_cost(prompt, n_items, pricing)["cost_usd"]
        async with sem:
            if budget.exhausted or not budget.allows(est):
                stats["skipped_budget"] += 1
                return
            try:
                reply = await call_grader(prompt, model_id=grader_model, pricing=pricing)
            except Exception as e:  # noqa: BLE001
                logger.error("Run %s: grader failed: %s", item.run_id, e)
                stats["call_failed"] += 1
                return
            budget.add(reply["cost_usd"])
            record = build_record(item, brief, prompt, reply, grader_model)
            async with lock:
                write_record(records_path, record)
            stats["graded"] += 1
            stats["cost_usd"] += reply["cost_usd"]
            if not record["parse_ok"]:
                stats["parse_failed"] += 1
                logger.warning("Run %s: parse failed: %s", item.run_id, record["parse_error"])
            else:
                s = record["scores"]
                logger.info(
                    "Run %s %s: strict %.2f april_equiv %.2f cost %.4f (spent %.2f)",
                    item.run_id, item.model_id.split("/")[-1][:20],
                    s["strict_overall_accuracy"], s["april_equiv_overall_accuracy"],
                    reply["cost_usd"], budget.spent,
                )

    await asyncio.gather(*(one(it) for it in items))
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summarise_selection(items: list[ProbeItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it.model_id] = counts.get(it.model_id, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-budget", type=float, default=None, help="USD cap for this invocation (required unless --dry-run)")
    parser.add_argument("--limit", type=int, default=None, help="grade at most N probes (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="print the selection and a token-based cost estimate; no API call")
    parser.add_argument("--retry-failed", action="store_true", help="re-grade records whose reply did not parse")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--grader", default=GRADER_MODEL)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_utc = dt.datetime.now(dt.timezone.utc)

    briefs = load_briefs()
    pricing = load_pricing(args.grader)
    stray = stray_run_ids(args.data_root)
    items = select_probes(args.data_root, stray=stray)
    counts = _summarise_selection(items)
    print(f"Selected {len(items)} probes ({CONDITION}, final probe, stray Sonnet runs dropped: {len(stray)})")
    for m, n in sorted(counts.items()):
        flag = "" if n == EXPECTED_PER_MODEL else f"  (expected {EXPECTED_PER_MODEL})"
        print(f"  {m}: {n}{flag}")
    n_trunc = sum(1 for it in items if it.probe_truncated)
    print(f"  probes with token_count >= {PROBE_TRUNCATION_TOKENS}: {n_trunc}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / RECORDS_NAME
    existing, removed = read_records(records_path, repair=not args.dry_run)
    done = done_run_ids(existing, retry_failed=args.retry_failed)
    pending = [it for it in items if it.run_id not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"On disk: {len(existing)} records ({removed} damaged trailing bytes{' removed' if removed and not args.dry_run else ''}); pending: {len(pending)}")

    est = estimate_total(pending, briefs, pricing)
    print(f"Estimate for pending: {est['n_calls']} calls, ~{est['est_input_tokens']:,} input tokens, "
          f"~{est['est_output_tokens']:,} output tokens, ~USD {est['est_cost_usd']:.2f} "
          f"(chars/{CHARS_PER_TOKEN:g} per token; {ASSUMED_OUTPUT_TOKENS_FIXED} + {ASSUMED_OUTPUT_TOKENS_PER_ITEM} per item output tokens)")

    if args.dry_run:
        return 0
    if args.max_budget is None:
        parser.error("--max-budget is required unless --dry-run")
    if not pending:
        print("Nothing to do.")
        return 0

    load_env()
    if args.retry_failed:
        keep = [r for r in existing if r.get("parse_ok")]
        if len(keep) != len(existing):
            rewrite_records(records_path, keep)
            print(f"Removed {len(existing) - len(keep)} unparsed records for retry")

    budget = Budget(args.max_budget)
    stats = asyncio.run(grade_items(pending, briefs, records_path, budget, pricing,
                                    grader_model=args.grader, concurrency=args.concurrency))
    end_utc = dt.datetime.now(dt.timezone.utc)
    try:
        from importlib.metadata import version
        litellm_version = version("litellm")
    except Exception:  # noqa: BLE001
        litellm_version = None
    log = {
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "argv": argv if argv is not None else os.sys.argv[1:],
        "grader_model": args.grader,
        "temperature": GRADER_TEMPERATURE,
        "max_tokens": GRADER_MAX_TOKENS,
        "litellm_version": litellm_version,
        "selected": len(items),
        "selection_counts": counts,
        "pending": len(pending),
        "estimate": est,
        "max_budget_usd": args.max_budget,
        **stats,
    }
    write_record(args.out_dir / RUN_LOG_NAME, log)
    print(json.dumps({k: v for k, v in log.items() if k not in ("estimate", "argv")}, indent=2))
    if budget.exhausted:
        print(f"STOPPED: budget cap USD {args.max_budget:.2f} reached after USD {budget.spent:.2f}; "
              f"{stats['skipped_budget']} probes not graded. Re-run to resume.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
