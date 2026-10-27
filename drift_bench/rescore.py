"""Re-score stored transcripts with a chosen judge.

Camera-ready tool for the crossed-judge (E1, E4) and per-turn (E5)
experiments. It reads transcripts from disk, scores them with a judge
pipeline, and writes one score file per run. It never calls the subject
models.

Directory convention (``--scores-dir`` is required; there is no default).
One prefix per directory:

    E1 crossed judge   drift_bench/data/camera_ready/e1_crossed_judge/opus/
                           prefix judge_opus  (Opus on Sonnet)
                       drift_bench/data/camera_ready/e1_crossed_judge/gpt/
                           prefix judge_gpt   (GPT-5.4 on the other six)
    E4 structured      drift_bench/data/camera_ready/e4_structured_opus/
                           prefix structured_opus, --mode structured
    E5 per turn        drift_bench/data/camera_ready/e5_per_turn/opus/
                           files judge_opus_t<K>_<run_id>.jsonl, --turn-indices 1 2 3 4 5

Per-turn files and final files never share a directory: the glob
``judge_opus_*`` also matches ``judge_opus_t3_*``. Never run
``aggregate_scores`` on an E1 directory: it globs ``judge_*.jsonl`` and
would take a crossed prefix for the published arm.
The tool refuses to write into the published score directories
(data/scores, data/openweight_subjects/scores, data/scores_skeptical,
data/monitored/scores), or next to files of the published shape
``<published prefix>_<12-hex run_id>.jsonl``, unless
``--allow-published-dir`` is given. Files of another crossed prefix
(``judge_opus_*`` next to ``judge_gpt_*``) are not a collision.

``--dry-run`` loads and filters the transcripts, prints the task count
and a token-based cost estimate from the config ``pricing`` map, makes
no network call and writes nothing (no score files, no summary record).
Estimate: input tokens = len(text the judge sees) / 4 + PROMPT_OVERHEAD_TOKENS;
output tokens = ASSUMED_OUTPUT_TOKENS[mode].

Usage:
    python -m drift_bench.rescore \\
        --transcripts-dir drift_bench/data/transcripts drift_bench/data/openweight_subjects/transcripts \\
        --scores-dir drift_bench/data/camera_ready/e1_crossed_judge/opus \\
        --judge-override anthropic/claude-opus-4-6 --output-prefix judge_opus \\
        --mode judge --conditions single_shot multi_turn_pressure \\
        --models anthropic/claude-sonnet-4-6 --max-budget 50 --concurrency 5

Resumable: a run whose output file exists and loads is skipped; a corrupt
file is re-scored. Every record carries ``judge_model`` (the id requested)
and ``judge_response_model`` (the id the API reported). Each invocation
appends a record to ``<scores_dir>/rescore_summary_<prefix>[_t<K..>].json``.

Cost guard: a successful call that reports zero cost is counted and
warned about. After more than ZERO_COST_LIMIT such calls the run aborts,
because the budget cap would be blind. config.yaml ``pricing`` gives a
token-based fallback estimate for the cap (see runners/base.py).
"""

import argparse
import asyncio
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import jsonlines

from drift_bench.judges.judge import (
    JudgePipeline,
    JudgeScore,
    score_from_record,
    score_output_path,
    truncate_messages_at_turn,
)
from drift_bench.judges.judge_blind import BlindJudgePipeline, BlindJudgeScore
from drift_bench.judges.judge_structured import StructuredJudgePipeline, StructuredJudgeScore
from drift_bench.runners.base import LITELLM_VERSION, Transcript
from drift_bench.schema.validate import load_all_briefs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DEFAULT_CONFIG = BASE_DIR / "config.yaml"
DEFAULT_BRIEFS = BASE_DIR / "briefs"
DEFAULT_TRANSCRIPTS = BASE_DIR / "data" / "transcripts"

MODES = {
    "judge": (JudgePipeline, JudgeScore),
    "structured": (StructuredJudgePipeline, StructuredJudgeScore),
    "blind": (BlindJudgePipeline, BlindJudgeScore),
}

# Prefixes that aggregate.py and the published analyses glob on.
PUBLISHED_PREFIXES = ("judge", "auditor", "structured", "blind", "skeptical")

# Score directories that hold the published arms. Refuse to write there.
PUBLISHED_SCORE_DIRS = (
    BASE_DIR / "data" / "scores",
    BASE_DIR / "data" / "openweight_subjects" / "scores",
    BASE_DIR / "data" / "scores_skeptical",
    BASE_DIR / "data" / "monitored" / "scores",
)

# Abort after more than this many successful calls with zero reported cost.
ZERO_COST_LIMIT = 5

# A published score file: <prefix>_<run_id>.jsonl with a 12-hex run_id and
# nothing else between prefix and suffix (RunMetadata.run_id = uuid4().hex[:12]).
PUBLISHED_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# --dry-run cost model. Rough on purpose; it feeds the budget cap only.
CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_TOKENS = 3000  # brief + rubrics + calibration examples + instructions
ASSUMED_OUTPUT_TOKENS = {"judge": 700, "structured": 1500, "blind": 600}


class RescoreRefused(RuntimeError):
    """Raised when the output directory or prefix would corrupt published data."""


def load_run_ids_file(path: Path) -> set[str]:
    """Read run ids from a text file: one id per line, blank lines and ``#`` comments ignored."""
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            token = line.split("#", 1)[0].strip()
            if token:
                ids.add(token)
    return ids


def load_transcript_dicts(
    transcripts_dirs: Path | list[Path],
    conditions: list[str] | None = None,
    models: list[str] | None = None,
    run_ids: set[str] | None = None,
    exclude_run_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load transcript records from one or more directories, filtered by condition and model.

    ``run_ids`` keeps only those runs (an include list); ``exclude_run_ids``
    drops those runs (E1 uses it for the 30 stray Sonnet runs). An empty
    set means no filter.
    """
    if isinstance(transcripts_dirs, (str, Path)):
        transcripts_dirs = [Path(transcripts_dirs)]
    out: list[dict[str, Any]] = []
    for transcripts_dir in transcripts_dirs:
        for path in sorted(Path(transcripts_dir).glob("*.jsonl")):
            try:
                with jsonlines.open(path) as reader:
                    record = next(iter(reader))
            except Exception as e:
                logger.warning(f"Skipping unreadable transcript {path.name}: {e}")
                continue
            meta = record.get("metadata", {})
            if conditions and meta.get("condition") not in conditions:
                continue
            if models and meta.get("model_id") not in models:
                continue
            if run_ids and meta.get("run_id") not in run_ids:
                continue
            if exclude_run_ids and meta.get("run_id") in exclude_run_ids:
                continue
            out.append(record)
    return out


def is_published_scores_dir(scores_dir: Path) -> bool:
    """True if scores_dir resolves to one of the published score directories."""
    resolved = Path(scores_dir).resolve()
    return any(resolved == d.resolve() for d in PUBLISHED_SCORE_DIRS)


def _is_per_turn_file(name: str, prefix: str) -> bool:
    """True for ``<prefix>_t<K>_<run_id>.jsonl``."""
    rest = name[len(prefix) + 1:]
    if not rest.startswith("t"):
        return False
    digits, sep, _ = rest[1:].partition("_")
    return bool(sep) and digits.isdigit()


def is_published_shape(name: str, base: str) -> bool:
    """True for ``<base>_<run_id>.jsonl`` with a 12-hex run_id and nothing else."""
    if not name.startswith(f"{base}_") or not name.endswith(".jsonl"):
        return False
    return bool(PUBLISHED_RUN_ID_RE.match(name[len(base) + 1:-len(".jsonl")]))


def prefix_collides_with_published(prefix: str, scores_dir: Path) -> str | None:
    """Return the published prefix whose aggregate glob would catch this prefix, if any.

    Only files of the published shape ``<base>_<12-hex run_id>.jsonl``
    count as foreign. Files of this prefix (``judge_opus_<id>``), of a
    sibling crossed prefix (``judge_gpt_<id>``) or per-turn files
    (``judge_opus_t2_<id>``) all have more than one segment after the
    base and are not a collision.
    """
    for base in PUBLISHED_PREFIXES:
        if prefix == base or not prefix.startswith(base):
            continue
        if any(is_published_shape(path.name, base) for path in scores_dir.glob(f"{base}_*.jsonl")):
            return base
    return None


def per_turn_collision(prefix: str, scores_dir: Path, per_turn: bool) -> str | None:
    """Describe a final/per-turn mix for this prefix in scores_dir, or None.

    A per-turn run must not sit next to final ``<prefix>_<run_id>.jsonl``
    files, and a final run must not sit next to ``<prefix>_t<K>_*`` files.
    Files of the same kind are fine; that is what resumability relies on.
    """
    existing = [p.name for p in scores_dir.glob(f"{prefix}_*.jsonl")]
    finals = [n for n in existing if not _is_per_turn_file(n, prefix)]
    turns = [n for n in existing if _is_per_turn_file(n, prefix)]
    if per_turn and finals:
        return f"{len(finals)} final {prefix}_*.jsonl file(s) already in {scores_dir}"
    if not per_turn and turns:
        return f"{len(turns)} per-turn {prefix}_t<K>_*.jsonl file(s) already in {scores_dir}"
    return None


def check_output_dir(
    scores_dir: Path, prefix: str, per_turn: bool, allow_published_dir: bool = False,
) -> None:
    """Refuse output locations that would corrupt or confuse published data."""
    problems: list[str] = []
    if is_published_scores_dir(scores_dir):
        problems.append(f"{scores_dir} is a published scores directory")
    if scores_dir.exists():
        base = prefix_collides_with_published(prefix, scores_dir)
        if base:
            problems.append(
                f"prefix {prefix!r} would be caught by the aggregate glob for published "
                f"{base!r} files in {scores_dir}"
            )
        mix = per_turn_collision(prefix, scores_dir, per_turn)
        if mix:
            problems.append(mix)
    if not problems:
        return
    message = "Refusing to write: " + "; ".join(problems)
    if allow_published_dir:
        logger.warning(message + " (continuing because --allow-published-dir was given)")
        return
    raise RescoreRefused(message + ". Use a separate --scores-dir, or pass --allow-published-dir.")


def summary_path(scores_dir: Path, prefix: str, turn_indices: list[int] | None) -> Path:
    suffix = f"_t{'-'.join(str(k) for k in turn_indices)}" if turn_indices else ""
    return scores_dir / f"rescore_summary_{prefix}{suffix}.json"


def append_summary(path: Path, record: dict[str, Any]) -> None:
    """Append one invocation record to a JSON list file."""
    records: list[dict[str, Any]] = []
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
            records = loaded if isinstance(loaded, list) else [loaded]
        except Exception as e:
            logger.warning(f"Could not read {path}; starting a new summary list: {e}")
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


async def rescore(
    transcripts_dirs: Path | list[Path],
    scores_dir: Path,
    judge_override: str | None,
    output_prefix: str,
    mode: str = "judge",
    conditions: list[str] | None = None,
    models: list[str] | None = None,
    turn_indices: list[int] | None = None,
    max_budget: float | None = None,
    concurrency: int = 5,
    config_path: Path = DEFAULT_CONFIG,
    briefs_dir: Path = DEFAULT_BRIEFS,
    prompts_dir: Path | None = None,
    rubrics_path: Path | None = None,
    calibration_path: Path | None = None,
    limit: int | None = None,
    allow_published_dir: bool = False,
    argv: list[str] | None = None,
    dry_run: bool = False,
    run_ids: set[str] | None = None,
    exclude_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Score every matching transcript. Returns the summary dict it also writes.

    With ``dry_run`` it only counts tasks and estimates cost; it makes no
    network call and writes nothing.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(MODES)}")
    pipeline_cls, score_cls = MODES[mode]
    scores_dir = Path(scores_dir)
    turn_list: list[int | None] = list(turn_indices) if turn_indices else [None]

    check_output_dir(scores_dir, output_prefix, per_turn=bool(turn_indices), allow_published_dir=allow_published_dir)

    pipeline = pipeline_cls(
        config_path,
        judge_override=judge_override,
        output_prefix=output_prefix,
        prompts_dir=prompts_dir,
        rubrics_path=rubrics_path,
        calibration_path=calibration_path,
    )

    briefs_by_id = {b["id"]: b for b in load_all_briefs(briefs_dir)}
    records = load_transcript_dicts(transcripts_dirs, conditions, models, run_ids, exclude_run_ids)
    if limit is not None:
        records = records[:limit]
    logger.info(f"Loaded {len(records)} transcripts from {transcripts_dirs}")

    if dry_run:
        return dry_run_estimate(pipeline, records, scores_dir, output_prefix, mode, turn_list)

    scores_dir.mkdir(parents=True, exist_ok=True)

    budget_cap = max_budget if max_budget is not None else float("inf")
    budget_spent = 0.0
    semaphore = asyncio.Semaphore(concurrency)
    counts = {
        "scored": 0, "skipped_existing": 0, "failed": 0,
        "budget_stopped": 0, "zero_cost": 0, "aborted": 0,
    }
    judge_ids_used: set[str] = set()
    response_models_used: set[str] = set()
    n_cost_estimated = 0
    abort = False

    async def score_one(record: dict[str, Any], turn_index: int | None) -> None:
        nonlocal budget_spent, abort, n_cost_estimated
        meta = record["metadata"]
        run_id = meta["run_id"]
        out_path = score_output_path(scores_dir, output_prefix, run_id, turn_index)
        if out_path.exists():
            # Skip only if the file loads; a corrupt file is re-scored.
            try:
                with jsonlines.open(out_path) as reader:
                    score_from_record(score_cls, next(iter(reader)))
                counts["skipped_existing"] += 1
                return
            except Exception as e:
                logger.warning(f"Re-scoring {run_id} turn={turn_index}: unreadable {out_path.name} ({e})")
        brief = briefs_by_id.get(meta["brief_id"])
        if brief is None:
            logger.error(f"Brief not found for {run_id}: {meta['brief_id']}")
            counts["failed"] += 1
            return
        # JudgePipeline takes Transcript objects; the others take raw dicts.
        subject: Any = Transcript.from_dict(record) if mode == "judge" else record

        async with semaphore:
            if abort:
                counts["aborted"] += 1
                return
            if budget_spent >= budget_cap:
                counts["budget_stopped"] += 1
                return
            try:
                score = await pipeline.score_transcript(
                    subject, brief, scores_dir, turn_index=turn_index,
                )
            except Exception as e:
                logger.error(f"Failed {run_id} turn={turn_index}: {e}")
                counts["failed"] += 1
                return
            cost = float(getattr(score, "cost_usd", 0.0) or 0.0)
            budget_spent += cost
            counts["scored"] += 1
            if getattr(score, "cost_estimated", False):
                n_cost_estimated += 1
            judge_ids_used.add(score.judge_model)
            if getattr(score, "judge_response_model", None):
                response_models_used.add(score.judge_response_model)
            if cost == 0.0:
                counts["zero_cost"] += 1
                logger.warning(
                    f"Zero cost reported for {run_id} with {score.judge_model} "
                    f"({counts['zero_cost']}/{ZERO_COST_LIMIT} allowed); the budget cap cannot see this call"
                )
                if counts["zero_cost"] > ZERO_COST_LIMIT:
                    abort = True
                    logger.error(
                        f"More than {ZERO_COST_LIMIT} zero-cost calls: aborting. Add the judge to the "
                        f"config.yaml `pricing` map so the cap can estimate cost from tokens."
                    )
            logger.info(
                f"Scored {run_id} turn={turn_index} with {score.judge_model} "
                f"(${budget_spent:.2f}/{budget_cap:.2f})"
            )

    tasks = [score_one(r, k) for k in turn_list for r in records]
    logger.info(
        f"Dispatching {len(tasks)} scoring tasks (mode={mode}, prefix={output_prefix}, "
        f"judge={judge_override or 'config routing'}, turns={turn_list})"
    )
    await asyncio.gather(*tasks)

    summary = {
        "argv": list(argv) if argv is not None else list(sys.argv),
        "utc_date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "litellm_version": LITELLM_VERSION,
        "mode": mode,
        "judge_override": judge_override,
        "judge_models_used": sorted(judge_ids_used),
        "judge_response_models": sorted(response_models_used),
        "output_prefix": output_prefix,
        "scores_dir": str(scores_dir),
        "transcripts_dirs": [str(d) for d in (transcripts_dirs if isinstance(transcripts_dirs, list) else [transcripts_dirs])],
        "conditions": conditions,
        "models": models,
        "run_ids": sorted(run_ids) if run_ids else None,
        "exclude_run_ids": sorted(exclude_run_ids) if exclude_run_ids else None,
        "transcripts": len(records),
        "turn_indices": turn_list,
        "max_budget_usd": max_budget,
        "budget_spent_usd": budget_spent,
        "n_cost_estimated": n_cost_estimated,
        "aborted_zero_cost": abort,
        **counts,
    }
    append_summary(summary_path(scores_dir, output_prefix, turn_indices), summary)
    logger.info(f"Done: {summary}")
    return summary


def _judge_input_text(record: dict[str, Any], mode: str, turn_index: int | None) -> str:
    """Text the judge would see from the transcript (not the fixed prompt parts)."""
    messages = record.get("messages", [])
    if turn_index is not None:
        messages = truncate_messages_at_turn(messages, turn_index)
    non_probe = [m for m in messages if not m.get("is_probe", False) and m.get("role") != "system"]
    if mode == "judge":
        # Full transcript plus the final proposal repeated.
        text = "".join(m.get("content", "") for m in non_probe)
        finals = [m for m in non_probe if m.get("role") == "assistant"]
        return text + (finals[-1].get("content", "") if finals else "")
    finals = [m for m in non_probe if m.get("role") == "assistant"]
    return finals[-1].get("content", "") if finals else ""


def dry_run_estimate(
    pipeline: Any,
    records: list[dict[str, Any]],
    scores_dir: Path,
    output_prefix: str,
    mode: str,
    turn_list: list[int | None],
) -> dict[str, Any]:
    """Count tasks and estimate cost from the config ``pricing`` map. No I/O."""
    pricing = pipeline.config.get("pricing") or {}
    out_tokens = ASSUMED_OUTPUT_TOKENS.get(mode, 700)
    tasks = 0
    skipped = 0
    unpriced: set[str] = set()
    judges: set[str] = set()
    est_input = 0
    est_output = 0
    est_cost = 0.0
    for turn_index in turn_list:
        for record in records:
            meta = record["metadata"]
            if score_output_path(scores_dir, output_prefix, meta["run_id"], turn_index).exists():
                skipped += 1
                continue
            try:
                text = _judge_input_text(record, mode, turn_index)
            except ValueError:
                continue  # fewer turns than turn_index: score_transcript would fail too
            tasks += 1
            judge_model = pipeline._select_judge_model(meta["model_id"])
            judges.add(judge_model)
            in_tokens = len(text) // CHARS_PER_TOKEN + PROMPT_OVERHEAD_TOKENS
            est_input += in_tokens
            est_output += out_tokens
            price = pricing.get(judge_model)
            if not price:
                unpriced.add(judge_model)
                continue
            est_cost += (in_tokens * float(price.get("input_cost_per_token", 0.0))
                         + out_tokens * float(price.get("output_cost_per_token", 0.0)))
    summary = {
        "dry_run": True,
        "mode": mode,
        "output_prefix": output_prefix,
        "scores_dir": str(scores_dir),
        "transcripts": len(records),
        "turn_indices": turn_list,
        "tasks": tasks,
        "skipped_existing": skipped,
        "judge_models": sorted(judges),
        "unpriced_judge_models": sorted(unpriced),
        "assumed_prompt_overhead_tokens": PROMPT_OVERHEAD_TOKENS,
        "assumed_output_tokens": out_tokens,
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_cost_usd": est_cost,
    }
    print(
        f"DRY RUN: {tasks} task(s) to score ({skipped} already on disk), judges {sorted(judges)}.\n"
        f"  Estimate: {est_input:,} input tokens (len(text)/{CHARS_PER_TOKEN} + {PROMPT_OVERHEAD_TOKENS} overhead each), "
        f"{est_output:,} output tokens ({out_tokens} each) -> about ${est_cost:.2f}."
        + (f"\n  No price in config.yaml `pricing` for: {sorted(unpriced)}" if unpriced else "")
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-score stored transcripts with a chosen judge.")
    parser.add_argument("--transcripts-dir", type=Path, nargs="+", default=[DEFAULT_TRANSCRIPTS],
                        help="One or more transcript directories (default: drift_bench/data/transcripts)")
    parser.add_argument("--scores-dir", type=Path, required=True,
                        help="Output directory. Required. Use a fresh camera_ready subdirectory "
                             "(see module docstring); published score directories are refused.")
    parser.add_argument("--judge-override", type=str, default=None,
                        help="Judge model id for every run (default: config cross-family routing)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Score file prefix (default: the mode's prefix: judge/structured/blind)")
    parser.add_argument("--mode", choices=sorted(MODES), default="judge")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--turn-index", type=int, default=None,
                        help="Score the transcript truncated after assistant turn K. K counts every "
                             "non-probe assistant message, including checkpoint replies in "
                             "checkpointed_pressure (there the probes sit after turns 1, 2, 4, 5, 7).")
    parser.add_argument("--turn-indices", type=int, nargs="+", default=None,
                        help="Loop over several turn indices (same counting rule as --turn-index)")
    parser.add_argument("--max-budget", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--run-ids-file", type=Path, default=None,
                        help="Score only the run ids listed in this file (one per line, # comments)")
    parser.add_argument("--exclude-run-ids-file", type=Path, default=None,
                        help="Skip the run ids listed in this file (one per line, # comments)")
    parser.add_argument("--limit", type=int, default=None, help="Score at most N transcripts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count tasks and estimate cost from config `pricing`; no API calls, no files written")
    parser.add_argument("--allow-published-dir", action="store_true",
                        help="Override the refusal to write into a published scores directory "
                             "or next to colliding prefixes. Not for normal use.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--briefs-dir", type=Path, default=DEFAULT_BRIEFS)
    parser.add_argument("--prompts-dir", type=Path, default=None)
    parser.add_argument("--rubrics", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.turn_index is not None and args.turn_indices:
        raise SystemExit("Pass --turn-index or --turn-indices, not both")
    turn_indices = args.turn_indices or ([args.turn_index] if args.turn_index is not None else None)
    output_prefix = args.output_prefix or args.mode
    try:
        summary = asyncio.run(rescore(
            transcripts_dirs=list(args.transcripts_dir),
            scores_dir=args.scores_dir,
            judge_override=args.judge_override,
            output_prefix=output_prefix,
            mode=args.mode,
            conditions=args.conditions,
            models=args.models,
            turn_indices=turn_indices,
            max_budget=args.max_budget,
            concurrency=args.concurrency,
            config_path=args.config,
            briefs_dir=args.briefs_dir,
            prompts_dir=args.prompts_dir,
            rubrics_path=args.rubrics,
            calibration_path=args.calibration,
            limit=args.limit,
            allow_published_dir=args.allow_published_dir,
            argv=sys.argv if argv is None else ["rescore", *argv],
            dry_run=args.dry_run,
            run_ids=load_run_ids_file(args.run_ids_file) if args.run_ids_file else None,
            exclude_run_ids=load_run_ids_file(args.exclude_run_ids_file) if args.exclude_run_ids_file else None,
        ))
    except RescoreRefused as e:
        raise SystemExit(str(e))
    if summary.get("aborted_zero_cost"):
        raise SystemExit(2)
    return summary


if __name__ == "__main__":
    main()
