"""Replay stored restatement probes at a higher token cap (camera-ready X3, used by E3).

For every stored transcript this runner rebuilds the conversation context the
model saw at a probe point and re-issues only the probe call. It mirrors
``MultiTurnRunner._run_restatement_probe`` in ``runners/multi_turn.py``:

    probe_messages = self._build_messages_list(conversation) + [
        {"role": "user", "content": get_restatement_probe()}
    ]

Here ``conversation`` is every stored message before the probe's user message.
``_build_messages_list`` drops messages with ``is_probe`` and keeps only
``role`` and ``content``. The probe prompt is taken from the stored probe user
message, so the replay sends byte-identical text even if the template changed.

What changes relative to the April call:
- max_tokens. That is the point: ``--probe-max-tokens`` replaces 1024.
- temperature stays at 0.0 by default (``--temperature original``), the value
  hard-coded in ``multi_turn.py`` for every April probe, so the cap is the only
  changed variable. ``configured`` (config.yaml value) or a float are options.

What cannot be reproduced from a stored transcript:
- the model version behind the API alias. Old transcripts do not log
  ``response_model``. New lines record the ``response_model`` the API returns.

Budget: spend is added after the await (as in pipeline.py), so the overshoot
past ``--max-budget`` is bounded by concurrency x the cost of one call.

Usage::

    python -m drift_bench.runners.reprobe \\
        --transcripts-dir drift_bench/data/transcripts \\
        --output-dir drift_bench/data/camera_ready/e3_reprobe_gemini_pro \\
        --models gemini/gemini-3.1-pro-preview \\
        --conditions multi_turn_neutral multi_turn_pressure checkpointed_pressure \\
        --turn final --probe-max-tokens 4096 --max-budget 15 --concurrency 4

Output: one ``reprobe_<run_id>.jsonl`` per run, one JSON line per replayed
probe. Resumable: probes that already have a line are not replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drift_bench.prompts.loader import get_restatement_probe
from drift_bench.runners.multi_turn import MultiTurnRunner

logger = logging.getLogger(__name__)

# A probe answer with at least this many tokens counts as truncated at the
# 1024 cap. Gemini Pro answers stop at 1,020 (see CHANGELOG).
TRUNCATION_TOKENS = 1020

# Conditions that never carry probes.
NO_PROBE_CONDITIONS = {"single_shot"}

OUTPUT_PREFIX = "reprobe_"
RUN_LOG_NAME = "reprobe_run_log.jsonl"


@dataclass
class ProbeSite:
    """Where one probe sits in a stored transcript."""

    turn_index: int           # number of non-probe user turns before the probe (as in analysis/probes.py)
    probe_user_index: int     # index of the probe's user message in transcript["messages"]
    probe_assistant_index: int  # index of the stored probe answer
    original_tokens: int | None
    original_truncated: bool
    is_final: bool = False


def load_transcript(path: Path) -> dict[str, Any]:
    """Load the single JSON record of a transcript file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"empty transcript file: {path}")


def find_probe_sites(transcript: dict[str, Any]) -> list[ProbeSite]:
    """Locate every probe pair (user prompt + assistant answer) in a transcript.

    ``turn_index`` counts non-probe user messages seen so far. This matches
    ``drift_bench.analysis.probes.extract_probes`` so that the replay joins
    to ``final_probe_accuracy.csv`` on (run_id, turn_index).
    """
    messages = transcript.get("messages", [])
    sites: list[ProbeSite] = []
    turn_idx = 0
    for i, msg in enumerate(messages):
        is_probe = bool(msg.get("is_probe", False))
        if msg.get("role") == "user" and not is_probe:
            turn_idx += 1
            continue
        if not (is_probe and msg.get("role") == "user"):
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if not (nxt and nxt.get("is_probe") and nxt.get("role") == "assistant"):
            logger.warning(
                "run %s: probe user message at index %d has no stored answer; skipped",
                transcript.get("metadata", {}).get("run_id"), i,
            )
            continue
        tokens = nxt.get("token_count")
        sites.append(ProbeSite(
            turn_index=turn_idx,
            probe_user_index=i,
            probe_assistant_index=i + 1,
            original_tokens=tokens,
            original_truncated=is_truncated(tokens),
        ))
    if sites:
        sites[-1].is_final = True
    return sites


def is_truncated(token_count: int | None, threshold: int = TRUNCATION_TOKENS) -> bool:
    """Original probe truncation flag: token_count >= 1020."""
    return token_count is not None and token_count >= threshold


def build_probe_context(messages: list[dict[str, Any]], site: ProbeSite) -> list[dict[str, str]]:
    """Rebuild the exact message list the original probe call sent.

    Mirrors ``MultiTurnRunner._run_restatement_probe``: the conversation up to
    the probe point, passed through ``BaseRunner._build_messages_list`` (drop
    ``is_probe`` messages, keep ``role`` and ``content``), then the probe prompt.
    The probe prompt is the stored probe user message.
    """
    context = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[:site.probe_user_index]
        if not m.get("is_probe", False)
    ]
    probe_msg = messages[site.probe_user_index]
    context.append({"role": "user", "content": probe_msg["content"]})
    return context


def select_sites(sites: list[ProbeSite], turn: str) -> list[ProbeSite]:
    """``final`` keeps the last probe of the run; ``all`` keeps every probe."""
    if turn == "final":
        return [s for s in sites if s.is_final]
    if turn == "all":
        return list(sites)
    raise ValueError(f"--turn must be 'final' or 'all', got {turn!r}")


def output_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / f"{OUTPUT_PREFIX}{run_id}.jsonl"


def read_existing_records(path: Path) -> list[dict[str, Any]]:
    """Read the lines already written for a run. Skips a damaged line."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping damaged line in %s", path)
    return records


def repair_partial_file(path: Path) -> int:
    """Drop a partial last line left by a crash mid-write.

    If the file does not end with a newline, rewrite it with only the valid
    JSON records. Returns the number of records kept. Without this repair an
    appended record would be glued to the damaged line and lost on resume.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        ends_with_newline = f.read(1) == b"\n"
    records = read_existing_records(path)
    if ends_with_newline:
        return len(records)
    logger.warning("repairing partial last line in %s", path)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(records)


def infer_finish_reason(output_tokens: int | None, probe_max_tokens: int) -> str:
    """Fallback when the API gives no finish_reason.

    Gemini answers stopped 4 tokens short of the 1,024 cap (at 1,020). The
    rule ``tokens >= cap - 4`` extrapolates that observation to other caps.
    """
    if output_tokens is not None and output_tokens >= probe_max_tokens - 4:
        return "length"
    return "stop"


def resolve_temperature(runner: MultiTurnRunner, model_id: str, spec: str) -> float:
    """``configured`` -> config.yaml value; ``original`` -> 0.0 (April probe call); else a float."""
    if spec == "configured":
        return runner._get_temperature(model_id)
    if spec == "original":
        return 0.0
    return float(spec)


class ReprobeRunner(MultiTurnRunner):
    """Re-issues probe calls. Inherits config, rate limiter, retry and the call wrapper."""

    async def replay_probe(
        self,
        transcript: dict[str, Any],
        site: ProbeSite,
        probe_max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Re-issue one probe and return the output record (one JSONL line)."""
        meta = transcript["metadata"]
        messages = transcript["messages"]
        probe_messages = build_probe_context(messages, site)
        stored_prompt = messages[site.probe_user_index]["content"]
        template_prompt = get_restatement_probe()

        result = await self._call_llm(
            model_id=meta["model_id"],
            messages=probe_messages,
            temperature=temperature,
            max_tokens=probe_max_tokens,
        )

        finish_reason = result.get("finish_reason")
        if finish_reason is None:
            finish_reason = infer_finish_reason(result.get("output_tokens"), probe_max_tokens)
            finish_reason_source = "inferred_from_tokens"
        else:
            finish_reason_source = "api"

        return {
            "run_id": meta["run_id"],
            "brief_id": meta["brief_id"],
            "model_id": meta["model_id"],
            "condition": meta["condition"],
            "turn_index": site.turn_index,
            "original_probe_tokens": site.original_tokens,
            "original_probe_truncated": site.original_truncated,
            "new_probe_content": result["content"],
            "new_probe_tokens": result["output_tokens"],
            "new_finish_reason": finish_reason,
            "cost_usd": result["cost"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probe_max_tokens": probe_max_tokens,
            # Extra fields, documented in the module docstring.
            "temperature": temperature,
            "finish_reason_source": finish_reason_source,
            "response_model": result.get("response_model"),
            "new_input_tokens": result["input_tokens"],
            "is_final_probe": site.is_final,
            "probe_prompt_matches_template": stored_prompt == template_prompt,
        }


def list_transcripts(
    transcripts_dir: Path,
    models: list[str] | None,
    conditions: list[str] | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load every transcript under ``transcripts_dir`` and apply the model and condition filters."""
    selected = []
    for path in sorted(transcripts_dir.glob("*.jsonl")):
        try:
            transcript = load_transcript(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("cannot read %s: %s", path, e)
            continue
        meta = transcript.get("metadata", {})
        if models and meta.get("model_id") not in models:
            continue
        if conditions and meta.get("condition") not in conditions:
            continue
        selected.append((path, transcript))
    return selected


async def replay_transcripts(
    runner: ReprobeRunner,
    transcripts: list[tuple[Path, dict[str, Any]]],
    output_dir: Path,
    turn: str,
    probe_max_tokens: int,
    max_budget: float,
    concurrency: int = 4,
    temperature_spec: str = "original",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replay the selected probes of every transcript. Returns the summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    budget_spent = 0.0
    summary: dict[str, Any] = {
        "runs_scanned": len(transcripts),
        "runs_single_shot_skipped": 0,
        "runs_without_probes": 0,
        "runs_already_complete": 0,
        "runs_replayed": 0,
        "probes_selected": 0,
        "probes_already_done": 0,
        "probes_replayed": 0,
        "probes_skipped_budget": 0,
        "probes_failed": 0,
        "cost_usd": 0.0,
        "turn": turn,
        "probe_max_tokens": probe_max_tokens,
        "temperature": temperature_spec,
        "dry_run": dry_run,
    }

    async def run_one(transcript: dict[str, Any]) -> None:
        nonlocal budget_spent
        meta = transcript["metadata"]
        run_id = meta["run_id"]
        if meta.get("condition") in NO_PROBE_CONDITIONS:
            summary["runs_single_shot_skipped"] += 1
            return
        sites = select_sites(find_probe_sites(transcript), turn)
        if not sites:
            summary["runs_without_probes"] += 1
            return
        summary["probes_selected"] += len(sites)

        # Resume: skip probes that already have a line in this run's output file.
        out_path = output_path(output_dir, run_id)
        repair_partial_file(out_path)
        done_turns = {r.get("turn_index") for r in read_existing_records(out_path)}
        todo = [s for s in sites if s.turn_index not in done_turns]
        summary["probes_already_done"] += len(sites) - len(todo)
        if not todo:
            summary["runs_already_complete"] += 1
            logger.info("skipping %s (output exists)", run_id)
            return

        temperature = resolve_temperature(runner, meta["model_id"], temperature_spec)
        replayed_here = 0
        for site in todo:
            if dry_run:
                logger.info(
                    "dry run: %s turn %d, %d context messages, cap %d, temperature %.2f",
                    run_id, site.turn_index,
                    len(build_probe_context(transcript["messages"], site)),
                    probe_max_tokens, temperature,
                )
                summary["probes_replayed"] += 1
                replayed_here += 1
                continue
            async with semaphore:
                # Budget check inside the semaphore, as in pipeline.py.
                if budget_spent >= max_budget:
                    logger.warning("budget %.2f USD reached; skipping %s turn %d",
                                   max_budget, run_id, site.turn_index)
                    summary["probes_skipped_budget"] += 1
                    continue
                try:
                    record = await runner.replay_probe(
                        transcript, site, probe_max_tokens, temperature,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error("failed %s turn %d: %s", run_id, site.turn_index, e)
                    summary["probes_failed"] += 1
                    continue
                budget_spent += record["cost_usd"]
                summary["cost_usd"] += record["cost_usd"]
                summary["probes_replayed"] += 1
                replayed_here += 1
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                logger.info(
                    "replayed %s turn %d: %s tokens, finish=%s, %.4f USD (total %.2f/%.2f)",
                    run_id, site.turn_index, record["new_probe_tokens"],
                    record["new_finish_reason"], record["cost_usd"], budget_spent, max_budget,
                )
        if replayed_here:
            summary["runs_replayed"] += 1

    await asyncio.gather(*(run_one(t) for _, t in transcripts))
    summary["cost_usd"] = round(summary["cost_usd"], 6)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("reprobe summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay stored restatement probes at a higher cap.")
    p.add_argument("--transcripts-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=None, help="model ids to include (default: all)")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="conditions to include (default: all; single_shot is always skipped)")
    p.add_argument("--turn", choices=["final", "all"], default="final")
    p.add_argument("--probe-max-tokens", type=int, default=None,
                   help="probe cap (default: experiment.probe_max_tokens in config.yaml)")
    p.add_argument("--max-budget", type=float, default=None,
                   help="USD cap for this invocation (required unless --dry-run)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", default="original",
                   help="'original' (0.0, as in every April probe; default), "
                        "'configured' (config.yaml value), or a float")
    p.add_argument("--config", type=Path, default=None, help="config.yaml path")
    p.add_argument("--limit", type=int, default=None, help="replay at most this many runs (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="rebuild contexts, make no API call")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    started = time.time()
    args = parse_args(argv)
    if args.max_budget is None and not args.dry_run:
        raise SystemExit("--max-budget is required unless --dry-run is set")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runner = ReprobeRunner(config_path=args.config)
    probe_max_tokens = args.probe_max_tokens or runner._get_probe_max_tokens()

    transcripts = list_transcripts(args.transcripts_dir, args.models, args.conditions)
    if args.limit is not None:
        transcripts = transcripts[:args.limit]
    logger.info("%d transcripts selected from %s", len(transcripts), args.transcripts_dir)

    summary = asyncio.run(replay_transcripts(
        runner=runner,
        transcripts=transcripts,
        output_dir=args.output_dir,
        turn=args.turn,
        probe_max_tokens=probe_max_tokens,
        max_budget=args.max_budget if args.max_budget is not None else 0.0,
        concurrency=args.concurrency,
        temperature_spec=args.temperature,
        dry_run=args.dry_run,
    ))
    summary["command"] = " ".join(sys.argv if argv is None else ["reprobe", *argv])
    summary["models"] = args.models
    summary["conditions"] = args.conditions
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["elapsed_s"] = round(time.time() - started, 1)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / RUN_LOG_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
