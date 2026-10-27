"""Revised coding-domain pilot: Gemini subjects, Opus judge, NO auditor.

Phase A: subject runs via the unmodified pipeline (skip_scoring=True).
Phase B: JudgePipeline.score_all on the resulting transcripts.

The auditor is skipped because its configured model (openai/gpt-5.4) has no
billing quota. KBV needs only the judge + the probe.

Cost for every API call is appended to cost_ledger.jsonl.
"""

import asyncio
import argparse
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

for line in (HERE / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ[k.strip()] = v.strip()

import jsonlines  # noqa: E402
from drift_bench.runners.base import BaseRunner, Transcript, RunMetadata, Message  # noqa: E402
from drift_bench.pipeline import run_experiment  # noqa: E402
from drift_bench.judges.judge import JudgePipeline  # noqa: E402
from drift_bench.schema.validate import load_all_briefs  # noqa: E402

LEDGER = HERE / "cost_ledger.jsonl"
HARD_CAP = float(os.environ.get("PILOT_HARD_CAP", "19.0"))

_lock = threading.Lock()
_total = 0.0
_session = 0.0
_calls = {"subject": 0, "judge": 0}


class HardBudgetExceeded(RuntimeError):
    pass


_orig_call = BaseRunner._call_llm


async def _instrumented(self, model_id, messages, temperature=0.7,
                        max_tokens=4096, response_format=None):
    global _total, _session
    role = "judge" if isinstance(self, JudgePipeline) else "subject"
    with _lock:
        if _total >= HARD_CAP:
            raise HardBudgetExceeded(f"${_total:.2f} >= cap ${HARD_CAP:.2f}")
    res = await _orig_call(self, model_id, messages, temperature=temperature,
                           max_tokens=max_tokens, response_format=response_format)
    with _lock:
        _total += res["cost"]
        _session += res["cost"]
        _calls[role] += 1
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"model": model_id, "cost": res["cost"],
                                "in": res["input_tokens"], "out": res["output_tokens"],
                                "role": role, "phase": "pilot2"}) + "\n")
    return res


BaseRunner._call_llm = _instrumented


def preload():
    t = 0.0
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            if ln.strip():
                t += json.loads(ln).get("cost", 0.0)
    return t


def load_transcripts(d: Path):
    out = []
    for p in sorted(d.glob("*.jsonl")):
        try:
            with jsonlines.open(p) as r:
                e = next(iter(r))
            out.append(Transcript(metadata=RunMetadata(**e["metadata"]),
                                  messages=[Message(**m) for m in e["messages"]]))
        except Exception as ex:
            print(f"  ! skip {p.name}: {ex}")
    return out


async def main():
    global _total
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-budget", type=float, default=20.0)
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()

    _total = preload()
    print(f"[ledger] preloaded ${_total:.4f} (cap ${HARD_CAP})")

    cfg = HERE / "drift_bench" / "config.yaml"

    if not a.judge_only:
        print("=== PHASE A: subject runs ===")
        s = await run_experiment(
            config_path=cfg, briefs_dir=a.briefs_dir, output_base=a.output,
            conditions=a.conditions, models=a.models, repetitions=1,
            enable_restatement_probe=True, brief_reinjection_ids=None,
            concurrency=a.concurrency, skip_scoring=True,
            max_budget=a.max_budget,
        )
        print(json.dumps(s, indent=2))

    print("=== PHASE B: judge (Opus 4.6), auditor SKIPPED (gpt-5.4 quota dead) ===")
    briefs = load_all_briefs(a.briefs_dir)
    ts = load_transcripts(a.output / "transcripts")
    print(f"  loaded {len(ts)} transcripts")
    judge = JudgePipeline(cfg)
    scores_dir = a.output / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    js = await judge.score_all(ts, briefs, scores_dir, a.concurrency)
    print(f"  judge scored {len(js)}/{len(ts)}")

    print(f"[ledger] session=${_session:.4f}  cumulative=${_total:.4f}")
    print(f"[calls] {_calls}")


if __name__ == "__main__":
    asyncio.run(main())
