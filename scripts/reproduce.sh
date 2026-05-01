#!/usr/bin/env bash
# One-command DriftBench reproduction.
#
#   bash scripts/reproduce.sh --hf         # fetch HF snapshot + reproduce paper numbers (no API calls)
#   bash scripts/reproduce.sh --aggregate  # re-aggregate locally-staged data
#   bash scripts/reproduce.sh --analyze    # run all priority-list analyses + figures
#   bash scripts/reproduce.sh --full       # full pipeline incl. subjects + judges (costs API)
#
# `--hf` is the path reviewers care about: it pulls the HF dataset, symlinks
# it into the canonical local paths, runs aggregation + analysis, and emits
# every CSV / parquet / figure the paper cites.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

run_aggregate() {
  echo "==> aggregating per-run JSONL -> parquet"
  $PY -c "
from pathlib import Path
from drift_bench.analysis.aggregate import aggregate_scores
out = Path('drift_bench/data/aggregated/all_scores.parquet')
out.parent.mkdir(parents=True, exist_ok=True)
df = aggregate_scores(Path('drift_bench/data/scores'), out)
print(f'  aggregated {len(df)} rows -> {out}')
"
}

run_analysis() {
  echo "==> running priority-list analyses (items 1-5)"
  $PY -m drift_bench.run_analysis --items all
  echo "==> regenerating figures"
  $PY -c "
from pathlib import Path
from drift_bench.analysis.figures import generate_all_figures
generate_all_figures(
    Path('drift_bench/data/aggregated/all_scores.parquet'),
    Path('drift_bench/data/figures'),
)
"
  echo "==> regenerating release manifest + paper macros"
  $PY -m drift_bench.analysis.release_manifest \
    --output drift_bench/data/analysis/release_manifest.json \
    --tex-output drift_bench/data/analysis/release_macros.tex
}

run_full() {
  echo "==> running full pipeline (subjects + judges + auditor; costs API budget)"
  $PY -m drift_bench.pipeline --max-budget 400
}

case "${1:-}" in
  --hf)
    echo "==> fetching DriftBench HF dataset"
    $PY scripts/fetch_from_hf.py
    run_aggregate
    run_analysis
    ;;
  --aggregate)
    run_aggregate
    ;;
  --analyze|--analysis)
    run_aggregate
    run_analysis
    ;;
  --full)
    run_full
    run_aggregate
    run_analysis
    ;;
  ""|--help|-h)
    cat <<EOF
usage: $0 <mode>

modes:
  --hf         fetch HF snapshot, symlink, aggregate, run analysis (no API)
  --aggregate  re-aggregate per-run JSONL into parquet
  --analyze    aggregate + run all priority-list analyses + figures
  --full       run subjects/judges/auditor (API-costly), then aggregate+analyze
EOF
    [ -z "${1:-}" ] && exit 2 || exit 0
    ;;
  *)
    echo "usage: $0 [--hf|--aggregate|--analyze|--full]" >&2
    exit 2
    ;;
esac

echo
echo "==> done."
