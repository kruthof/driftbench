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
  echo "==> aggregating per-run JSONL -> parquet (commercial + open-weight)"
  $PY -c "
import pandas as pd
from pathlib import Path
from drift_bench.analysis.aggregate import aggregate_scores

agg_dir = Path('drift_bench/data/aggregated')
agg_dir.mkdir(parents=True, exist_ok=True)

main_out = agg_dir / 'main_scores.parquet'
open_out = agg_dir / 'openweight_scores.parquet'
combined_out = agg_dir / 'all_scores.parquet'

main_df = aggregate_scores(Path('drift_bench/data/scores'), main_out)
print(f'  commercial: {len(main_df)} rows -> {main_out}')

open_src = Path('drift_bench/data/openweight_subjects/scores')
if open_src.is_dir() and any(open_src.iterdir()):
    open_df = aggregate_scores(open_src, open_out)
    print(f'  open-weight: {len(open_df)} rows -> {open_out}')
    combined = pd.concat([main_df, open_df], ignore_index=True)
else:
    print('  open-weight: no scores found, skipping')
    combined = main_df

combined.to_parquet(combined_out, index=False)
print(f'  combined: {len(combined)} rows -> {combined_out}')
"
}

run_analysis() {
  echo "==> running priority-list analyses (items 1-6)"
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
  # release_manifest also regenerates drift_bench/data/camera_ready/analysis/cr_macros.tex
  # (drift_bench.analysis.camera_ready.macros) and makes release_macros.tex \input it.
  $PY -m drift_bench.analysis.release_manifest \
    --output drift_bench/data/analysis/release_manifest.json \
    --tex-output drift_bench/data/analysis/release_macros.tex
}

run_full() {
  echo "==> running full pipeline (subjects + judges + auditor; uses your API quota)"
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
