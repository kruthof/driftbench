#!/usr/bin/env bash
# One-command DriftBench reproduction.
#
#   bash scripts/reproduce.sh --hf         # fetch HF snapshot + reproduce paper numbers (no API calls)
#   bash scripts/reproduce.sh --aggregate  # re-aggregate locally-staged data
#   bash scripts/reproduce.sh --analyze    # run all priority-list analyses + figures
#   bash scripts/reproduce.sh --full       # full pipeline incl. subjects + judges (costs API)
#
# `--hf` is the path reviewers care about: it pulls the HF dataset, symlinks
# it into the canonical local paths, runs aggregation + analysis, emits every
# CSV / parquet / figure the paper cites, and checks the regenerated numbers
# against the snapshot's MANIFEST.txt (scripts/verify_reproduction.py).
# Extra arguments after --hf go to scripts/fetch_from_hf.py
# (e.g. `--hf --revision main`).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
# litellm otherwise downloads its model map on import; the bundled map keeps the
# context windows used by tokens_context fixed for the pinned litellm version.
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"

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
  echo "==> regenerating submission-era figures (drift_bench/data/figures)"
  $PY -c "
from pathlib import Path
from drift_bench.analysis.figures import generate_all_figures
generate_all_figures(
    Path('drift_bench/data/aggregated/all_scores.parquet'),
    Path('drift_bench/data/figures'),
)
"
  echo "==> regenerating the six camera-ready figures (figures/)"
  $PY -m drift_bench.analysis.camera_ready.figures_cr
  # sensitivity checks its cells against the fig_sensitivity sidecar, which
  # exists only now; re-run it so its provenance records that check.
  echo "==> re-running sensitivity against the new figure sidecar"
  $PY -m drift_bench.analysis.camera_ready.sensitivity > /dev/null
  echo "==> regenerating release manifest + paper macros"
  # release_manifest also regenerates drift_bench/data/camera_ready/analysis/cr_macros.tex
  # (drift_bench.analysis.camera_ready.macros) and makes release_macros.tex \input it.
  $PY -m drift_bench.analysis.release_manifest \
    --output drift_bench/data/analysis/release_manifest.json \
    --tex-output drift_bench/data/analysis/release_macros.tex
}

# The run overwrites the released files through the symlinks, so keep a copy
# of each checked file (confirmed against MANIFEST.txt) for the cell-level
# comparison when a hash differs.
run_save_reference() {
  echo "==> saving the released copies of the checked files"
  $PY scripts/verify_reproduction.py --save-reference drift_bench/data/released_reference "$@"
}

run_verify() {
  echo "==> comparing the regenerated numbers with the snapshot's MANIFEST.txt"
  $PY scripts/verify_reproduction.py --since "$START" "$@"
}

run_full() {
  echo "==> running full pipeline (subjects + judges + auditor; uses your API quota)"
  $PY -m drift_bench.pipeline --max-budget 400
}

case "${1:-}" in
  --hf)
    shift
    echo "==> fetching DriftBench HF dataset"
    $PY scripts/fetch_from_hf.py "$@"
    run_save_reference
    START="$($PY -c 'import time; print(time.time())')"
    run_aggregate
    run_analysis
    run_verify
    ;;
  --aggregate)
    run_aggregate
    ;;
  --analyze|--analysis)
    run_save_reference --allow-missing-manifest
    START="$($PY -c 'import time; print(time.time())')"
    run_aggregate
    run_analysis
    run_verify --allow-missing-manifest
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
  --hf         fetch HF snapshot, symlink, aggregate, run analysis + figures,
               verify against MANIFEST.txt (no API); extra args go to fetch_from_hf.py
  --aggregate  re-aggregate per-run JSONL into parquet
  --analyze    aggregate + run all priority-list analyses + figures + verify
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
