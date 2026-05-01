"""Pull the DriftBench HF dataset, then symlink at the canonical local paths
that the analysis pipeline expects.

The HF dataset stores files under friendly names (e.g. `transcripts/`,
`scores/`, `openweight/transcripts/`). The local `drift_bench/` analysis
pipeline reads from `drift_bench/data/transcripts/`,
`drift_bench/data/openweight_subjects/transcripts/`, etc. This script bridges
the two with symlinks so reviewers can run

    python -m drift_bench.run_analysis --items all

against an HF snapshot without any code changes.

Usage (from repo root):

    python scripts/fetch_from_hf.py --repo-id <USER>/DriftBench
    # or set DRIFTBENCH_HF_REPO instead.

Pass --no-fetch to skip download (assumes the cache dir is already populated).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = ROOT / "drift_bench" / "data" / "hf_cache"

# Inverse of build_dataset_release.COPY_PLAN: (HF subpath, local repo path).
# Keep these two lists strictly inverse — see the note in the build script.
PATH_MAP: list[tuple[str, str]] = [
    # Benchmark definition lives in the source tree already; we don't symlink
    # it back. Only data + analysis outputs get linked.

    # Core experiment data
    ("transcripts", "drift_bench/data/transcripts"),
    ("scores", "drift_bench/data/scores"),

    # Open-weight subject extension
    ("openweight/transcripts", "drift_bench/data/openweight_subjects/transcripts"),
    ("openweight/scores", "drift_bench/data/openweight_subjects/scores"),
    ("openweight/aggregated/scores.parquet",
     "drift_bench/data/openweight_subjects/aggregated/scores.parquet"),

    # Monitored experiment
    ("monitored/transcripts", "drift_bench/data/monitored/transcripts"),
    ("monitored/scores", "drift_bench/data/monitored/scores"),
    ("monitored/aggregated/scores.parquet",
     "drift_bench/data/monitored/aggregated/scores.parquet"),

    # Sensitivity follow-ups
    ("followup_a/transcripts", "drift_bench/data/followup_a/transcripts"),
    ("followup_a/scores", "drift_bench/data/followup_a/scores"),
    ("followup_a/aggregated/all_scores.parquet",
     "drift_bench/data/followup_a/aggregated/all_scores.parquet"),
    ("followup_b/transcripts", "drift_bench/data/followup_b/transcripts"),
    ("followup_b/scores", "drift_bench/data/followup_b/scores"),
    ("followup_b/aggregated/all_scores.parquet",
     "drift_bench/data/followup_b/aggregated/all_scores.parquet"),

    # Aggregated parquets (top-level)
    ("aggregated/all_scores.parquet", "drift_bench/data/aggregated/all_scores.parquet"),
    ("aggregated/main_scores.parquet", "drift_bench/data/aggregated/main_scores.parquet"),
    ("aggregated/openweight_scores.parquet",
     "drift_bench/data/aggregated/openweight_scores.parquet"),

    # Pre-computed analysis outputs
    ("analysis", "drift_bench/data/analysis"),

    # Human validation
    ("human_validation", "drift_bench/data/human_validation"),
]


def link(src: Path, dst: Path) -> bool:
    """Create symlink at dst -> src. Refuse to clobber a real file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        print(f"  SKIP: {dst.relative_to(ROOT)} is a real file/dir; not clobbering",
              file=sys.stderr)
        return False
    dst.symlink_to(src)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--repo-id",
        default=os.environ.get("DRIFTBENCH_HF_REPO", "anonymous-driftbench/DriftBench"),
        help="HuggingFace dataset repo id (default: anonymous-driftbench/DriftBench). "
             "Override via $DRIFTBENCH_HF_REPO or --repo-id.",
    )
    p.add_argument("--cache-dir", default=str(HF_CACHE_DIR),
                   help=f"Where to download the HF snapshot (default: {HF_CACHE_DIR.relative_to(ROOT)})")
    p.add_argument("--revision", default=None,
                   help="HF revision/tag/branch (default: latest main)")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the HF download; just (re)link from cache-dir")
    args = p.parse_args()

    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    if not args.no_fetch:
        if not args.repo_id:
            print(
                "ERROR: pass --repo-id user/DriftBench or set DRIFTBENCH_HF_REPO",
                file=sys.stderr,
            )
            return 2
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "ERROR: huggingface-hub not installed. Run: pip install huggingface-hub",
                file=sys.stderr,
            )
            return 2
        print(f"Downloading {args.repo_id} -> {cache.relative_to(ROOT)}")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(cache),
        )

    n_linked = n_already = n_missing = 0
    for hf_rel, local_rel in PATH_MAP:
        src = cache / hf_rel
        dst = ROOT / local_rel
        if not src.exists():
            print(f"  WARN: missing in cache: {hf_rel}", file=sys.stderr)
            n_missing += 1
            continue
        if link(src.resolve(), dst):
            n_linked += 1
        else:
            n_already += 1

    print(
        f"\nlinked {n_linked} new symlinks; "
        f"{n_already} already-real (skipped); "
        f"{n_missing} missing in cache"
    )
    if n_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
