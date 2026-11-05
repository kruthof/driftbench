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

    python scripts/fetch_from_hf.py                              # driftbench/DriftBench
    python scripts/fetch_from_hf.py --repo-id <USER>/DriftBench  # a fork or mirror
    # or set DRIFTBENCH_HF_REPO instead.

The default revision is the tag `camera-ready`: the exact snapshot the paper's
numbers come from. `--revision main` follows the latest commit instead.

The snapshot has about 19,000 files. Anonymous downloads can hit the Hugging
Face rate limit (HTTP 429). Set HF_TOKEN to a free read token
(https://huggingface.co/settings/tokens) and re-run; the download resumes and
skips the files it already has.

Pass --no-fetch to skip download (assumes the cache dir is already populated).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = ROOT / "drift_bench" / "data" / "hf_cache"
# The HF tag on the dataset commit the paper's numbers come from.
DEFAULT_REVISION = "camera-ready"
TOKEN_HINT = (
    "Set HF_TOKEN to a free Hugging Face read token "
    "(https://huggingface.co/settings/tokens) and re-run; the download resumes."
)

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

    # Run summaries (read by the camera-ready modules failed_runs,
    # corrections and sensitivity)
    ("run_summary.json", "drift_bench/data/run_summary.json"),
    ("openweight/run_summary.json", "drift_bench/data/openweight_subjects/run_summary.json"),
    ("monitored/run_summary.json", "drift_bench/data/monitored/run_summary.json"),
    ("followup_a/run_summary.json", "drift_bench/data/followup_a/run_summary.json"),
    ("followup_b/run_summary.json", "drift_bench/data/followup_b/run_summary.json"),

    # Aggregated parquets (top-level)
    ("aggregated/all_scores.parquet", "drift_bench/data/aggregated/all_scores.parquet"),
    ("aggregated/main_scores.parquet", "drift_bench/data/aggregated/main_scores.parquet"),
    ("aggregated/openweight_scores.parquet",
     "drift_bench/data/aggregated/openweight_scores.parquet"),

    # Pre-computed analysis outputs
    ("analysis", "drift_bench/data/analysis"),

    # Second-judge scorings
    ("scores_skeptical", "drift_bench/data/scores_skeptical"),
    ("openweight_judges", "drift_bench/data/openweight_judges"),

    # Camera-ready experiments and analysis CSVs
    ("camera_ready", "drift_bench/data/camera_ready"),

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


def _shown(path: Path) -> str:
    """`path` relative to the repository root when it lies inside it."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def download_error_message(exc: BaseException, repo_id: str, revision: str | None) -> str | None:
    """One line that tells the user what to do after a failed download.

    Returns None for errors that are not HTTP or network errors, so that bugs
    keep their traceback.
    """
    from huggingface_hub.utils import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, RevisionNotFoundError):
        return (f"ERROR: revision '{revision}' not found on {repo_id}. "
                "Pass --revision main for the latest commit.")
    if isinstance(exc, RepositoryNotFoundError):
        return f"ERROR: dataset {repo_id} not found. Check --repo-id. {TOKEN_HINT}"
    module = type(exc).__module__.split(".")[0]
    if not (isinstance(exc, (HfHubHTTPError, OSError))
            or module in ("httpx", "httpcore", "requests", "urllib3")):
        return None
    # The HTTP status can sit on a chained error: a 429 on the repo lookup
    # surfaces as LocalEntryNotFoundError whose __cause__ holds the response.
    status = None
    for err in (exc, exc.__cause__, exc.__context__):
        status = getattr(getattr(err, "response", None), "status_code", None)
        if status:
            break
    what = f"HTTP {status}" if status else type(exc).__name__
    if status == 429:
        what = "HTTP 429, rate limit"
    return f"ERROR: download stopped ({what}). {TOKEN_HINT}"


def incomplete_download_message(cache: Path) -> str | None:
    """One line when files listed in MANIFEST.txt are absent from `cache`, else None.

    snapshot_download can return a partial local_dir without raising when the
    Hub answers 429 or cannot be reached, so the files are checked here.
    """
    manifest = cache / "MANIFEST.txt"
    if not manifest.is_file():
        return f"ERROR: download incomplete (no MANIFEST.txt). {TOKEN_HINT}"
    listed = [line.split(maxsplit=1)[1] for line in manifest.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not listed:
        return f"ERROR: download incomplete (MANIFEST.txt lists no files). {TOKEN_HINT}"
    absent = sum(not (cache / rel).is_file() for rel in listed)
    if not absent:
        return None
    return f"ERROR: download incomplete ({absent:,} of {len(listed):,} listed files missing). {TOKEN_HINT}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--repo-id",
        default=os.environ.get("DRIFTBENCH_HF_REPO", "driftbench/DriftBench"),
        help="HuggingFace dataset repo id (default: driftbench/DriftBench). "
             "Override via $DRIFTBENCH_HF_REPO or --repo-id.",
    )
    p.add_argument("--cache-dir", default=str(HF_CACHE_DIR),
                   help=f"Where to download the HF snapshot (default: {HF_CACHE_DIR.relative_to(ROOT)})")
    p.add_argument("--revision", default=DEFAULT_REVISION,
                   help=f"HF revision/tag/branch (default: the tag {DEFAULT_REVISION}, "
                        "the snapshot of the paper; 'main' follows the latest commit)")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the HF download; just (re)link from cache-dir")
    return p


def main() -> int:
    args = build_parser().parse_args()

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
        print(f"Downloading {args.repo_id} (revision {args.revision}) -> {_shown(cache)}")
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=str(cache),
            )
        except Exception as exc:  # noqa: BLE001 - HTTP and network errors get one line
            message = download_error_message(exc, args.repo_id, args.revision)
            if message is None:
                raise
            print(message, file=sys.stderr)
            return 1
        message = incomplete_download_message(cache)
        if message is not None:
            print(message, file=sys.stderr)
            return 1

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
