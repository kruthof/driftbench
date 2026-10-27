"""Stage everything we publish on HuggingFace into Dataset/.

Run from the repo root:

    python scripts/build_dataset_release.py --clean

The output `Dataset/` directory is the literal HF dataset repo layout — it
mirrors the local `drift_bench/data/` paths under friendly names so that
`scripts/fetch_from_hf.py` can symlink HF -> local for the reverse direction.
Anonymization replacements are loaded from `.anonymize.json` (gitignored);
forbidden substrings and forbidden file extensions cause the build to fail
hard so we never accidentally publish copyrighted PDFs or identifying strings.
The forbidden-substring list is empty since the camera-ready de-anonymization.

The build strips local path prefixes (the absolute repository root and Claude
Code scratchpad directories) from the staged copies of text files, so recorded
paths ship repository-relative. It rewrites the staged copies only, never the
source files, and fails if the repository root or the user's home directory is
still present afterwards.

The staging dir must not be the repository root, overlap a source (including
docs/dataset_card.md and .anonymize.json), or be a non-empty directory without
a MANIFEST.txt from an earlier build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source path under repo, target path under Dataset/) — the inverse of
# scripts/fetch_from_hf.PATH_MAP. Keep these two lists strictly inverse:
# a change to one needs the same change to the other or --hf repro silently
# misses files.
COPY_PLAN: list[tuple[str, str]] = [
    # ── Benchmark definition ────────────────────────────────────────────────
    ("drift_bench/briefs", "briefs"),
    ("drift_bench/prompts", "prompts"),
    ("drift_bench/schema/brief_schema.json", "brief_schema.json"),
    ("drift_bench/judges/rubrics.yaml", "rubrics.yaml"),
    ("drift_bench/judges/calibration.yaml", "calibration.yaml"),

    # ── Core experiment data (5-model commercial subset) ───────────────────
    ("drift_bench/data/transcripts", "transcripts"),
    ("drift_bench/data/scores", "scores"),

    # ── Open-weight subject extension ──────────────────────────────────────
    ("drift_bench/data/openweight_subjects/transcripts", "openweight/transcripts"),
    ("drift_bench/data/openweight_subjects/scores", "openweight/scores"),

    # ── Monitored-warning experiment ───────────────────────────────────────
    ("drift_bench/data/monitored/transcripts", "monitored/transcripts"),
    ("drift_bench/data/monitored/scores", "monitored/scores"),

    # ── Sensitivity follow-ups ─────────────────────────────────────────────
    ("drift_bench/data/followup_a/transcripts", "followup_a/transcripts"),
    ("drift_bench/data/followup_a/scores", "followup_a/scores"),
    ("drift_bench/data/followup_b/transcripts", "followup_b/transcripts"),
    ("drift_bench/data/followup_b/scores", "followup_b/scores"),

    # ── Run summaries (completed and failed run counts). The camera-ready
    #     modules failed_runs, corrections and sensitivity read them. The
    #     top-level one was overwritten by the rigor-pressure pass. ─────────
    ("drift_bench/data/run_summary.json", "run_summary.json"),
    ("drift_bench/data/openweight_subjects/run_summary.json", "openweight/run_summary.json"),
    ("drift_bench/data/monitored/run_summary.json", "monitored/run_summary.json"),
    ("drift_bench/data/followup_a/run_summary.json", "followup_a/run_summary.json"),
    ("drift_bench/data/followup_b/run_summary.json", "followup_b/run_summary.json"),

    # ── Aggregated parquets (lets reviewers verify paper numbers without
    #     re-running aggregation) ────────────────────────────────────────────
    ("drift_bench/data/aggregated/all_scores.parquet", "aggregated/all_scores.parquet"),
    ("drift_bench/data/aggregated/main_scores.parquet", "aggregated/main_scores.parquet"),
    ("drift_bench/data/aggregated/openweight_scores.parquet", "aggregated/openweight_scores.parquet"),
    ("drift_bench/data/openweight_subjects/aggregated/scores.parquet",
     "openweight/aggregated/scores.parquet"),
    ("drift_bench/data/monitored/aggregated/scores.parquet",
     "monitored/aggregated/scores.parquet"),
    ("drift_bench/data/followup_a/aggregated/all_scores.parquet",
     "followup_a/aggregated/all_scores.parquet"),
    ("drift_bench/data/followup_b/aggregated/all_scores.parquet",
     "followup_b/aggregated/all_scores.parquet"),

    # ── Analysis outputs (CSVs/JSON the paper macros derive from) ──────────
    ("drift_bench/data/analysis", "analysis"),

    # ── Second-judge scorings cited in the paper (skeptical GPT-5.4 prompt;
    #     Llama/Qwen as judges) ───────────────────────────────────────────────
    ("drift_bench/data/scores_skeptical", "scores_skeptical"),
    ("drift_bench/data/openweight_judges", "openweight_judges"),

    # ── Camera-ready experiments (crossed judge, per-turn, probe replay,
    #     SWE pilot, typology, mechanism, checkers) and their analysis CSVs ───
    ("drift_bench/data/camera_ready", "camera_ready"),

    # ── Human validation (non-rater files; raters handled separately) ──────
    ("drift_bench/data/human_validation/scoring_items.json",
     "human_validation/scoring_items.json"),
    # Set 2 as the rater app showed it (plan F12; copied from the gitignored
    # scoring_app_standalone_set2/ app).
    ("drift_bench/data/human_validation/scoring_items_set2.json",
     "human_validation/scoring_items_set2.json"),
    ("drift_bench/data/human_validation/scoring_form.md",
     "human_validation/scoring_form.md"),
    # The rater app's instructions, verbatim (Appendix C quotes them).
    ("drift_bench/data/human_validation/rater_instructions.md",
     "human_validation/rater_instructions.md"),
    ("drift_bench/data/human_validation/human_scores.json",
     "human_validation/human_scores.json"),
    ("drift_bench/data/human_validation/README.md",
     "human_validation/README.md"),
]

# Per-rater files copied with anonymization replacements applied.
ANNOTATION_FILE_MAP: list[tuple[str, str]] = [
    (f"drift_bench/data/human_validation/raters/{name}",
     f"human_validation/raters/{name}")
    for name in [
        "human_rater_set1_rater_A.json",
        "human_rater_set1_rater_B.json",
        "human_rater_set1_rater_C.json",
        "human_rater_set2_rater_A.json",
        "human_rater_set2_rater_C.json",
        "human_rater_set2_rater_D.json",
    ]
]

ANON_CONFIG_PATH = ROOT / ".anonymize.json"

FORBIDDEN_EXTENSIONS = (".pdf", ".html", ".PDF", ".HTML")
FORBIDDEN_BASENAMES = (".DS_Store", "__pycache__")
# structure_* score files and prompts/structure_extraction.j2 ship: Appendix A
# reports the LLM structural counts, and the paper promises all prompts and scores.

# Staged files with these suffixes are binary; the local-path transform skips them.
BINARY_SUFFIXES = (
    ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc",
    ".npy", ".npz", ".pkl", ".zip", ".gz",
)
# A Claude Code scratchpad directory, e.g.
# /private/tmp/claude-501/<project-slug>/<session-uuid>/scratchpad/
SCRATCHPAD_RE = re.compile(r"/private/tmp/claude-\d+/[^/\s\"']+/[0-9a-f-]+/scratchpad/")
SCRATCHPAD_PLACEHOLDER = "<scratchpad>/"


def _load_anon_config() -> dict:
    if not ANON_CONFIG_PATH.exists():
        print(
            f"  WARN: missing {ANON_CONFIG_PATH.name} — proceeding with no replacements/sweep",
            file=sys.stderr,
        )
        return {"rater_replacements": {}, "forbidden_substrings": []}
    return json.loads(ANON_CONFIG_PATH.read_text())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_excluded(path: Path) -> bool:
    if path.suffix in FORBIDDEN_EXTENSIONS:
        return True
    if path.name in FORBIDDEN_BASENAMES:
        return True
    if any(part in FORBIDDEN_BASENAMES for part in path.parts):
        return True
    return False


def strip_local_paths_text(text: str, root: Path = ROOT) -> str:
    """Return `text` without local path prefixes.

    `<root>/x` becomes `x`. A bare `<root>` becomes `.`. A scratchpad prefix
    becomes `<scratchpad>/`.
    """
    root_str = str(root)
    text = text.replace(root_str + "/", "")
    # A bare root not followed by a path character (so `<root>_old` is kept).
    text = re.sub(re.escape(root_str) + r"(?![\w.-])", ".", text)
    return SCRATCHPAD_RE.sub(SCRATCHPAD_PLACEHOLDER, text)


def strip_local_paths(out: Path, root: Path = ROOT) -> list[Path]:
    """Rewrite the staged text files under `out` without local path prefixes.

    Only files under `out` are touched; the caller guarantees `out` holds
    copies, not sources (see `_check_out_dir`). Binary files and files that
    are not UTF-8 are left as they are. Returns the rewritten files.
    """
    changed: list[Path] = []
    for f in sorted(out.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        if f.suffix.lower() in BINARY_SUFFIXES:
            continue
        raw = f.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = strip_local_paths_text(text, root)
        if new != text:
            f.write_bytes(new.encode("utf-8"))
            changed.append(f)
    return changed


def _check_out_dir(out: Path, root: Path = ROOT) -> None:
    """Refuse a staging dir that is the repo root, overlaps a source path, or is foreign.

    `--clean` deletes the staging dir and the path transform rewrites files
    in it, so it must never contain or sit inside a source. The sources include
    the dataset card and `.anonymize.json`, which the build reads but does not
    copy under their own names. A non-empty dir is accepted only if it holds a
    MANIFEST.txt from an earlier build.
    """
    out = out.resolve()
    if out == root.resolve():
        raise SystemExit(f"refusing staging dir {out}: it is the repository root")
    for src_rel, _ in COPY_PLAN + ANNOTATION_FILE_MAP + [("docs/dataset_card.md", "README.md"), (".anonymize.json", "")]:
        src = (root / src_rel).resolve()
        if out == src or out.is_relative_to(src) or src.is_relative_to(out):
            raise SystemExit(f"refusing staging dir {out}: it overlaps source {src_rel}")
    if out.exists() and not out.is_dir():
        raise SystemExit(f"refusing staging dir {out}: it is a file")
    if out.exists() and any(out.iterdir()) and not (out / "MANIFEST.txt").is_file():
        raise SystemExit(f"refusing staging dir {out}: not empty and not an earlier build (no MANIFEST.txt)")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def copy_tree_filtered(src: Path, dst: Path) -> list[Path]:
    copied: list[Path] = []
    if not src.exists():
        print(f"  WARN: source missing: {src}", file=sys.stderr)
        return copied
    if src.is_file():
        if _is_excluded(src):
            return copied
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return [dst]
    for f in src.rglob("*"):
        if not f.is_file() or _is_excluded(f):
            continue
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied.append(target)
    return copied


def _apply_replacements_in_obj(obj, replacements: dict) -> None:
    """In-place: replace any string value matching a replacement key."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str) and v in replacements:
                obj[k] = replacements[v]
            else:
                _apply_replacements_in_obj(v, replacements)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v in replacements:
                obj[i] = replacements[v]
            else:
                _apply_replacements_in_obj(v, replacements)


def anonymize_annotation(src: Path, dst: Path, replacements: dict | None) -> None:
    data = json.loads(src.read_text())
    if replacements:
        _apply_replacements_in_obj(data, replacements)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_license(out: Path) -> None:
    (out / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
        "Human-annotation files under human_validation/raters/ are released under\n"
        "CC BY-NC 4.0 (non-commercial). See human_validation/README.md.\n"
    )


def write_dataset_card(out: Path) -> None:
    card_src = ROOT / "docs" / "dataset_card.md"
    if card_src.exists():
        shutil.copy2(card_src, out / "README.md")
    else:
        print(
            f"  WARN: missing {card_src.relative_to(ROOT)} — Dataset/README.md not written",
            file=sys.stderr,
        )


def write_manifest(out: Path) -> int:
    lines = []
    for f in sorted(out.rglob("*")):
        if not f.is_file() or f.name == "MANIFEST.txt":
            continue
        lines.append(f"{sha256_of(f)}  {f.relative_to(out)}")
    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_gates(out: Path, forbidden: tuple[str, ...], root: Path = ROOT,
                 home: Path | None = None) -> int:
    bad_ext = [
        p for p in out.rglob("*")
        if p.is_file() and p.suffix in FORBIDDEN_EXTENSIONS
    ]
    if bad_ext:
        print("FAIL: forbidden extensions in Dataset/:", file=sys.stderr)
        for p in bad_ext:
            print(f"  {p.relative_to(out)}", file=sys.stderr)
        return 2

    root_str = str(root)
    # The home directory catches local paths outside the repository too.
    # A root-level home ("/") would match every path, so it is skipped.
    home_path = Path.home() if home is None else home
    home_str = str(home_path) if len(home_path.parts) > 1 else None
    for f in out.rglob("*"):
        if not f.is_file() or f.suffix in {".parquet"}:
            continue
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        if root_str in txt or (home_str and home_str in txt) or SCRATCHPAD_RE.search(txt):
            print(
                f"FAIL: {f.relative_to(out)}: contains a local path prefix",
                file=sys.stderr,
            )
            return 2

    if forbidden:
        for f in out.rglob("*"):
            if not f.is_file() or f.name == "MANIFEST.txt":
                continue
            if f.suffix in {".parquet"}:
                # Binary parquet/arrow files: skip text scan.
                continue
            try:
                txt = f.read_text(errors="ignore")
            except Exception:
                continue
            for s in forbidden:
                if s and s in txt:
                    print(
                        f"FAIL: {f.relative_to(out)}: contains forbidden substring {s!r}",
                        file=sys.stderr,
                    )
                    return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="Dataset",
                   help="Staging directory, relative to the repo root or absolute "
                        "(default: Dataset). The build refuses the repository root, a "
                        "directory that overlaps a source, and a non-empty directory "
                        "without a MANIFEST.txt from an earlier build")
    p.add_argument("--clean", action="store_true",
                   help="Remove the staging dir before rebuilding")
    p.add_argument("--skip-gates", action="store_true",
                   help="Skip the no-PDF / no-real-name verification (NOT for release builds)")
    args = p.parse_args()

    out = ROOT / args.out  # an absolute --out replaces ROOT
    _check_out_dir(out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = _load_anon_config()
    rater_repl_by_file = cfg.get("rater_replacements", {}) or {}
    forbidden = tuple(cfg.get("forbidden_substrings", []) or [])

    print(f"Staging into {_display(out)}/  (clean={args.clean})")
    total_copied = 0
    for src_rel, tgt_rel in COPY_PLAN:
        copied = copy_tree_filtered(ROOT / src_rel, out / tgt_rel)
        total_copied += len(copied)
        print(f"  {len(copied):>5} files  {src_rel}  ->  {tgt_rel}")

    for src_rel, tgt_rel in ANNOTATION_FILE_MAP:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  WARN: rater source missing: {src_rel}", file=sys.stderr)
            continue
        anonymize_annotation(
            src, out / tgt_rel,
            rater_repl_by_file.get(Path(src_rel).name),
        )
        total_copied += 1
    print(f"  {len(ANNOTATION_FILE_MAP):>5} rater files anonymized")

    write_license(out)
    write_dataset_card(out)

    stripped = strip_local_paths(out)
    print(f"  {len(stripped):>5} staged text files had local path prefixes stripped")

    n_files = write_manifest(out)
    total_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / (1024 ** 2)
    print(f"\n{n_files} files manifested; {total_mb:.1f} MB total")

    if args.skip_gates:
        print("WARNING: --skip-gates set, no anonymization/PDF verification ran")
        return 0

    rc = verify_gates(out, forbidden)
    if rc != 0:
        return rc
    print("OK: gates passed (no forbidden extensions, no local path prefixes, "
          "no forbidden substrings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
