"""Check regenerated paper numbers against the released snapshot.

`bash scripts/reproduce.sh --hf` (or `--analyze`) regenerates the files that
carry the paper's numbers. This script hashes them (SHA-256) and compares
each hash with the snapshot's MANIFEST.txt, which the build wrote before
release and which no analysis step rewrites:

* camera_ready/analysis/cr_macros.json   every \\cr macro value of the paper
* camera_ready/analysis/*.csv            the CSVs the macros are computed from
* analysis/release_macros.tex            the release macros file

Provenance sidecars and cr_macros.tex are skipped: they carry a generation
date, and cr_macros.json carries the same values as cr_macros.tex.
swe_verify_results.csv is compared as shipped: the SWE harness that writes it
executes model-written code, so reproduce.sh does not run it.

The run is byte-identical on macOS arm64. On other platforms (other BLAS
builds of numpy and scipy) the last floating-point digits of intermediate CSV
values may differ while every printed number is identical. So when a hash
differs, the file is compared with the released copy that
`--save-reference` saved before the run (reproduce.sh does this; the run
overwrites the snapshot through the symlinks). The file passes "within
tolerance" when a CSV has the same shape and header, every non-numeric cell
is equal and every numeric cell is within rtol 1e-9 (atol 1e-12), and when
cr_macros.json has the same macros with every printed value (the `value`
field, the string the paper prints) and every other field equal, and every
numeric `raw` value within the same tolerance (their largest relative
difference is reported). Anything else fails, with the cells that differ.

The six camera-ready figures (figures/*.pdf) are not in the dataset, so they
are compared with scripts/figures_reference.json instead. The numbers each
figure draws (the `numbers_drawn` list of its provenance sidecar) must match.
The PDF bytes are compared too, but a byte difference only warns: the PDFs
embed the system font STIX Two Text (shipped with macOS), and another font
version or a fallback font changes the bytes, not the numbers.

Usage (from the repository root):

    python scripts/verify_reproduction.py --save-reference DIR   # before the run
    python scripts/verify_reproduction.py                        # after the run

The snapshot is found through the drift_bench/data/camera_ready symlink
(else drift_bench/data/hf_cache); `--snapshot <dir>` names it.

Exit status: 0 when every file is identical or identical at printed
precision, 1 on any difference, missing file or file not regenerated in this
run, 2 when no MANIFEST.txt is found (0 with --allow-missing-manifest).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "drift_bench" / "data"
HF_CACHE_DIR = DATA_ROOT / "hf_cache"
REFERENCE_DIR = DATA_ROOT / "released_reference"
FIGURES_DIR = ROOT / "figures"
FIGURE_REFERENCE = Path(__file__).resolve().parent / "figures_reference.json"
FIGURE_FONT = "STIX Two Text"

MACROS_JSON = "camera_ready/analysis/cr_macros.json"
CR_ANALYSIS = "camera_ready/analysis/"
RELEASE_MACROS = "analysis/release_macros.tex"
# Shipped files that reproduce.sh does not regenerate on purpose: exempt from
# the --since check, still compared with MANIFEST.txt.
NOT_REGENERATED_BY_DESIGN = {
    "camera_ready/analysis/swe_verify_results.csv":
        "swe_verify executes model-written code and is run by hand",
}
RTOL = 1e-9
ATOL = 1e-12
MAX_CELLS_SHOWN = 5
NO_COPY = ("hash differs and no released copy was saved to compare cells "
           "(reproduce.sh saves one with --save-reference)")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numbers_drawn_sha256(provenance_path: Path) -> str:
    """SHA-256 of the `numbers_drawn` list of a figure sidecar, in canonical JSON."""
    drawn = json.loads(provenance_path.read_text(encoding="utf-8"))["numbers_drawn"]
    canonical = json.dumps(drawn, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    """Parse MANIFEST.txt (`<sha256>  <relative path>` per line)."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        entries[rel.strip()] = digest
    return entries


def find_snapshot(data_root: Path = DATA_ROOT) -> Path:
    """The snapshot dir: where drift_bench/data/camera_ready links to, else hf_cache."""
    link = data_root / "camera_ready"
    if link.is_symlink():
        target = link.resolve().parent
        if (target / "MANIFEST.txt").is_file():
            return target
    return HF_CACHE_DIR


def manifest_groups(manifest: dict[str, str]) -> list[tuple[str, list[str]]]:
    """The files to check, grouped, as paths relative to the data root."""
    csvs = sorted(rel for rel in manifest
                  if rel.startswith(CR_ANALYSIS) and rel.endswith(".csv")
                  and "/" not in rel[len(CR_ANALYSIS):])
    return [
        ("cr_macros.json", [MACROS_JSON]),
        ("camera-ready CSVs", csvs),
        ("release_macros.tex", [RELEASE_MACROS]),
    ]


# ---------------------------------------------------------------------------
# Tolerant comparison (used only when the hash differs)
# ---------------------------------------------------------------------------


def _as_float(text):
    if isinstance(text, bool) or text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _close(a: float, b: float) -> tuple[bool, float]:
    """(within tolerance, relative difference) for two floats; b is the released value."""
    if math.isnan(a) and math.isnan(b):
        return True, 0.0
    if a == b:
        return True, 0.0
    diff = abs(a - b)
    rel = diff / abs(b) if b else math.inf
    return diff <= ATOL + RTOL * abs(b), rel


def compare_csv(new: Path, released: Path) -> tuple[bool, list[str], float]:
    """(passes within tolerance, differing cells, max relative difference)."""
    with new.open(newline="", encoding="utf-8") as fh:
        a = list(csv.reader(fh))
    with released.open(newline="", encoding="utf-8") as fh:
        b = list(csv.reader(fh))
    if len(a) != len(b):
        return False, [f"{len(a) - 1} rows vs released {len(b) - 1}"], math.nan
    if not a or a[0] != b[0]:
        return False, ["header differs"], math.nan
    header = b[0]
    problems: list[str] = []
    max_rel = 0.0
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), start=1):
        if len(ra) != len(rb):
            problems.append(f"row {i}: {len(ra)} cells vs released {len(rb)}")
            continue
        for j, (ca, cb) in enumerate(zip(ra, rb)):
            if ca == cb:
                continue
            fa, fb = _as_float(ca), _as_float(cb)
            col = header[j] if j < len(header) else str(j)
            if fa is None or fb is None:
                problems.append(f"row {i} column {col}: {ca!r} vs released {cb!r}")
                continue
            ok, rel = _close(fa, fb)
            if ok:
                max_rel = max(max_rel, rel)
            else:
                problems.append(f"row {i} column {col}: {ca} vs released {cb}")
    return not problems, problems, max_rel


def _raw_close(a, b) -> tuple[bool, float]:
    """Compare two `raw` macro fields: numbers within tolerance, lists elementwise."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, math.nan
        worst = 0.0
        for x, y in zip(a, b):
            ok, rel = _raw_close(x, y)
            if not ok:
                return False, rel
            worst = max(worst, rel)
        return True, worst
    if a == b:
        return True, 0.0
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or fb is None:
        return False, math.nan
    return _close(fa, fb)


def _raw_numeric(a, b) -> bool:
    """True when both `raw` fields are numbers, or lists of numbers of the same length."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_raw_numeric(x, y) for x, y in zip(a, b))
    return _as_float(a) is not None and _as_float(b) is not None


def compare_macros_json(new: Path, released: Path) -> tuple[bool, list[str], float]:
    """(passes, differing macros, max relative difference of the raw values)."""
    a = json.loads(new.read_text(encoding="utf-8"))
    b = json.loads(released.read_text(encoding="utf-8"))
    if set(a) != set(b):
        return False, ["top-level keys differ"], math.nan
    problems: list[str] = []
    for key in sorted(set(a) - {"macros"}):
        if a[key] != b[key]:
            problems.append(f"'{key}' differs")
    ma, mb = a.get("macros", []), b.get("macros", [])
    if [m.get("name") for m in ma] != [m.get("name") for m in mb]:
        return False, problems + ["the list of macros differs"], math.nan
    max_rel = 0.0
    for x, y in zip(ma, mb):
        if x.get("value") != y.get("value"):
            problems.append(f"{x.get('name')}: printed {x.get('value')!r} vs released {y.get('value')!r}")
            continue
        if {k: v for k, v in x.items() if k != "raw"} != {k: v for k, v in y.items() if k != "raw"}:
            problems.append(f"{x.get('name')}: a field other than the value differs")
            continue
        # A numeric raw value must also be within tolerance; a text raw value
        # (a model id, a pair label) is not judged beyond the equal printed value.
        if not _raw_numeric(x.get("raw"), y.get("raw")):
            continue
        ok, rel = _raw_close(x.get("raw"), y.get("raw"))
        if not ok:
            problems.append(f"{x.get('name')}: raw {x.get('raw')!r} vs released {y.get('raw')!r} "
                            f"(beyond rtol {RTOL:g})")
            continue
        max_rel = max(max_rel, rel)
    return not problems, problems, max_rel


def tolerant_compare(rel: str, new: Path, released: Path) -> tuple[bool, list[str], float]:
    if rel.endswith(".csv"):
        return compare_csv(new, released)
    if rel.endswith(".json"):
        return compare_macros_json(new, released)
    return False, ["bytes differ (no tolerance for this file type)"], math.nan


# ---------------------------------------------------------------------------
# Comparison against MANIFEST.txt
# ---------------------------------------------------------------------------


def _empty() -> dict[str, list]:
    return {k: [] for k in ("identical", "tolerance", "different", "missing", "unlisted", "stale")}


def compare(files: list[str], expected: dict[str, str], actual_root: Path,
            since: float | None = None, reference_dir: Path | None = None) -> dict[str, list]:
    """Compare files under `actual_root` with `expected` hashes.

    Returns lists keyed "identical" (paths), "tolerance" ((path, note)),
    "different" ((path, problems)), "missing", "unlisted" and "stale" (paths;
    files not rewritten after `since`, epoch seconds).
    """
    out = _empty()
    for rel in files:
        path = actual_root / rel
        if rel not in expected:
            out["unlisted"].append(rel)
        elif not path.is_file():
            out["missing"].append(rel)
        else:
            if (since is not None and path.stat().st_mtime < since
                    and rel not in NOT_REGENERATED_BY_DESIGN):
                out["stale"].append(rel)
            if sha256_of(path) == expected[rel]:
                out["identical"].append(rel)
                continue
            released = reference_dir / rel if reference_dir is not None else None
            if released is None or not released.is_file() or sha256_of(released) != expected[rel]:
                out["different"].append((rel, [NO_COPY]))
                continue
            ok, problems, max_rel = tolerant_compare(rel, path, released)
            if ok:
                out["tolerance"].append((rel, f"max relative difference {max_rel:.1e}"))
            else:
                out["different"].append((rel, problems))
    return out


def save_reference(snapshot: Path, reference_dir: Path, data_root: Path = DATA_ROOT,
                   allow_missing_manifest: bool = False) -> int:
    """Copy the released versions of the checked files to `reference_dir`.

    Each copy is confirmed against MANIFEST.txt first. A copy already in
    `reference_dir` that matches is kept, so a second run still has it.
    """
    manifest_path = snapshot / "MANIFEST.txt"
    if not manifest_path.is_file():
        print(f"verify_reproduction: no MANIFEST.txt in {_shown(snapshot)}; no released copies saved.")
        return 0 if allow_missing_manifest else 2
    manifest = read_manifest(manifest_path)
    saved = kept = 0
    not_saved: list[str] = []
    for _label, files in manifest_groups(manifest):
        for rel in files:
            target = reference_dir / rel
            if target.is_file() and sha256_of(target) == manifest[rel]:
                kept += 1
                continue
            source = next((p for p in (snapshot / rel, data_root / rel)
                           if p.is_file() and sha256_of(p) == manifest[rel]), None)
            if source is None:
                not_saved.append(rel)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            saved += 1
    print(f"verify_reproduction: released copies in {_shown(reference_dir)}: {saved} saved, "
          f"{kept} already there, {len(not_saved)} not matching MANIFEST.txt (not saved)")
    for rel in not_saved[:MAX_CELLS_SHOWN]:
        print(f"    {rel}")
    return 0


def figure_font_available() -> bool:
    try:
        from matplotlib import font_manager
        font_manager.findfont(font_manager.FontProperties(family=FIGURE_FONT), fallback_to_default=False)
        return True
    except Exception:  # noqa: BLE001 - any failure means the font is not usable
        return False


def check_figures(figures_dir: Path, reference_path: Path, since: float | None = None):
    """Compare the regenerated figures with the reference hashes.

    Returns (numbers result, pdf result) in the shape of `compare`.
    """
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["figures"]
    numbers, pdfs = _empty(), _empty()
    for name, ref in sorted(reference.items()):
        pdf = figures_dir / f"{name}.pdf"
        sidecar = figures_dir / f"{name}.provenance.json"
        shown = f"figures/{sidecar.name}"
        if not sidecar.is_file():
            numbers["missing"].append(shown)
        elif numbers_drawn_sha256(sidecar) != ref["numbers_drawn_sha256"]:
            numbers["different"].append((shown, ["numbers drawn differ from scripts/figures_reference.json"]))
        else:
            numbers["identical"].append(shown)
            if since is not None and sidecar.stat().st_mtime < since:
                numbers["stale"].append(shown)
        if not pdf.is_file():
            pdfs["missing"].append(f"figures/{pdf.name}")
        elif sha256_of(pdf) != ref["pdf_sha256"]:
            pdfs["different"].append((f"figures/{pdf.name}", ["PDF bytes differ"]))
        else:
            pdfs["identical"].append(f"figures/{pdf.name}")
    return numbers, pdfs


def write_figure_reference(figures_dir: Path, reference_path: Path) -> dict:
    """Write the reference hashes from the figures in `figures_dir` (maintainers)."""
    figures = {}
    for pdf in sorted(figures_dir.glob("fig_*.pdf")):
        sidecar = pdf.with_suffix(".provenance.json")
        figures[pdf.stem] = {
            "pdf_sha256": sha256_of(pdf),
            "numbers_drawn_sha256": numbers_drawn_sha256(sidecar),
        }
    data = {
        "note": ("SHA-256 of the six camera-ready figures and of the numbers_drawn list of "
                 "each provenance sidecar (canonical JSON: sorted keys, no spaces). "
                 "Written by scripts/verify_reproduction.py --write-figure-reference."),
        "font": FIGURE_FONT,
        "figures": figures,
    }
    reference_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _line(label: str, result: dict[str, list]) -> str:
    n_id, n_tol = len(result["identical"]), len(result["tolerance"])
    n_diff, n_miss = len(result["different"]), len(result["missing"]) + len(result["unlisted"])
    n = n_id + n_tol + n_diff + n_miss
    text = f"  {label:<26} {n_id}/{n} identical"
    if n_tol:
        text += f", {n_tol} identical at printed precision"
    if n_diff:
        text += f", {n_diff} DIFFERENT"
    if n_miss:
        text += f", {n_miss} missing"
    return text


def verify(snapshot: Path, data_root: Path = DATA_ROOT, figures_dir: Path | None = FIGURES_DIR,
           figure_reference: Path = FIGURE_REFERENCE, since: float | None = None,
           allow_missing_manifest: bool = False, reference_dir: Path | None = REFERENCE_DIR) -> int:
    manifest_path = snapshot / "MANIFEST.txt"
    if not manifest_path.is_file():
        print(f"verify_reproduction: no MANIFEST.txt in {_shown(snapshot)}; nothing to compare. "
              "Fetch the snapshot with scripts/fetch_from_hf.py or pass --snapshot <dir>.")
        return 0 if allow_missing_manifest else 2
    manifest = read_manifest(manifest_path)
    print(f"verify_reproduction: regenerated files under {_shown(data_root)} "
          f"vs {_shown(manifest_path)} ({len(manifest):,} entries)")

    tolerance: list[tuple[str, str]] = []
    different: list[tuple[str, list[str]]] = []
    missing: list[str] = []
    stale: list[str] = []
    for label, files in manifest_groups(manifest):
        result = compare(files, manifest, data_root, since, reference_dir)
        print(_line(label, result))
        tolerance += result["tolerance"]
        different += result["different"]
        missing += result["missing"] + [f"{rel} (not in MANIFEST.txt)" for rel in result["unlisted"]]
        stale += result["stale"]

    pdf_warning: list[str] = []
    if figures_dir is not None:
        numbers, pdfs = check_figures(figures_dir, figure_reference, since)
        print(_line("figures (numbers drawn)", numbers))
        different += numbers["different"]
        missing += numbers["missing"] + pdfs["missing"]
        stale += numbers["stale"]
        text = _line("figures (PDF bytes)", pdfs)
        print(text + ("  (warning only)" if pdfs["different"] else ""))
        if pdfs["different"]:
            font = "installed" if figure_font_available() else "not installed (a fallback font was used)"
            names = ", ".join(rel for rel, _ in pdfs["different"])
            pdf_warning.append(
                f"  WARNING: PDF bytes differ for {names}. The figures embed the font {FIGURE_FONT} "
                f"({font} here); another font version changes the bytes. The numbers drawn are checked above.")

    for rel, note in tolerance:
        print(f"  PASS-WITHIN-TOLERANCE {rel}: every printed value equal; {note}")
    for rel, problems in different:
        print(f"  FAIL {rel}:")
        for p in problems[:MAX_CELLS_SHOWN]:
            print(f"    {p}")
        if len(problems) > MAX_CELLS_SHOWN:
            print(f"    ... {len(problems) - MAX_CELLS_SHOWN} more")
    for rel in missing:
        print(f"  MISSING {rel}")
    for line in pdf_warning:
        print(line)
    for rel, why in NOT_REGENERATED_BY_DESIGN.items():
        if rel in manifest:
            print(f"  note: {rel} is compared as shipped ({why})")
    if stale:
        print(f"  NOT REGENERATED in this run ({len(stale)}): the file is older than the run.")
        for rel in stale:
            print(f"    {rel}")

    uncompared = [rel for rel, problems in different if problems == [NO_COPY]]
    if len(uncompared) < len(different):
        print(f"verify_reproduction: DIFFERENT. {len(different) - len(uncompared)} file(s) differ beyond "
              "the last float digits. Check that requirements-lock.txt is installed; other library "
              "versions can change values.")
    if uncompared:
        print(f"verify_reproduction: NOT COMPARED. {len(uncompared)} file(s) differ in hash and no released "
              "copy was saved before the run. Run scripts/reproduce.sh, which saves them.")
    if missing or stale:
        print(f"verify_reproduction: INCOMPLETE. {len(missing) + len(stale)} file(s) missing or not "
              "regenerated in this run: a step failed or was skipped. Check the analysis log above.")
    if different or missing or stale:
        return 1
    if tolerance:
        print("verify_reproduction: identical at printed precision (last float digits differ). "
              "Every printed number matches the release.")
    else:
        print("verify_reproduction: identical. Every checked file matches the release byte for byte.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", type=Path, default=None,
                   help="Dir with the snapshot's MANIFEST.txt (default: where drift_bench/data/"
                        "camera_ready links to, else drift_bench/data/hf_cache)")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help="Where the regenerated files are (default: drift_bench/data)")
    p.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR,
                   help="Released copies saved before the run (default: drift_bench/data/released_reference)")
    p.add_argument("--save-reference", type=Path, default=None, metavar="DIR",
                   help="Before the run: copy the released checked files to DIR (each confirmed "
                        "against MANIFEST.txt) and exit")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                   help="Where figures_cr wrote the figures (default: figures/)")
    p.add_argument("--no-figures", action="store_true", help="Skip the figure check")
    p.add_argument("--since", type=float, default=None,
                   help="Epoch seconds of the run start; a matching file older than this "
                        "was not regenerated and counts as a failure")
    p.add_argument("--allow-missing-manifest", action="store_true",
                   help="Exit 0 with a note when no MANIFEST.txt is found")
    p.add_argument("--write-figure-reference", action="store_true",
                   help="Maintainers: rewrite scripts/figures_reference.json from --figures-dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_figure_reference:
        data = write_figure_reference(args.figures_dir, FIGURE_REFERENCE)
        print(f"wrote {_shown(FIGURE_REFERENCE)} ({len(data['figures'])} figures)")
        return 0
    data_root = Path(os.path.abspath(args.data_root))
    snapshot = Path(os.path.abspath(args.snapshot if args.snapshot is not None else find_snapshot(data_root)))
    if args.save_reference is not None:
        return save_reference(snapshot, Path(os.path.abspath(args.save_reference)), data_root,
                              args.allow_missing_manifest)
    return verify(
        snapshot=snapshot,
        data_root=data_root,
        figures_dir=None if args.no_figures else Path(os.path.abspath(args.figures_dir)),
        since=args.since,
        allow_missing_manifest=args.allow_missing_manifest,
        reference_dir=Path(os.path.abspath(args.reference_dir)),
    )


if __name__ == "__main__":
    sys.exit(main())
