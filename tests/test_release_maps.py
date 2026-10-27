"""The HF release copy plan and the HF fetch path map must stay inverse.

`scripts/build_dataset_release.py::COPY_PLAN` maps repo paths to HF paths.
`scripts/fetch_from_hf.py::PATH_MAP` maps HF paths back to repo paths. Every
data entry shipped must be fetchable at the same local path, or the
`reproduce.sh --hf` loop silently misses files.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _covered(src: str, dst: str, path_map: list[tuple[str, str]]) -> bool:
    """A copy entry is covered if a fetch entry maps the same HF path (or a
    parent directory of it) back to the same repo path (or its parent)."""
    for hf, local in path_map:
        if dst == hf and src == local:
            return True
        if dst.startswith(hf + "/") and src.startswith(local + "/"):
            if dst[len(hf) + 1:] == src[len(local) + 1:]:
                return True
    return False


def test_every_data_copy_entry_has_an_inverse_fetch_entry() -> None:
    build = _load("build_dataset_release")
    fetch = _load("fetch_from_hf")
    data_entries = [(s, d) for s, d in build.COPY_PLAN if s.startswith("drift_bench/data/")]
    missing = [(s, d) for s, d in data_entries if not _covered(s, d, fetch.PATH_MAP)]
    assert not missing, f"COPY_PLAN entries with no inverse in PATH_MAP: {missing}"


def test_every_fetch_entry_has_a_copy_entry() -> None:
    build = _load("build_dataset_release")
    fetch = _load("fetch_from_hf")
    copy_dsts = {d for _, d in build.COPY_PLAN}
    copy_dsts |= {d for _, d in build.ANNOTATION_FILE_MAP}
    missing = []
    for hf, local in fetch.PATH_MAP:
        ok = hf in copy_dsts or any(d.startswith(hf + "/") for d in copy_dsts)
        if not ok:
            missing.append((hf, local))
    assert not missing, f"PATH_MAP entries nothing in COPY_PLAN produces: {missing}"


def test_camera_ready_data_is_in_both_maps() -> None:
    build = _load("build_dataset_release")
    fetch = _load("fetch_from_hf")
    assert ("drift_bench/data/camera_ready", "camera_ready") in build.COPY_PLAN
    assert ("camera_ready", "drift_bench/data/camera_ready") in fetch.PATH_MAP


def test_run_summaries_the_camera_ready_modules_read_ship_and_link() -> None:
    """failed_runs, corrections and sensitivity read every run_summary.json.

    Without them a fresh clone fed from the HF snapshot fails those three
    modules (F9 fresh-clone check, 2026-10-27).
    """
    from drift_bench.analysis.camera_ready.failed_runs import RUN_SUMMARY_DIRS

    build = _load("build_dataset_release")
    fetch = _load("fetch_from_hf")
    copied = {s for s, _ in build.COPY_PLAN}
    linked = {local for _, local in fetch.PATH_MAP}
    for rel in RUN_SUMMARY_DIRS:
        local = f"drift_bench/data/{rel}/run_summary.json" if rel else "drift_bench/data/run_summary.json"
        assert local in copied, f"{local} is not in COPY_PLAN"
        assert local in linked, f"{local} is not in PATH_MAP"
