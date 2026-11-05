"""scripts/verify_reproduction.py and the fetch defaults.

The verify tests build a tiny synthetic snapshot (MANIFEST.txt plus a few
files) in a temp dir. No network, no released data needed.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = {
    "fig_complexity_vs_tokens", "fig_drift_checkpoint", "fig_drift_classification",
    "fig_kbv_by_judge_arm", "fig_sensitivity", "fig_trajectories",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verify_mod = _load("verify_reproduction")
fetch_mod = _load("fetch_from_hf")

MACROS = {
    "macros": [
        {"name": "\\crA", "csv": "a.csv", "fmt": "dec3", "value": "0.124", "raw": "0.12345678901234"},
        {"name": "\\crB", "csv": "a.csv", "fmt": "cidec2", "value": "[0.10, 0.20]",
         "raw": ["0.1012345678901", "0.2012345678901"]},
    ],
    "missing": [],
}
CSV_A = "model,estimate,p\ngpt,0.12345678901234,0.0312\nqwen,1.5,0.5\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def snapshot(tmp_path):
    """A snapshot dir with MANIFEST.txt, a data root with the regenerated
    files, a saved reference dir, and a figures dir with one figure."""
    files = {
        "camera_ready/analysis/cr_macros.json": json.dumps(MACROS, indent=1).encode(),
        "camera_ready/analysis/a.csv": CSV_A.encode(),
        "camera_ready/analysis/b.csv": b"x\n3\n",
        "camera_ready/analysis/a.csv.provenance.json": b'{"date": "2026-10-27"}\n',
        "camera_ready/e1/sub.csv": b"not checked\n",
        "analysis/release_macros.tex": b"\\newcommand{\\A}{1}\n",
    }
    snap = tmp_path / "snap"
    data = tmp_path / "data"
    for rel, content in files.items():
        _write(snap / rel, content)
        _write(data / rel, content)
    manifest = "".join(f"{_sha(c)}  {rel}\n" for rel, c in sorted(files.items()))
    (snap / "MANIFEST.txt").write_text(manifest)

    figs = tmp_path / "figures"
    sidecar = {"date": "2026-10-26", "numbers_drawn": [{"n": 76, "rate": 0.25}]}
    _write(figs / "fig_one.pdf", b"%PDF fake")
    _write(figs / "fig_one.provenance.json", json.dumps(sidecar).encode())
    reference = tmp_path / "figures_reference.json"
    reference.write_text(json.dumps({"figures": {"fig_one": {
        "pdf_sha256": _sha(b"%PDF fake"),
        "numbers_drawn_sha256": verify_mod.numbers_drawn_sha256(figs / "fig_one.provenance.json"),
    }}}))
    ref_dir = tmp_path / "released_reference"
    assert verify_mod.save_reference(snap, ref_dir, data) == 0
    return SimpleNamespace(snap=snap, data=data, figs=figs, reference=reference, ref_dir=ref_dir)


def _verify(s, **kw):
    kw.setdefault("reference_dir", s.ref_dir)
    return verify_mod.verify(snapshot=s.snap, data_root=s.data, figures_dir=s.figs,
                             figure_reference=s.reference, **kw)


def _set_macros(s, macros: dict) -> None:
    (s.data / verify_mod.MACROS_JSON).write_text(json.dumps(macros, indent=1))


def test_identical_files_pass(snapshot, capsys):
    assert _verify(snapshot) == 0
    out = capsys.readouterr().out
    assert "cr_macros.json             1/1 identical" in out
    assert "camera-ready CSVs          2/2 identical" in out  # sub-dir CSVs and sidecars are not checked
    assert "release_macros.tex         1/1 identical" in out
    assert "figures (numbers drawn)    1/1 identical" in out
    assert "verify_reproduction: identical." in out


def test_last_digit_differences_pass_within_tolerance(snapshot, capsys):
    # Another BLAS build: last float digits move, printed values do not.
    (snapshot.data / "camera_ready/analysis/a.csv").write_text(CSV_A.replace("0.12345678901234", "0.12345678901237"))
    macros = json.loads(json.dumps(MACROS))
    macros["macros"][0]["raw"] = "0.12345678901236"
    macros["macros"][1]["raw"] = ["0.1012345678903", "0.2012345678901"]
    _set_macros(snapshot, macros)
    assert _verify(snapshot) == 0
    out = capsys.readouterr().out
    assert "cr_macros.json             0/1 identical, 1 identical at printed precision" in out
    assert "camera-ready CSVs          1/2 identical, 1 identical at printed precision" in out
    assert "PASS-WITHIN-TOLERANCE camera_ready/analysis/a.csv" in out
    assert "PASS-WITHIN-TOLERANCE camera_ready/analysis/cr_macros.json" in out
    assert "identical at printed precision (last float digits differ)" in out


def test_printed_value_change_fails(snapshot, capsys):
    macros = json.loads(json.dumps(MACROS))
    macros["macros"][0]["value"] = "0.125"
    _set_macros(snapshot, macros)
    assert _verify(snapshot) == 1
    out = capsys.readouterr().out
    assert "FAIL camera_ready/analysis/cr_macros.json" in out
    assert "\\crA: printed '0.125' vs released '0.124'" in out
    assert "DIFFERENT" in out


def test_raw_macro_change_beyond_tolerance_fails(snapshot, capsys):
    # Same printed value, but the raw number moved beyond rtol 1e-9: FAIL.
    macros = json.loads(json.dumps(MACROS))
    macros["macros"][0]["raw"] = "0.12345678"
    _set_macros(snapshot, macros)
    assert _verify(snapshot) == 1
    out = capsys.readouterr().out
    assert "FAIL camera_ready/analysis/cr_macros.json" in out
    assert "\\crA: raw '0.12345678' vs released '0.12345678901234' (beyond rtol 1e-09)" in out


def test_raw_macro_list_change_beyond_tolerance_fails(snapshot, capsys):
    macros = json.loads(json.dumps(MACROS))
    macros["macros"][1]["raw"] = ["0.1012345678901", "0.2012"]
    _set_macros(snapshot, macros)
    assert _verify(snapshot) == 1
    assert "\\crB: raw ['0.1012345678901', '0.2012'] vs released" in capsys.readouterr().out


def test_text_raw_macro_is_not_judged(tmp_path):
    # A text raw value (a model id) is not a number; the equal printed value decides.
    base = {"macros": [{"name": "\\crM", "value": "GPT-5.4", "raw": "openai/gpt-5.4"}]}
    other = {"macros": [{"name": "\\crM", "value": "GPT-5.4", "raw": "openai/gpt-5.4 "}]}
    new, old = tmp_path / "new.json", tmp_path / "old.json"
    new.write_text(json.dumps(other))
    old.write_text(json.dumps(base))
    ok, problems, max_rel = verify_mod.compare_macros_json(new, old)
    assert ok and problems == [] and max_rel == 0.0


def test_string_cell_change_fails(snapshot, capsys):
    (snapshot.data / "camera_ready/analysis/a.csv").write_text(CSV_A.replace("qwen", "llama"))
    assert _verify(snapshot) == 1
    out = capsys.readouterr().out
    assert "FAIL camera_ready/analysis/a.csv" in out
    assert "row 2 column model: 'llama' vs released 'qwen'" in out


def test_numeric_change_beyond_tolerance_fails(snapshot, capsys):
    (snapshot.data / "camera_ready/analysis/a.csv").write_text(CSV_A.replace("0.0312", "0.0313"))
    assert _verify(snapshot) == 1
    assert "row 1 column p: 0.0313 vs released 0.0312" in capsys.readouterr().out


def test_shape_change_fails(snapshot, capsys):
    (snapshot.data / "camera_ready/analysis/b.csv").write_bytes(b"x\n3\n4\n")
    assert _verify(snapshot) == 1
    out = capsys.readouterr().out
    assert "FAIL camera_ready/analysis/b.csv:\n    2 rows vs released 1" in out


def test_hash_difference_without_saved_copy_fails(snapshot, capsys):
    (snapshot.data / "camera_ready/analysis/b.csv").write_bytes(b"x\n3.0000000000001\n")
    assert _verify(snapshot, reference_dir=snapshot.snap.parent / "nowhere") == 1
    out = capsys.readouterr().out
    assert "no released copy was saved" in out
    assert "NOT COMPARED" in out and "library" not in out


def test_missing_file_is_incomplete_not_a_library_issue(snapshot, capsys):
    (snapshot.data / "analysis/release_macros.tex").unlink()
    assert _verify(snapshot) == 1
    out = capsys.readouterr().out
    assert "MISSING analysis/release_macros.tex" in out
    assert "INCOMPLETE" in out
    assert "library" not in out


def test_sidecar_date_change_is_ignored(snapshot):
    (snapshot.data / "camera_ready/analysis/a.csv.provenance.json").write_text('{"date": "2026-11-05"}\n')
    sidecar = snapshot.figs / "fig_one.provenance.json"
    d = json.loads(sidecar.read_text())
    d["date"] = "2026-11-05"
    sidecar.write_text(json.dumps(d, indent=2))  # other formatting, same numbers
    assert _verify(snapshot) == 0


def test_figure_numbers_change_fails(snapshot, capsys):
    sidecar = snapshot.figs / "fig_one.provenance.json"
    d = json.loads(sidecar.read_text())
    d["numbers_drawn"][0]["rate"] = 0.26
    sidecar.write_text(json.dumps(d))
    assert _verify(snapshot) == 1
    assert "FAIL figures/fig_one.provenance.json" in capsys.readouterr().out


def test_figure_pdf_bytes_only_warn(snapshot, capsys):
    (snapshot.figs / "fig_one.pdf").write_bytes(b"%PDF other font")
    assert _verify(snapshot) == 0
    out = capsys.readouterr().out
    assert "figures (PDF bytes)        0/1 identical, 1 DIFFERENT  (warning only)" in out
    assert "WARNING: PDF bytes differ" in out


def _age_all(s, seconds: int) -> None:
    for path in list(s.data.rglob("*")) + list(s.figs.iterdir()):
        if path.is_file():
            os.utime(path, (seconds, seconds))


def test_file_not_rewritten_in_this_run_fails(snapshot, capsys):
    _age_all(snapshot, 2_000_000_000)
    os.utime(snapshot.data / "camera_ready/analysis/a.csv", (1_000_000_000, 1_000_000_000))
    assert _verify(snapshot, since=1_500_000_000) == 1
    out = capsys.readouterr().out
    assert "NOT REGENERATED in this run (1)" in out
    assert "INCOMPLETE" in out and "library" not in out


def test_differing_file_not_rewritten_in_this_run_fails(snapshot, capsys):
    # A within-tolerance file left over from an earlier run still counts as not regenerated.
    _age_all(snapshot, 2_000_000_000)
    path = snapshot.data / "camera_ready/analysis/a.csv"
    path.write_text(CSV_A.replace("0.12345678901234", "0.12345678901237"))
    os.utime(path, (1_000_000_000, 1_000_000_000))
    assert _verify(snapshot, since=1_500_000_000) == 1
    out = capsys.readouterr().out
    assert "PASS-WITHIN-TOLERANCE camera_ready/analysis/a.csv" in out
    assert "NOT REGENERATED in this run (1)" in out


def test_hand_run_file_is_exempt_from_the_since_check(snapshot, capsys, monkeypatch):
    monkeypatch.setattr(verify_mod, "NOT_REGENERATED_BY_DESIGN",
                        {"camera_ready/analysis/b.csv": "run by hand"})
    _age_all(snapshot, 2_000_000_000)
    os.utime(snapshot.data / "camera_ready/analysis/b.csv", (1_000_000_000, 1_000_000_000))
    assert _verify(snapshot, since=1_500_000_000) == 0
    assert "camera_ready/analysis/b.csv is compared as shipped (run by hand)" in capsys.readouterr().out


def test_save_reference_keeps_only_released_copies(snapshot, capsys):
    # The saved copies match MANIFEST; a regenerated file is not saved over them.
    (snapshot.data / "camera_ready/analysis/b.csv").write_bytes(b"x\n4\n")
    (snapshot.snap / "camera_ready/analysis/b.csv").write_bytes(b"x\n4\n")
    assert verify_mod.save_reference(snapshot.snap, snapshot.ref_dir, snapshot.data) == 0
    assert "0 saved, 4 already there" in capsys.readouterr().out
    assert (snapshot.ref_dir / "camera_ready/analysis/b.csv").read_bytes() == b"x\n3\n"
    fresh = snapshot.snap.parent / "fresh_reference"
    verify_mod.save_reference(snapshot.snap, fresh, snapshot.data)
    assert "3 saved, 0 already there, 1 not matching MANIFEST.txt" in capsys.readouterr().out
    assert not (fresh / "camera_ready/analysis/b.csv").exists()


def test_missing_manifest(snapshot, capsys):
    (snapshot.snap / "MANIFEST.txt").unlink()
    assert _verify(snapshot) == 2
    assert _verify(snapshot, allow_missing_manifest=True) == 0
    assert verify_mod.save_reference(snapshot.snap, snapshot.ref_dir, snapshot.data) == 2
    assert verify_mod.save_reference(snapshot.snap, snapshot.ref_dir, snapshot.data,
                                     allow_missing_manifest=True) == 0
    assert "no MANIFEST.txt" in capsys.readouterr().out


def test_snapshot_found_through_symlink(tmp_path):
    snap = tmp_path / "cache"
    (snap / "camera_ready").mkdir(parents=True)
    (snap / "MANIFEST.txt").write_text("")
    data = tmp_path / "data"
    data.mkdir()
    (data / "camera_ready").symlink_to(snap / "camera_ready")
    assert verify_mod.find_snapshot(data) == snap.resolve()


def test_figure_reference_lists_the_six_figures():
    ref = json.loads(verify_mod.FIGURE_REFERENCE.read_text())["figures"]
    assert set(ref) == FIGURE_NAMES
    for entry in ref.values():
        assert len(entry["pdf_sha256"]) == 64
        assert len(entry["numbers_drawn_sha256"]) == 64


@pytest.mark.skipif(not (ROOT / "figures" / "fig_trajectories.provenance.json").is_file(),
                    reason="figures/ not generated (run figures_cr or reproduce.sh)")
def test_figure_reference_matches_the_numbers_of_figures_on_disk():
    ref = json.loads(verify_mod.FIGURE_REFERENCE.read_text())["figures"]
    for name, entry in ref.items():
        sidecar = ROOT / "figures" / f"{name}.provenance.json"
        assert verify_mod.numbers_drawn_sha256(sidecar) == entry["numbers_drawn_sha256"], name


def test_fetch_defaults_to_the_camera_ready_tag():
    assert fetch_mod.DEFAULT_REVISION == "camera-ready"
    args = fetch_mod.build_parser().parse_args([])
    assert args.revision == "camera-ready"
    assert fetch_mod.build_parser().parse_args(["--revision", "main"]).revision == "main"


def test_fetch_http_errors_print_the_token_hint():
    class RateLimited(OSError):
        response = SimpleNamespace(status_code=429)

    msg = fetch_mod.download_error_message(RateLimited("429"), "driftbench/DriftBench", "camera-ready")
    assert msg.startswith("ERROR: download stopped (HTTP 429, rate limit).")
    assert "HF_TOKEN" in msg and "re-run" in msg and "\n" not in msg
    assert "HF_TOKEN" in fetch_mod.download_error_message(ConnectionError("reset"), "r", "camera-ready")
    # Not an HTTP or network error: no message, so the traceback is kept.
    assert fetch_mod.download_error_message(ValueError("bug"), "r", "camera-ready") is None


def test_fetch_reads_the_status_of_a_chained_error():
    # A 429 on the repo lookup surfaces as a local-entry error caused by the HTTP error.
    class RateLimited(OSError):
        response = SimpleNamespace(status_code=429)

    try:
        try:
            raise RateLimited("429")
        except RateLimited as cause:
            raise FileNotFoundError("cannot find the requested files locally") from cause
    except FileNotFoundError as exc:
        msg = fetch_mod.download_error_message(exc, "r", "camera-ready")
    assert msg.startswith("ERROR: download stopped (HTTP 429, rate limit).")


def test_fetch_flags_an_incomplete_download(tmp_path):
    cache = tmp_path / "cache"
    files = {"a/x.json": b"{}", "b/y.csv": b"1\n", "c.parquet": b"PAR1"}
    for rel, content in files.items():
        _write(cache / rel, content)
    (cache / "MANIFEST.txt").write_text("".join(f"{_sha(c)}  {rel}\n" for rel, c in files.items()))
    assert fetch_mod.incomplete_download_message(cache) is None
    (cache / "b/y.csv").unlink()
    msg = fetch_mod.incomplete_download_message(cache)
    assert msg.startswith("ERROR: download incomplete (1 of 3 listed files missing).")
    assert "HF_TOKEN" in msg
    (cache / "MANIFEST.txt").write_text("\n")
    assert fetch_mod.incomplete_download_message(cache).startswith(
        "ERROR: download incomplete (MANIFEST.txt lists no files).")
    (cache / "MANIFEST.txt").unlink()
    msg = fetch_mod.incomplete_download_message(cache)
    assert msg.startswith("ERROR: download incomplete (no MANIFEST.txt).")
    assert "listed files missing" not in msg and "HF_TOKEN" in msg
