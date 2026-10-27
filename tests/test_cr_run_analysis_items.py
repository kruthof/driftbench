"""X7: run_analysis --items accepts numbers and names; item 6 discovers camera_ready modules."""

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from drift_bench.run_analysis import (
    ALL_ITEMS,
    ITEM_NAMES,
    discover_camera_ready_modules,
    parse_items,
    run_item_6,
)


def test_all_includes_item_6() -> None:
    assert ALL_ITEMS == [1, 2, 3, 4, 5, 6]
    assert parse_items(["all"]) == [1, 2, 3, 4, 5, 6]
    assert ITEM_NAMES["camera_ready"] == 6


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["1", "2", "3"], [1, 2, 3]),          # --items 1 2 3
        (["1,2,3"], [1, 2, 3]),                # --items 1,2,3 (old form)
        ("1,2", [1, 2]),                       # bare string
        (["camera_ready"], [6]),
        (["camera-ready"], [6]),
        (["Reliability", "6"], [1, 6]),
        (["3", "probes", "3"], [3]),           # duplicates dropped
        (["1, 2 ,", "validation"], [1, 2, 4]),  # whitespace and empty parts
    ],
)
def test_parse_items(tokens, expected) -> None:
    assert parse_items(tokens) == expected


@pytest.mark.parametrize("bad", [["7"], ["0"], ["nope"], ["1", "x"]])
def test_parse_items_rejects_unknown(bad) -> None:
    with pytest.raises(ValueError):
        parse_items(bad)


def test_real_package_discovery_returns_list() -> None:
    found = discover_camera_ready_modules()
    assert isinstance(found, list)
    for name, module in found:
        assert callable(module.run), name
    names = [name for name, _ in found]
    assert names[-1] == "macros", "macros must regenerate after every other module"
    assert "swe_verify" not in names, "the SWE harness executes model code; never run it implicitly"


def test_discovery_on_synthetic_package(tmp_path: Path, monkeypatch) -> None:
    pkg = tmp_path / "cr_synthetic_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text(
        "import pandas as pd\n"
        "def run(data_root, out_dir):\n"
        "    return {'alpha_table': pd.DataFrame({'x': [1, 2]})}\n"
    )
    (pkg / "beta.py").write_text("VALUE = 1\n")  # no run()
    (pkg / "_private.py").write_text("def run(data_root, out_dir):\n    return {}\n")
    (pkg / "harness.py").write_text(
        "RUN_IN_ITEM_6 = False\n"
        "def run(data_root, out_dir):\n    return {}\n"
    )
    (pkg / "macros.py").write_text("def run(data_root, out_dir):\n    return {}\n")
    (pkg / "aaa_after_macros.py").write_text("def run(data_root, out_dir):\n    return {}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        found = discover_camera_ready_modules("cr_synthetic_pkg")
        # harness opts out; macros runs last even though "aaa_..." sorts after it.
        assert [name for name, _ in found] == ["aaa_after_macros", "alpha", "macros"]
        result = dict(found)["alpha"].run(tmp_path, tmp_path)
        assert list(result) == ["alpha_table"]
    finally:
        for key in [k for k in sys.modules if k.startswith("cr_synthetic_pkg")]:
            del sys.modules[key]


def test_run_item_6_runs_every_module_and_survives_failures(tmp_path: Path, monkeypatch) -> None:
    good = ModuleType("good")
    good.run = lambda data_root, out_dir: {"t": pd.DataFrame({"a": [1]})}
    bad = ModuleType("bad")

    def _fail(data_root, out_dir):
        raise RuntimeError("boom")
    bad.run = _fail

    nondict = ModuleType("nondict")
    nondict.run = lambda data_root, out_dir: None  # returns nothing
    listy = ModuleType("listy")
    listy.run = lambda data_root, out_dir: [1, 2]  # wrong type

    monkeypatch.setattr(
        "drift_bench.run_analysis.discover_camera_ready_modules",
        lambda package_name="x": [("bad", bad), ("nondict", nondict), ("listy", listy), ("good", good)],
    )
    out = tmp_path / "out"
    results = run_item_6(tmp_path, out)
    assert out.is_dir()
    assert results["bad"] == {"error": "boom"}
    assert "error" in results["nondict"] and "NoneType" in results["nondict"]["error"]
    assert "error" in results["listy"] and "list" in results["listy"]["error"]
    # The module after the bad ones still ran.
    assert len(results["good"]["t"]) == 1
