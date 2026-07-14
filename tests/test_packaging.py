"""Regression tests for Audit #6 (packaging / release engineering).

Two real bugs were found and fixed here:

1. `src/experiment_audit_mcp/reasoning/__init__.py` did not exist, unlike
   every sibling package (`analysis/`, `backends/`,
   `reasoning/scientific_rules/`). Because Python 3 supports implicit
   namespace packages, imports of `experiment_audit_mcp.reasoning.*`
   still worked and the wheel still happened to contain the loose `.py`
   files -- but `setuptools.find_packages()` (the mechanism
   `[tool.setuptools.packages.find]` uses) silently did NOT register
   `experiment_audit_mcp.reasoning` or
   `experiment_audit_mcp.reasoning.scientific_rules` as packages at all.
   That's a landmine: any future setuptools/packaging change that relies
   on the registered package list (package_data, py.typed markers,
   stricter wheel builders, etc.) could silently drop the entire
   reasoning engine from a release with no build error.

2. There was no `MANIFEST.in`. Without one, setuptools' default sdist
   file list only picks up a flat `tests/test*.py` glob (a legacy
   distutils convention) and does not recurse into subdirectories, so
   `tests/reasoning/`, `tests/validation/`, and `tests/fixtures/` (more
   than half of the suite) were silently missing from
   `experiment-audit-mcp-*.tar.gz`. Anyone installing from the sdist and
   running `pytest` to verify the release would run an incomplete suite
   with no error or warning.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

from setuptools import find_packages

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def test_reasoning_is_a_registered_package():
    """`experiment_audit_mcp.reasoning` must be discovered by find_packages().

    This is what `[tool.setuptools.packages.find]` uses at build time.
    A package that isn't in this list isn't a package as far as
    setuptools is concerned, regardless of whether its modules happen to
    get swept into the wheel some other way.
    """
    packages = set(find_packages(where=str(SRC_ROOT)))

    assert "experiment_audit_mcp.reasoning" in packages
    assert "experiment_audit_mcp.reasoning.scientific_rules" in packages


def test_every_module_directory_under_src_has_an_init_file():
    """Every directory containing .py modules has __init__.py.

    Broader guard than the single `reasoning` case above: catches the
    same class of bug anywhere else under src/, now or in the future.
    """
    missing = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "egg-info" in path.parts or "__pycache__" in path.parts:
            continue
        package_dir = path.parent
        if not (package_dir / "__init__.py").exists():
            missing.append(str(package_dir.relative_to(SRC_ROOT)))

    assert not missing, f"Directories with .py modules but no __init__.py: {sorted(set(missing))}"


def test_sdist_contains_the_full_test_suite(tmp_path):
    """`python -m build --sdist` output must contain every tests/**/*.py file.

    Builds a real sdist (no network needed: --no-isolation reuses the
    already-installed build backend) and inspects the resulting tarball,
    which is the only way to actually catch this class of bug -- unit
    tests that merely check for a MANIFEST.in's existence wouldn't catch
    a typo'd pattern inside it.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(tmp_path),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"sdist build failed:\n{result.stdout}\n{result.stderr}"

    (sdist_path,) = tmp_path.glob("*.tar.gz")
    with tarfile.open(sdist_path) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]

    # Strip the leading "<name>-<version>/" component tarballs are rooted at.
    shipped = {name.split("/", 1)[1] for name in members if "/" in name}

    expected = {
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").rglob("*.py")
    }

    missing = expected - shipped
    assert not missing, f"Test files missing from sdist: {sorted(missing)}"