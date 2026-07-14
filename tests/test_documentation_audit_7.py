"""Regression tests for documentation-correctness defects found and fixed in
Audit #7 (documentation/examples/DX audit only -- see the audit report).

These tests do not exercise the MCP server, W&B backend, reasoning engine,
or reporting layer; they only check that README.md / CHANGELOG.md / docs/*
stay internally consistent and consistent with the shipped `server.py`
behavior that they describe, and that the two example scripts backing
`docs/PROOF_CASE.md`'s claims still actually produce the numbers/verdict the
doc quotes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CHANGELOG.md",
    *sorted((REPO_ROOT / "docs").glob("*.md")),
]

# Backtick-quoted paths under these top-level dirs, as referenced from docs.
_PATH_RE = re.compile(
    r"`((?:docs|tests|scripts|src|validation|examples)/[A-Za-z0-9_./-]*[A-Za-z0-9_])`"
)


def test_all_referenced_paths_in_docs_exist():
    """Every backtick-quoted docs/tests/scripts/src/validation/examples path
    mentioned in README, CHANGELOG, or docs/*.md must exist on disk.
    Catches stale references to renamed/removed files."""
    missing = []
    for doc in DOC_FILES:
        text = doc.read_text(encoding="utf-8")
        for match in _PATH_RE.findall(text):
            if not (REPO_ROOT / match).exists():
                missing.append(f"{doc.relative_to(REPO_ROOT)} references missing path: {match}")
    assert not missing, "\n".join(missing)


def test_readme_does_not_claim_test_connection_autoruns_on_start():
    """`test_connection` is never invoked automatically by `build_server()`/
    `main()` in server.py -- only WANDB_API_KEY *presence* is fail-fast
    checked at startup (auth.py). The README must not claim the tool itself
    runs automatically on start (Audit #7 finding #3)."""
    assert "runs automatically on server start" not in README
    # The corrected wording should still explain the real fail-fast behavior.
    assert "not invoked automatically on server start" in README or (
        "not... invoked automatically" in README
    ) or "call it explicitly" in README.lower() or "not invoked automatically" in README.lower()


def test_pip_install_from_pypi_is_caveated():
    """`pip install experiment-audit-mcp` currently fails (package not yet
    published to PyPI, confirmed live during Audit #7). The README's Install
    section must not present it as the sole/first working install path
    without the "not yet published" caveat, and must offer a from-source
    install that actually works today."""
    install_section = README.split("## Install", 1)[1].split("## Quick start", 1)[0]
    lowered = install_section.lower()
    assert "not yet published" in lowered or "not published" in lowered
    assert "git clone" in install_section
    assert "pip install -e ." in install_section


def test_claude_code_command_puts_options_before_server_name():
    """Per Claude Code's documented CLI syntax
    (`claude mcp add [options] <name> -- <command>`), option flags like
    `-e` must precede the positional server name. Putting `-e` after the
    name has repeatedly triggered 'Invalid environment variable format'
    failures in Claude Code (Audit #7 finding #2)."""
    match = re.search(r"claude mcp add ([^\n]+)", README)
    assert match, "Claude Code install command not found in README"
    command_line = match.group(1)
    e_flag_index = command_line.index("-e ")
    name_index = command_line.index("experiment-audit ")
    assert e_flag_index < name_index, (
        f"-e flag must appear before the server name in: {command_line!r}"
    )


def test_proof_case_example_still_produces_documented_verdict():
    """`docs/PROOF_CASE.md` quotes exact accuracy numbers and a `confounded`
    / `high` verdict produced by running `examples/run_experiment.py` then
    `examples/run_audit.py`. Re-run both for real and check the verdict
    (not the exact floats, which are legitimately sensitive to sklearn/numpy
    version drift) still matches what the doc claims, so the doc doesn't
    silently go stale if the analysis logic ever changes."""
    proof_case = (REPO_ROOT / "docs" / "PROOF_CASE.md").read_text(encoding="utf-8")
    assert '"verdict": "confounded"' in proof_case
    assert '"confidence": "high"' in proof_case

    subprocess.run(
        [sys.executable, "examples/run_experiment.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, "examples/run_audit.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output_json = result.stdout.split("===\n", 1)[1]
    audit = json.loads(output_json)
    assert audit["verdict"] == "confounded"
    assert audit["confidence"] == "high"
    param_names = {p["param"] for p in audit["differing_params"]}
    assert {"use_memory", "batch_size"} <= param_names

