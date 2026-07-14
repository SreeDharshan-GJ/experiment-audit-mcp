"""
Regression tests for release metadata.

These exist because `pyproject.toml`'s `[project.urls]` table shipped in
the v1.0.0 release with literal `your-username` placeholders instead of
the real GitHub org (`SreeDharshan-GJ`) -- a real bug, not a hypothetical
one; see CHANGELOG.md's v1.0.0 entry and the fix that accompanies this
test. Nothing else in the test suite touches packaging metadata, so a
future edit to `pyproject.toml` (e.g. bumping the version, adding a URL)
could silently reintroduce a placeholder or a typo'd org name with no
test catching it.

This file does not validate the *content* of docs (README prose,
changelog history, etc.) -- only the structured, load-bearing package
metadata that PyPI, `pip install`, and MCP registries actually read.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# Patterns that indicate a template value was never filled in. Kept
# general (not just "your-username") so this test also catches the next
# placeholder convention someone reaches for, e.g. <org>, TODO, CHANGEME.
PLACEHOLDER_PATTERNS = [
    re.compile(r"your-username", re.IGNORECASE),
    re.compile(r"<[a-z-]+>"),  # e.g. <your-username>, <org>
    re.compile(r"\bTODO\b"),
    re.compile(r"\bCHANGEME\b", re.IGNORECASE),
]

EXPECTED_GITHUB_ORG = "SreeDharshan-GJ"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def test_pyproject_urls_have_no_placeholder_values():
    """No [project.urls] entry contains an unfilled template placeholder."""
    data = _load_pyproject()
    urls = data["project"]["urls"]
    assert urls, "[project.urls] must not be empty"

    offenders = []
    for name, url in urls.items():
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(url):
                offenders.append((name, url, pattern.pattern))

    assert not offenders, (
        f"Placeholder value(s) found in [project.urls] -- these ship to PyPI verbatim: {offenders}"
    )


def test_pyproject_urls_point_to_the_real_github_org():
    """Every github.com URL in [project.urls] points at the real org/repo."""
    data = _load_pyproject()
    urls = data["project"]["urls"]

    github_urls = {name: url for name, url in urls.items() if "github.com" in url}
    assert github_urls, "expected at least one github.com URL in [project.urls]"

    for name, url in github_urls.items():
        assert f"github.com/{EXPECTED_GITHUB_ORG}/" in url, (
            f"[project.urls].{name} = {url!r} does not point at "
            f"github.com/{EXPECTED_GITHUB_ORG}/ -- did the org name get "
            "typo'd or left as a placeholder?"
        )


def test_changelog_release_links_have_no_placeholder_values():
    """CHANGELOG.md's version links (e.g. `[1.0.0]: https://...`) are filled in.

    Regression coverage for the same bug class as the pyproject checks
    above: CHANGELOG.md's v1.0.0 release-tag link shipped as
    `https://github.com/<your-username>/...`.
    """
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    # Only check version-link reference lines, e.g. "[1.0.0]: https://...",
    # not prose elsewhere in the file (which may legitimately show a
    # generic `git clone` example for contributors forking the repo).
    version_link_re = re.compile(r"^\[\d+\.\d+\.\d+\]:\s+https?://")
    link_lines = [line for line in changelog.splitlines() if version_link_re.match(line)]
    assert link_lines, "expected at least one '[x.y.z]: https://...' release link in CHANGELOG.md"

    offenders = []
    for line in link_lines:
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(line):
                offenders.append(line)

    assert not offenders, f"Placeholder value(s) found in CHANGELOG.md release links: {offenders}"
def test_readme_has_no_placeholder_values():
    """README.md contains no unfilled template placeholders.

    Regression coverage for the same bug class as the pyproject/CHANGELOG
    checks above, but for README.md specifically: it shipped a
    `git clone https://github.com/<your-username>/...` instruction in its
    Development section that the earlier, narrower checks (which only
    looked at [project.urls] and CHANGELOG.md release links) never
    covered.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    offenders = []
    for line_no, line in enumerate(readme.splitlines(), start=1):
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(line):
                offenders.append((line_no, line.strip(), pattern.pattern))

    assert not offenders, f"Placeholder value(s) found in README.md: {offenders}"
    