"""Milestone 9 — consolidated adversarial fixture suite, MCP layer.

Closes the gap the roadmap's Milestone 9 entry names: analysis-level
tests (test_confound.py, test_divergence.py, test_sensitivity.py) and
scattered tool-level tests (test_server.py, added incrementally across
Milestones 4-8) both already exercise every spec §7 adversarial case
correctly — this suite doesn't change that behavior at all. What it adds
is a single, spec-numbered index: one file that lines up all six
design-spec-v1.md §7 point-2 cases against the actual MCP protocol round
trip (`fastmcp.Client(mcp).call_tool(...)`), so "is every §7 case covered
end-to-end through the MCP layer" is answerable by reading this file
alone, not by cross-referencing five others.

Fixture construction lives in `tests/fixtures/adversarial_cases.py`
(`ADVERSARIAL_CASES`), not here, so the *data* for each case and the
*assertion* that it produces the correct MCP-layer behavior stay
separately reviewable — a reviewer checking "did they build the right
adversarial scenario" doesn't have to read pytest parametrization
machinery to do it.
"""

from __future__ import annotations

import pytest
from fastmcp import Client

from experiment_audit_mcp.server import build_server
from tests.fixtures.adversarial_cases import ADVERSARIAL_CASES


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=[case.name for case in ADVERSARIAL_CASES],
)
@pytest.mark.asyncio
async def test_spec_section_7_adversarial_case_through_mcp_layer(case):
    """One assertion group per design-spec-v1.md §7 point-2 bullet,
    invoked exactly as a real MCP client would: through the protocol
    layer, not by calling the underlying Python function directly.
    """
    backend = case.build_backend()
    mcp = build_server(backends={"fake": backend})

    async with Client(mcp) as client:
        result = await client.call_tool(case.tool_name, case.tool_args())

    case.assert_result(result.data)


def test_all_six_spec_section_7_cases_are_present():
    """A cheap guard against silent fixture-set shrinkage: spec §7 point 2
    names exactly six adversarial/edge cases. If this count ever drifts,
    it means a case was removed (or the spec grew a new one that hasn't
    been added here yet) — either way, that's worth a deliberate look,
    not a silent pass.
    """
    assert len(ADVERSARIAL_CASES) == 6
    assert len({case.name for case in ADVERSARIAL_CASES}) == 6, "case names must be unique"


def test_every_case_documents_its_spec_reference():
    """Each case's `spec_ref` should read as a real quoted bullet, not a
    placeholder — this is what makes the module auditable against the
    spec text by eye."""
    for case in ADVERSARIAL_CASES:
        assert case.spec_ref, case.name
        assert len(case.spec_ref) > 20, f"{case.name}: spec_ref looks like a placeholder"
