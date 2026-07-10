"""Offline sanity checks for the Milestone 9 tool-selection eval.

These do not call a live model (see docs/tool-selection-eval.md for why
that hasn't happened yet) — they check the parts of
scripts/tool_selection_eval.py and scripts/tool_selection_prompts.py that
*can* be verified without one: the prompt set is well-formed, every
`expected_tool` names a tool the server actually registers, the set
covers all eight tools, and the harness's MCP-to-Anthropic tool-schema
conversion produces a shape the Messages API expects. This keeps the
harness from silently bit-rotting (e.g. if a tool is renamed) even though
the eval itself can't run in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tool_selection_eval import _mcp_tools_as_anthropic_tools  # noqa: E402
from tool_selection_prompts import TOOL_SELECTION_PROMPTS  # noqa: E402

from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.server import build_server

_ALL_TOOL_NAMES = {
    "test_connection",
    "list_runs",
    "get_run_summary",
    "get_metric_history",
    "compare_runs",
    "audit_training_curve",
    "audit_ablation",
    "audit_sweep",
}


def test_prompt_set_has_at_least_ten_and_at_most_fifteen_prompts():
    # Roadmap: "a fixed prompt set (~10-15 representative phrasings)".
    assert 10 <= len(TOOL_SELECTION_PROMPTS) <= 15


def test_every_expected_tool_is_a_real_registered_tool():
    for case in TOOL_SELECTION_PROMPTS:
        assert case.expected_tool in _ALL_TOOL_NAMES, case.prompt


def test_every_tool_is_the_expected_tool_for_at_least_one_prompt():
    covered = {case.expected_tool for case in TOOL_SELECTION_PROMPTS}
    missing = _ALL_TOOL_NAMES - covered
    assert not missing, f"no prompt targets: {missing}"


def test_prompts_are_nonempty_and_unique():
    prompts = [case.prompt for case in TOOL_SELECTION_PROMPTS]
    assert all(p.strip() for p in prompts)
    assert len(prompts) == len(set(prompts))


def test_every_case_has_a_rationale():
    for case in TOOL_SELECTION_PROMPTS:
        assert case.rationale.strip(), case.prompt


@pytest.mark.asyncio
async def test_mcp_tools_convert_to_anthropic_tool_shape():
    """The eval harness's schema-conversion step (server.py tool schemas
    -> Anthropic Messages API `tools` shape) works and yields exactly the
    eight registered tools, each with a non-empty description and a
    dict-shaped input_schema -- independent of any live API call."""
    tools = await _mcp_tools_as_anthropic_tools()
    names = {t["name"] for t in tools}
    assert names == _ALL_TOOL_NAMES
    for tool in tools:
        assert tool["description"], tool["name"]
        assert isinstance(tool["input_schema"], dict)


@pytest.mark.asyncio
async def test_seeded_backend_in_eval_script_resolves_every_expected_tool_call():
    """Best-effort check that the eval script's representative seed data
    (`_seeded_backend`) is rich enough that, *if* a live client selects
    the intended tool for each prompt, the call wouldn't fail purely for
    lack of seeded data. This doesn't run the live eval; it just checks
    the fixture backend against a plausible call for each expected tool.
    """
    from tool_selection_eval import _seeded_backend

    backend: FakeBackend = _seeded_backend()
    mcp = build_server(backends={"fake": backend})

    ref_a = {"backend": "fake", "entity": "test-entity", "project": "mamfac", "run_id": "xj29fk1a"}
    ref_b = {"backend": "fake", "entity": "test-entity", "project": "mamfac", "run_id": "run-b"}
    plausible_calls = {
        "test_connection": {},
        "list_runs": {"backend": "fake", "project": "mamfac"},
        "get_run_summary": {"ref": ref_a},
        "get_metric_history": {"ref": ref_a, "metric": "reward"},
        "compare_runs": {"refs": [ref_a, ref_b]},
        "audit_training_curve": {"ref": ref_a, "metric": "reward"},
        "audit_ablation": {
            "baseline": ref_a,
            "ablation": ref_b,
            "claimed_variable": "learning_rate",
        },
        "audit_sweep": {
            "sweep_ref": {
                "backend": "fake",
                "entity": "test-entity",
                "project": "mamfac",
                "sweep_id": "sweep-1",
            }
        },
    }
    assert set(plausible_calls) == _ALL_TOOL_NAMES

    async with Client(mcp) as client:
        for tool_name, args in plausible_calls.items():
            result = await client.call_tool(tool_name, args)
            data = result.data or {}
            # ToolError-shaped failures are {"error": {"error_type": ..., ...}}
            # (server.py's _error_dict); test_connection's own success
            # shape happens to also have a top-level `error` key (`None`
            # on success, per its ConnectionStatus.to_dict()), so check
            # for the structured-failure shape specifically rather than
            # mere key presence.
            failure = isinstance(data.get("error"), dict)
            assert not failure, (tool_name, data)
