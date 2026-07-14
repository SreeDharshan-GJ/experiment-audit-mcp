"""Regression tests for the MCP-server audit findings (server.py).

Two categories of fix are covered here:

1. **Tool-schema hygiene.** `SweepRefInput`'s class docstring previously
   contained a multi-paragraph developer-facing rationale that FastMCP
   surfaced verbatim as the JSON-schema `description` for every
   `audit_sweep` call's `sweep_ref` parameter — directly contradicting
   this file's own stated principle (see server.py's module docstring)
   that MCP tool schemas carry short, LLM-facing text, not prose meant
   for a human reading the source. `test_sweep_ref_schema_description_*`
   below pins the fix.

2. **Boundary argument validation.** Several MCP-facing arguments had no
   validation at all: blank identifiers (`backend`/`entity`/`project`/
   `run_id`/`sweep_id`/`metric`/`claimed_variable`), a non-positive or
   unbounded `page_size`, and a `step_range` whose start is after its
   end. All four now fail cleanly at the MCP boundary — the first three
   via the tool's JSON schema (a `fastmcp.exceptions.ToolError` raised
   before the tool body ever runs, which is the standard MCP way to
   reject malformed arguments), the last via an explicit in-tool check
   returning the same `ToolError`-shaped dict every other business-logic
   failure in this file uses (pydantic can constrain each tuple element
   independently but not their relative order).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef
from experiment_audit_mcp.server import build_server

_ENTITY = "test-entity"
_PROJECT = "mamfac"
_BACKEND = "fake"


def _make_run(run_id: str = "run1", **overrides) -> Run:
    defaults = dict(
        ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id=run_id),
        name=f"run-{run_id}",
        tags=[],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 12.5},
    )
    defaults.update(overrides)
    return Run(**defaults)


def _ref(run_id: str = "run1") -> dict:
    return {"backend": _BACKEND, "entity": _ENTITY, "project": _PROJECT, "run_id": run_id}


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mcp(fake_backend: FakeBackend):
    return build_server(backends={"fake": fake_backend})


# ---------------------------------------------------------------------------
# 1. Tool-schema hygiene: SweepRefInput no longer leaks developer prose.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_ref_schema_description_is_short_not_leaked_rationale(mcp):
    async with Client(mcp) as client:
        tools = await client.list_tools()
    sweep_tool = next(t for t in tools if t.name == "audit_sweep")
    description = sweep_tool.inputSchema["properties"]["sweep_ref"]["description"]

    # The bug: the class docstring's multi-paragraph internal rationale
    # (mentioning `Sweep.ref`, `models.py`, "placeholder") was appearing
    # here verbatim. Assert both a tight length bound and the absence of
    # the tell-tale developer-facing terms that rationale used.
    assert len(description) < 100, description
    for leaked_term in ("models.py", "placeholder", "Sweep.ref", "RunRefInput"):
        assert leaked_term not in description, (leaked_term, description)


@pytest.mark.asyncio
async def test_run_ref_schema_description_still_short(mcp):
    """Sanity check that the working case (RunRefInput, never buggy) keeps
    matching the same short-description standard the fix restores for
    SweepRefInput — a baseline so future regressions on either type are
    caught the same way."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    get_run_summary = next(t for t in tools if t.name == "get_run_summary")
    description = get_run_summary.inputSchema["properties"]["ref"]["description"]
    assert len(description) < 100, description


# ---------------------------------------------------------------------------
# 2a. Non-blank identifier validation (RunRefInput / SweepRefInput fields).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("blank_field", ["backend", "entity", "project", "run_id"])
async def test_get_run_summary_rejects_blank_run_ref_fields(mcp, blank_field):
    ref = _ref()
    ref[blank_field] = ""
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_run_summary", {"ref": ref})


@pytest.mark.asyncio
async def test_list_runs_rejects_blank_backend(mcp):
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("list_runs", {"backend": "", "project": _PROJECT})


@pytest.mark.asyncio
async def test_list_runs_rejects_blank_project(mcp):
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("list_runs", {"backend": _BACKEND, "project": ""})


@pytest.mark.asyncio
async def test_get_metric_history_rejects_blank_metric(mcp, fake_backend):
    fake_backend.seed_run(_make_run())
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_metric_history", {"ref": _ref(), "metric": ""})


@pytest.mark.asyncio
async def test_audit_ablation_rejects_blank_claimed_variable(mcp, fake_backend):
    fake_backend.seed_run(_make_run("baseline"))
    fake_backend.seed_run(_make_run("ablation"))
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "audit_ablation",
                {
                    "baseline": _ref("baseline"),
                    "ablation": _ref("ablation"),
                    "claimed_variable": "",
                },
            )


@pytest.mark.asyncio
async def test_audit_sweep_rejects_blank_sweep_id(mcp):
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "",
                    }
                },
            )


# ---------------------------------------------------------------------------
# 2b. page_size bounds (list_runs).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_page_size", [0, -1, 501])
async def test_list_runs_rejects_out_of_bounds_page_size(mcp, bad_page_size):
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "list_runs",
                {"backend": _BACKEND, "project": _PROJECT, "page_size": bad_page_size},
            )


@pytest.mark.asyncio
async def test_list_runs_accepts_page_size_at_bounds(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1"))
    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_runs", {"backend": _BACKEND, "project": _PROJECT, "page_size": 1}
        )
    assert "error" not in result.data


# ---------------------------------------------------------------------------
# 2c. step_range ordering (get_metric_history).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metric_history_rejects_reversed_step_range(mcp, fake_backend):
    fake_backend.seed_run(_make_run())
    fake_backend.seed_metric_history(
        MetricHistory(
            ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="run1"),
            metric_name="loss",
            points=[MetricPoint(step=0, value=1.0)],
        )
    )
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_metric_history",
            {"ref": _ref(), "metric": "loss", "step_range": [5, 1]},
            raise_on_error=False,
        )
    assert result.is_error is True
    assert result.structured_content["error"]["error_type"] == "unknown"
    assert "step_range" in result.structured_content["error"]["message"]
    assert result.structured_content["error"]["recoverable"] is False


@pytest.mark.asyncio
async def test_get_metric_history_accepts_equal_step_range_bounds(mcp, fake_backend):
    fake_backend.seed_run(_make_run())
    fake_backend.seed_metric_history(
        MetricHistory(
            ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="run1"),
            metric_name="loss",
            points=[MetricPoint(step=2, value=0.5)],
        )
    )
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_metric_history",
            {"ref": _ref(), "metric": "loss", "step_range": [2, 2]},
        )
    assert "error" not in result.data
