"""Integration tests for server.py (Milestone 4).

Per the roadmap's Milestone 4 completion criteria, these invoke each tool
through the actual MCP protocol layer — via `fastmcp.Client(mcp)`, an
in-memory MCP client/server pair — not by calling the registered Python
functions directly. This confirms JSON schema compliance and the full
request/response round-trip end-to-end, not just the underlying logic.

Uses `FakeBackend` (Milestone 2), not a `WandbBackend` fake-client setup:
`FakeBackend` is a fully public, exported `ExperimentBackend`
implementation built specifically to be "the primary tool for testing
every subsequent milestone" (fake_backend.py's own module docstring;
roadmap's Milestone 2 rationale) — reusing it here avoids duplicating a
second hand-rolled fake W&B client just for the server layer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastmcp import Client

from experiment_audit_mcp.backends.base import ConnectionStatus
from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef
from experiment_audit_mcp.server import build_server

_ENTITY = "test-entity"
_PROJECT = "mamfac"


def _make_run(run_id: str = "run1", project: str = _PROJECT, **overrides) -> Run:
    defaults = dict(
        ref=RunRef(backend="fake", entity=_ENTITY, project=project, run_id=run_id),
        name=f"run-{run_id}",
        tags=["baseline"],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 12.5},
    )
    defaults.update(overrides)
    return Run(**defaults)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mcp(fake_backend: FakeBackend):
    return build_server(backends={"fake": fake_backend})


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_tool_returns_connection_status_shape(mcp, fake_backend):
    async with Client(mcp) as client:
        result = await client.call_tool("test_connection", {})
    assert result.data == {
        "backend": "fake",
        "authenticated": True,
        "scopes_detected": ["read"],
        "error": None,
    }


@pytest.mark.asyncio
async def test_connection_tool_surfaces_auth_failure(mcp, fake_backend):
    fake_backend.set_connection_status(
        ConnectionStatus(backend="fake", authenticated=False, error="bad key")
    )
    async with Client(mcp) as client:
        result = await client.call_tool("test_connection", {})
    assert result.data["authenticated"] is False
    assert result.data["error"] == "bad key"


@pytest.mark.asyncio
async def test_connection_tool_has_no_backend_parameter(mcp):
    # Spec §4.2: test_connection() is a literal zero-argument tool.
    async with Client(mcp) as client:
        tools = await client.list_tools()
    tool = next(t for t in tools if t.name == "test_connection")
    assert tool.inputSchema.get("properties", {}) == {}


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_returns_run_summary_shape_not_full_run(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1"))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_runs", {"backend": "fake", "project": _PROJECT}
        )

    assert result.data["next_cursor"] is None
    assert len(result.data["items"]) == 1
    summary = result.data["items"][0]
    # Cheap fields present:
    assert summary["ref"] == {
        "backend": "fake", "entity": _ENTITY, "project": _PROJECT, "run_id": "run1"
    }
    assert summary["name"] == "run-run1"
    assert summary["tags"] == ["baseline"]
    assert summary["status"] == "finished"
    assert summary["data_completeness"] == "unknown"
    assert "created_at" in summary
    # Expensive fields deliberately absent (spec §4.2 "omits full
    # config/metrics"):
    assert "config" not in summary
    assert "summary_metrics" not in summary


@pytest.mark.asyncio
async def test_list_runs_applies_filters_through_the_mcp_layer(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1", tags=["ablation"], status="finished"))
    fake_backend.seed_run(_make_run("run2", tags=["baseline"], status="crashed"))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_runs",
            {
                "backend": "fake",
                "project": _PROJECT,
                "filters": {"tags": ["ablation"], "status": "finished"},
            },
        )

    assert [r["ref"]["run_id"] for r in result.data["items"]] == ["run1"]


@pytest.mark.asyncio
async def test_list_runs_unknown_backend_returns_structured_error_not_exception(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_runs", {"backend": "not-a-real-backend", "project": _PROJECT}
        )
    assert result.data["error"]["error_type"] == "unknown"
    assert "not-a-real-backend" in result.data["error"]["message"]
    assert result.data["error"]["recoverable"] is False


# ---------------------------------------------------------------------------
# get_run_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_summary_returns_full_run_including_config(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1", config={"lr": 0.01}, summary_metrics={"reward": 9.0}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_run_summary",
            {"ref": {"backend": "fake", "entity": _ENTITY, "project": _PROJECT, "run_id": "run1"}},
        )

    assert result.data["config"] == {"lr": 0.01}
    assert result.data["summary_metrics"] == {"reward": 9.0}


@pytest.mark.asyncio
async def test_get_run_summary_not_found_returns_structured_error(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_run_summary",
            {
                "ref": {
                    "backend": "fake",
                    "entity": _ENTITY,
                    "project": _PROJECT,
                    "run_id": "does-not-exist",
                }
            },
        )
    assert result.data["error"]["error_type"] == "run_not_found"
    assert result.data["error"]["recoverable"] is False


# ---------------------------------------------------------------------------
# get_metric_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metric_history_preserves_null_points_over_mcp(mcp, fake_backend):
    ref = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run1")
    fake_backend.seed_run(_make_run("run1"))
    fake_backend.seed_metric_history(
        MetricHistory(
            ref=ref,
            metric_name="loss",
            points=[
                MetricPoint(step=0, value=1.0),
                MetricPoint(step=1, value=None),  # logged NaN
                MetricPoint(step=2, value=0.5),
            ],
        )
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_metric_history",
            {
                "ref": {
                    "backend": "fake", "entity": _ENTITY, "project": _PROJECT, "run_id": "run1"
                },
                "metric": "loss",
            },
        )

    # The critical adversarial-case assertion (spec §2, §7): the None
    # point must survive the full MCP JSON round-trip, not be dropped.
    assert result.data["points"] == [
        {"step": 0, "value": 1.0},
        {"step": 1, "value": None},
        {"step": 2, "value": 0.5},
    ]


@pytest.mark.asyncio
async def test_get_metric_history_respects_step_range(mcp, fake_backend):
    ref = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run1")
    fake_backend.seed_run(_make_run("run1"))
    fake_backend.seed_metric_history(
        MetricHistory(
            ref=ref,
            metric_name="loss",
            points=[MetricPoint(step=i, value=float(i)) for i in range(5)],
        )
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_metric_history",
            {
                "ref": {
                    "backend": "fake", "entity": _ENTITY, "project": _PROJECT, "run_id": "run1"
                },
                "metric": "loss",
                "step_range": [1, 3],
            },
        )

    assert [p["step"] for p in result.data["points"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_metric_history_not_found_returns_structured_error(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1"))
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_metric_history",
            {
                "ref": {
                    "backend": "fake", "entity": _ENTITY, "project": _PROJECT, "run_id": "run1"
                },
                "metric": "never_logged",
            },
        )
    assert result.data["error"]["error_type"] == "run_not_found"


# ---------------------------------------------------------------------------
# Tool schema sanity (spec §4.1 naming convention / §6 minimal descriptions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_eight_tools_are_registered(mcp):
    # Updated in Milestone 8 to include audit_sweep, the same way
    # Milestone 7 updated this test for audit_ablation, Milestone 6 for
    # audit_training_curve, and the way Revisions 1/2 touched other
    # already-approved files when a later milestone's change had a
    # direct, mechanical effect on them.
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "test_connection",
        "list_runs",
        "get_run_summary",
        "get_metric_history",
        "compare_runs",
        "audit_training_curve",
        "audit_ablation",
        "audit_sweep",
    }


@pytest.mark.asyncio
async def test_tool_descriptions_do_not_inline_audit_methodology(mcp):
    # Spec §6: tool descriptions stay minimal; no audit_* tools exist yet
    # in Milestone 4, so there's no docs/audit-methods.md pointer to check
    # for, but descriptions should still be short (a proxy for "minimal").
    async with Client(mcp) as client:
        tools = await client.list_tools()
    for tool in tools:
        assert tool.description is not None
        assert len(tool.description) < 400


# ---------------------------------------------------------------------------
# compare_runs (Milestone 5)
# ---------------------------------------------------------------------------


def _ref_input(run_id: str, project: str = _PROJECT) -> dict:
    return {"backend": "fake", "entity": _ENTITY, "project": project, "run_id": run_id}


@pytest.mark.asyncio
async def test_compare_runs_returns_config_and_metric_diff_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(
        _make_run("run1", config={"lr": 0.001}, summary_metrics={"reward": 10.0})
    )
    fake_backend.seed_run(
        _make_run("run2", config={"lr": 0.01}, summary_metrics={"reward": 15.0})
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_runs", {"refs": [_ref_input("run1"), _ref_input("run2")]}
        )

    assert result.data["config_diff"][0]["param"] == "lr"
    assert result.data["metric_diff"][0]["metric"] == "reward"
    assert result.data["metric_diff"][0]["delta"] == 5.0


@pytest.mark.asyncio
async def test_compare_runs_nway_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1", config={"seed": 1}))
    fake_backend.seed_run(_make_run("run2", config={"seed": 2}))
    fake_backend.seed_run(_make_run("run3", config={"seed": 3}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_runs",
            {"refs": [_ref_input("run1"), _ref_input("run2"), _ref_input("run3")]},
        )

    entry = result.data["config_diff"][0]
    assert entry["param"] == "seed"
    assert len(entry["values"]) == 3


@pytest.mark.asyncio
async def test_compare_runs_cross_project_echoes_project_in_ref_keys(mcp, fake_backend):
    fake_backend.seed_run(
        _make_run("run1", project="mamfac", config={"lr": 0.001})
    )
    fake_backend.seed_run(
        _make_run("run2", project="carm-plus-plus", config={"lr": 0.01})
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_runs",
            {"refs": [_ref_input("run1", "mamfac"), _ref_input("run2", "carm-plus-plus")]},
        )

    keys = set(result.data["config_diff"][0]["values"].keys())
    assert any("mamfac" in k for k in keys)
    assert any("carm-plus-plus" in k for k in keys)


@pytest.mark.asyncio
async def test_compare_runs_missing_run_returns_structured_error_not_exception(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1"))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_runs", {"refs": [_ref_input("run1"), _ref_input("does-not-exist")]}
        )

    assert result.data["error"]["error_type"] == "run_not_found"


@pytest.mark.asyncio
async def test_compare_runs_single_ref_returns_structured_error(mcp, fake_backend):
    fake_backend.seed_run(_make_run("run1"))

    async with Client(mcp) as client:
        result = await client.call_tool("compare_runs", {"refs": [_ref_input("run1")]})

    assert result.data["error"]["error_type"] == "unknown"
    assert "at least 2 runs" in result.data["error"]["message"]


# ---------------------------------------------------------------------------
# audit_training_curve (Milestone 6)
# ---------------------------------------------------------------------------


def _seed_history(fake_backend, ref, metric, values):
    fake_backend.seed_metric_history(
        MetricHistory(
            ref=ref,
            metric_name=metric,
            points=[MetricPoint(step=i, value=v) for i, v in enumerate(values)],
        )
    )


@pytest.mark.asyncio
async def test_audit_training_curve_returns_schema_version_2(mcp, fake_backend):
    ref = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run1")
    _seed_history(fake_backend, ref, "loss", [10.0 - i for i in range(11)])

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_training_curve", {"ref": _ref_input("run1"), "metric": "loss"}
        )

    assert result.data["schema_version"] == 2
    assert result.data["metric_type_assumed"] == "loss"
    assert result.data["signals"] == []
    assert "docs/audit-methods.md#training-curve" in result.data["method"]


@pytest.mark.asyncio
async def test_audit_training_curve_flags_nan_mid_curve_over_mcp(mcp, fake_backend):
    ref = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run1")
    _seed_history(fake_backend, ref, "loss", [1.0, 2.0, None, 4.0, 5.0])

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_training_curve", {"ref": _ref_input("run1"), "metric": "loss"}
        )

    signals = {s["signal"] for s in result.data["signals"]}
    assert "null_values" in signals
    null_signal = next(s for s in result.data["signals"] if s["signal"] == "null_values")
    assert null_signal["step_range"] == [2, 2]
    assert null_signal["confidence"] == "high"


@pytest.mark.asyncio
async def test_audit_training_curve_flags_oscillation_over_mcp(mcp, fake_backend):
    ref = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run1")
    values = [5.0 if i % 2 == 0 else 15.0 for i in range(10)]
    _seed_history(fake_backend, ref, "reward", values)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_training_curve", {"ref": _ref_input("run1"), "metric": "reward"}
        )

    signals = {s["signal"] for s in result.data["signals"]}
    assert "high_frequency_oscillation" in signals
    assert result.data["metric_type_assumed"] == "reward"


@pytest.mark.asyncio
async def test_audit_training_curve_missing_history_returns_structured_error(mcp, fake_backend):
    # Deliberately do not seed any metric history for this ref/metric.
    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_training_curve", {"ref": _ref_input("run1"), "metric": "never_logged"}
        )

    assert result.data["error"]["error_type"] == "run_not_found"


@pytest.mark.asyncio
async def test_audit_training_curve_unknown_backend_returns_structured_error(mcp, fake_backend):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_training_curve",
            {
                "ref": {
                    "backend": "not-a-real-backend",
                    "entity": _ENTITY,
                    "project": _PROJECT,
                    "run_id": "run1",
                },
                "metric": "loss",
            },
        )

    assert result.data["error"]["error_type"] == "unknown"


# ---------------------------------------------------------------------------
# audit_ablation (Milestone 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_ablation_clean_single_claimed_variable_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(_make_run("baseline", config={"lr": 0.001, "seed": 42}))
    fake_backend.seed_run(_make_run("ablation", config={"lr": 0.01, "seed": 42}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("ablation"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["verdict"] == "clean"
    assert result.data["confidence"] == "high"
    assert result.data["differing_params"] == [
        {"param": "lr", "baseline_value": 0.001, "ablation_value": 0.01, "likely_intentional": True}
    ]
    assert "docs/audit-methods.md#ablation" in result.data["method"]


@pytest.mark.asyncio
async def test_audit_ablation_seed_only_difference_is_clean_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(_make_run("baseline", config={"lr": 0.001, "seed": 1}))
    fake_backend.seed_run(_make_run("ablation", config={"lr": 0.001, "seed": 2}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("ablation"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["verdict"] == "clean"
    entry = next(p for p in result.data["differing_params"] if p["param"] == "seed")
    assert entry["likely_intentional"] is True


@pytest.mark.asyncio
async def test_audit_ablation_two_nonallowlisted_params_is_confounded_over_mcp(
    mcp, fake_backend
):
    fake_backend.seed_run(
        _make_run("baseline", config={"lr": 0.001, "batch_size": 32})
    )
    fake_backend.seed_run(
        _make_run("ablation", config={"lr": 0.01, "batch_size": 64})
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("ablation"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["verdict"] == "confounded"
    params = {p["param"]: p["likely_intentional"] for p in result.data["differing_params"]}
    assert params == {"lr": True, "batch_size": False}


@pytest.mark.asyncio
async def test_audit_ablation_partial_data_downgrades_confidence_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(
        _make_run("baseline", config={"lr": 0.001}, data_completeness="partial")
    )
    fake_backend.seed_run(_make_run("ablation", config={"lr": 0.01}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("ablation"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["verdict"] == "clean"
    assert result.data["confidence"] == "low"
    assert "partial data" in result.data["method"]


@pytest.mark.asyncio
async def test_audit_ablation_evidence_is_full_comparison_over_mcp(mcp, fake_backend):
    fake_backend.seed_run(
        _make_run("baseline", config={"lr": 0.001}, summary_metrics={"reward": 10.0})
    )
    fake_backend.seed_run(
        _make_run("ablation", config={"lr": 0.01}, summary_metrics={"reward": 15.0})
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("ablation"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["evidence"]["config_diff"][0]["param"] == "lr"
    assert result.data["evidence"]["metric_diff"][0]["metric"] == "reward"
    assert result.data["evidence"]["metric_diff"][0]["delta"] == 5.0


@pytest.mark.asyncio
async def test_audit_ablation_missing_run_returns_structured_error_not_exception(
    mcp, fake_backend
):
    fake_backend.seed_run(_make_run("baseline", config={"lr": 0.001}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("does-not-exist"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["error"]["error_type"] == "run_not_found"


@pytest.mark.asyncio
async def test_audit_ablation_same_ref_returns_structured_error(mcp, fake_backend):
    fake_backend.seed_run(_make_run("baseline", config={"lr": 0.001}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": _ref_input("baseline"),
                "claimed_variable": "lr",
            },
        )

    assert result.data["error"]["error_type"] == "unknown"
    assert "duplicate RunRefs" in result.data["error"]["message"]


@pytest.mark.asyncio
async def test_audit_ablation_unknown_backend_returns_structured_error(mcp, fake_backend):
    fake_backend.seed_run(_make_run("baseline", config={"lr": 0.001}))

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_ablation",
            {
                "baseline": _ref_input("baseline"),
                "ablation": {
                    "backend": "not-a-real-backend",
                    "entity": _ENTITY,
                    "project": _PROJECT,
                    "run_id": "ablation",
                },
                "claimed_variable": "lr",
            },
        )

    assert result.data["error"]["error_type"] == "unknown"


# ---------------------------------------------------------------------------
# audit_sweep (Milestone 8)
# ---------------------------------------------------------------------------


def _sweep_ref_input(sweep_id: str = "sweep-1") -> dict:
    return {"backend": "fake", "entity": _ENTITY, "project": _PROJECT, "sweep_id": sweep_id}


def _seed_linear_sweep(fake_backend: FakeBackend, n: int = 12, sweep_id: str = "sweep-1"):
    from experiment_audit_mcp.models import Sweep

    runs = []
    for i in range(1, n + 1):
        run = _make_run(
            run_id=f"r{i}",
            config={"lr": float(i), "seed": 42},
            summary_metrics={"reward": 10.0 * i},
        )
        fake_backend.seed_run(run)
        runs.append(run)
    sweep = Sweep(
        ref=RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
        sweep_id=sweep_id,
        method="grid",
        run_refs=[r.ref for r in runs],
        target_metric="reward",
    )
    fake_backend.seed_sweep(sweep)
    return sweep, runs


@pytest.mark.asyncio
async def test_audit_sweep_returns_ranking_through_mcp_layer(mcp, fake_backend):
    _seed_linear_sweep(fake_backend)

    async with Client(mcp) as client:
        result = await client.call_tool("audit_sweep", {"sweep_ref": _sweep_ref_input()})

    assert result.data["sweep_size"] == 12
    assert result.data["usable_run_count"] == 12
    assert result.data["parameter_importance"][0]["param"] == "lr"
    assert result.data["confidence"] == "high"
    assert "error" not in result.data


@pytest.mark.asyncio
async def test_audit_sweep_explicit_target_metric_overrides_sweep_default(mcp, fake_backend):
    from experiment_audit_mcp.models import Sweep

    runs = []
    for i in range(1, 13):
        run = _make_run(
            run_id=f"r{i}",
            config={"lr": float(i)},
            summary_metrics={"reward": float(i), "loss": float(20 - i)},
        )
        fake_backend.seed_run(run)
        runs.append(run)
    sweep = Sweep(
        ref=RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
        sweep_id="sweep-1",
        method="grid",
        run_refs=[r.ref for r in runs],
        target_metric="reward",
    )
    fake_backend.seed_sweep(sweep)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_sweep", {"sweep_ref": _sweep_ref_input(), "target_metric": "loss"}
        )

    assert result.data["target_metric"] == "loss"


@pytest.mark.asyncio
async def test_audit_sweep_too_small_returns_insufficient_samples_error(mcp, fake_backend):
    _seed_linear_sweep(fake_backend, n=3)

    async with Client(mcp) as client:
        result = await client.call_tool("audit_sweep", {"sweep_ref": _sweep_ref_input()})

    assert result.data["error"]["error_type"] == "insufficient_samples"
    assert result.data["run_count"] == 3
    assert result.data["minimum_required"] == 10


@pytest.mark.asyncio
async def test_audit_sweep_unknown_sweep_id_returns_structured_error(mcp, fake_backend):
    _seed_linear_sweep(fake_backend, sweep_id="sweep-1")

    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_sweep", {"sweep_ref": _sweep_ref_input(sweep_id="does-not-exist")}
        )

    assert result.data["error"]["error_type"] == "unknown"


@pytest.mark.asyncio
async def test_audit_sweep_unknown_backend_returns_structured_error(mcp, fake_backend):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_sweep",
            {
                "sweep_ref": {
                    "backend": "not-a-real-backend",
                    "entity": _ENTITY,
                    "project": _PROJECT,
                    "sweep_id": "sweep-1",
                }
            },
        )

    assert result.data["error"]["error_type"] == "unknown"


@pytest.mark.asyncio
async def test_audit_sweep_unsupported_capability_returns_structured_error(mcp):
    capability_less_backend = FakeBackend(capabilities=set())
    mcp_no_sweeps = build_server(backends={"fake": capability_less_backend})
    capability_less_backend.seed_run(_make_run())

    async with Client(mcp_no_sweeps) as client:
        result = await client.call_tool("audit_sweep", {"sweep_ref": _sweep_ref_input()})

    assert result.data["error"]["error_type"] == "backend_unsupported_capability"


@pytest.mark.asyncio
async def test_audit_sweep_covarying_params_surfaces_warning_through_mcp_layer(mcp, fake_backend):
    from experiment_audit_mcp.models import Sweep

    runs = []
    for i in range(1, 13):
        run = _make_run(
            run_id=f"r{i}",
            config={"learning_rate": float(i), "batch_size": float(i) * 10, "seed": 42},
            summary_metrics={"reward": float(i)},
        )
        fake_backend.seed_run(run)
        runs.append(run)
    sweep = Sweep(
        ref=RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
        sweep_id="sweep-1",
        method="grid",
        run_refs=[r.ref for r in runs],
        target_metric="reward",
    )
    fake_backend.seed_sweep(sweep)

    async with Client(mcp) as client:
        result = await client.call_tool("audit_sweep", {"sweep_ref": _sweep_ref_input()})

    by_param = {p["param"]: p for p in result.data["parameter_importance"]}
    assert "co-varies with" in by_param["learning_rate"]["warning"]
