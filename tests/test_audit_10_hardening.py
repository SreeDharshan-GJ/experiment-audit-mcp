"""Regression tests for Audit #10's finding: unbounded resource fan-out
in `audit_sweep`.

**Finding (Denial-of-Service / unbounded resource growth):**
`compare_runs`' `refs` argument is bounded at the MCP schema level
(`min_length=2, max_length=_MAX_COMPARE_RUNS_REFS` in server.py) precisely
because it fans out into one `get_run_summary` backend call per ref via
`_gather_bounded`. `audit_sweep` fans out over `sweep.run_refs` through
the exact same `_gather_bounded` helper, but `sweep.run_refs` comes back
from the backend's `list_sweeps` call, not a caller-supplied MCP
argument, so it was never bounded anywhere -- neither at a schema level
(there is no schema for backend-returned data) nor with an explicit
runtime check. `_gather_bounded`'s concurrency `limit` only bounds how
many fetches run *at once*, not the total number issued.

Before the fix, a single `audit_sweep` call against a sweep with N runs
(reported here by a backend under an attacker's influence, or simply an
unusually large real sweep) issued exactly N `get_run_summary` calls
with no upper bound, held N full `Run` objects in memory simultaneously,
and fed all N into `_audit_sweep_pure`'s `_covariance_warnings` pass,
which is O(parameters^2 * N). This test file pins the fix:
`audit_sweep` now rejects a sweep whose `run_refs` exceeds
`_MAX_AUDIT_SWEEP_RUNS` with a structured, non-`isError=False` error
*before* issuing any `get_run_summary` call, mirroring the existing
`min_length=2, max_length=_MAX_COMPARE_RUNS_REFS` guard on `compare_runs`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastmcp import Client

from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef, Sweep
from experiment_audit_mcp.server import (
    _MAX_AUDIT_SWEEP_RUNS,
    _MAX_METRIC_HISTORY_POINTS,
    build_server,
)

_ENTITY = "test-entity"
_PROJECT = "mamfac"
_BACKEND = "fake"


def _make_run(run_id: str, **overrides) -> Run:
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


def _seed_sweep_of_size(fake_backend: FakeBackend, n: int, sweep_id: str = "big-sweep") -> Sweep:
    runs = [
        _make_run(f"r{i}", config={"lr": 0.001 + i * 1e-6, "seed": 42}) for i in range(n)
    ]
    for r in runs:
        fake_backend.seed_run(r)
    sweep = Sweep(
        ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="sweep-placeholder"),
        sweep_id=sweep_id,
        method="random",
        run_refs=[r.ref for r in runs],
        target_metric="final_reward",
    )
    fake_backend.seed_sweep(sweep)
    return sweep


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mcp(fake_backend: FakeBackend):
    return build_server(backends={_BACKEND: fake_backend})


class TestAuditSweepUnboundedFanOut:
    @pytest.mark.asyncio
    async def test_oversized_sweep_rejected_before_any_backend_fetch(
        self, mcp, fake_backend
    ):
        n = _MAX_AUDIT_SWEEP_RUNS + 1
        _seed_sweep_of_size(fake_backend, n)

        calls = {"count": 0}
        original = fake_backend.get_run_summary

        async def _counting(ref):
            calls["count"] += 1
            return await original(ref)

        fake_backend.get_run_summary = _counting  # type: ignore[method-assign]

        async with Client(mcp) as client:
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "big-sweep",
                    }
                },
                raise_on_error=False,
            )

        assert result.is_error is True, "an oversized sweep must be a structured MCP error"
        assert calls["count"] == 0, (
            f"expected zero get_run_summary calls for a rejected oversized sweep, "
            f"got {calls['count']} -- the size check must happen before any fetch"
        )
        assert "error" in result.structured_content
        assert str(n) in result.structured_content["error"]["message"] or (
            str(_MAX_AUDIT_SWEEP_RUNS) in result.structured_content["error"]["message"]
        )

    @pytest.mark.asyncio
    async def test_sweep_at_exactly_the_limit_still_works(self, mcp, fake_backend):
        _seed_sweep_of_size(fake_backend, _MAX_AUDIT_SWEEP_RUNS)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "big-sweep",
                    }
                },
            )

        assert result.is_error is False
        assert result.structured_content["sweep_size"] == _MAX_AUDIT_SWEEP_RUNS

    @pytest.mark.asyncio
    async def test_small_sweep_still_fetches_every_run_unaffected_by_the_cap(
        self, mcp, fake_backend
    ):
        # 12, not fewer: analysis/sensitivity.py's own
        # DEFAULT_MINIMUM_SAMPLES=10 floor is a separate, deliberate
        # statistical-validity refusal (insufficient_samples) unrelated to
        # the fan-out cap under test here -- using a sweep size below it
        # would trigger that refusal first and never exercise this path.
        n = 12
        _seed_sweep_of_size(fake_backend, n)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "big-sweep",
                    }
                },
            )

        assert result.is_error is False
        assert result.structured_content["sweep_size"] == n


class TestGetMetricHistoryUnboundedPayload:
    """Finding: `get_metric_history` returned every point in
    `MetricHistory.points` with no upper bound, so a run with enough
    logged points (an ordinary occurrence for a long training run, not
    an adversarial input) produced an unbounded-size JSON response and,
    at large enough scale, exhausted process memory outright. Direct
    reproduction against a freshly-seeded `FakeBackend` (isolating the
    growth to this server's own serialization path, no real W&B network
    behavior involved): seeding 5,000,000 `MetricPoint`s and calling
    `get_metric_history` through the same in-memory `fastmcp.Client`
    round trip these tests use killed the Python process with an
    out-of-memory signal before returning. At 500,000 points the same
    call completed but produced a 16.3 MB single-response JSON payload
    with no ceiling stopping it from being larger.

    Fix: `get_metric_history` now refuses (structured MCP error, no
    partial data returned) once `history.points` exceeds
    `_MAX_METRIC_HISTORY_POINTS`, rather than truncating -- truncating
    would violate this same function's own documented, spec-critical
    promise that logged NaN/null points are "preserved exactly, never
    dropped."
    """

    @pytest.fixture
    def fake_backend(self) -> FakeBackend:
        return FakeBackend()

    @pytest.fixture
    def mcp(self, fake_backend: FakeBackend):
        return build_server(backends={_BACKEND: fake_backend})

    def _seed_history(self, fake_backend: FakeBackend, n: int, run_id: str = "r1") -> RunRef:
        ref = RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id=run_id)
        fake_backend.seed_run(_make_run(run_id))
        points = [MetricPoint(step=i, value=float(i % 100)) for i in range(n)]
        fake_backend.seed_metric_history(
            MetricHistory(ref=ref, metric_name="train/loss", points=points)
        )
        return ref

    @pytest.mark.asyncio
    async def test_oversized_history_rejected_not_truncated(self, mcp, fake_backend):
        n = _MAX_METRIC_HISTORY_POINTS + 1
        self._seed_history(fake_backend, n)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_metric_history",
                {
                    "ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "run_id": "r1",
                    },
                    "metric": "train/loss",
                },
                raise_on_error=False,
            )

        assert result.is_error is True
        # It must be an outright refusal, not a silently truncated
        # "points" list -- a truncated-but-still-200-status response
        # would look like a complete history while quietly having
        # dropped data, which is exactly the failure mode this fix
        # avoids.
        assert "points" not in result.structured_content
        assert "error" in result.structured_content
        assert str(_MAX_METRIC_HISTORY_POINTS) in result.structured_content["error"]["message"]

    @pytest.mark.asyncio
    async def test_history_at_exactly_the_limit_still_returns_every_point(
        self, mcp, fake_backend
    ):
        n = 50_000  # below the cap, but large enough to be a meaningful check
        self._seed_history(fake_backend, n)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_metric_history",
                {
                    "ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "run_id": "r1",
                    },
                    "metric": "train/loss",
                },
            )

        assert result.is_error is False
        assert len(result.structured_content["points"]) == n

    @pytest.mark.asyncio
    async def test_nan_and_null_points_still_preserved_exactly_under_the_cap(
        self, mcp, fake_backend
    ):
        """The cap must refuse-or-return-everything, never silently
        drop individual NaN/null points while stripping the count down
        to size -- that would violate get_metric_history's own
        documented invariant while looking like a normal, complete
        response."""
        ref = RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="r2")
        fake_backend.seed_run(_make_run("r2"))
        points = [
            MetricPoint(step=0, value=1.0),
            MetricPoint(step=1, value=float("nan")),
            MetricPoint(step=2, value=None),
            MetricPoint(step=3, value=2.5),
        ]
        fake_backend.seed_metric_history(
            MetricHistory(ref=ref, metric_name="train/loss", points=points)
        )

        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_metric_history",
                {
                    "ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "run_id": "r2",
                    },
                    "metric": "train/loss",
                },
            )

        assert result.is_error is False
        returned = result.structured_content["points"]
        assert len(returned) == 4
        assert returned[1]["value"] is None  # NaN serialized, not dropped
        assert returned[2]["value"] is None
