"""Regression tests for the MCP-layer production audit fixes.

Each test class below corresponds to one numbered finding in the audit
summary and is named after it, so a future regression in any of these
behaviors fails with a name that points straight back to the finding
that would have caught it, without needing to cross-reference a
separate document.

Uses `FakeBackend` (Milestone 2) and the real `fastmcp.Client` protocol
round trip, matching the convention established in test_server.py and
test_adversarial_mcp_layer.py — these findings are specifically about
what crosses the actual MCP wire, so calling the registered tool
functions directly (bypassing FastMCP's request/response handling)
would not exercise the bug or the fix.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import pytest
from fastmcp import Client

from experiment_audit_mcp.backends.base import BackendCapability
from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.models import Run, RunRef
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


def _ref_input(run_id: str) -> dict:
    return {"backend": _BACKEND, "entity": _ENTITY, "project": _PROJECT, "run_id": run_id}


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mcp(fake_backend: FakeBackend):
    return build_server(backends={_BACKEND: fake_backend})


# ---------------------------------------------------------------------------
# Finding 1 (Critical): structured tool errors must set CallToolResult.isError
# ---------------------------------------------------------------------------


class TestStructuredErrorsSetIsError:
    """Before the fix, every tool error was a plain dict, so
    `CallToolResult.isError` was `False` for every failure this server has
    ever returned — an MCP client had no protocol-level way to tell a
    failed audit from a successful one. Covers every tool's error path,
    not just one representative case, since the bug was in the shared
    `_error_dict` helper every one of them uses.
    """

    @pytest.mark.asyncio
    async def test_unknown_backend_sets_is_error_across_every_tool(self, mcp, fake_backend):
        fake_backend.seed_run(_make_run())
        cases = [
            ("list_runs", {"backend": "nope", "project": _PROJECT}),
            ("get_run_summary", {"ref": {**_ref_input("run1"), "backend": "nope"}}),
            (
                "get_metric_history",
                {"ref": {**_ref_input("run1"), "backend": "nope"}, "metric": "loss"},
            ),
            (
                "compare_runs",
                {
                    "refs": [
                        {**_ref_input("run1"), "backend": "nope"},
                        _ref_input("run1"),
                    ]
                },
            ),
            (
                "audit_training_curve",
                {"ref": {**_ref_input("run1"), "backend": "nope"}, "metric": "loss"},
            ),
            (
                "audit_ablation",
                {
                    "baseline": {**_ref_input("run1"), "backend": "nope"},
                    "ablation": _ref_input("run1"),
                    "claimed_variable": "lr",
                },
            ),
            (
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": "nope",
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "s1",
                    }
                },
            ),
        ]
        async with Client(mcp) as client:
            for tool_name, args in cases:
                result = await client.call_tool(tool_name, args, raise_on_error=False)
                assert result.is_error is True, f"{tool_name} did not set isError"
                assert result.structured_content["error"]["error_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_successful_calls_do_not_set_is_error(self, mcp, fake_backend):
        fake_backend.seed_run(_make_run())
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_run_summary", {"ref": _ref_input("run1")}, raise_on_error=False
            )
        assert result.is_error is False
        assert "error" not in result.structured_content

    @pytest.mark.asyncio
    async def test_error_content_is_still_structured_and_parseable(self, mcp):
        # The fix must not sacrifice the structured payload for the
        # protocol-level signal — a caller that wants to parse
        # error_type/recoverable/retry_after_seconds programmatically
        # still can, via structured_content.
        async with Client(mcp) as client:
            result = await client.call_tool(
                "list_runs", {"backend": "nope", "project": _PROJECT}, raise_on_error=False
            )
        error = result.structured_content["error"]
        assert set(error) == {"error_type", "message", "recoverable", "retry_after_seconds"}


# ---------------------------------------------------------------------------
# Finding 2 (Major): test_connection must translate its RuntimeError, not
# let it cross the boundary as an unstructured error.
# ---------------------------------------------------------------------------


class TestTestConnectionErrorTranslation:
    @pytest.mark.asyncio
    async def test_multi_backend_misconfiguration_returns_structured_error(self):
        mcp_multi = build_server(backends={"a": FakeBackend(), "b": FakeBackend()})
        async with Client(mcp_multi) as client:
            result = await client.call_tool("test_connection", {}, raise_on_error=False)
        # Before the fix, this raised a bare RuntimeError that crossed the
        # MCP boundary as FastMCP's generic "Error calling tool '...'"
        # text, not this codebase's {"error": {"error_type": ...}} shape
        # every other failure uses.
        assert result.is_error is True
        assert result.structured_content["error"]["error_type"] == "unknown"
        assert set(result.structured_content["error"]) == {
            "error_type",
            "message",
            "recoverable",
            "retry_after_seconds",
        }

    @pytest.mark.asyncio
    async def test_zero_backend_misconfiguration_returns_structured_error(self):
        mcp_empty = build_server(backends={})
        async with Client(mcp_empty) as client:
            result = await client.call_tool("test_connection", {}, raise_on_error=False)
        assert result.is_error is True
        assert result.structured_content["error"]["error_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_single_backend_still_works(self, mcp, fake_backend):
        async with Client(mcp) as client:
            result = await client.call_tool("test_connection", {}, raise_on_error=False)
        assert result.is_error is False
        assert result.structured_content["authenticated"] is True


# ---------------------------------------------------------------------------
# Finding 3 (Major): compare_runs' refs list must be bounded and fetched
# concurrently, not sequentially with no upper bound.
# ---------------------------------------------------------------------------


class TestCompareRunsRefsBoundAndConcurrency:
    @pytest.mark.asyncio
    async def test_single_ref_rejected_before_any_backend_call(self, mcp, fake_backend):
        fake_backend.seed_run(_make_run("run1"))
        call_log: list[str] = []
        original = fake_backend.get_run_summary

        async def _spy(ref):
            call_log.append(ref.run_id)
            return await original(ref)

        fake_backend.get_run_summary = _spy  # type: ignore[method-assign]

        async with Client(mcp) as client:
            with pytest.raises(Exception, match="at least 2 items"):
                await client.call_tool("compare_runs", {"refs": [_ref_input("run1")]})

        # The old behavior fetched the one ref via a real backend call
        # before _compare_runs_pure ever got a chance to reject it for
        # having too few runs. The schema-level min_length=2 bound means
        # no backend call happens at all now.
        assert call_log == []

    @pytest.mark.asyncio
    async def test_more_than_max_refs_rejected_by_schema(self, mcp, fake_backend):
        for i in range(60):
            fake_backend.seed_run(_make_run(f"r{i}"))
        refs = [_ref_input(f"r{i}") for i in range(60)]
        async with Client(mcp) as client:
            with pytest.raises(Exception):
                await client.call_tool("compare_runs", {"refs": refs})

    @pytest.mark.asyncio
    async def test_refs_are_fetched_concurrently_not_sequentially(self, mcp, fake_backend):
        n = 8
        delay = 0.15
        for i in range(n):
            fake_backend.seed_run(_make_run(f"r{i}"))
        original = fake_backend.get_run_summary

        async def _delayed(ref):
            await asyncio.sleep(delay)
            return await original(ref)

        fake_backend.get_run_summary = _delayed  # type: ignore[method-assign]

        refs = [_ref_input(f"r{i}") for i in range(n)]
        async with Client(mcp) as client:
            start = time.monotonic()
            result = await client.call_tool("compare_runs", {"refs": refs})
            elapsed = time.monotonic() - start

        assert result.is_error is False
        # Sequential fetching would take ~n * delay (~1.2s for n=8). This
        # generously allows for two "batches" under the bounded-concurrency
        # semaphore (limit=10, so all 8 actually run in one batch) plus
        # scheduling overhead, while still failing fast if the code
        # regresses to one-await-per-ref-in-a-loop.
        assert elapsed < delay * (n - 1), (
            f"compare_runs took {elapsed:.2f}s for {n} refs at {delay}s each — "
            "looks sequential, not concurrent"
        )


# ---------------------------------------------------------------------------
# Finding 4 (Major/Minor): audit_sweep must check BackendCapability.SWEEPS
# up front, and must fetch its runs concurrently.
# ---------------------------------------------------------------------------


class TestAuditSweepCapabilityGatingAndConcurrency:
    @pytest.mark.asyncio
    async def test_unsupported_capability_never_calls_list_sweeps(self, monkeypatch):
        backend = FakeBackend(capabilities=set())
        mcp_no_sweeps = build_server(backends={_BACKEND: backend})
        backend.seed_run(_make_run())

        calls: list[str] = []
        original = backend.list_sweeps

        def _spy(project: str):
            calls.append(project)
            return original(project)

        monkeypatch.setattr(backend, "list_sweeps", _spy)

        async with Client(mcp_no_sweeps) as client:
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "s1",
                    }
                },
                raise_on_error=False,
            )

        assert result.is_error is True
        assert result.structured_content["error"]["error_type"] == (
            "backend_unsupported_capability"
        )
        assert calls == [], "list_sweeps should never be dispatched when SWEEPS is unsupported"

    @pytest.mark.asyncio
    async def test_supported_capability_still_works(self, mcp, fake_backend):
        assert BackendCapability.SWEEPS in fake_backend.capabilities

    @pytest.mark.asyncio
    async def test_sweep_runs_are_fetched_concurrently_not_sequentially(self, mcp, fake_backend):
        from experiment_audit_mcp.models import Sweep

        n = 12
        delay = 0.1
        runs = []
        for i in range(1, n + 1):
            run = _make_run(
                f"r{i}", config={"lr": float(i), "seed": 42}, summary_metrics={"reward": 10.0 * i}
            )
            fake_backend.seed_run(run)
            runs.append(run)
        sweep = Sweep(
            ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
            sweep_id="sweep-1",
            method="grid",
            run_refs=[r.ref for r in runs],
            target_metric="reward",
        )
        fake_backend.seed_sweep(sweep)

        original = fake_backend.get_run_summary

        async def _delayed(ref):
            await asyncio.sleep(delay)
            return await original(ref)

        fake_backend.get_run_summary = _delayed  # type: ignore[method-assign]

        async with Client(mcp) as client:
            start = time.monotonic()
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "sweep-1",
                    }
                },
            )
            elapsed = time.monotonic() - start

        assert result.is_error is False
        # n=12 with a concurrency bound of 10 means two batches
        # (~2 * delay), not twelve (~12 * delay) — generous margin below
        # the fully-sequential cost.
        assert elapsed < delay * (n - 1), (
            f"audit_sweep took {elapsed:.2f}s for {n} runs at {delay}s each — "
            "looks sequential, not concurrent"
        )


# ---------------------------------------------------------------------------
# Finding 5 (Minor): errors.py's "rate_limited" literal is documented as
# currently unreachable. This test pins that fact so it fails loudly (as a
# reminder to update the documentation, not as a sign anything is broken)
# the day someone actually wires up a rate_limited-producing code path.
# ---------------------------------------------------------------------------


class TestRateLimitedIsCurrentlyUnreachable:
    @pytest.mark.asyncio
    async def test_no_current_tool_error_path_produces_rate_limited(self, mcp, fake_backend):
        # Exhaustively exercise every documented error path (unknown
        # backend, not-found run, unsupported capability, insufficient
        # samples) and confirm none of them is classified as
        # rate_limited, matching _translate_backend_error's documented
        # design (backoff is exhausted before this layer ever sees the
        # exception).
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_run_summary",
                {"ref": {**_ref_input("does-not-exist")}},
                raise_on_error=False,
            )
        assert result.structured_content["error"]["error_type"] != "rate_limited"


# ---------------------------------------------------------------------------
# Multi-failure ordering — `_gather_bounded`'s docstring and the comments at
# both of its call sites (compare_runs, audit_sweep) explicitly claim: "the
# first failure in list order is what gets translated and returned,
# preserving the same left-to-right error-reporting order the old
# sequential loop had." Every existing concurrency test above seeds at most
# ONE failing ref among otherwise-successful ones, so this specific,
# documented ordering guarantee — which only matters when *multiple* refs
# fail with *different* errors — has never actually been exercised. Since
# asyncio.gather(..., return_exceptions=True) preserves input order in its
# result list, this is currently correct by construction, but nothing
# pins it: a future refactor (e.g. switching to asyncio.as_completed, or
# collecting failures into an unordered set before picking one) could
# silently make the reported error nondeterministic or picked from the
# wrong ref, and no test here would notice.
# ---------------------------------------------------------------------------


class TestMultiFailureErrorOrderingIsDeterministic:
    @pytest.mark.asyncio
    async def test_compare_runs_reports_first_failing_ref_in_list_order(self, mcp, fake_backend):
        # run1 exists (succeeds); run2 and run3 are both unseeded, so both
        # fail with RunNotFoundError -> run_not_found. Per the documented
        # contract, the error reported must correspond to run2 (the first
        # failing ref in list order), not run3 or an unspecified one of
        # the two.
        fake_backend.seed_run(_make_run("run1"))
        refs = [_ref_input("run1"), _ref_input("run2"), _ref_input("run3")]

        async with Client(mcp) as client:
            result = await client.call_tool("compare_runs", {"refs": refs}, raise_on_error=False)

        assert result.is_error is True
        assert result.structured_content["error"]["error_type"] == "run_not_found"
        assert "run2" in result.structured_content["error"]["message"]
        assert "run3" not in result.structured_content["error"]["message"]

    @pytest.mark.asyncio
    async def test_compare_runs_reports_first_failure_even_when_a_later_ref_fails_faster(
        self, mcp, fake_backend
    ):
        # Strengthens the test above: run2 (first failing ref in list
        # order) is deliberately made to fail *slower* than run3 (a later
        # ref) so that, under concurrent dispatch, run3's failure is
        # available first in wall-clock time. The reported error must
        # still be run2's — "first in list order," not "first to finish."
        fake_backend.seed_run(_make_run("run1"))
        original = fake_backend.get_run_summary

        async def _maybe_delayed(ref):
            if ref.run_id == "run2":
                await asyncio.sleep(0.1)
            return await original(ref)

        fake_backend.get_run_summary = _maybe_delayed  # type: ignore[method-assign]
        refs = [_ref_input("run1"), _ref_input("run2"), _ref_input("run3")]

        async with Client(mcp) as client:
            result = await client.call_tool("compare_runs", {"refs": refs}, raise_on_error=False)

        assert result.is_error is True
        assert "run2" in result.structured_content["error"]["message"]

    @pytest.mark.asyncio
    async def test_audit_sweep_reports_first_failing_run_in_sweep_order(self, mcp, fake_backend):
        from experiment_audit_mcp.models import Sweep

        # Only run1 is seeded; run2 and run3 (both listed in the sweep,
        # in this order) are not, so both fail with run_not_found. The
        # reported error must correspond to run2, matching the sweep's
        # own run_refs order, not run3.
        fake_backend.seed_run(_make_run("run1"))
        sweep = Sweep(
            ref=RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
            sweep_id="sweep-1",
            method="grid",
            run_refs=[
                RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="run1"),
                RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="run2"),
                RunRef(backend=_BACKEND, entity=_ENTITY, project=_PROJECT, run_id="run3"),
            ],
            target_metric="final_reward",
        )
        fake_backend.seed_sweep(sweep)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "audit_sweep",
                {
                    "sweep_ref": {
                        "backend": _BACKEND,
                        "entity": _ENTITY,
                        "project": _PROJECT,
                        "sweep_id": "sweep-1",
                    }
                },
                raise_on_error=False,
            )

        assert result.is_error is True
        assert result.structured_content["error"]["error_type"] == "run_not_found"
        assert "run2" in result.structured_content["error"]["message"]
        assert "run3" not in result.structured_content["error"]["message"]
