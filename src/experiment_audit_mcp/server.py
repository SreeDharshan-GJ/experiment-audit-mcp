"""MCP server entrypoint (FastMCP). Registers all tools.

Milestone 4: wires up `test_connection`, `list_runs`, `get_run_summary`,
`get_metric_history` as real MCP tools per design-spec-v1.md §4.2,
backed by `WandbBackend`. Milestone 5 adds `compare_runs`, a thin
MCP-boundary wrapper around the pure `analysis.comparison.compare_runs`
library function: this tool's only job is resolving each `RunRef` into
a `Run` via the target backend(s) and translating errors — the actual
diffing logic lives entirely in `analysis/comparison.py` and has no
knowledge that an MCP call is involved. Milestone 6 adds
`audit_training_curve` following the identical pattern: this tool's only
job is fetching the metric's `MetricHistory` via `get_metric_history` and
translating errors — the four signal detectors live entirely in
`analysis/divergence.py` and likewise have no knowledge that an MCP call
is involved. Milestone 7 adds `audit_ablation`, again the same pattern:
this tool's only job is resolving `baseline`/`ablation` into `Run`s via
`get_run_summary` and translating errors — the verdict/confidence/
allowlist logic lives entirely in `analysis/confound.py`, which itself
calls `analysis/comparison.py`'s `compare_runs` directly rather than
duplicating diff logic, and neither module has any knowledge that an MCP
call is involved. Extended in Milestone 8 with the remaining audit tool.

**Structured errors, not bare exceptions, cross the MCP boundary**
(spec §5): every tool catches the backend-specific exceptions it can
raise and returns a `ToolError`-shaped dict instead of letting the
exception propagate through the MCP protocol layer as an opaque
failure. See `_translate_backend_error` below.

**Testability / credential injection:** `build_server()` takes an
optional `backends` mapping. Called with no arguments (what `main()`
does), it constructs a real `WandbBackend()`, which eagerly loads
`WANDB_API_KEY` via `auth.py`'s fail-fast contract and raises
`MissingCredentialsError` immediately if it's unset — this is
deliberate (spec §6) and not caught here. Tests pass an explicit
`backends` dict (typically a `WandbBackend` constructed with an
injected fake client) so the tool layer is fully testable without real
credentials or network access, consistent with Milestone 3's approach.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from experiment_audit_mcp.analysis.comparison import CompareRunsError
from experiment_audit_mcp.analysis.comparison import compare_runs as _compare_runs_pure
from experiment_audit_mcp.analysis.confound import audit_ablation as _audit_ablation_pure
from experiment_audit_mcp.analysis.divergence import (
    audit_training_curve as _audit_training_curve_pure,
)
from experiment_audit_mcp.analysis.sensitivity import InsufficientSamplesError, SweepAuditError
from experiment_audit_mcp.analysis.sensitivity import audit_sweep as _audit_sweep_pure
from experiment_audit_mcp.auth import MissingCredentialsError
from experiment_audit_mcp.backends.base import (
    BackendCapability,
    ExperimentBackend,
    NotSupportedError,
    RunFilter,
)
from experiment_audit_mcp.backends.fake_backend import (
    MetricHistoryNotFoundError as FakeMetricHistoryNotFoundError,
)
from experiment_audit_mcp.backends.fake_backend import (
    RunNotFoundError as FakeRunNotFoundError,
)
from experiment_audit_mcp.backends.wandb_backend import (
    WandbAuthenticationError,
    WandbBackend,
    WandbRunNotFoundError,
)
from experiment_audit_mcp.models import Run, RunRef

logger = logging.getLogger("experiment_audit_mcp")

_DEFAULT_LIST_RUNS_PAGE_SIZE = 25

# **Bugfix:** `compare_runs`/`audit_sweep` used to fetch every ref's `Run`
# sequentially (one `await` per ref, in a plain `for` loop) with no upper
# bound on how many refs a single call could request. A single tool call
# with N refs therefore cost N sequential backend round trips (each
# possibly carrying its own retry/backoff delay) before returning
# anything, and nothing stopped a caller from passing an arbitrarily
# large `refs` list. `_MAX_CONCURRENT_BACKEND_FETCHES` bounds how many of
# those calls run at once (via `_gather_bounded` below) so a large
# request fans out instead of serializing, without opening unbounded
# concurrent connections against the backend. `_MAX_COMPARE_RUNS_REFS`
# caps `compare_runs`' `refs` list length at the schema level, the same
# way `list_runs`' `page_size` is already bounded (`ge=1, le=500` below)
# rather than left unbounded.
_MAX_CONCURRENT_BACKEND_FETCHES = 10
_MAX_COMPARE_RUNS_REFS = 50

# **Bugfix (Audit #10 finding):** `audit_sweep` fetches a `Run` for every
# ref in `sweep.run_refs` (below) via the same `_gather_bounded` helper
# `compare_runs` uses -- but unlike `compare_runs`, whose `refs` list is a
# caller-supplied MCP argument capped at the schema level by
# `_MAX_COMPARE_RUNS_REFS`, `sweep.run_refs` comes back from the backend's
# `list_sweeps` call with no length limit anywhere. `_gather_bounded`'s
# `limit` only bounds how many fetches run *concurrently*, not how many
# fetches are issued in total: a sweep with, say, 20,000 runs still
# resolves into 20,000 `get_run_summary` calls (confirmed by direct
# reproduction against `FakeBackend` -- see
# tests/test_audit_10_hardening.py), each held in memory as a full `Run`
# (config + summary_metrics) simultaneously in the `runs` list before
# `_audit_sweep_pure` even starts, and `_audit_sweep_pure`'s own
# `_covariance_warnings` pass is O(parameters^2 * runs) on top of that.
# A backend under an attacker's influence (or simply a very large,
# legitimate sweep) can therefore turn one `audit_sweep` call into
# unbounded backend fan-out, unbounded held memory, and unbounded CPU --
# the exact resource-exhaustion shape `_MAX_COMPARE_RUNS_REFS` exists to
# prevent for `compare_runs`, just left unaddressed on this second path
# that has the identical fan-out structure. `_MAX_AUDIT_SWEEP_RUNS`
# closes the same gap here: sized generously above any real sweep this
# tool is meant to audit (spec's own sample-size guidance tops out in the
# low hundreds of runs), rejected before any backend fetch is issued.
_MAX_AUDIT_SWEEP_RUNS = 2000

# **Bugfix (Audit #10 finding):** `get_metric_history` returns every
# point `MetricHistory.points` holds, with no upper bound, via
# `history.to_dict()`. That's unlike `_json_safe`'s handling of `config`/
# `summary_metrics` (models.py), which caps container size at
# `_MAX_CONTAINER_ITEMS` -- there is no equivalent cap for metric points.
# Long training runs (a multi-million-step RL run logging a metric every
# step is ordinary, not adversarial) turn a single `get_metric_history`
# call into a response tens or hundreds of megabytes of JSON, and, direct
# reproduction confirms, enough of them (5,000,000 points, seeded via
# `FakeBackend` so the growth is attributable to this server's own
# serialization path rather than any real W&B network behavior) OOM-kills
# the server process outright -- see tests/test_audit_10_hardening.py.
#
# Truncating `points` the way `_json_safe` truncates dict items is *not*
# a safe fix here, unlike for config/summary: this module's own
# docstring states NaN/null metric points "must never be silently
# dropped... during filtering, aggregation, or serialization" as a
# spec-critical invariant (models.py, MetricHistory), and
# `get_metric_history`'s own docstring repeats the same promise. Silently
# truncating would violate that promise while looking like a complete,
# trustworthy history -- worse than refusing. So this cap refuses with a
# structured error instead of truncating, consistent with this
# codebase's established "refuse rather than mislead" pattern
# (`insufficient_samples` in analysis/sensitivity.py; the step_range
# ordering check in `get_metric_history` itself, just below).
#
# This closes the acute half of the issue: the oversized JSON payload
# never crosses the MCP boundary to a client. It does not eliminate the
# backend-side cost of the initial fetch itself (by the time this check
# runs, `WandbBackend.get_metric_history`'s `scan_history` call has
# already pulled every point into memory in this process) -- narrowing
# that further would mean pushing a point limit into the backend
# interface's `get_metric_history` signature so backends can stream/page
# rather than fetch-then-check, a larger interface change than this
# audit's MCP-layer scope. Documented here as an acknowledged residual
# gap, the same way `list_sweeps`' async-cancellation limitation is
# documented and left open elsewhere in this file.
_MAX_METRIC_HISTORY_POINTS = 200_000


async def _gather_bounded(
    coros: list[Any], limit: int = _MAX_CONCURRENT_BACKEND_FETCHES
) -> list[Any]:
    """Run `coros` concurrently, at most `limit` at a time, collecting
    exceptions instead of raising them (`return_exceptions=True`) so the
    caller can translate the first failure in the same left-to-right
    order the old sequential `for` loop used to report it in."""
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros), return_exceptions=True)


# -- Tool-layer boundary types ------------------------------------------
#
# Deliberately separate from the models.py / backends.base types they
# wrap: MCP tool JSON schemas need short, LLM-facing field descriptions
# (spec §6 — "verbose tool descriptions cost context budget on every
# turn"), not the developer-facing prose docstrings RunRef/RunFilter
# carry for humans reading the source (e.g. RunRef's docstring explains
# the Revision 1 entity migration — useful context in models.py, dead
# weight repeated in every tool call's schema). No frozen model or
# backend-abstraction type changed to introduce these; they exist only
# at this boundary and convert to the real types immediately.


@dataclass
class RunRefInput:
    """Identifies one run: backend name, entity, project, run_id."""

    backend: Annotated[
        str, Field(description="Configured backend name, e.g. 'wandb'.", min_length=1)
    ]
    entity: Annotated[str, Field(description="W&B team/user namespace.", min_length=1)]
    project: Annotated[str, Field(description="Project name within the entity.", min_length=1)]
    run_id: Annotated[str, Field(description="Run ID within the project.", min_length=1)]

    def to_model(self) -> RunRef:
        return RunRef(
            backend=self.backend,
            entity=self.entity,
            project=self.project,
            run_id=self.run_id,
        )


@dataclass
class RunFilterInput:
    """Optional list_runs filters. An unset field applies no filtering."""

    tags: Annotated[
        list[str] | None,
        Field(description="Only runs with at least one of these tags."),
    ] = None
    status: Annotated[
        str | None,
        Field(
            description=(
                "Only runs with this exact status (backend-native value, "
                "e.g. 'running', 'finished', 'crashed')."
            )
        ),
    ] = None
    created_after: Annotated[
        datetime | None,
        Field(description="Only runs created at or after this ISO-8601 timestamp."),
    ] = None
    created_before: Annotated[
        datetime | None,
        Field(description="Only runs created at or before this ISO-8601 timestamp."),
    ] = None

    def to_model(self) -> RunFilter:
        return RunFilter(
            tags=self.tags,
            status=self.status,
            created_after=self.created_after,
            created_before=self.created_before,
        )


# Mirrors `RunRefInput`'s shape but for sweeps. `Sweep.ref` (models.py) is
# itself a `RunRef` whose `run_id` field is an unused placeholder (see that
# model's docstring) — this boundary type uses `sweep_id` instead of
# `run_id` precisely so that placeholder confusion doesn't leak into the
# MCP tool schema too. NOTE: this rationale lives in a comment, not the
# class docstring — FastMCP surfaces a dataclass's docstring verbatim as
# the JSON-schema `description` for any tool parameter of this type (see
# `sweep_ref` on `audit_sweep`), so anything written there is LLM-facing
# tool-selection text, not developer documentation (this exact class
# previously leaked this whole paragraph into every audit_sweep call's
# schema before the fix that added this note).
@dataclass
class SweepRefInput:
    """Identifies one sweep: backend name, entity, project, sweep_id."""

    backend: Annotated[
        str, Field(description="Configured backend name, e.g. 'wandb'.", min_length=1)
    ]
    entity: Annotated[str, Field(description="W&B team/user namespace.", min_length=1)]
    project: Annotated[str, Field(description="Project name within the entity.", min_length=1)]
    sweep_id: Annotated[str, Field(description="Sweep ID within the project.", min_length=1)]


# -- Error translation (spec §5) -----------------------------------------


def _error_payload(
    error_type: str,
    message: str,
    recoverable: bool,
    retry_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Build a `ToolError`-shaped dict (spec §5) — the JSON *content* of a
    tool error, with no opinion on how it crosses the MCP boundary.

    Constructs an actual `errors.ToolError` first (so the `error_type`
    literal set stays the single source of truth defined in Milestone 2)
    and serializes it here, rather than adding a `to_dict()` method to
    the Milestone-2-approved `ToolError` dataclass — this keeps that file
    untouched for a Milestone 4 concern. Split out from `_error_dict`
    below so `audit_sweep`'s `insufficient_samples` case — the one error
    shape that needs extra top-level fields (`run_count`,
    `minimum_required`) alongside the standard `error` object — can reuse
    this single source of truth instead of duplicating the four-field
    construction inline.
    """
    from experiment_audit_mcp.errors import ToolError

    error = ToolError(
        error_type=error_type,  # type: ignore[arg-type]
        message=message,
        recoverable=recoverable,
        retry_after_seconds=retry_after_seconds,
    )
    return {
        "error": {
            "error_type": error.error_type,
            "message": error.message,
            "recoverable": error.recoverable,
            "retry_after_seconds": error.retry_after_seconds,
        }
    }


def _error_dict(
    error_type: str,
    message: str,
    recoverable: bool,
    retry_after_seconds: int | None = None,
) -> ToolResult:
    """Build a `ToolError`-shaped result (spec §5) as a tool's return value.

    **Bugfix:** this used to return a bare `dict` shaped like
    `{"error": {...}}`. That dict is exactly what spec §5 asks for as
    *content*, but returning a plain dict from an `@mcp.tool` function
    only ever produces a *successful* `CallToolResult` — FastMCP has no
    way to know from an ordinary dict that it represents a failure, so
    `CallToolResult.isError` was `False` for every single error this
    server has ever returned (auth failures, not-found runs, unsupported
    capabilities, insufficient samples, everything). Any MCP client that
    branches on `isError` — which is the protocol's actual mechanism for
    "this tool call failed", not a convention this codebase invented —
    saw every one of these as an ordinary success and had no
    protocol-level signal to treat the result differently (retry it,
    surface it as a failure to a human, stop a multi-step plan that
    depended on the data actually being there). Verified against a real
    `fastmcp.Client` round trip: a dict return is `isError=False`
    on the wire regardless of its content; a `ToolResult(...,
    is_error=True)` return is `isError=True`, with the same JSON payload
    still available as `structured_content` for a caller that wants to
    parse `error_type`/`recoverable`/`retry_after_seconds` programmatically.
    Returning `ToolResult` here (rather than only fixing this at each of
    the ~10 call sites) makes the fix apply uniformly and makes it
    impossible for a future error path to reintroduce the bug by
    forgetting to set `is_error`.
    """
    payload = _error_payload(error_type, message, recoverable, retry_after_seconds)
    return ToolResult(structured_content=payload, is_error=True)


def _unknown_backend_error(name: str, known: set[str]) -> ToolResult:
    return _error_dict(
        error_type="unknown",
        message=f"Unknown backend {name!r}. Configured backends: {sorted(known)}.",
        recoverable=False,
    )


def _translate_backend_error(exc: Exception) -> ToolResult:
    """Map a backend-raised exception onto a structured ToolError dict,
    so a bare exception never crosses the MCP tool boundary.

    Rate-limiting is deliberately *not* classified here: backoff/retry is
    handled once, in the backend layer (spec §5 — "never duplicated
    per-tool"). By the time an exception reaches this function, the
    backend has already exhausted its retries; re-guessing a
    `retry_after_seconds` here would duplicate a decision the backend
    already made and reject.
    """
    not_found_types = (WandbRunNotFoundError, FakeRunNotFoundError, FakeMetricHistoryNotFoundError)
    if isinstance(exc, not_found_types):
        return _error_dict("run_not_found", str(exc), recoverable=False)
    if isinstance(exc, (MissingCredentialsError, WandbAuthenticationError)):
        return _error_dict("auth_failed", str(exc), recoverable=False)
    if isinstance(exc, NotSupportedError):
        return _error_dict("backend_unsupported_capability", str(exc), recoverable=False)
    logger.exception("Unhandled backend error in tool call")
    return _error_dict("unknown", str(exc), recoverable=False)


def _to_run_summary_dict(run: Run) -> dict[str, Any]:
    """Project a full `Run` down to the cheap `RunSummary` shape spec
    §4.2 describes for `list_runs` — no `config`/`summary_metrics`.
    Reuses `Run.to_dict()` (Milestone 1, already tested) rather than
    re-implementing serialization, then drops the expensive keys.

    **Interpretation note (not a model/spec change):** §4.2's prose lists
    RunSummary's fields as "id, name, tags, status, created_at" — but §2's
    non-negotiable design principle #1 is "Never pass a bare run_id
    string between tools," precisely because a bare id can't be used to
    construct the `RunRef` a follow-up `get_run_summary`/
    `get_metric_history` call needs. Read "id" here as shorthand for "the
    identifying fields" (i.e. the full `ref`), not as license to drop
    entity/project/backend scoping from every list_runs result — dropping
    it would make list_runs's output a dead end. `data_completeness` is
    also included: it's already present on the `Run` object at zero extra
    cost, and omitting it would let a partial-data run look identical to
    a complete one at listing time, which is exactly the "confidently
    wrong" failure mode spec §5 exists to prevent.
    """
    full = run.to_dict()
    return {
        "ref": full["ref"],
        "name": full["name"],
        "tags": full["tags"],
        "status": full["status"],
        "created_at": full["created_at"],
        "data_completeness": full["data_completeness"],
    }


# -- Server construction ---------------------------------------------------


def build_server(backends: dict[str, ExperimentBackend] | None = None) -> FastMCP:
    """Construct the FastMCP server and register the Milestone 4 tools.

    See the module docstring for the `backends`-injection rationale.
    """
    if backends is None:
        backends = {"wandb": WandbBackend()}

    mcp: FastMCP = FastMCP(
        name="experiment-audit-mcp",
        instructions=(
            "Audits ML experiments (W&B today; MLflow planned) for confounded "
            "ablations, training-curve pathologies, and misleading sweep "
            "conclusions. list_runs/get_run_summary/get_metric_history are "
            "deterministic reads; audit_* tools (added in later milestones) "
            "are heuristic judgments and always report their own confidence "
            "and evidence — never trust a bare verdict from one."
        ),
    )

    def _resolve_backend(name: str) -> ExperimentBackend | None:
        return backends.get(name)

    def _primary_backend() -> ExperimentBackend:
        """`test_connection()` has no `backend` parameter in spec §4.2 —
        its signature is literally `test_connection()`. That's only
        resolvable while exactly one backend is configured; this raises
        loudly rather than silently guessing if that stops being true
        (e.g. once an MLflow backend is added in v2). Deciding how
        `test_connection` should address multiple backends is a v2
        design question, not resolved here.
        """
        if len(backends) != 1:
            raise RuntimeError(
                "test_connection() has no `backend` parameter (spec §4.2) "
                "and assumes exactly one configured backend; found "
                f"{sorted(backends)}. Multi-backend selection for this tool "
                "is an open design question for when a second backend is "
                "actually added."
            )
        return next(iter(backends.values()))

    @mcp.tool
    async def test_connection() -> dict[str, Any] | ToolResult:
        """Validate credentials against the configured backend. Call this
        first — failing fast here is cheaper than failing three tool
        calls deep into a task."""
        # **Bugfix:** this used to call `_primary_backend()` unguarded.
        # Every other tool in this file wraps every backend call in a
        # `try/except Exception` that routes through `_translate_backend_error`
        # (per this module's own "structured errors, not bare exceptions
        # cross the MCP boundary" contract, spec §5) — this was the one
        # tool that didn't, so the moment a second backend is configured
        # (the module docstring and `_primary_backend`'s own docstring
        # both anticipate this happening once MLflow support lands),
        # `test_connection` would let a bare `RuntimeError` propagate
        # straight through the MCP boundary as an unstructured, generic
        # FastMCP-formatted error message instead of this codebase's
        # `{"error": {"error_type": ..., ...}}` shape every other failure
        # uses — the one tool call a caller can't parse the same way as
        # every other one. Caught here and translated the same way.
        try:
            target = _primary_backend()
        except RuntimeError as exc:
            return _error_dict("unknown", str(exc), recoverable=False)
        try:
            status = await target.test_connection()
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        return status.to_dict()

    @mcp.tool
    async def list_runs(
        backend: Annotated[
            str, Field(description="Configured backend name, e.g. 'wandb'.", min_length=1)
        ],
        project: Annotated[str, Field(description="Project name to list runs from.", min_length=1)],
        filters: RunFilterInput | None = None,
        cursor: Annotated[
            str | None,
            Field(
                description=(
                    "Opaque next_cursor from a previous list_runs call. Omit for the first page."
                )
            ),
        ] = None,
        page_size: Annotated[
            int,
            Field(description="Max runs to return in this page.", ge=1, le=500),
        ] = _DEFAULT_LIST_RUNS_PAGE_SIZE,
    ) -> dict[str, Any] | ToolResult:
        """List runs in a project. Returns a lightweight summary per run
        (no config or metrics) — call get_run_summary for the full
        record on any run you need to inspect closely."""
        target = _resolve_backend(backend)
        if target is None:
            return _unknown_backend_error(backend, set(backends))
        try:
            page = await target.list_runs(
                project,
                filters=filters.to_model() if filters is not None else None,
                cursor=cursor,
                page_size=page_size,
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            return _translate_backend_error(exc)
        return {
            "items": [_to_run_summary_dict(r) for r in page.items],
            "next_cursor": page.next_cursor,
        }

    @mcp.tool
    async def get_run_summary(ref: RunRefInput) -> dict[str, Any] | ToolResult:
        """Fetch one run's full config and summary metrics. Does not
        include metric history — call get_metric_history separately for
        that (a deliberate, explicit split; see design-spec-v1.md §4.3)."""
        target = _resolve_backend(ref.backend)
        if target is None:
            return _unknown_backend_error(ref.backend, set(backends))
        try:
            run = await target.get_run_summary(ref.to_model())
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        return run.to_dict()

    @mcp.tool
    async def get_metric_history(
        ref: RunRefInput,
        metric: Annotated[
            str, Field(description="Metric key as logged, e.g. 'train/loss'.", min_length=1)
        ],
        step_range: Annotated[
            tuple[int, int] | None,
            Field(
                description=(
                    "Inclusive [start_step, end_step] to restrict the "
                    "fetch. Omit for the full history."
                )
            ),
        ] = None,
    ) -> dict[str, Any] | ToolResult:
        """Fetch the full recorded history of one metric for one run.
        Logged NaN/null points are preserved exactly, never dropped."""
        if step_range is not None and step_range[0] > step_range[1]:
            # Argument-validation gap: pydantic can constrain each tuple
            # element individually but not their relative order, so this
            # check is done here rather than in the schema. Caught before
            # any backend call so a malformed range never turns into a
            # silently-empty (and therefore misleading) history.
            return _error_dict(
                "unknown",
                f"step_range start ({step_range[0]}) must be <= end ({step_range[1]}).",
                recoverable=False,
            )
        target = _resolve_backend(ref.backend)
        if target is None:
            return _unknown_backend_error(ref.backend, set(backends))
        try:
            history = await target.get_metric_history(ref.to_model(), metric, step_range)
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        if len(history.points) > _MAX_METRIC_HISTORY_POINTS:
            # See `_MAX_METRIC_HISTORY_POINTS`'s module-level comment:
            # refusing here, rather than truncating `points`, is
            # deliberate -- truncation would silently violate this same
            # docstring's "never dropped" promise for NaN/null points.
            return _error_dict(
                "unknown",
                f"Metric {metric!r} has {len(history.points)} recorded points, "
                f"which exceeds the {_MAX_METRIC_HISTORY_POINTS} this tool will "
                f"return in one call. Narrow the request with step_range and "
                f"page through the history in smaller windows.",
                recoverable=False,
            )
        return history.to_dict()

    @mcp.tool
    async def compare_runs(
        refs: Annotated[
            list[RunRefInput],
            Field(
                description="Two or more runs to diff (config + summary metrics).",
                # **Bugfix:** unlike every other bounded input in this file
                # (page_size: ge=1,le=500; every string field: min_length=1),
                # `refs` had no length constraint at all. `compare_runs`
                # itself (analysis/comparison.py) already rejects fewer
                # than 2 runs, but that check only happens *after* this
                # tool has already resolved a backend and awaited a real
                # `get_run_summary` call for every ref supplied — so a
                # single-ref call (structurally guaranteed to fail) still
                # paid for a full backend round trip before reporting
                # that. `min_length=2` rejects that case at the schema
                # level, before any I/O. `max_length` caps the other end:
                # nothing bounded how many runs one call could request,
                # so an arbitrarily long `refs` list could fan out into
                # an arbitrarily large number of backend calls from a
                # single tool invocation.
                min_length=2,
                max_length=_MAX_COMPARE_RUNS_REFS,
            ),
        ],
    ) -> dict[str, Any] | ToolResult:
        """Diff config and summary metrics across two or more runs.
        Pure deterministic diffing — no verdict, no confidence field
        (spec §4.1: this is why the tool keeps the compare_ prefix
        rather than audit_). Runs may span different projects or even
        different backends; each value in the result is keyed by its
        run's full ref so nothing is ambiguous."""
        # Resolve every ref's backend up front — this is a synchronous,
        # in-process dict lookup (`_resolve_backend`), not I/O — so an
        # unknown backend anywhere in `refs` is caught before any network
        # call is made for *any* ref, not just the ones before it in the
        # list.
        targets: list[ExperimentBackend] = []
        for ref_input in refs:
            target = _resolve_backend(ref_input.backend)
            if target is None:
                return _unknown_backend_error(ref_input.backend, set(backends))
            targets.append(target)

        # **Bugfix:** this used to fetch each ref's `Run` sequentially (one
        # `await target.get_run_summary(...)` per ref, in a `for` loop),
        # so an N-run comparison cost N sequential backend round trips.
        # Fetched concurrently now (bounded by `_MAX_CONCURRENT_BACKEND_FETCHES`
        # so a large `refs` list can't open unbounded concurrent backend
        # calls at once), with `return_exceptions=True` so a failure on
        # one ref doesn't cancel the others mid-flight — the first
        # failure in list order is what gets translated and returned,
        # preserving the same left-to-right error-reporting order the old
        # sequential loop had.
        fetches = [
            target.get_run_summary(ref_input.to_model())
            for target, ref_input in zip(targets, refs, strict=True)
        ]
        fetch_results = await _gather_bounded(fetches)
        runs: list[Run] = []
        for result in fetch_results:
            if isinstance(result, BaseException):
                return _translate_backend_error(result)  # type: ignore[arg-type]
            runs.append(result)

        try:
            result = _compare_runs_pure(runs)
        except CompareRunsError as exc:
            return _error_dict("unknown", str(exc), recoverable=False)
        return result.to_dict()

    @mcp.tool
    async def audit_training_curve(
        ref: RunRefInput,
        metric: Annotated[
            str, Field(description="Metric key as logged, e.g. 'train/loss'.", min_length=1)
        ],
    ) -> dict[str, Any] | ToolResult:
        """Score one metric's history for training pathologies: logged
        NaNs, sudden level shifts, stalled (low-variance) plateaus, and
        high-frequency oscillation. Always fetches live data via
        get_metric_history. Heuristic judgment - reports continuous
        scores and confidence per signal, never a bare verdict. See
        docs/audit-methods.md#training-curve for exact thresholds."""
        target = _resolve_backend(ref.backend)
        if target is None:
            return _unknown_backend_error(ref.backend, set(backends))
        try:
            history = await target.get_metric_history(ref.to_model(), metric)
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        result = _audit_training_curve_pure(history)
        return result.to_dict()

    @mcp.tool
    async def audit_ablation(
        baseline: RunRefInput,
        ablation: RunRefInput,
        claimed_variable: Annotated[
            str,
            Field(
                description="Config key the ablation claims to isolate, e.g. 'learning_rate'.",
                min_length=1,
            ),
        ],
    ) -> dict[str, Any] | ToolResult:
        """Judge whether an ablation pair cleanly isolates claimed_variable.
        Diffs baseline vs ablation config and checks each differing
        parameter against claimed_variable and a conservative allowlist
        (seed, device, run name/id fields). Heuristic judgment - reports
        verdict, confidence, and the full diff as evidence, never a bare
        verdict. See docs/audit-methods.md#ablation."""
        baseline_target = _resolve_backend(baseline.backend)
        if baseline_target is None:
            return _unknown_backend_error(baseline.backend, set(backends))
        ablation_target = _resolve_backend(ablation.backend)
        if ablation_target is None:
            return _unknown_backend_error(ablation.backend, set(backends))

        try:
            baseline_run = await baseline_target.get_run_summary(baseline.to_model())
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        try:
            ablation_run = await ablation_target.get_run_summary(ablation.to_model())
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)

        try:
            result = _audit_ablation_pure(baseline_run, ablation_run, claimed_variable)
        except CompareRunsError as exc:
            return _error_dict("unknown", str(exc), recoverable=False)
        return result.to_dict()

    @mcp.tool
    async def audit_sweep(
        sweep_ref: SweepRefInput,
        target_metric: Annotated[
            str | None,
            Field(
                description=(
                    "Metric to rank hyperparameters against. Defaults "
                    "to the sweep's own target metric."
                )
            ),
        ] = None,
    ) -> dict[str, Any] | ToolResult:
        """Rank hyperparameter importance across a sweep via pairwise
        Pearson correlation with target_metric (defaults to the sweep's
        own target metric). Refuses with insufficient_samples below a
        run-count floor rather than a shaky ranking; flags co-varying
        hyperparameter pairs. Heuristic judgment - reports
        confidence/evidence/method. See docs/audit-methods.md#sweep
        (correlation is linear-only)."""
        target = _resolve_backend(sweep_ref.backend)
        if target is None:
            return _unknown_backend_error(sweep_ref.backend, set(backends))

        # **Bugfix:** this used to always dispatch `list_sweeps` (via
        # `asyncio.to_thread`, since `list_sweeps` is a plain synchronous
        # method — base.py) and rely on catching the `NotSupportedError`
        # it raises for a backend without sweep support. base.py's own
        # `BackendCapability` docstring documents the intended pattern as
        # the opposite of that: "an audit tool checks [capabilities]
        # before calling an optional method" (its exact words), precisely
        # so a caller gets "a clear, structured refusal instead of a
        # fictional shim" without needing to attempt the call at all.
        # Checking `target.capabilities` here is a synchronous, in-process
        # set-membership test — no thread dispatch, no backend call —
        # so an unsupported backend is now rejected for free, consistent
        # with how the ABC that defines this capability says it should be
        # used.
        if BackendCapability.SWEEPS not in target.capabilities:
            return _error_dict(
                "backend_unsupported_capability",
                f"Backend {sweep_ref.backend!r} does not support capability "
                f"{BackendCapability.SWEEPS.value!r}.",
                recoverable=False,
            )

        # **Known limitation (documented, not fixed here — see the audit
        # summary's cancellation-handling finding):** if the MCP client
        # cancels this tool call while `list_sweeps` is running, asyncio
        # will detach from this `await` and return control to the caller,
        # but the underlying thread-pool worker keeps executing the
        # blocking call (and any retry/backoff sleep inside it) to
        # completion regardless — `asyncio.to_thread` offers no way to
        # interrupt work already dispatched to a thread. That worker
        # occupies one of the default executor's bounded slots for the
        # rest of that duration, so a client that cancels and retries
        # repeatedly against a slow/rate-limited backend can exhaust the
        # executor and stall unrelated concurrent tool calls. A complete
        # fix needs the blocking call itself (inside `WandbBackend`, out
        # of this audit's MCP-layer scope) to poll a cancellation signal;
        # nothing at this boundary can abort a thread already in flight.
        try:
            sweeps = await asyncio.to_thread(target.list_sweeps, sweep_ref.project)
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)

        sweep = next((s for s in sweeps if s.sweep_id == sweep_ref.sweep_id), None)
        if sweep is None:
            # No literal in errors.py's frozen error_type set (§5) names
            # "sweep not found" — run_not_found is about runs
            # specifically, so reusing it here would misdescribe the
            # failure. "unknown" is the most honest fit available; see
            # the Milestone 8 summary for this flagged gap in the frozen
            # taxonomy rather than silently overloading run_not_found.
            return _error_dict(
                "unknown",
                f"No sweep {sweep_ref.sweep_id!r} found in backend "
                f"{sweep_ref.backend!r}, project {sweep_ref.project!r}.",
                recoverable=False,
            )

        # **Bugfix (Audit #10):** `sweep.run_refs` is backend-controlled
        # (returned by `list_sweeps`, not a caller-supplied MCP argument),
        # so it can't be bounded at the schema level the way
        # `compare_runs`' `refs` is (`min_length`/`max_length` above).
        # Checked here instead, before any `get_run_summary` fetch is
        # issued, so an oversized sweep is rejected with a structured
        # error at zero fetch cost rather than fanning out into thousands
        # of backend calls first. See `_MAX_AUDIT_SWEEP_RUNS`'s
        # module-level comment for the full rationale and
        # tests/test_audit_10_hardening.py for the reproduction this
        # closes.
        if len(sweep.run_refs) > _MAX_AUDIT_SWEEP_RUNS:
            return _error_dict(
                "unknown",
                f"Sweep {sweep_ref.sweep_id!r} has {len(sweep.run_refs)} runs, "
                f"which exceeds the {_MAX_AUDIT_SWEEP_RUNS} this tool will "
                f"audit in one call. Split the sweep or audit a subset "
                f"of its runs.",
                recoverable=False,
            )

        # **Bugfix:** this used to fetch each of the sweep's runs
        # sequentially (one `await target.get_run_summary(ref)` per run,
        # in a `for` loop). Sweeps routinely contain dozens to hundreds of
        # runs, so a single `audit_sweep` call cost that many sequential
        # backend round trips — the largest instance of the same
        # sequential-fetch pattern fixed in `compare_runs` above, and for
        # the same reason: nothing about resolving each ref into a `Run`
        # depends on any other ref, so there's no correctness reason for
        # it to be sequential. Bounded concurrent fetch via
        # `_gather_bounded`, same as `compare_runs`.
        fetches = [target.get_run_summary(ref) for ref in sweep.run_refs]
        fetch_results = await _gather_bounded(fetches)
        runs: list[Run] = []
        for result in fetch_results:
            if isinstance(result, BaseException):
                return _translate_backend_error(result)  # type: ignore[arg-type]
            runs.append(result)

        try:
            result = _audit_sweep_pure(sweep, runs, target_metric)
        except InsufficientSamplesError as exc:
            # spec §4.2's literal shape is
            # {error: "insufficient_samples", run_count, minimum_required}
            # — this codebase's uniform ToolError payload (_error_payload,
            # the single source of truth every other tool's error also
            # builds from) is preserved and extended with those two
            # fields alongside it, rather than returning a one-off shape
            # just for this tool.
            #
            # **Bugfix:** this used to do `{**_error_dict(...), ...}` —
            # dict-unpacking a `ToolResult` (now that `_error_dict`
            # returns one instead of a plain dict, per the `isError` fix
            # above) raises `TypeError` immediately, since `ToolResult` is
            # a Pydantic model, not a `Mapping`. Built from
            # `_error_payload` (the plain-dict layer) instead, then
            # wrapped in `ToolResult` once, the same way every other error
            # path here now returns.
            payload = {
                **_error_payload("insufficient_samples", str(exc), recoverable=False),
                "run_count": exc.run_count,
                "minimum_required": exc.minimum_required,
            }
            return ToolResult(structured_content=payload, is_error=True)
        except SweepAuditError as exc:
            return _error_dict("unknown", str(exc), recoverable=False)
        return result.to_dict()

    return mcp


def main() -> None:
    """Production entrypoint. Never logs to stdout: the MCP stdio
    transport uses stdout for protocol messages, so all diagnostic
    output goes to stderr."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp = build_server()
    mcp.run()


if __name__ == "__main__":
    main()