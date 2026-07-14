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
from typing import Any

from fastmcp import FastMCP

from experiment_audit.analysis.comparison import CompareRunsError
from experiment_audit.analysis.comparison import compare_runs as _compare_runs_pure
from experiment_audit.analysis.confound import audit_ablation as _audit_ablation_pure
from experiment_audit.analysis.divergence import (
    audit_training_curve as _audit_training_curve_pure,
)
from experiment_audit.analysis.sensitivity import InsufficientSamplesError, SweepAuditError
from experiment_audit.analysis.sensitivity import audit_sweep as _audit_sweep_pure
from experiment_audit.auth import MissingCredentialsError
from experiment_audit.backends.base import (
    ExperimentBackend,
    NotSupportedError,
    RunFilter,
)
from experiment_audit.backends.fake_backend import (
    MetricHistoryNotFoundError as FakeMetricHistoryNotFoundError,
)
from experiment_audit.backends.fake_backend import (
    RunNotFoundError as FakeRunNotFoundError,
)
from experiment_audit.backends.wandb_backend import (
    WandbAuthenticationError,
    WandbBackend,
    WandbRunNotFoundError,
)
from experiment_audit.models import Run, RunRef

logger = logging.getLogger("experiment_audit")

_DEFAULT_LIST_RUNS_PAGE_SIZE = 25


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

    backend: str
    entity: str
    project: str
    run_id: str

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

    tags: list[str] | None = None
    status: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    def to_model(self) -> RunFilter:
        return RunFilter(
            tags=self.tags,
            status=self.status,
            created_after=self.created_after,
            created_before=self.created_before,
        )


@dataclass
class SweepRefInput:
    """Identifies one sweep: backend name, entity, project, sweep_id.

    Mirrors `RunRefInput`'s shape but for sweeps. `Sweep.ref` (models.py)
    is itself a `RunRef` whose `run_id` field is an unused placeholder
    (see that model's docstring) — this boundary type uses `sweep_id`
    instead of `run_id` precisely so that placeholder confusion doesn't
    leak into the MCP tool schema too.
    """

    backend: str
    entity: str
    project: str
    sweep_id: str


# -- Error translation (spec §5) -----------------------------------------


def _error_dict(
    error_type: str,
    message: str,
    recoverable: bool,
    retry_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Build a `ToolError`-shaped dict (spec §5) as a tool's return value.

    Constructs an actual `errors.ToolError` first (so the `error_type`
    literal set stays the single source of truth defined in Milestone 2)
    and serializes it here, rather than adding a `to_dict()` method to
    the Milestone-2-approved `ToolError` dataclass — this keeps that file
    untouched for a Milestone 4 concern.
    """
    from experiment_audit.errors import ToolError

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


def _unknown_backend_error(name: str, known: set[str]) -> dict[str, Any]:
    return _error_dict(
        error_type="unknown",
        message=f"Unknown backend {name!r}. Configured backends: {sorted(known)}.",
        recoverable=False,
    )


def _translate_backend_error(exc: Exception) -> dict[str, Any]:
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
    async def test_connection() -> dict[str, Any]:
        """Validate credentials against the configured backend. Call this
        first — failing fast here is cheaper than failing three tool
        calls deep into a task."""
        target = _primary_backend()
        status = await target.test_connection()
        return status.to_dict()

    @mcp.tool
    async def list_runs(
        backend: str,
        project: str,
        filters: RunFilterInput | None = None,
        cursor: str | None = None,
        page_size: int = _DEFAULT_LIST_RUNS_PAGE_SIZE,
    ) -> dict[str, Any]:
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
    async def get_run_summary(ref: RunRefInput) -> dict[str, Any]:
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
        metric: str,
        step_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Fetch the full recorded history of one metric for one run.
        Logged NaN/null points are preserved exactly, never dropped."""
        target = _resolve_backend(ref.backend)
        if target is None:
            return _unknown_backend_error(ref.backend, set(backends))
        try:
            history = await target.get_metric_history(ref.to_model(), metric, step_range)
        except Exception as exc:  # noqa: BLE001
            return _translate_backend_error(exc)
        return history.to_dict()

    @mcp.tool
    async def compare_runs(refs: list[RunRefInput]) -> dict[str, Any]:
        """Diff config and summary metrics across two or more runs.
        Pure deterministic diffing — no verdict, no confidence field
        (spec §4.1: this is why the tool keeps the compare_ prefix
        rather than audit_). Runs may span different projects or even
        different backends; each value in the result is keyed by its
        run's full ref so nothing is ambiguous."""
        runs: list[Run] = []
        for ref_input in refs:
            target = _resolve_backend(ref_input.backend)
            if target is None:
                return _unknown_backend_error(ref_input.backend, set(backends))
            try:
                run = await target.get_run_summary(ref_input.to_model())
            except Exception as exc:  # noqa: BLE001
                return _translate_backend_error(exc)
            runs.append(run)

        try:
            result = _compare_runs_pure(runs)
        except CompareRunsError as exc:
            return _error_dict("unknown", str(exc), recoverable=False)
        return result.to_dict()

    @mcp.tool
    async def audit_training_curve(ref: RunRefInput, metric: str) -> dict[str, Any]:
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
        baseline: RunRefInput, ablation: RunRefInput, claimed_variable: str
    ) -> dict[str, Any]:
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
        sweep_ref: SweepRefInput, target_metric: str | None = None
    ) -> dict[str, Any]:
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

        runs: list[Run] = []
        for ref in sweep.run_refs:
            try:
                runs.append(await target.get_run_summary(ref))
            except Exception as exc:  # noqa: BLE001
                return _translate_backend_error(exc)

        try:
            result = _audit_sweep_pure(sweep, runs, target_metric)
        except InsufficientSamplesError as exc:
            # spec §4.2's literal shape is
            # {error: "insufficient_samples", run_count, minimum_required}
            # — this codebase's uniform ToolError dict (_error_dict, used
            # by every other tool) is preserved and extended with those
            # two fields alongside it, rather than returning a one-off
            # shape just for this tool.
            return {
                **_error_dict("insufficient_samples", str(exc), recoverable=False),
                "run_count": exc.run_count,
                "minimum_required": exc.minimum_required,
            }
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