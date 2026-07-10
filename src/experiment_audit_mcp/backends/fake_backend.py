"""In-memory ExperimentBackend test double, seeded and adversary-capable.

This is the primary tool used to test tool logic in later milestones
(compare_runs, audit_training_curve, audit_ablation, audit_sweep) without
a live API dependency or W&B rate limits — see design-spec-v1.md §7 and
the roadmap's rationale for building this before the real W&B backend.

Design choice: rather than hard-coding named methods for each adversarial
scenario (`seed_tiny_sweep()`, `seed_nan_curve()`, ...), `FakeBackend`
exposes generic seeding primitives (`seed_run`, `seed_metric_history`,
`seed_sweep`) that construct arbitrary `Run`/`MetricHistory`/`Sweep`
objects. The adversarial *scenarios* themselves (tiny sweep, partial
data, NaN mid-curve, correlated hyperparameters) are then just specific
data shapes built from these primitives, normally via pytest fixtures
in the tests that need them. This keeps `FakeBackend` a general-purpose
double rather than a growing pile of scenario-specific methods — new
adversarial cases in later milestones don't require touching this file.
"""

from __future__ import annotations

from experiment_audit_mcp.backends.base import (
    BackendCapability,
    ConnectionStatus,
    ExperimentBackend,
    RunFilter,
)
from experiment_audit_mcp.models import MetricHistory, Page, Run, RunRef, Sweep


class RunNotFoundError(Exception):
    """Raised by FakeBackend when a requested RunRef has not been seeded.

    Mirrors the `run_not_found` error_type from errors.py at the backend
    layer; translating this into a `ToolError` is a tool-layer (Milestone
    4+) concern, not this backend's.
    """

    def __init__(self, ref: RunRef) -> None:
        self.ref = ref
        super().__init__(
            f"No run seeded for backend={ref.backend!r} "
            f"project={ref.project!r} run_id={ref.run_id!r}."
        )


class MetricHistoryNotFoundError(Exception):
    """Raised by FakeBackend when no history was seeded for (ref, metric)."""

    def __init__(self, ref: RunRef, metric: str) -> None:
        self.ref = ref
        self.metric = metric
        super().__init__(
            f"No metric history seeded for run_id={ref.run_id!r}, "
            f"metric={metric!r}."
        )


class FakeBackend(ExperimentBackend):
    """Fully in-memory backend used only in tests.

    `capabilities` defaults to `{SWEEPS}` since most tests want a fully
    capable fake; pass `capabilities=set()` to construct a capability-less
    fake for testing the `NotSupportedError` default-method path (spec §3,
    validated here in code per the Milestone 2 completion criteria).
    """

    name = "fake"

    def __init__(self, capabilities: set[BackendCapability] | None = None) -> None:
        self.capabilities = (
            capabilities if capabilities is not None else {BackendCapability.SWEEPS}
        )
        self._runs: dict[RunRef, Run] = {}
        self._histories: dict[tuple[RunRef, str], MetricHistory] = {}
        self._sweeps: dict[tuple[str, str, str], Sweep] = {}  # (backend, project, sweep_id)
        self._connection_status: ConnectionStatus | None = None

    # -- seeding / adversarial-state injection -----------------------------

    def seed_run(self, run: Run) -> None:
        """Register a run so it's returned by list_runs/get_run_summary.

        Used directly for happy-path runs, and equally for adversarial
        states — e.g. a run with `data_completeness="partial"` — since
        `Run` already carries that field (spec §2); FakeBackend doesn't
        need a separate code path for it.
        """
        self._runs[run.ref] = run

    def seed_metric_history(self, history: MetricHistory) -> None:
        """Register a metric history, keyed by (ref, metric_name).

        Used for both clean curves and adversarial ones — a history
        whose `points` include `MetricPoint(step, value=None)` entries
        represents a logged NaN mid-curve (spec §7) and needs no special
        handling here beyond preserving the points list exactly as given.
        """
        self._histories[(history.ref, history.metric_name)] = history

    def seed_sweep(self, sweep: Sweep) -> None:
        """Register a sweep so it's returned by list_sweeps.

        Tiny sweeps (few run_refs) and sweeps with correlated
        hyperparameters (encoded in the seeded runs' `config` dicts, via
        `seed_run`) are both just ordinary `Sweep`/`Run` objects — no
        dedicated method needed for either adversarial case.
        """
        key = (sweep.ref.backend, sweep.ref.project, sweep.sweep_id)
        self._sweeps[key] = sweep

    def set_connection_status(self, status: ConnectionStatus) -> None:
        """Override what test_connection() returns (e.g. to simulate an
        auth failure) instead of the default authenticated response."""
        self._connection_status = status

    # -- ExperimentBackend contract -----------------------------------------

    async def test_connection(self) -> ConnectionStatus:
        if self._connection_status is not None:
            return self._connection_status
        return ConnectionStatus(
            backend=self.name, authenticated=True, scopes_detected=["read"]
        )

    async def list_runs(
        self,
        project: str,
        filters: RunFilter | None = None,
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Page[Run]:
        runs = [r for r in self._runs.values() if r.ref.project == project]
        if filters is not None:
            runs = [r for r in runs if _matches(r, filters)]
        # No pagination logic yet: FakeBackend returns everything seeded
        # for the project in a single page. Real cursor-based pagination
        # is exercised against the actual W&B API in Milestone 3 — nothing
        # in spec §7's adversarial cases or Milestone 2's completion
        # criteria requires multi-page behavior from the fake.
        #
        # `page_size` (Revision 2) is accepted, for ABC-signature parity
        # with `WandbBackend`, but deliberately not enforced here — same
        # rationale as `cursor` above. A test that needs to verify
        # page_size *behavior* exercises it against `WandbBackend`
        # (Milestone 3+), not this fake.
        return Page(items=runs, next_cursor=None)

    async def get_run_summary(self, ref: RunRef) -> Run:
        try:
            return self._runs[ref]
        except KeyError:
            raise RunNotFoundError(ref) from None

    async def get_metric_history(
        self,
        ref: RunRef,
        metric: str,
        step_range: tuple[int, int] | None = None,
    ) -> MetricHistory:
        try:
            history = self._histories[(ref, metric)]
        except KeyError:
            raise MetricHistoryNotFoundError(ref, metric) from None

        if step_range is None:
            return history

        lo, hi = step_range
        filtered_points = [p for p in history.points if lo <= p.step <= hi]
        return MetricHistory(
            ref=history.ref,
            metric_name=history.metric_name,
            points=filtered_points,
            schema_version=history.schema_version,
        )

    def list_sweeps(self, project: str) -> list[Sweep]:
        if BackendCapability.SWEEPS not in self.capabilities:
            # Falls through to ExperimentBackend's default, which raises
            # NotSupportedError — this is the exact mechanism spec §3
            # describes, exercised here against a fake before it has to
            # hold up against a real second backend (Milestone 3+).
            return super().list_sweeps(project)
        return [s for s in self._sweeps.values() if s.ref.project == project]


def _matches(run: Run, filters: RunFilter) -> bool:
    if filters.tags is not None and not set(filters.tags).issubset(run.tags):
        return False
    if filters.status is not None and run.status != filters.status:
        return False
    if filters.created_after is not None and run.created_at < filters.created_after:
        return False
    if filters.created_before is not None and run.created_at > filters.created_before:
        return False
    return True
