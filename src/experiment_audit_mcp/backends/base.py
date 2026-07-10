"""ExperimentBackend ABC and BackendCapability enum.

Implements the frozen contract from design-spec-v1.md §3. The key design
decision carried over from review: capability declaration, not blanket
abstract methods. `list_sweeps` has a default implementation that raises
`NotSupportedError` rather than being a required abstract method — this
lets a backend like MLflow (no first-class Sweep object) declare
`capabilities = {ARTIFACTS}` (no `SWEEPS`) and get a clear, structured
refusal instead of a fictional shim (see design-spec-v1.md §3, Appendix A).

`ConnectionStatus` and `RunFilter` are defined here rather than in
`models.py` because they are backend-abstraction concerns (the shapes
`test_connection` and `list_runs` speak in), not core experiment data —
`models.py`'s Milestone 1 contract was scoped explicitly to
RunRef/Run/MetricPoint/MetricHistory/Sweep/Page[T] (spec §2) and these two
types are part of §3, not §2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from experiment_audit_mcp.models import MetricHistory, Page, Run, RunRef, Sweep


class BackendCapability(Enum):
    """Optional capabilities a backend may declare support for.

    A backend's `capabilities` set is the single source of truth an
    audit tool checks before calling an optional method — see
    `NotSupportedError` below and design-spec-v1.md §3.
    """

    SWEEPS = "sweeps"
    ARTIFACTS = "artifacts"


class NotSupportedError(Exception):
    """Raised when a backend is asked to do something it doesn't support.

    Carries the backend name and the missing capability so a caller (or
    the MCP tool layer in a later milestone) can surface a clear
    `backend_unsupported_capability` error rather than an opaque
    AttributeError/NotImplementedError.
    """

    def __init__(self, backend_name: str, capability: BackendCapability) -> None:
        self.backend_name = backend_name
        self.capability = capability
        super().__init__(
            f"Backend {backend_name!r} does not support capability "
            f"{capability.value!r}."
        )


@dataclass
class ConnectionStatus:
    """Result of `test_connection()` — spec §4.2: {backend, authenticated,
    scopes_detected, error?}."""

    backend: str
    authenticated: bool
    scopes_detected: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "authenticated": self.authenticated,
            "scopes_detected": list(self.scopes_detected),
            "error": self.error,
        }


@dataclass
class RunFilter:
    """Filters accepted by `list_runs` — spec §4.2: tags, status,
    created_after, created_before. All fields optional; an unset field
    applies no filtering on that dimension."""

    tags: list[str] | None = None
    status: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


class ExperimentBackend(ABC):
    """Common interface every backend (W&B, MLflow, ...) implements.

    `name` and `capabilities` are per-instance declarations, not class
    constants, so a backend can in principle vary capabilities based on
    what it detects at connection time (not exercised until a later
    milestone, but the interface doesn't foreclose it).

    `list_runs`'s `page_size` (Revision 2): a per-call override of the
    backend's configured default page size. `None` means "use whatever
    default the backend instance was constructed with" — this closed a
    real gap between design-spec-v1.md §4.2's `list_runs(..., page_size=25)`
    tool signature and this ABC, which originally had no per-call
    page-size parameter at all. See the Revision 2 entry in
    design-spec-v1.md's Revision Log for the full rationale.
    """

    name: str
    capabilities: set[BackendCapability]

    @abstractmethod
    async def test_connection(self) -> ConnectionStatus: ...

    @abstractmethod
    async def list_runs(
        self,
        project: str,
        filters: RunFilter | None = None,
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Page[Run]: ...

    @abstractmethod
    async def get_run_summary(self, ref: RunRef) -> Run: ...

    @abstractmethod
    async def get_metric_history(
        self,
        ref: RunRef,
        metric: str,
        step_range: tuple[int, int] | None = None,
    ) -> MetricHistory: ...

    def list_sweeps(self, project: str) -> list[Sweep]:
        """Default: unsupported. Backends that declare SWEEPS override this.

        Kept synchronous and non-abstract per spec §3's exact code block —
        this asymmetry (every other method is `async def` and abstract,
        this one is plain `def` with a default body) is intentional in the
        frozen spec, not an oversight, so it's preserved exactly here.
        """
        raise NotSupportedError(self.name, BackendCapability.SWEEPS)
