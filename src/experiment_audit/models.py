"""Shared data models: RunRef, Run, MetricPoint, MetricHistory, Sweep, Page[T].

Implements the frozen contract from design-spec-v1.md §2. Every model that
crosses an MCP tool boundary provides a `to_dict()` method producing a
JSON-serializable structure, since MCP tool responses are JSON. Datetimes
are serialized as ISO-8601 strings; `None` values (representing logged
NaN/null metric points) are always preserved explicitly, never dropped or
coerced — this is spec-critical behavior, not an implementation detail
(see design-spec-v1.md §7, adversarial case: "a real crashed run with
logged NaN values mid-curve").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

DataCompleteness = Literal["complete", "partial", "unknown"]
_VALID_DATA_COMPLETENESS = {"complete", "partial", "unknown"}


@dataclass(frozen=True)
class RunRef:
    """Fully-scoped run identity.

    Never pass a bare run_id string between tools — run IDs are
    project-scoped in W&B, and cross-project ambiguity was a design flaw
    identified and fixed in the frozen spec (§2). Frozen and therefore
    hashable, so it can be used as a dict key (e.g. in compare_runs'
    per-run value maps) and passed safely across tool boundaries without
    risk of mutation.

    `entity` (Revision 1): W&B scopes every run as `entity/project/run_id`
    — `entity` is the team or user namespace a project lives under, and
    two different entities can each have a project named e.g. "mamfac".
    `project` alone is therefore not a sufficient scope key against the
    real API; omitting `entity` was the design flaw this revision fixes.
    Required (not optional) so a RunRef is always fully resolvable against
    the real W&B API without an implicit default. MLflow has no entity
    concept; MLflowBackend (v2) is expected to set this to a fixed
    placeholder (e.g. `"default"`) — a decision deferred to v2, not made
    here, since only the W&B backend exists in this codebase today.
    """

    backend: str
    entity: str
    project: str
    run_id: str


@dataclass
class Run:
    """A single experiment run's summary — config and final/latest metrics.

    Does NOT include metric history; that is fetched separately via
    get_metric_history / MetricHistory, per the explicit data-flow
    separation in spec §4.3 (avoids the old bolted-on `full_history: bool`
    flag problem flagged in design review).
    """

    ref: RunRef
    name: str
    tags: list[str]
    status: str
    created_at: datetime
    config: dict[str, Any]
    summary_metrics: dict[str, float]
    data_completeness: DataCompleteness = "unknown"

    def __post_init__(self) -> None:
        if self.data_completeness not in _VALID_DATA_COMPLETENESS:
            raise ValueError(
                f"data_completeness must be one of {_VALID_DATA_COMPLETENESS}, "
                f"got {self.data_completeness!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": _runref_to_dict(self.ref),
            "name": self.name,
            "tags": list(self.tags),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "config": dict(self.config),
            "summary_metrics": dict(self.summary_metrics),
            "data_completeness": self.data_completeness,
        }


@dataclass
class MetricPoint:
    """A single (step, value) sample from a metric history.

    `value` is `float | None` — None represents a logged NaN/null and
    must never be silently dropped during filtering, aggregation, or
    serialization (spec §2, §7).
    """

    step: int
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "value": self.value}


@dataclass
class MetricHistory:
    """The full recorded history of one metric for one run."""

    ref: RunRef
    metric_name: str
    points: list[MetricPoint]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": _runref_to_dict(self.ref),
            "metric_name": self.metric_name,
            "points": [p.to_dict() for p in self.points],
            "schema_version": self.schema_version,
        }


@dataclass
class Sweep:
    """A hyperparameter sweep, scoped to a backend/project via `ref`.

    Note: `ref.run_id` is not a real run — it's used here only to carry
    backend/project scoping consistently with every other model. The
    sweep's own identity is `sweep_id`.
    """

    ref: RunRef
    sweep_id: str
    method: str  # "grid" | "random" | "bayes" | "unsupported"
    run_refs: list[RunRef]
    target_metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": _runref_to_dict(self.ref),
            "sweep_id": self.sweep_id,
            "method": self.method,
            "run_refs": [_runref_to_dict(r) for r in self.run_refs],
            "target_metric": self.target_metric,
        }


T = TypeVar("T")


@dataclass
class Page(Generic[T]):
    """Generic cursor-paginated result wrapper.

    Every list-returning backend/tool method returns Page[T], per spec
    §3 — pagination is part of the interface from v1, not retrofitted
    later once a project has enough runs for it to matter.
    """

    items: list[T]
    next_cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [_serialize_item(item) for item in self.items],
            "next_cursor": self.next_cursor,
        }


def _serialize_item(item: Any) -> Any:
    if hasattr(item, "to_dict") and callable(item.to_dict):
        return item.to_dict()
    return item


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }
