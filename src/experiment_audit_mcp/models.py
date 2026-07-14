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

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

DataCompleteness = Literal["complete", "partial", "unknown"]
_VALID_DATA_COMPLETENESS = {"complete", "partial", "unknown"}

# -- JSON-safety sanitization (Audit #9 security/robustness finding) --------
#
# This module's own docstring promises that every model's `to_dict()`
# "produces a JSON-serializable structure". That promise was not actually
# enforced anywhere: `Run.config`/`Run.summary_metrics` and
# `MetricPoint.value` are populated from data a *backend* controls (a W&B
# run's logged config/summary/history), not data this codebase validates
# on the way in (see backends/wandb_backend.py's `_to_run`, which does
# `dict(wandb_run.config or {})` unmodified). Three concrete, reproducible
# failure modes follow from trusting that data to already be JSON-safe:
#
# 1. A real Python `float('nan')`/`float('inf')` (as opposed to the string
#    sentinels `"NaN"`/`"Infinity"` the wandb API is documented to emit)
#    reaching `json.dumps` with its default `allow_nan=True` produces the
#    bare tokens `NaN`/`Infinity` in the output — valid to Python's own
#    lenient decoder but not valid JSON per RFC 8259, so any
#    standards-compliant MCP client (most non-Python clients, strict
#    validators) fails to parse the response.
# 2. A hyperparameter logged as a numpy scalar (`numpy.float32`,
#    `numpy.int64` — extremely common in ML config dicts) or any other
#    object without native JSON support raises a bare `TypeError` out of
#    the serializer, after the tool has already returned — bypassing this
#    codebase's entire "structured errors, not bare exceptions cross the
#    MCP boundary" contract (server.py's module docstring) for exactly the
#    kind of malformed/adversarial backend response this contract exists
#    to guard against.
# 3. A pathologically large or deeply nested config/summary blob (a
#    crafted or corrupted run) costs unbounded memory/CPU to serialize and
#    transmit, with no limit anywhere in this codebase today.
#
# `_json_safe` is the single choke point that makes the docstring's promise
# actually true: every value written into a `to_dict()` output for
# externally-sourced data is routed through it first.

_MAX_SANITIZE_DEPTH = 20
_MAX_CONTAINER_ITEMS = 2000
_MAX_STRING_LENGTH = 20_000
_TRUNCATION_MARKER = "<truncated>"


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Recursively coerce `value` into a structure safe to pass to
    `json.dumps` — no NaN/Infinity floats, no non-JSON-native types, no
    unbounded depth/size. Never raises: anything it can't confidently
    represent is replaced with a `repr()`-based fallback (length-capped)
    rather than propagating an exception across the MCP boundary.
    """
    if _depth > _MAX_SANITIZE_DEPTH:
        return _TRUNCATION_MARKER

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            # Matches this codebase's own established convention (spec
            # §2): a non-finite numeric reading is represented as `None`,
            # the same way a logged NaN/null metric point already is —
            # never a raw token that isn't valid JSON.
            return None
        return value

    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            return value[:_MAX_STRING_LENGTH] + _TRUNCATION_MARKER
        return value

    if isinstance(value, dict):
        items = list(value.items())
        truncated = len(items) > _MAX_CONTAINER_ITEMS
        result: dict[str, Any] = {}
        for key, val in items[:_MAX_CONTAINER_ITEMS]:
            # JSON object keys must be strings; a malformed backend
            # response could carry any hashable as a dict key.
            safe_key = key if isinstance(key, str) else repr(key)
            result[safe_key] = _json_safe(val, _depth=_depth + 1)
        if truncated:
            result[f"_{_TRUNCATION_MARKER}_extra_keys"] = len(items) - _MAX_CONTAINER_ITEMS
        return result

    if isinstance(value, (list, tuple)):
        truncated = len(value) > _MAX_CONTAINER_ITEMS
        result_list = [_json_safe(v, _depth=_depth + 1) for v in value[:_MAX_CONTAINER_ITEMS]]
        if truncated:
            result_list.append(_TRUNCATION_MARKER)
        return result_list

    # Common escape hatch for numpy/pandas scalar types (float32, int64,
    # ...), which are not JSON-native but expose `.item()` to convert to
    # the equivalent native Python type. Guarded: `.item()` is only ever
    # called on something that isn't already one of the JSON-native types
    # handled above, and any failure falls through to the repr() fallback.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method(), _depth=_depth + 1)
        except Exception:  # noqa: BLE001 - defensive fallback, never raise
            pass

    # Last resort: never let an unrecognized type raise TypeError out of
    # json.dumps later. A capped repr() preserves some diagnostic value
    # without risking an unbounded string from a pathological __repr__.
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - even repr() can raise on a broken object
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > _MAX_STRING_LENGTH:
        text = text[:_MAX_STRING_LENGTH] + _TRUNCATION_MARKER
    return text


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
        # `config`/`summary_metrics` are backend-controlled data (a W&B
        # run's logged config/summary), not validated on the way in — see
        # `_json_safe`'s module-level docstring above for the concrete
        # malformed-response failure modes this guards against.
        return {
            "ref": _runref_to_dict(self.ref),
            "name": self.name,
            "tags": list(self.tags),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "config": _json_safe(dict(self.config)),
            "summary_metrics": _json_safe(dict(self.summary_metrics)),
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
        # `value` originates from a backend's metric history and, for the
        # real W&B backend, is expected to already be normalized to
        # finite-float-or-None (see wandb_backend._normalize_metric_value)
        # — but MetricPoint can also be constructed directly (any backend,
        # or a test/fixture), so this stays defensive rather than trusting
        # that normalization always ran. `_json_safe` is a no-op for an
        # already-finite float or None.
        return {"step": self.step, "value": _json_safe(self.value)}


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