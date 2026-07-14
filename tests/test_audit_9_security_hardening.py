"""Audit #9 — security / robustness / production-hardening regression tests.

Scope: this audit reviews security, robustness, defensive programming, and
production hardening only (backend correctness, the MCP protocol layer,
the scientific reasoning engine, reporting, packaging, documentation, and
performance/scalability already passed dedicated audits and are not
re-litigated here).

**Finding (High): `to_dict()` did not actually guarantee JSON-safe output.**

models.py's own module docstring states: "Every model that crosses an MCP
tool boundary provides a `to_dict()` method producing a JSON-serializable
structure." That promise was not enforced. `Run.config`/`Run
.summary_metrics` and `MetricPoint.value` are populated from data a
*backend* controls (a W&B run's logged config/summary/history) and were
passed through to `to_dict()`'s output completely unsanitized
(`backends/wandb_backend.py`'s `_to_run`: `dict(wandb_run.config or {})`).

Reproducible failure modes this allowed, all confirmed against the
pre-fix code:

1. A real Python `float('nan')`/`float('inf')` value anywhere in a run's
   config, summary metrics, or metric-history points serialized to the
   bare tokens `NaN`/`Infinity` via `json.dumps`'s default
   `allow_nan=True` — valid to Python's own decoder, but **not valid
   JSON** per RFC 8259, breaking any standards-compliant (i.e.
   non-Python) MCP client that tries to parse the response.
2. A hyperparameter logged as a numpy scalar (`numpy.float32`,
   `numpy.int64` — routine in ML config dicts) or any other
   non-JSON-native object raised a bare `TypeError` *after* the tool had
   already returned, downstream of every `try/except` in server.py —
   directly violating this codebase's own stated "structured errors, not
   bare exceptions cross the MCP boundary" contract (server.py's module
   docstring) for precisely the malformed/adversarial-backend-response
   case that contract exists to cover.
3. Nothing bounded the size/depth of config/summary blobs relayed to a
   client — a large or deeply nested run could blow up serialized
   response size or nesting depth with no limit anywhere in this
   codebase.

Fixed by routing all backend-sourced values through `models._json_safe`
(a recursive, depth- and size-bounded, never-raising sanitizer) at the
single point every model's `to_dict()` already funnels through, so the
module docstring's promise is now actually true.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.backends.wandb_backend import (
    _normalize_metric_value,
    _to_summary_metrics,
)
from experiment_audit_mcp.models import (
    MetricHistory,
    MetricPoint,
    Run,
    RunRef,
    _json_safe,
)

# ---------------------------------------------------------------------------
# _json_safe — unit-level behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_json_safe_maps_non_finite_floats_to_none(bad):
    """NaN/Infinity/-Infinity are not valid JSON tokens (RFC 8259) — must
    become `None`, consistent with this codebase's existing convention
    that `None` represents a non-finite/logged-NaN numeric reading."""
    assert _json_safe(bad) is None


def test_json_safe_preserves_finite_numbers_and_none_and_bool():
    assert _json_safe(3.14) == 3.14
    assert _json_safe(42) == 42
    assert _json_safe(None) is None
    assert _json_safe(True) is True
    assert _json_safe(False) is False


def test_json_safe_recurses_into_nested_containers():
    value = {"a": [1, float("nan"), {"b": float("inf")}], "c": (1, 2, float("-inf"))}
    safe = _json_safe(value)
    assert safe == {"a": [1, None, {"b": None}], "c": [1, 2, None]}
    # Must be actually JSON-round-trippable, not just "no exception".
    assert json.loads(json.dumps(safe)) == safe


def test_json_safe_coerces_non_string_dict_keys():
    safe = _json_safe({1: "int-key", (1, 2): "tuple-key"})
    assert safe == {"1": "int-key", "(1, 2)": "tuple-key"}
    json.dumps(safe)  # must not raise


def test_json_safe_never_raises_on_arbitrary_unrecognized_object():
    class Unserializable:
        def __repr__(self) -> str:
            return "<Unserializable object>"

    safe = _json_safe(Unserializable())
    assert isinstance(safe, str)
    json.dumps(safe)  # must not raise


def test_json_safe_handles_numpy_scalar_like_objects_via_item():
    """Simulates numpy.float32/int64 — extremely common in logged ML
    hyperparameters — without importing numpy as a test dependency."""

    class NumpyFloat32Like:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return float(self._value)

        def __repr__(self) -> str:  # pragma: no cover - fallback path only
            return f"np.float32({self._value})"

    safe = _json_safe(NumpyFloat32Like(0.001))
    assert safe == 0.001


def test_json_safe_caps_recursion_depth_without_raising():
    """A maliciously/pathologically deep structure must not blow the
    Python recursion limit (a real DoS vector against a long-running MCP
    server process shared by many callers)."""
    deep: dict = {}
    cursor = deep
    for _ in range(500):
        cursor["x"] = {}
        cursor = cursor["x"]
    safe = _json_safe(deep)  # must not raise RecursionError
    json.dumps(safe)  # must still be valid, boundedly-sized JSON


def test_json_safe_caps_container_size_without_raising():
    """A pathologically large container must not produce unbounded
    memory/JSON growth in a single tool response."""
    huge = {f"key_{i}": i for i in range(50_000)}
    safe = _json_safe(huge)
    serialized = json.dumps(safe)
    assert len(safe) < 50_000  # truncated, not passed through whole
    assert len(serialized) < 500_000


def test_json_safe_caps_string_length_without_raising():
    huge_string = "x" * 1_000_000
    safe = _json_safe(huge_string)
    assert len(safe) < 1_000_000


# ---------------------------------------------------------------------------
# Run.to_dict() / MetricPoint.to_dict() — integration through the actual
# MCP-boundary serialization path, with adversarial backend-shaped data.
# ---------------------------------------------------------------------------


def _make_ref() -> RunRef:
    return RunRef(backend="wandb", entity="e", project="p", run_id="r1")


def test_run_to_dict_with_real_nan_in_summary_metrics_produces_valid_json():
    """A backend (or a bug upstream of the documented string-sentinel
    contract) that hands back a real float('nan') summary value must not
    corrupt the wire format."""
    run = Run(
        ref=_make_ref(),
        name="run",
        tags=[],
        status="finished",
        created_at=datetime.now(UTC),
        config={},
        summary_metrics={"loss": float("nan"), "acc": float("inf")},
    )
    serialized = json.dumps(run.to_dict())
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    reparsed = json.loads(serialized)
    assert reparsed["summary_metrics"]["loss"] is None
    assert reparsed["summary_metrics"]["acc"] is None


def test_run_to_dict_with_non_json_native_config_values_does_not_raise():
    """A malformed/adversarial run config containing objects with no
    native JSON representation (numpy scalars, arbitrary objects, non-str
    keys) must not raise a bare exception out of to_dict() — that would
    cross the MCP boundary as an unstructured failure, defeating this
    codebase's entire ToolError-translation contract (server.py)."""

    class NumpyFloat32Like:
        def item(self) -> float:
            return 0.001

    class Unserializable:
        def __repr__(self) -> str:
            return "<Unserializable object>"

    run = Run(
        ref=_make_ref(),
        name="run",
        tags=[],
        status="finished",
        created_at=datetime.now(UTC),
        config={
            "lr": NumpyFloat32Like(),
            "weird": Unserializable(),
            "nested_bad_key": {1: "x"},
        },
        summary_metrics={},
    )
    serialized = json.dumps(run.to_dict())  # must not raise TypeError
    reparsed = json.loads(serialized)
    assert reparsed["config"]["lr"] == 0.001


def test_metric_point_to_dict_with_real_nan_value_produces_valid_json():
    point = MetricPoint(step=1, value=float("nan"))
    history = MetricHistory(ref=_make_ref(), metric_name="loss", points=[point])
    serialized = json.dumps(history.to_dict())
    assert "NaN" not in serialized
    assert json.loads(serialized)["points"][0]["value"] is None


# ---------------------------------------------------------------------------
# wandb_backend normalization — defense in depth at the source, not just
# at the final serialization choke point.
# ---------------------------------------------------------------------------


def test_normalize_metric_value_maps_real_nan_and_infinity_to_none():
    """Previously only the documented string sentinels ("NaN", "Infinity")
    were normalized to None; a raw numeric NaN/Infinity passed straight
    through as a non-finite float, in violation of this function's own
    `float | None` (never non-finite) return-type contract."""
    assert _normalize_metric_value(float("nan")) is None
    assert _normalize_metric_value(float("inf")) is None
    assert _normalize_metric_value(float("-inf")) is None
    # Existing string-sentinel and finite-value behavior must be unchanged.
    assert _normalize_metric_value("NaN") is None
    assert _normalize_metric_value("Infinity") is None
    assert _normalize_metric_value(3.14) == 3.14
    assert _normalize_metric_value(None) is None


def test_to_summary_metrics_drops_real_nan_and_infinity_values():
    result = _to_summary_metrics({"loss": float("nan"), "acc": 0.95, "bad": float("inf")})
    assert result == {"acc": 0.95}
    assert not any(math.isnan(v) or math.isinf(v) for v in result.values())