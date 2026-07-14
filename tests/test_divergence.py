"""Tests for analysis/divergence.py (Milestone 6).

Per the roadmap's Milestone 6 completion criteria: "Every adversarial curve
fixture from spec §7 produces the expected signal; a clean, well-behaved
curve produces no high-confidence signals (explicit false-positive test,
since a judgment tool crying wolf is as damaging as missing a real issue)."

These use hand-constructed `MetricHistory`/`MetricPoint` objects, not
recorded fixtures, following the same rationale test_comparison.py (Milestone
5) documents: this logic is pure and doesn't need real API shape to
validate — fast, isolated tests exercised independently of the MCP layer
(covered separately in test_server.py).
"""

from __future__ import annotations

import pytest

from experiment_audit.analysis.divergence import (
    SCHEMA_VERSION,
    audit_training_curve,
    infer_metric_type,
)
from experiment_audit.models import MetricHistory, MetricPoint, RunRef

_REF = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="run1")


def _history(metric_name: str, values: list[float | None]) -> MetricHistory:
    return MetricHistory(
        ref=_REF,
        metric_name=metric_name,
        points=[MetricPoint(step=i, value=v) for i, v in enumerate(values)],
    )


def _signal_names(result) -> set[str]:
    return {s.signal for s in result.signals}


# ---------------------------------------------------------------------------
# metric_type_assumed inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_name,expected",
    [
        ("train/loss", "loss"),
        ("val_loss", "loss"),
        ("episode_reward", "reward"),
        ("reward/mean", "reward"),
        ("gradient_norm", "unknown"),
    ],
)
def test_infer_metric_type(metric_name, expected):
    assert infer_metric_type(metric_name) == expected


def test_metric_type_can_be_overridden():
    history = _history("custom_metric", [1.0, 2.0, 3.0])
    result = audit_training_curve(history, metric_type="reward")
    assert result.metric_type_assumed == "reward"


def test_schema_version_is_2():
    history = _history("loss", [1.0, 2.0])
    result = audit_training_curve(history)
    assert result.schema_version == SCHEMA_VERSION == 2


# ---------------------------------------------------------------------------
# Clean curve — false-positive test (roadmap Milestone 6 completion criteria)
# ---------------------------------------------------------------------------


def test_clean_linear_curve_produces_no_signals():
    history = _history("loss", [10.0 - i for i in range(11)])
    result = audit_training_curve(history)
    assert result.signals == []


def test_clean_curve_to_dict_has_empty_signals_list():
    history = _history("loss", [10.0 - i for i in range(11)])
    result = audit_training_curve(history)
    assert result.to_dict()["signals"] == []


def test_empty_history_produces_no_signals():
    history = _history("loss", [])
    result = audit_training_curve(history)
    assert result.signals == []


# ---------------------------------------------------------------------------
# null_values — a real crashed run with logged NaN values mid-curve (spec §7)
# ---------------------------------------------------------------------------


def test_nan_mid_curve_produces_null_values_signal_with_correct_step_range():
    history = _history("loss", [1.0, 2.0, None, 4.0, 5.0])
    result = audit_training_curve(history)
    assert "null_values" in _signal_names(result)
    signal = next(s for s in result.signals if s.signal == "null_values")
    assert signal.step_range == (2, 2)
    assert signal.confidence == "high"
    assert signal.evidence["null_step_count"] == 1
    assert signal.evidence["null_steps"] == [2]


def test_multiple_nan_points_produce_step_range_spanning_all_of_them():
    history = _history("loss", [1.0, None, 3.0, None, 5.0, None])
    result = audit_training_curve(history)
    signal = next(s for s in result.signals if s.signal == "null_values")
    assert signal.step_range == (1, 5)
    assert signal.evidence["null_step_count"] == 3


def test_nan_gap_does_not_falsely_trigger_sudden_jump():
    # A NaN between two points widens the effective step gap; the
    # rate-of-change normalization (docs/audit-methods.md#training-curve)
    # exists specifically so this does not register as a sudden_jump.
    history = _history("loss", [1.0, 2.0, None, 4.0, 5.0])
    result = audit_training_curve(history)
    assert "sudden_jump" not in _signal_names(result)


def test_clean_curve_has_no_null_values_signal():
    history = _history("loss", [10.0 - i for i in range(11)])
    result = audit_training_curve(history)
    assert "null_values" not in _signal_names(result)


# ---------------------------------------------------------------------------
# sudden_jump — an abrupt level shift
# ---------------------------------------------------------------------------


def test_genuine_level_shift_triggers_sudden_jump():
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    history = _history("reward", values)
    result = audit_training_curve(history)
    assert "sudden_jump" in _signal_names(result)
    signal = next(s for s in result.signals if s.signal == "sudden_jump")
    assert signal.step_range == (4, 5)
    assert signal.evidence["value_before"] == 1.0
    assert signal.evidence["value_after"] == 50.0
    assert signal.confidence in {"high", "medium"}


def test_too_few_points_does_not_trigger_sudden_jump():
    history = _history("reward", [1.0, 100.0])
    result = audit_training_curve(history)
    assert "sudden_jump" not in _signal_names(result)


# ---------------------------------------------------------------------------
# low_variance_plateau — training has stalled
# ---------------------------------------------------------------------------


def test_flat_plateau_triggers_low_variance_plateau_signal():
    declining = [10.0 - i for i in range(5)]
    plateau = [5.0 + 0.001 * ((-1) ** j) for j in range(7)]
    declining_again = [5.0 - 1.0 - k for k in range(4)]
    values = declining + plateau + declining_again
    history = _history("loss", values)
    result = audit_training_curve(history)
    assert "low_variance_plateau" in _signal_names(result)
    signal = next(s for s in result.signals if s.signal == "low_variance_plateau")
    assert signal.step_range[0] >= 4  # plateau starts around index 5, tolerate window edge
    assert signal.evidence["run_length"] >= 5


def test_clean_monotonic_curve_has_no_plateau_signal():
    history = _history("loss", [10.0 - i for i in range(11)])
    result = audit_training_curve(history)
    assert "low_variance_plateau" not in _signal_names(result)


def test_short_history_does_not_trigger_plateau():
    history = _history("loss", [5.0, 5.0, 5.0])
    result = audit_training_curve(history)
    assert "low_variance_plateau" not in _signal_names(result)


# ---------------------------------------------------------------------------
# high_frequency_oscillation — jagged up/down pattern
# ---------------------------------------------------------------------------


def test_oscillating_curve_triggers_oscillation_signal():
    values = [5.0 if i % 2 == 0 else 15.0 for i in range(10)]
    history = _history("reward", values)
    result = audit_training_curve(history)
    assert "high_frequency_oscillation" in _signal_names(result)
    signal = next(s for s in result.signals if s.signal == "high_frequency_oscillation")
    assert signal.evidence["sign_flip_ratio"] == 1.0
    assert signal.confidence == "high"


def test_clean_curve_has_no_oscillation_signal():
    history = _history("loss", [10.0 - i for i in range(11)])
    result = audit_training_curve(history)
    assert "high_frequency_oscillation" not in _signal_names(result)


def test_too_few_points_does_not_trigger_oscillation():
    history = _history("reward", [5.0, 15.0, 5.0, 15.0])
    result = audit_training_curve(history)
    assert "high_frequency_oscillation" not in _signal_names(result)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_to_dict_serializes_step_range_as_list_not_tuple():
    history = _history("loss", [1.0, 2.0, None, 4.0, 5.0])
    result = audit_training_curve(history)
    payload = result.to_dict()
    null_entry = next(s for s in payload["signals"] if s["signal"] == "null_values")
    assert isinstance(null_entry["step_range"], list)
    assert null_entry["step_range"] == [2, 2]


def test_to_dict_includes_method_pointer_to_docs():
    history = _history("loss", [1.0, 2.0])
    result = audit_training_curve(history)
    assert "docs/audit-methods.md#training-curve" in result.to_dict()["method"]
