"""Tests for analysis/sensitivity.py (Milestone 8).

Per the roadmap's Milestone 8 completion criteria: "The insufficient-
samples refusal is un-bypassable through any parameter combination
(explicit test attempting to sneak a small sweep past the floor via
edge-case inputs); co-variance warning fires correctly on the specific
adversarial fixture from spec §7." Spec §7's three sweep-specific
adversarial cases are each covered below:

    1. a sweep with 3 runs -> insufficient_samples, never a ranking
    2. two correlated hyperparameters in a grid sweep -> audit_sweep
       surfaces the co-variance warning, not two independently
       important parameters
    3. a well-powered, independent sweep -> a sensible ranking with
       caveat and n populated correctly

Plus the statistical-validity edge cases identified in the module's own
review (see analysis/sensitivity.py's module docstring): non-numeric
parameters, constant parameters/target metric, missing target_metric,
the per-parameter overlap floor, and the Fisher z-transform confidence
derivation.

These use hand-constructed `Run`/`Sweep` objects, not recorded fixtures,
for the same reason test_comparison.py/test_divergence.py/
test_confound.py do: this logic is pure and doesn't need real API shape
to validate. MCP-layer coverage (the `audit_sweep` tool itself) lives in
test_server.py, per that file's existing Milestone 5/6/7 pattern.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from experiment_audit.analysis.sensitivity import (
    DEFAULT_MINIMUM_SAMPLES,
    InsufficientSamplesError,
    SweepAuditError,
    audit_sweep,
)
from experiment_audit.models import Run, RunRef, Sweep

_ENTITY = "test-entity"
_PROJECT = "mamfac"


def _make_run(run_id: str, **overrides) -> Run:
    defaults: dict = dict(
        ref=RunRef(backend="wandb", entity=_ENTITY, project=_PROJECT, run_id=run_id),
        name=f"run-{run_id}",
        tags=[],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={},
        summary_metrics={},
        data_completeness="complete",
    )
    defaults.update(overrides)
    return Run(**defaults)


def _make_sweep(runs: list[Run], target_metric: str | None = "reward") -> Sweep:
    return Sweep(
        ref=RunRef(backend="wandb", entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
        sweep_id="sweep-1",
        method="grid",
        run_refs=[r.ref for r in runs],
        target_metric=target_metric,
    )


def _linear_sweep(n: int, noise: float = 0.0) -> tuple[Sweep, list[Run]]:
    """n runs where reward = 10 * lr (perfectly linear, modulo `noise`),
    seed held constant (uninformative), so `lr` should rank #1 and
    `seed` should be excluded as constant."""
    runs = [
        _make_run(
            f"r{i}",
            config={"lr": float(i), "seed": 42},
            summary_metrics={"reward": 10.0 * i + noise * ((-1) ** i)},
        )
        for i in range(1, n + 1)
    ]
    return _make_sweep(runs), runs


# ---------------------------------------------------------------------------
# spec §7 case 1: sweep too small -> insufficient_samples, never a ranking
# ---------------------------------------------------------------------------


def test_three_run_sweep_raises_insufficient_samples_on_raw_count():
    sweep, runs = _linear_sweep(3)
    with pytest.raises(InsufficientSamplesError) as excinfo:
        audit_sweep(sweep, runs)
    assert excinfo.value.run_count == 3
    assert excinfo.value.minimum_required == DEFAULT_MINIMUM_SAMPLES


def test_insufficient_samples_refusal_happens_before_any_ranking_logic():
    """Un-bypassable: even sweep runs with wildly different, highly
    informative configs still refuse below the floor -- there is no
    parameter combination that produces a ranking from 3 runs."""
    runs = [
        _make_run("r1", config={"lr": 0.001}, summary_metrics={"reward": 1.0}),
        _make_run("r2", config={"lr": 0.01}, summary_metrics={"reward": 10.0}),
        _make_run("r3", config={"lr": 0.1}, summary_metrics={"reward": 100.0}),
    ]
    sweep = _make_sweep(runs)
    with pytest.raises(InsufficientSamplesError):
        audit_sweep(sweep, runs)


def test_insufficient_samples_on_usable_run_count_when_metric_missing():
    """A nominally large-enough sweep where most runs never logged the
    target metric must still refuse -- the *usable* count is what
    matters, per the module's extension of the sample-size floor."""
    runs = [
        _make_run(f"r{i}", config={"lr": float(i)}, summary_metrics={"reward": float(i)})
        for i in range(1, 4)
    ] + [
        _make_run(f"r{i}", config={"lr": float(i)}, summary_metrics={})  # never logged reward
        for i in range(4, 13)
    ]
    sweep = _make_sweep(runs)
    with pytest.raises(InsufficientSamplesError) as excinfo:
        audit_sweep(sweep, runs)
    assert excinfo.value.run_count == 3  # only 3 runs actually usable
    assert excinfo.value.minimum_required == DEFAULT_MINIMUM_SAMPLES


def test_minimum_samples_is_configurable():
    sweep, runs = _linear_sweep(5)
    result = audit_sweep(sweep, runs, minimum_samples=5)
    assert result.sweep_size == 5
    assert result.parameter_importance  # ranking was actually computed


# ---------------------------------------------------------------------------
# spec §7 case 2: co-varying hyperparameters -> warning, not independent
# importance
# ---------------------------------------------------------------------------


def test_covarying_hyperparameters_are_flagged_on_both_sides():
    """learning_rate and batch_size move in lockstep (a non-orthogonal
    grid) -- both should carry a co-variance warning naming the other."""
    runs = [
        _make_run(
            f"r{i}",
            config={"learning_rate": float(i), "batch_size": float(i) * 10, "seed": 42},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    by_param = {p.param: p for p in result.parameter_importance}
    assert by_param["learning_rate"].warning is not None
    assert "batch_size" in by_param["learning_rate"].warning
    assert by_param["batch_size"].warning is not None
    assert "learning_rate" in by_param["batch_size"].warning


def test_independent_hyperparameters_are_not_flagged():
    runs = [
        _make_run(
            f"r{i}",
            config={"lr": float(i % 4), "batch_size": float((i * 7) % 5), "seed": 42},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    by_param = {p.param: p for p in result.parameter_importance}
    # lr and batch_size are constructed to not move together; neither
    # should carry a co-variance warning against the other.
    if "lr" in by_param and by_param["lr"].warning is not None:
        assert "batch_size" not in by_param["lr"].warning


# ---------------------------------------------------------------------------
# spec §7 case 3: well-powered, independent sweep -> sensible ranking with
# caveat and n populated correctly
# ---------------------------------------------------------------------------


def test_well_powered_sweep_ranks_the_informative_parameter_first():
    sweep, runs = _linear_sweep(12)

    result = audit_sweep(sweep, runs)

    assert result.sweep_size == 12
    assert result.usable_run_count == 12
    assert result.target_metric == "reward"
    assert "n=12 of 12" in result.caveat
    top = result.parameter_importance[0]
    assert top.param == "lr"
    assert top.rank == 1
    assert top.correlation == pytest.approx(1.0, abs=1e-6)
    assert result.confidence == "high"


def test_constant_parameter_is_excluded_not_ranked():
    sweep, runs = _linear_sweep(12)  # seed is constant across all runs

    result = audit_sweep(sweep, runs)

    ranked_params = {p.param for p in result.parameter_importance}
    assert "seed" not in ranked_params
    excluded = {e.param: e.reason for e in result.excluded_parameters}
    assert excluded["seed"] == "constant"


def test_rank_ties_use_competition_ranking():
    """Two parameters with identical |correlation| should share a rank,
    and the next distinct rank should skip accordingly (1, 1, 3 -- not
    1, 1, 2)."""
    runs = [
        _make_run(
            f"r{i}",
            config={"a": float(i), "b": float(i), "c": float(i % 3)},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    by_param = {p.param: p for p in result.parameter_importance}
    assert by_param["a"].rank == by_param["b"].rank == 1
    assert by_param["c"].rank == 3


# ---------------------------------------------------------------------------
# Non-numeric / categorical parameters (module docstring point 3)
# ---------------------------------------------------------------------------


def test_categorical_parameter_is_excluded_as_non_numeric():
    runs = [
        _make_run(
            f"r{i}",
            config={"lr": float(i), "optimizer": "adam" if i % 2 == 0 else "sgd"},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    ranked_params = {p.param for p in result.parameter_importance}
    assert "optimizer" not in ranked_params
    excluded = {e.param: e.reason for e in result.excluded_parameters}
    assert excluded["optimizer"] == "non_numeric"
    assert "lr" in ranked_params


def test_boolean_parameter_is_treated_as_numeric_point_biserial():
    runs = [
        _make_run(
            f"r{i}",
            config={"use_batchnorm": i % 2 == 0, "seed": 42},
            summary_metrics={"reward": 10.0 if i % 2 == 0 else 1.0},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    ranked_params = {p.param for p in result.parameter_importance}
    assert "use_batchnorm" in ranked_params


def test_parameter_present_on_too_few_runs_is_insufficient_overlap():
    runs = [
        _make_run(f"r{i}", config={"lr": float(i)}, summary_metrics={"reward": float(i)})
        for i in range(1, 11)
    ]
    # a conditional parameter only present on 2 of the 10 runs
    runs[0].config["dropout"] = 0.1
    runs[1].config["dropout"] = 0.2
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    excluded = {e.param: e.reason for e in result.excluded_parameters}
    assert excluded["dropout"] == "insufficient_overlap"


# ---------------------------------------------------------------------------
# target_metric resolution and "nothing rankable" cases (module docstring
# points 5 and 7)
# ---------------------------------------------------------------------------


def test_target_metric_falls_back_to_sweep_target_metric():
    sweep, runs = _linear_sweep(12)
    result = audit_sweep(sweep, runs, target_metric=None)
    assert result.target_metric == "reward"


def test_explicit_target_metric_overrides_sweep_target_metric():
    runs = [
        _make_run(
            f"r{i}",
            config={"lr": float(i)},
            summary_metrics={"reward": float(i), "loss": float(20 - i)},
        )
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs, target_metric="reward")

    result = audit_sweep(sweep, runs, target_metric="loss")

    assert result.target_metric == "loss"
    assert result.parameter_importance[0].correlation < 0  # lr and loss move oppositely


def test_missing_target_metric_raises_sweep_audit_error():
    sweep, runs = _linear_sweep(12)
    sweep_without_metric = Sweep(
        ref=sweep.ref,
        sweep_id=sweep.sweep_id,
        method=sweep.method,
        run_refs=sweep.run_refs,
        target_metric=None,
    )
    with pytest.raises(SweepAuditError):
        audit_sweep(sweep_without_metric, runs, target_metric=None)


def test_constant_target_metric_returns_empty_ranking_not_an_error():
    runs = [
        _make_run(f"r{i}", config={"lr": float(i)}, summary_metrics={"reward": 5.0})
        for i in range(1, 13)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    assert result.parameter_importance == []
    assert result.confidence == "low"
    excluded = {e.param: e.reason for e in result.excluded_parameters}
    assert excluded["lr"] == "target_metric_constant"
    assert "did not vary" in result.caveat


def test_missing_run_in_runs_list_is_excluded_from_sweep_size_accounting():
    sweep, runs = _linear_sweep(13)
    # simulate a run whose get_run_summary failed upstream and was never
    # appended to `runs` by the caller
    partial_runs = runs[:12]

    result = audit_sweep(sweep, partial_runs)

    assert result.sweep_size == 13  # sweep still claims 13 run_refs
    assert result.usable_run_count == 12  # but only 12 were actually usable


# ---------------------------------------------------------------------------
# Confidence derivation (module docstring point 6: Fisher z-transform)
# ---------------------------------------------------------------------------


def test_weak_correlation_from_noisy_data_yields_low_confidence():
    rng = random.Random(0)
    runs = [
        _make_run(
            f"r{i}",
            config={"lr": rng.uniform(0.0, 1.0), "seed": 42},
            summary_metrics={"reward": 5.0 + rng.gauss(0, 1)},
        )
        for i in range(1, 21)
    ]
    sweep = _make_sweep(runs)

    result = audit_sweep(sweep, runs)

    assert result.confidence == "low"
    assert result.parameter_importance[0].p_value is not None
    assert result.parameter_importance[0].p_value > 0.05


def test_perfect_correlation_yields_high_confidence():
    sweep, runs = _linear_sweep(12)
    result = audit_sweep(sweep, runs)
    assert result.confidence == "high"
    assert result.parameter_importance[0].p_value < 0.01


# ---------------------------------------------------------------------------
# Output shape / schema
# ---------------------------------------------------------------------------


def test_to_dict_matches_spec_shape_plus_additive_fields():
    sweep, runs = _linear_sweep(12)
    result = audit_sweep(sweep, runs)
    payload = result.to_dict()

    assert payload["schema_version"] == 1
    assert payload["sweep_size"] == 12
    assert payload["usable_run_count"] == 12
    assert isinstance(payload["parameter_importance"], list)
    assert payload["parameter_importance"][0]["param"] == "lr"
    assert "rank" in payload["parameter_importance"][0]
    assert "correlation" in payload["parameter_importance"][0]
    assert isinstance(payload["excluded_parameters"], list)
    assert payload["confidence"] in {"high", "medium", "low"}
    assert "Pearson" in payload["method"]
    assert isinstance(payload["caveat"], str) and payload["caveat"]


def test_every_ranked_parameter_carries_its_own_evidence():
    """Design principle #2: judgment tools show their work. Every
    parameter_importance entry must carry enough to audit the claim."""
    sweep, runs = _linear_sweep(12)
    result = audit_sweep(sweep, runs)
    for p in result.parameter_importance:
        d = p.to_dict()
        assert "correlation" in d
        assert "p_value" in d
        assert "rank" in d
