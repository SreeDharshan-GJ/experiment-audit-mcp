"""Tests for experiment_audit_mcp.backends.fake_backend — Milestone 2.

Covers the roadmap's Milestone 2 completion criteria directly: FakeBackend
satisfies the ExperimentBackend ABC contract, a capability-less fake's
list_sweeps() raises NotSupportedError, and every adversarial scenario
named in design-spec-v1.md §7 (tiny sweep, partial data, NaN mid-curve,
correlated hyperparameters) can be produced on demand via seeding.
"""

from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.backends.base import (
    BackendCapability,
    ConnectionStatus,
    ExperimentBackend,
    NotSupportedError,
    RunFilter,
)
from experiment_audit_mcp.backends.fake_backend import (
    FakeBackend,
    MetricHistoryNotFoundError,
    RunNotFoundError,
)
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef, Sweep


def _make_run(run_id="run1", **overrides) -> Run:
    defaults = dict(
        ref=RunRef(backend="fake", entity="test-entity", project="mamfac", run_id=run_id),
        name=f"run-{run_id}",
        tags=["baseline"],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 12.5},
    )
    defaults.update(overrides)
    return Run(**defaults)


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


def test_fake_backend_is_an_experiment_backend():
    backend = FakeBackend()
    assert isinstance(backend, ExperimentBackend)


def test_fake_backend_default_capabilities_include_sweeps():
    backend = FakeBackend()
    assert BackendCapability.SWEEPS in backend.capabilities


@pytest.mark.asyncio
async def test_fake_backend_default_connection_status_is_authenticated():
    backend = FakeBackend()
    status = await backend.test_connection()
    assert status == ConnectionStatus(
        backend="fake", authenticated=True, scopes_detected=["read"]
    )


@pytest.mark.asyncio
async def test_fake_backend_connection_status_can_be_overridden():
    backend = FakeBackend()
    backend.set_connection_status(
        ConnectionStatus(backend="fake", authenticated=False, error="bad key")
    )
    status = await backend.test_connection()
    assert status.authenticated is False
    assert status.error == "bad key"


# ---------------------------------------------------------------------------
# list_sweeps / NotSupportedError — Milestone 2's core validation target
# ---------------------------------------------------------------------------


def test_list_sweeps_on_capability_less_fake_raises_not_supported_error():
    backend = FakeBackend(capabilities=set())
    with pytest.raises(NotSupportedError) as exc_info:
        backend.list_sweeps("mamfac")
    assert exc_info.value.backend_name == "fake"
    assert exc_info.value.capability is BackendCapability.SWEEPS


def test_list_sweeps_on_capable_fake_returns_seeded_sweeps():
    backend = FakeBackend(capabilities={BackendCapability.SWEEPS})
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="sweep-scope")
    sweep = Sweep(ref=ref, sweep_id="sweep-1", method="grid", run_refs=[])
    backend.seed_sweep(sweep)

    result = backend.list_sweeps("mamfac")
    assert result == [sweep]


def test_list_sweeps_filters_by_project():
    backend = FakeBackend()
    ref_a = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="scope-a")
    ref_b = RunRef(backend="fake", entity="test-entity", project="carm-plus-plus", run_id="scope-b")
    backend.seed_sweep(Sweep(ref=ref_a, sweep_id="s1", method="grid", run_refs=[]))
    backend.seed_sweep(Sweep(ref=ref_b, sweep_id="s2", method="grid", run_refs=[]))

    result = backend.list_sweeps("mamfac")
    assert [s.sweep_id for s in result] == ["s1"]


# ---------------------------------------------------------------------------
# list_runs / get_run_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_returns_seeded_runs_for_project():
    backend = FakeBackend()
    backend.seed_run(_make_run("run1"))
    backend.seed_run(_make_run("run2"))

    page = await backend.list_runs("mamfac")
    assert {r.ref.run_id for r in page.items} == {"run1", "run2"}
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_list_runs_accepts_page_size_without_enforcing_it():
    # Revision 2: page_size is part of the ABC signature FakeBackend must
    # satisfy, but FakeBackend deliberately doesn't implement pagination
    # (see the docstring in fake_backend.py) — passing it must not raise
    # or truncate results.
    backend = FakeBackend()
    backend.seed_run(_make_run("run1"))
    backend.seed_run(_make_run("run2"))

    page = await backend.list_runs("mamfac", page_size=1)
    assert {r.ref.run_id for r in page.items} == {"run1", "run2"}


@pytest.mark.asyncio
async def test_list_runs_excludes_other_projects():
    backend = FakeBackend()
    backend.seed_run(_make_run("run1", ref=RunRef("fake", "test-entity", "mamfac", "run1")))
    backend.seed_run(_make_run("run2", ref=RunRef("fake", "test-entity", "other-project", "run2")))

    page = await backend.list_runs("mamfac")
    assert [r.ref.run_id for r in page.items] == ["run1"]


@pytest.mark.asyncio
async def test_list_runs_filters_by_tags_and_status():
    backend = FakeBackend()
    backend.seed_run(_make_run("run1", tags=["ablation"], status="finished"))
    backend.seed_run(_make_run("run2", tags=["baseline"], status="crashed"))

    page = await backend.list_runs(
        "mamfac", filters=RunFilter(tags=["ablation"], status="finished")
    )
    assert [r.ref.run_id for r in page.items] == ["run1"]


@pytest.mark.asyncio
async def test_list_runs_filters_by_tags_excludes_non_matching():
    backend = FakeBackend()
    backend.seed_run(_make_run("run1", tags=["baseline"]))

    page = await backend.list_runs("mamfac", filters=RunFilter(tags=["ablation"]))
    assert page.items == []


@pytest.mark.asyncio
async def test_list_runs_filters_by_status_excludes_non_matching():
    backend = FakeBackend()
    backend.seed_run(_make_run("run1", status="finished"))
    backend.seed_run(_make_run("run2", status="crashed"))

    page = await backend.list_runs("mamfac", filters=RunFilter(status="finished"))
    assert [r.ref.run_id for r in page.items] == ["run1"]


@pytest.mark.asyncio
async def test_list_runs_filters_by_created_after_and_before():
    backend = FakeBackend()
    backend.seed_run(_make_run("early", created_at=datetime(2026, 1, 1, tzinfo=UTC)))
    backend.seed_run(_make_run("mid", created_at=datetime(2026, 6, 1, tzinfo=UTC)))
    backend.seed_run(_make_run("late", created_at=datetime(2026, 12, 1, tzinfo=UTC)))

    page = await backend.list_runs(
        "mamfac",
        filters=RunFilter(
            created_after=datetime(2026, 3, 1, tzinfo=UTC),
            created_before=datetime(2026, 9, 1, tzinfo=UTC),
        ),
    )
    assert [r.ref.run_id for r in page.items] == ["mid"]


@pytest.mark.asyncio
async def test_get_run_summary_returns_seeded_run():
    backend = FakeBackend()
    run = _make_run("run1")
    backend.seed_run(run)

    result = await backend.get_run_summary(run.ref)
    assert result is run


@pytest.mark.asyncio
async def test_get_run_summary_raises_run_not_found_for_unseeded_ref():
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="does-not-exist")

    with pytest.raises(RunNotFoundError):
        await backend.get_run_summary(ref)


# ---------------------------------------------------------------------------
# get_metric_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metric_history_returns_seeded_history():
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="run1")
    history = MetricHistory(
        ref=ref,
        metric_name="reward",
        points=[MetricPoint(step=0, value=1.0), MetricPoint(step=1, value=2.0)],
    )
    backend.seed_metric_history(history)

    result = await backend.get_metric_history(ref, "reward")
    assert result is history


@pytest.mark.asyncio
async def test_get_metric_history_raises_when_not_seeded():
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="run1")

    with pytest.raises(MetricHistoryNotFoundError):
        await backend.get_metric_history(ref, "reward")


@pytest.mark.asyncio
async def test_get_metric_history_applies_step_range():
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="run1")
    history = MetricHistory(
        ref=ref,
        metric_name="reward",
        points=[MetricPoint(step=s, value=float(s)) for s in range(10)],
    )
    backend.seed_metric_history(history)

    result = await backend.get_metric_history(ref, "reward", step_range=(3, 5))
    assert [p.step for p in result.points] == [3, 4, 5]


# ---------------------------------------------------------------------------
# Adversarial scenarios required by spec §7 / Milestone 2 completion criteria
# ---------------------------------------------------------------------------


def test_adversarial_tiny_sweep_representable():
    """A 3-run sweep — must trigger insufficient_samples logic in
    Milestone 8, but here we only need FakeBackend to represent it."""
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="sweep-scope")
    run_refs = [
        RunRef(backend="fake", entity="test-entity", project="mamfac", run_id=f"run{i}")
        for i in range(3)
    ]
    tiny_sweep = Sweep(
        ref=ref,
        sweep_id="tiny-sweep",
        method="random",
        run_refs=run_refs,
        target_metric="final_reward",
    )
    backend.seed_sweep(tiny_sweep)

    [result] = backend.list_sweeps("mamfac")
    assert result.sweep_id == "tiny-sweep"
    assert len(result.run_refs) == 3


@pytest.mark.asyncio
async def test_adversarial_partial_data_run_representable():
    """A run mid-ingestion — Run.data_completeness already models this
    (spec §2); FakeBackend needs no special-casing to seed it."""
    backend = FakeBackend()
    run = _make_run("partial-run", data_completeness="partial")
    backend.seed_run(run)

    result = await backend.get_run_summary(run.ref)
    assert result.data_completeness == "partial"


@pytest.mark.asyncio
async def test_adversarial_nan_mid_curve_representable():
    """A real crashed run with logged NaN values mid-curve — represented
    as a MetricPoint(value=None) that must survive the round trip
    unchanged (not skipped, not coerced)."""
    backend = FakeBackend()
    ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="crashed-run")
    history = MetricHistory(
        ref=ref,
        metric_name="loss",
        points=[
            MetricPoint(step=0, value=0.5),
            MetricPoint(step=1, value=0.4),
            MetricPoint(step=2, value=None),  # logged NaN
            MetricPoint(step=3, value=0.6),
        ],
    )
    backend.seed_metric_history(history)

    result = await backend.get_metric_history(ref, "loss")
    assert result.points[2].value is None
    assert result.points[2].step == 2


@pytest.mark.asyncio
async def test_adversarial_correlated_hyperparameters_sweep_representable():
    """Two hyperparameters (learning_rate, batch_size) that co-vary across
    every run in a grid sweep — the correlation itself is Milestone 8
    logic; here FakeBackend just needs to let the config values encode it."""
    backend = FakeBackend()
    sweep_ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id="sweep-scope")
    run_refs = []
    for i, (lr, batch_size) in enumerate([(0.1, 128), (0.01, 64), (0.001, 32)]):
        ref = RunRef(backend="fake", entity="test-entity", project="mamfac", run_id=f"grid-run-{i}")
        backend.seed_run(
            _make_run(
                f"grid-run-{i}",
                ref=ref,
                config={"learning_rate": lr, "batch_size": batch_size},
                summary_metrics={"final_reward": 10.0 + i},
            )
        )
        run_refs.append(ref)

    sweep = Sweep(
        ref=sweep_ref,
        sweep_id="correlated-sweep",
        method="grid",
        run_refs=run_refs,
        target_metric="final_reward",
    )
    backend.seed_sweep(sweep)

    runs = [await backend.get_run_summary(r) for r in run_refs]
    lrs = [r.config["learning_rate"] for r in runs]
    batch_sizes = [r.config["batch_size"] for r in runs]
    # Sanity check the fixture actually encodes correlated params — a
    # perfectly monotonic co-variation, same as a real learning_rate vs.
    # batch_size grid.
    assert sorted(lrs, reverse=True) == lrs
    assert sorted(batch_sizes, reverse=True) == batch_sizes

