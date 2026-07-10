"""Tests for experiment_audit_mcp.models — Milestone 1.

Written before the implementation (TDD), per the roadmap's note that the
serialization round-trip tests, including None-preservation for logged
NaN values, should drive the design of the to-dict helpers rather than
just verify them after the fact.

Covers, per design-spec-v1.md §2 and the roadmap's Milestone 1 deliverables:
- construction of every model
- RunRef frozen/hashable behavior
- serialization round-trip, including None survival (spec §2, §7)
- data_completeness default behavior
"""

from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.models import (
    MetricHistory,
    MetricPoint,
    Page,
    Run,
    RunRef,
    Sweep,
)

# ---------------------------------------------------------------------------
# RunRef
# ---------------------------------------------------------------------------


def test_runref_construction():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    assert ref.backend == "wandb"
    assert ref.project == "mamfac"
    assert ref.run_id == "abc123"


def test_runref_is_frozen():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    with pytest.raises(Exception):
        ref.run_id = "different"  # type: ignore[misc]


def test_runref_is_hashable_and_usable_as_dict_key():
    ref_a = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    ref_b = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    ref_c = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="different")

    # Equal refs hash the same and are usable as dict keys — required
    # because compare_runs' output keys diffs by RunRef (spec §2, §4.2).
    d = {ref_a: "first"}
    d[ref_b] = "second"
    assert len(d) == 1
    assert d[ref_a] == "second"

    d[ref_c] = "third"
    assert len(d) == 2


def test_runref_equality_requires_all_fields_to_match():
    ref_a = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    ref_b = RunRef(backend="wandb", entity="test-entity", project="other_project", run_id="abc123")
    assert ref_a != ref_b


def test_runref_distinguishes_same_project_different_entity():
    """Revision 1 regression test.

    Two different W&B entities (teams/users) can each own a project with
    the same name (e.g. "mamfac"). Before this revision, RunRef had no
    `entity` field, so these two runs were indistinguishable and would
    silently collide as the same dict key / same identity. This is the
    exact design flaw the entity field fixes.
    """
    ref_a = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")
    ref_b = RunRef(backend="wandb", entity="collaborator-lab", project="mamfac", run_id="abc123")
    assert ref_a != ref_b
    assert hash(ref_a) != hash(ref_b)
    d = {ref_a: "mine", ref_b: "theirs"}
    assert len(d) == 2


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _make_run(**overrides) -> Run:
    defaults = dict(
        ref=RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123"),
        name="baseline-run",
        tags=["baseline"],
        status="finished",
        created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 12.5},
    )
    defaults.update(overrides)
    return Run(**defaults)


def test_run_construction():
    run = _make_run()
    assert run.name == "baseline-run"
    assert run.config["learning_rate"] == 0.001


def test_run_data_completeness_defaults_to_unknown_when_not_specified():
    # A backend that hasn't explicitly checked completeness must not
    # silently claim "complete" — that would be exactly the kind of
    # false confidence audit tools must avoid (spec §5).
    run = _make_run()
    assert run.data_completeness == "unknown"


def test_run_data_completeness_can_be_explicitly_set():
    run = _make_run(data_completeness="partial")
    assert run.data_completeness == "partial"


def test_run_data_completeness_rejects_invalid_value():
    with pytest.raises(ValueError):
        _make_run(data_completeness="totally_fine_trust_me")


# ---------------------------------------------------------------------------
# MetricPoint / MetricHistory — None (NaN) preservation
# ---------------------------------------------------------------------------


def test_metric_point_accepts_none_value():
    # Represents a logged NaN/null in the source data. Must never be
    # silently dropped or coerced to 0.0 — spec §2, adversarial case in §7.
    point = MetricPoint(step=10, value=None)
    assert point.value is None
    assert point.step == 10


def test_metric_history_construction():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    history = MetricHistory(
        ref=ref,
        metric_name="reward",
        points=[
            MetricPoint(step=0, value=1.0),
            MetricPoint(step=1, value=None),  # a logged NaN mid-curve
            MetricPoint(step=2, value=3.5),
        ],
    )
    assert history.schema_version == 1
    assert history.points[1].value is None


def test_metric_history_serialization_round_trip_preserves_none():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="abc123")
    history = MetricHistory(
        ref=ref,
        metric_name="reward",
        points=[
            MetricPoint(step=0, value=1.0),
            MetricPoint(step=1, value=None),
            MetricPoint(step=2, value=3.5),
        ],
    )

    serialized = history.to_dict()

    # The None at step=1 must survive serialization explicitly as null,
    # not be dropped from the list and not be coerced to 0.0/NaN-as-float.
    assert len(serialized["points"]) == 3
    assert serialized["points"][1]["value"] is None
    assert serialized["points"][1]["step"] == 1
    assert serialized["points"][0]["value"] == 1.0
    assert serialized["ref"] == {
        "backend": "wandb",
        "entity": "test-entity",
        "project": "mamfac",
        "run_id": "abc123",
    }
    assert serialized["metric_name"] == "reward"
    assert serialized["schema_version"] == 1


def test_run_serialization_round_trip_datetime_is_iso_string():
    run = _make_run()
    serialized = run.to_dict()
    assert serialized["created_at"] == "2026-06-01T12:00:00+00:00"
    assert serialized["ref"]["run_id"] == "abc123"
    assert serialized["data_completeness"] == "unknown"
    assert serialized["tags"] == ["baseline"]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def test_sweep_construction():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="sweep-scope")
    run_refs = [
        RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="run1"),
        RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="run2"),
    ]
    sweep = Sweep(
        ref=ref,
        sweep_id="sweep-001",
        method="bayes",
        run_refs=run_refs,
        target_metric="final_reward",
    )
    assert sweep.sweep_id == "sweep-001"
    assert len(sweep.run_refs) == 2
    assert sweep.target_metric == "final_reward"


def test_sweep_target_metric_is_optional():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="sweep-scope")
    sweep = Sweep(
        ref=ref, sweep_id="sweep-001", method="grid", run_refs=[], target_metric=None
    )
    assert sweep.target_metric is None


def test_sweep_serialization_round_trip():
    ref = RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="sweep-scope")
    run_refs = [RunRef(backend="wandb", entity="test-entity", project="mamfac", run_id="run1")]
    sweep = Sweep(
        ref=ref,
        sweep_id="sweep-001",
        method="bayes",
        run_refs=run_refs,
        target_metric="final_reward",
    )

    serialized = sweep.to_dict()

    assert serialized["sweep_id"] == "sweep-001"
    assert serialized["method"] == "bayes"
    assert serialized["target_metric"] == "final_reward"
    assert serialized["ref"] == {
        "backend": "wandb",
        "entity": "test-entity",
        "project": "mamfac",
        "run_id": "sweep-scope",
    }
    assert serialized["run_refs"] == [
        {"backend": "wandb", "entity": "test-entity", "project": "mamfac", "run_id": "run1"}
    ]


# ---------------------------------------------------------------------------
# Page[T]
# ---------------------------------------------------------------------------


def test_page_construction_and_default_no_next_cursor():
    run = _make_run()
    page = Page(items=[run], next_cursor=None)
    assert page.items == [run]
    assert page.next_cursor is None


def test_page_to_dict_serializes_dataclass_items():
    run = _make_run()
    page = Page(items=[run], next_cursor="cursor-abc")

    serialized = page.to_dict()

    assert serialized["next_cursor"] == "cursor-abc"
    assert len(serialized["items"]) == 1
    assert serialized["items"][0]["ref"]["run_id"] == "abc123"


def test_page_to_dict_handles_empty_items():
    page: Page[Run] = Page(items=[], next_cursor=None)
    serialized = page.to_dict()
    assert serialized["items"] == []
    assert serialized["next_cursor"] is None


def test_page_to_dict_passes_through_plain_non_dataclass_items():
    # Page[T] is generic — not every T is guaranteed to have a to_dict
    # method (e.g. a page of plain strings, such as project names).
    # The fallback path must pass such items through unchanged rather
    # than erroring.
    page: Page[str] = Page(items=["project-a", "project-b"], next_cursor=None)
    serialized = page.to_dict()
    assert serialized["items"] == ["project-a", "project-b"]
