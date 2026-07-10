"""Tests for analysis/comparison.py (Milestone 5).

Per the roadmap's Milestone 5 TDD approach: these tests use
hand-constructed `Run` objects, not recorded fixtures — the diffing
logic is pure and doesn't need real API shape to validate, so these
are fast, isolated tests exercised before (and independently of) the
MCP tool wiring in `server.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.analysis.comparison import (
    CompareRunsError,
    compare_runs,
)
from experiment_audit_mcp.models import Run, RunRef

_ENTITY = "test-entity"


def _make_run(run_id: str, project: str = "mamfac", **overrides) -> Run:
    defaults: dict = dict(
        ref=RunRef(backend="wandb", entity=_ENTITY, project=project, run_id=run_id),
        name=f"run-{run_id}",
        tags=[],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 10.0},
    )
    defaults.update(overrides)
    return Run(**defaults)


# ---------------------------------------------------------------------------
# Input-contract validation
# ---------------------------------------------------------------------------


def test_requires_at_least_two_runs():
    with pytest.raises(CompareRunsError, match="at least 2 runs"):
        compare_runs([_make_run("run1")])


def test_requires_at_least_two_runs_empty_list():
    with pytest.raises(CompareRunsError, match="at least 2 runs"):
        compare_runs([])


def test_rejects_duplicate_refs():
    run = _make_run("run1")
    with pytest.raises(CompareRunsError, match="duplicate RunRefs"):
        compare_runs([run, run])


# ---------------------------------------------------------------------------
# Config diff correctness (added / removed / changed params)
# ---------------------------------------------------------------------------


def test_config_diff_omits_identical_params():
    run_a = _make_run("a", config={"learning_rate": 0.001, "seed": 42})
    run_b = _make_run("b", config={"learning_rate": 0.001, "seed": 42})
    result = compare_runs([run_a, run_b])
    assert result.config_diff == []


def test_config_diff_flags_changed_param():
    run_a = _make_run("a", config={"learning_rate": 0.001})
    run_b = _make_run("b", config={"learning_rate": 0.01})
    result = compare_runs([run_a, run_b])
    assert len(result.config_diff) == 1
    entry = result.config_diff[0]
    assert entry.param == "learning_rate"
    assert entry.values[run_a.ref].value == 0.001
    assert entry.values[run_b.ref].value == 0.01
    assert entry.values[run_a.ref].present is True


def test_config_diff_flags_added_param():
    # run_b has a param run_a's config never had.
    run_a = _make_run("a", config={"seed": 42})
    run_b = _make_run("b", config={"seed": 42, "dropout": 0.1})
    result = compare_runs([run_a, run_b])
    params = {e.param for e in result.config_diff}
    assert params == {"dropout"}
    entry = result.config_diff[0]
    assert entry.values[run_a.ref].present is False
    assert entry.values[run_a.ref].value is None
    assert entry.values[run_b.ref].present is True
    assert entry.values[run_b.ref].value == 0.1


def test_config_diff_flags_removed_param():
    run_a = _make_run("a", config={"seed": 42, "dropout": 0.1})
    run_b = _make_run("b", config={"seed": 42})
    result = compare_runs([run_a, run_b])
    params = {e.param for e in result.config_diff}
    assert params == {"dropout"}


def test_config_diff_distinguishes_missing_from_explicit_none():
    # run_a explicitly logged flag=None; run_b never had "flag" at all.
    run_a = _make_run("a", config={"flag": None})
    run_b = _make_run("b", config={})
    result = compare_runs([run_a, run_b])
    entry = next(e for e in result.config_diff if e.param == "flag")
    assert entry.values[run_a.ref].present is True
    assert entry.values[run_a.ref].value is None
    assert entry.values[run_b.ref].present is False


# ---------------------------------------------------------------------------
# Metric diff correctness
# ---------------------------------------------------------------------------


def test_metric_diff_omits_identical_metrics():
    run_a = _make_run("a", summary_metrics={"final_reward": 10.0})
    run_b = _make_run("b", summary_metrics={"final_reward": 10.0})
    result = compare_runs([run_a, run_b])
    assert result.metric_diff == []


def test_metric_diff_computes_delta_for_two_runs():
    run_a = _make_run("a", summary_metrics={"final_reward": 10.0})
    run_b = _make_run("b", summary_metrics={"final_reward": 15.0})
    result = compare_runs([run_a, run_b])
    assert len(result.metric_diff) == 1
    entry = result.metric_diff[0]
    assert entry.metric == "final_reward"
    assert entry.values[run_a.ref] == 10.0
    assert entry.values[run_b.ref] == 15.0
    assert entry.delta == 5.0


def test_metric_diff_handles_metric_missing_on_one_run():
    run_a = _make_run("a", summary_metrics={"final_reward": 10.0})
    run_b = _make_run("b", summary_metrics={})
    result = compare_runs([run_a, run_b])
    entry = next(e for e in result.metric_diff if e.metric == "final_reward")
    assert entry.values[run_a.ref] == 10.0
    assert entry.values[run_b.ref] is None
    # A delta needs two present values to be meaningful.
    assert entry.delta is None


# ---------------------------------------------------------------------------
# N-way comparison (not just pairwise) — roadmap Milestone 5 requirement,
# since audit_ablation (Milestone 7) is a 2-run special case of this.
# ---------------------------------------------------------------------------


def test_nway_config_diff_across_three_runs():
    run_a = _make_run("a", config={"learning_rate": 0.001})
    run_b = _make_run("b", config={"learning_rate": 0.01})
    run_c = _make_run("c", config={"learning_rate": 0.1})
    result = compare_runs([run_a, run_b, run_c])
    entry = result.config_diff[0]
    assert entry.values[run_a.ref].value == 0.001
    assert entry.values[run_b.ref].value == 0.01
    assert entry.values[run_c.ref].value == 0.1


def test_nway_metric_diff_delta_is_max_minus_min_spread():
    run_a = _make_run("a", summary_metrics={"final_reward": 10.0})
    run_b = _make_run("b", summary_metrics={"final_reward": 15.0})
    run_c = _make_run("c", summary_metrics={"final_reward": 7.0})
    result = compare_runs([run_a, run_b, run_c])
    entry = result.metric_diff[0]
    assert entry.delta == 15.0 - 7.0


def test_nway_comparison_of_four_runs_with_mixed_diffs():
    runs = [
        _make_run("a", config={"seed": 1}, summary_metrics={"loss": 1.0}),
        _make_run("b", config={"seed": 2}, summary_metrics={"loss": 1.0}),
        _make_run("c", config={"seed": 3}, summary_metrics={"loss": 2.0}),
        _make_run("d", config={"seed": 4}, summary_metrics={"loss": 1.0}),
    ]
    result = compare_runs(runs)
    seed_entry = next(e for e in result.config_diff if e.param == "seed")
    assert {v.value for v in seed_entry.values.values()} == {1, 2, 3, 4}
    loss_entry = next(e for e in result.metric_diff if e.metric == "loss")
    assert loss_entry.delta == 1.0


# ---------------------------------------------------------------------------
# Cross-project comparison (spec §2: explicitly supported, project echoed)
# ---------------------------------------------------------------------------


def test_cross_project_comparison_echoes_each_runs_project():
    run_a = _make_run("a", project="mamfac", config={"learning_rate": 0.001})
    run_b = _make_run("b", project="carm-plus-plus", config={"learning_rate": 0.01})
    result = compare_runs([run_a, run_b])
    entry = result.config_diff[0]
    assert run_a.ref in entry.values
    assert run_b.ref in entry.values
    assert run_a.ref.project == "mamfac"
    assert run_b.ref.project == "carm-plus-plus"


# ---------------------------------------------------------------------------
# Serialization (to_dict) — JSON-safe keys, since RunRef itself can't be a
# JSON object key.
# ---------------------------------------------------------------------------


def test_to_dict_produces_json_safe_string_keys():
    run_a = _make_run("a", config={"learning_rate": 0.001})
    run_b = _make_run("b", config={"learning_rate": 0.01})
    result = compare_runs([run_a, run_b]).to_dict()

    entry = result["config_diff"][0]
    assert entry["param"] == "learning_rate"
    expected_key_a = f"wandb/{_ENTITY}/mamfac/a"
    expected_key_b = f"wandb/{_ENTITY}/mamfac/b"
    assert set(entry["values"].keys()) == {expected_key_a, expected_key_b}
    assert entry["values"][expected_key_a] == {"present": True, "value": 0.001}
    assert entry["values"][expected_key_b] == {"present": True, "value": 0.01}


def test_to_dict_metric_diff_shape():
    run_a = _make_run("a", summary_metrics={"final_reward": 10.0})
    run_b = _make_run("b", summary_metrics={"final_reward": 15.0})
    result = compare_runs([run_a, run_b]).to_dict()

    entry = result["metric_diff"][0]
    assert entry["metric"] == "final_reward"
    assert entry["delta"] == 5.0
    assert set(entry["values"].values()) == {10.0, 15.0}


def test_to_dict_returns_empty_lists_when_nothing_differs():
    run_a = _make_run("a")
    run_b = _make_run("b")
    result = compare_runs([run_a, run_b]).to_dict()
    assert result == {"config_diff": [], "metric_diff": []}
