"""Tests for analysis/confound.py (Milestone 7).

Per the roadmap's Milestone 7 completion criteria: "All four named
adversarial ablation cases from the spec pass exactly as specified —
this tool's correctness on these specific cases matters more than broad
coverage." Those four cases (spec §7) are each covered below:

    1. seed-only difference -> clean / likely_intentional: true
    2. two non-allowlisted differing params -> confounded
    3. single claimed-variable-only difference -> clean, high confidence
    4. partial-data run involved -> confidence downgraded

These use hand-constructed `Run` objects, not recorded fixtures, for the
same reason test_comparison.py (Milestone 5) and test_divergence.py
(Milestone 6) do: this logic is pure and doesn't need real API shape to
validate. MCP-layer coverage (the `audit_ablation` tool itself) lives in
test_server.py, per that file's existing Milestone 5/6 pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.analysis.comparison import CompareRunsError
from experiment_audit_mcp.analysis.confound import (
    ALLOWLIST_PARAMS,
    SCHEMA_VERSION,
    audit_ablation,
)
from experiment_audit_mcp.models import Run, RunRef

_ENTITY = "test-entity"
_PROJECT = "mamfac"


def _make_run(run_id: str, **overrides) -> Run:
    defaults: dict = dict(
        ref=RunRef(backend="wandb", entity=_ENTITY, project=_PROJECT, run_id=run_id),
        name=f"run-{run_id}",
        tags=[],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.001, "seed": 42},
        summary_metrics={"final_reward": 10.0},
        data_completeness="complete",
    )
    defaults.update(overrides)
    return Run(**defaults)


# ---------------------------------------------------------------------------
# Named adversarial case 1: seed-only difference
# ---------------------------------------------------------------------------


def test_seed_only_difference_is_clean_and_likely_intentional():
    baseline = _make_run("baseline", config={"learning_rate": 0.001, "seed": 1})
    ablation = _make_run("ablation", config={"learning_rate": 0.001, "seed": 2})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"
    seed_entry = next(p for p in result.differing_params if p.param == "seed")
    assert seed_entry.likely_intentional is True


@pytest.mark.parametrize("field", sorted(ALLOWLIST_PARAMS))
def test_every_allowlisted_field_is_treated_as_intentional(field):
    baseline = _make_run("baseline", config={"learning_rate": 0.001, field: "a"})
    ablation = _make_run("ablation", config={"learning_rate": 0.001, field: "b"})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"
    entry = next(p for p in result.differing_params if p.param == field)
    assert entry.likely_intentional is True


def test_allowlist_is_case_insensitive():
    baseline = _make_run("baseline", config={"learning_rate": 0.001, "Seed": 1})
    ablation = _make_run("ablation", config={"learning_rate": 0.001, "Seed": 2})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"


def test_allowlist_does_not_fuzzy_match_substrings():
    # "device_batch_size" is NOT "device" - a real confound must not be
    # silently waved through just because its name contains an
    # allowlisted substring (module docstring rationale).
    baseline = _make_run("baseline", config={"learning_rate": 0.001, "device_batch_size": 8})
    ablation = _make_run("ablation", config={"learning_rate": 0.001, "device_batch_size": 16})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "confounded"
    entry = next(p for p in result.differing_params if p.param == "device_batch_size")
    assert entry.likely_intentional is False


# ---------------------------------------------------------------------------
# Named adversarial case 2: two non-allowlisted differing params
# ---------------------------------------------------------------------------


def test_two_nonallowlisted_differing_params_is_confounded():
    baseline = _make_run(
        "baseline", config={"learning_rate": 0.001, "batch_size": 32, "dropout": 0.1}
    )
    ablation = _make_run(
        "ablation", config={"learning_rate": 0.01, "batch_size": 64, "dropout": 0.1}
    )

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "confounded"
    params = {p.param: p.likely_intentional for p in result.differing_params}
    assert params == {"learning_rate": True, "batch_size": False}


def test_confounded_verdict_still_reports_all_differing_params():
    baseline = _make_run("baseline", config={"a": 1, "b": 1, "seed": 1})
    ablation = _make_run("ablation", config={"a": 2, "b": 2, "seed": 2})

    result = audit_ablation(baseline, ablation, claimed_variable="a")

    assert result.verdict == "confounded"
    assert len(result.differing_params) == 3
    by_param = {p.param: p for p in result.differing_params}
    assert by_param["a"].likely_intentional is True
    assert by_param["b"].likely_intentional is False
    assert by_param["seed"].likely_intentional is True


# ---------------------------------------------------------------------------
# Named adversarial case 3: single claimed-variable-only difference
# ---------------------------------------------------------------------------


def test_single_claimed_variable_only_difference_is_clean_high_confidence():
    baseline = _make_run("baseline", config={"learning_rate": 0.001, "seed": 42})
    ablation = _make_run("ablation", config={"learning_rate": 0.01, "seed": 42})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"
    assert result.confidence == "high"
    assert len(result.differing_params) == 1
    entry = result.differing_params[0]
    assert entry.param == "learning_rate"
    assert entry.baseline_value == 0.001
    assert entry.ablation_value == 0.01
    assert entry.likely_intentional is True


# ---------------------------------------------------------------------------
# Named adversarial case 4: partial-data run involved
# ---------------------------------------------------------------------------


def test_partial_data_on_baseline_downgrades_confidence_to_low():
    baseline = _make_run("baseline", config={"learning_rate": 0.001}, data_completeness="partial")
    ablation = _make_run("ablation", config={"learning_rate": 0.01}, data_completeness="complete")

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"  # verdict itself is unaffected by partial data
    assert result.confidence == "low"
    assert "partial data" in result.method


def test_partial_data_on_ablation_downgrades_confidence_to_low():
    baseline = _make_run("baseline", config={"learning_rate": 0.001}, data_completeness="complete")
    ablation = _make_run(
        "ablation", config={"learning_rate": 0.01, "batch_size": 64}, data_completeness="partial"
    )

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "confounded"  # partial data downgrades confidence, not verdict
    assert result.confidence == "low"
    assert "partial data" in result.method


def test_unknown_data_completeness_does_not_trigger_partial_downgrade():
    # "unknown" (the Run default) is a distinct state from a confirmed
    # "partial" ingestion - spec §5 flags "partial" specifically, so an
    # unknown-completeness run should not be penalized as if it were
    # known to be partial.
    baseline = _make_run("baseline", config={"learning_rate": 0.001}, data_completeness="unknown")
    ablation = _make_run("ablation", config={"learning_rate": 0.01}, data_completeness="unknown")

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.confidence == "high"


# ---------------------------------------------------------------------------
# "uncertain" verdict: no config differences at all
# ---------------------------------------------------------------------------


def test_identical_configs_is_uncertain_not_clean():
    baseline = _make_run("baseline", config={"learning_rate": 0.001, "seed": 42})
    ablation = _make_run("ablation", config={"learning_rate": 0.001, "seed": 42})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "uncertain"
    assert result.confidence == "low"
    assert result.differing_params == []


def test_uncertain_verdict_overridden_by_partial_data_message_but_stays_low_confidence():
    baseline = _make_run("baseline", config={"learning_rate": 0.001}, data_completeness="partial")
    ablation = _make_run("ablation", config={"learning_rate": 0.001})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "uncertain"
    assert result.confidence == "low"
    assert "partial data" in result.method


# ---------------------------------------------------------------------------
# Evidence, method, and schema shape
# ---------------------------------------------------------------------------


def test_evidence_is_full_compare_runs_style_diff_including_metrics():
    baseline = _make_run(
        "baseline",
        config={"learning_rate": 0.001},
        summary_metrics={"final_reward": 10.0},
    )
    ablation = _make_run(
        "ablation",
        config={"learning_rate": 0.01},
        summary_metrics={"final_reward": 15.0},
    )

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert "config_diff" in result.evidence
    assert "metric_diff" in result.evidence
    assert result.evidence["metric_diff"][0]["metric"] == "final_reward"
    assert result.evidence["metric_diff"][0]["delta"] == 5.0


def test_method_references_audit_methods_doc():
    baseline = _make_run("baseline", config={"learning_rate": 0.001})
    ablation = _make_run("ablation", config={"learning_rate": 0.01})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert "docs/audit-methods.md#ablation" in result.method


def test_schema_version_is_1():
    baseline = _make_run("baseline", config={"learning_rate": 0.001})
    ablation = _make_run("ablation", config={"learning_rate": 0.01})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.schema_version == SCHEMA_VERSION == 1


def test_to_dict_round_trips_expected_keys():
    baseline = _make_run("baseline", config={"learning_rate": 0.001})
    ablation = _make_run("ablation", config={"learning_rate": 0.01})

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate").to_dict()

    assert set(result.keys()) == {
        "schema_version",
        "verdict",
        "confidence",
        "differing_params",
        "method",
        "evidence",
    }
    assert result["differing_params"][0] == {
        "param": "learning_rate",
        "baseline_value": 0.001,
        "ablation_value": 0.01,
        "likely_intentional": True,
    }


# ---------------------------------------------------------------------------
# Input-contract validation (reused from compare_runs, not duplicated)
# ---------------------------------------------------------------------------


def test_same_ref_for_baseline_and_ablation_raises_compare_runs_error():
    run = _make_run("only-run", config={"learning_rate": 0.001})
    with pytest.raises(CompareRunsError, match="duplicate RunRefs"):
        audit_ablation(run, run, claimed_variable="learning_rate")


# ---------------------------------------------------------------------------
# Cross-project ablation pairs (compare_runs already supports this; confirm
# audit_ablation doesn't add an unwarranted same-project restriction)
# ---------------------------------------------------------------------------


def test_cross_project_ablation_pair_is_supported():
    baseline = _make_run("baseline", config={"learning_rate": 0.001})
    ablation = Run(
        ref=RunRef(backend="wandb", entity=_ENTITY, project="carm-plus-plus", run_id="ablation"),
        name="run-ablation",
        tags=[],
        status="finished",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        config={"learning_rate": 0.01},
        summary_metrics={},
        data_completeness="complete",
    )

    result = audit_ablation(baseline, ablation, claimed_variable="learning_rate")

    assert result.verdict == "clean"
