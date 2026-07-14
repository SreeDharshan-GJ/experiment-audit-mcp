"""Consolidated adversarial fixture set — one builder per spec §7 case.

**Milestone 9 deliverable.** design-spec-v1.md §7 ("Testing Strategy")
names six adversarial/edge cases under point 2. Each one already has
dedicated coverage at the analysis-function level (test_sensitivity.py,
test_confound.py, test_divergence.py) and, scattered across
test_server.py's Milestone 4-8 additions, at the MCP-tool level too. What
was missing was a single place where a reviewer can see all six cases
lined up against the spec's own numbering and confirm each is exercised
*through the MCP protocol layer specifically* — not just re-derive that
fact by reading five different test files. That's this module's job: it
is a fixture module, not a test file, so it has no assertions of its own
and no pytest collection concerns; `tests/test_adversarial_mcp_layer.py`
imports `ADVERSARIAL_CASES` and does the asserting.

This does not replace the existing analysis-level or scattered MCP-level
tests — deleting Milestone 4-8-approved test coverage is out of scope for
a milestone whose own roadmap entry says "resolved by editing tool
descriptions... not renaming tools or changing schemas." It is additive:
one canonical, spec-numbered index into behavior that was already
correct, so the correspondence between the spec's list and the
implementation is explicit rather than implicit.

Every builder returns a seeded `FakeBackend` plus the exact MCP tool name
and JSON-shaped arguments a real MCP client would send — the same
`fastmcp.Client(mcp)` in-memory protocol round trip test_server.py
already established in Milestone 4, reused here rather than reinvented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experiment_audit_mcp.backends.fake_backend import FakeBackend
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef, Sweep

_ENTITY = "test-entity"
_PROJECT = "mamfac"
_BACKEND_NAME = "fake"


def _ref(run_id: str, project: str = _PROJECT) -> RunRef:
    return RunRef(backend=_BACKEND_NAME, entity=_ENTITY, project=project, run_id=run_id)


def _run(run_id: str, **overrides: Any) -> Run:
    defaults: dict[str, Any] = dict(
        ref=_ref(run_id),
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


def _ref_input(ref: RunRef) -> dict[str, str]:
    """The `RunRefInput` JSON shape a real MCP client sends (server.py)."""
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }


@dataclass(frozen=True)
class AdversarialCase:
    """One spec §7 case: how to seed it, which tool exercises it, and how
    to recognize a correct MCP-layer response.

    `spec_ref` is the exact bullet under design-spec-v1.md §7 point 2,
    quoted verbatim so drift between this module and the spec text is
    visible on read, not just asserted in code.
    """

    name: str
    spec_ref: str
    build_backend: Callable[[], FakeBackend]
    tool_name: str
    tool_args: Callable[[], dict[str, Any]]
    assert_result: Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Case 1 — "A sweep with 3 runs -> must trigger insufficient_samples, not
#           a ranking."
# ---------------------------------------------------------------------------


def _build_tiny_sweep_backend() -> FakeBackend:
    backend = FakeBackend()
    runs = [_run(f"r{i}", config={"lr": float(i), "seed": 42}) for i in range(1, 4)]
    for run in runs:
        backend.seed_run(run)
    backend.seed_sweep(
        Sweep(
            ref=_ref("sweep-ref"),
            sweep_id="sweep-tiny",
            method="grid",
            run_refs=[r.ref for r in runs],
            target_metric="final_reward",
        )
    )
    return backend


def _assert_tiny_sweep(result: dict[str, Any]) -> None:
    assert "error" in result, result
    assert result["error"]["error_type"] == "insufficient_samples"
    assert result["run_count"] == 3
    assert result["minimum_required"] == 10


CASE_TINY_SWEEP = AdversarialCase(
    name="sweep_too_small",
    spec_ref="A sweep with 3 runs -> must trigger insufficient_samples, not a ranking.",
    build_backend=_build_tiny_sweep_backend,
    tool_name="audit_sweep",
    tool_args=lambda: {
        "sweep_ref": {
            "backend": _BACKEND_NAME,
            "entity": _ENTITY,
            "project": _PROJECT,
            "sweep_id": "sweep-tiny",
        }
    },
    assert_result=_assert_tiny_sweep,
)


# ---------------------------------------------------------------------------
# Case 2 — "An ablation pair where exactly one parameter differs and it's
#           a random seed -> must be flagged likely_intentional: true via
#           the allowlist, verdict should lean clean, not confounded."
# ---------------------------------------------------------------------------


def _build_seed_only_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.seed_run(_run("baseline", config={"learning_rate": 0.001, "seed": 1}))
    backend.seed_run(_run("ablation", config={"learning_rate": 0.001, "seed": 2}))
    return backend


def _assert_seed_only(result: dict[str, Any]) -> None:
    assert "error" not in result, result
    assert result["verdict"] == "clean"
    seed_entry = next(p for p in result["differing_params"] if p["param"] == "seed")
    assert seed_entry["likely_intentional"] is True


CASE_SEED_ONLY_DIFF = AdversarialCase(
    name="ablation_seed_only_diff",
    spec_ref=(
        "An ablation pair where exactly one parameter differs and it's a "
        "random seed -> must be flagged likely_intentional: true via the "
        "allowlist, verdict should lean clean, not confounded."
    ),
    build_backend=_build_seed_only_backend,
    tool_name="audit_ablation",
    tool_args=lambda: {
        "baseline": _ref_input(_ref("baseline")),
        "ablation": _ref_input(_ref("ablation")),
        "claimed_variable": "learning_rate",
    },
    assert_result=_assert_seed_only,
)


# ---------------------------------------------------------------------------
# Case 3 — "An ablation pair with two differing params, neither on the
#           allowlist -> must return confounded."
# ---------------------------------------------------------------------------


def _build_two_nonallowlisted_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.seed_run(
        _run("baseline", config={"learning_rate": 0.001, "batch_size": 32, "seed": 42})
    )
    backend.seed_run(_run("ablation", config={"learning_rate": 0.01, "batch_size": 64, "seed": 42}))
    return backend


def _assert_two_nonallowlisted(result: dict[str, Any]) -> None:
    assert "error" not in result, result
    assert result["verdict"] == "confounded"
    by_param = {p["param"]: p["likely_intentional"] for p in result["differing_params"]}
    assert by_param["batch_size"] is False


CASE_TWO_NONALLOWLISTED_DIFFS = AdversarialCase(
    name="ablation_two_nonallowlisted_diffs",
    spec_ref=(
        "An ablation pair with two differing params, neither on the "
        "allowlist -> must return confounded."
    ),
    build_backend=_build_two_nonallowlisted_backend,
    tool_name="audit_ablation",
    tool_args=lambda: {
        "baseline": _ref_input(_ref("baseline")),
        "ablation": _ref_input(_ref("ablation")),
        "claimed_variable": "learning_rate",
    },
    assert_result=_assert_two_nonallowlisted,
)


# ---------------------------------------------------------------------------
# Case 4 — "A real crashed run with logged NaN values mid-curve ->
#           audit_training_curve must surface a null_values signal, not
#           silently skip the points."
# ---------------------------------------------------------------------------


def _build_crashed_run_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.seed_run(_run("crashed", status="crashed"))
    values: list[float | None] = [1.0, 0.9, 0.8, None, None, 0.7, 0.6]
    backend.seed_metric_history(
        MetricHistory(
            ref=_ref("crashed"),
            metric_name="train/loss",
            points=[MetricPoint(step=i, value=v) for i, v in enumerate(values)],
        )
    )
    return backend


def _assert_crashed_run(result: dict[str, Any]) -> None:
    assert "error" not in result, result
    signal_names = {s["signal"] for s in result["signals"]}
    assert "null_values" in signal_names


CASE_CRASHED_RUN_NAN = AdversarialCase(
    name="crashed_run_with_nan",
    spec_ref=(
        "A real crashed run with logged NaN values mid-curve -> "
        "audit_training_curve must surface a null_values signal, not "
        "silently skip the points."
    ),
    build_backend=_build_crashed_run_backend,
    tool_name="audit_training_curve",
    tool_args=lambda: {"ref": _ref_input(_ref("crashed")), "metric": "train/loss"},
    assert_result=_assert_crashed_run,
)


# ---------------------------------------------------------------------------
# Case 5 — "A run mid-ingestion (data_completeness: 'partial') -> any
#           audit tool touching it must downgrade confidence and explain
#           why."
# ---------------------------------------------------------------------------


def _build_partial_data_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.seed_run(
        _run(
            "baseline",
            config={"learning_rate": 0.001, "seed": 42},
            data_completeness="partial",
        )
    )
    backend.seed_run(_run("ablation", config={"learning_rate": 0.01, "seed": 42}))
    return backend


def _assert_partial_data(result: dict[str, Any]) -> None:
    assert "error" not in result, result
    assert result["confidence"] == "low"
    assert "partial" in result["method"].lower()


CASE_PARTIAL_DATA_RUN = AdversarialCase(
    name="partial_data_run",
    spec_ref=(
        'A run mid-ingestion (data_completeness: "partial") -> any audit '
        "tool touching it must downgrade confidence and explain why."
    ),
    build_backend=_build_partial_data_backend,
    tool_name="audit_ablation",
    tool_args=lambda: {
        "baseline": _ref_input(_ref("baseline")),
        "ablation": _ref_input(_ref("ablation")),
        "claimed_variable": "learning_rate",
    },
    assert_result=_assert_partial_data,
)


# ---------------------------------------------------------------------------
# Case 6 — "Two correlated hyperparameters in a grid sweep ->
#           audit_sweep must surface the co-variance warning, not present
#           both as independently important."
# ---------------------------------------------------------------------------


def _build_correlated_hparams_backend() -> FakeBackend:
    backend = FakeBackend()
    runs = [
        _run(
            f"r{i}",
            config={"learning_rate": float(i), "batch_size": float(i) * 10, "seed": 42},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    for run in runs:
        backend.seed_run(run)
    backend.seed_sweep(
        Sweep(
            ref=_ref("sweep-ref"),
            sweep_id="sweep-correlated",
            method="grid",
            run_refs=[r.ref for r in runs],
            target_metric="reward",
        )
    )
    return backend


def _assert_correlated_hparams(result: dict[str, Any]) -> None:
    assert "error" not in result, result
    by_param = {p["param"]: p for p in result["parameter_importance"]}
    assert by_param["learning_rate"]["warning"] is not None
    assert "batch_size" in by_param["learning_rate"]["warning"]
    assert by_param["batch_size"]["warning"] is not None
    assert "learning_rate" in by_param["batch_size"]["warning"]


CASE_CORRELATED_HYPERPARAMS = AdversarialCase(
    name="sweep_correlated_hyperparams",
    spec_ref=(
        "Two correlated hyperparameters in a grid sweep -> audit_sweep "
        "must surface the co-variance warning, not present both as "
        "independently important."
    ),
    build_backend=_build_correlated_hparams_backend,
    tool_name="audit_sweep",
    tool_args=lambda: {
        "sweep_ref": {
            "backend": _BACKEND_NAME,
            "entity": _ENTITY,
            "project": _PROJECT,
            "sweep_id": "sweep-correlated",
        }
    },
    assert_result=_assert_correlated_hparams,
)


# ---------------------------------------------------------------------------
# The consolidated index, in spec §7 point 2's own bullet order.
# ---------------------------------------------------------------------------

ADVERSARIAL_CASES: list[AdversarialCase] = [
    CASE_TINY_SWEEP,
    CASE_SEED_ONLY_DIFF,
    CASE_TWO_NONALLOWLISTED_DIFFS,
    CASE_CRASHED_RUN_NAN,
    CASE_PARTIAL_DATA_RUN,
    CASE_CORRELATED_HYPERPARAMS,
]
