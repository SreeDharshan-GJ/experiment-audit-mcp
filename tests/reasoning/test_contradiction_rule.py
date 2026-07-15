"""Tests for `ContradictionRule` (R003).

Focused on the rule's comparability boundary: two `EvidenceItem`s are
only ever compared when they share both their `EvidenceKind` and their
normalized `key` (evidence.py). This is a regression suite for a bug
where `_grouped_by_key` grouped only by `key`, so two unrelated facts
that happen to reuse the same key string in different evidence kinds
(e.g. a hardware bundle's `{"name": "A100"}` and a dataset bundle's
`{"name": "benchmark-a"}`) were reported as a contradiction even
though they were never the same fact to begin with.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from experiment_audit.models import Run, RunRef
from experiment_audit.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit.reasoning.evidence import Evidence
from experiment_audit.reasoning.hypotheses import HypothesisSet
from experiment_audit.reasoning.observations import ObservationSet
from experiment_audit.reasoning.rules import RuleContext
from experiment_audit.reasoning.scientific_rules.contradiction_rule import ContradictionRule

_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _run_ref(run_id: str) -> RunRef:
    return RunRef(backend="wandb", entity="test-team", project="proj", run_id=run_id)


def _run(ref: RunRef, **summary_metrics: float) -> Run:
    return Run(
        ref=ref,
        name=ref.run_id,
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config={},
        summary_metrics=summary_metrics,
    )


def _claim(claim_id: str, subject: str, evidence_trace: tuple[RunRef, ...], **scope: str) -> Claim:
    return Claim(
        id=claim_id,
        subject=subject,
        statement=f"Statement for {claim_id}.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(**scope),
        evidence_trace=evidence_trace,
    )


def _context(evidence: list[Evidence], claims: list[Claim]) -> RuleContext:
    return RuleContext(
        evidence=evidence,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet(claims),
    )


@pytest.fixture
def rule() -> ContradictionRule:
    return ContradictionRule()


def test_same_key_different_kind_is_not_a_contradiction(rule: ContradictionRule) -> None:
    """Regression test: a hardware `name` and a dataset `name` sharing a
    key string must never be reported as conflicting evidence, since
    they are two different facts, not two recorded values of one fact.
    """
    ref = _run_ref("run-001")
    ev = Evidence(
        ref=ref,
        run=_run(ref, accuracy=0.93),
        hardware={"name": "A100"},
        dataset={"name": "benchmark-a"},
    )
    claim = _claim("C001", "Model X accuracy", (ref,), dataset="benchmark-a", hardware="A100")

    result = rule.evaluate(_context([ev], [claim]))

    assert result.triggered is False
    assert result.contradictions == ()


def test_same_key_same_kind_within_claim_is_a_contradiction(rule: ContradictionRule) -> None:
    """Two metric items sharing a key *and* a kind, both attributed to
    the same claim, with different values -- a genuine "impossible
    simultaneous state" this rule must still catch.
    """
    ref = _run_ref("run-001")
    ev = Evidence(ref=ref, run=_run(ref, accuracy=0.93))
    # A second, conflicting metric item for the same run/claim, added
    # by hand so both values are attributed to the same claim.
    ev.add_item(
        type(ev.items[0])(
            kind=ev.items[0].kind,
            key="accuracy",
            value=0.50,
            source=ref,
        )
    )
    claim = _claim("C001", "Model X accuracy", (ref,))

    result = rule.evaluate(_context([ev], [claim]))

    assert result.triggered is True
    assert len(result.contradictions) == 1
    assert "metric.accuracy" in result.contradictions[0]
    assert "0.93" in result.contradictions[0]
    assert "0.5" in result.contradictions[0]


def test_cross_claim_same_kind_conflict_is_detected(rule: ContradictionRule) -> None:
    """The canonical Runtime = 42ms vs. Runtime = 87ms pattern: two
    claims, same subject, compatible scope, attributed evidence
    disagreeing on the same metric.
    """
    ref_a = _run_ref("run-001")
    ref_b = _run_ref("run-002")
    ev_a = Evidence(ref=ref_a, run=_run(ref_a, latency_ms=42.0))
    ev_b = Evidence(ref=ref_b, run=_run(ref_b, latency_ms=87.0))

    claim_a = _claim(
        "C004",
        "SparseAttn inference latency on Benchmark-B",
        (ref_a,),
        dataset="benchmark-b",
        hardware="A100",
    )
    claim_b = _claim(
        "C005",
        "SparseAttn inference latency on Benchmark-B",
        (ref_b,),
        dataset="benchmark-b",
        hardware="A100",
    )

    result = rule.evaluate(_context([ev_a, ev_b], [claim_a, claim_b]))

    assert result.triggered is True
    assert any(
        "metric.latency_ms" in c and "42.0" in c and "87.0" in c for c in result.contradictions
    )


def test_cross_claim_different_kind_same_key_is_not_a_contradiction(
    rule: ContradictionRule,
) -> None:
    """Two claims about the same subject and compatible scope, whose
    evidence happens to reuse a key string across different kinds
    (hardware.name vs. dataset.name), must not be reported as
    conflicting -- this is the cross-claim analogue of the
    within-claim regression above.
    """
    ref_a = _run_ref("run-001")
    ref_b = _run_ref("run-002")
    ev_a = Evidence(ref=ref_a, run=_run(ref_a, accuracy=0.9), hardware={"name": "A100"})
    ev_b = Evidence(ref=ref_b, run=_run(ref_b, accuracy=0.9), dataset={"name": "benchmark-a"})

    claim_a = _claim("C001", "Model X accuracy", (ref_a,), dataset="benchmark-a")
    claim_b = _claim("C002", "Model X accuracy", (ref_b,), dataset="benchmark-a")

    result = rule.evaluate(_context([ev_a, ev_b], [claim_a, claim_b]))

    assert result.triggered is False
    assert result.contradictions == ()


def test_diverging_scope_is_not_compared(rule: ContradictionRule) -> None:
    """Two claims about the same subject but explicitly different
    declared scope (CIFAR-10 vs. ImageNet) must never be reported as a
    contradiction, regardless of what their evidence records.
    """
    ref_a = _run_ref("run-001")
    ref_b = _run_ref("run-002")
    ev_a = Evidence(ref=ref_a, run=_run(ref_a, accuracy=0.9))
    ev_b = Evidence(ref=ref_b, run=_run(ref_b, accuracy=0.1))

    claim_a = _claim("C001", "Model X accuracy", (ref_a,), dataset="cifar-10")
    claim_b = _claim("C002", "Model X accuracy", (ref_b,), dataset="imagenet")

    result = rule.evaluate(_context([ev_a, ev_b], [claim_a, claim_b]))

    assert result.triggered is False
    assert result.contradictions == ()
