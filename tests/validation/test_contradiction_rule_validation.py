"""Scientific validation of `ContradictionRule` (R003).

R003 detects genuine, structural contradictions -- two `EvidenceItem`s
that share the same `EvidenceKind` *and* normalized `key` but record
different values -- within one claim's own attributed evidence, across
two claims sharing a subject and a compatible scope, and carries
forward any already-known, unresolved `Contradiction` naming a claim in
scope. Per Chapter 8 §2, a scope difference is never a contradiction,
and per specification §8, an item sharing only a key string across two
different `EvidenceKind`s is never comparable at all.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionResolution,
    ContradictionResolutionKind,
    ContradictionStatus,
)
from experiment_audit_mcp.reasoning.scientific_rules.contradiction_rule import ContradictionRule

from .builders import claim, context, evidence, run_ref, scope


@pytest.fixture
def rule() -> ContradictionRule:
    return ContradictionRule()


def test_genuine_metric_contradiction_across_claims(rule: ContradictionRule) -> None:
    """Two claims about the same subject, under a matched, compatible
    scope (same hardware, same dataset), whose attributed evidence
    disagrees on the exact same metric -- 42ms vs. 87ms latency for
    "SparseAttn inference latency on an A100" -- cannot both be an
    accurate account of the same quantity. This is a genuine
    scientific contradiction, not a scope difference.
    """
    ref_a = run_ref("run-201")
    ref_b = run_ref("run-202")
    ev_a = evidence(
        ref_a, hardware={"name": "A100"}, dataset={"name": "benchmark-b"}, latency_ms=42.0
    )
    ev_b = evidence(
        ref_b, hardware={"name": "A100"}, dataset={"name": "benchmark-b"}, latency_ms=87.0
    )

    claim_a = claim(
        "C201",
        "SparseAttn inference latency on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref_a,),
        claim_scope=scope(hardware="A100", dataset="benchmark-b"),
    )
    claim_b = claim(
        "C202",
        "SparseAttn inference latency on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref_b,),
        claim_scope=scope(hardware="A100", dataset="benchmark-b"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev_a, ev_b], claims=[claim_a, claim_b]))

    assert result.triggered is True
    assert any("42.0" in c and "87.0" in c for c in result.contradictions)
    assert {c.id for c in result.affected_claims} == {"C201", "C202"}


def test_same_key_different_evidence_kind_is_not_a_contradiction(rule: ContradictionRule) -> None:
    """A hardware bundle's `name` ("A100") and a dataset bundle's
    `name` ("benchmark-a") share the key string `"name"`, but they are
    two unrelated facts, not two recorded values of one fact -- two
    different `EvidenceKind`s are never comparable, regardless of key
    overlap.
    """
    ref = run_ref("run-203")
    ev = evidence(ref, hardware={"name": "A100"}, dataset={"name": "benchmark-a"}, accuracy=0.93)
    c = claim(
        "C203",
        "Model X accuracy",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(dataset="benchmark-a", hardware="A100"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is False
    assert result.contradictions == ()


def test_duplicate_evidence_is_not_a_contradiction(rule: ContradictionRule) -> None:
    """The same fact, recorded twice with the *same* value (e.g. a
    metric re-logged identically), is a duplicate, not a conflict --
    `_values_conflict` only fires on genuinely different values, so
    two identical evidence items sharing a kind and key must never be
    reported as contradicting one another.
    """
    ref = run_ref("run-204")
    ev = evidence(ref, accuracy=0.93)
    # Duplicate the exact same metric value under the same key, as if
    # the same result had been logged twice.
    ev.add_item(type(ev.items[0])(kind=ev.items[0].kind, key="accuracy", value=0.93, source=ref))
    c = claim("C204", "Model X accuracy", ClaimCategory.PERFORMANCE, evidence_trace=(ref,))

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is False
    assert result.contradictions == ()


def test_resolved_contradiction_is_not_carried_forward_as_a_finding(
    rule: ContradictionRule,
) -> None:
    """A `Contradiction` already advanced to `RESOLVED` (Chapter 8 §5)
    is a closed matter, not an active conflict this rule should keep
    re-surfacing on every pass -- R003 only carries forward
    *unresolved* known contradictions.
    """
    ref = run_ref("run-205")
    ev = evidence(ref, accuracy=0.9)
    c = claim("C205", "Model X accuracy", ClaimCategory.PERFORMANCE, evidence_trace=(ref,))
    counterpart = claim(
        "C205b", "Model X accuracy (independent replication)", ClaimCategory.PERFORMANCE
    )

    resolved = Contradiction(
        id="CD-001",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.RESOLVED,
        claims=(c, counterpart),
        resolution=ContradictionResolution(
            kind=ContradictionResolutionKind.SCOPE_DIFFERENCE_CLARIFIED,
            explanation="The two runs were found to target different eval splits.",
        ),
    )

    result = rule.evaluate(
        context(evidence_bundles=[ev], claims=[c], detected_contradictions=(resolved,))
    )

    assert result.triggered is False
    assert result.contradictions == ()


def test_unresolved_known_contradiction_is_carried_forward(rule: ContradictionRule) -> None:
    """A `Contradiction` still at `DETECTED` (never resolved) and
    naming a claim in scope must be surfaced by R003, per
    specification §3's requirement that a rule take relevant,
    already-recorded contradictions as input rather than silently
    dropping them.
    """
    ref = run_ref("run-206")
    ev = evidence(ref, accuracy=0.9)
    c = claim("C206", "Model X accuracy", ClaimCategory.PERFORMANCE, evidence_trace=(ref,))
    counterpart = claim(
        "C206b", "Model X accuracy (independent replication)", ClaimCategory.PERFORMANCE
    )

    unresolved = Contradiction(
        id="CD-002",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    result = rule.evaluate(
        context(evidence_bundles=[ev], claims=[c], detected_contradictions=(unresolved,))
    )

    assert result.triggered is True
    assert any("CD-002" in finding for finding in result.contradictions)
    assert any("remains unresolved" in finding for finding in result.contradictions)


def test_multiple_claims_within_and_cross_claim_findings_combine(rule: ContradictionRule) -> None:
    """A three-claim context exercising two conflict types at once: one
    claim (C301) whose own attributed evidence disagrees with itself
    (an "impossible simultaneous state" -- two recorded values for the
    same run's accuracy), and a *separate* pair of claims (C302, C303)
    about a different subject that genuinely conflict with each other
    under matched scope. All findings must surface together, each
    correctly attributed.
    """
    ref_self_conflict = run_ref("run-301")
    ev_self_conflict = evidence(ref_self_conflict, accuracy=0.95)
    ev_self_conflict.add_item(
        type(ev_self_conflict.items[0])(
            kind=ev_self_conflict.items[0].kind,
            key="accuracy",
            value=0.40,
            source=ref_self_conflict,
        )
    )
    claim_self_conflict = claim(
        "C301", "Model Y accuracy", ClaimCategory.PERFORMANCE, evidence_trace=(ref_self_conflict,)
    )

    ref_b = run_ref("run-302")
    ref_c = run_ref("run-303")
    ev_b = evidence(ref_b, hardware={"name": "A100"}, throughput=120.0)
    ev_c = evidence(ref_c, hardware={"name": "A100"}, throughput=95.0)
    claim_b = claim(
        "C302",
        "Model Z throughput on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref_b,),
        claim_scope=scope(hardware="A100"),
    )
    claim_c = claim(
        "C303",
        "Model Z throughput on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref_c,),
        claim_scope=scope(hardware="A100"),
    )

    result = rule.evaluate(
        context(
            evidence_bundles=[ev_self_conflict, ev_b, ev_c],
            claims=[claim_self_conflict, claim_b, claim_c],
        )
    )

    assert result.triggered is True
    assert len(result.contradictions) == 2
    assert any("cannot both hold" in c for c in result.contradictions)
    assert any("120.0" in c and "95.0" in c for c in result.contradictions)
    assert {c.id for c in result.affected_claims} == {"C301", "C302", "C303"}
