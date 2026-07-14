"""Scientific validation of `RecommendationRule` (R006).

R006 turns R001-R005's already-closed findings into concrete, traceable
next steps, via four independent pathways per claim: (1) an
already-identified missing-evidence/scope gap, (2) an unresolved
contradiction, (3) confidence below this rule's own low-confidence
threshold (0.5), and (4) a non-SUPPORTED judgment. All four may fire
for the same claim; recommendations are deduplicated only on exact
text coincidence, and pathway order (1 -> 2 -> 3 -> 4) is the order
recommendations are emitted in.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionStatus,
)
from experiment_audit_mcp.reasoning.scientific_rules.recommendation_rule import RecommendationRule

from .builders import GapRecord, claim, context, run_ref


@pytest.fixture
def rule() -> RecommendationRule:
    return RecommendationRule()


def test_no_recommendations_for_a_clean_supported_claim(rule: RecommendationRule) -> None:
    """A claim with no missing-evidence gap, no unresolved
    contradiction, high confidence, and a SUPPORTED judgment gives
    R006 nothing traceable to recommend -- a scientifically sound
    absence of action, not an oversight.
    """
    c = claim("C601", "SparseAttn accuracy on CIFAR-10", ClaimCategory.PERFORMANCE)

    result = rule.evaluate(
        context(
            claims=[c],
            metadata={
                "confidence_adjustments": {"C601": 0.9},
                "judgments": {"C601": "supported"},
            },
        )
    )

    assert result.triggered is False
    assert result.recommendations == ()
    assert result.affected_claims == ()


def test_grouped_recommendations_for_a_claim_hit_by_every_pathway(
    rule: RecommendationRule,
) -> None:
    """A single claim simultaneously carrying an open missing-evidence
    gap (R001/R002), an unresolved contradiction (R003), low
    confidence (R004), and a PARTIALLY_SUPPORTED judgment (R005) earns
    one recommendation from each pathway, all grouped under that one
    claim -- none of the four is dropped merely because the others
    also fired.
    """
    c = claim("C602", "SparseAttn generalizes to speech data", ClaimCategory.GENERALIZATION)
    gap = "Claim C602 ('SparseAttn generalizes to speech data'): Additional datasets"
    counterpart = claim(
        "C602b",
        "SparseAttn generalizes to speech data (independent replication)",
        ClaimCategory.GENERALIZATION,
    )
    unresolved = Contradiction(
        id="CD-030",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    result = rule.evaluate(
        context(
            claims=[c],
            missing_evidence=(gap,),
            detected_contradictions=(unresolved,),
            metadata={
                "confidence_adjustments": {"C602": 0.2},
                "judgments": {"C602": "partially_supported"},
            },
        )
    )

    assert result.triggered is True
    assert len(result.recommendations) == 4
    assert result.affected_claims == (c,)
    joined = " ".join(result.recommendations)
    assert "collect the missing evidence" in joined
    assert "CD-030" in joined
    assert "Strengthen evidence quality" in joined
    assert "Resolve the outstanding missing-evidence" in joined


def test_priority_ordering_follows_the_documented_pathway_sequence(
    rule: RecommendationRule,
) -> None:
    """When several pathways fire for one claim, R006 always emits
    them in the same fixed order its own docstring documents:
    missing-evidence/scope (1), unresolved contradiction (2), low
    confidence (3), then judgment (4) -- never re-prioritized by
    severity or any other runtime-computed ranking.
    """
    c = claim("C603", "SparseAttn's robustness to adversarial noise", ClaimCategory.ROBUSTNESS)
    gap = (
        "Claim C603 ('SparseAttn's robustness to adversarial noise'): "
        "Evidence under perturbed conditions"
    )
    counterpart = claim(
        "C603b",
        "SparseAttn's robustness to adversarial noise (independent replication)",
        ClaimCategory.ROBUSTNESS,
    )
    unresolved = Contradiction(
        id="CD-031",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    result = rule.evaluate(
        context(
            claims=[c],
            missing_evidence=(gap,),
            detected_contradictions=(unresolved,),
            metadata={
                "confidence_adjustments": {"C603": 0.1},
                "judgments": {"C603": "unsupported"},
            },
        )
    )

    assert len(result.recommendations) == 4
    assert "collect the missing evidence" in result.recommendations[0]
    assert "CD-031" in result.recommendations[1]
    assert "Strengthen evidence quality" in result.recommendations[2]
    assert "Strengthen or replace the evidence" in result.recommendations[3]


def test_duplicated_recommendation_text_is_deduplicated_across_claims(
    rule: RecommendationRule,
) -> None:
    """A single, context-wide missing-evidence gap -- one that names no
    specific claim (no `claim_id`/`claim_ids` attribute) -- is
    conservatively attributed to *every* claim in scope, per this
    rule's own documented attribution fallback. When that gap's own
    `recommendation` text does not itself vary by claim, the identical
    string is produced for both claims, and R006 must deduplicate it
    to a single entry in `recommendations` even though it counts both
    claims as affected.
    """
    ref_a = run_ref("run-701")
    ref_b = run_ref("run-702")
    claim_a = claim("C701", "Model X accuracy", ClaimCategory.STATISTICAL, evidence_trace=(ref_a,))
    claim_b = claim("C702", "Model Y accuracy", ClaimCategory.STATISTICAL, evidence_trace=(ref_b,))
    shared_gap = GapRecord(recommendation="Record at least two seeds across the sweep.")

    result = rule.evaluate(context(claims=[claim_a, claim_b], missing_evidence=(shared_gap,)))

    assert result.triggered is True
    assert result.recommendations == ("Record at least two seeds across the sweep.",)
    assert {c.id for c in result.affected_claims} == {"C701", "C702"}
