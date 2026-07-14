"""End-to-end scientific validation of `ScientificReasoningPipeline`.

Each scenario below runs the full, fixed R001 -> R002 -> R003 -> R004
-> R005 -> R006 sequence (pipeline.py's `ScientificReasoningPipeline`)
against a realistic experiment-audit situation and checks the resulting
`ScientificReport` end to end: which rules triggered, how the context
was threaded between stages (`missing_evidence` accumulation,
`metadata["confidence_adjustment"/"confidence_adjustments"/"judgments"]`),
and what the terminal judgment and recommendations were. No rule module
is modified, wrapped, or bypassed; every assertion below is checking
the frozen implementation's own, real output.

For each scenario this file states, in its own docstring, the
scientific reasoning behind every expected output -- why the evidence
is missing what it's missing, why the scope holds or doesn't, why a
contradiction should or shouldn't be found, what confidence and
judgment that evidentiary picture warrants, and what a researcher
should therefore be told to do next.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionStatus,
)
from experiment_audit_mcp.reasoning.pipeline import ScientificReasoningPipeline

from .builders import claim, evidence, run_ref, scope
from .builders import context as build_context

# NOTE: `ScientificReasoningPipeline.build_initial_context` is a thin
# pass-through to `RuleContext`'s constructor, but as shipped it never
# forwards `observations`/`hypotheses` (both required, no-default
# fields on `RuleContext`), so calling it raises `TypeError` for every
# input. Per this task's "treat the implementation as frozen"
# constraint we do not patch that method; instead we build the
# `RuleContext` directly (via this suite's own `build_context` helper,
# which does supply empty `ObservationSet`/`HypothesisSet` instances)
# and pass it straight to `pipeline.execute(...)`, exactly as
# `build_initial_context` would have if it worked.


def _pipeline() -> ScientificReasoningPipeline:
    return ScientificReasoningPipeline()


def test_well_evidenced_claim_is_supported_end_to_end() -> None:
    """Scenario: a performance claim about SparseAttn's CIFAR-10
    accuracy, backed by two independent runs, each contributing a seed,
    one of which links to a prior experiment, and both agreeing on the
    reported accuracy.

    Expected reasoning, stage by stage:
    - R001 (Missing Evidence): the claim is PERFORMANCE, which only
      requires evaluation-metric evidence; both runs report `accuracy`,
      so nothing is missing.
    - R002 (Scope): the claim declares dataset="cifar-10", matching
      both runs' recorded dataset -- no violation.
    - R003 (Contradiction): both runs report the *same* accuracy value
      (0.912), so no conflict exists to detect.
    - R004 (Confidence): two independent sources, >=2 seeds
      (statistical support), one linked prior experiment
      (reproducibility), and a fully resolvable trace (traceability)
      -- every strengthening factor applies and nothing reduces it, so
      confidence is the maximum, 1.0.
    - R005 (Judgment): no missing-evidence gap, no contradiction, and
      confidence >= 0.5 -> SUPPORTED.
    - R006 (Recommendation): a SUPPORTED judgment with no open gap or
      conflict implies no corrective action.
    """
    prior_ref = run_ref("run-901-prior")
    prior = evidence(prior_ref, dataset={"name": "cifar-10"}, accuracy=0.905)

    ref_a = run_ref("run-901")
    ref_b = run_ref("run-902")
    ev_a = evidence(
        ref_a, dataset={"name": "cifar-10"}, seeds=[7], previous_experiments=[prior], accuracy=0.912
    )
    ev_b = evidence(ref_b, dataset={"name": "cifar-10"}, seeds=[7], accuracy=0.912)

    c = claim(
        "C901",
        "SparseAttn attains 91.2% accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref_a, ref_b),
        claim_scope=scope(dataset="cifar-10"),
    )

    ctx = build_context(claims=[c], evidence_bundles=[ev_a, ev_b])
    report = _pipeline().execute(ctx)

    assert [record.rule_id for record in report.execution_trace] == [
        "R001",
        "R002",
        "R003",
        "R004",
        "R005",
        "R006",
    ]
    assert all(record.applied for record in report.execution_trace)

    assert report.by_rule_id("R001").triggered is False
    assert report.by_rule_id("R002").triggered is False
    assert report.by_rule_id("R003").triggered is False
    assert report.by_rule_id("R004").confidence_adjustment == pytest.approx(1.0)
    assert report.by_rule_id("R005").metadata["judgments"]["C901"] == "supported"
    assert report.by_rule_id("R006").recommendations == ()

    # Context threading: R004's aggregate and per-claim adjustments, and
    # R005's judgments, must both have been copied forward into the
    # final context's metadata, per pipeline.py's documented contract.
    assert report.context.metadata["confidence_adjustment"] == pytest.approx(1.0)
    assert report.context.metadata["confidence_adjustments"]["C901"] == pytest.approx(1.0)
    assert report.context.metadata["judgments"]["C901"] == "supported"


def test_under_evidenced_statistical_claim_is_downgraded_end_to_end() -> None:
    """Scenario: a statistical claim ("SparseAttn's accuracy gain over
    the dense baseline is real") backed by only a single seed on a
    single run.

    Expected reasoning, stage by stage:
    - R001: STATISTICAL claims require >=2 recorded seeds; only one
      was logged, so this is reported as missing.
    - R002: the claim's declared scope (dataset="cifar-10") matches
      the evidence -- no scope violation.
    - R003: only one claim, one run -- nothing to compare against, no
      contradiction.
    - R004: no reproducibility evidence, no statistical support (one
      seed), only one source, but the trace is resolvable
      (traceability): 0.25 strengths. One missing-evidence gap
      (from R001) attributed to this claim costs -0.20. Net: 0.05,
      a claim whose evidentiary support is thin but not actively
      contradicted.
    - R005: no unresolved contradiction, but a missing-evidence gap is
      present and confidence (0.05) is below the 0.5 supported
      threshold and above the -0.25 unsupported threshold ->
      PARTIALLY_SUPPORTED.
    - R006: the still-open missing-evidence gap (pathway 1) and the
      non-SUPPORTED judgment (pathway 4) both warrant a
      recommendation; confidence (0.05) is also below R006's own 0.5
      low-confidence threshold (pathway 3).
    """
    ref = run_ref("run-903")
    ev = evidence(ref, dataset={"name": "cifar-10"}, seeds=[42], accuracy=0.93)

    c = claim(
        "C903",
        "SparseAttn's accuracy gain over the dense baseline is a real effect",
        ClaimCategory.STATISTICAL,
        evidence_trace=(ref,),
        claim_scope=scope(dataset="cifar-10"),
    )

    ctx = build_context(claims=[c], evidence_bundles=[ev])
    report = _pipeline().execute(ctx)

    assert report.by_rule_id("R001").triggered is True
    assert any("C903" in m for m in report.by_rule_id("R001").missing_evidence)

    assert report.by_rule_id("R002").triggered is False

    assert report.by_rule_id("R004").confidence_adjustment == pytest.approx(0.05)
    assert report.by_rule_id("R005").metadata["judgments"]["C903"] == "partially_supported"

    recommendations = report.by_rule_id("R006").recommendations
    assert len(recommendations) == 3
    assert any("collect the missing evidence" in r for r in recommendations)
    assert any("Strengthen evidence quality" in r for r in recommendations)
    assert any("Resolve the outstanding missing-evidence" in r for r in recommendations)

    # Context threading: R001's missing-evidence findings must have
    # been carried into the context R002 (and everything downstream)
    # saw, and accumulated further if R002 also found something (it
    # did not, here, so the count is exactly R001's one finding).
    assert len(report.context.missing_evidence) == 1


def test_carried_forward_contradiction_caps_judgment_end_to_end() -> None:
    """Scenario: a throughput claim about SparseAttn on an A100, where
    an earlier audit pass already detected and recorded an unresolved
    contradiction naming this exact claim (e.g. a second, conflicting
    run was found after the claim was first formulated). This
    contradiction is supplied as already-known input
    (`detected_contradictions`), exactly as a real second pipeline
    pass over previously-flagged findings would receive it --
    `ContradictionRule`'s own newly-detected findings are never fed
    forward within a single pipeline run (pipeline.py's own documented
    contract), so only a *pre-existing* `Contradiction` can affect
    R004/R005/R006 in one pass.

    Expected reasoning, stage by stage:
    - R001/R002: complete, on-scope evidence -- neither triggers.
    - R003: carries the already-known, unresolved contradiction
      forward as a finding (it does not re-detect it; it was supplied,
      not derived from this pass's own evidence).
    - R004: traceability (+0.25) is the only strengthening factor
      available; the unresolved contradiction penalty (-0.35) leaves
      a net confidence of -0.10.
    - R005: an unresolved contradiction against this claim, with
      confidence above the unsupported threshold (-0.10 > -0.25),
      caps the verdict at PARTIALLY_SUPPORTED -- never SUPPORTED,
      regardless of how the rest of the evidence looks.
    - R006: the unresolved contradiction (pathway 2) and the
      non-SUPPORTED judgment (pathway 4) both warrant action; -0.10
      is also below R006's low-confidence threshold (pathway 3).
    """
    ref = run_ref("run-905")
    ev = evidence(ref, hardware={"name": "A100"}, throughput=118.0)

    c = claim(
        "C905",
        "SparseAttn's throughput improvement on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref,),
        claim_scope=scope(hardware="A100"),
    )
    counterpart = claim(
        "C905b",
        "SparseAttn's throughput improvement on an A100 (independent replication)",
        ClaimCategory.EFFICIENCY,
        claim_scope=scope(hardware="A100"),
    )

    known_contradiction = Contradiction(
        id="CD-905",
        categories=(ContradictionCategory.EXPERIMENTAL,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    ctx = build_context(
        claims=[c], evidence_bundles=[ev], detected_contradictions=(known_contradiction,)
    )
    report = _pipeline().execute(ctx)

    assert report.by_rule_id("R001").triggered is False
    assert report.by_rule_id("R002").triggered is False

    assert report.by_rule_id("R003").triggered is True
    assert any("CD-905" in finding for finding in report.by_rule_id("R003").contradictions)

    assert report.by_rule_id("R004").confidence_adjustment == pytest.approx(-0.10)
    assert report.by_rule_id("R005").metadata["judgments"]["C905"] == "partially_supported"

    recommendations = report.by_rule_id("R006").recommendations
    assert any("CD-905" in r for r in recommendations)
    assert any("Strengthen evidence quality" in r for r in recommendations)


def test_generalization_claim_without_cross_dataset_evidence_is_capped_end_to_end() -> None:
    """Scenario: a generalization claim ("SparseAttn's accuracy gains
    generalize across image classification tasks") whose scope leaves
    `dataset` undeclared -- a Generalization claim's own category
    asserts applicability beyond a single dataset (claims.py §5) -- but
    whose only evidence is a single CIFAR-10 run.

    Expected reasoning, stage by stage:
    - R001: GENERALIZATION requires cross-domain evaluation and
      additional datasets; only one dataset was ever evidenced, so
      both expectations are reported missing.
    - R002: the claim leaves `dataset` undeclared, which -- for a
      GENERALIZATION claim -- asserts breadth beyond one dataset; the
      evidence covers only one distinct dataset value, so this is
      *also* reported as a scope violation (the breadth check), a
      second, independent gap on top of R001's category-driven ones.
    - R003: only one run, one claim -- nothing to compare, no
      contradiction.
    - R004: no reproducibility, no statistical support, only one
      source, but the trace does resolve (traceability, +0.25);
      three missing-evidence/scope gaps (2 from R001, 1 from R002)
      apply, each costing -0.20, for a combined penalty of -0.60
      (0.25 - 0.60 = -0.35, clamped within [-1, 1]).
    - R005: confidence (-0.35) is below the -0.25 unsupported
      threshold, with no unresolved contradiction -> UNSUPPORTED: the
      evidence on hand does not justify a claim of this breadth at
      all.
    - R006: the three missing-evidence/scope gaps (pathway 1) and the
      UNSUPPORTED judgment (pathway 4) both warrant action; -0.35 is
      also below R006's low-confidence threshold (pathway 3).
    """
    ref = run_ref("run-907")
    ev = evidence(ref, dataset={"name": "cifar-10"}, accuracy=0.90)

    c = claim(
        "C907",
        "SparseAttn's accuracy gains generalize across image classification tasks",
        ClaimCategory.GENERALIZATION,
        evidence_trace=(ref,),
        # `dataset` is deliberately left undeclared -- the breadth-sensitive
        # dimension for GENERALIZATION -- while `hardware` is declared so
        # `Scope.is_unspecified()` is False and R002's breadth check actually
        # runs (a fully empty `Scope()` would instead be skipped outright as
        # scope ambiguity, per specification Section 8).
        claim_scope=scope(hardware="A100"),
    )

    ctx = build_context(claims=[c], evidence_bundles=[ev])
    report = _pipeline().execute(ctx)

    assert report.by_rule_id("R001").triggered is True
    assert len(report.by_rule_id("R001").missing_evidence) == 2

    assert report.by_rule_id("R002").triggered is True
    assert len(report.by_rule_id("R002").missing_evidence) == 1

    # All three gaps must have been threaded forward into the shared
    # context.missing_evidence R004/R005/R006 all read from.
    assert len(report.context.missing_evidence) == 3

    assert report.by_rule_id("R004").confidence_adjustment == pytest.approx(-0.35)
    assert report.by_rule_id("R005").metadata["judgments"]["C907"] == "unsupported"

    recommendations = report.by_rule_id("R006").recommendations
    assert any("collect the missing evidence" in r for r in recommendations)
    assert any("Strengthen or replace the evidence" in r for r in recommendations)
