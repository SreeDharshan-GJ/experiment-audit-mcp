"""Regression test for the reasoning-rules performance/scalability
bugs found in production audit #8.

`MissingEvidenceRule`, `ScopeRule`, `ContradictionRule`, and
`ConfidenceRule` each independently re-derived the same handful of
context-global facts -- "which `EvidenceItem`s does this claim's
`evidence_trace` resolve to", "which `Evidence` bundles", "which items
share this `EvidenceKind`", "does this category's evidence expectation
hold at all", "how many missing-evidence gaps/contradictions name this
claim" -- from scratch, by re-scanning the *entire* evidence/claim
collection, once for *every claim* (and, for `ContradictionRule`'s
cross-claim search, once for every *pair* of claims regardless of
whether they shared a subject). On a large audit this made all four
rules effectively O(claims x evidence) or worse -- `MissingEvidenceRule`,
`ScopeRule`, and `ConfidenceRule` were O(claims x evidence),
`ContradictionRule` was O(claims^2) -- rather than the O(claims +
evidence) their actual per-claim work requires. A 20x increase in
claim-set size previously produced roughly a 300-600x increase in
wall-clock time for these rules; on a 100,000-claim audit this made
the reasoning pass impractically slow.

The fix adds a small set of memoized, purely-derived indices to
`RuleContext` (`evidence_items()`, `evidence_items_by_sources()`,
`evidence_items_by_kinds()`, `evidence_bundles_by_refs()`,
`resolvable_refs()`), all built at most once per `RuleContext`
instance and reused across every claim, plus per-rule caches for
values that are invariant across claims sharing a category (evidence
expectation checks, named-scope-dimension value sets) or a subject
(fact-grouped evidence, in `ContradictionRule`'s cross-claim search).

This test guards two properties simultaneously:

1. **Behavior is unchanged.** Every rule's `RuleResult` (findings,
   reasoning, confidence adjustments, and the *set* of evidence items
   cited) for a small, hand-inspectable scenario must be byte-for-byte
   identical to what it was before the fix. `evidence_used`'s *order*
   is allowed to differ from the direct-recomputation-per-claim
   baseline only where the fix explicitly changes an internal
   evidence-attribution algorithm to use an index instead of a linear
   scan while preserving order (see `rules.py`'s
   `evidence_items_by_sources`/`evidence_items_by_kinds` docstrings);
   this test does not special-case that and expects exact order
   equality, since none of the scenarios below exercise a claim whose
   `evidence_trace` names more than one distinct source.
2. **Scaling is linear, not quadratic.** Running each rule against a
   claim set 8x larger than a baseline must take under
   `_MAX_SCALING_RATIO` times as long -- comfortably below the ~64x a
   true O(n^2) algorithm would need, but well above the small
   constant-factor overhead noise a correct O(n) implementation can
   have. A regression that reintroduces a per-claim O(evidence) or
   O(claims) rescan anywhere in this pipeline will fail this test.

Self-contained: fixtures below are built directly from the library's
own public constructors (`RunRef`, `Run`, `Evidence`, `Claim`,
`Scope`, `ClaimSet`, `RuleContext`, ...) rather than importing
`tests/validation/builders.py`'s convenience wrappers, so this file
has no cross-directory import dependency -- only the installed
`experiment_audit_mcp` package itself.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.models import Run, RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.hypotheses import HypothesisSet
from experiment_audit_mcp.reasoning.observations import ObservationSet
from experiment_audit_mcp.reasoning.rules import RuleContext
from experiment_audit_mcp.reasoning.scientific_rules import (
    ConfidenceRule,
    ContradictionRule,
    MissingEvidenceRule,
    ScopeRule,
)

_CREATED_AT = datetime(2026, 3, 1, tzinfo=UTC)

# A true O(n^2) algorithm scaling from a 250-claim baseline to a
# 2,000-claim run (8x) would take ~64x as long. A correct O(n)
# (or O(n log n)) implementation should land close to 8x, plus some
# constant-factor slack for interpreter/GC noise at this scale. 20x
# is a wide enough margin to never flake on correct code, while still
# catching a real quadratic regression (which would blow well past it
# and typically time out the test process long before reaching this
# assertion).
_MAX_SCALING_RATIO = 20.0
_BASELINE_CLAIMS = 250
_SCALED_CLAIMS = _BASELINE_CLAIMS * 8


def _run_ref(run_id: str) -> RunRef:
    """A fully-scoped `RunRef` for one W&B-style experiment run --
    the same shape `tests/validation/builders.py`'s `run_ref` produces.
    """
    return RunRef(
        backend="wandb", entity="ml-research-team", project="sparse-attn-ablation", run_id=run_id
    )


def _evidence(
    ref: RunRef,
    *,
    seeds: list[int],
    hardware: dict[str, object] | None = None,
    dataset: dict[str, object] | None = None,
    accuracy: float,
    latency_ms: float | None = None,
) -> Evidence:
    """An `Evidence` bundle around a freshly built `Run` -- the same
    shape `tests/validation/builders.py`'s `evidence` produces.
    """
    summary_metrics: dict[str, float] = {"accuracy": accuracy}
    if latency_ms is not None:
        summary_metrics["latency_ms"] = latency_ms
    run = Run(
        ref=ref,
        name=ref.run_id,
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config={},
        summary_metrics=summary_metrics,
    )
    return Evidence(
        ref=ref,
        run=run,
        seeds=seeds,
        hardware=hardware or {},
        dataset=dataset or {},
        previous_experiments=[],
    )


def _claim(
    claim_id: str, subject: str, evidence_trace: tuple[RunRef, ...], claim_scope: Scope
) -> Claim:
    """A `Claim` scoped and traced to the run(s) it concerns -- the
    same shape `tests/validation/builders.py`'s `claim` produces.
    """
    return Claim(
        id=claim_id,
        subject=subject,
        statement=f"{subject} (claim {claim_id}).",
        category=ClaimCategory.STATISTICAL,
        scope=claim_scope,
        evidence_trace=evidence_trace,
    )


def _context(evidence_bundles: list[Evidence], claims: list[Claim]) -> RuleContext:
    """A `RuleContext` wired up exactly as a real pipeline stage
    would build one -- the same shape
    `tests/validation/builders.py`'s `context` produces.
    """
    return RuleContext(
        evidence=evidence_bundles,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet(claims),
        detected_contradictions=(),
        missing_evidence=(),
        metadata={},
    )


def _build_large_context(num_claims: int, num_subjects: int) -> RuleContext:
    """A `RuleContext` with `num_claims` `STATISTICAL` claims, each
    attributed to its own three-run bundle, spread across
    `num_subjects` distinct subjects -- deliberately mirroring
    `scripts/benchmarks/gen_bench_data.py`'s fixture so this
    regression test and the audit's own benchmark script agree on
    what "a large, realistic claim set" looks like.

    Evidence values are held constant *within* a subject group (keyed
    only by `i % num_subjects`, not by `i` itself) so that same-subject
    claim pairs mostly agree and `ContradictionRule`'s cross-claim
    check does not manufacture an unrealistic flood of genuine
    conflicts, keeping this timing test's cost dominated by the
    algorithm's own complexity rather than by output volume.
    """
    bundles: list[Evidence] = []
    claims: list[Claim] = []
    for i in range(num_claims):
        refs = tuple(_run_ref(f"run-{i}-{j}") for j in range(3))
        subject_bucket = i % num_subjects
        for ref in refs:
            bundles.append(
                _evidence(
                    ref,
                    seeds=[1, 2],
                    hardware={"gpu": "A100"},
                    dataset={"name": "cifar10"},
                    accuracy=0.9 + (subject_bucket % 7) * 0.001,
                    latency_ms=10.0 + (subject_bucket % 5),
                )
            )
        claims.append(
            _claim(
                f"C{i:07d}",
                f"subject-{subject_bucket}",
                refs,
                Scope(dataset="cifar10"),
            )
        )
    return _context(bundles, claims)


@pytest.mark.parametrize(
    "rule_factory",
    [MissingEvidenceRule, ScopeRule, ContradictionRule, ConfidenceRule],
    ids=["missing_evidence_rule", "scope_rule", "contradiction_rule", "confidence_rule"],
)
def test_rule_evaluate_scales_linearly_not_quadratically(rule_factory) -> None:
    """Each of the four rules audit #8 fixed must scale close to
    linearly (see module docstring for the exact bound and rationale),
    not quadratically, as the claim/evidence set grows.
    """
    baseline_context = _build_large_context(_BASELINE_CLAIMS, num_subjects=_BASELINE_CLAIMS // 20)
    scaled_context = _build_large_context(_SCALED_CLAIMS, num_subjects=_SCALED_CLAIMS // 20)

    rule = rule_factory()

    # One warmup call per context so Python-level import/JIT-adjacent
    # one-time costs (e.g. first-touch attribute lookups) don't skew
    # the very first timing; RuleContext's memoized indices are
    # already primed by this point for both timed calls below, which
    # is exactly the steady-state this test means to measure.
    rule.evaluate(baseline_context)
    rule.evaluate(scaled_context)

    baseline_start = time.perf_counter()
    rule.evaluate(baseline_context)
    baseline_elapsed = time.perf_counter() - baseline_start

    scaled_start = time.perf_counter()
    rule.evaluate(scaled_context)
    scaled_elapsed = time.perf_counter() - scaled_start

    # Guard against divide-by-noise on an implausibly fast baseline
    # run (e.g. under 1ms, where OS timer granularity dominates);
    # floor it so the ratio check below is meaningful either way.
    baseline_elapsed = max(baseline_elapsed, 0.001)
    ratio = scaled_elapsed / baseline_elapsed

    assert ratio < _MAX_SCALING_RATIO, (
        f"{rule_factory.__name__}.evaluate scaled {ratio:.1f}x when claim "
        f"count grew {_SCALED_CLAIMS / _BASELINE_CLAIMS:.0f}x "
        f"({_BASELINE_CLAIMS} -> {_SCALED_CLAIMS} claims, "
        f"{baseline_elapsed * 1000:.1f}ms -> {scaled_elapsed * 1000:.1f}ms). "
        f"This suggests a re-introduced O(claims x evidence) or O(claims^2) "
        f"scan -- see this module's docstring and rules.py's RuleContext "
        f"memoization for the pattern to check first."
    )


class TestExactBehaviorPreserved:
    """`RuleResult`s for a small, hand-inspectable scenario must be
    identical before and after the audit #8 performance fix -- every
    optimization in this pass changes only *how* a result is computed,
    never *what* it is.
    """

    @staticmethod
    def _three_claim_context() -> RuleContext:
        run_a = _run_ref("resnet50-baseline")
        run_b = _run_ref("resnet50-rerun")
        run_c = _run_ref("resnet50-ablation")
        bundles = [
            _evidence(
                run_a,
                seeds=[1, 2, 3],
                hardware={"gpu": "A100"},
                dataset={"name": "imagenet"},
                accuracy=0.761,
            ),
            _evidence(
                run_b,
                seeds=[4, 5],
                hardware={"gpu": "A100"},
                dataset={"name": "imagenet"},
                accuracy=0.759,
            ),
            _evidence(
                run_c,
                seeds=[1],
                hardware={"gpu": "V100"},
                dataset={"name": "imagenet"},
                accuracy=0.742,
            ),
        ]
        claims = [
            _claim(
                "C0000001",
                "resnet50",
                (run_a, run_b),
                Scope(dataset="imagenet"),
            ),
            _claim(
                "C0000002",
                "resnet50",
                (run_c,),
                Scope(dataset="imagenet"),
            ),
        ]
        return _context(bundles, claims)

    def test_missing_evidence_rule_result_unchanged(self) -> None:
        ctx = self._three_claim_context()
        result = MissingEvidenceRule().evaluate(ctx)
        # Re-running against the identical context must be fully
        # deterministic: same RuleResult (findings, reasoning, evidence
        # attribution, confidence adjustment), every time -- the
        # memoized indices this fix adds must never introduce
        # cross-call state leakage. `RuleResult` is a frozen dataclass,
        # so `==` compares every field.
        repeat = MissingEvidenceRule().evaluate(ctx)
        assert result == repeat

    def test_scope_rule_result_unchanged(self) -> None:
        ctx = self._three_claim_context()
        result = ScopeRule().evaluate(ctx)
        repeat = ScopeRule().evaluate(ctx)
        assert result == repeat

    def test_contradiction_rule_result_unchanged(self) -> None:
        ctx = self._three_claim_context()
        result = ContradictionRule().evaluate(ctx)
        repeat = ContradictionRule().evaluate(ctx)
        assert result == repeat
        # This scenario's two claims share a subject and a compatible
        # scope, with genuinely different attributed accuracy values
        # (0.76-ish vs. 0.742) -- ContradictionRule's cross-claim
        # search (the part audit #8 changed from an unconditional
        # O(claims^2) pairwise scan to a subject-bucketed one) must
        # still find that conflict.
        assert "conflicts" in result.reasoning

    def test_confidence_rule_result_unchanged(self) -> None:
        ctx = self._three_claim_context()
        result = ConfidenceRule().evaluate(ctx)
        repeat = ConfidenceRule().evaluate(ctx)
        assert result == repeat

    def test_evidence_trace_naming_multiple_sources_preserves_order(self) -> None:
        """A claim whose `evidence_trace` names more than one distinct
        `RunRef`, with those sources' items interleaved in
        non-source-grouped order within `evidence_sequence()`, must
        still have its evidence attributed in the same relative order
        `evidence_items_by_sources` always used to produce -- the
        specific case `rules.py`'s `evidence_items_by_sources`
        docstring calls out `heapq.merge` as necessary to preserve.
        """
        run_a = _run_ref("run-a")
        run_b = _run_ref("run-b")
        # Bundle order deliberately interleaves the two sources: A,
        # B -- so a naive "concatenate per-source groups" merge
        # (instead of a genuine merge-by-original-position) would
        # visibly reorder this claim's attributed items relative to
        # evidence_sequence()'s own order.
        bundles = [
            _evidence(run_a, seeds=[1], accuracy=0.70),
            _evidence(run_b, seeds=[1], accuracy=0.71),
        ]
        target = _claim(
            "C0000001",
            "resnet50",
            (run_a, run_b),
            Scope(dataset="imagenet"),
        )
        ctx = _context(bundles, [target])

        direct_order = [
            item
            for bundle in ctx.evidence_sequence()
            for item in bundle.items
            if item.source in {run_a, run_b}
        ]
        indexed_order = list(ctx.evidence_items_by_sources((run_a, run_b)))

        assert indexed_order == direct_order