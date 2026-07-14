"""Shared synthetic large-scale fixture generator for audit #8 benchmarks.

Self-contained: builds `RuleContext`s directly from the library's own
public constructors (`RunRef`, `Run`, `Evidence`, `Claim`, `Scope`,
`ClaimSet`, `RuleContext`, ...), the same objects
`tests/validation/builders.py` wraps for the test suite -- but without
importing that test-only helper module, so this script has no
dependency on the `tests/` directory being present or on your
`PYTHONPATH`/working directory when you run it. The only requirement
is that the `experiment_audit_mcp` package itself is installed
(`pip install -e .` from the repo root).
"""

from __future__ import annotations

from datetime import UTC, datetime

from experiment_audit_mcp.models import Run, RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.hypotheses import HypothesisSet
from experiment_audit_mcp.reasoning.observations import ObservationSet
from experiment_audit_mcp.reasoning.rules import RuleContext

_CREATED_AT = datetime(2026, 3, 1, tzinfo=UTC)


def _run_ref(run_id: str) -> RunRef:
    return RunRef(
        backend="wandb", entity="ml-research-team", project="sparse-attn-ablation", run_id=run_id
    )


def _evidence(ref: RunRef, *, accuracy: float, latency_ms: float, seeds: list[int]) -> Evidence:
    run = Run(
        ref=ref,
        name=ref.run_id,
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config={},
        summary_metrics={"accuracy": accuracy, "latency_ms": latency_ms},
    )
    return Evidence(
        ref=ref,
        run=run,
        seeds=seeds,
        hardware={"gpu": "A100"},
        dataset={"name": "cifar10"},
        previous_experiments=[],
    )


def _claim(
    claim_id: str, subject: str, evidence_trace: tuple[RunRef, ...], claim_scope: Scope
) -> Claim:
    return Claim(
        id=claim_id,
        subject=subject,
        statement=f"{subject} (claim {claim_id}).",
        category=ClaimCategory.STATISTICAL,
        scope=claim_scope,
        evidence_trace=evidence_trace,
    )


def build_context(num_claims: int, evidence_per_claim: int = 3, num_subjects: int = 200):
    """Build a RuleContext with `num_claims` claims, each attributed to
    its own small run, spread across `num_subjects` distinct subjects
    (so ContradictionRule's cross-claim search has realistic bucket
    sizes rather than one giant same-subject group).

    Evidence values are held constant *within* a subject group (keyed
    only by `i % num_subjects`, not by `i` itself) so that same-subject
    claim pairs mostly agree and ContradictionRule's cross-claim check
    does not manufacture an unrealistic flood of genuine conflicts --
    realistic ML-audit claim sets have most same-subject pairs
    agreeing, with contradictions the exception this rule exists to
    catch, not the norm.
    """
    bundles: list[Evidence] = []
    claims: list[Claim] = []
    for i in range(num_claims):
        refs = tuple(_run_ref(f"run-{i}-{j}") for j in range(evidence_per_claim))
        subject_bucket = i % num_subjects
        for ref in refs:
            bundles.append(
                _evidence(
                    ref,
                    accuracy=0.9 + (subject_bucket % 7) * 0.001,
                    latency_ms=10.0 + (subject_bucket % 5),
                    seeds=[1, 2],
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
    return RuleContext(
        evidence=bundles,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet(claims),
        detected_contradictions=(),
        missing_evidence=(),
        metadata={},
    )