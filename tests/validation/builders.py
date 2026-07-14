"""Shared, realistic builders for the Scientific Validation suite.

Every helper here is a thin, literal wrapper around the frozen
implementation's own public constructors (`RunRef`, `Run`, `Evidence`,
`Claim`, `Scope`, `RuleContext`, ...). Nothing here monkeypatches,
subclasses, or otherwise alters the implementation under test -- these
are convenience factories only, mirroring the pattern already
established in `tests/reasoning/test_contradiction_rule.py`.

Scenarios throughout this suite are drawn from realistic ML-experiment
situations (a sparse-attention ablation, a vision-model benchmark
sweep, a reproducibility audit of a published result) rather than
placeholder ints and single-letter names, per this task's "realistic
scientific scenarios rather than toy examples" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experiment_audit_mcp.models import Run, RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.hypotheses import HypothesisSet
from experiment_audit_mcp.reasoning.observations import ObservationSet
from experiment_audit_mcp.reasoning.rules import RuleContext

_CREATED_AT = datetime(2026, 3, 1, tzinfo=UTC)


def run_ref(
    run_id: str,
    *,
    project: str = "sparse-attn-ablation",
    entity: str = "ml-research-team",
) -> RunRef:
    """A fully-scoped `RunRef` for one W&B-style experiment run."""
    return RunRef(backend="wandb", entity=entity, project=project, run_id=run_id)


def run(ref: RunRef, *, config: dict[str, Any] | None = None, **summary_metrics: float) -> Run:
    """A `Run` summary with realistic config/metric shape."""
    return Run(
        ref=ref,
        name=ref.run_id,
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config=dict(config) if config else {},
        summary_metrics=summary_metrics,
    )


def evidence(
    ref: RunRef,
    *,
    config: dict[str, Any] | None = None,
    seeds: list[int] | None = None,
    hardware: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    previous_experiments: list[Evidence] | None = None,
    include_run: bool = True,
    **summary_metrics: float,
) -> Evidence:
    """An `Evidence` bundle around a freshly built `Run`, plus whatever
    additional evidence dimensions (`seeds`, `hardware`, `dataset`,
    `previous_experiments`) the scenario calls for.
    """
    return Evidence(
        ref=ref,
        run=run(ref, config=config, **summary_metrics) if include_run else None,
        seeds=seeds or [],
        hardware=hardware or {},
        dataset=dataset or {},
        previous_experiments=previous_experiments or [],
    )


def scope(**fields: Any) -> Scope:
    """A `Scope` with only the named dimensions declared -- everything
    else defaults to `None` / empty, exactly as `Scope` itself does.
    """
    return Scope(**fields)


def claim(
    claim_id: str,
    subject: str,
    category: ClaimCategory,
    evidence_trace: tuple[RunRef, ...] = (),
    *,
    statement: str | None = None,
    claim_scope: Scope | None = None,
) -> Claim:
    """A `Claim` with a realistic subject/statement, scoped and traced
    to the run(s) it concerns.
    """
    return Claim(
        id=claim_id,
        subject=subject,
        statement=statement or f"{subject} (claim {claim_id}).",
        category=category,
        scope=claim_scope if claim_scope is not None else Scope(),
        evidence_trace=evidence_trace,
    )


def context(
    *,
    evidence_bundles: list[Evidence] | None = None,
    claims: list[Claim] | None = None,
    detected_contradictions: tuple[Any, ...] = (),
    missing_evidence: tuple[Any, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> RuleContext:
    """A `RuleContext` wired up exactly as a real pipeline stage would
    build one: empty `ObservationSet` / `HypothesisSet` (these rules
    never read them), the evidence and claims a scenario supplies, and
    whatever upstream findings (`detected_contradictions`,
    `missing_evidence`, `metadata`) the scenario wants pre-seeded.
    """
    return RuleContext(
        evidence=evidence_bundles or [],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet(claims) if claims is not None else None,
        detected_contradictions=detected_contradictions,
        missing_evidence=missing_evidence,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class GapRecord:
    """A minimal, duck-typed stand-in for a future, structured
    `MissingEvidenceRecord` (evidence.py) -- carries only a
    `recommendation` attribute (no `claim_id` / `claim_ids`), used to
    exercise `RecommendationRule`'s "context-wide gap, attributed to
    every claim" branch deterministically, per that rule's own
    documented attribution fallback.
    """

    recommendation: str
