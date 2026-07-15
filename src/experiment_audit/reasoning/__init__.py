"""experiment_audit.reasoning: the Scientific Reasoning Engine.

This package is the public, importable entry point for the reasoning
engine. It contains two independent pipelines built for different
purposes; both are exported here rather than one being hidden behind
the other, since they answer different questions and neither is a
strict subset of the other.

``ScientificReasoningPipeline`` (see ``pipeline.py``) is the concrete,
fixed six-rule pipeline: it takes ``Claim``s and ``EvidenceItem``s and
runs them through ``MissingEvidenceRule -> ScopeRule ->
ContradictionRule -> ConfidenceRule -> JudgmentRule ->
RecommendationRule`` to produce a ``ScientificReport``. This is the
pipeline that performs claim/contradiction analysis and structured
report generation.

``ScientificReasoningEngine`` (see ``engine.py``) is a separate,
generic, injectable pipeline: ``Evidence -> Observations -> Hypotheses
-> (rule engine) -> Confidence -> Judgment -> Recommendation``. It is
independent of ``ScientificReasoningPipeline`` and does not require
``Claim``s.

Example (concrete six-rule pipeline)::

    from experiment_audit.reasoning import (
        ScientificReasoningPipeline,
        ScientificReport,
        Claim, ClaimCategory, Scope,
        EvidenceItem, EvidenceKind,
    )

    pipeline = ScientificReasoningPipeline()
    context = pipeline.build_initial_context(
        claims=[...],      # Sequence[Claim]
        evidence=[...],    # Sequence[EvidenceItem]
    )
    pipeline_report = pipeline.execute(context)
    report = ScientificReport.from_pipeline_report(pipeline_report)
    print(report.to_markdown())
"""

from __future__ import annotations

from experiment_audit.reasoning.claims import (
    Claim,
    ClaimCategory,
    ClaimLifecycleStage,
    ClaimSet,
    ClaimStrength,
    Scope,
    UnsupportedReason,
)
from experiment_audit.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionSet,
)
from experiment_audit.reasoning.engine import ScientificReasoningEngine
from experiment_audit.reasoning.evidence import Evidence, EvidenceItem, EvidenceKind
from experiment_audit.reasoning.pipeline import (
    PipelineConfigurationError,
    RuleContext,
    ScientificReasoningPipeline,
)
from experiment_audit.reasoning.scientific_report import ScientificReport

__all__ = [
    # Concrete six-rule pipeline (claims/evidence -> report)
    "ScientificReasoningPipeline",
    "RuleContext",
    "PipelineConfigurationError",
    "ScientificReport",
    "Claim",
    "ClaimSet",
    "ClaimCategory",
    "ClaimLifecycleStage",
    "ClaimStrength",
    "Scope",
    "UnsupportedReason",
    "Contradiction",
    "ContradictionSet",
    "ContradictionCategory",
    "EvidenceItem",
    "EvidenceKind",
    # Generic injectable pipeline (evidence -> observations -> ... )
    "ScientificReasoningEngine",
    "Evidence",
]
