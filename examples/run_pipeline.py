"""
Experiment Audit Scientific Reasoning Engine

Example: run_pipeline

The first executable, end-to-end demonstration of Experiment Audit.

This script builds a small but realistic experiment record -- evidence
collected from four training runs, the observations and hypotheses
derived from that evidence, and five scientific claims made about it --
and hands the whole thing to `ScientificReasoningPipeline`. The
pipeline runs all six Scientific Rules, in their required order:

    R001 MissingEvidenceRule    -- Claim C002 (a statistical claim) is
                                    missing the repeated-measurement
                                    evidence its category requires.
    R002 ScopeRule               -- Claim C003 declares a hardware
                                    scope ("H100") its own attributed
                                    evidence does not support (the
                                    evidence was gathered on "A100").
    R003 ContradictionRule       -- Claims C004 and C005 share a
                                    subject and a compatible scope, but
                                    their attributed evidence reports
                                    two different values for the same
                                    metric ("latency_ms": 42 vs. 87) --
                                    a structural contradiction.
    R004 ConfidenceRule          -- Aggregates every claim's
                                    strengthening and reducing factors
                                    into a signed confidence adjustment.
    R005 JudgmentRule            -- Synthesizes R001-R004's findings
                                    into a terminal standing for every
                                    claim.
    R006 RecommendationRule      -- Turns those standings into concrete
                                    next steps.

Claim C001 is deliberately left clean -- adequate evidence for its
category, a declared scope its evidence actually supports, and no part
in any detected conflict -- so the report also has a claim whose record
raised no rule's concern.

No object here is mocked. Every `Evidence`, `Claim`, `Observation`, and
`Hypothesis` is built through that type's own public constructor, and
`ObservationExtractor` / `HypothesisGenerator` are the same pure,
deterministic extractors the rest of the pipeline relies on.

Run with:

    python examples/run_pipeline.py
"""

from __future__ import annotations

from datetime import UTC, datetime

from experiment_audit_mcp.models import Run, RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.hypotheses import HypothesisGenerator
from experiment_audit_mcp.reasoning.observations import ObservationExtractor
from experiment_audit_mcp.reasoning.pipeline import ScientificReasoningPipeline
from experiment_audit_mcp.reasoning.rules import RuleContext
from experiment_audit_mcp.reasoning.scientific_report import ScientificReport

# ---------------------------------------------------------------------
# 1. Evidence
#
# Four runs of a fictional method, "SparseAttn": one accuracy run on
# Benchmark-A (a single seed -- deliberately too few to support a
# statistical claim), one accuracy run on Benchmark-A run a second time
# on different hardware, and two latency runs on Benchmark-B that
# disagree with each other.
# ---------------------------------------------------------------------

ref_accuracy_a100 = RunRef(
    backend="wandb", entity="research-team", project="sparse-attn", run_id="run-001"
)
ref_latency_a100_trial1 = RunRef(
    backend="wandb", entity="research-team", project="sparse-attn", run_id="run-002"
)
ref_latency_a100_trial2 = RunRef(
    backend="wandb", entity="research-team", project="sparse-attn", run_id="run-003"
)
ref_accuracy_h100 = RunRef(
    backend="wandb", entity="research-team", project="sparse-attn", run_id="run-004"
)

evidence_accuracy_a100 = Evidence(
    ref=ref_accuracy_a100,
    run=Run(
        ref=ref_accuracy_a100,
        name="run-001",
        tags=["baseline"],
        status="finished",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config={"model": "sparse-attn-base", "learning_rate": 0.0003},
        summary_metrics={"accuracy": 0.93},
    ),
    seeds=[7],
    hardware={"name": "A100"},
    dataset={"name": "benchmark-a"},
)

evidence_latency_trial1 = Evidence(
    ref=ref_latency_a100_trial1,
    run=Run(
        ref=ref_latency_a100_trial1,
        name="run-002",
        tags=[],
        status="finished",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config={"model": "sparse-attn-base"},
        summary_metrics={"latency_ms": 42.0},
    ),
    hardware={"name": "A100"},
    dataset={"name": "benchmark-b"},
)

evidence_latency_trial2 = Evidence(
    ref=ref_latency_a100_trial2,
    run=Run(
        ref=ref_latency_a100_trial2,
        name="run-003",
        tags=[],
        status="finished",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config={"model": "sparse-attn-base"},
        summary_metrics={"latency_ms": 87.0},
    ),
    hardware={"name": "A100"},
    dataset={"name": "benchmark-b"},
)

evidence_accuracy_h100 = Evidence(
    ref=ref_accuracy_h100,
    run=Run(
        ref=ref_accuracy_h100,
        name="run-004",
        tags=[],
        status="finished",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config={"model": "sparse-attn-base"},
        summary_metrics={"accuracy": 0.81},
    ),
    hardware={"name": "A100"},
    dataset={"name": "benchmark-a"},
)

all_evidence = [
    evidence_accuracy_a100,
    evidence_latency_trial1,
    evidence_latency_trial2,
    evidence_accuracy_h100,
]


# ---------------------------------------------------------------------
# 2. Observations and Hypotheses
#
# Both derived, not hand-written: `ObservationExtractor` reads the
# evidence bundles above, and `HypothesisGenerator` reads whatever
# observations that extraction actually produced. Neither performs any
# scoring, judgment, or recommendation of its own.
# ---------------------------------------------------------------------

observations = ObservationExtractor().extract(all_evidence)
hypotheses = HypothesisGenerator().generate(observations)


# ---------------------------------------------------------------------
# 3. Claims
#
#   C001 -- clean: adequate category evidence, a declared scope its own
#           evidence supports, no part in any conflict.
#   C002 -- missing evidence: a statistical claim, but the evidentiary
#           record has only one seed in total (fewer than the two a
#           statistical claim requires).
#   C003 -- scope limitation: declares its evidence was gathered on
#           "H100" hardware, but the evidence actually attributed to it
#           was gathered on "A100".
#   C004 / C005 -- contradiction: the same subject and a compatible
#           declared scope, but their attributed evidence reports two
#           different latency figures for the same run of the method.
# ---------------------------------------------------------------------

claim_clean_performance = Claim(
    id="C001",
    subject="SparseAttn accuracy on Benchmark-A",
    statement="SparseAttn attains at least 90% accuracy on Benchmark-A.",
    category=ClaimCategory.PERFORMANCE,
    scope=Scope(dataset="benchmark-a", hardware="A100"),
    evidence_trace=(ref_accuracy_a100,),
)

claim_missing_statistical_evidence = Claim(
    id="C002",
    subject="SparseAttn accuracy-gain significance on Benchmark-A",
    statement=(
        "SparseAttn's accuracy gain on Benchmark-A is statistically significant across seeds."
    ),
    category=ClaimCategory.STATISTICAL,
    scope=Scope(dataset="benchmark-a"),
    evidence_trace=(ref_accuracy_a100,),
)

claim_scope_violation = Claim(
    id="C003",
    subject="SparseAttn accuracy on Benchmark-A (H100 replication)",
    statement="SparseAttn attains at least 90% accuracy on Benchmark-A on H100 hardware.",
    category=ClaimCategory.PERFORMANCE,
    scope=Scope(dataset="benchmark-a", hardware="H100"),
    evidence_trace=(ref_accuracy_h100,),
)

claim_latency_42ms = Claim(
    id="C004",
    subject="SparseAttn inference latency on Benchmark-B",
    statement="SparseAttn achieves an inference latency of approximately 42ms on Benchmark-B.",
    category=ClaimCategory.EFFICIENCY,
    scope=Scope(dataset="benchmark-b", hardware="A100"),
    evidence_trace=(ref_latency_a100_trial1,),
)

claim_latency_87ms = Claim(
    id="C005",
    subject="SparseAttn inference latency on Benchmark-B",
    statement="SparseAttn achieves an inference latency of approximately 87ms on Benchmark-B.",
    category=ClaimCategory.EFFICIENCY,
    scope=Scope(dataset="benchmark-b", hardware="A100"),
    evidence_trace=(ref_latency_a100_trial2,),
)

claims = ClaimSet(
    [
        claim_clean_performance,
        claim_missing_statistical_evidence,
        claim_scope_violation,
        claim_latency_42ms,
        claim_latency_87ms,
    ]
)


# ---------------------------------------------------------------------
# 4. RuleContext
#
# Everything R001-R006 need to reason over, assembled directly through
# `RuleContext`'s own public constructor.
# ---------------------------------------------------------------------

context = RuleContext(
    evidence=all_evidence,
    observations=observations,
    hypotheses=hypotheses,
    claims=claims,
)


# ---------------------------------------------------------------------
# 5. Run the pipeline and build the report
# ---------------------------------------------------------------------

pipeline = ScientificReasoningPipeline()
pipeline_report = pipeline.execute(context)
report = ScientificReport.from_pipeline_report(pipeline_report)


# ---------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------

print("=" * 60)
print("Experiment Audit")
print("Executive Summary")
print("=" * 60)
print(report.summary())

print()
print("=" * 60)
print("Full Scientific Report")
print("=" * 60)
print(report.to_text())
