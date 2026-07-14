"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.missing_evidence

Defines `MissingEvidenceRule`, the first concrete `ScientificRule`
(rules.py) in this pipeline. It implements the check named directly in
01_evidence.md, Section 8 ("Missing Evidence") and echoed in
02_claims.md, Section 8 ("Unsupported Claims" -> `missing_evidence`):
whether the evidence already gathered for a claim covers every
evidence dimension that claim's own declared category and scope call
for, and, where it does not, which specific dimension is absent.

**What this module is not.** It is not a confidence estimator, a
verifier, or a judge. Per 01_evidence.md Section 8, "the absence of
evidence is not the same as the absence of information" -- this rule's
entire job is to make that absence explicit and traceable, never to
decide what the absence *means* for the claim's ultimate standing.
Accordingly this rule never touches `RuleResult.confidence_adjustment`
(it is left at its `0.0` default on every `RuleResult` this rule
returns) and never populates `RuleResult.contradictions`: scoring a
gap's consequence is confidence.py's and judgment.py's job (rules.py's
own module docstring, "Never generate judgments. Those belong to later
rules."), not this rule's.

**What "missing" means here, precisely.** A dimension counts as
missing only when it is a *structural absence* the framework can
verify directly from `RuleContext` -- a dataset dimension for a
generalization claim with evidence from only one dataset, a seed count
of fewer than two for a statistical claim, an absent baseline for a
comparison claim, and so on (see `_CATEGORY_EXPECTATIONS`, below).
This rule never infers a gap from what a claim's evidence *might*
plausibly need beyond what its declared `ClaimCategory` (claims.py,
Chapter 2 Section 4) and scope (claims.py's `Scope`, Chapter 2 Section
5) actually commit it to -- a performance claim scoped to one dataset
is not flagged for lacking cross-dataset evidence, because nothing
about a performance claim, as such, asserts anything cross-dataset.
Conversely this rule never invents evidence that was not supplied: an
expectation is satisfied only by evidence actually present in
`RuleContext.evidence`, never assumed, extrapolated, or filled in by
this rule on the claim's behalf.

**Determinism.** Every check this rule performs is a fixed, literal
inspection of `RuleContext` -- counting recorded seeds, counting
distinct dataset identifiers, checking for the presence of an
`EvidenceKind` category, matching a small, fixed vocabulary of
resource-cost metric-name substrings. None of it is probabilistic,
none of it calls out to a model, and re-running this rule against an
unchanged `RuleContext` always yields the same `RuleResult`s.

**Architectural constraint, mirrored from every other module in this
package.** This module depends only on `rules.py` (the framework it
plugs into), `evidence.py` and `claims.py` (the normalized types it
reasons over, reused as-is), and the Python standard library. It has
no dependency on FastMCP, MCP transport, `server.py`, or any backend
implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory
from experiment_audit_mcp.reasoning.evidence import EvidenceItem, EvidenceKind
from experiment_audit_mcp.reasoning.rules import (
    OutputCategory,
    RuleContext,
    RuleResult,
    ScientificRule,
)

# ---------------------------------------------------------------------
# Evidence-dimension checks
#
# Each function below is a narrow, literal predicate over a
# `RuleContext`: "is evidence bearing on dimension X actually present."
# Every function reasons only over `RuleContext.evidence` (via
# `RuleContext.evidence_sequence()` / `evidence_items()`), never over
# `RuleContext.observations` or `RuleContext.hypotheses`, since those
# are already *interpretations* of evidence (per 01_evidence.md,
# Section 2, "Evidence vs. Observation") and this rule's job is to
# check the evidentiary record itself, not someone else's reading of
# it. Each function is deliberately small and literal -- no scoring,
# no thresholds beyond the structural minimum the dimension requires
# (e.g. "at least two seeds," matching 01_evidence.md's own Rule 002
# framing of statistical evidence), no probability.
# ---------------------------------------------------------------------


def _has_performance_evidence(context: RuleContext) -> bool:
    """Whether at least one metric (summary or full curve) was recorded
    for any evidence bundle in `context` -- the minimal evaluation
    evidence a performance claim (claims.py Section 4) requires.
    """
    return any(bundle.metric_names() for bundle in context.evidence_sequence())


def _has_baseline_evidence(context: RuleContext) -> bool:
    """Whether `context` contains evidence of a reference point to
    compare against -- either an explicit prior-experiment link
    (`EvidenceKind.PREVIOUS_EXPERIMENT`) or a second evidence bundle
    alongside the one under study. Either constitutes the baseline
    evidence a comparison claim (claims.py Section 4) requires per
    01_evidence.md Section 4's "Baseline Evidence."
    """
    bundles = context.evidence_sequence()
    if len(bundles) >= 2:
        return True
    return any(bundle.previous_experiments for bundle in bundles)


def _distinct_dataset_identifiers(context: RuleContext) -> set[str]:
    """Every distinct dataset identifier recorded across `context`'s
    evidence bundles, derived only from each bundle's declared
    `dataset` facts -- never inferred from a claim's wording.

    Prefers the conventional `\"name\"` key when a bundle's `dataset`
    dict sets one; otherwise falls back to the full, sorted content of
    that bundle's `dataset` dict as its identifier, so two bundles
    with genuinely different dataset facts are still counted as
    distinct even when neither used the `\"name\"` convention.
    """
    identifiers: set[str] = set()
    for bundle in context.evidence_sequence():
        if not bundle.dataset:
            continue
        name = bundle.dataset.get("name")
        if name is not None:
            identifiers.add(str(name))
        else:
            identifiers.add(repr(sorted(bundle.dataset.items(), key=lambda kv: kv[0])))
    return identifiers


def _has_generalization_evidence(context: RuleContext) -> bool:
    """Whether `context` contains evidence from more than one distinct
    dataset -- the minimal cross-domain coverage a generalization
    claim (claims.py Section 4) requires. A single dataset, however
    thoroughly evaluated, is exactly the case 01_evidence.md's own
    worked example treats as insufficient: evidence of the method
    working on CIFAR-10 alone does not warrant a claim that it
    generalizes.
    """
    return len(_distinct_dataset_identifiers(context)) >= 2


_ROBUSTNESS_KEYWORDS: tuple[str, ...] = (
    "perturb",
    "noise",
    "adversarial",
    "corrupt",
    "shift",
    "ood",
    "out_of_distribution",
)


def _has_robustness_evidence(context: RuleContext) -> bool:
    """Whether `context` contains evidence explicitly recorded under a
    perturbed, noisy, or adversarial condition -- the evidence a
    robustness claim (claims.py Section 4) requires, per that
    section's own statement that such a claim "requires evidence
    collected specifically under the perturbed conditions being
    claimed about, not evidence collected only under nominal
    conditions."

    Checks only `EvidenceKind.DATASET` and `EvidenceKind.CONFIG`
    items' recorded keys and values against a small, fixed keyword
    vocabulary -- a literal presence check, not a probabilistic
    judgment about whether a given configuration is "robust enough."
    """
    for bundle in context.evidence_sequence():
        for item in bundle.items:
            if item.kind not in (EvidenceKind.DATASET, EvidenceKind.CONFIG):
                continue
            haystack = f"{item.key} {item.value}".lower()
            if any(keyword in haystack for keyword in _ROBUSTNESS_KEYWORDS):
                return True
    return False


_RESOURCE_METRIC_KEYWORDS: tuple[str, ...] = (
    "time",
    "latency",
    "throughput",
    "runtime",
    "speed",
    "flops",
    "memory",
    "duration",
)


def _has_resource_cost_evidence(context: RuleContext) -> bool:
    """Whether `context` contains a metric whose name suggests it
    measures resource cost (wall-clock time, latency, throughput,
    memory, ...) rather than result quality -- the runtime-benchmark
    evidence an efficiency claim (claims.py Section 4) requires.

    Matches metric names against a small, fixed keyword vocabulary.
    This is a literal substring check over evidence the caller already
    supplied, not a guess at what the metric measures.
    """
    for bundle in context.evidence_sequence():
        for name in bundle.metric_names():
            lowered = name.lower()
            if any(keyword in lowered for keyword in _RESOURCE_METRIC_KEYWORDS):
                return True
    return False


def _has_hardware_evidence(context: RuleContext) -> bool:
    """Whether `context` contains any recorded hardware fact -- the
    hardware-specification evidence an efficiency claim (claims.py
    Section 4) requires, since a resource-cost measurement is only
    interpretable together with the hardware it was measured on
    (01_evidence.md Section 3, "Contextual").
    """
    return any(bundle.hardware for bundle in context.evidence_sequence())


_SCALE_CONFIG_KEYWORDS: tuple[str, ...] = (
    "model_size",
    "num_params",
    "parameters",
    "batch_size",
    "compute",
    "data_volume",
    "num_agents",
    "scale",
)


def _has_scalability_evidence(context: RuleContext) -> bool:
    """Whether `context` contains evidence at more than one point along
    some declared scaling factor -- the range of evidence a
    scalability claim (claims.py Section 4) requires, per that
    section's own statement that such a claim "requires evidence
    gathered across a range of that factor, not evidence from a single
    point along it."

    Groups each evidence bundle's config values by config key (limited
    to keys matching a small, fixed scaling-factor vocabulary) and
    checks whether any one key took on two or more distinct values
    across bundles -- a literal count over evidence already supplied,
    never an assumption about what the claim's scaling factor is.
    """
    values_by_key: dict[str, set[str]] = {}
    for bundle in context.evidence_sequence():
        if bundle.run is None:
            continue
        for key, value in bundle.run.config.items():
            lowered_key = key.lower()
            if any(keyword in lowered_key for keyword in _SCALE_CONFIG_KEYWORDS):
                values_by_key.setdefault(lowered_key, set()).add(repr(value))
    return any(len(values) >= 2 for values in values_by_key.values())


def _has_statistical_evidence(context: RuleContext) -> bool:
    """Whether `context` records at least two random seeds in total
    across its evidence bundles -- the minimal repeated-measurement
    evidence a statistical claim (claims.py Section 4) requires, per
    01_evidence.md's own Rule 002 framing ("only one random seed ->
    low statistical confidence"). A single recorded seed, however
    favorable its result, cannot distinguish an effect from noise.
    """
    total_seeds = sum(len(bundle.seeds) for bundle in context.evidence_sequence())
    return total_seeds >= 2


def _has_reproducibility_evidence(context: RuleContext) -> bool:
    """Whether `context` contains evidence of an independent
    re-execution of the same nominal experiment -- the reproducibility
    evidence a reproducibility claim (claims.py Section 4) requires,
    per that section's own statement that such a claim "is not
    satisfied by the original result alone, however carefully that
    original result was recorded."

    Proxied by the presence of at least one linked prior experiment
    (`Evidence.previous_experiments`), the same structural signal
    01_evidence.md Section 4 uses for reproducibility evidence: a
    recorded, separate execution this bundle's result can be checked
    against.
    """
    return any(bundle.previous_experiments for bundle in context.evidence_sequence())


# ---------------------------------------------------------------------
# _EvidenceExpectation
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EvidenceExpectation:
    """One named, checkable evidence dimension a claim of a given
    `ClaimCategory` is expected to have supporting evidence for.

    An `_EvidenceExpectation` is the unit `MissingEvidenceRule` reasons
    over: a `check` predicate that inspects a `RuleContext` and
    reports whether the dimension is actually covered, plus the fixed,
    human-readable text this rule reports when it is not. `kinds`
    names which `EvidenceKind` categories are relevant to this
    dimension, so `MissingEvidenceRule.evaluate` can populate
    `RuleResult.evidence_used` with exactly the items it actually
    inspected in reaching its conclusion, per rules.py's traceability
    requirement.

    Attributes:
        name: A short, stable identifier for this expectation (e.g.
            `"baseline_comparison"`), used only for internal grouping;
            never shown to a caller directly.
        missing_description: The plain-language sentence this rule
            reports in `RuleResult.missing_evidence` when `check`
            returns `False` for a given claim.
        recommendation: The plain-language suggestion this rule
            reports in `RuleResult.recommendations` when `check`
            returns `False` for a given claim.
        kinds: Which `EvidenceKind` categories this expectation's
            `check` inspects, for `evidence_used` bookkeeping.
        check: The deterministic predicate itself.
    """

    name: str
    missing_description: str
    recommendation: str
    kinds: tuple[EvidenceKind, ...]
    check: Callable[[RuleContext], bool]


# ---------------------------------------------------------------------
# Category -> expectations
#
# Chapter 2, Section 4's per-category evidence dependencies, and
# Chapter 1, Section 8's named "common forms of missing evidence"
# (missing baselines, missing seeds, missing evaluation metrics,
# missing statistical analysis, missing environment details), given a
# fixed, closed mapping from `ClaimCategory` to the expectations that
# category's own defining section already states it requires. Every
# `ClaimCategory` member has at least one expectation registered here,
# so `MissingEvidenceRule` never has to guess at an unmapped category.
# ---------------------------------------------------------------------

_CATEGORY_EXPECTATIONS: dict[ClaimCategory, tuple[_EvidenceExpectation, ...]] = {
    ClaimCategory.PERFORMANCE: (
        _EvidenceExpectation(
            name="evaluation_metric",
            missing_description=(
                "Evaluation metric evidence quantifying the stated performance objective"
            ),
            recommendation=(
                "Record a summary metric or metric history for the objective this claim asserts."
            ),
            kinds=(EvidenceKind.METRIC, EvidenceKind.CURVE),
            check=_has_performance_evidence,
        ),
    ),
    ClaimCategory.COMPARISON: (
        _EvidenceExpectation(
            name="baseline_comparison",
            missing_description="Baseline or alternative-method evidence to compare against",
            recommendation=(
                "Collect evidence from at least one baseline or alternative run "
                "under matched conditions."
            ),
            kinds=(EvidenceKind.PREVIOUS_EXPERIMENT,),
            check=_has_baseline_evidence,
        ),
    ),
    ClaimCategory.GENERALIZATION: (
        _EvidenceExpectation(
            name="cross_domain_evaluation",
            missing_description="Cross-domain evaluation",
            recommendation=(
                "Evaluate the method on at least one additional dataset or task "
                "outside the one it was evidenced on."
            ),
            kinds=(EvidenceKind.DATASET,),
            check=_has_generalization_evidence,
        ),
        _EvidenceExpectation(
            name="additional_datasets",
            missing_description="Additional datasets",
            recommendation=(
                "Record dataset evidence for each additional dataset the "
                "generalization claim is meant to cover."
            ),
            kinds=(EvidenceKind.DATASET,),
            check=_has_generalization_evidence,
        ),
    ),
    ClaimCategory.ROBUSTNESS: (
        _EvidenceExpectation(
            name="perturbed_condition_evidence",
            missing_description=(
                "Evidence collected under perturbed, noisy, or adversarial conditions"
            ),
            recommendation=(
                "Evaluate the method under the perturbed or adversarial conditions "
                "this claim asserts robustness to, and record that evidence."
            ),
            kinds=(EvidenceKind.DATASET, EvidenceKind.CONFIG),
            check=_has_robustness_evidence,
        ),
    ),
    ClaimCategory.EFFICIENCY: (
        _EvidenceExpectation(
            name="resource_cost_metric",
            missing_description="Resource-cost metric evidence (e.g. runtime benchmark)",
            recommendation=(
                "Record a resource-cost metric (wall-clock time, latency, throughput, "
                "or memory) for the efficiency dimension this claim concerns."
            ),
            kinds=(EvidenceKind.METRIC,),
            check=_has_resource_cost_evidence,
        ),
        _EvidenceExpectation(
            name="hardware_specification",
            missing_description="Hardware specification evidence",
            recommendation=("Record the hardware the resource-cost measurement was taken on."),
            kinds=(EvidenceKind.HARDWARE,),
            check=_has_hardware_evidence,
        ),
    ),
    ClaimCategory.SCALABILITY: (
        _EvidenceExpectation(
            name="scaling_range_evidence",
            missing_description=(
                "Evidence gathered across a range of the scaling factor, rather than "
                "a single point along it"
            ),
            recommendation=(
                "Collect evidence at multiple points along the scaling factor "
                "(e.g. model size, data volume, compute budget) this claim concerns."
            ),
            kinds=(EvidenceKind.CONFIG,),
            check=_has_scalability_evidence,
        ),
    ),
    ClaimCategory.STATISTICAL: (
        _EvidenceExpectation(
            name="repeated_measurement_evidence",
            missing_description=(
                "Statistical evidence across repeated measurement (multiple seeds or runs)"
            ),
            recommendation=(
                "Record results from at least two seeds or repeated runs so an "
                "effect can be distinguished from chance variation."
            ),
            kinds=(EvidenceKind.SEED,),
            check=_has_statistical_evidence,
        ),
    ),
    ClaimCategory.REPRODUCIBILITY: (
        _EvidenceExpectation(
            name="independent_reexecution_evidence",
            missing_description=(
                "Reproducibility evidence from an independent re-execution of the "
                "same nominal experiment"
            ),
            recommendation=(
                "Attempt an independent re-execution of the experiment and record "
                "the resulting reproducibility evidence."
            ),
            kinds=(EvidenceKind.PREVIOUS_EXPERIMENT,),
            check=_has_reproducibility_evidence,
        ),
    ),
}


# ---------------------------------------------------------------------
# MissingEvidenceRule
# ---------------------------------------------------------------------


class MissingEvidenceRule(ScientificRule):
    """Determines whether the evidence already gathered is sufficient
    to support each claim in a `RuleContext`, per 01_evidence.md
    Section 8 and 02_claims.md Section 8.

    For every `Claim` in `RuleContext.claims`, this rule looks up the
    fixed set of evidence dimensions that claim's `ClaimCategory`
    requires (`_CATEGORY_EXPECTATIONS`) and checks each one, literally,
    against the evidence actually present in `RuleContext.evidence`.
    A dimension with no supporting evidence is reported by name in the
    returned `RuleResult.missing_evidence`, alongside a matching
    suggestion in `RuleResult.recommendations`; a claim whose every
    expected dimension is covered contributes nothing to either field.

    This rule never invents evidence that was not supplied, and never
    infers a gap beyond what a claim's own declared category and scope
    commit it to -- see this module's docstring for the precise
    boundary. It never sets `RuleResult.confidence_adjustment` away
    from its `0.0` default and never populates
    `RuleResult.contradictions`: scoring or judging a gap's
    consequence belongs to later rules, per rules.py's own stated
    scope boundary.
    """

    @property
    def id(self) -> str:
        return "R001"

    @property
    def name(self) -> str:
        return "Missing Evidence"

    @property
    def description(self) -> str:
        return (
            "Checks each claim's declared category against the evidence "
            "already gathered for it, and reports any evidence dimension "
            "that category expects but that evidence does not cover."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to check.

        With no claims present, there is nothing this rule's evidence
        dimensions could possibly be checked against, so it declines
        to run rather than reporting on evidence in the abstract.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Check every claim in `context.claims` against its category's
        expected evidence dimensions and report what, if anything, is
        missing.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.
        """
        assert context.claims is not None  # guaranteed by applies()

        missing_evidence: list[str] = []
        recommendations: list[str] = []
        evidence_used: list[EvidenceItem] = []
        seen_evidence_ids: set[int] = set()
        affected_claims: list[Claim] = []
        reasoning_lines: list[str] = []

        # Performance note (audit #8): `_EvidenceExpectation.check` and
        # `expectation.kinds` (via `_items_of_kinds`) both take only
        # `context`, never `claim` -- per `_EvidenceExpectation`'s own
        # docstring, `check` "inspects a `RuleContext`", full stop. Every
        # claim of the same `ClaimCategory` therefore shares the exact
        # same `expectation.check(context)` verdict and the exact same
        # `_items_of_kinds(context, expectation.kinds)` result, since
        # `_CATEGORY_EXPECTATIONS` hands out the same `_EvidenceExpectation`
        # instance to every claim of a given category rather than building
        # a fresh one per claim. Without this cache, both were recomputed
        # from scratch for every claim that shared a category -- on a
        # large audit with many claims of the same category, the dominant
        # cost. `id(expectation)` is a stable, safe cache key here because
        # `_CATEGORY_EXPECTATIONS` is a fixed module-level mapping: the
        # same expectation object is handed out every time, for the
        # lifetime of this `evaluate()` call and beyond.
        #
        # `recorded_expectations` closes a second, subtler instance of the
        # same problem: caching `items` avoids rebuilding that list, but
        # without this guard the `seen_evidence_ids` dedup loop below would
        # still walk the *entire* (now-cached, but still O(len(items)))
        # list again for every claim that shares the expectation -- a
        # no-op after the first claim, since every item is already in
        # `seen_evidence_ids`, but still O(len(items)) wasted work per
        # claim. With many claims sharing a category, and `items` itself
        # growing with the size of the audit, that no-op walk is exactly
        # the O(claims x evidence) behavior this whole cache exists to
        # eliminate. Skipping it once an expectation's items have already
        # been folded into `evidence_used` changes nothing about the
        # result -- every one of those walks was already contributing
        # zero new items -- only how many times it's performed.
        expectation_items_cache: dict[int, tuple[EvidenceItem, ...]] = {}
        expectation_check_cache: dict[int, bool] = {}
        recorded_expectations: set[int] = set()

        for claim in context.claims:
            affected_claims.append(claim)
            expectations = _CATEGORY_EXPECTATIONS.get(claim.category, ())
            claim_missing: list[str] = []

            for expectation in expectations:
                key = id(expectation)
                items = expectation_items_cache.get(key)
                if items is None:
                    items = tuple(_items_of_kinds(context, expectation.kinds))
                    expectation_items_cache[key] = items
                if key not in recorded_expectations:
                    recorded_expectations.add(key)
                    for item in items:
                        if id(item) not in seen_evidence_ids:
                            seen_evidence_ids.add(id(item))
                            evidence_used.append(item)

                check_passed = expectation_check_cache.get(key)
                if check_passed is None:
                    check_passed = expectation.check(context)
                    expectation_check_cache[key] = check_passed
                if check_passed:
                    continue

                claim_missing.append(expectation.missing_description)
                missing_evidence.append(
                    f"Claim {claim.id} ({claim.subject!r}): {expectation.missing_description}"
                )
                recommendations.append(f"Claim {claim.id}: {expectation.recommendation}")

            if claim_missing:
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.category.value}) is missing: "
                    + "; ".join(claim_missing)
                    + "."
                )
            else:
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.category.value}) has evidence "
                    "covering every dimension its category expects."
                )

        triggered = len(missing_evidence) > 0
        reasoning = (
            " ".join(reasoning_lines)
            if reasoning_lines
            else ("No claims were available to check for missing evidence.")
        )

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=triggered,
            output_category=OutputCategory.MISSING_EVIDENCE,
            reasoning=reasoning,
            evidence_used=tuple(evidence_used),
            affected_claims=tuple(affected_claims),
            missing_evidence=tuple(missing_evidence),
            recommendations=tuple(recommendations),
        )


def _items_of_kinds(context: RuleContext, kinds: tuple[EvidenceKind, ...]) -> list[EvidenceItem]:
    """Every `EvidenceItem` in `context` whose `kind` is in `kinds`.

    A small local helper so `MissingEvidenceRule.evaluate` can populate
    `RuleResult.evidence_used` with exactly the items each expectation
    actually inspected, without each expectation re-implementing the
    same filter over `context.evidence_items()`.

    Delegates to `RuleContext.evidence_items_by_kinds`, which answers
    this same filter using a per-context, once-built index rather than
    a fresh `O(len(evidence_items()))` scan. `evaluate` below calls
    this once per claim per category expectation, so on a large audit
    that indexing is the difference between one full evidence scan and
    one full evidence scan per claim per expectation.
    """
    return list(context.evidence_items_by_kinds(kinds))


__all__ = ["MissingEvidenceRule"]