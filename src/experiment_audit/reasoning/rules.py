"""
Experiment Audit Scientific Reasoning Engine

Module: rules

Defines the Scientific Rules stage of the reasoning pipeline (Evidence ->
Observations -> Hypotheses -> **Rules** -> Confidence -> Judgment ->
Recommendation, per research/07_reasoning_engine/reasoning-engine.md).
This corresponds to that document's "Contradiction Search" step and to
research/07_reasoning_engine's description of a Scientific Rule Engine:
a deterministic, auditable layer that inspects the evidence, observations,
and hypotheses already on hand and decides which named scientific rules
(e.g. reasoning-rules.md's Rule 001-005) fire, why, and with what effect
on the reasoning that follows.

**Scope, strictly bounded -- this is a framework, not a rule set.** This
module builds the infrastructure a scientific rule runs inside: the base
class every rule inherits from (`ScientificRule`), the read-only bundle
of context a rule receives (`RuleContext`), the traceable result a rule
produces (`RuleResult`), and the deterministic machinery that registers
and executes rules (`RuleRegistry`, `RuleEngine`). It deliberately
contains **no scientific reasoning of its own** -- no rule that inspects
a metric, a hyperparameter, a seed count, or a contradiction actually
lives here. Concretely, per this task's explicit boundary: *what
evidence exists* is `evidence.py`'s job; *what is measurably true about
it* is `observations.py`'s; *why might that be happening* is
`hypotheses.py`'s; *how sure are we* is `confidence.py`'s; *what do we
conclude* is `judgment.py`'s; *what should change* is
`recommendation.py`'s. *Is this a known scientific pattern, and what
does it imply* belongs to a concrete `ScientificRule` subclass, living
in a future module (e.g. `MissingEvidenceRule`, `ScopeRule`,
`ConfidenceRule`, `ContradictionRule`, `RecommendationRule`,
`JudgmentRule`) -- never in this file.

Every abstract hook this module defines (`ScientificRule.applies`,
`ScientificRule.evaluate`) raises `NotImplementedError` in its base
form and carries a docstring describing the *contract* a concrete rule
must satisfy, never a worked example that quietly becomes a de facto
rule. A reviewer should be able to read this file end to end and learn
exactly how rules are declared, registered, and run -- and learn nothing
about overfitting, seed variance, or any other scientific judgment.

**Architectural constraint, mirrored from every other module in this
package.** This module has no dependency on FastMCP, MCP transport,
`server.py`, or any backend implementation, and no dependency outside
the Python standard library. It operates only on the normalized types
already established upstream in this pipeline -- `evidence.py`'s
`Evidence` / `EvidenceItem`, `observations.py`'s `ObservationSet`,
`hypotheses.py`'s `Hypothesis` / `HypothesisSet`, and `models.py`'s
`RunRef` -- imported and reused as-is rather than re-declared, per this
task's explicit instruction to reuse existing structures. `confidence.py`
and `judgment.py`'s `ConfidenceSet` / `JudgmentSet` are referenced only
as optional, already-computed context a rule may consult (see
`RuleContext.confidence` / `RuleContext.judgment` below); importing them
only under `TYPE_CHECKING` -- the same pattern `judgment.py` already
established for its own upstream references -- keeps this module's
runtime import graph a strict superset of nothing but what it actually
executes against. `claims.py`'s `Claim` / `ClaimSet` / `Scope`,
`contradictions.py`'s `Contradiction`, and `evidence.py`'s
`MissingEvidenceRecord` are referenced under the same `TYPE_CHECKING`
discipline (see `RuleContext.claims` / `.scope` /
`.detected_contradictions` / `.missing_evidence` and
`RuleResult.affected_claims` below), for the same reason.

**Determinism, mandatory.** This module never introduces randomness,
wall-clock dependence, or nondeterministic ordering. `RuleRegistry`
preserves registration order exactly; `RuleRegistry.evaluate_all` and
`RuleEngine.evaluate` always visit rules in that order and return
results in that order. Re-running the same `RuleContext` through the
same `RuleRegistry` always yields the same sequence of `RuleResult`s.

**Traceability, mandatory.** Per this task's explicit requirement, every
`RuleResult` a rule produces must record which rule ran (`rule_id`,
`rule_name`), what evidence it inspected (`evidence_used`), what claims
it affected (`affected_claims`), and what reasoning it produced
(`reasoning`) -- enforced structurally by `RuleResult`'s shape, not left
to convention.

**Revision 2.** `Claim` is now a first-class concept, distinct from
`Hypothesis` per Chapter 2's explicit "a claim is not a hypothesis":
`RuleContext.claims` and `RuleResult.affected_claims` both reference it
directly rather than overloading `hypotheses.py`'s types (the field that
previously did so is preserved, correctly retyped, as
`affected_hypotheses`). `RuleResult` also now carries an explicit
`output_category` (see `OutputCategory` below), so a result's place in
the specification's Section 4 output taxonomy is a structural fact
rather than something inferred from which optional fields happen to be
populated. `RuleContext` gains three further optional, structured input
fields -- `scope`, `detected_contradictions`, `missing_evidence` --
covering input categories the specification's Section 3 requires but
that previously had no home other than the untyped, uninterpreted
`metadata` mapping.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from experiment_audit.models import RunRef
from experiment_audit.reasoning.evidence import Evidence, EvidenceItem, EvidenceKind
from experiment_audit.reasoning.hypotheses import Hypothesis, HypothesisSet
from experiment_audit.reasoning.observations import ObservationSet

if TYPE_CHECKING:
    # Imported only for static typing, mirroring judgment.py's own
    # TYPE_CHECKING-only reference to its upstream stages. `rules.py`
    # never inspects a `ConfidenceSet` or `JudgmentSet` internally; it
    # only carries whatever a caller already computed through
    # `RuleContext` so a rule *may* consult it (e.g. a rule that
    # re-evaluates evidence in light of an existing judgment). Neither
    # import is required at runtime.
    # Same pattern, extended to the Claims stage (Chapter 2 of the
    # Experiment Audit methodology). A `Claim` is a distinct concept
    # from a `Hypothesis` -- "a claim is not a hypothesis" is explicit
    # in that chapter -- so it is imported under its own name rather
    # than reusing `hypotheses.py`'s types. `Scope` is imported from
    # the same module since the methodology ties a claim's declared
    # scope directly to the claim itself (Chapter 2, Section 5).
    from experiment_audit.reasoning.claims import Claim, ClaimSet, Scope
    from experiment_audit.reasoning.confidence import ConfidenceSet

    # `Contradiction` corresponds to the Contradiction Engine described
    # in reasoning-engine.md and Chapter 8 of the methodology. Not yet
    # implemented as a concrete module at the time of this revision;
    # imported only under TYPE_CHECKING, exactly like `ConfidenceSet`
    # and `JudgmentSet` above, so this file's runtime import graph does
    # not grow ahead of what actually exists.
    from experiment_audit.reasoning.contradictions import Contradiction

    # `MissingEvidenceRecord` corresponds to the missing-evidence
    # categories defined in Chapter 1, Section 8. Kept in the same
    # module as `Evidence` / `EvidenceItem` since it describes a gap in
    # that same evidentiary record, imported under TYPE_CHECKING for
    # the same reason as the other forward references above.
    from experiment_audit.reasoning.evidence import MissingEvidenceRecord
    from experiment_audit.reasoning.judgment import JudgmentSet


# ---------------------------------------------------------------------
# Errors
#
# Local, narrow exception types, matching this codebase's convention of
# one small hierarchy per module (e.g. `analysis/comparison.py`'s
# `CompareRunsError`) rather than raising bare `Exception` or reusing a
# stdlib exception for a domain-specific failure.
# ---------------------------------------------------------------------


class RuleRegistrationError(ValueError):
    """Raised by `RuleRegistry` for a registration-time misuse.

    Covers registering an object that is not a `ScientificRule`,
    registering two rules that share an `id`, and unregistering or
    looking up an `id` that was never registered. Kept separate from
    `RuleEvaluationError` (below) because a registration failure is a
    programming error discoverable at wiring time, whereas an
    evaluation failure happens while a specific `RuleContext` is being
    processed and needs to carry that context.
    """


class RuleEvaluationError(RuntimeError):
    """Raised when a registered rule's `applies` or `evaluate` raises.

    `RuleRegistry.evaluate_all` and `RuleEngine.evaluate` never let a
    misbehaving rule's raw exception propagate unannotated: per this
    module's traceability requirement, a caller must always be able to
    tell *which rule* failed and *at which hook*, even when the failure
    originates deep inside a concrete `ScientificRule` subclass this
    module has never heard of.

    Attributes:
        rule_id: The `id` of the rule that raised.
        stage: Which hook was being called -- `"applies"` or
            `"evaluate"`.
        original_error: The exception the rule itself raised. Also
            chained via `raise ... from original_error`, so both the
            structured attribute and the standard traceback chain are
            available to a caller.
    """

    def __init__(self, rule_id: str, stage: str, original_error: BaseException) -> None:
        self.rule_id = rule_id
        self.stage = stage
        self.original_error = original_error
        super().__init__(
            f"Rule {rule_id!r} raised {type(original_error).__name__} "
            f"during {stage!r}: {original_error}"
        )


# ---------------------------------------------------------------------
# RuleContext
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a `ScientificRule` is given to reason over.

    A `RuleContext` is assembled once per reasoning pass (typically by
    whatever orchestrates the pipeline stage that runs after
    `hypotheses.py`, e.g. a future concrete wiring of `engine.py`'s
    `ScientificRuleEngine` protocol) and then handed unchanged to every
    registered rule's `applies` and `evaluate` calls. It is frozen: a
    rule reads from it but never mutates it, so two rules evaluated
    against the same context can never observe each other's side
    effects -- a prerequisite for this module's determinism guarantee.

    This class invents no new structures. Every field reuses a type
    already established earlier in the pipeline, per this task's
    explicit instruction to reuse rather than re-declare:

    Attributes:
        evidence: The raw evidence this reasoning pass concerns -- a
            single `Evidence` bundle (evidence.py), or a group of them
            considered together (e.g. a baseline and its ablations),
            matching the same `Evidence | Sequence[Evidence]` union
            `observations.py`'s `ObservationExtractor.extract` and
            `engine.py`'s `ScientificReasoningEngine.review` already
            accept. Most rules should prefer `observations` and
            `hypotheses`, which are cheaper to reason over and already
            traceable; this field is for a rule that genuinely needs
            raw evidence.
        observations: Every `Observation` (observations.py) extracted
            from `evidence` for this pass -- the measurable, literal
            facts a rule pattern-matches over.
        hypotheses: Every `Hypothesis` (hypotheses.py) generated from
            `observations` for this pass -- the candidate explanations
            a rule may confirm, contradict, or otherwise act on. Kept
            as a `HypothesisSet` (not a bare sequence) so a rule gets
            that class's lookup helpers (`by_kind`, `by_subject`, ...)
            for free.
        confidence: The `ConfidenceSet` (confidence.py) already
            computed for this pass, if any. `None` on a first pass,
            since Confidence Estimation runs *after* the Rules stage
            in the canonical pipeline order -- this field exists for
            the less common case of a rule re-evaluating evidence in
            light of an already-computed confidence, and a rule must
            not assume it is populated.
        judgment: The `JudgmentSet` (judgment.py) already reached for
            this pass, if any. Same rationale and "may be `None`"
            caveat as `confidence`.
        claims: The `ClaimSet` (claims.py) this pass concerns, if any --
            the Chapter 2 `Claim`s that `hypotheses` and `observations`
            ultimately bear on. Kept distinct from `hypotheses`
            deliberately: a `Claim` is not a `Hypothesis` (Chapter 2,
            Section 2), and this field must never be populated with
            `Hypothesis` instances standing in for claims. `None` when
            no claim has been formulated yet for this pass (e.g. a
            purely exploratory reasoning pass with no claim attached),
            consistent with `confidence` / `judgment`'s "may be `None`"
            pattern above.
        scope: The declared `Scope` (claims.py) of the claim(s) under
            evaluation, if declared. Kept as a field distinct from
            `claims` itself, per the specification's Section 3, since a
            rule's correct behavior frequently depends on comparing
            declared scope against the actual scope of the evidence
            available, independently of whatever else the claim states.
            `None` when scope has not been declared -- a rule that
            requires scope to proceed must treat that absence as scope
            ambiguity (specification, Section 8), not assume an
            unstated default.
        detected_contradictions: Every `Contradiction` (contradictions.py)
            already recorded and bearing on this pass, in caller-supplied
            order. Per the specification's Section 3, a rule reasoning
            about a claim must take any relevant recorded contradiction
            as input rather than discovering it anew or ignoring it;
            this field is that input. Defaults to an empty tuple, not
            `None`, since "no contradictions recorded" is itself a
            meaningful, iterable-safe value.
        missing_evidence: Every `MissingEvidenceRecord` (evidence.py)
            already identified as an evidentiary gap relevant to this
            pass (Chapter 1, Section 8), in caller-supplied order.
            Distinct from `RuleResult.missing_evidence`: this field is
            an *input* -- a gap already known before this rule runs --
            while `RuleResult.missing_evidence` is a rule's own,
            newly-reported findings. Defaults to an empty tuple for the
            same reason as `detected_contradictions`.
        metadata: Free-form, caller-supplied context that does not fit
            any of the above (e.g. `{"audit_reason": "pre-merge
            review"}`). Never interpreted by this module or by
            `RuleRegistry` / `RuleEngine`. Defaults to an empty
            mapping rather than `None` so a rule can always safely
            call `.get(...)` on it without a `None` check. Concepts
            with a dedicated field above (`claims`, `scope`,
            `detected_contradictions`, `missing_evidence`) must not be
            smuggled through `metadata` instead -- a rule or caller that
            needs one of those concepts should populate the named field
            so it remains inspectable without prior knowledge of that
            rule's private conventions.
    """

    evidence: Evidence | Sequence[Evidence]
    observations: ObservationSet
    hypotheses: HypothesisSet
    confidence: ConfidenceSet | None = None
    judgment: JudgmentSet | None = None
    claims: ClaimSet | None = None
    scope: Scope | None = None
    detected_contradictions: tuple[Contradiction, ...] = ()
    missing_evidence: tuple[MissingEvidenceRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Performance note (audit #8, performance/scalability pass).
    #
    # These private slots are pure memoization storage for the
    # read-only, purely-derived views `evidence_sequence()`,
    # `evidence_items()`, `evidence_items_by_sources()`,
    # `evidence_items_by_kinds()`, `evidence_bundles_by_refs()`, and
    # `resolvable_refs()` compute below. They add no new *information*
    # to `RuleContext` -- everything they hold is 100% reconstructable
    # from `evidence` alone -- and they change no observable behavior:
    # every one of those methods still returns exactly the value it
    # always returned, in exactly the same order, for the same
    # `RuleContext`. They exist solely so that a `RuleContext` built
    # once and consulted by `MissingEvidenceRule`, `ScopeRule`,
    # `ContradictionRule`, and `ConfidenceRule` -- all four of which
    # need this same "which evidence items/bundles belong to which
    # claim / which kind" answer, once per claim -- pay the cost of
    # normalizing, flattening, and indexing `evidence` once per
    # `RuleContext`, not once per call. `init=False` keeps them out of
    # `__init__`'s signature and `__repr__`; `compare=False` keeps them
    # out of the generated `__eq__` (two contexts with the same
    # `evidence` are still equal regardless of which one has already
    # computed and cached its index), matching this dataclass's
    # existing `frozen=True` contract -- the cache is populated via
    # `object.__setattr__`, the same escape hatch a frozen dataclass's
    # own `__post_init__` would use, and never through ordinary
    # attribute assignment.
    _evidence_items_cache: tuple[EvidenceItem, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_sequence_cache: tuple[Evidence, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_position_by_source_cache: dict[RunRef, list[int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_position_by_kind_cache: dict[EvidenceKind, list[int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _bundle_position_by_ref_cache: dict[RunRef, list[int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _resolvable_refs_cache: frozenset[RunRef] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def evidence_sequence(self) -> tuple[Evidence, ...]:
        """`evidence`, normalized to a tuple regardless of whether a
        single `Evidence` bundle or a sequence of them was supplied.

        Mirrors the normalization `observations.py`'s
        `ObservationExtractor.extract` performs internally for the same
        union type, so a rule never has to special-case "one bundle vs.
        many" itself.

        Memoized on this `RuleContext` instance for the same reason
        `evidence_items()` is (see the cache-slots note above):
        `evidence_bundles_by_refs`, `_position_by_kind`, and
        `_bundle_position_by_ref` all call this once per lookup, and
        `ConfidenceRule` performs one such lookup per claim -- without
        memoization, this normalization (trivial for one bundle, an
        O(len(evidence)) `tuple(...)` copy for a sequence) would
        otherwise re-run once per claim rather than once per context.
        """
        if self._evidence_sequence_cache is None:
            if isinstance(self.evidence, Evidence):
                sequence = (self.evidence,)
            else:
                sequence = tuple(self.evidence)
            object.__setattr__(self, "_evidence_sequence_cache", sequence)
        return self._evidence_sequence_cache

    def evidence_items(self) -> tuple[EvidenceItem, ...]:
        """Every `EvidenceItem` (evidence.py) recorded across every
        `Evidence` bundle in this context, in bundle order then
        recorded order.

        A convenience for rules that build `RuleResult.evidence_used`
        directly from raw evidence rather than from `observations`;
        most rules should prefer citing the `Observation`s or
        `Hypothesis`es that already summarize the relevant items.

        Memoized on this `RuleContext` instance (see the cache slots
        declared above): the flattening walk over every bundle's
        `items` runs at most once per context, however many times a
        rule -- or several different rules -- call this method. A
        `RuleContext` is immutable once built, so a cached result can
        never go stale.
        """
        if self._evidence_items_cache is None:
            items: list[EvidenceItem] = []
            for bundle in self.evidence_sequence():
                items.extend(bundle.items)
            object.__setattr__(self, "_evidence_items_cache", tuple(items))
        return self._evidence_items_cache

    def _position_by_source(self) -> dict[RunRef, list[int]]:
        """`evidence_items()`'s indices, grouped by `EvidenceItem.source`,
        each group's index list kept in ascending (i.e. original,
        `evidence_items()`) order. Built once per context and memoized.
        """
        if self._evidence_position_by_source_cache is None:
            index: dict[RunRef, list[int]] = {}
            for position, item in enumerate(self.evidence_items()):
                if item.source is not None:
                    index.setdefault(item.source, []).append(position)
            object.__setattr__(self, "_evidence_position_by_source_cache", index)
        return self._evidence_position_by_source_cache

    def _position_by_kind(self) -> dict[EvidenceKind, list[int]]:
        """`evidence_items()`'s indices, grouped by `EvidenceItem.kind`,
        each group's index list kept in ascending order. Built once per
        context and memoized.
        """
        if self._evidence_position_by_kind_cache is None:
            index: dict[EvidenceKind, list[int]] = {}
            for position, item in enumerate(self.evidence_items()):
                index.setdefault(item.kind, []).append(position)
            object.__setattr__(self, "_evidence_position_by_kind_cache", index)
        return self._evidence_position_by_kind_cache

    def _bundle_position_by_ref(self) -> dict[RunRef, list[int]]:
        """`evidence_sequence()`'s indices, grouped by `Evidence.ref`,
        each group's index list kept in ascending order. Built once per
        context and memoized.
        """
        if self._bundle_position_by_ref_cache is None:
            index: dict[RunRef, list[int]] = {}
            for position, bundle in enumerate(self.evidence_sequence()):
                index.setdefault(bundle.ref, []).append(position)
            object.__setattr__(self, "_bundle_position_by_ref_cache", index)
        return self._bundle_position_by_ref_cache

    def evidence_items_by_sources(self, sources: Iterable[RunRef]) -> tuple[EvidenceItem, ...]:
        """Every `EvidenceItem` in `evidence_items()` whose `source` is
        one of `sources`, in the exact same relative order
        `[item for item in self.evidence_items() if item.source in
        set(sources)]` would produce.

        Equivalent, item-for-item and in the same order, to filtering
        `evidence_items()` directly -- but does so in
        `O(len(sources) + number of matching items)` using a
        per-context, once-built position index (`_position_by_source`),
        instead of `O(len(evidence_items()))`. This is the difference
        between a single rule call and a call made once per claim
        (`MissingEvidenceRule`, `ContradictionRule`, `ConfidenceRule`
        all attribute evidence to a claim this way): re-scanning every
        evidence item for every one of, say, 100,000 claims is the
        dominant cost in a large audit; consulting a small, pre-built
        per-source index is not.
        """
        index = self._position_by_source()
        unique_sources = dict.fromkeys(sources)
        position_lists = [index[source] for source in unique_sources if source in index]
        if not position_lists:
            return ()
        if len(position_lists) == 1:
            positions: Iterable[int] = position_lists[0]
        else:
            # `heapq.merge` over already-sorted per-source position
            # lists reconstructs the exact ascending, de-duplicated-by-
            # construction order `evidence_items()` itself uses, even
            # when a claim's evidence_trace names several distinct
            # sources whose items are interleaved in the original
            # bundle order. A plain concatenation of `index[source]`
            # per source would instead group results by source and
            # silently change this method's output order relative to
            # the direct-filter behavior it must match exactly.
            positions = heapq.merge(*position_lists)
        items = self.evidence_items()
        return tuple(items[position] for position in positions)

    def evidence_items_by_kinds(self, kinds: Iterable[EvidenceKind]) -> tuple[EvidenceItem, ...]:
        """Every `EvidenceItem` in `evidence_items()` whose `kind` is
        one of `kinds`, in the same relative order
        `[item for item in self.evidence_items() if item.kind in
        set(kinds)]` would produce.

        Same rationale, same complexity characteristics, and the same
        `heapq.merge`-based order-preservation as
        `evidence_items_by_sources` above, keyed by `kind` instead of
        `source` -- the lookup `MissingEvidenceRule` and `ScopeRule`
        both perform once per claim per evidence expectation.
        """
        index = self._position_by_kind()
        unique_kinds = dict.fromkeys(kinds)
        position_lists = [index[kind] for kind in unique_kinds if kind in index]
        if not position_lists:
            return ()
        if len(position_lists) == 1:
            positions: Iterable[int] = position_lists[0]
        else:
            positions = heapq.merge(*position_lists)
        items = self.evidence_items()
        return tuple(items[position] for position in positions)

    def evidence_bundles_by_refs(self, refs: Iterable[RunRef]) -> tuple[Evidence, ...]:
        """Every `Evidence` bundle in `evidence_sequence()` whose `ref`
        is one of `refs`, in the same relative order
        `[bundle for bundle in self.evidence_sequence() if bundle.ref
        in set(refs)]` would produce.

        Same rationale as `evidence_items_by_sources`, applied to
        bundles rather than items -- `ConfidenceRule`'s
        `_bundles_for_claim` performs exactly this lookup once per
        claim.
        """
        index = self._bundle_position_by_ref()
        unique_refs = dict.fromkeys(refs)
        position_lists = [index[ref] for ref in unique_refs if ref in index]
        if not position_lists:
            return ()
        if len(position_lists) == 1:
            positions: Iterable[int] = position_lists[0]
        else:
            positions = heapq.merge(*position_lists)
        bundles = self.evidence_sequence()
        return tuple(bundles[position] for position in positions)

    def resolvable_refs(self) -> frozenset[RunRef]:
        """Every distinct `RunRef` this context has an `Evidence`
        bundle for -- i.e. `{bundle.ref for bundle in
        self.evidence_sequence()}`, but derived from the already-built,
        memoized `_bundle_position_by_ref` index rather than
        reconstructing the set from a fresh scan.

        A convenience for a rule checking whether a claim's
        `evidence_trace` fully resolves against this context's
        evidence (`ConfidenceRule`'s `_is_fully_traceable`, Section 3's
        "traceability" read literally as end-to-end link resolution):
        without this, that check would otherwise rebuild the same set
        comprehension over `evidence_sequence()` once per claim.

        The resulting `frozenset` is itself memoized (a second cache
        slot, distinct from `_bundle_position_by_ref_cache`): building
        a `frozenset` from that index's keys is still an
        O(len(resolvable refs)) walk, so without this second cache a
        rule calling `resolvable_refs()` once per claim -- exactly what
        `_is_fully_traceable` does -- would still rebuild that
        `frozenset` from scratch on every call.
        """
        refs = self._resolvable_refs_cache
        if refs is None:
            refs = frozenset(self._bundle_position_by_ref())
            object.__setattr__(self, "_resolvable_refs_cache", refs)
        return refs

    def subjects(self) -> tuple[RunRef, ...]:
        """Every distinct `RunRef` this context concerns, deduplicated
        and in first-seen order, unioned across `observations` and
        `hypotheses`.

        A convenience for rules that need "which run(s) is this pass
        about" without re-walking both sets themselves.
        """
        seen: dict[RunRef, None] = {}
        for observation in self.observations:
            for ref in observation.subjects:
                seen.setdefault(ref, None)
        for hypothesis in self.hypotheses:
            for ref in hypothesis.evidence_trace:
                seen.setdefault(ref, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the `to_dict()`
        convention every concrete type in this package follows.

        `confidence`, `judgment`, `claims`, and `scope` are serialized
        via their own `to_dict()` when present and represented as
        `None` otherwise, so this method never has to import any of
        those modules just to check for absence. `detected_contradictions`
        and `missing_evidence` are serialized item-by-item the same way
        `evidence` already is.
        """
        return {
            "evidence": [bundle.to_dict() for bundle in self.evidence_sequence()],
            "observations": self.observations.to_dict(),
            "hypotheses": self.hypotheses.to_dict(),
            "confidence": self.confidence.to_dict() if self.confidence is not None else None,
            "judgment": self.judgment.to_dict() if self.judgment is not None else None,
            "claims": self.claims.to_dict() if self.claims is not None else None,
            "scope": self.scope.to_dict() if self.scope is not None else None,
            "detected_contradictions": [
                contradiction.to_dict() for contradiction in self.detected_contradictions
            ],
            "missing_evidence": [
                record.to_dict() if hasattr(record, "to_dict") else record
                for record in self.missing_evidence
            ],
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# OutputCategory
# ---------------------------------------------------------------------


class OutputCategory(StrEnum):
    """The specification's Section 4 taxonomy of scientific rule
    outputs, as a closed, structural enum rather than something a
    reader has to infer from which optional `RuleResult` fields happen
    to be populated.

    Every `RuleResult` carries exactly one `output_category`, chosen by
    the rule that produced it. This module does not validate that a
    rule's chosen category actually matches its output's content (that
    would require this framework to understand scientific content it
    deliberately knows nothing about); it only guarantees that some
    category, drawn from this closed set, is always present and
    inspectable.

    Members are ordered least- to most-consequential, mirroring the
    specification's own statement that outputs "are ordered by
    consequence, from Observation and Hypothesis at the least
    consequential end through Judgment and Recommendation at the most
    consequential end." `Missing Evidence`, `Request More Evidence`,
    and `Abstain` sit outside that consequence ordering entirely: each
    is a rule's explicit refusal to produce a more consequential
    output, per the specification's Section 8 ("Rule Failures").

    Values are lowercase strings (not opaque `auto()` integers) so a
    serialized `RuleResult.to_dict()` remains self-describing without
    a separate lookup table.
    """

    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    CONTRADICTION = "contradiction"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    JUDGMENT = "judgment"
    RECOMMENDATION = "recommendation"
    MISSING_EVIDENCE = "missing_evidence"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    ABSTAIN = "abstain"


# ---------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleResult:
    """The fully traceable outcome of evaluating one `ScientificRule`
    against one `RuleContext`.

    Every `ScientificRule.evaluate` call must return exactly one of
    these, whether or not the rule actually fired: a rule that did not
    trigger still returns a `RuleResult` with `triggered=False` and a
    `reasoning` string explaining why, rather than returning `None` or
    raising -- a rule that did not fire is itself a traceable fact
    (per this module's traceability requirement, "what reasoning was
    produced" applies equally to a non-trigger).

    Frozen, matching `Observation` / `Hypothesis` / `Judgment`'s
    convention elsewhere in this package: a result, once produced, is
    not mutated in place. A later re-evaluation produces a new
    `RuleResult`, never an edited one.

    Attributes:
        rule_id: The `id` of the `ScientificRule` that produced this
            result. Always equal to `rule.id` for whichever rule
            called `evaluate`; `RuleRegistry.evaluate_all` does not
            re-derive or override it.
        rule_name: The `name` of the `ScientificRule` that produced
            this result, copied for convenience so a reader inspecting
            a `RuleResult` in isolation (e.g. after serializing to
            JSON) does not need to look the rule up by `rule_id` to
            know what produced it.
        triggered: Whether this rule's condition was met for this
            `RuleContext`. `False` is a legitimate, informative
            result -- see class docstring -- not an error state.
        output_category: Which of the specification's Section 4 output
            categories this result belongs to (see `OutputCategory`).
            Required, not defaulted: per the specification, mislabeling
            an output's consequence is worse than the framework simply
            refusing to guess one on a rule's behalf.
        reasoning: A human-readable account of *why* `triggered` has
            the value it has, referencing the specific evidence,
            observations, or hypotheses that led to that conclusion.
            Never empty: even a non-trigger must say why (e.g. "no
            observation of kind X was present for any subject").
        evidence_used: Every `EvidenceItem` (evidence.py) this rule
            actually inspected in reaching its conclusion -- the
            "what evidence was inspected" half of this module's
            traceability requirement. May be empty for a rule that
            reasons only over `Observation`s or `Hypothesis`es rather
            than raw evidence; never populated with items the rule did
            not actually look at.
        affected_claims: Every `Claim` (claims.py) this result bears
            on -- confirmed, weakened, contradicted, or otherwise
            implicated. The "what claims were affected" half of this
            module's traceability requirement (specification, Section
            9: "Claim"). Deliberately typed as `Claim`, never
            `Hypothesis`: Chapter 2 is explicit that "a claim is not a
            hypothesis," and this field must not be populated with
            `Hypothesis` instances standing in for claims -- see
            `affected_hypotheses` below for that. May be empty for a
            rule whose finding does not bear on any specific claim.
        affected_hypotheses: Every `Hypothesis` (hypotheses.py) this
            result bears on, confirmed, weakened, contradicted, or
            otherwise implicated -- the same relationship
            `affected_claims` describes, but to a `Hypothesis` rather
            than a `Claim`. Kept as a distinct field, rather than
            folded into `affected_claims`, precisely so the two
            concepts are never conflated. May be empty for a rule
            whose finding does not bear on any specific hypothesis.
        confidence_adjustment: A signed, bounded suggestion for how
            this finding should shift downstream confidence scoring,
            in `[-1.0, 1.0]` -- positive strengthens, negative weakens,
            `0.0` (the default) is neutral. This module does not
            interpret the value; `confidence.py` decides how, or
            whether, to fold it into a `ConfidenceAssessment.score`.
            Bounded to the same closed interval `score` itself uses,
            so a downstream consumer never has to renormalize.
        contradictions: Plain-language descriptions of any
            contradictions this rule found between evidence,
            observations, or hypotheses (reasoning-engine.md's
            "Contradiction Search" step). Detecting *what* a
            contradiction is remains a concrete rule's job; this field
            only gives that rule somewhere to report one.
        missing_evidence: Plain-language descriptions of evidence this
            rule expected but did not find while evaluating
            `triggered` (e.g. "no seed information available for
            run X"). Distinct from `reasoning`: `reasoning` explains
            the verdict, `missing_evidence` specifically enumerates
            gaps a later stage can act on without re-parsing prose.
        recommendations: Plain-language suggestions this rule's
            finding implies (e.g. "record a random seed before
            drawing conclusions from this run"). Deliberately `str`,
            not `recommendation.py`'s `Recommendation` dataclass --
            per this module's scope boundary, `rules.py` does not
            depend on `recommendation.py`; a concrete rule that wants
            a structured `Recommendation` may build one upstream of
            that module, using this field's text as input.
        metadata: Free-form, rule-supplied detail that does not fit
            any of the above (e.g. threshold values used, an internal
            sub-check identifier). Never interpreted by this module.
    """

    rule_id: str
    rule_name: str
    triggered: bool
    output_category: OutputCategory
    reasoning: str
    evidence_used: tuple[EvidenceItem, ...] = ()
    affected_claims: tuple[Claim, ...] = ()
    affected_hypotheses: tuple[Hypothesis, ...] = ()
    confidence_adjustment: float = 0.0
    contradictions: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("RuleResult.rule_id must be a non-empty string.")
        if not isinstance(self.output_category, OutputCategory):
            raise ValueError(
                "RuleResult.output_category must be an OutputCategory member, "
                f"got {self.output_category!r}."
            )
        if not self.rule_name:
            raise ValueError("RuleResult.rule_name must be a non-empty string.")
        if not self.reasoning:
            raise ValueError(
                "RuleResult.reasoning must be a non-empty string -- a rule must "
                "explain itself whether or not it triggered."
            )
        if not -1.0 <= self.confidence_adjustment <= 1.0:
            raise ValueError(
                "RuleResult.confidence_adjustment must be in [-1.0, 1.0], got "
                f"{self.confidence_adjustment}."
            )

    def subjects(self) -> tuple[RunRef, ...]:
        """Every distinct `RunRef` this result concerns, deduplicated
        and in first-seen order, unioned across `evidence_used`'s
        sources, `affected_claims`' evidence traces, and
        `affected_hypotheses`' evidence traces.
        """
        seen: dict[RunRef, None] = {}
        for item in self.evidence_used:
            if item.source is not None:
                seen.setdefault(item.source, None)
        for claim in self.affected_claims:
            for ref in claim.evidence_trace:
                seen.setdefault(ref, None)
        for hypothesis in self.affected_hypotheses:
            for ref in hypothesis.evidence_trace:
                seen.setdefault(ref, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "triggered": self.triggered,
            "output_category": self.output_category.value,
            "reasoning": self.reasoning,
            "evidence_used": [item.to_dict() for item in self.evidence_used],
            "affected_claims": [claim.to_dict() for claim in self.affected_claims],
            "affected_hypotheses": [
                hypothesis.to_dict() for hypothesis in self.affected_hypotheses
            ],
            "confidence_adjustment": self.confidence_adjustment,
            "contradictions": list(self.contradictions),
            "missing_evidence": list(self.missing_evidence),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# ScientificRule
# ---------------------------------------------------------------------


class ScientificRule(ABC):
    """Abstract base class every scientific rule inherits from.

    A `ScientificRule` is a named, versioned, deterministic unit of
    scientific reasoning: given a `RuleContext`, it decides whether its
    condition is met (`applies`) and, if so -- or even if not --
    produces a fully traceable `RuleResult` (`evaluate`). This base
    class defines that contract and nothing else. It contains no
    scientific logic; every abstract method below raises
    `NotImplementedError` in words, not in a default implementation
    that happens to return something plausible.

    **Identity.** `id`, `name`, `description`, and `version` are
    abstract read-only properties rather than constructor arguments or
    class attributes, so a concrete rule can compute them however suits
    it (a class constant is the common case, but nothing here forces
    that). `id` is the stable key `RuleRegistry` indexes rules by and
    `RuleResult.rule_id` cites -- it must be unique within any one
    `RuleRegistry` and, per this module's determinism guarantee, must
    not change between runs of the same rule class.

    **Two-phase evaluation.** `applies` is a cheap, side-effect-free
    predicate: "is this rule even relevant to this context?"
    `RuleRegistry.evaluate_all` calls `evaluate` only for rules whose
    `applies` returned `True`, so `evaluate` may assume its own
    applicability condition already holds. Splitting the two lets a
    registry report, cheaply, which rules are relevant to a context
    without paying the cost of full evaluation for the rest.

    **Statelessness.** A `ScientificRule` instance must not accumulate
    state across calls to `applies` or `evaluate` -- every fact a rule
    needs comes from the `RuleContext` argument, and every fact it
    produces goes into the returned `RuleResult`. This is what lets
    `RuleRegistry` evaluate the same rule against many contexts, or
    many rules against the same context, without one call influencing
    another.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """A short, stable, globally-unique identifier for this rule
        (e.g. `"R001"`), analogous to `hypotheses.py`'s internal
        `_RULE_*` identifiers made public and formal. Must never
        change once a rule ships, since `RuleResult.rule_id` and any
        persisted audit trail cite it by value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, human-readable name for this rule (e.g. "Missing
        Baseline Comparison"), suitable for display in a report or log
        line without further formatting.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """A longer, human-readable account of what this rule checks
        for and why it is scientifically relevant -- the rule's own
        analog of this package's module-level "Scope" docstrings, but
        one rule at a time.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        """This rule's version (e.g. `"1.0.0"`), incremented whenever
        its evaluation logic changes in a way that could alter past
        results. Exists so a `RuleResult` -- or anything that persists
        one -- can distinguish "this rule found nothing" from "this
        finding came from a rule version that has since been revised."
        """
        raise NotImplementedError

    @abstractmethod
    def applies(self, context: RuleContext) -> bool:
        """Return whether this rule is relevant to `context`.

        Must be side-effect-free and inexpensive relative to
        `evaluate`: `RuleRegistry.evaluate_all` calls this for every
        registered rule on every `RuleContext`, and only calls
        `evaluate` for rules that return `True` here. A concrete rule
        typically checks for the presence of specific `Observation`
        kinds, `Hypothesis` kinds, or evidence fields in `context`
        before deciding it has anything to say.

        Args:
            context: The evidence, observations, hypotheses, and
                optional prior confidence/judgment this rule may
                consider.

        Returns:
            `True` if `evaluate(context)` should be called; `False`
            if this rule has nothing to contribute for `context`. Must
            be an actual `bool`, not merely a truthy or falsy value:
            `RuleRegistry` enforces this strictly, raising
            `RuleEvaluationError` on a violation, rather than silently
            coercing an unexpected return type via `bool(...)`, since a
            rule returning e.g. `None` by falling off the end of a
            function is a bug worth surfacing, not a quiet `False`.

        Raises:
            NotImplementedError: Always, in this base class. Concrete
                scientific rules must override this method; this
                framework never guesses at applicability.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, context: RuleContext) -> RuleResult:
        """Evaluate this rule against `context` and return a fully
        traceable `RuleResult`.

        Called by `RuleRegistry.evaluate_all` only when
        `self.applies(context)` has already returned `True` -- a
        concrete implementation may rely on that precondition and need
        not re-check it. Must return a `RuleResult` whether or not the
        rule's condition is actually met (see `RuleResult.triggered`'s
        docstring): returning `None`, raising to signal "did not
        trigger", or fabricating an empty-but-`triggered=True` result
        are all contract violations.

        Args:
            context: The evidence, observations, hypotheses, and
                optional prior confidence/judgment this rule reasons
                over.

        Returns:
            A `RuleResult` describing this rule's conclusion for
            `context`, with `rule_id == self.id` and
            `rule_name == self.name`.

        Raises:
            NotImplementedError: Always, in this base class. Concrete
                scientific rules must override this method; this
                framework never performs scientific evaluation itself.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """A one-line, human-readable summary of this rule's identity --
        `"{id} v{version}: {name} -- {description}"`.

        A concrete convenience, not scientific logic: useful for log
        lines, CLI listings, or a registry dump, so callers do not each
        reinvent the same formatting.
        """
        return f"{self.id} v{self.version}: {self.name} -- {self.description}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, version={self.version!r})"


# ---------------------------------------------------------------------
# RuleRegistry
# ---------------------------------------------------------------------


class RuleRegistry:
    """A deterministic, ordered collection of `ScientificRule`s.

    `RuleRegistry` owns exactly one responsibility: knowing which rules
    exist and in what order they should run. It performs no scientific
    reasoning itself -- `evaluate_all` only calls each registered
    rule's own `applies`/`evaluate`, in registration order, and
    collects what they return.

    **Determinism.** Rules are kept in an insertion-ordered mapping
    keyed by `id` (a plain `dict`, whose insertion-order guarantee is
    part of the language since Python 3.7). `list_rules` and
    `evaluate_all` always iterate in that same order; unregistering and
    re-registering a rule moves it to the end, since that is a new
    insertion -- this is documented explicitly in `unregister`'s
    docstring so it is never a surprise. No method in this class sorts,
    shuffles, or otherwise reorders rules by any criterion other than
    "when was this `id` most recently inserted."
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: Iterable[ScientificRule] | None = None) -> None:
        """Create a registry, optionally pre-populated with `rules`.

        Args:
            rules: Rules to register immediately, in the order given.
                Equivalent to calling `register` on each in turn.
                Defaults to an empty registry.
        """
        self._rules: dict[str, ScientificRule] = {}
        if rules is not None:
            for rule in rules:
                self.register(rule)

    def register(self, rule: ScientificRule) -> None:
        """Add `rule` to this registry.

        Args:
            rule: The rule to register. Must be a `ScientificRule`
                instance whose `id` is not already registered.

        Raises:
            RuleRegistrationError: If `rule` is not a `ScientificRule`,
                or if a rule with the same `id` is already registered
                (registering a genuinely updated rule means giving it
                a new `version` and, if its behavior changed enough to
                warrant one, a new `id` -- never silently overwriting
                the old registration).
        """
        if not isinstance(rule, ScientificRule):
            raise RuleRegistrationError(
                f"Cannot register {rule!r}: expected a ScientificRule instance."
            )
        if rule.id in self._rules:
            existing = self._rules[rule.id]
            raise RuleRegistrationError(
                f"A rule with id {rule.id!r} is already registered "
                f"({existing!r}); cannot register {rule!r} under the same id."
            )
        self._rules[rule.id] = rule

    def unregister(self, rule_id: str) -> None:
        """Remove the rule registered under `rule_id`.

        Note that registering a *different* rule under the same `id`
        afterward places it at the end of iteration order, since that
        registration is a new insertion -- `RuleRegistry` does not
        preserve a removed rule's original position for a later
        replacement.

        Args:
            rule_id: The `id` of the rule to remove.

        Raises:
            RuleRegistrationError: If no rule with `rule_id` is
                registered.
        """
        if rule_id not in self._rules:
            raise RuleRegistrationError(f"No rule with id {rule_id!r} is registered.")
        del self._rules[rule_id]

    def list_rules(self) -> tuple[ScientificRule, ...]:
        """Every registered rule, in deterministic registration order."""
        return tuple(self._rules.values())

    def find_rule(self, rule_id: str) -> ScientificRule | None:
        """The rule registered under `rule_id`, or `None` if none is."""
        return self._rules.get(rule_id)

    def evaluate_all(self, context: RuleContext) -> tuple[RuleResult, ...]:
        """Evaluate every registered rule against `context`.

        For each registered rule, in registration order: calls
        `rule.applies(context)`; if that returns `True`, calls
        `rule.evaluate(context)` and collects the result. A rule whose
        `applies` returns `False` contributes no `RuleResult` at all --
        `evaluate_all`'s output is "what every relevant rule
        concluded," not "one entry per registered rule regardless of
        relevance."

        Args:
            context: The `RuleContext` to evaluate every registered
                rule against.

        Returns:
            A tuple of `RuleResult`s, one per rule whose `applies`
            returned `True`, in the same deterministic order
            `list_rules` would report those rules in.

        Raises:
            RuleEvaluationError: If any rule's `applies` or `evaluate`
                raises; if `applies` returns something other than an
                actual `bool`; or if `evaluate` returns something
                other than a `RuleResult`. Wraps and re-raises rather
                than letting a single misbehaving rule's exception
                surface unannotated, or a malformed return value pass
                silently -- see `RuleEvaluationError`'s docstring.
        """
        results: list[RuleResult] = []
        for rule in self._rules.values():
            if self._rule_applies(rule, context):
                results.append(self._rule_evaluate(rule, context))
        return tuple(results)

    def evaluate_one(self, rule_id: str, context: RuleContext) -> RuleResult:
        """Evaluate a single registered rule against `context`,
        unconditionally -- bypassing its own `applies` check.

        Intended for testing, debugging, and interactive tooling where
        a caller has already decided a specific rule should run
        regardless of what `applies` would report, not for normal
        pipeline use (which should call `evaluate_all`, so
        applicability is always honored).

        Args:
            rule_id: The `id` of the rule to evaluate.
            context: The `RuleContext` to evaluate it against.

        Returns:
            That rule's `RuleResult` for `context`.

        Raises:
            RuleRegistrationError: If no rule with `rule_id` is
                registered.
            RuleEvaluationError: If the rule's `evaluate` raises, or
                returns something other than a `RuleResult`.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            raise RuleRegistrationError(f"No rule with id {rule_id!r} is registered.")
        return self._rule_evaluate(rule, context)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_applies(rule: ScientificRule, context: RuleContext) -> bool:
        try:
            result = rule.applies(context)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised typed
            raise RuleEvaluationError(rule.id, "applies", exc) from exc
        # Strictly type-checked, mirroring `_rule_evaluate` below: a rule
        # returning something other than an actual `bool` (e.g. `None`,
        # `""`, `[]`) is a contract violation to surface, not a value to
        # silently coerce via `bool(...)` into a possibly-wrong `False`.
        if not isinstance(result, bool):
            raise RuleEvaluationError(
                rule.id,
                "applies",
                TypeError(f"expected a bool, got {type(result).__name__}"),
            )
        return result

    @staticmethod
    def _rule_evaluate(rule: ScientificRule, context: RuleContext) -> RuleResult:
        try:
            result = rule.evaluate(context)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised typed
            raise RuleEvaluationError(rule.id, "evaluate", exc) from exc
        if not isinstance(result, RuleResult):
            raise RuleEvaluationError(
                rule.id,
                "evaluate",
                TypeError(f"expected a RuleResult, got {type(result).__name__}"),
            )
        return result

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[ScientificRule]:
        return iter(self._rules.values())

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._rules)} rule(s): {list(self._rules)!r})"


# ---------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------


class RuleEngine:
    """Runs every registered scientific rule against a `RuleContext`
    and returns a deterministic, ordered list of `RuleResult`s.

    `RuleEngine` is a thin, intentionally minimal wrapper around a
    `RuleRegistry`: it owns no reasoning of its own and performs no
    orchestration across pipeline stages (assembling `Evidence` into
    `Observation`s, `Observation`s into `Hypothesis`es, or carrying
    `RuleResult`s into `confidence.py` / `judgment.py` is
    `engine.py`'s `ScientificReasoningEngine`'s job, per this module's
    scope boundary -- not this class's). Its entire contract is: given
    a `RuleContext`, consult every registered rule, in a fixed order,
    and report what each relevant one concluded.

    A `RuleEngine` is constructed with either an existing `RuleRegistry`
    (to share one registry across several engines or callers) or a bare
    iterable of rules (the common case, for a single self-contained
    engine) -- never both.
    """

    __slots__ = ("_registry",)

    def __init__(
        self,
        rules: Iterable[ScientificRule] | None = None,
        *,
        registry: RuleRegistry | None = None,
    ) -> None:
        """Create a rule engine.

        Args:
            rules: Rules to register into a new, internally-owned
                `RuleRegistry`. Ignored if `registry` is given.
            registry: An existing `RuleRegistry` to use directly,
                rather than constructing a new one. Useful when several
                `RuleEngine`s (or other callers) should share the same
                set of registered rules. Mutually exclusive with
                `rules`.

        Raises:
            ValueError: If both `rules` and `registry` are given --
                the two are alternative ways to supply the same thing,
                and accepting both silently would leave it ambiguous
                which one wins.
        """
        if rules is not None and registry is not None:
            raise ValueError("RuleEngine accepts either 'rules' or 'registry', not both.")
        self._registry = registry if registry is not None else RuleRegistry(rules)

    @property
    def registry(self) -> RuleRegistry:
        """The `RuleRegistry` this engine consults. Exposed directly so
        a caller can register or unregister rules on a running engine
        without this class needing to re-expose every `RuleRegistry`
        method as a passthrough.
        """
        return self._registry

    def evaluate(
        self, context: RuleContext, *, triggered_only: bool = False
    ) -> tuple[RuleResult, ...]:
        """Run every registered, applicable rule against `context`.

        Args:
            context: The `RuleContext` to evaluate.
            triggered_only: If `True`, omit results whose `triggered`
                is `False` from the returned tuple. Relative order
                among the remaining results is unchanged. Defaults to
                `False`, returning every applicable rule's result
                (triggered or not) -- the fuller, more traceable
                default, per this module's traceability requirement;
                a caller that only wants findings can opt in to the
                narrower view.

        Returns:
            A deterministically ordered tuple of `RuleResult`s, per
            `RuleRegistry.evaluate_all`.

        Raises:
            RuleEvaluationError: Propagated from
                `RuleRegistry.evaluate_all` if any registered rule
                fails during `applies` or `evaluate`.
        """
        results = self._registry.evaluate_all(context)
        if triggered_only:
            return tuple(result for result in results if result.triggered)
        return results

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[ScientificRule]:
        return iter(self._registry)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(registry={self._registry!r})"