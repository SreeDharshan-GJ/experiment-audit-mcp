"""
Experiment Audit Scientific Reasoning Engine

Module: pipeline

Defines `ScientificReasoningPipeline`, the orchestration layer that runs
the six concrete `ScientificRule` implementations (`scientific_rules`)
in their required order and assembles their output into one
`ScientificReport`.

**Why orchestration is a separate layer from reasoning.** Every
scientific question this pipeline can answer -- whether evidence a
claim's category expects is missing (R001, `MissingEvidenceRule`),
whether a claim's evidence matches its own declared scope (R002,
`ScopeRule`), whether claims or evidence items genuinely conflict
(R003, `ContradictionRule`), how strongly a claim is, on balance,
justified (R004, `ConfidenceRule`), what a claim's current scientific
standing is (R005, `JudgmentRule`), and what should be done about it
(R006, `RecommendationRule`) -- is already answered, completely and
independently, inside that rule's own `evaluate` method. Each of those
six modules is explicit, in its own docstring, that it depends on
nothing but `rules.py` and the normalized domain types (`claims.py`,
`evidence.py`, `contradictions.py`), and never on any sibling rule
module's internals. That independence is only meaningful, however, if
something else is responsible for actually running the six rules in
the right sequence and handing each one what the rule before it
produced. This module is that "something else," and nothing more.

Concretely, this module:

- Never inspects `Evidence` or `EvidenceItem` content itself -- it never
  calls `RuleContext.evidence_sequence()` / `RuleContext.evidence_items()`
  and reads nothing out of them.
- Never inspects `Claim` content itself -- it never reads a claim's
  `category`, `scope`, `statement`, or `evidence_trace`.
- Never computes a confidence adjustment, detects a contradiction,
  reaches a judgment, or proposes a recommendation. Every one of those
  values in the final `ScientificReport` was produced by calling
  `rule.evaluate(context)` on the appropriate rule -- this module only
  decides *when* to make that call and *what to do with the result*
  once it has been produced elsewhere.
- Never invents a value not already present in a `RuleResult`. Where
  this module updates the running `RuleContext` between rules (see
  `_advance_context`, below), it only ever copies a value a upstream
  `RuleResult` already computed into the field or `metadata` key a
  downstream rule's own documented contract says it will read from --
  it never transforms, scores, or reinterprets that value on the way.

Keeping this boundary strict is what makes the six rule modules
independently testable and independently replaceable: a future revision
to, say, `ConfidenceRule`'s aggregation weights can change freely
without this module changing at all, because this module never knew
*how* `ConfidenceRule` computed its adjustment -- only that it would
produce one, in a documented shape, that a later rule's documented
contract says how to consume.

**Why rule order matters.** The six rules are not interchangeable or
independently orderable; each rule after the first is written, by its
own explicit documentation, to *consume* one or more of the fields the
rules before it populate:

1. **R001 (`MissingEvidenceRule`)** and **R002 (`ScopeRule`)** are the
   only two rules that can run with nothing but the caller's original
   claims and evidence -- neither depends on any other rule's output.
   Both report their findings as `RuleResult.missing_evidence`, and, per
   `scope.py`'s own docstring, both use the *same* `OutputCategory.
   MISSING_EVIDENCE` for exactly this reason: so their combined output
   can be consumed by later rules as a single, undifferentiated stream
   of "evidentiary gaps," without a later rule needing to know or care
   which of the two produced any given gap.
2. **R003 (`ContradictionRule`)** runs next because it, too, needs only
   the original claims and evidence (plus whatever contradictions were
   already known coming in) -- it does not depend on R001 or R002's
   findings. It is ordered here, before R004, simply to match the
   specification's own consequence ordering (Missing Evidence and Scope
   before Contradiction before Confidence), not because of a data
   dependency on R001/R002.
3. **R004 (`ConfidenceRule`)** is explicit, in its own docstring, that
   it is "an aggregator, not a fourth independent structural check": it
   reads `RuleContext.missing_evidence` (R001 + R002's combined output)
   and `RuleContext.detected_contradictions` (R003's relevant input/
   carry-forward) to compute its penalties, and therefore cannot
   meaningfully run before both of those have already executed.
4. **R005 (`JudgmentRule`)** is explicit that it performs "no scientific
   detection of its own": it synthesizes R001-R004's already-closed
   findings, and documents a specific `RuleContext.metadata` contract
   (`confidence_adjustments` / `confidence_adjustment`) for reading
   R004's output, since `RuleContext` has no dedicated field for a
   per-claim confidence figure. It cannot run before R004 has produced
   that value for this module to place into `metadata`.
5. **R006 (`RecommendationRule`)** is explicit that it depends on
   R001-R005's combined findings -- missing-evidence/scope gaps (R001/
   R002), unresolved contradictions (R003), confidence (R004, via the
   same `metadata` contract as R005), and judgment (R005, via
   `metadata["judgments"]`, which R005 itself documents as the channel
   it writes to precisely so R006 can read it). It must run last.

Running the rules in any other order would mean a downstream rule
consuming a field or `metadata` key that has not yet been populated --
not a wrong answer, but no traceable answer at all, which the
specification's own Section 1 ("a conclusion that cannot be
independently regenerated from its stated inputs has not ... been shown
to follow from those inputs at all") treats as a failure in its own
right. This module's fixed `_RULE_SEQUENCE` exists to make that ordering
a structural property of the pipeline, not a convention a caller could
accidentally violate.

**Why this layer is deterministic.** Every step this module performs is
a fixed, literal transformation of already-computed inputs: iterating a
fixed tuple of rule classes in a fixed order, calling each rule's own
`applies()` and `evaluate()` methods exactly once per run, and copying
specific, named fields out of a `RuleResult` into specific, named
locations on the next `RuleContext` -- never a computation over
`Evidence`, `Claim`, or any other raw domain content. This module never
calls out to a model, never branches on wall-clock time, and never
introduces randomness of any kind; re-running `execute()` against an
unchanged input `RuleContext` and an unchanged set of rule instances
always produces the same `ScientificReport`, field for field. The one
piece of caller-supplied nondeterminism this module explicitly accepts
-- an optional `detected_at`-style timestamp on a `ScientificReport`,
were one ever added -- is deliberately *not* included here, mirroring
`evidence.py`'s `Evidence.collected_at` and `contradictions.py`'s
`Contradiction.detected_at` convention of leaving timestamps entirely to
the caller rather than this module reading the clock itself.

**Architectural constraint, mirrored from every module in this
package.** This module depends only on `rules.py` (the framework every
rule plugs into) and the six concrete `ScientificRule` implementations
in `scientific_rules` (`MissingEvidenceRule`, `ScopeRule`,
`ContradictionRule`, `ConfidenceRule`, `JudgmentRule`,
`RecommendationRule`), plus the Python standard library. It has no
dependency on FastMCP, MCP transport, `server.py`, or any backend
implementation, and no dependency on `claims.py`, `evidence.py`, or
`contradictions.py` directly -- this module never constructs or
inspects a `Claim`, `Evidence`, or `Contradiction` itself; it only
passes an already-built `RuleContext` (which may reference those types)
from one rule to the next.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from experiment_audit_mcp.reasoning.rules import RuleContext, RuleResult, ScientificRule
from experiment_audit_mcp.reasoning.scientific_rules import (
    ConfidenceRule,
    ContradictionRule,
    JudgmentRule,
    MissingEvidenceRule,
    RecommendationRule,
    ScopeRule,
)

# ---------------------------------------------------------------------
# Fixed, deterministic rule sequence
#
# The one and only place in this module where rule *order* is decided.
# Declared as a plain tuple of classes -- not discovered via a registry
# scan, not sorted by any runtime-computed key -- so the execution order
# is a structural fact of this module's source, exactly as
# `claims.py`'s `ClaimLifecycleStage` fixes Chapter 2, Section 3's
# lifecycle order by declaration order rather than by a computed sort.
# See this module's own docstring, "Why rule order matters," for the
# data-dependency justification behind this exact sequence.
# ---------------------------------------------------------------------

_RULE_SEQUENCE: tuple[type[ScientificRule], ...] = (
    MissingEvidenceRule,
    ScopeRule,
    ContradictionRule,
    ConfidenceRule,
    JudgmentRule,
    RecommendationRule,
)

#: The rule `id` each position in `_RULE_SEQUENCE` is expected to
#: report, in order. Used only as a construction-time sanity check
#: (`ScientificReasoningPipeline.__init__`) that a caller-supplied
#: custom rule sequence has not silently dropped or reordered one of
#: the six required stages -- never used to *decide* the order itself,
#: which `_RULE_SEQUENCE` above already fixes.
_EXPECTED_RULE_IDS: tuple[str, ...] = ("R001", "R002", "R003", "R004", "R005", "R006")


class PipelineConfigurationError(ValueError):
    """Raised when `ScientificReasoningPipeline` is constructed with a
    rule sequence that does not match `_EXPECTED_RULE_IDS`.

    A narrow, local exception type, matching this package's convention
    of one small, purpose-specific error type per module (e.g.
    `rules.py`'s own registration/evaluation errors,
    `contradictions.py`'s `ContradictionError`) rather than raising a
    bare `ValueError` indistinguishable from any other misuse.
    """


# ---------------------------------------------------------------------
# RuleExecutionRecord
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RuleExecutionRecord:
    """One entry in a `ScientificReport`'s execution trace: exactly what
    happened when this pipeline considered running one rule, in one
    position of the fixed sequence.

    Recorded for *every* rule this pipeline considers, whether or not
    that rule actually ran -- a rule skipped because its own
    `applies(context)` returned `False` is exactly as traceable a fact
    as a rule that ran and produced a `RuleResult`, per this package's
    general traceability discipline (e.g. `evidence.py`'s "evidence is
    never discarded," applied here to "no execution step is ever
    silently omitted from the trace").

    Attributes:
        order: This rule's fixed position in `_RULE_SEQUENCE`, starting
            at `1` -- the same order `_RULE_SEQUENCE` itself declares,
            recorded here so a reader of the trace alone (without
            re-consulting this module's source) can confirm the
            pipeline ran its rules in the required sequence.
        rule_id: The `ScientificRule.id` of the rule this record
            concerns (e.g. `"R001"`).
        rule_name: The `ScientificRule.name` of the rule this record
            concerns (e.g. `"Missing Evidence"`).
        applied: Whether this rule's own `applies(context)` returned
            `True` for the `RuleContext` it was offered. When `False`,
            `result` is always `None` -- a rule that declined to apply
            was never asked to `evaluate` at all, per `ScientificRule`'s
            own two-phase `applies` / `evaluate` contract.
        result: The `RuleResult` this rule's `evaluate(context)` call
            produced, when `applied` is `True`. `None` when `applied` is
            `False`.
    """

    order: int
    rule_id: str
    rule_name: str
    applied: bool
    result: RuleResult | None

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the
        `to_dict()` convention every concrete type in this package
        follows.
        """
        return {
            "order": self.order,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "applied": self.applied,
            "result": self.result.to_dict() if self.result is not None else None,
        }


# ---------------------------------------------------------------------
# ScientificReport
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScientificReport:
    """The complete, traceable output of one full pipeline run.

    A `ScientificReport` bundles three things together, all already
    fully computed by the time this module constructs one: the final
    `RuleContext` (the original context plus every documented,
    field-for-field update this module applied on a downstream rule's
    behalf between stages -- see `ScientificReasoningPipeline.
    _advance_context`), every `RuleResult` produced by a rule that
    actually ran, and the full, ordered `execution_trace` covering
    every rule this pipeline considered, applied or not.

    Frozen, matching this package's convention elsewhere (`Evidence`,
    `Claim`, `Contradiction`, every `RuleResult`): a `ScientificReport`,
    once produced, is not mutated in place. A caller wanting to
    "re-run" the pipeline with new information constructs a new
    `RuleContext` and calls `ScientificReasoningPipeline.execute` again,
    producing a new, independent `ScientificReport`.

    Attributes:
        context: The `RuleContext` as it stood after the last rule in
            `_RULE_SEQUENCE` finished running (or declined to apply) --
            the original input `RuleContext`, plus every update
            documented in `ScientificReasoningPipeline._advance_context`.
        results: Every `RuleResult` produced by a rule whose
            `applies(context)` returned `True`, in the fixed order
            `_RULE_SEQUENCE` declares. A rule that did not apply
            contributes no entry here (see `execution_trace` for a
            record of that fact instead).
        execution_trace: One `RuleExecutionRecord` per rule in
            `_RULE_SEQUENCE`, in order, covering every rule this
            pipeline considered -- whether or not it actually ran.
    """

    context: RuleContext
    results: tuple[RuleResult, ...]
    execution_trace: tuple[RuleExecutionRecord, ...]

    def by_rule_id(self, rule_id: str) -> RuleResult | None:
        """The `RuleResult` produced by the rule with `rule_id`, or
        `None` if no rule with that id ran (either because no such
        rule was in this pipeline's sequence, or because it declined to
        apply).
        """
        for result in self.results:
            if result.rule_id == rule_id:
                return result
        return None

    def triggered_results(self) -> tuple[RuleResult, ...]:
        """Every `RuleResult` in `results` whose own `triggered` flag is
        `True`, in this report's order -- a convenience filter over
        already-computed results, not a new determination.
        """
        return tuple(result for result in self.results if result.triggered)

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the
        `to_dict()` convention every concrete type in this package
        follows.
        """
        return {
            "context": self.context.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "execution_trace": [record.to_dict() for record in self.execution_trace],
        }


# ---------------------------------------------------------------------
# ScientificReasoningPipeline
# ---------------------------------------------------------------------


class ScientificReasoningPipeline:
    """Orchestrates R001-R006 in their required, fixed order and
    assembles a `ScientificReport`.

    This class performs no scientific reasoning of any kind -- see this
    module's own docstring for the full boundary. Its entire
    responsibility is:

    1. Accept, or construct, the initial `RuleContext` a pipeline run
       starts from.
    2. Walk `_RULE_SEQUENCE` (or a caller-supplied, validated
       substitute -- see `__init__`) in order, and for each rule:
       a. Call that rule's own `applies(context)`.
       b. If `False`, record a skipped `RuleExecutionRecord` and move
          on -- this pipeline never overrides a rule's own decision
          about whether it is relevant to the current `RuleContext`.
       c. If `True`, call that rule's own `evaluate(context)`, record
          the resulting `RuleResult`, and -- only where a *documented,
          named* upstream-to-downstream contract exists (see
          `_advance_context`) -- copy specific fields of that result
          into the `RuleContext` the next rule will see.
    3. Return one `ScientificReport` bundling the final `RuleContext`,
       every `RuleResult` produced, and the complete execution trace.

    Every context update this class performs is a literal copy of a
    value a rule already computed into the field or `metadata` key that
    rule's own module documents a downstream rule as reading from --
    never a transformation, an inference, or a value this class
    computes itself. Where no such documented contract exists (e.g.
    `ContradictionRule`'s newly detected conflicts, which its own
    docstring is explicit are reported only as plain-language strings
    in `RuleResult.contradictions` and never constructed into a
    `contradictions.py` `Contradiction` instance), this class leaves the
    corresponding `RuleContext` field untouched rather than inventing a
    conversion no rule module asked for.
    """

    def __init__(self, rules: Sequence[ScientificRule] | None = None) -> None:
        """Construct a pipeline over a fixed sequence of rule instances.

        Args:
            rules: The rule instances to run, in the order they should
                run. Defaults to one fresh instance of each class in
                `_RULE_SEQUENCE`, constructed here. A caller may supply
                its own sequence (e.g. pre-configured rule instances,
                or a sequence built for testing), but that sequence's
                rule `id`s must exactly match `_EXPECTED_RULE_IDS`, in
                order -- this class validates that invariant at
                construction time rather than silently running whatever
                sequence it is given, since running these six rules out
                of order (or with one missing) would break the very
                data dependencies this module's own docstring documents
                as the reason this fixed order is required.

        Raises:
            PipelineConfigurationError: If `rules` is supplied and its
                rule `id`s, in order, do not exactly match
                `_EXPECTED_RULE_IDS`.
        """
        if rules is None:
            self._rules: tuple[ScientificRule, ...] = tuple(
                rule_cls() for rule_cls in _RULE_SEQUENCE
            )
        else:
            self._rules = tuple(rules)

        actual_ids = tuple(rule.id for rule in self._rules)
        if actual_ids != _EXPECTED_RULE_IDS:
            raise PipelineConfigurationError(
                "ScientificReasoningPipeline requires rules R001-R006, in "
                f"that exact order; got {actual_ids!r}."
            )

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_initial_context(
        *,
        claims: Any = None,
        evidence: Any = None,
        detected_contradictions: Any = (),
        missing_evidence: Any = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RuleContext:
        """Construct the `RuleContext` a pipeline run starts from.

        A thin, literal pass-through to `RuleContext`'s own constructor
        -- this method does not validate, normalize, or interpret any
        of its arguments beyond forwarding them, and does not decide
        what a "valid" claim, evidence, or contradiction collection
        looks like. It exists only so a caller does not need to import
        `RuleContext` directly to start a pipeline run, mirroring this
        package's general preference for small, named convenience
        constructors (e.g. `evidence.py`'s `Evidence.from_run`) over
        requiring every caller to know a type's full constructor
        signature.

        Args:
            claims: The claims this pipeline run should reason over,
                exactly as `RuleContext` itself accepts them.
            evidence: The evidence this pipeline run should reason
                over, exactly as `RuleContext` itself accepts it.
            detected_contradictions: Any already-known contradictions
                (e.g. from a prior pipeline run, or supplied directly by
                a caller) this run should carry forward, exactly as
                `RuleContext` itself accepts them. Defaults to an empty
                collection.
            missing_evidence: Any already-known missing-evidence
                findings this run should carry forward. Defaults to an
                empty collection.
            metadata: Free-form, caller-supplied context forwarded
                as-is to `RuleContext.metadata`. Defaults to an empty
                mapping, matching this package's "absence is itself a
                meaningful, safely-`.get()`-able value" convention.

        Returns:
            A new `RuleContext` built from the given arguments.
        """
        return RuleContext(
            claims=claims,
            evidence=evidence,
            detected_contradictions=detected_contradictions,
            missing_evidence=missing_evidence,
            metadata=dict(metadata) if metadata is not None else {},
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: RuleContext) -> ScientificReport:
        """Run every rule in this pipeline's fixed sequence against
        `context`, advancing the context between stages per each
        downstream rule's own documented contract, and return the
        resulting `ScientificReport`.

        Args:
            context: The initial `RuleContext` to reason over -- either
                built directly, or via `build_initial_context`.

        Returns:
            A `ScientificReport` bundling the final `RuleContext`,
            every `RuleResult` a rule actually produced, and the
            complete, ordered execution trace.
        """
        current_context = context
        results: list[RuleResult] = []
        trace: list[RuleExecutionRecord] = []

        for order, rule in enumerate(self._rules, start=1):
            if not rule.applies(current_context):
                trace.append(
                    RuleExecutionRecord(
                        order=order,
                        rule_id=rule.id,
                        rule_name=rule.name,
                        applied=False,
                        result=None,
                    )
                )
                continue

            result = rule.evaluate(current_context)
            results.append(result)
            trace.append(
                RuleExecutionRecord(
                    order=order,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    result=result,
                )
            )

            current_context = self._advance_context(current_context, rule.id, result)

        return ScientificReport(
            context=current_context,
            results=tuple(results),
            execution_trace=tuple(trace),
        )

    # ------------------------------------------------------------------
    # Context advancement between stages
    #
    # The one place in this class where a rule's output is fed forward
    # into the context the next rule will see. Every branch below
    # implements exactly one documented upstream-produces /
    # downstream-consumes contract already stated in the corresponding
    # rule module's own docstring -- this method adds no new contract
    # of its own and performs no computation over the values it copies.
    # ------------------------------------------------------------------

    @staticmethod
    def _advance_context(context: RuleContext, rule_id: str, result: RuleResult) -> RuleContext:
        """Return a `RuleContext` reflecting `result` per `rule_id`'s
        own documented downstream contract, or `context` unchanged if
        that rule defines no such contract.

        - **R001 / R002** (`MissingEvidenceRule`, `ScopeRule`): both
          report their findings via `RuleResult.missing_evidence`, using
          the same `OutputCategory.MISSING_EVIDENCE` precisely so their
          combined output can be read, undifferentiated, as
          `RuleContext.missing_evidence` by `ConfidenceRule`,
          `JudgmentRule`, and `RecommendationRule` (all three read this
          field by the same documented duck-typed attribution rule).
          This method appends `result.missing_evidence` to whatever
          `context.missing_evidence` already held, preserving order.
        - **R003** (`ContradictionRule`): reports newly detected
          conflicts as plain-language strings in `RuleResult.contradictions`,
          and its own docstring is explicit that it "never constructs...
          a `contradictions.py` `Contradiction` instance." There is
          therefore no documented field this method can advance
          `RuleContext.detected_contradictions` with from this rule's
          output; that field continues to reflect only whatever
          contradictions were already known when the pipeline's
          `RuleContext` was first built. This method does, however,
          forward the subset of `result.contradictions` that are *not*
          already-known, carried-forward findings (those begin with
          the literal prefix `"Contradiction "`; see `ContradictionRule`'s
          "Known, carried-forward contradictions" check) into
          `context.metadata["newly_detected_contradictions"]`, appending
          to whatever that key already held. `ConfidenceRule` reads this
          key, alongside `RuleContext.detected_contradictions`, so a
          contradiction detected during *this* pass is not silently
          invisible to confidence scoring -- see `confidence_rule.py`'s
          `_new_contradiction_claim_ids` / `_contradiction_counts_for_claim`
          for the consuming side of this contract. Carried-forward
          findings are deliberately excluded here precisely because they
          are already counted via `RuleContext.detected_contradictions`
          itself; forwarding them too would double-count them.
        - **R004** (`ConfidenceRule`): reports one aggregate
          `RuleResult.confidence_adjustment`. `JudgmentRule` and
          `RecommendationRule` both document reading this exact value
          from `context.metadata["confidence_adjustment"]` as their
          documented fallback source (used when no more precise
          per-claim `"confidence_adjustments"` mapping is present).
          This method writes `result.confidence_adjustment` into that
          key.
        - **R005** (`JudgmentRule`): reports its per-claim judgments in
          `RuleResult.metadata["judgments"]`, a mapping from `Claim.id`
          to a judgment string, and its own docstring states this is
          the channel `RecommendationRule` reads via
          `context.metadata["judgments"]`. This method copies that
          mapping into `context.metadata` under the same key.
        - **R006** (`RecommendationRule`): the terminal rule; no
          downstream rule consumes its output within this pipeline, so
          this method defines no advancement for it.
        """
        updates: dict[str, Any] = {}

        if rule_id in ("R001", "R002") and result.missing_evidence:
            updates["missing_evidence"] = tuple(context.missing_evidence) + tuple(
                result.missing_evidence
            )

        if rule_id == "R003" and result.contradictions:
            new_findings = tuple(
                text for text in result.contradictions if not text.startswith("Contradiction ")
            )
            if new_findings:
                metadata = dict(context.metadata)
                metadata["newly_detected_contradictions"] = (
                    tuple(metadata.get("newly_detected_contradictions", ())) + new_findings
                )
                updates["metadata"] = metadata

        if rule_id == "R004":
            metadata = dict(context.metadata)
            metadata["confidence_adjustment"] = result.confidence_adjustment
            metadata["confidence_adjustments"] = result.metadata.get("confidence_adjustments", {})
            updates["metadata"] = metadata

        if rule_id == "R005":
            judgments = result.metadata.get("judgments") if result.metadata else None
            if judgments:
                metadata = dict(updates.get("metadata", context.metadata))
                metadata["judgments"] = judgments
                updates["metadata"] = metadata

        if not updates:
            return context
        return dataclasses.replace(context, **updates)


__all__ = [
    "ScientificReasoningPipeline",
    "ScientificReport",
    "RuleExecutionRecord",
    "PipelineConfigurationError",
]
