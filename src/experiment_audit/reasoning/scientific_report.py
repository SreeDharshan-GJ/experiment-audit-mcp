"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_report

Defines `ScientificReport`, the public, canonical output of the
Scientific Reasoning Engine: the object every downstream interface --
CLI, Python SDK, MCP tool, IDE plugin, GitHub Action, web dashboard --
is meant to consume, so that none of them has to know how R001-R006
work, in what order they ran, or which optional `RuleContext.metadata`
key a given finding happens to live under.

**Why reporting is a separate layer from reasoning.** Every scientific
question this package can answer was already, completely and
independently, answered before this module runs: `pipeline.py`'s
`ScientificReasoningPipeline` has already executed `MissingEvidenceRule`
(R001) through `RecommendationRule` (R006), in their required order, and
assembled the result into a `pipeline.ScientificReport` -- a fixed
`RuleContext`, an ordered tuple of `RuleResult`s, and a complete
execution trace. That object is already a lossless, fully traceable
record of the reasoning that occurred. What it is *not* is something a
person, or another program with no knowledge of this package's internal
rule-numbering scheme, can consume comfortably: reading it means knowing
that "Scope Findings" live under `by_rule_id("R002")`, that a claim's
confidence adjustment lives under `context.metadata["confidence_adjustment"]`
because `RuleResult` has no dedicated field for it, and that "Scientific
Judgment" is a `dict[str, str]` tucked inside `context.metadata["judgments"]`
because `RuleContext` was never extended with a dedicated field for it
either. Those are correct, deliberate implementation details of the
reasoning layer (see `pipeline.py`'s own docstring for why they are
shaped that way) -- and exactly the kind of detail a *reporting* layer
exists to absorb on a caller's behalf, once, so that every future
interface built on top of this package never has to rediscover it
independently.

Concretely, this module:

- Never re-implements, re-derives, or second-guesses a scientific
  finding. Every fact this module's `summary()`, `statistics()`,
  `to_dict()`, `to_json()`, `to_markdown()`, and `to_text()` expose was
  already present, verbatim, in the `pipeline.ScientificReport` this
  report was built from -- copied, organized, and formatted, never
  recomputed. In particular, this module never adjusts a confidence
  value, never reaches a judgment, never proposes a recommendation, and
  never decides that a contradiction exists; it only presents whatever
  `ConfidenceRule` (R004), `JudgmentRule` (R005), `RecommendationRule`
  (R006), and `ContradictionRule` (R003) already concluded.
- Never inspects raw evidence, raw claims, or raw contradictions beyond
  what is needed to organize and format them. Where a section of this
  report needs to show "the claims," "the evidence," or "the
  contradictions," it reads `RuleContext.claims`, `RuleContext.
  evidence_sequence()`, and `RuleContext.detected_contradictions`
  directly and serializes them via each type's own `to_dict()` -- it
  never opens up a `Claim`, `Evidence`, or `Contradiction` and
  reinterprets its fields.
- Never calls a rule, a `RuleRegistry`, a `RuleEngine`, or
  `ScientificReasoningPipeline.execute` itself. This module's only input
  is an already-produced `pipeline.ScientificReport`; producing one is
  entirely the orchestration layer's responsibility, not this module's.

**Why the report is immutable.** `ScientificReport` is a frozen,
`slots=True` dataclass wrapping a single already-frozen
`pipeline.ScientificReport`, matching this package's convention
throughout (`Evidence`, `Claim`, `Contradiction`, every `RuleResult`,
`pipeline.ScientificReport` itself): a report, once produced, is a
fixed record of one completed reasoning pass, not a mutable object a
caller could accidentally edit findings into after the fact. A caller
wanting an updated report -- because new evidence arrived, a claim was
re-scoped, or a contradiction was resolved -- re-runs the pipeline
against a new `RuleContext` and wraps the new `pipeline.ScientificReport`
in a new `ScientificReport`; nothing here supports patching an existing
report in place. This is also what makes every serialization method on
this class safe to call repeatedly, cache, or hand to several callers
concurrently: nothing about calling `to_markdown()` can ever change what
`to_json()` would return a moment later.

**Why this becomes the public API of Experiment Audit.** Per this
package's architecture --

    Scientific Rules -> Pipeline -> ScientificReport -> CLI -> Python SDK
    -> MCP -> VS Code -> GitHub Actions -> Web Dashboard

-- every interface downstream of the reasoning engine is meant to depend
on this module's `ScientificReport` and nothing beneath it. A CLI prints
`to_text()`; a GitHub Action renders `to_markdown()` into a check
summary; an MCP tool or Python SDK call returns `to_dict()` or
`to_json()`; a web dashboard renders `statistics()` into charts and
`summary()` into a headline. None of those callers needs to import
`pipeline.py`, `rules.py`, or any of the six rule modules directly, and
none of them needs to change when this package's internal rule-ordering,
`metadata` key conventions, or `RuleResult` shape changes -- only this
module does, in one place, because it is the only module that knows
those details are conventions of the reasoning layer rather than a
public contract.

**Architectural constraint, mirrored from every module in this
package.** This module depends only on `pipeline.py` (for `pipeline.
ScientificReport` and `pipeline.RuleExecutionRecord`) and the Python
standard library. It has no dependency on FastMCP, MCP transport,
`server.py`, or any backend implementation, and no direct dependency on
`rules.py`, `claims.py`, `evidence.py`, `contradictions.py`,
`confidence.py`, `judgment.py`, or `recommendation.py` -- every value
this module reads from those modules' types (a `Claim`'s fields, an
`Evidence` bundle's items, a `Contradiction`'s categories, a
`RuleResult`'s reasoning) is read only via that type's own already-tested
`to_dict()`, never by importing the type itself to inspect or reconstruct
it. The one exception is purely for static typing: this module imports
`rules.RuleResult` under `TYPE_CHECKING` so its own method signatures can
be precise, exactly as `rules.py` itself does for `confidence.py` /
`judgment.py` / `claims.py` / `contradictions.py` -- never at runtime.

**Determinism, mandatory.** Every method on `ScientificReport` is a
fixed, literal transformation of the `pipeline.ScientificReport` it
wraps: iterating already-computed tuples and mappings, calling each
already-computed value's own `to_dict()`, and formatting the result into
a string. No method here calls out to a model, reads the wall clock, or
introduces randomness. Re-calling any method on an unchanged
`ScientificReport` always returns the same result; `to_json()` sorts its
keys explicitly for the same reason `pipeline.py`'s `_RULE_SEQUENCE` is a
literal tuple rather than a runtime-computed ordering -- so that
"deterministic JSON serialization" is a structural property of this
class, not a side effect of `dict` insertion order that a future Python
version, or a future field reordering, could silently change.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from experiment_audit.reasoning.pipeline import (
    RuleExecutionRecord,
)
from experiment_audit.reasoning.pipeline import (
    ScientificReport as PipelineScientificReport,
)

if TYPE_CHECKING:
    # Imported only for static typing, mirroring `rules.py`'s own
    # TYPE_CHECKING-only references to its upstream stages. This module
    # never constructs, inspects the internals of, or imports at
    # runtime a `RuleResult` -- it only calls the already-computed
    # instances' own documented methods (`to_dict()`, attribute reads)
    # on objects `pipeline.ScientificReport` already handed it.
    from experiment_audit.reasoning.rules import RuleResult

__all__ = [
    "ScientificReport",
    "ScientificReportError",
]


# ---------------------------------------------------------------------
# Errors
#
# A local, narrow exception type, matching this package's convention of
# one small, purpose-specific error type per module (e.g. `pipeline.py`'s
# `PipelineConfigurationError`, `rules.py`'s registration/evaluation
# errors) rather than raising a bare `ValueError` indistinguishable from
# any other misuse.
# ---------------------------------------------------------------------


class ScientificReportError(ValueError):
    """Raised when `ScientificReport` is constructed from something
    other than an actual `pipeline.ScientificReport`.

    This module performs no reasoning and therefore has nothing
    meaningful to validate about the *content* of a report -- every
    fact it organizes was already validated by the rule that produced
    it, by `RuleResult.__post_init__`, or by `pipeline.py`'s own
    construction-time checks. The one thing worth guarding here is
    structural: that this class is always built from the object its
    entire contract assumes it was built from.
    """


# ---------------------------------------------------------------------
# Rule id -> section name, used only to organize an already-produced
# tuple of RuleResults for display. Not a decision about *order*
# (pipeline.py's `_RULE_SEQUENCE` already fixed that, and this module
# never re-runs or reorders anything) -- only a fixed, literal lookup
# table from a rule's stable `id` to the human-readable section of the
# report its finding belongs in, mirroring pipeline.py's own
# `_EXPECTED_RULE_IDS` convention of naming rule ids as plain module
# constants rather than magic strings scattered through the file.
# ---------------------------------------------------------------------

_MISSING_EVIDENCE_RULE_IDS: tuple[str, ...] = ("R001", "R002")
_SCOPE_RULE_ID = "R002"
_CONTRADICTION_RULE_ID = "R003"
_CONFIDENCE_RULE_ID = "R004"
_JUDGMENT_RULE_ID = "R005"
_RECOMMENDATION_RULE_ID = "R006"

_SECTION_TITLES: Mapping[str, str] = {
    "claims": "Claims",
    "evidence": "Evidence Summary",
    "missing_evidence": "Missing Evidence",
    "scope": "Scope Findings",
    "contradictions": "Contradictions",
    "confidence": "Confidence",
    "judgment": "Scientific Judgment",
    "recommendations": "Recommendations",
    "execution_trace": "Execution Trace",
    "metadata": "Metadata",
}


def _md_table_cell(value: Any) -> str:
    """Render `value` as text safe to place inside a single Markdown
    table cell in `to_markdown()`.

    `to_markdown()`'s tables (Claims, Scientific Judgment) interpolate
    caller-supplied free text -- `Claim.id`, `Claim.subject` -- that
    this module never validates or restricts (per `claims.py`'s own
    contract, both are only required to be non-empty strings). Neither
    a literal `|` nor an embedded newline is escaped before Markdown's
    `| --- | --- |` table syntax, so a claim whose `subject` or `id`
    contains either one silently splits into extra columns or extra
    rows, corrupting every row after it in the same table. This
    function escapes `|` as `\\|` and collapses embedded newlines to a
    single space so one logical cell always renders as exactly one
    cell -- a purely textual escaping, not a reinterpretation of what
    the value means.
    """
    text = str(value)
    text = text.replace("|", "\\|")
    text = " ".join(text.splitlines())
    return text


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of an arbitrary, caller-supplied value
    into something JSON-serializable, for use only on free-form
    `metadata` mappings this module never otherwise interprets.

    Mirrors `evidence.py`'s own `_json_safe` helper: normalizes
    anything with a `to_dict()` method by calling it, recurses into
    plain mappings and sequences, and passes everything else through
    unchanged for `json.dumps`'s own `default=str` fallback to handle.
    This function performs no reasoning and asserts nothing about
    `value`'s meaning -- it only makes free-form data safe to
    serialize, the same guarantee every other `to_dict()` in this
    package already provides for its own typed fields.
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _rule_result_to_dict(result: RuleResult | None) -> dict[str, Any] | None:
    """`result.to_dict()`, or `None` if `result` is `None`.

    A one-line convenience used throughout this module's `to_dict()`
    so every "did this rule run" branch reads the same way -- not a
    place where this module interprets what a `RuleResult` means.
    """
    return result.to_dict() if result is not None else None


def _context_to_dict(context: Any) -> dict[str, Any]:
    """A lossless, JSON-safe serialization of a `RuleContext`, built
    here rather than by delegating to `RuleContext.to_dict()` itself.

    `RuleContext.to_dict()` (`rules.py`, not modifiable by this
    module) assumes every entry of `missing_evidence` carries its own
    `to_dict()` method, matching that field's declared
    `tuple[MissingEvidenceRecord, ...]` type. In practice, once a
    `pipeline.ScientificReasoningPipeline` run has completed,
    `context.missing_evidence` also carries the plain-language `str`
    gaps `MissingEvidenceRule` (R001) and `ScopeRule` (R002) reported
    via `RuleResult.missing_evidence` -- `pipeline.py`'s own
    documented `_advance_context` contract promotes those strings
    directly into this field, and a `str` has no `to_dict()`. Calling
    `RuleContext.to_dict()` on a context that has been through even
    one ordinary pipeline run therefore raises `AttributeError` for
    any report with at least one missing-evidence finding, which this
    module's own `to_dict()` must not do. This function reproduces
    `RuleContext.to_dict()`'s exact field layout, but serializes
    `missing_evidence` (and, for the same reason, `metadata`, which is
    equally free-form) via `_json_safe` instead of assuming a
    `to_dict()` every entry might not have.
    """
    return {
        "evidence": [bundle.to_dict() for bundle in context.evidence_sequence()],
        "observations": context.observations.to_dict(),
        "hypotheses": context.hypotheses.to_dict(),
        "confidence": (context.confidence.to_dict() if context.confidence is not None else None),
        "judgment": (context.judgment.to_dict() if context.judgment is not None else None),
        "claims": context.claims.to_dict() if context.claims is not None else None,
        "scope": context.scope.to_dict() if context.scope is not None else None,
        "detected_contradictions": [
            contradiction.to_dict() for contradiction in context.detected_contradictions
        ],
        "missing_evidence": [_json_safe(gap) for gap in context.missing_evidence],
        "metadata": {key: _json_safe(value) for key, value in context.metadata.items()},
    }


# ---------------------------------------------------------------------
# ScientificReport
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScientificReport:
    """The canonical, public output of Experiment Audit's Scientific
    Reasoning Engine.

    A `ScientificReport` wraps exactly one already-completed
    `pipeline.ScientificReport` and organizes its contents -- Claims,
    Evidence Summary, Missing Evidence, Scope Findings, Contradictions,
    Confidence, Scientific Judgment, Recommendations, Execution Trace,
    and Metadata -- into the shape every downstream interface (CLI,
    Python SDK, MCP, IDE plugin, GitHub Action, web dashboard) is meant
    to consume, via `summary()`, `statistics()`, `to_dict()`,
    `to_json()`, `to_markdown()`, and `to_text()`.

    This class performs no scientific reasoning of any kind -- see this
    module's own docstring for the full boundary. It never modifies a
    finding it was given, never recomputes a confidence value, and
    never generates a recommendation; it only reads what
    `pipeline_report` already contains and presents it.

    Frozen and `slots=True`, matching every other concrete type in this
    package: a `ScientificReport`, once constructed, is a fixed record
    of one reasoning pass. See this module's docstring, "Why the report
    is immutable," for the full rationale.

    Attributes:
        pipeline_report: The already-completed `pipeline.
            ScientificReport` this report organizes and presents. Every
            value exposed by this class's methods is read from this
            object -- directly, or via its `context` and `results` --
            and nothing else.
        generated_at: When this `ScientificReport` was assembled, if the
            caller chooses to record it. `None` by default and never
            set automatically by this module, mirroring `evidence.py`'s
            `Evidence.collected_at` and `contradictions.py`'s
            `Contradiction.detected_at` convention of leaving timestamps
            entirely to the caller rather than a module reading the
            wall clock itself -- consistent with this module's
            determinism requirement.
    """

    pipeline_report: PipelineScientificReport
    generated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_report, PipelineScientificReport):
            raise ScientificReportError(
                "ScientificReport requires a pipeline.ScientificReport instance; "
                f"got {type(self.pipeline_report).__name__}."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline_report(
        cls,
        pipeline_report: PipelineScientificReport,
        *,
        generated_at: datetime | None = None,
    ) -> ScientificReport:
        """Construct a `ScientificReport` from an already-completed
        `pipeline.ScientificReport`.

        A thin, literal, named constructor -- exists only so a caller
        does not need to know this class's field name to build one,
        mirroring this package's general preference for small, named
        convenience constructors (e.g. `evidence.py`'s `Evidence.
        from_run`, `pipeline.py`'s own `ScientificReasoningPipeline.
        build_initial_context`) over requiring every caller to know a
        type's full constructor signature.

        Args:
            pipeline_report: The completed `pipeline.ScientificReport`
                to wrap.
            generated_at: When this report was assembled, if the caller
                chooses to record it. Defaults to `None`.

        Returns:
            A new `ScientificReport` wrapping `pipeline_report`.
        """
        return cls(pipeline_report=pipeline_report, generated_at=generated_at)

    # ------------------------------------------------------------------
    # Section accessors
    #
    # Each accessor below reads one named section of this report
    # directly from `pipeline_report` -- its `context` for input-side
    # concepts (claims, evidence, scope, already-known
    # contradictions/missing-evidence), or `by_rule_id` for a specific
    # rule's finding. None of these methods filters, transforms, or
    # reinterprets what it reads; each is a one-line lookup, named for
    # the report section it serves so the rest of this class (and any
    # caller reading it) never has to know a rule's numeric id to find
    # its output.
    # ------------------------------------------------------------------

    def claims(self) -> tuple[Any, ...]:
        """Every `Claim` (claims.py) this report concerns, in
        `RuleContext.claims`'s own order, or an empty tuple if no
        `ClaimSet` was supplied to this reasoning pass.
        """
        claim_set = self.pipeline_report.context.claims
        if claim_set is None:
            return ()
        return tuple(claim_set)

    def evidence(self) -> tuple[Any, ...]:
        """Every `Evidence` bundle (evidence.py) this report concerns,
        exactly as `RuleContext.evidence_sequence()` normalizes them.
        """
        return self.pipeline_report.context.evidence_sequence()

    def scope(self) -> Any | None:
        """The declared `Scope` (claims.py) of the claim(s) under
        evaluation, or `None` if scope was not declared for this pass.
        """
        return self.pipeline_report.context.scope

    def missing_evidence(self) -> tuple[Any, ...]:
        """Every evidentiary gap on record for this report, in
        `RuleContext.missing_evidence`'s own order -- gaps already
        known when this pass began, plus (per `pipeline.py`'s
        `_advance_context`) every gap `MissingEvidenceRule` (R001) and
        `ScopeRule` (R002) reported while this pipeline ran.
        """
        return tuple(self.pipeline_report.context.missing_evidence)

    def missing_evidence_findings(self) -> tuple[RuleResult, ...]:
        """The `RuleResult`s produced by `MissingEvidenceRule` (R001)
        and `ScopeRule` (R002) -- the two rules whose combined output
        `missing_evidence()` above reports -- in that order, omitting
        either rule that did not apply to this context.
        """
        return tuple(
            result
            for rule_id in _MISSING_EVIDENCE_RULE_IDS
            if (result := self.pipeline_report.by_rule_id(rule_id)) is not None
        )

    def scope_findings(self) -> RuleResult | None:
        """The `RuleResult` produced by `ScopeRule` (R002), or `None`
        if it did not apply to this context.
        """
        return self.pipeline_report.by_rule_id(_SCOPE_RULE_ID)

    def detected_contradictions(self) -> tuple[Any, ...]:
        """Every `Contradiction` (contradictions.py) already recorded
        and carried into this report, in `RuleContext.
        detected_contradictions`'s own order. Per `pipeline.py`'s own
        documentation, `ContradictionRule` (R003) reports newly found
        conflicts only as plain-language strings (see
        `contradiction_findings()` below), so this tuple reflects only
        contradictions that were already known before the pipeline ran.
        """
        return tuple(self.pipeline_report.context.detected_contradictions)

    def contradiction_findings(self) -> RuleResult | None:
        """The `RuleResult` produced by `ContradictionRule` (R003), or
        `None` if it did not apply to this context. Its own
        `.contradictions` field carries any newly-found conflicts as
        plain-language strings.
        """
        return self.pipeline_report.by_rule_id(_CONTRADICTION_RULE_ID)

    def confidence_findings(self) -> RuleResult | None:
        """The `RuleResult` produced by `ConfidenceRule` (R004), or
        `None` if it did not apply to this context.
        """
        return self.pipeline_report.by_rule_id(_CONFIDENCE_RULE_ID)

    def confidence_adjustment(self) -> float | None:
        """The aggregate confidence adjustment `ConfidenceRule` (R004)
        computed, read from `RuleContext.metadata["confidence_adjustment"]`
        per `pipeline.py`'s own documented `_advance_context` contract,
        or `None` if R004 did not apply to this context (and therefore
        never wrote that key).
        """
        return self.pipeline_report.context.metadata.get("confidence_adjustment")

    def judgment_findings(self) -> RuleResult | None:
        """The `RuleResult` produced by `JudgmentRule` (R005), or
        `None` if it did not apply to this context.
        """
        return self.pipeline_report.by_rule_id(_JUDGMENT_RULE_ID)

    def judgments(self) -> Mapping[str, str]:
        """The per-claim judgment mapping `JudgmentRule` (R005)
        produced -- `Claim.id` to judgment string -- read from
        `RuleContext.metadata["judgments"]` per `pipeline.py`'s own
        documented `_advance_context` contract. An empty mapping if
        R005 did not apply to this context, or applied but reported no
        judgments.
        """
        judgments = self.pipeline_report.context.metadata.get("judgments")
        if not judgments:
            return {}
        return dict(judgments)

    def recommendation_findings(self) -> RuleResult | None:
        """The `RuleResult` produced by `RecommendationRule` (R006), or
        `None` if it did not apply to this context.
        """
        return self.pipeline_report.by_rule_id(_RECOMMENDATION_RULE_ID)

    def recommendations(self) -> tuple[str, ...]:
        """Every plain-language recommendation `RecommendationRule`
        (R006) produced, in its own order. An empty tuple if R006 did
        not apply to this context, or applied but recommended nothing.
        """
        result = self.recommendation_findings()
        if result is None:
            return ()
        return tuple(result.recommendations)

    def execution_trace(self) -> tuple[RuleExecutionRecord, ...]:
        """The complete, ordered execution trace `pipeline.py` already
        assembled -- one `RuleExecutionRecord` per rule it considered,
        applied or not. See `pipeline.RuleExecutionRecord`'s own
        docstring for what each entry records.
        """
        return self.pipeline_report.execution_trace

    def metadata(self) -> Mapping[str, Any]:
        """The final `RuleContext.metadata` this pipeline run produced
        -- the original caller-supplied metadata, plus every key
        `pipeline.py`'s `_advance_context` documented itself as
        writing (`confidence_adjustment`, `judgments`).
        """
        return dict(self.pipeline_report.context.metadata)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """Structured, already-computed statistics about this report.

        Every figure below is a literal count, or a `Counter` over an
        already-computed field, of data this report already carries --
        never a new scientific determination. In particular, the
        "confidence distribution" counts each known claim's own
        `Claim.strength` (claims.py, Chapter 2, Section 6 -- set, if at
        all, by whatever Confidence Assessment step ran before this
        report was built) and the "judgment distribution" counts each
        entry of `judgments()`'s already-produced values; this method
        computes neither strength nor judgment itself.

        Returns:
            A dictionary with the following keys:

            - `claim_count`: Number of claims in `claims()`.
            - `evidence_bundle_count`: Number of `Evidence` bundles in
              `evidence()`.
            - `evidence_item_count`: Total number of `EvidenceItem`s
              across every bundle in `evidence()`.
            - `missing_evidence_count`: Number of entries in
              `missing_evidence()`.
            - `known_contradiction_count`: Number of entries in
              `detected_contradictions()` -- contradictions already on
              record before this pipeline run.
            - `newly_reported_contradiction_count`: Number of
              plain-language conflicts `contradiction_findings()`
              reported, `0` if R003 did not apply.
            - `confidence_adjustment`: The value `confidence_adjustment()`
              returns (may be `None`).
            - `confidence_distribution`: A mapping from each observed
              `Claim.strength` value (or `"unassessed"` for a claim with
              `strength is None`) to the number of claims at that
              strength.
            - `judgment_distribution`: A mapping from each distinct
              judgment string in `judgments()` to how many claims
              received it.
            - `recommendation_count`: Number of entries in
              `recommendations()`.
            - `rules_evaluated`: Number of rules whose `applies()`
              returned `True` for this context (i.e. `len(pipeline_report.
              results)`).
            - `rules_triggered`: Number of those rules whose own
              `RuleResult.triggered` is `True`.
        """
        claims = self.claims()
        evidence_bundles = self.evidence()
        evidence_item_count = sum(len(bundle.items) for bundle in evidence_bundles)
        contradiction_result = self.contradiction_findings()
        newly_reported_contradictions = (
            len(contradiction_result.contradictions) if contradiction_result is not None else 0
        )

        confidence_distribution: Counter[str] = Counter()
        for claim in claims:
            strength = getattr(claim, "strength", None)
            label = strength.value if strength is not None else "unassessed"
            confidence_distribution[label] += 1

        judgment_distribution: Counter[str] = Counter(self.judgments().values())

        results = self.pipeline_report.results

        return {
            "claim_count": len(claims),
            "evidence_bundle_count": len(evidence_bundles),
            "evidence_item_count": evidence_item_count,
            "missing_evidence_count": len(self.missing_evidence()),
            "known_contradiction_count": len(self.detected_contradictions()),
            "newly_reported_contradiction_count": newly_reported_contradictions,
            "confidence_adjustment": self.confidence_adjustment(),
            "confidence_distribution": dict(sorted(confidence_distribution.items())),
            "judgment_distribution": dict(sorted(judgment_distribution.items())),
            "recommendation_count": len(self.recommendations()),
            "rules_evaluated": len(results),
            "rules_triggered": len(self.pipeline_report.triggered_results()),
        }

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """A concise, human-readable executive summary of this report.

        Built entirely from `statistics()` and `judgments()` -- a
        handful of sentences stating how many claims and evidence
        bundles this pass concerned, what evidentiary or scope gaps and
        contradictions were found, what the aggregate confidence
        adjustment and per-claim judgments were, and how many
        recommendations followed. States a fact only when the
        underlying section actually has one to report (e.g. omits the
        confidence sentence entirely if `ConfidenceRule` never applied),
        rather than printing a placeholder for an absent finding.

        Returns:
            A short paragraph suitable for a report header, a chat
            reply, or a notification body.
        """
        stats = self.statistics()
        lines: list[str] = []

        claim_word = "claim" if stats["claim_count"] == 1 else "claims"
        evidence_word = "bundle" if stats["evidence_bundle_count"] == 1 else "bundles"
        lines.append(
            f"Reviewed {stats['claim_count']} {claim_word} against "
            f"{stats['evidence_bundle_count']} evidence {evidence_word} "
            f"({stats['evidence_item_count']} evidence item(s))."
        )

        if stats["missing_evidence_count"]:
            gap_word = "gap" if stats["missing_evidence_count"] == 1 else "gaps"
            lines.append(f"Identified {stats['missing_evidence_count']} evidentiary {gap_word}.")
        else:
            lines.append("No evidentiary gaps identified.")

        total_contradictions = (
            stats["known_contradiction_count"] + stats["newly_reported_contradiction_count"]
        )
        if total_contradictions:
            lines.append(
                f"{stats['known_contradiction_count']} known and "
                f"{stats['newly_reported_contradiction_count']} newly reported "
                "contradiction(s) on record."
            )
        else:
            lines.append("No contradictions on record.")

        if self.confidence_findings() is not None:
            adjustment = stats["confidence_adjustment"]
            lines.append(f"Aggregate confidence adjustment: {adjustment:+.2f}.")

        judgments = self.judgments()
        if judgments:
            distribution = ", ".join(
                f"{count} {label}" for label, count in stats["judgment_distribution"].items()
            )
            lines.append(
                f"Scientific judgment reached for {len(judgments)} claim(s): {distribution}."
            )

        if stats["recommendation_count"]:
            rec_word = "recommendation" if stats["recommendation_count"] == 1 else "recommendations"
            lines.append(f"{stats['recommendation_count']} {rec_word} issued.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # to_dict()
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Lossless, JSON-safe serialization of this report.

        Contains two layers, both derived from `pipeline_report`
        without alteration:

        - `"pipeline_report"`: the full `RuleContext`, every
          `RuleResult`, and the complete execution trace, exactly as
          `pipeline.py` produced them -- assembled via this module's
          own `_context_to_dict` (see that function's docstring for
          why this module does not call `pipeline_report.to_dict()`
          directly) plus each `RuleResult` and `RuleExecutionRecord`'s
          own already-tested `to_dict()`. This is what makes this
          serialization *lossless*: nothing this class organizes below
          is information `pipeline_report` did not already contain.
        - `"sections"`: the same information reorganized under the
          named headings this report exposes (`claims`, `evidence`,
          `missing_evidence`, `scope`, `contradictions`, `confidence`,
          `judgment`, `recommendations`, `execution_trace`, `metadata`),
          each built by calling that section's own accessor above and
          that value's own `to_dict()` -- never a re-derivation.
        - `"statistics"`: `statistics()`'s output, included so a
          consumer of `to_dict()` does not need to recompute it.
        - `"generated_at"`: `generated_at.isoformat()` if set, else
          `None`.

        Returns:
            A dictionary safe to pass to `json.dumps` (see `to_json()`).
        """
        generated_at = self.generated_at.isoformat() if self.generated_at else None
        pipeline_results = self.pipeline_report.results
        pipeline_trace = self.pipeline_report.execution_trace
        return {
            "generated_at": generated_at,
            "pipeline_report": {
                "context": _context_to_dict(self.pipeline_report.context),
                "results": [result.to_dict() for result in pipeline_results],
                "execution_trace": [record.to_dict() for record in pipeline_trace],
            },
            "sections": {
                "claims": [claim.to_dict() for claim in self.claims()],
                "evidence": [bundle.to_dict() for bundle in self.evidence()],
                "missing_evidence": {
                    "gaps": [_json_safe(gap) for gap in self.missing_evidence()],
                    "findings": [result.to_dict() for result in self.missing_evidence_findings()],
                },
                "scope": {
                    "declared": scope.to_dict() if (scope := self.scope()) is not None else None,
                    "findings": _rule_result_to_dict(self.scope_findings()),
                },
                "contradictions": {
                    "detected": [
                        contradiction.to_dict() for contradiction in self.detected_contradictions()
                    ],
                    "findings": _rule_result_to_dict(self.contradiction_findings()),
                },
                "confidence": {
                    "adjustment": self.confidence_adjustment(),
                    "findings": _rule_result_to_dict(self.confidence_findings()),
                },
                "judgment": {
                    "judgments": dict(self.judgments()),
                    "findings": _rule_result_to_dict(self.judgment_findings()),
                },
                "recommendations": {
                    "items": list(self.recommendations()),
                    "findings": _rule_result_to_dict(self.recommendation_findings()),
                },
                "execution_trace": [record.to_dict() for record in self.execution_trace()],
                "metadata": {key: _json_safe(value) for key, value in self.metadata().items()},
            },
            "statistics": self.statistics(),
        }

    # ------------------------------------------------------------------
    # to_json()
    # ------------------------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        """Deterministic JSON serialization of this report.

        Serializes `to_dict()` with `sort_keys=True`, so the key order
        of the resulting document depends only on the key names
        themselves -- never on `dict` insertion order, which could
        otherwise vary with, e.g., a future reordering of this class's
        own field construction. `default=str` is passed only as a
        last-resort safety net for a stray, non-`to_dict()`-bearing
        value a caller placed in free-form `metadata` that `_json_safe`
        could not normalize; every value this module itself places into
        `to_dict()` is already JSON-safe.

        Args:
            indent: Passed through to `json.dumps`. Defaults to `2`,
                for readability; pass `None` for a compact, single-line
                document.

        Returns:
            A JSON string equivalent to `to_dict()`.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    # ------------------------------------------------------------------
    # to_markdown()
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """A professional Markdown report, suitable for a GitHub Actions
        check summary, a pull-request comment, or a saved audit
        artifact.

        Renders the same sections `to_dict()` exposes, in the same
        order this module's documentation lists them (Claims, Evidence
        Summary, Missing Evidence, Scope Findings, Contradictions,
        Confidence, Scientific Judgment, Recommendations, Execution
        Trace, Metadata), each under its own `##` heading. A section
        with nothing to report states so explicitly (e.g. "_No
        contradictions on record._") rather than being silently
        omitted, so a reader can trust that an absent finding was
        actually checked for, not merely never rendered.

        Returns:
            A complete Markdown document as a single string.
        """
        stats = self.statistics()
        parts: list[str] = ["# Experiment Audit -- Scientific Report", ""]

        if self.generated_at is not None:
            parts.append(f"_Generated: {self.generated_at.isoformat()}_")
            parts.append("")

        parts.append("## Executive Summary")
        parts.append("")
        parts.append(self.summary())
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['claims']}")
        parts.append("")
        claims = self.claims()
        if claims:
            parts.append("| ID | Subject | Category | Stage | Strength |")
            parts.append("| --- | --- | --- | --- | --- |")
            for claim in claims:
                strength = claim.strength.value if claim.strength is not None else "unassessed"
                parts.append(
                    f"| {_md_table_cell(claim.id)} | {_md_table_cell(claim.subject)} | "
                    f"{claim.category.value} | {claim.lifecycle_stage.value} | {strength} |"
                )
        else:
            parts.append("_No claims recorded for this pass._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['evidence']}")
        parts.append("")
        evidence_bundles = self.evidence()
        parts.append(
            f"{stats['evidence_bundle_count']} evidence bundle(s), "
            f"{stats['evidence_item_count']} evidence item(s) total."
        )
        for bundle in evidence_bundles:
            parts.append(f"- `{bundle.ref}`: {len(bundle.items)} item(s)")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['missing_evidence']}")
        parts.append("")
        gaps = self.missing_evidence()
        if gaps:
            for gap in gaps:
                parts.append(f"- {gap}")
        else:
            parts.append("_No evidentiary gaps identified._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['scope']}")
        parts.append("")
        scope = self.scope()
        scope_findings = self.scope_findings()
        if scope is not None and not scope.is_unspecified():
            for key, value in scope.to_dict().items():
                if value:
                    parts.append(f"- **{key}**: {value}")
        else:
            parts.append("_Scope not declared for this pass._")
        if scope_findings is not None:
            parts.append("")
            parts.append(f"> {scope_findings.reasoning}")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['contradictions']}")
        parts.append("")
        detected = self.detected_contradictions()
        contradiction_result = self.contradiction_findings()
        newly_reported = contradiction_result.contradictions if contradiction_result else ()
        if detected or newly_reported:
            for contradiction in detected:
                parts.append(f"- (known) `{contradiction.id}`: {contradiction.status.value}")
            for description in newly_reported:
                parts.append(f"- (newly reported) {description}")
        else:
            parts.append("_No contradictions on record._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['confidence']}")
        parts.append("")
        confidence_result = self.confidence_findings()
        if confidence_result is not None:
            adjustment = stats["confidence_adjustment"]
            parts.append(f"Aggregate confidence adjustment: `{adjustment:+.2f}`")
            parts.append("")
            parts.append(f"> {confidence_result.reasoning}")
            if stats["confidence_distribution"]:
                parts.append("")
                parts.append("| Strength | Claims |")
                parts.append("| --- | --- |")
                for label, count in stats["confidence_distribution"].items():
                    parts.append(f"| {label} | {count} |")
        else:
            parts.append("_Confidence assessment did not apply to this pass._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['judgment']}")
        parts.append("")
        judgments = self.judgments()
        if judgments:
            parts.append("| Claim | Judgment |")
            parts.append("| --- | --- |")
            for claim_id, judgment in judgments.items():
                parts.append(f"| {_md_table_cell(claim_id)} | {_md_table_cell(judgment)} |")
        else:
            parts.append("_No scientific judgment reached for this pass._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['recommendations']}")
        parts.append("")
        recommendations = self.recommendations()
        if recommendations:
            for recommendation in recommendations:
                parts.append(f"- {recommendation}")
        else:
            parts.append("_No recommendations issued._")
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['execution_trace']}")
        parts.append("")
        parts.append("| Order | Rule | Applied | Triggered |")
        parts.append("| --- | --- | --- | --- |")
        for record in self.execution_trace():
            triggered = record.result.triggered if record.result is not None else "--"
            parts.append(
                f"| {record.order} | {record.rule_id} ({record.rule_name}) | "
                f"{record.applied} | {triggered} |"
            )
        parts.append("")

        parts.append(f"## {_SECTION_TITLES['metadata']}")
        parts.append("")
        metadata = self.metadata()
        if metadata:
            for key, value in metadata.items():
                parts.append(f"- **{key}**: {_json_safe(value)!r}")
        else:
            parts.append("_No metadata recorded._")
        parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # to_text()
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """A plain-text rendering of this report, suitable for CLI
        output (no Markdown syntax, fixed-width friendly).

        Covers the same sections as `to_markdown()`, in the same order,
        using simple headers and indentation instead of Markdown
        formatting.

        Returns:
            A complete plain-text report as a single string.
        """
        stats = self.statistics()
        lines: list[str] = ["EXPERIMENT AUDIT -- SCIENTIFIC REPORT", "=" * 40, ""]

        if self.generated_at is not None:
            lines.append(f"Generated: {self.generated_at.isoformat()}")
            lines.append("")

        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(self.summary())
        lines.append("")

        lines.append(_SECTION_TITLES["claims"].upper())
        lines.append("-" * 40)
        claims = self.claims()
        if claims:
            for claim in claims:
                strength = claim.strength.value if claim.strength is not None else "unassessed"
                lines.append(
                    f"  {claim.id}: {claim.statement} "
                    f"[{claim.category.value}, {claim.lifecycle_stage.value}, "
                    f"{strength}]"
                )
        else:
            lines.append("  No claims recorded for this pass.")
        lines.append("")

        lines.append(_SECTION_TITLES["evidence"].upper())
        lines.append("-" * 40)
        lines.append(
            f"  {stats['evidence_bundle_count']} evidence bundle(s), "
            f"{stats['evidence_item_count']} evidence item(s) total."
        )
        lines.append("")

        lines.append(_SECTION_TITLES["missing_evidence"].upper())
        lines.append("-" * 40)
        gaps = self.missing_evidence()
        if gaps:
            for gap in gaps:
                lines.append(f"  - {gap}")
        else:
            lines.append("  No evidentiary gaps identified.")
        lines.append("")

        lines.append(_SECTION_TITLES["scope"].upper())
        lines.append("-" * 40)
        scope = self.scope()
        scope_findings = self.scope_findings()
        if scope is not None and not scope.is_unspecified():
            for key, value in scope.to_dict().items():
                if value:
                    lines.append(f"  {key}: {value}")
        else:
            lines.append("  Scope not declared for this pass.")
        if scope_findings is not None:
            lines.append(f"  Finding: {scope_findings.reasoning}")
        lines.append("")

        lines.append(_SECTION_TITLES["contradictions"].upper())
        lines.append("-" * 40)
        detected = self.detected_contradictions()
        contradiction_result = self.contradiction_findings()
        newly_reported = contradiction_result.contradictions if contradiction_result else ()
        if detected or newly_reported:
            for contradiction in detected:
                lines.append(f"  - (known) {contradiction.id}: {contradiction.status.value}")
            for description in newly_reported:
                lines.append(f"  - (newly reported) {description}")
        else:
            lines.append("  No contradictions on record.")
        lines.append("")

        lines.append(_SECTION_TITLES["confidence"].upper())
        lines.append("-" * 40)
        confidence_result = self.confidence_findings()
        if confidence_result is not None:
            adjustment = stats["confidence_adjustment"]
            lines.append(f"  Aggregate confidence adjustment: {adjustment:+.2f}")
            lines.append(f"  Reasoning: {confidence_result.reasoning}")
            for label, count in stats["confidence_distribution"].items():
                lines.append(f"    {label}: {count}")
        else:
            lines.append("  Confidence assessment did not apply to this pass.")
        lines.append("")

        lines.append(_SECTION_TITLES["judgment"].upper())
        lines.append("-" * 40)
        judgments = self.judgments()
        if judgments:
            for claim_id, judgment in judgments.items():
                lines.append(f"  {claim_id}: {judgment}")
        else:
            lines.append("  No scientific judgment reached for this pass.")
        lines.append("")

        lines.append(_SECTION_TITLES["recommendations"].upper())
        lines.append("-" * 40)
        recommendations = self.recommendations()
        if recommendations:
            for recommendation in recommendations:
                lines.append(f"  - {recommendation}")
        else:
            lines.append("  No recommendations issued.")
        lines.append("")

        lines.append(_SECTION_TITLES["execution_trace"].upper())
        lines.append("-" * 40)
        for record in self.execution_trace():
            status = "applied" if record.applied else "skipped"
            triggered = ""
            if record.result is not None:
                triggered = " (triggered)" if record.result.triggered else " (not triggered)"
            lines.append(
                f"  {record.order}. {record.rule_id} {record.rule_name}: {status}{triggered}"
            )
        lines.append("")

        lines.append(_SECTION_TITLES["metadata"].upper())
        lines.append("-" * 40)
        metadata = self.metadata()
        if metadata:
            for key, value in metadata.items():
                lines.append(f"  {key}: {_json_safe(value)!r}")
        else:
            lines.append("  No metadata recorded.")
        lines.append("")

        return "\n".join(lines)