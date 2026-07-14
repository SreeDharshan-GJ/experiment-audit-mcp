# 1. Purpose

Evidence (Chapter 1) is what a claim (Chapter 2) is answerable to, but
evidence attached to a claim does not, by itself, settle how much weight
that claim deserves. Two claims can each have evidence linked to them and
still merit very different degrees of trust — one supported by a single,
narrow, unreplicated run, the other by evidence spanning multiple seeds,
multiple independent sources, and a successful reproduction attempt. A
methodology that stops at "evidence exists" without assessing how strongly
that evidence justifies the claim has not finished the scientific task; it
has only completed the bookkeeping stage of it.

Confidence is what closes that gap. It is the assessment of how strongly
the available evidence, taken as a whole and evaluated against the
properties defined in Chapter 1, justifies accepting a given claim. Without
a confidence assessment, every claim with any linked evidence would appear
equally warranted, which is false in essentially every real case and which
would make Experiment Audit unable to distinguish a well-established result
from a suggestive but preliminary one. Confidence assessment is therefore
not an optional refinement layered on top of claim evaluation — it is the
mechanism by which claim evaluation becomes actionable, since a claim's
strength position (Chapter 2, Section 6) is meaningless without a
disciplined account of what justifies placing it there.

--------------------------------------------------

# 2. Definition of Scientific Confidence

**Scientific confidence** is an assessment of how strongly the available
evidence justifies accepting a claim, as evaluated against defined
properties of that evidence rather than against any internal sense of
certainty held by the party making the assessment.

This definition excludes several things confidence is commonly, and
incorrectly, conflated with. Scientific confidence is not a probability
estimate over outcomes; it is not a numeric score produced by a statistical
model; and it is not a measure of how certain any computational system is
about its own output. Confidence, as defined here, is entirely external to
the party assessing it — it is a property of the relationship between a
claim and the evidence offered in its support, not a property of the
assessor's internal state.

Confidence is built from the properties of evidence quality already
established in Chapter 1 — completeness, consistency, freshness,
reliability, coverage, repeatability, and integrity — together with the
degree to which the evidence has actually been reproduced (Chapter 1's
reproducibility evidence category) and the degree to which independent
sources agree. It is not built from the volume of evidence alone, a
distinction Chapter 1 already establishes for evidence and Chapter 2
already establishes for claim strength, and which applies with equal force
here: a confidence assessment that increases simply because more evidence
items exist, without regard to their quality or independence, is not a
scientific confidence assessment under this methodology.

--------------------------------------------------

# 3. Sources of Confidence

The following factors contribute to justified confidence in a claim. Each
addresses a distinct way in which evidence can make a claim more
trustworthy, and none is sufficient on its own.

**Evidence quality.** Confidence draws directly on the quality dimensions
defined in Chapter 1. Evidence that is complete, consistent, fresh,
reliably sourced, well covered, repeatable, and intact contributes more to
justified confidence than evidence that is technically present but weak
along these dimensions.

**Independent evidence.** Confidence is strengthened when supporting
evidence comes from sources that did not influence one another's
production, per the independence property defined in Chapter 1. Ten
metric readings from a single run contribute far less independent support
than the same number of readings drawn from separately configured,
separately executed experiments, because the former share a single point
of possible failure that the latter do not.

**Reproducibility.** Confidence is strengthened specifically by
reproducibility evidence — a documented, independent attempt to regenerate
a result. A claim supported only by its original measurement, however
carefully that measurement was recorded, has not yet been tested against
the possibility that the original result was itself an artifact of a
particular execution.

**Statistical support.** Confidence is strengthened when a claim is backed
by statistical evidence establishing that an observed effect is
distinguishable from the variation expected across repeated measurement,
rather than resting on a single point estimate that could plausibly have
arisen from chance.

**Consistency.** Confidence is strengthened when the evidence bearing on a
claim agrees with itself — different runs, different evaluation passes,
and different sources all pointing toward the same conclusion, rather than
producing a scattered or contradictory picture.

**Coverage.** Confidence is strengthened when evidence spans the range of
conditions the claim's scope (Chapter 2, Section 5) actually implies —
multiple seeds, multiple data subsets, multiple environments — rather than
resting on a narrow slice of those conditions.

**Traceability.** Confidence is strengthened when the evidentiary chain
behind a claim can be followed without gaps, consistent with the
traceability requirement introduced in Chapter 1 and Chapter 2. A claim
whose supporting evidence cannot be traced to its origin cannot be
confidently assessed, regardless of how favorable that evidence appears on
its face.

**Agreement across evidence sources.** Confidence is strengthened when
evidence collected from genuinely different sources — different
backends, different evaluation protocols, different execution
environments — converges on the same conclusion. Agreement across sources
that share no common origin is a stronger signal than agreement among
evidence items that are, on inspection, derived from the same underlying
measurement.

--------------------------------------------------

# 4. Factors that Reduce Confidence

The following factors act against justified confidence, independently of
whatever positive evidence a claim may also have.

**Missing evidence.** Per Chapter 1's treatment of absence as itself
meaningful, a claim missing an evidence category its own category
(Chapter 2, Section 4) would require — a comparison claim without baseline
evidence, a statistical claim without statistical evidence — cannot be
assessed as though that gap did not exist.

**Conflicting evidence.** Evidence items that contradict one another,
rather than simply being weak, leave a claim's evidentiary position
genuinely unresolved. Conflicting evidence reduces confidence more sharply
than an equivalent quantity of merely absent evidence, because it
indicates an active disagreement in the record rather than an
unaddressed gap.

**Weak baselines.** A comparison or performance claim assessed against a
baseline that is poorly matched, outdated, or otherwise not representative
of a fair reference point cannot support the same confidence as one
assessed against a rigorously matched baseline, even if every other
aspect of the evidence is strong.

**Single-seed evaluation.** A claim resting on a single run, without
statistical evidence across repeated initialization, cannot rule out the
possibility that the observed result reflects the particular seed rather
than the method itself. This is among the most common and most
underweighted sources of unjustified confidence in machine learning
research.

**Poor reproducibility.** A documented reproduction attempt that fails to
regenerate the original result, or that regenerates it only partially,
reduces confidence directly — and does so more severely than the mere
absence of a reproduction attempt, since it constitutes positive evidence
against the claim's durability rather than an unaddressed gap.

**Limited scope.** Evidence gathered under a narrow set of conditions
relative to the scope the claim actually asserts (Chapter 2, Section 5)
cannot justify the same confidence as evidence gathered across the full
range that scope implies, even where the evidence within that narrow slice
is otherwise strong.

**Incomplete reporting.** Evidence associated with a claim that omits
context necessary to interpret it correctly — missing configuration,
missing environment details, missing operational history — reduces
confidence because it prevents the evidence quality dimensions of Chapter
1 from being properly assessed in the first place, independent of what the
reported values themselves show.

--------------------------------------------------

# 5. Confidence Levels

Experiment Audit recognizes the following conceptual levels of confidence,
ordered from weakest to strongest. These levels describe a scientific
judgment about the relationship between a claim and its evidence; they are
not numeric scores and this methodology does not assign thresholds,
percentages, or scoring formulas to them.

**Very Low.** The evidence available is minimal, unreliable, in conflict
with itself, or largely absent relative to what the claim requires. A
claim at this level should be treated as an unsupported assertion
(Chapter 2, Section 8) rather than as a scientifically warranted position,
regardless of how the claim is phrased.

**Low.** Some relevant evidence exists, but it is narrow in coverage,
drawn from a single source, unreplicated, or otherwise insufficient to
rule out the more mundane explanations — chance variation, a
non-representative run, an unmatched baseline — for the observed result.
A claim at this level warrants continued investigation, not adoption.

**Moderate.** The evidence is relevant, reasonably consistent, and covers
a meaningful portion of the claim's scope, but has not yet been
independently reproduced, or has been reproduced only partially, and has
not yet accumulated the breadth of independent sources associated with
higher confidence. A claim at this level is a reasonable working position
but should not be treated as settled.

**High.** The evidence is consistent, reasonably broad in coverage, drawn
from more than one independent source, and has withstood at least one
genuine attempt to reproduce or contradict it. A claim at this level can
serve as a basis for further work, while remaining open, in principle, to
revision by new evidence.

**Very High.** The evidence is extensive, consistent across genuinely
independent sources, has been reproduced under conditions matching the
claim's stated scope, and has been actively tested against plausible
alternative explanations without those tests overturning it. A claim at
this level functions, for practical purposes, as an established position,
while remaining formally revisable if new, sufficiently strong evidence
were to emerge.

Movement between these levels is governed by the sources and reducing
factors defined in Sections 3 and 4, not by any numeric aggregation of
them. Two claims can occupy the same confidence level for different
reasons, and a single missing element — such as any reproducibility
evidence at all — can be sufficient to keep a claim from rising above
Moderate regardless of how strong the rest of its evidence is.

--------------------------------------------------

# 6. Confidence Lifecycle

Confidence is not assigned once and left fixed. It progresses through the
following stages as the evidence underlying a claim changes:

**Initial.** The first confidence assessment made once a claim has
reached the Supported stage of its own lifecycle (Chapter 2, Section 3),
based on whatever evidence is linked to it at that point. An initial
assessment is provisional by construction, since it reflects only the
evidence available at the moment the claim was first supported.

**Updated.** The stage at which a confidence assessment is revised to
reflect newly linked evidence — additional runs, additional sources,
additional context — without that new evidence necessarily indicating a
directional change in the claim's standing.

**Strengthened.** The stage at which new evidence increases confidence —
a successful reproduction, additional independent sources converging on
the same result, resolution of a previously identified gap. A
strengthened assessment must be traceable to the specific evidence that
justified the increase (Section 9).

**Weakened.** The stage at which new evidence decreases confidence — a
failed reproduction attempt, newly discovered conflicting evidence, or the
identification of a scope or baseline problem in previously accepted
evidence. A weakened assessment carries the same traceability obligation
as a strengthened one, and this methodology treats a confidence decrease
as no less legitimate an outcome than an increase.

**Re-evaluated.** The stage triggered by the claim's own re-evaluation
(Chapter 2, Section 3) — a full reassessment of confidence against the
complete current body of evidence, rather than an incremental update
against only what has newly arrived. Re-evaluation exists because
incremental updates alone can drift from what a fresh, whole-evidence
assessment would produce, particularly after many small updates have
accumulated.

A confidence assessment that has not passed through this lifecycle in a
traceable way — for example, one that jumps directly to a high level
without any recorded justification for the intermediate stages — has not
met the standard this methodology sets for a legitimate assessment,
regardless of whether the final level happens to be correct.

--------------------------------------------------

# 7. Confidence Relationships

Confidence does not stand alone; it is defined in relation to the other
constructs this methodology establishes.

**Confidence and Evidence.** Confidence is derived from evidence, per
Sections 3 and 4, and must change when the underlying evidence changes.
Confidence that persists unchanged after evidence relevant to a claim has
been added, removed, or contradicted has become disconnected from its own
basis.

**Confidence and Claims.** Confidence is always confidence in a specific
claim, at that claim's specific stated scope (Chapter 2, Section 5).
Confidence assessed for a claim at one scope must not be silently carried
over to a broader or narrower restatement of that claim; a change in
scope requires its own confidence assessment.

**Confidence and Contradictions.** A claim with unresolved contradictory
evidence, or a contradictory relationship with another claim (Chapter 2,
Section 7), cannot be assessed at a high confidence level while that
contradiction remains unaddressed. Confidence assessment and contradiction
handling are distinct processes, but confidence must reflect the presence
of an unresolved contradiction rather than being computed as though the
contradiction were not present.

**Confidence and Judgment.** A scientific judgment (addressed in a later
chapter of this methodology) is formed from a claim together with its
confidence assessment. Confidence is an input to judgment, not a
substitute for it — a high confidence assessment describes how well
evidence supports a claim, while judgment is the further act of deciding
what should be concluded or done given that support.

**Confidence and Recommendations.** A recommendation (addressed in a
later chapter) inherits the confidence of the judgment it is based on.
A recommendation built on a claim assessed at Low or Very Low confidence
carries that limitation forward explicitly; confidence is not permitted
to be dropped silently as reasoning moves from claim to judgment to
recommendation.

--------------------------------------------------

# 8. Confidence Limitations

Positive evidence for a claim does not automatically warrant a high
confidence assessment. This methodology recognizes several situations in
which confidence should remain low even when the available evidence looks
favorable on its face:

- Evidence that is favorable but drawn from a single, non-independent
  source, regardless of how strong that single source's result appears.
- Evidence that is favorable but has not been tested against the
  possibility of a weak or unmatched baseline.
- Evidence that is favorable but has never been subjected to a genuine
  reproduction attempt, leaving open the possibility that the result was
  specific to its original execution.
- Evidence that is favorable but narrow in scope relative to the breadth
  of the claim being made, such that the favorable result may not
  generalize to the rest of the claim's stated scope.
- Evidence that is favorable but incompletely reported, such that its
  actual quality along the dimensions of Chapter 1 cannot be properly
  assessed.

In each of these situations, this methodology requires that confidence be
withheld rather than inferred optimistically from the evidence that is
present. Abstaining from a confidence assessment — recording that
available evidence is insufficient to place a claim at any level beyond
Very Low or Low — is a legitimate and often correct outcome, and is
preferable in every case to overstating certainty on the basis of evidence
that looks encouraging but has not actually been tested against the
factors in Section 4. A methodology that treats favorable-looking but
untested evidence as grounds for high confidence has abandoned scientific
caution for the appearance of decisiveness.

--------------------------------------------------

# 9. Traceability

Every confidence assessment must be explainable: it must be possible to
state, for any assigned level, which specific evidence and which specific
factors from Sections 3 and 4 justify that level rather than a higher or
lower one. A confidence assessment that cannot be decomposed into the
evidence and rules that produced it is not a scientific confidence
assessment under this methodology, regardless of how plausible the
assigned level appears.

This requirement connects directly to the traceability chain established
in Chapter 2, Section 9. Confidence sits between scientific rules and
judgment in that chain, which means a confidence assessment must itself be
traceable both downward, to the specific evidence and observations it
rests on, and it must in turn be usable as a traceable input to any
judgment built on top of it. A confidence level that appears without this
justification — asserted rather than derived — breaks the chain at
exactly the link this methodology treats as most consequential, since
confidence is the point at which evidentiary detail is compressed into a
single conceptual level that later components will treat as authoritative.
That compression is only legitimate if it remains reversible: any party
examining a confidence assessment must be able to expand it back into the
specific evidence and rules that produced it.

--------------------------------------------------

# 10. Future Integration

This document defines confidence independently of the components that will
consume it. Those components are described elsewhere in this methodology;
this section notes only their conceptual dependency on the definitions
above.

**Claim Verification.** Consumes a claim's supporting evidence to confirm
that the claim is actually consistent with it (Chapter 2), providing the
verified basis that confidence assessment then evaluates for strength.
Verification and confidence are sequential and distinct: a claim must be
verified as consistent with its evidence before the strength of that
evidence can be meaningfully assessed.

**Scientific Rule Engine.** Applies the rules that determine which
sources and reducing factors (Sections 3 and 4) are relevant to a given
claim category (Chapter 2, Section 4), since not every factor applies
with equal weight to every kind of claim — a reproducibility claim depends
more heavily on reproducibility evidence specifically than a performance
claim does, for instance.

**Judgment Generation.** Consumes a claim together with its confidence
assessment to form a terminal scientific judgment, depending on the
traceability established in Section 9 to ensure the judgment itself
remains explainable back through confidence to evidence.

**Recommendation Engine.** Consumes judgments, and through them their
underlying confidence assessments, to propose actions, depending on the
confidence relationships defined in Section 7 to ensure that a
recommendation never presents more certainty than the confidence behind it
actually warrants.

No component in this list is permitted to treat a confidence level as
self-justifying or to raise a claim's effective standing beyond what its
recorded confidence assessment supports. Every confidence level that
reaches any of these components must arrive with the traceable
justification this chapter requires.
