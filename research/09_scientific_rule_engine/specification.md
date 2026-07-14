# 1 Purpose

Every prior chapter of this methodology — Philosophy (Chapter 0), Evidence
(Chapter 1), Claims (Chapter 2), and Contradictions (Chapter 8) — defines
concepts. None of them define how those concepts are actually combined to
produce a scientific output. Evidence sits inert until something inspects
it against a claim. A contradiction is not detected by itself; something
must compare two evidence items and determine that they conflict. The
Scientific Rule Engine is the component of Experiment Audit responsible
for that combination step: the place where the definitions established
elsewhere in this methodology are actually applied to a specific body of
evidence to produce a specific reasoning output.

The central design commitment of the Scientific Rule Engine is that this
application must be deterministic. Given the same evidence, the same
claims, and the same rule set, the engine must produce the same output,
every time, with no dependency on execution order, sampling, model
temperature, or any other source of variation that is not itself part of
the recorded input. This is not a preference for simplicity over
sophistication. It is a direct consequence of the traceability requirement
established throughout this methodology (Chapter 1, Section 9; Chapter 2,
Section 9; Chapter 8, Section 9): a conclusion that cannot be
independently regenerated from its stated inputs has not, by the
standards of this methodology, been shown to follow from those inputs at
all. It has merely been asserted alongside them.

Opaque reasoning — reasoning whose internal steps cannot be inspected,
whose output cannot be regenerated from the same inputs, or whose
conclusion depends on factors outside the recorded evidence — is
unacceptable within Experiment Audit for a reason distinct from
correctness. An opaque process might, on any given evaluation, produce a
correct-looking conclusion. What it cannot do is allow a reviewer,
without trusting the process itself, to verify that the conclusion
follows from the evidence rather than from something else entirely:
prior exposure to similar cases, an unstated preference, an artifact of
how a question happened to be phrased. A deterministic rule, by contrast,
is checkable by construction. Its inputs are fixed and recorded (Section
3), its output is fixed and recorded (Section 4), and the relationship
between the two is stated explicitly enough that a second party, given
the same inputs, can confirm the same output would follow — without
needing to trust whoever or whatever executed the rule the first time.

This requirement is why the Scientific Rule Engine is explicitly not an
instance of the categories that commonly perform this kind of reasoning
elsewhere: it is not a large language model prompted to render a
judgment, it is not a heuristic scorer that combines signals into a
weighted output, and it is not a machine learning model trained to
predict a conclusion from examples. Each of those approaches produces
outputs that are difficult or impossible to regenerate deterministically
from a fixed set of stated inputs, and each therefore breaks the
traceability chain this methodology requires at every other layer. The
Scientific Rule Engine is a deterministic reasoning system: a defined set
of rules, each with declared inputs and declared outputs, applied to
evidence and claims in a way that can be re-executed, inspected, and
independently verified by anyone with access to the same evidence.

--------------------------------------------------

# 2 Definition

A **Scientific Rule** is a defined, deterministic procedure that consumes
a specified set of inputs drawn from evidence, claims, and their
associated metadata (Section 3), and produces one or more specified
reasoning outputs (Section 4), according to a fixed and inspectable
relationship between the two.

Four properties are load-bearing in this definition:

- **Consumes evidence.** A rule operates over evidence admitted under
  Chapter 1's definition. It does not operate over raw, unvalidated
  artifacts, and it does not operate over impressions or summaries of
  evidence that have not themselves been recorded as evidence.
- **Inspects claims.** A rule examines the claims (Chapter 2) that
  evidence has been attached to, and the relationship between that
  evidence and those claims, rather than reasoning about evidence in a
  vacuum disconnected from the claim it bears on.
- **Evaluates relationships.** A rule's function is to evaluate a
  relationship — between evidence and a claim, between two evidence
  items, between two claims, or between evidence and a scientific
  principle such as those referenced in Chapter 8, Section 9 — and to
  characterize that relationship according to a fixed procedure.
- **Generates reasoning outputs.** A rule's output is one of the defined
  categories in Section 4: a new statement about what the inputs show,
  not a modification of the inputs themselves.

Two restrictions apply to every rule without exception, and both follow
directly from the philosophy established in Chapter 0.

**A rule never invents evidence.** A rule may read, compare, and
characterize evidence that already exists under Chapter 1's definition.
It may never introduce a value, a measurement, or a fact into its
reasoning that is not traceable to an existing evidence item. Where a
rule's procedure would require a value that is not present in the
evidence available to it, the correct behavior is not to estimate,
infer, or assume that value — it is to produce a Missing Evidence or
Abstain output (Section 4), reporting the absence rather than filling
it.

**A rule never invents claims.** A rule may inspect, verify, or evaluate
claims that already exist under Chapter 2's definition, and may produce
new reasoning outputs — an observation, a hypothesis, a contradiction —
about those claims. It may never originate a claim on its own initiative
and then proceed to evaluate that self-originated claim as though a
researcher had asserted it. Claims enter this methodology only through
the process defined in Chapter 2; the Scientific Rule Engine consumes
that process's output, it does not substitute for it.

A rule, under this definition, is therefore best understood as a fixed
lens applied to existing material — evidence and claims — rather than as
a generative process that produces new material of its own. Everything a
rule outputs must be decomposable, without exception, into the specific
evidence and claims it was given as input, together with the fixed
procedure that connects them.

--------------------------------------------------

# 3 Rule Inputs

A rule's inputs must be fully specified and fully recorded before the
rule is applied, so that the rule's application can later be
regenerated exactly as described in Section 1. This methodology
recognizes the following categories of input.

**Evidence.** One or more evidence items, admitted under Chapter 1's
definition, together with their full provenance (Chapter 1, Section 9)
and quality assessment (Chapter 1, Section 7). A rule that consumes
evidence without its provenance and quality metadata cannot perform the
comparisons most rules require — for example, weighing evidence produced
by a single run against evidence produced by repeated runs (Chapter 1,
Section 3).

**Claims.** One or more claims, admitted under Chapter 2's definition,
together with their declared scope (Chapter 2, Section 5) and current
lifecycle stage (Chapter 2, Section 3). A rule that inspects a claim
without knowing its declared scope cannot correctly distinguish a
genuine relationship from an unstated scope mismatch, per the
distinction established in Chapter 8, Section 2.

**Observations.** Prior statements about what specific evidence shows,
where these have already been produced by an earlier stage of reasoning
(Chapter 1, Section 10). Rules that operate at a level above raw
evidence — for example, rules that reason about patterns across several
observations — take observations as input rather than re-deriving them
from evidence directly, so that the observation-forming step remains a
distinct, inspectable stage in the reasoning chain.

**Metadata.** Structural information attached to evidence or claims that
is not itself a measurement — configuration identifiers, dataset
identifiers, environment identifiers, timestamps, and versioning
information. Metadata is what allows a rule to determine whether two
evidence items are actually comparable, per the sources of contradiction
enumerated in Chapter 8, Section 4 (different datasets, different
environments, different implementations).

**Scope.** The declared boundary of applicability for a claim (Chapter
2, Section 5) or for the evidence being evaluated against it. Scope is
listed as a distinct input, separate from the claim or evidence it
belongs to, because a rule's correct behavior frequently depends on
comparing declared scope against the actual scope of the evidence
available — the central check required by Chapter 8, Section 2 before
any conflict may be classified as a genuine contradiction.

**Context.** The broader circumstances under which evidence was
produced or a claim was made, where such circumstances have been
explicitly recorded — for example, that a body of evidence was collected
as part of a specific ablation study, or that a claim was made in
response to a specific prior contradiction. Context is admitted as input
only where it has been recorded as part of the evidence or claim record
itself; a rule may not draw on unrecorded context supplied informally at
the time the rule is applied, since doing so would break the
regenerability requirement in Section 1.

**Existing Confidence.** Where a claim already carries a confidence
assessment produced by an earlier stage of reasoning, that existing
assessment is a valid input to a rule that adjusts, extends, or
re-evaluates confidence in light of new evidence. A rule that produces a
Confidence Adjustment output (Section 4) is, definitionally, operating
on an existing confidence value as one of its inputs.

**Detected Contradictions.** Contradictions already recorded under
Chapter 8, together with their lifecycle stage (Chapter 8, Section 5)
and quality assessment (Chapter 8, Section 8). A rule reasoning about a
claim must take any relevant recorded contradiction as input, since
Chapter 8, Section 6 establishes that a claim's evidentiary status
cannot be correctly assessed while ignoring contradictions that bear on
it.

**Missing Evidence.** The explicit record of evidentiary gaps already
identified under Chapter 1, Section 8 — for example, that a claim
requiring baseline evidence has no baseline evidence attached. Missing
evidence is admitted as an input in its own right, distinct from the
absence of an input, because a rule's correct behavior in the presence
of a known gap (for example, producing a Missing Evidence or Abstain
output, Section 4) depends on that gap being explicitly flagged rather
than merely being unaddressed.

Every rule application must record which specific instances of each
applicable input category were used, with sufficient identifying detail
(matching the provenance standard of Chapter 1, Section 9) that the
exact input set could be reassembled independently. An input that
influenced a rule's output but was not recorded as part of the rule's
input set constitutes a violation of the determinism principle
established in Section 1 and Section 7.

--------------------------------------------------

# 4 Rule Outputs

A rule's output must belong to one of the following defined categories.
A rule may produce more than one output of different categories from a
single application where its procedure genuinely warrants it, but every
individual output must be separately categorized and separately
traceable (Section 9).

**Observation.** A statement about what a specific piece of evidence
shows, scoped tightly to that evidence, consistent with the definition
of observation established in Chapter 1, Section 2. This is the most
granular output category: it does not yet assert anything about a claim
in general, only about what a given evidence item, on inspection,
contains.

**Hypothesis.** A proposed explanation for a pattern identified across
one or more observations, consistent with the definition established in
Chapter 1, Section 2. A rule producing a hypothesis output is proposing
a candidate account of why the evidence looks the way it does; the
hypothesis remains provisional and answerable to further evidence, and
must not be presented, at this stage, as though it had already been
established.

**Contradiction.** A record of a relationship meeting the definition in
Chapter 8, Section 2, including the category or categories from Chapter
8, Section 3 the rule has determined apply, and the specific evidence
items or claims found to be in conflict. A rule producing a contradiction
output initiates the lifecycle defined in Chapter 8, Section 5; it does
not, by itself, resolve that lifecycle.

**Confidence Adjustment.** A change to a claim's existing confidence
value (taken as input per Section 3), together with the specific reason
for the adjustment — new supporting evidence, newly discovered missing
evidence, a newly detected or newly resolved contradiction, or a
statistical finding bearing on the claim. A confidence adjustment output
must state both the direction and the specific justification of the
change; an adjustment without a stated justification is not a valid
output under this methodology.

**Judgment.** A terminal scientific determination about a claim,
produced only once a rule's procedure has established sufficient basis
under the standards defined elsewhere in this methodology (claim
verification and confidence assessment, both addressed further in
Section 10). A judgment output is the output category with the most
downstream consequence, and correspondingly carries the strictest
traceability obligation (Section 9).

**Recommendation.** A proposed action grounded in one or more judgments
— for example, that a comparison should not be trusted without
additional controls, or that a claim's stated scope should be narrowed.
A recommendation output is only valid when it is traceable, through the
judgments it is grounded in, back to the underlying evidence, consistent
with the requirement established in Chapter 8, Section 6 that
recommendations inherit the qualifications of the judgments beneath
them.

**Missing Evidence.** A rule's explicit report that evidence required
for its procedure to proceed is absent, using the categories established
in Chapter 1, Section 8. This output category exists so that an
evidentiary gap discovered during rule application is recorded with the
same explicitness as a gap discovered during initial evidence collection,
rather than causing the rule to silently produce a weaker or partial
result without flagging why.

**Request More Evidence.** A rule's explicit statement that a specific,
named category of additional evidence, if obtained, would allow it to
proceed toward a Judgment or Confidence Adjustment output that it cannot
currently produce. This output is distinguished from Missing Evidence in
that it is forward-looking and actionable: it names what evidence would
resolve the gap, rather than only recording that the gap exists.

**Abstain.** A rule's explicit refusal to produce a Judgment,
Confidence Adjustment, or Recommendation output for the input it was
given, because the conditions defined in Section 8 for a rule failure
are met. Abstention is not the absence of an output; it is itself a
recorded output, stating that no further conclusion can responsibly be
drawn from the available inputs, together with the specific reason.

Rule outputs are ordered by consequence, from Observation and Hypothesis
at the least consequential end through Judgment and Recommendation at
the most consequential end. This methodology does not permit a rule to
produce a highly consequential output (Judgment, Recommendation) as a
substitute for a less consequential one (Observation, Hypothesis) that
its actual procedure only supports; an output must be labeled according
to what the rule's inputs actually establish, not according to what
would be most useful to a downstream consumer.

--------------------------------------------------

# 5 Rule Lifecycle

A Scientific Rule, distinct from any single application of that rule, is
itself an artifact of this methodology with its own lifecycle. This
lifecycle governs when a rule may be applied, and what obligations
precede or follow each stage.

Draft
↓
Validated
↓
Applied
↓
Reviewed
↓
Updated
↓
Deprecated

**Draft.** A rule has been specified — its inputs (Section 3), its
possible outputs (Section 4), and the fixed relationship between them —
but has not yet been checked against the principles in Section 7 or
tested against known cases. A draft rule must not be applied to produce
outputs that enter the permanent evidentiary or claim record of an
experiment under audit.

**Validated.** A rule has been checked to confirm that it satisfies the
principles in Section 7, in particular determinism and evidence-first
reasoning: that its output is fully determined by its declared inputs,
and that it does not depend on any input outside those declared in
Section 3. Validation also confirms that the rule's defined inputs and
outputs are each drawn from the categories established in Sections 3
and 4, with no undeclared category introduced informally.

**Applied.** A validated rule has been executed against a specific,
recorded set of inputs to produce a specific, recorded output, with the
full traceability record required by Section 9. A rule may be applied
any number of times, against any number of distinct input sets, once it
has reached this stage; each individual application is a separate
traceable event.

**Reviewed.** The outputs a rule has produced across its applications
are examined — by a human reviewer, by comparison against known correct
outcomes, or by a later stage of this methodology not yet defined — to
determine whether the rule continues to produce outputs consistent with
the principles in Section 7 and the definitions in Chapter 0 through
Chapter 8. Review may confirm the rule is functioning as intended, or
may identify a defect: a rule that produces an output type inconsistent
with its declared procedure, or a rule whose procedure turns out to
depend on an input it did not declare.

**Updated.** A defect identified during review is corrected by revising
the rule's specification — its inputs, its outputs, or the relationship
between them — and returning the revised rule to the Validated stage
before it may resume production of applications that enter the
permanent record. An updated rule is a distinct version of the rule; its
prior applications, made under the earlier version, remain part of the
historical record and are not retroactively altered, consistent with the
immutability principle established for evidence generally (Chapter 1,
Section 3).

**Deprecated.** A rule is withdrawn from further application, either
because it has been superseded by an updated version, because the
scientific principle it implemented has itself been revised, or because
review has found it unsound in a way that revision cannot correct. A
deprecated rule's prior applications remain part of the historical
record; deprecation governs future use of the rule, not the standing of
outputs it has already produced, which retain whatever standing their
own traceability record supports.

--------------------------------------------------

# 6 Rule Composition

A single scientific conclusion is rarely the product of a single rule.
This section governs how multiple rules combine.

**Independent Rules.** Two rules are independent when neither rule's
input set depends on the other's output. Independent rules may be
applied in any order, or concurrently, without affecting either rule's
result, since determinism (Section 7) guarantees that each rule's output
depends only on its own declared inputs.

**Dependent Rules.** A rule is dependent on another when its input set
includes an output the other rule produces — for example, a rule that
consumes an Observation output (Section 4) to produce a Hypothesis
output, or a rule that consumes a Confidence Adjustment to produce a
Judgment. Dependent rules must be applied in an order consistent with
this dependency: a dependent rule may not be applied before the rule it
depends on has produced the output it requires.

**Rule Ordering.** Where dependency (above) does not itself fully
determine an order, an explicit ordering must still be defined and
recorded for any set of rules applied together, so that the same set of
rules, applied to the same inputs, produces the same sequence of
applications on every execution. An unordered rule set applied
concurrently without declared independence is not a valid configuration
under this methodology, since it would reintroduce the non-determinism
Section 1 exists to eliminate.

**Rule Priorities.** Where more than one rule is capable of producing an
output of the same category from a given input set, a declared priority
determines which rule's output is retained as the operative one for that
category, and which, if any, are retained as secondary or dissenting
outputs rather than discarded. Priority is a declared property of the
rule set's configuration, not a runtime decision made ad hoc at the
moment two applicable rules are found.

**Conflict Resolution.** Where two rules produce outputs that
contradict each other under the definition in Chapter 8, Section 2, that
outcome is not treated as a defect in rule composition to be silently
resolved by priority alone. It is treated as a contradiction in its own
right, subject to the full lifecycle defined in Chapter 8, Section 5.
Priority (above) may determine which output is treated as operative in
the interim, but the conflict itself must be recorded and carried
forward exactly as any other contradiction is, per Chapter 8, Section 6.

**Rule Chaining.** A sequence of dependent rules, applied in the order
their dependencies require, forms a chain in which each rule's output
becomes the next rule's input. A chain is only valid under this
methodology if every link in it is individually traceable (Section 9);
a chain's final output must be decomposable back through every
intermediate rule application to the original evidence and claims that
began the chain, with no step compressed or omitted from the record.

Rule composition, across all of the mechanisms above, is governed by a
single constraint: the combined behavior of multiple rules must remain
as deterministic, traceable, and regenerable as the behavior of any
single rule considered in isolation. A composition mechanism that
introduces order-dependence, priority ambiguity, or untracked conflict
resolution violates the same principles that govern individual rules.

--------------------------------------------------

# 7 Rule Principles

The following principles govern every rule and every rule application
admitted under this methodology. They take precedence over convenience,
performance, or the desire for a rule to produce a more complete-looking
output than its inputs actually support.

**Determinism.** A rule's output is a fixed function of its declared
inputs. The same inputs, applied to the same version of a rule, produce
the same output on every execution, with no dependency on execution
order (except as governed by Section 6), timing, or any external state
not itself declared as an input.

**Transparency.** A rule's procedure — the relationship it applies
between its declared inputs and its declared outputs — is stated
explicitly and is available for inspection by anyone reviewing the
rule's applications, rather than existing only as an unstated internal
mechanism.

**Explainability.** Every output a rule produces is accompanied by a
statement of why that output follows from the specific inputs the rule
was given, sufficient for a domain expert to follow the reasoning
without independently re-deriving it from scratch.

**Reproducibility.** A rule's application, given its recorded inputs
and its recorded version (Section 5), can be re-executed by an
independent party to confirm the same output is produced. This is
determinism (above) considered as a property that must remain checkable
after the fact, not only true at the moment of original execution.

**Modularity.** A rule performs one defined function, with declared
inputs and declared outputs, and can be examined, tested, validated, or
deprecated independently of other rules, consistent with the modularity
design goal established for the methodology as a whole in Chapter 0,
Section 5.

**Traceability.** Every application of a rule is recorded in sufficient
detail to satisfy the requirements of Section 9: the specific inputs
used, the reasoning applied, and the specific output produced, in a form
that can be walked back to the underlying evidence and claims.

**Composability.** A rule is specified in terms of the input and output
categories defined in Sections 3 and 4, so that it can be combined with
other rules under the mechanisms defined in Section 6 without requiring
special-case handling for each new combination.

**Backend Independence.** A rule's specification does not depend on any
particular storage system, evidence tracker, or execution environment,
consistent with the backend independence design goal established in
Chapter 0, Section 5. The rule is defined by its inputs, its outputs, and
the relationship between them, not by the infrastructure used to execute
it.

**Evidence First.** A rule's procedure begins from evidence and claims
already admitted under Chapters 1 and 2, and never begins from an
assumed or preferred conclusion that evidence is subsequently searched
to support. This principle is a direct application, at the level of
individual rules, of the core philosophical commitment established in
Chapter 0: evidence before conclusions.

**Abstain When Necessary.** A rule that cannot produce a Judgment,
Confidence Adjustment, or Recommendation output consistent with the
Evidence First principle, given the inputs it has been provided, is
required to produce an Abstain output (Section 4) rather than a
weaker or hedged version of the more consequential output. Abstention
is treated as correct rule behavior under insufficient input, not as a
failure of the rule.

--------------------------------------------------

# 8 Rule Failures

A rule failure, under this methodology, is not a software malfunction.
It is a situation in which the inputs available to a rule do not meet
the conditions its procedure requires in order to responsibly produce a
consequential output. This section defines the conditions under which a
rule must refuse to continue toward such an output, and must instead
produce a Missing Evidence, Request More Evidence, or Abstain output
(Section 4).

**Missing evidence.** The rule's procedure requires an evidence category
that is not present among its inputs, per the missing evidence
categories established in Chapter 1, Section 8. A rule encountering this
condition must not proceed as though the missing category were
immaterial to its conclusion; it must report the gap explicitly.

**Scope ambiguity.** The rule cannot determine, from its declared
inputs, whether the evidence and the claim under evaluation share the
same scope, per the distinction established in Chapter 8, Section 2. A
rule that proceeds to characterize a relationship as a genuine
contradiction, or as genuine support, without first resolving scope
ambiguity has violated the Evidence First principle (Section 7).

**Conflicting evidence.** The rule's inputs include evidence items that
are themselves in an unresolved contradiction, per Chapter 8, Section 5.
A rule whose procedure would require treating that evidence as settled
in order to proceed must instead produce an output that reflects the
unresolved status of the underlying evidence, consistent with Chapter 8,
Section 6.

**Unknown claim.** The rule has been asked to evaluate a relationship
involving a claim that has not been formulated under the process defined
in Chapter 2, or whose scope (Chapter 2, Section 5) has not been
declared. A rule may not proceed by inferring or constructing a claim on
the input's behalf, per the restriction in Section 2 that a rule never
invents claims.

**Incomplete metadata.** The rule's procedure requires metadata — a
dataset identifier, an environment identifier, a configuration version —
that is not present among its inputs, making it impossible to determine
whether evidence being compared is actually comparable, per the sources
of contradiction discussed in Chapter 8, Section 4.

Abstaining under any of these conditions is preferable to unsupported
reasoning for a reason that follows directly from the philosophy
established in Chapter 0: a confident-looking conclusion produced
despite an unresolved gap is not more useful than an honest statement
that the gap exists — it is worse, because it presents an unsupported
conclusion with the same appearance as a supported one, and nothing in
its output distinguishes the two. A rule that abstains has correctly
reported the actual state of the evidence. A rule that does not abstain
under these conditions, and instead produces a Judgment or Confidence
Adjustment regardless, has produced an output indistinguishable in form
from a properly supported one, which is precisely the failure mode this
methodology is designed to prevent at every other layer.

--------------------------------------------------

# 9 Rule Traceability

Every application of a rule must record the following, in full, before
its output may be treated as part of the permanent evidentiary or claim
record of an experiment under audit:

**Input.** The complete, specific set of inputs used for this
application, identified per the categories in Section 3, with
sufficient detail (matching the provenance standard of Chapter 1,
Section 9) that the same input set could be independently reassembled.

**Reasoning.** The specific procedure the rule applied to its inputs to
arrive at its output, stated explicitly enough that a reviewer can
follow the connection between input and output without needing to trust
the rule's execution, consistent with the transparency and
explainability principles in Section 7.

**Evidence Used.** The specific evidence items (Chapter 1) that were
load-bearing in this application, distinguished from evidence that may
have been available as context but did not actually inform the output
produced.

**Claim.** The specific claim or claims (Chapter 2) this application
bears on, consistent with the requirement that a rule's output always be
connectable to a claim it affects.

**Output.** The specific output produced, categorized per Section 4,
including its full content — not merely its category label.

This record is essential for the same reason traceability is essential
at every other layer of this methodology: it is what allows a rule's
application to be independently checked, disputed, or reproduced by a
party other than the one that executed it. A rule application that
cannot be decomposed into its recorded input, reasoning, evidence used,
claim, and output has not satisfied the traceability principle in
Section 7, regardless of how correct its output happens to be; a correct
conclusion reached through an untraceable process is, under this
methodology, indistinguishable from a fortunate guess.

--------------------------------------------------

# 10 Future Integration

This document specifies the Scientific Rule Engine independently of the
components that will consume its outputs or supply inputs to it beyond
what has already been recorded as evidence and claims. Those components
are described elsewhere in this methodology or remain future work; this
section notes only their conceptual dependency on the definitions above,
without describing how any of them will be built.

**Claim Verification.** A future component responsible for determining
whether a stated claim is supported by its attached evidence. Claim
Verification is expected to be implemented substantially as an
application of Scientific Rules that consume evidence and a claim
(Section 3) and produce Observation, Contradiction, or Confidence
Adjustment outputs (Section 4) bearing on that claim's support.

**Confidence Assessment.** A future component responsible for producing
a calibrated estimate of how strongly evidence supports a claim.
Confidence Assessment is expected to consume the Confidence Adjustment
outputs (Section 4) produced by individual rule applications, combining
them under the rule composition mechanisms defined in Section 6 into a
claim's current confidence standing.

**Contradiction Engine.** A future component responsible for the
detection, investigation, and lifecycle management of contradictions
defined in Chapter 8. The Contradiction Engine is expected to be
implemented as a category of Scientific Rule whose output is
specifically the Contradiction output defined in Section 4, applying the
categories and sources established in Chapter 8, Sections 3 and 4 as
part of its declared procedure.

**Judgment Generation.** A future component responsible for producing a
terminal scientific determination about a claim. Judgment Generation is
expected to consume verified claims (from Claim Verification) and their
current confidence (from Confidence Assessment) as inputs to rules whose
output category is Judgment (Section 4), remaining subject to the
Abstain When Necessary principle (Section 7) wherever those inputs are
insufficient.

**Recommendation Engine.** A future component responsible for proposing
actions grounded in judgments. The Recommendation Engine is expected to
consume Judgment outputs as inputs to rules whose output category is
Recommendation (Section 4), inheriting the full traceability chain
(Section 9) back through those judgments to the underlying evidence.

**Review Report Generator.** A future component responsible for
assembling the outputs of rule applications — observations, hypotheses,
contradictions, confidence adjustments, judgments, and recommendations —
into a document intended for human review. The Review Report Generator
is expected to depend entirely on the traceability record defined in
Section 9, since a report that presents a rule's output without the
ability to trace it back to input, reasoning, evidence, and claim would
reintroduce, at the reporting layer, the opacity this entire methodology
is designed to eliminate.

No component in this list is permitted to treat a rule's output as
authoritative without the traceability record required by Section 9, and
none is permitted to supply a rule with an input that has not been
admitted as evidence, a claim, or one of the other categories defined in
Section 3. That constraint is the reason this document exists as a
specification prior to any implementation: the Scientific Rule Engine is
the layer at which every definitional commitment made earlier in this
methodology is either honored in practice or silently abandoned, and
this document exists to ensure it is the former.
