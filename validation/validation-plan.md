# Validation Plan — Does Experiment Audit v1 Genuinely Help ML Researchers?

**Status:** First real pass. `validation/` was scaffolded (five 0-byte
files) by `ff494be`, "Create validation framework," and populated for the
first time in this pass. Written directly on top of
`research/04_benchmarks/benchmark-plan.md` (hereafter "the benchmark
plan"), which already did the literature synthesis and methodology design
this document operationalizes. **This document does not re-derive that
synthesis.** Where the benchmark plan already answered a question (what
existing benchmarks get right, what datasets exist, how ground truth
should be built), this plan cites the section and moves straight to making
it executable: who does what, in what order, against what acceptance
bar, reported in what shape.

**What this pass is not:** it is not Version 2 planning, not new tooling,
not production code, and not the empirical validation itself. Per the
task brief, it is the protocol another researcher could pick up and run
without having to re-invent any of the above. Executing it is future work
(see `research/research-progress.md` for the updated priority order).

---

## 1. Purpose and framing

Experiment Audit has already answered "does the code do what it claims"
(233 tests, 96% coverage, `research-progress.md` §2) and has a designed-
but-unrun answer to "does an agent call the right tool"
(`scripts/tool_selection_eval.py`). Neither answers the question that
actually determines whether this project should continue: **would an
independent ML researcher, looking at real, messy experimental data,
agree with what the three judgment tools (`audit_ablation`,
`audit_training_curve`, `audit_sweep`) tell them — and would that
agreement change how they work?**

This plan treats that as two separable questions, deliberately not
conflated, because a positive answer to the first does not imply a
positive answer to the second:

- **Q_correctness — calibration:** does each tool's verdict match what an
  independent expert panel would conclude from the same raw data? (The
  benchmark plan's "category 2," §3.1.)
- **Q_value — utility:** even where a tool is well-calibrated, does
  surfacing its verdict unprompted change a researcher's belief, save
  them time, or catch something they would otherwise have missed? Pure
  calibration accuracy does not establish this — a tool can be accurate
  and still not be something researchers want in their workflow (the
  core UX bet named as unvalidated in `research-progress.md` §7, §9).

V1 is judged to "genuinely help" only if there is affirmative evidence for
**both** questions, not one. A validation pass that only measures
calibration and stops there would answer half the mission and could be
mistaken for having answered all of it — the same conflation risk the
benchmark plan already named for its own scope (Part 4) and Decision
Impact ("stop conflating 'tests pass' with 'verdicts are correct'"). This
plan extends that same discipline one level further: don't conflate
"verdicts are correct" with "researchers are helped."

## 2. Non-goals (explicit, per task brief)

- No new `audit_*` tool, threshold, or MCP capability is proposed or
  built here.
- No production code is written; scripts referenced below
  (`record_wandb_fixtures.py`, `tool_selection_eval.py`) already exist
  and are cited, not extended.
- No experiments are analyzed in this pass — this document defines how
  they *will* be analyzed once ground-truth data exists.
- No Version 2 roadmap decisions are made. Findings from executing this
  plan are Version 2 *input*, not Version 2 itself (see closing note in
  `research/research-progress.md`).

## 3. Structure of this validation package

| File | Answers |
|---|---|
| `validation-plan.md` (this file) | Why the protocol is shaped this way; phases; acceptance bar; reporting cadence; risk register |
| `validation/capability-matrix.md` | Per-capability (per-tool): research question, hypothesis, failure modes, ground truth, metrics, acceptable FP/FN rates, datasets — one row per tool, cross-referenced |
| `validation/datasets.md` | Exactly what data is needed per tool, where it comes from, how much, how it's split and versioned |
| `validation/annotation-guidelines.md` | The operational instructions an actual human annotator follows — label definitions, blinding procedure, disagreement handling, IRR computation |

Read in that order. `capability-matrix.md` is the index; the other two
are the operational detail two of its columns point into.

## 4. Guiding principles (inherited, not re-derived)

Every principle below is argued for at length in the benchmark plan; they
are restated here only as the constraints this plan is bound by, with a
pointer to where the argument lives:

1. **Human-labeled ground truth only — no LLM-as-judge**, because the
   judgment tools' entire structural advantage is being deterministic and
   non-LLM at inference time; validating them with an LLM grader would
   discard that advantage (benchmark plan §1.4, §3.4).
2. **Per-verdict-type metrics, never one aggregate score** (benchmark
   plan §1.3, §3.5) — an "88% accurate" headline is a category error for
   a tool whose value is in *which* errors it makes and how it fails.
3. **Disagreement is a reported category, not resolved noise** (benchmark
   plan §1.6, §3.4) — contested cases go in a labeled `contested` bucket,
   never silently folded into majority vote.
4. **Everything is versioned and reproducible from one command per
   metric** (benchmark plan §1.1, §1.3, §3.7, §3.9) — a cited number must
   always be traceable to a frozen dataset version and a runnable script.
5. **Retrieval tools get correctness tests, not calibration benchmarks**
   (benchmark plan §3.1) — `test_connection`, `list_runs`,
   `get_run_summary`, `get_metric_history`, `compare_runs` have no
   judgment to calibrate; applying panel-agreement metrics to them would
   be a design mistake, not extra rigor.
6. **A human baseline and an unaided-Claude baseline accompany every
   judgment-tool metric** (benchmark plan §3.5, §3.10) — a tool's score
   is uninterpretable without a reference point on the same cases.

## 5. Phases

Five phases, sequenced so that cheap, already-built work happens first
and the expensive net-new work (human annotation) is piloted before it is
scaled. This ordering matches `research-progress.md` §11's updated
priority list; this section adds the acceptance gate between each phase.

### Phase 0 — Run what's already built (cost: near-zero, blocked only on credentials)

- Run `scripts/record_wandb_fixtures.py` against a real W&B project.
- Run `scripts/tool_selection_eval.py` (`docs/tool-selection-eval.md`'s
  15 prompts, 4 distractor pairs).
- Add the determinism check named in benchmark plan §3.7: three
  independent reruns of each judgment tool against the same fixture,
  asserting bit-identical output. Free given the already-verified
  pure-function architecture (`research-progress.md` §2); no reason to
  defer it to a later phase.
- **Gate to Phase 1:** none — Phase 1 does not depend on Phase 0's
  results, only reuses its infrastructure. Phase 0 can run in parallel
  with Phase 1's dataset-sourcing work.

### Phase 1 — Source and freeze datasets (`validation/datasets.md`)

- Pull real ablation-shaped and curve-shaped cases from Open RL
  Benchmark (benchmark plan §1.5, §3.2).
- Construct the adversarial "trap" cases named in benchmark plan §3.8
  (allowlist-too-narrow, allowlist-too-permissive, scheduled-jump,
  below-threshold pathology, structural-vs-accidental covariance,
  contested cross-tool cases).
- Freeze `experiment-audit-bench-v0.1-pilot` — see `datasets.md` §5 for
  the exact versioning scheme.
- **Gate to Phase 2:** dataset frozen, versioned, and reviewed by at
  least one person other than whoever assembled it, for whether cases are
  realistic (not whether labels are correct — no labels exist yet).

### Phase 2 — Pilot the annotation protocol (`validation/annotation-guidelines.md`)

- Run the full annotation procedure on a small sample (10-15 cases per
  tool, per `research-progress.md` §11 item 8) before committing to the
  full panel effort.
- Compute inter-annotator agreement (Fleiss' kappa) on the pilot alone.
- **Gate to Phase 3 (hard gate, not advisory):** if pilot IRR is below
  the threshold in `annotation-guidelines.md` §6, the protocol itself is
  revised (clearer instructions, better examples, tighter label
  definitions) and re-piloted before any full-scale labeling begins.
  Scaling a protocol with poor agreement produces a ground-truth set that
  looks authoritative but isn't — exactly the failure this plan exists to
  prevent, not a formality to skip under time pressure.

### Phase 3 — Full ground-truth labeling

- Scale the piloted-and-passed protocol to the full dataset size target
  per tool (`datasets.md` §2-§4).
- Publish raw per-annotator labels alongside the aggregated set
  (benchmark plan §3.7, §3.9) — never only the majority-vote result.
- `audit_ablation` is labeled first, per the existing priority ranking
  (`research/03_workflows/workflow-ranking.md`, reaffirmed in benchmark
  plan Decision Impact); `audit_training_curve` and `audit_sweep` follow.
- **Gate to Phase 4:** full-set IRR computed and published, meeting or
  explicitly falling short of (with reasons stated) the same bar as
  Phase 2.

### Phase 4 — Score, baseline, and report

- Compute per-tool, per-verdict-type precision/recall/F1, confusion
  matrix, confidence calibration, and abstention-correctness
  (`capability-matrix.md`, benchmark plan §3.5).
- Compute the human-panel-agreement ceiling and the unaided-Claude
  baseline on the same cases (benchmark plan §3.5, §3.10 item 3) — this
  is the step that actually tests the project's central differentiation
  claim (`research-progress.md` §8), not an optional comparison.
- Report FP/FN rates in the tool-specific directions defined in
  `capability-matrix.md`, never merged into one number without the split
  shown (benchmark plan §3.6).
- Produce one report per tool using the template in §8 below, plus one
  cross-tool summary.
- **This phase does not have a downstream gate** — it is the deliverable.
  Its own output (§7) determines what, if anything, is built next.

### Phase 5 (separate track, can run alongside Phases 1-4) — Utility evidence

Calibration alone cannot answer Q_value (§1). This track is explicitly
**not** a benchmark in the Part 1/Part 3 sense — no ground-truth labels,
no precision/recall. It is the closest this plan comes to a qualitative
study, kept structurally distinct so it is never mistaken for a
statistical result:

- Recruit 3-5 real researchers (`research-progress.md` §11 item 2),
  prioritizing the Open RL Benchmark / CleanRL tracked-experiment
  community as the friendliest-evidenced entry point, per the same
  section.
- For each, run a small number of their own real `audit_*` calls (or, if
  they're not yet using the tool, a structured walkthrough against their
  own historical runs) and record, in their own words: did the verdict
  tell them something they didn't already know; would they want this
  surfaced unprompted or only on demand; did they trust the confidence
  field.
- Report as structured qualitative notes per researcher, not aggregated
  into a score — forcing a quantitative summary onto 3-5 interviews would
  manufacture false precision the sample size can't support.
- This track's findings feed §7's decision criteria alongside Phase 4's
  quantitative results; neither track substitutes for the other.

## 6. Roles

- **Dataset assembler** — sources and freezes cases (Phase 1). Must not
  also be an annotator on the same tool's dataset, to keep the panel
  blind to any pre-existing framing the assembler introduced.
- **Annotation panel** — minimum 3 ML researchers per case
  (`annotation-guidelines.md` §1), independent of this project's own
  design/engineering work on `audit_*`.
- **Protocol owner** — runs the pilot, computes IRR, decides pass/revise
  at each gate (§5 Phase 2/3 gates). Should be a different person from
  the dataset assembler where feasible, for the same blinding reason.
- **Reporter** — compiles Phase 4's per-tool reports; does not relabel or
  adjudicate contested cases (that's the panel's job, per
  `annotation-guidelines.md` §5).

A single person can hold more than one role only where team size forces
it; where that happens, it should be stated explicitly in the report
(§8's provenance field) rather than left implicit, per the same
provenance discipline `research-progress.md` §12 already established for
this project's research documents.

## 7. Decision criteria — what counts as "genuinely helps"

Stated up front, before any data exists, specifically so the bar can't be
quietly moved after seeing results:

- **Calibration bar, per tool:** F1 (per verdict type) is reported
  alongside the panel's own inter-annotator F1 as a ceiling (benchmark
  plan §3.5). A tool is judged calibration-validated for a verdict type
  if its F1 is within a pre-registered margin of that ceiling — the exact
  margin is a judgment call to be fixed *before* Phase 4 scoring begins
  (recommended default: within 15 percentage points of the panel ceiling,
  revisited only with a stated reason, never silently, per the FP/FN
  regression-gate discipline in benchmark plan §3.6).
- **Differentiation bar:** the unaided-Claude baseline (§5 Phase 4) must
  score measurably below the tool on the same cases for the calibration
  result to count as evidence *for* the tool's existence, not just for
  its correctness. A tool that matches unaided Claude is calibrated but
  not yet shown to be *worth having* as a separate deterministic
  component — this is the direct empirical test of
  `research-progress.md` §8's differentiation claim.
- **Utility bar:** at least a plurality of Phase 5's 3-5 researchers
  report the verdict told them something they didn't already know, or
  changed how they'd act, on at least one real case. This is a low bar
  deliberately — 3-5 interviews cannot support a high-confidence
  threshold — but a plan that sets no bar at all cannot be said to have
  tested the utility question either.
- **Honesty bar (process, not outcome):** contested/ambiguous cases and
  any adversarial trap-case failures (benchmark plan §3.8) are reported
  in the same document as successes, not in an appendix or omitted. A
  report meeting the calibration and differentiation bars while hiding
  known trap-case failures does not meet this plan's bar, regardless of
  the headline numbers.

**If a tool fails the calibration or differentiation bar:** that is a
valid, reportable outcome, not a failed validation pass. The purpose of
this protocol is to find out, not to produce a passing grade. A
tool-specific failure should be reported with the same rigor as a
success, including the FP/FN direction (`capability-matrix.md`) so a
future decision about revising or retiring that specific tool has
evidence to act on.

## 8. Reporting template (per tool)

Every Phase 4 report follows this shape, so results are comparable across
tools and across future re-runs:

```
# Validation Report -- <tool_name> -- <dataset_version>

## Provenance
- Dataset version, freeze date, assembler
- Annotation panel size, dates, protocol version
- Roles held by the same person (if any), per Section 6

## Ground truth summary
- N cases (total, per verdict type in the panel's labels)
- N contested (excluded from main scoring, reported separately)
- Inter-annotator agreement (Fleiss' kappa), with 95% CI or explicit
  small-sample caveat

## Calibration results
- Confusion matrix (tool verdict x panel verdict)
- Precision / recall / F1 per verdict type, with CI or small-sample caveat
- Confidence calibration: accuracy at each confidence level (high/medium/low)
- Abstention rate and abstention-correctness (where applicable)

## Baselines
- Panel inter-annotator F1 (the ceiling)
- Unaided-Claude F1 on the same cases
- Tool F1 vs. both, stated as a gap, not just three numbers side by side

## Error analysis
- False-positive rate and false-negative rate, tool-specific direction
  per capability-matrix.md
- Every adversarial trap case (benchmark plan Section 3.8): pass/fail, with the
  actual tool output shown
- Qualitative pattern in the errors, if one exists (e.g. "every false
  negative involved a differently-named seed field") -- only claimed if
  actually observed, not inferred from a small N

## Reproduction
- Exact command(s) that regenerate every number above
- Dataset and protocol version this report is pinned to

## Verdict against Section 7's decision criteria
- Calibration bar: met / not met, with the numbers
- Differentiation bar: met / not met, with the numbers
- Honesty bar: confirms contested cases and trap-case failures are all
  reported above, not selectively omitted
```

## 9. Risk register

- **Small-sample overconfidence.** Every dataset in `datasets.md` is
  small relative to a production ML benchmark. Every metric in Section 8
  carries a confidence interval or explicit small-sample caveat
  (benchmark plan §1.7, Dror et al.) — a report that drops this is
  non-compliant with this plan, not just imprecise.
- **Annotator anchoring.** Panel members labeling with knowledge of the
  tool's own verdict would inflate agreement artificially.
  `annotation-guidelines.md` §2's blinding procedure exists specifically
  to prevent this; any deviation must be disclosed in the report's
  provenance section.
- **Dataset skew toward RL.** Open RL Benchmark, the primary data source
  (benchmark plan §1.5), is RL-heavy — the same skew
  `research-progress.md` §7 already names as a property of what's
  publicly documented, not necessarily of the target user base. A result
  from this protocol validates (or invalidates) the tools' behavior on
  RL-shaped data specifically; generalizing to NLP/CV supervised
  fine-tuning workflows requires either additional data sourced from that
  domain or an explicit, stated scope limitation in the final report —
  not silent generalization.
- **Threshold-tuning temptation.** Per benchmark plan Decision Impact, no
  threshold or allowlist change should be made "to improve the
  benchmark" during or immediately after this protocol runs, except as a
  clearly-cited response to a specific FP/FN finding, versioned and
  re-tested against the frozen ground-truth set, never a speculative fix
  made in anticipation of a result.
- **Protocol drift between pilot and full run.** Phase 2's gate exists
  precisely so the piloted protocol is what scales, not a protocol that
  quietly changed between the 10-15 case pilot and the full set. Any
  change to the guidelines after Phase 2 passes requires a new pilot.
- **Conflating calibration with utility (Section 1).** Restated here as a
  risk, not just a framing choice: a strong Phase 4 result with no
  Phase 5 evidence should never be reported as "validated" without the
  utility caveat attached (Section 7's utility bar).

## 10. What this plan deliberately leaves open

- The exact numeric margin in Section 7's calibration bar ("within 15
  points of ceiling") is a recommended default, not a number this pass
  has evidence to fix precisely — it should be confirmed or revised by
  whoever runs Phase 2's pilot, before Phase 4 scoring, and stated
  explicitly in the first report rather than silently assumed.
- Whether 3-5 researchers (Phase 5) is enough to say anything about
  utility at all is itself an open methodological question this plan
  does not resolve — it is the minimum `research-progress.md` §11
  already named as actionable, not a number derived from a power
  calculation.
- Whether NLP/CV-domain data should be sourced for Phase 1 in parallel
  with Open RL Benchmark, rather than as a stated follow-on limitation,
  is a scope decision left to whoever executes this plan, given
  time/resource constraints not fixed here.
