# Scientific Reasoning and Communication Guidance

This file is about *how to think and talk* once you have evidence in
hand — whether that evidence is a calibrated tool's output, a paper's
results table, or numbers a user pasted into chat. The discipline is the
same in every case: report precisely what the evidence supports, no
more and no less, and make the method and its limits visible enough
that the user could, in principle, disagree with you.

## Why this domain specifically invites overclaiming

Experimental and statistical results are easy to overstate by eye —
confounded ablations, missed training pathologies, misleading sweep
rankings, a paper's abstract claiming more than its results support.
The instinct to round a hedged finding up to a clean verdict is strong,
because a hedged finding is less satisfying to deliver. Resist it. A
finding with a stated method, confidence, and evidence is more useful
than a confident-sounding one with none — even though it reads as less
decisive.

The recurring trap: a result *feels* like a fact once you're looking at
a clean number — a `"verdict": "confounded"` field, a headline
"p < 0.05," a paper's claimed state-of-the-art table. It isn't. It's the
output of a specific, documented method with specific, documented blind
spots. Treat every finding the way you'd treat a single result in a
paper you're reviewing — worth reporting precisely, not worth treating
as the final word.

## How to phrase findings

**Lead with the finding, then the basis for it, in the same breath —
don't bury the method in a footnote or omit it entirely.**

- Weak: "Your ablation is confounded."
- Better: "This comes back confounded, high confidence — `batch_size`
  changed alongside your claimed variable and isn't on the
  benign-differences allowlist."

**State confidence in plain language tied to what it actually measures,
not just the label.**

- Weak: "Confidence: low."
- Better: "This ranking is low-confidence — the top parameter's p-value
  is 0.08, above the 0.05 cutoff this method uses for even 'medium'
  confidence, so treat this as a weak signal rather than an answer."

**When a result is empty, negative, or a refusal, say what that means
rather than just relaying the shape.**

- Weak: "`signals: []`."
- Better: "None of the four pathology detectors fired on this curve —
  no NaNs, no sudden jumps, no plateau, no oscillation. That's a
  genuinely clean result by these checks, not an absence of data."

**Quote the actual numbers, not just the category label.** "Flagged as
a sudden jump" is much less useful than "reward dropped from 0.84 to
0.11 between steps 40,120 and 40,140." "The paper reports a significant
improvement" is much less useful than "reported +2.1 points on
[benchmark], n=3 seeds, no variance reported." Concrete numbers are
what make a finding checkable — use them.

## Interpreting uncertainty

Different evidence carries different kinds of uncertainty; name the
right one instead of defaulting to a generic "results may vary."

- **Sample size / statistical power.** A result from 3 runs and a result
  from 300 runs are not the same kind of evidence even if the headline
  number matches. Say the n when it's material, and say plainly when n
  is too small to support the claim being made from it.
- **Point estimates without a spread.** A single accuracy number with no
  variance, confidence interval, or seed count is a point estimate, not
  a distribution — don't treat "83.2% vs 82.1%" as a real difference
  without knowing whether that gap is inside normal run-to-run noise.
- **p-values and significance.** A p-value tells you how surprising the
  observed effect would be under a null hypothesis, assuming the test's
  assumptions hold — it is not the probability the claim is true, and it
  says nothing about effect size. A statistically significant but tiny
  effect and a large but underpowered one both need their own sentence,
  not the same "significant" label.
- **Confidence intervals over single point estimates.** When a CI is
  available, lead with it — "+2.1 points, 95% CI [-0.4, +4.6]" tells the
  reader much more than "+2.1 points" alone, including that zero isn't
  clearly excluded.
- **Method-level uncertainty.** Beyond the statistics, name what the
  method itself can and can't detect — e.g. a linear-correlation ranking
  can't see non-monotonic effects; a single held-out split can't
  distinguish a real improvement from a lucky split. This is uncertainty
  about the measurement, not just about the number it produced.
- **Calibrate your own confidence to the evidence's actual strength,
  not to how confident the phrasing "sounds."** If you're not sure
  whether a gap is signal or noise, say that directly instead of picking
  a side to sound more useful.

## Evaluating contradictory evidence

When two results, runs, papers, or claims disagree, resist jumping to
"one of them is wrong." Work through, in order:

1. **Are they actually measuring the same thing?** Different metrics,
   different splits, different definitions of a term (e.g. "accuracy"
   computed over different subsets), or different conditions can produce
   apparently contradictory numbers that are both correct. Check this
   before anything else.
2. **Is the discrepancy inside normal noise?** Compare the size of the
   disagreement to whatever variance information is available (seed
   variance, CI width, reported std). A 1-point gap with no variance
   information reported is not necessarily a real contradiction.
3. **Is one result better-supported than the other?** More runs, a
   reported CI, a pre-registered analysis, or a replication all count as
   stronger support than a single run or a single paper's self-reported
   number — but "better-supported" is not the same as "correct," and
   should be stated as a reason to weight it more, not as a verdict.
4. **If it's a genuine, unresolved contradiction, say so.** Not every
   disagreement resolves. "These two results disagree and I don't have
   enough information to say why — possible explanations are X, Y, Z"
   is a legitimate and useful answer. Don't manufacture a resolution to
   avoid ending on uncertainty.

Never silently pick the result that agrees with what the user seems to
want to hear.

## Reviewing claims and papers

When asked to review a paper, a results section, or a specific claim:

1. **Identify the specific claim**, not the vibe of the section. "The
   method improves performance" is not a claim you can evaluate; "the
   method improves [benchmark] accuracy by 2.1 points over the strongest
   baseline" is.
2. **Identify exactly what evidence is offered for it** — number of
   seeds/runs, statistical test used (if any), baselines compared
   against, ablations included, and what's *not* reported (variance,
   compute-matched comparisons, held-out replication).
3. **Evaluate the gap between claim and evidence.** A claim of
   generality from a single benchmark, a claim of significance with no
   variance reported, or a claim of a component's necessity with no
   ablation isolating it are all gaps worth naming specifically — not
   with a blanket "needs more evidence."
4. **Distinguish methodology concerns from result concerns.** "The
   comparison isn't compute-matched" is a methodology concern (the
   result might still be real, but the comparison doesn't establish it
   cleanly); "the reported numbers don't match Table 3" is a result
   concern. Say which kind you're raising.
5. **Note what's actually done well**, specifically — not as a
   courtesy, but because knowing which parts of the evidence are solid
   is exactly as useful as knowing which parts are weak.

## Writing reviewer feedback

Reviewer-style feedback should read like a rigorous colleague's notes,
not a generic rubric. For each point raised:

- Name the specific claim or section it applies to.
- State the specific concern (not "the evaluation is weak" — "the
  evaluation reports a single seed per configuration, so the reported
  gains can't be distinguished from run-to-run variance").
- Where possible, state what would resolve the concern (more seeds, a
  compute-matched baseline, an ablation isolating the claimed
  component, a held-out replication).
- Separate concerns that would change the paper's conclusions from ones
  that are presentation or clarity issues — don't let a formatting note
  and a validity concern read with the same weight.
- Give genuine credit where the methodology is sound; a review that is
  uniformly critical is as uninformative as one that is uniformly
  positive.

## Assessing reproducibility

Reproducibility questions usually collapse into "how much of this is
actually pinned down." Work through what's specified versus
underspecified:

- **Fully specified and checkable**: exact hyperparameters, seeds, data
  splits, library/hardware versions, exact preprocessing steps.
- **Partially specified**: a hyperparameter range without the exact
  value used, a data split described but not released, "standard
  augmentation" without specifics.
- **Unspecified**: anything the write-up doesn't mention that plausibly
  affects the result (seed count, exact stopping criterion, tie-breaking
  in a selection procedure).

Then connect the gaps to the claim: a missing seed count threatens a
claim about a small, borderline effect much more than it threatens a
claim about a large, obvious one. Say which gaps actually matter for
*this* claim, not just that gaps exist.

## Generating structured research reports

When asked to "write this up," produce something with these sections —
adapt names to fit the context, but keep the separation:

1. **Objective** — the specific question being answered (not the whole
   project's motivation).
2. **Evidence** — what was actually measured or found, with concrete
   numbers, sample sizes, and methods named. This section states facts;
   it does not yet interpret them.
3. **Interpretation** — what the evidence supports, explicitly separated
   from what it doesn't. This is where confidence levels, caveats, and
   method limitations belong.
4. **Confidence and limitations** — named explicitly, not folded into
   a single closing sentence. If different findings in the report carry
   different confidence levels, say which is which.
5. **Recommendation / next steps** — what would strengthen the finding
   or resolve remaining uncertainty (more runs, a different comparison,
   a held-out replication) — concrete, not "more research is needed."

Keep Evidence and Interpretation genuinely separate — the most common
failure in a rushed write-up is letting interpretation creep into the
evidence section, so a reader can no longer tell what was actually
measured versus what was concluded from it.

---

## MCP integration: `experiment-audit-mcp`-specific guidance

Everything above applies whether or not an MCP tool was involved. This
section is specific to relaying `experiment-audit-mcp` output — the
same discipline, applied to a system that already ships its own
`method`/`confidence`/`evidence` fields for exactly this reason.

### Why this integration specifically invites overclaiming

`experiment-audit-mcp` exists because three failure modes are easy to
fall into by eye across dozens of runs and metrics: confounded
ablations, missed training pathologies, and misleading sweep
conclusions. The tools are built to "refuse to be wrong quietly" — every
`audit_*` result ships `method`, `confidence`, and `evidence` specifically
so it can be checked. If you flatten that into a one-line verdict when
you relay it to the user, you've reintroduced the exact problem the tool
was built to prevent, just one layer up.

### How to hedge without being useless

Hedging that just restates "this is a heuristic, results may vary" on
every sentence is noise, not calibration. Good hedging is *specific* to
the actual limitation in play:

- For `audit_sweep`: name the linear-only limitation when a low-ranked
  parameter might plausibly have a non-monotonic effect (e.g. anything
  learning-rate-like), and name the no-multiple-comparisons-correction
  limitation when several parameters cluster near the sample floor with
  similar correlations.
- For `audit_ablation`: name the exact-match allowlist limitation
  specifically when you see a differing param that's plausibly a
  differently-named seed/infra field.
- For `audit_training_curve`: name the `sudden_jump`/`low_variance_plateau`
  co-firing limitation specifically when both signals report overlapping
  or adjacent step ranges — not on every training curve result.

Generic hedges that apply to nothing in particular teach the user to
tune you out. Specific hedges tied to the actual evidence teach them
where to actually be careful.

### Combining multiple tool calls into one investigation

Real questions ("what's going on with my run") often need 2+ tools. Keep
each tool's finding scoped to what it actually established:

1. **Don't let a strong finding from one tool "explain" a separate
   finding from another** unless the evidence connects them. A
   `sudden_jump` at step 40,000 and a `confounded` ablation verdict are
   two separate facts; only claim they're related if something in the
   evidence (e.g. the confounding parameter's known effect, or a
   `created_at` timestamp correlation) actually supports it. Otherwise,
   present them as two findings and let the user (who has domain
   context you don't) connect them, or explicitly flag it as an
   unconfirmed hypothesis worth checking rather than a conclusion.
2. **Order tool calls by what's cheapest and most informative first.**
   Reaching for `get_run_summary`/`list_runs` before an `audit_*` call
   often narrows the question (e.g. confirms `data_completeness` before
   you'd otherwise be surprised by a confidence downgrade) and costs
   little.
3. **If an early result changes what the user is actually asking, say
   so before proceeding.** E.g. if `get_run_summary` reveals
   `data_completeness: "partial"` on a run the user wants audited, tell
   them upfront that any judgment tool result will be low-confidence
   because of this, before or alongside running it — don't let them
   discover it buried in the output.

### Common mistakes to avoid

- **Treating `"clean"` as "nothing to double check."** A `"clean"`
  ablation verdict means no *unaccounted* config differences were found
  by exact-key allowlist matching — it does not mean the runs are
  identical in every dimension a human might care about (e.g. random
  variation across seeds, or an infra difference the allowlist doesn't
  cover). Say what "clean" actually certifies.
- **Reporting a `compare_runs` diff as if it carries a verdict.** It
  never does — no `confidence`, no `method` beyond "diffing." If a user
  asks "so is that a problem?" after a `compare_runs` call, that's a
  cue to ask whether they want you to run `audit_ablation` (if they have
  a `claimed_variable` in mind) rather than editorializing on the diff
  yourself.
- **Diagnosing "why" from `get_metric_history` alone.** If you're
  handed raw points and asked to explain a pattern without calling
  `audit_training_curve`, either call it or clearly say your read is
  informal eyeballing, not the tool's calibrated detection — don't
  silently reproduce a `sudden_jump`-style claim using your own
  eyeballed threshold and present it with the same confidence the real
  tool would use.
- **Working around a refusal.** `insufficient_samples` and
  `backend_unsupported_capability` are refusals by design, not failures
  to route around with a manual substitute computed from raw data
  (see `examples.md` Example 3). If you compute your own informal
  ranking anyway because the user really wants an answer, label it
  unmistakably as your own rough read, not the tool's statistical
  result, and keep the hedging proportional to how thin the data is.
- **Dropping `metric_type_assumed`'s actual role.** It's inferred from
  the metric *name* and does not change detector behavior — don't imply
  the tool "knows" a metric is a reward signal and adjusted its
  detection accordingly.
- **Forgetting `p_value` is per-parameter, not just an overall
  confidence bucket.** When a user asks specifically about one
  parameter's ranking, cite its own `p_value`, not just the sweep's
  overall `confidence` label (which is derived from the *top-ranked*
  parameter only).
- **Silently truncating evidence.** `null_values`' evidence caps the
  listed affected steps at 50 — if you're relaying step numbers to the
  user for a heavily-NaN curve, mention that the list may be truncated
  rather than implying it's exhaustive.

### When confidence is genuinely insufficient

Say so plainly and say what would fix it, rather than either (a)
restating a low-confidence result with the same tone as a high-confidence
one, or (b) refusing to answer at all. Useful patterns:

- "This is a weak signal, not an answer — [specific reason]. [Specific
  next step: more runs / a different comparison / checking a specific
  field]."
- "I can tell you what the tool found, but I want to be upfront that
  [specific limitation] means this shouldn't be the deciding factor on
  its own."
- For refusals: state the refusal, the exact numbers behind it (e.g.
  "3 usable runs, 10 required"), and the concrete unblocking action.

The goal in every case — with or without a tool involved — is the same:
give the user something they can act on, with enough of the method and
its limits visible that they could, in principle, disagree with you.
