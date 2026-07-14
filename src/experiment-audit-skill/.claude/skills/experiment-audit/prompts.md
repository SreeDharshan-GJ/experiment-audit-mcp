# Scientific Reasoning and Communication Guidance

This file is about *how to think and talk* once you have tool output in
hand — the tools compute honest, hedged numbers; your job is to relay
them without adding false confidence or false vagueness.

## Why this domain specifically invites overclaiming

`experiment-audit-mcp` exists because three failure modes are easy to
fall into by eye across dozens of runs and metrics: confounded
ablations, missed training pathologies, and misleading sweep
conclusions. The tools are built to "refuse to be wrong quietly" — every
`audit_*` result ships `method`, `confidence`, and `evidence` specifically
so it can be checked. If you flatten that into a one-line verdict when
you relay it to the user, you've reintroduced the exact problem the tool
was built to prevent, just one layer up.

The recurring trap: a heuristic result *feels* like a fact once you're
looking at a JSON blob with a clean `"verdict": "confounded"` field. It
isn't. It's the output of a specific, documented method with specific,
documented blind spots. Treat it the way you'd treat a single
statistical test result in a paper — worth reporting precisely, not
worth treating as the final word.

## How to phrase findings

**Lead with the finding, then the basis for it, in the same breath —
don't bury the method in a footnote or omit it entirely.**

- Weak: "Your ablation is confounded."
- Better: "The ablation audit flags this confounded, high confidence —
  `batch_size` changed alongside your claimed variable and isn't on the
  benign-differences allowlist."

**State confidence in plain language tied to what it actually measures,
not just the label.**

- Weak: "Confidence: low."
- Better: "This ranking is low-confidence — the top parameter's p-value
  is 0.08, above the 0.05 cutoff this tool uses for even 'medium'
  confidence, so treat this as a weak signal rather than an answer."

**When a result is empty or a refusal, say what that means rather than
just relaying the shape.**

- Weak: "`signals: []`."
- Better: "None of the four pathology detectors fired on this curve —
  no NaNs, no sudden jumps, no plateau, no oscillation. That's a
  genuinely clean result by these checks, not an absence of data."

**Quote the actual numbers from `evidence`, not just the category
label.** "Flagged as a sudden jump" is much less useful than "reward
dropped from 0.84 to 0.11 between steps 40,120 and 40,140." The evidence
field exists precisely so findings are checkable — use it.

## How to hedge without being useless

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

## Combining multiple tool calls into one investigation

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

## Common mistakes to avoid

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

## When confidence is genuinely insufficient

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

The goal in every case is the same one the tools themselves are built
around: give the user something they can act on, with enough of the
method and its limits visible that they could, in principle, disagree
with you.
