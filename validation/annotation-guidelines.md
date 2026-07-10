# Annotation Guidelines — Human Expert Ground-Truth Protocol

Operationalizes `research/04_benchmarks/benchmark-plan.md` §3.4 (adapted
from SWE-bench Verified's clarity/correctness/solvability triad) into
instructions an actual annotator can follow without needing to read the
benchmark plan first. This is the document handed to panel members; it
assumes no prior familiarity with this project's internals beyond what's
given here.

---

## 1. Panel composition

- **Minimum 3 annotators per case**, all cases in a given tool's dataset
  labeled by the same fixed panel where feasible (rotating panel members
  across cases within one tool's set makes inter-annotator agreement
  harder to interpret; avoid unless panel size genuinely requires it, and
  disclose in the report's provenance section if it happens).
- Annotators must be ML researchers with hands-on experience running and
  debugging training experiments (not necessarily in RL specifically —
  general experience with reading training curves, config diffs, and
  sweep results is what's being tested against, not RL-domain expertise
  per se, though RL familiarity helps given the dataset's RL skew,
  `datasets.md` §6).
- **Annotators must not be this project's own tool designers or
  engineers.** The point of an independent panel is that it isn't
  anchored to how `audit_ablation`/`audit_training_curve`/`audit_sweep`
  were designed to think about the problem.

## 2. Blinding procedure (mandatory)

- Annotators see **only the raw data**: for `audit_ablation`, the two
  runs' full config and summary metrics (i.e., the same input
  `compare_runs` would show) plus the stated `claimed_variable`; for
  `audit_training_curve`, the raw metric history (steps + values,
  including any nulls) plus the metric name; for `audit_sweep`, the raw
  per-run configs and target-metric values across the sweep.
- Annotators **never see**: the tool's own verdict, confidence, evidence,
  or method string, at any point before submitting their label. This is
  not merely "don't mention it" — the annotation form/interface must not
  render it, to prevent accidental exposure.
- Annotators are told the *general* purpose (evaluating whether this data
  is a clean test of a claimed variable / contains a pathology / shows a
  meaningful hyperparameter relationship) but not which specific project
  or tool the labels will validate, to reduce the chance of guessing at
  and anchoring toward an expected "house" answer.
- **Deviations must be disclosed.** If blinding is broken for any case
  (e.g. an annotator recognizes a run from their own prior work), that
  case is flagged and excluded from the panel's agreement computation for
  that annotator, not silently kept.

## 3. `audit_ablation` labeling rubric

For each baseline/ablation pair, the annotator sees: both runs' full
config dicts, both runs' summary metrics, and the stated
`claimed_variable`. They assign exactly one label:

- **`clean`** — every config difference between the two runs is either
  the claimed variable itself, or a parameter you'd consider
  reproducibility-noise (not a substantive experimental variable) even
  if you personally wouldn't have named the same allowlist the tool uses.
  Use your own judgment about what counts as noise — you are not asked
  to apply `ALLOWLIST_PARAMS` yourself, only to judge the pair
  independently.
- **`confounded`** — at least one config difference exists that is
  neither the claimed variable nor something you'd dismiss as noise; a
  reader could not attribute the metric difference to the claimed
  variable alone.
- **`uncertain`** — you cannot determine cleanliness from the data shown
  (e.g. no config differences are visible at all, which could mean a
  truly identical run or missing config logging).
- **`contested`** — not a label you personally choose; this is assigned
  automatically when the panel's individual labels don't reach majority
  agreement (§5). Do not attempt to predict or avoid triggering it.

Three properties to check per case before finalizing a label (adapted
from SWE-bench Verified's triad, benchmark plan §3.4):

1. **Realism** — does this look like real research data (messy,
   plausible field names, realistic values), not an artificially clean
   or artificially confusing construction? Note your impression even if
   it doesn't change your label; the dataset assembler uses this signal
   to catch unrealistic cases in future dataset versions.
2. **Determinacy** — are you confident, or is this genuinely a coin flip
   for you? Low personal confidence should be reflected honestly (§4's
   confidence field), not resolved by picking whichever label feels
   safer.
3. **Label-confidence-under-added-context** — would your label plausibly
   change if you knew the researcher's actual intent? If yes, note it —
   this flags cases where the ground truth is inherently
   context-dependent, which is itself useful information, not a defect
   in your labeling.

## 4. `audit_training_curve` labeling rubric

For each curve, the annotator sees the full step/value series (including
nulls) and the metric name. For **each of the four signal types
separately** (not one combined judgment), assign:

- **Present** — you would flag this signal on this curve if a colleague
  showed it to you and asked "does anything look off here."
- **Absent** — you would not flag it.
- **Uncertain** — genuinely unclear (e.g. borderline plateau length).

Explicit instruction for negative cases (`datasets.md` §3): some curves
are deliberately included that *resemble* a signal but shouldn't be
flagged — e.g. a large jump that coincides with a known LR-warmup restart
pattern, or a flat region that represents genuine convergence rather than
stalled training. Annotators are given the same contextual information a
real researcher glancing at a dashboard would plausibly have (e.g. "this
is a standard cosine-with-warmup schedule" where relevant) — the point is
not to trick annotators with information withheld from a real user, but
to test whether the signal, on its own, would still look like a false
positive to someone reading the curve shape without that context. Record
both: (a) would you flag this from the curve shape alone, and (b) does
the additional context change your answer. Both are used in the false
positive analysis (`capability-matrix.md` §2).

## 5. `audit_sweep` labeling rubric

For each sweep, the annotator sees all runs' configs and target-metric
values. Two separate judgments:

1. **Importance ranking** — rank the varied hyperparameters by how much
   you believe each one drove the target metric, using whatever method
   you'd normally use (visual inspection, your own quick correlation
   check, domain intuition) — you are not required to use Pearson
   correlation or any specific statistic; the point is capturing your
   independent judgment, not replicating the tool's method.
2. **Covariance judgment** — for any pair of hyperparameters that appear
   to move together across the sweep, judge whether that's **structural**
   (mathematically or logically related by how the sweep was
   constructed — e.g. one is derived from the other) or **accidental**
   (happened to correlate in this particular sweep's sampled values, no
   inherent relationship).

## 6. Disagreement handling and inter-annotator agreement

- **Majority vote (>=2 of 3, or the equivalent majority for larger
  panels) determines the recorded label**, except:
- **No majority → `contested`.** Reported and scored as its own bucket
  (`validation-plan.md` §4 principle 3), never folded into the nearest
  majority label.
- **Compute Fleiss' kappa** across the full panel for each tool's dataset
  (pilot and full separately — §7). Interpretation bands (standard
  Landis & Koch scale, stated here so there's no ambiguity about what
  counts as acceptable):
  - kappa >= 0.60: acceptable — proceed past the Phase 2 gate
    (`validation-plan.md` §5).
  - 0.40 <= kappa < 0.60: marginal — proceed only with the specific
    sources of disagreement documented and a stated reason the gate is
    still being passed (e.g. a genuinely ambiguous case category, not a
    protocol defect); otherwise revise and re-pilot.
  - kappa < 0.40: do not proceed. Revise the rubric (clearer label
    definitions, better worked examples in §8) and re-pilot on a fresh
    10-15 case sample — reusing the same pilot cases after revision would
    let annotators anchor on their own earlier answers.
- Report kappa with a 95% CI where sample size allows a meaningful one;
  for pilot-sized samples (10-15 cases), state the small-sample caveat
  explicitly rather than presenting the point estimate alone
  (`validation-plan.md` §9).

## 7. Data handling and output schema

- Raw per-annotator labels are recorded and published alongside the
  aggregated majority label — never only the aggregate
  (`validation-plan.md` §4 principle 4, benchmark plan §3.7).
- Minimum recorded fields per case per annotator: case ID, dataset
  version, annotator ID (pseudonymous is fine — consistency across cases
  matters more than real identity), label, the tool-specific confidence
  fields from §3-§5, free-text rationale (short, optional but
  encouraged — useful for the qualitative error-pattern analysis in
  `validation-plan.md` §8), timestamp.
- Label version identifier: `<dataset-version>-labels-v<n>` (e.g.
  `experiment-audit-bench-ablation-v1.0-labels-v1`) — bumped whenever
  labels are added, corrected, or a contested case is resolved by a
  larger panel (§6), independent of the dataset version itself
  (`datasets.md` §5).

## 8. Worked examples (to be filled in during Phase 2 piloting)

This section is intentionally a placeholder in this initial pass — worked
examples should be drawn from real pilot cases once Phase 1's dataset
exists (`validation-plan.md` §5), not invented synthetically here, since
a hand-built illustrative example risks looking cleaner than the real
data annotators will actually see and could mis-calibrate expectations.
When Phase 2 runs, add one worked example per label per tool here,
showing the raw data, the majority label, and a short note on what made
it clear-cut or borderline — this becomes the reference annotators
consult mid-task if unsure, and should be updated (with the dataset/label
version noted) if the pilot's revision cycle (§6) changes the rubric.

## 9. Explicit non-goals of this protocol

- This protocol does not ask annotators to reproduce or approximate
  `audit_*`'s own internal method (Pearson correlation, MAD-based
  z-scores, exact-match allowlists) — doing so would just be
  re-implementing the tool by hand, not producing an independent
  judgment to calibrate it against.
- This protocol does not use any LLM at any stage of label production,
  per `validation-plan.md` §4 principle 1 and benchmark plan §1.4 — an
  LLM may not be used to pre-filter, pre-label, or suggest labels to
  annotators, even as a "starting point they can overrule," since that
  still anchors the human judgment being collected.
