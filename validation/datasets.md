# Datasets — Sourcing, Sizing, and Versioning for Empirical Validation

Operationalizes `research/04_benchmarks/benchmark-plan.md` §3.2-§3.3
(what can be reused, what must be created) into exactly what to pull, how
much, and how to freeze it. **Does not re-derive** why Open RL Benchmark
is the primary source, why `adversarial_cases.py`'s six existing cases
can't be repurposed, or why fANOVA isn't a ground-truth oracle — those
arguments live in the benchmark plan and are cited, not repeated.

---

## 1. Sourcing principles (inherited)

- **Real data over synthetic wherever possible** (benchmark plan §1.5)
  — a case's realism cannot be assumed just because it's plausible-looking;
  synthetic cases are permitted only where no real-data example of a
  specific documented failure mode can be found (§2-§4 below note where
  this applies).
- **Public, citable, re-pullable** (benchmark plan §3.9) — a third party
  must be able to fetch the exact same raw data. Open RL Benchmark
  qualifies (public, W&B-hosted, versioned); privately-held data does
  not, and should not be used even if convenient.
- **Datasets and labels are frozen separately.** A dataset version
  (this file) fixes which *cases* exist. A label version
  (`annotation-guidelines.md` §7) fixes what the panel said about them.
  Re-labeling a frozen dataset produces a new label version, not a new
  dataset version; adding/removing cases produces a new dataset version
  and requires re-labeling.

## 2. `audit_ablation` dataset

**Target size:** 40-60 real baseline/ablation pairs for the full set;
10-15 for the Phase 2 pilot (`validation-plan.md` §5), drawn from the
same pool but excluded from the full-set count once the pilot completes
(re-labeled once as part of the full run, not double-counted).

**Sourcing:**
- Open RL Benchmark / CleanRL corpus: identify run groups sharing an
  algorithm and environment that differ by exactly one or two
  hyperparameters (same algorithm, different seeds; same algorithm,
  single hyperparameter varied) — the corpus's own full config logging
  (benchmark plan §1.5) makes these pairs directly extractable via
  `WandbBackend` against the public project.
- Target a mix across at least 3 distinct algorithm/environment
  combinations, to avoid the ground-truth set being dominated by one
  algorithm's particular logging conventions.

**Stratification (required, not optional):**
- ~40% "obvious" cases (single clearly-intentional or clearly-confounded
  diff) — establishes a floor the tool should not fail.
- ~40% "moderate" cases (2-3 differing params, mixed allowlisted/not)
  — the realistic middle of the distribution.
- ~20% adversarial trap cases (below).

**Adversarial trap cases (from benchmark plan §3.8, listed here for
completeness, not redesigned):**
1. Allowlist-too-narrow: a real, reproducibility-irrelevant differing
   param outside `ALLOWLIST_PARAMS` (logging dir, wandb tag, timestamp).
2. Allowlist-too-permissive: a field literally named `seed` that is
   semantically not a model-init seed.
3. At least one case drawn from the panel's own `contested` bucket
   (populated only after Phase 2/3 labeling begins — see note below).

**Note on trap case 3:** the contested-case trap requires labels to
exist before it can be selected, so this specific case is added to the
frozen set in a follow-up minor version after Phase 2's pilot produces
at least one contested result, rather than blocking the initial freeze.

## 3. `audit_training_curve` dataset

**Target size:** 40-60 real curves for the full set (roughly 10-15 per
signal type, since each curve is scored per-signal, not once overall);
10-15 for the pilot.

**Sourcing:**
- Open RL Benchmark curves spanning multiple algorithms/environments,
  selected to include: curves with no pathology (clean), curves with
  null/NaN-logged points, curves with plateaus, curves with visible
  oscillation, and curves with abrupt level shifts.
- Curves must be selected for *shape*, not for whether they happen to
  cross the tool's own current fixed thresholds — selecting only
  threshold-crossing examples would make the ground-truth set circular
  (it would only ever confirm the tool agrees with itself).

**Stratification (required):**
- One dedicated negative sub-set: curves that resemble a signal but the
  panel should not flag as pathological (a scheduled LR-warmup jump; a
  genuinely converged, intentional plateau). This is the category
  `adversarial_cases.py` currently has zero coverage of (benchmark plan
  §3.3 item 2) and is not optional padding — it is the set that tests
  whether the tool's false-positive rate (`capability-matrix.md` §2) is
  actually measurable at all.
- One dedicated hard-negative sub-set: a real, below-every-threshold
  pathology (slow-onset issue per `related-work.md` §5's reward-hacking
  literature) that the panel flags but no detector currently would.

**Adversarial trap cases (from benchmark plan §3.8):**
1. Scheduled/intended jump misread as `sudden_jump`.
2. Real pathology below every fixed threshold.

## 4. `audit_sweep` dataset

**Target size:** 15-25 real sweeps (smaller than the other two because
each sweep is itself a multi-run unit requiring the 10-run floor
`_MIN_SWEEP_RUNS` already enforces — a "case" here is a whole sweep, not
a single run or pair); 5-8 for the pilot.

**Sourcing:**
- Open RL Benchmark sweep-shaped groups (same algorithm/environment,
  multiple hyperparameters varied across >= 10 runs).
- At least 3-5 sweeps must have a parallel or matched Optuna-run version
  available (or be re-run through Optuna if the original wasn't
  Optuna-managed) specifically to compute the fANOVA divergence
  characterization (`capability-matrix.md` §3) — this subset is smaller
  than the full sweep set because it has a stricter sourcing
  requirement, and that's expected, not a shortfall.
- At least one sweep must include a non-numeric/categorical
  hyperparameter, to exercise the exclusion-to-`excluded_parameters`
  path (`sensitivity.py` review comment 3).

**Adversarial trap cases (from benchmark plan §3.8):**
1. Non-monotonic hyperparameter (interior-optimum learning rate or
   similar) — tests H2 in `capability-matrix.md` §3.
2. Structural-vs-accidental covariance: a sweep containing two
   parameters correlated by mathematical construction (e.g.
   `total_steps = batch_size x num_epochs`).

## 5. Versioning scheme

- Dataset identifier format: `experiment-audit-bench-<tool>-v<major>.<minor>`,
  e.g. `experiment-audit-bench-ablation-v0.1-pilot`,
  `experiment-audit-bench-ablation-v1.0`.
- **v0.x-pilot** = Phase 2 pilot sets (10-15 cases per tool). Never used
  for a published Phase 4 report — pilot sets exist to validate the
  *protocol*, not to produce a citable metric (`validation-plan.md` §5
  Phase 2).
- **v1.0** = first full frozen set per tool, used for the first Phase 4
  report.
- Any change to case membership (add/remove/replace a case) bumps the
  minor version and requires the affected cases to be re-labeled
  (`annotation-guidelines.md` §7); it does not require re-labeling
  unaffected cases.
- Every frozen version ships with: the raw case data (or, for Open RL
  Benchmark cases, the exact W&B run IDs/project path needed to re-pull
  it — not a copy, per benchmark plan §3.9's reproducibility requirement),
  a manifest listing which cases are "real" vs "hand-constructed" and
  why (per §1's sourcing principle), and the stratification breakdown
  (§2-§4 above) so a reviewer can confirm the required mix was actually
  met, not just claimed.
- **Defect handling:** if a case is later found to be mislabeled as data
  (e.g. the underlying run was actually crashed/corrupted, not a valid
  case at all — a data defect, distinct from a *label* disagreement),
  the fix is a new dataset version with the defect and correction both
  stated in the manifest, never a silent edit — directly following
  MLE-bench's and SWE-bench Verified's precedent (benchmark plan §1.1,
  §1.2, §3.9).

## 6. What is explicitly out of scope for this file

- NLP/CV-domain (non-RL) data sourcing — named as a stated limitation in
  `validation-plan.md` §9, not addressed here. If added later, it should
  get its own dataset version and manifest entry, not be silently mixed
  into the RL-sourced sets in a way that obscures the domain split.
- Any dataset for retrieval-tool or tool-selection validation beyond what
  already exists (`scripts/record_wandb_fixtures.py`,
  `scripts/tool_selection_prompts.py`) — per `capability-matrix.md` §4-§5,
  these are category 1/3 concerns with their own existing tooling, not
  new ground-truth sets this file needs to specify.
