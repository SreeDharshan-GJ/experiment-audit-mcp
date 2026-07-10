# Pain Points — Evidence-Backed Catalogue of Real Researcher Friction

**Status:** First real pass, companion to `researcher-workflows.md`. Each
entry below is a specific, named friction point with at least one
independent source, mapped (where applicable) to the Experiment Audit tool
that already addresses it, partially addresses it, or does not address it
at all. Pain points with no current tool mapping are flagged as such
explicitly rather than silently omitted -- an honest gap list is more
useful than a list that only shows tools in a flattering light.

---

## 1. "I lost track of which run is which" (volume/organization pain, general ML)

**Evidence:** A first-hand practitioner account describes exactly this
failure spiraling out of manual tracking: <cite index="54-1">after three days I had 47 notebook cells with different model configurations, a spreadsheet where I was manually logging results, and absolutely no idea which combination of preprocessing, model, hyperparameters, and threshold had given me the best score</cite>.
The same account frames the fix as adopting a tracking tool at all, not as
adding judgment on top of one: <cite index="54-1">that weekend I rebuilt the entire pipeline with MLflow</cite>.

**Severity/frequency signal:** Anecdotal but vivid and specific (named
number of cells, named tools tried). One account only -- treat as
illustrative, not as a frequency estimate.

**Tool mapping:** Out of scope for Experiment Audit by design. This is a
*retrieval/organization* pain point, and Experiment Audit's stated
position (`research-progress.md` §1) is explicitly not to compete on
retrieval. `list_runs` / `get_run_summary` provide baseline retrieval but
are not differentiated from what any tracker's UI already does. **No
judgment tool addresses this because none should** -- flagging this
explicitly so it doesn't get treated as an oversight later.

---

## 2. Confounded ablations: the claimed variable isn't the only thing that changed

**Evidence:** Named directly as a recurring methodological pitfall,
independent of any tool's framing: <cite index="24-1">recurring pitfalls in controlled ablation studies include misalignment between ablation design and the original system or context, leading to confounded outcomes; ambiguous ablation boundaries; and incomplete procedural specification</cite>.
Reinforced at the institutional level: <cite index="33-1">the expectation across top venues is that architecture claims come with ablation evidence, not anecdote</cite>,
which raises the cost of a confound slipping through (a weakened or
rejected claim), not just the cost of wasted compute.

**Severity/frequency signal:** Institutional (venue checklist-level), which
is stronger evidence of real, recurring frequency than an anecdote -- venue
policies don't get written in response to rare events.

**Tool mapping:** Directly addressed by `audit_ablation` -- exact-match
allowlist diff of claimed-constant fields (seed, device, run_name, etc.)
against actual field values across the compared pair. **Known, documented
limitation carried over from `research-progress.md` §3:** the allowlist is
name-based (`seed`, not `random_seed`), so a naming-convention mismatch
between a project's config schema and the tool's allowlist produces a
false "no confound found" — a conservative-direction failure the tool's
own design accepts rather than hides, but one this pass did not get new
evidence to resolve (still blocked on live-fixture data, per
`research-progress.md` §11 item 1).

---

## 3. Training curve pathologies that are individually easy to miss, and collectively costly

**Evidence:** RL debugging guidance lists several distinct pathology
classes practitioners are told to watch for by eye: <cite index="23-1">stagnant performance, where the reward curve flattens out early and policy/value loss stop improving</cite>; <cite index="23-1">unstable training or divergence, where metrics suddenly shoot up to NaN or infinity, or performance collapses dramatically after a period of improvement</cite>; and <cite index="23-1">policy collapse, where the agent converges to a near-deterministic suboptimal policy, indicated by rapidly decreasing policy entropy and stagnant rewards</cite>.
Each of these is explicitly described as something a practitioner must
notice, not something existing tooling flags automatically.

**Severity/frequency signal:** Multiple independently-named pathology
classes converging in one widely-used practitioner reference is moderate
evidence this is a recurring, recognized (if not automatically caught)
category of problem, not a single edge case.

**Tool mapping:** `audit_training_curve` covers a documented subset:
null-value detection, sudden-jump detection (global-MAD z-score), plateau
detection (coefficient of variation), and oscillation detection
(sign-flip ratio). This maps well onto the "stagnant performance" and
"unstable training/divergence" pathology classes above. It does **not**
currently cover "policy collapse" as RL practitioners define it (entropy
collapse specifically), which requires a second, correlated metric
(policy entropy) rather than a single curve's own statistics. This is a
concrete, evidence-backed scope gap worth naming plainly rather than
implying `audit_training_curve` is a complete RL-pathology detector.

---

## 4. Sweep "importance" being read as causal when it's correlational

**Evidence:** The market-leading product itself warns against this
directly in its own documentation -- <cite index="44-1">correlations show evidence of association, not necessarily causation, and are sensitive to outliers, especially with a small sample size of hyperparameters tried</cite> --
which is itself evidence the misreading is common enough to warrant a
standing warning. Real users asking basic clarifying questions about the
importance/correlation distinction on W&B's own issue tracker and
community forum (<cite index="45-1">asking about the difference between importance and correlation for a categorical CNN architecture search</cite>; <cite index="46-1">asking whether the number of runs in a Bayesian sweep affects how trustworthy the importance chart is</cite>) are further evidence the caveat doesn't fully prevent the misreading in practice.

**Severity/frequency signal:** Direct vendor documentation warning plus
multiple independent user questions on the exact distinction is
moderate-to-strong evidence this is a recurring point of confusion, not a
one-off.

**Tool mapping:** Partially addressed by `audit_sweep`'s Fisher
z-transform significance testing (distinguishes "correlated" from
"correlated and statistically distinguishable from noise given this many
runs") and its hard 10-run floor (directly answers the "does sweep size
affect trustworthiness" question users are asking, deterministically,
rather than leaving it to intuition). **Not** addressed: `audit_sweep`
uses Pearson correlation only, which is linear-only and cannot capture the
interaction effects that W&B's tree-based importance panel already
handles (see `researcher-workflows.md` §4) -- an interior-optimum
hyperparameter (e.g., a learning rate with an internal best value) will
rank artificially low under Pearson even where a tree-based method would
rank it correctly. `audit_sweep`'s own module docstring already flags
this per `research-progress.md` §3; this pass adds independent evidence
that the market leader has already solved the specific failure mode with
a different method, which sharpens (rather than removes) the honesty
obligation in how this limitation is documented.

---

## 5. Multiple-comparisons / p-hacking in hyperparameter and ablation reporting

**Evidence:** A direct, methodology-focused synthesis states plainly that
<cite index="92-1">to take a p-value at face value requires that just one hypothesis test was conducted, which does not always -- or often -- hold in ML, and, in the authors' experience, explicit correction for multiple comparisons is rare</cite>.
This is a stronger and more specific claim than the general "p-hacking
exists" literature -- it is specifically about ML practice, not science
broadly, and specifically diagnoses the correction step as the one that
gets skipped, not the testing step itself.

**Severity/frequency signal:** A direct claim of practice-wide rarity from
a source studying ML research practice specifically. Single source for the
ML-specific claim, though it sits inside a broader, well-established
statistics literature on multiple comparisons generally.

**Tool mapping:** Explicitly **not** addressed by `audit_sweep` today, and
by design -- `research-progress.md` §3 documents that the module
"explicitly declines to add [a multiple-comparisons correction] silently
rather than inventing an ad hoc one." This pass's finding corroborates
that this is a real, literature-grounded gap worth deciding on
deliberately (already listed as an open methodological question in
`research-progress.md` §7 and a v2/v3 candidate in §10) rather than either
ignoring it or bolting on an arbitrary correction.

---

## 6. Manual, error-prone reproduction of a specific past run

**Evidence:** Framed as a known, general reproducibility problem in
computational research: <cite index="76-1">reproducing experimental results is often challenging due to evolving codebases, incomplete hyperparameter listings, version discrepancies, and compatibility issues</cite>.
The fact that CleanRL/Open RL Benchmark built a dedicated `reproduce`
utility specifically to close this gap -- rather than relying on a tracked
run's logged config being sufficient on its own -- is itself evidence that
*even with full experiment tracking already in place*, reproduction
remains effortful enough to justify dedicated tooling.

**Severity/frequency signal:** Moderate -- one ecosystem (CleanRL/Open RL
Benchmark) building dedicated tooling for this is a real signal, but it is
one ecosystem, and the underlying problem (environment/dependency drift)
is broader software-engineering hygiene, not unique to experiment
*judgment*.

**Tool mapping:** Out of scope for Experiment Audit by design -- this is a
reproduction/retrieval problem (getting a run to run again), not a
judgment problem (deciding whether a comparison between two already-run
experiments is trustworthy). Flagged explicitly as adjacent-but-different
from Experiment Audit's mission, to keep the "not another tracker"
positioning honest rather than stretching scope to cover it.

---

## 7. General experimentation friction from team/org factors, not tooling

**Evidence:** A broader MLOps-focused piece notes <cite index="65-1">studies show 70% of production ML issues are organizational, not technical</cite>,
and separately that unclear ownership (<cite index="65-1">who signs off on releasing a model, who owns a given system in the org chart</cite>) compounds every other friction point.

**Severity/frequency signal:** Cited statistic from a single secondary
source (a DEV Community post aggregating "studies," not itself a primary
study) -- treat as directional, not as a precise, verifiable figure. This
entry is flagged with lower confidence than the others in this document
specifically because the source is a blog aggregating unnamed studies
rather than a named, checkable one.

**Tool mapping:** Out of scope entirely -- organizational and ownership
friction is not something any analysis tool, judgment-layer or otherwise,
can address. Included here only because it is a genuine, if
lower-confidence, finding from this pass's search, and omitting it would
make this document look like it only surfaced pain points convenient for
the product.

---

## Decision Impact

**What assumptions were confirmed?**
- Confounded ablations (pain point 2) and training-curve pathologies
  (pain point 3) are both independently evidenced as real, recurring,
  named failure modes -- not invented problems retrofitted to justify
  `audit_ablation` and `audit_training_curve`.
- The correlation-read-as-causation confusion in sweep analysis (pain
  point 4) is real and documented by the market leader itself, and
  `audit_sweep`'s significance testing and run-count floor map onto real,
  observed user confusion (not hypothetical confusion).
- Multiple-comparisons correction being skipped in practice (pain point 5)
  is corroborated by an ML-practice-specific source, strengthening the
  case that this is a genuine open methodological decision rather than a
  low-priority nice-to-have.

**What assumptions were disproven?**
- None of the five previously-open "still lacking evidence" items from
  `research-progress.md` §7 are fully closed by this pass -- this document
  adds *supporting* evidence for problem existence, not validation of
  Experiment Audit's specific heuristics against real data. That
  distinction should not collapse in future summarization of this
  document: "the problem is real" and "our specific fix is correctly
  calibrated" remain two separate, separately-unresolved claims.

**What should we build because of this?**
- Nothing new is recommended for immediate building. This pass sharpens
  two already-known candidates rather than introducing new scope: (a) a
  documented, literature-grounded decision on multiple-comparisons
  correction for `audit_sweep` (pain point 5, already tracked in
  `research-progress.md` §7/§10), and (b) clearer, more specific limitation
  language in `audit_sweep`'s own docs about Pearson vs. tree-based
  importance, now that a concrete comparison point (W&B's shipped panel)
  exists (pain point 4).

**What should we stop building because of this?**
- Nothing currently planned should stop. This pass does argue against ever
  expanding Experiment Audit's judgment tools toward pain points 1, 6, and
  7 (organization/tracking, reproduction tooling, org-process friction) --
  each is real, but each is retrieval, infrastructure, or organizational in
  nature, not judgment, and building toward them would blur the
  "not another tracker" positioning this project has deliberately held to.
