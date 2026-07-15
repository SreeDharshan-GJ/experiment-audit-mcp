# Examples: Worked Conversations and Tool-Selection Cases

This file has two parts. Part 1 covers the `experiment-audit-mcp`
integration — tool selection and relaying calibrated tool output. Part 2
covers the reasoning engine applied with no MCP call at all — paper
review, statistical claims, reproducibility, contradictory evidence, and
structured write-ups. The discipline in `prompts.md` applies to both;
these examples show it in practice.

---

# Part 1 — The MCP integration (`experiment-audit-mcp`)

These reflect real output shapes from `experiment-audit-mcp` v1.0.0, and
the distractor pairs are taken directly from this project's own
tool-selection eval set (`scripts/tool_selection_prompts.py`) — they
represent cases the server's authors specifically designed to be
confusable, so pay attention to what makes each pair resolve differently.

## Distractor pairs (retrieval/diffing vs. judgment)

These are the sharpest tool-selection tests. In each pair, both prompts
share vocabulary, but only one wants a heuristic verdict.

**Pair A — training curve**
- *"Why did my reward crash partway through training?"* → `audit_training_curve`
  (wants a diagnosis).
- *"What do the reward values actually look like over time for this run?"*
  → `get_metric_history` (wants the raw series, no diagnosis implied).

**Pair B — ablation**
- *"Did I mess up this ablation? I only meant to change the learning
  rate."* → `audit_ablation` (wants a verdict on whether the change was
  clean).
- *"I changed the learning rate for my ablation run — what else differs
  from baseline?"* → `compare_runs` (wants the factual diff; no verdict
  requested, even though the word "ablation" appears). This is the
  softest case in the whole set — a client choosing `audit_ablation`
  here isn't *wrong*, but the more literal read of the request is a diff,
  and defaulting to the lighter tool avoids manufacturing a verdict
  nobody asked for.

**Pair C — sweep**
- *"Which hyperparameter mattered most in my sweep?"* → `audit_sweep`
  (wants an importance judgment).
- *"List every run in my learning-rate sweep so I can eyeball them
  myself."* → `list_runs` (wants enumeration, explicitly says they'll do
  the judging themselves).

**Also worth knowing:** *"My run's data looks incomplete — can you tell
me if it finished ingesting?"* → `get_run_summary` (the answer is in
`data_completeness`, not a judgment call), and *"Has my training run
flatlined? The loss hasn't moved in a while"* → `audit_training_curve`
(this is the `low_variance_plateau` detector's exact use case).

---

## Example 1: Diagnosing a training curve

**User:** "Why did the reward on run xj29fk1a in my mamfac project crash
around step 40,000?"

**Tool call:**
```
audit_training_curve(
  ref={backend: "wandb", entity: "your-team", project: "mamfac", run_id: "xj29fk1a"},
  metric="reward"
)
```

**Result (abridged):**
```json
{
  "schema_version": 2,
  "metric_type_assumed": "reward",
  "signals": [
    {"signal": "sudden_jump", "score": 0.91, "step_range": [40120, 40140],
     "evidence": {"value_before": 0.84, "value_after": 0.11, "z_score": 7.3},
     "confidence": "high"}
  ],
  "method": "threshold-based, see docs/audit-methods.md#training-curve"
}
```

**Good answer to the user:** "There's a real level shift here, not just
noise: the reward dropped from about 0.84 to 0.11 between steps 40,120
and 40,140 — a z-score of 7.3 on the rate-of-change, well above the
detection threshold, so this is flagged high-confidence. No plateau or
oscillation signals fired elsewhere in the curve and there are no logged
NaNs. That step range is where I'd look for a cause — a checkpoint
restore, an LR schedule change, or a logging discontinuity around there."

Notice what this answer does: names the exact numbers from `evidence`,
states the confidence and why, explicitly notes what *didn't* fire
(useful negative information), and stops short of claiming to know the
cause — the tool detects the shift, not its origin.

---

## Example 2: Checking an ablation for confounds

**User:** "Did I mess up my memory-ablation run? Compare mamfac-baseline
and mamfac-no-memory in the mamfac project and check whether the only
real difference is use_memory."

**Tool call:**
```
audit_ablation(
  baseline={backend: "wandb", entity: "your-team", project: "mamfac", run_id: "mamfac-baseline"},
  ablation={backend: "wandb", entity: "your-team", project: "mamfac", run_id: "mamfac-no-memory"},
  claimed_variable="use_memory"
)
```

**Result (abridged):**
```json
{
  "verdict": "confounded",
  "confidence": "high",
  "differing_params": [
    {"param": "use_memory", "baseline_value": true, "ablation_value": false, "likely_intentional": true},
    {"param": "batch_size", "baseline_value": 64, "ablation_value": 32, "likely_intentional": false}
  ],
  "method": "full config diff against claimed_variable; params tagged intentional if name matches claimed_variable or is on the allowlist (seed, device, run name/id)",
  "evidence": {"config_diff": ["..."], "metric_diff": ["..."]}
}
```

**Good answer:** "Yes — this comes back confounded, high confidence.
`use_memory` did change as intended (true → false), but `batch_size`
also silently changed (64 → 32) and isn't on the allowlist of
expected/benign differences, so the reward delta between these two runs
can't be cleanly attributed to memory alone. I'd rerun the ablation with
`batch_size` held fixed before drawing conclusions from the metric
difference." (Then, if the user wants to know what the metric actually
did despite the confound, read `metric_diff` from `evidence` and report
it as "moved by X, but under a confounded comparison" — not as clean
evidence either way.)

**What NOT to do:** don't say "your ablation is broken" as a flat
statement, and don't silently ignore `batch_size` because it seems minor
— a 2x batch size change is a real, plausible confound, not a rounding
artifact.

---

## Example 3: Ranking sweep hyperparameters — refusal case

**User:** "Which hyperparameter mattered most in my lr-sweep sweep?"

**Tool call:**
```
audit_sweep(
  sweep_ref={backend: "wandb", entity: "your-team", project: "mamfac", sweep_id: "lr-sweep"}
)
```

**Result:**
```json
{"error": {"error_type": "insufficient_samples", "message": "Sweep has only 3 usable run(s) (...); audit_sweep requires at least 10 to compute a statistically meaningful importance ranking.", "recoverable": false}, "run_count": 3, "minimum_required": 10}
```

**Good answer:** "I can't give you a reliable ranking here — the sweep
only has 3 usable runs (ones that actually logged the target metric),
and this tool refuses to rank below 10 by design, since a ranking from
that few runs would be more noise than signal. If you want, I can list
the sweep's runs so you can see what's there, or you can let the sweep
run further and I'll re-check."

**What NOT to do:** don't manually eyeball the 3 runs' configs and
metrics and present an informal ranking as if it were a substitute —
that reproduces exactly the "shaky number" the refusal exists to
prevent. If the user explicitly wants your own informal read of 3 runs,
that's fine, but label it clearly as your own eyeballing, not a
statistical result, and keep it far more hedged than the tool's own
output would be.

---

## Example 4: Ranking sweep hyperparameters — success case with a co-variance warning

**User:** "Is learning_rate and batch_size confounded in my grid sweep,
or independently important?"

**Tool call:**
```
audit_sweep(
  sweep_ref={backend: "wandb", entity: "your-team", project: "mamfac", sweep_id: "grid-sweep-1"},
  target_metric="val_accuracy"
)
```

**Result (abridged):**
```json
{
  "sweep_size": 24, "usable_run_count": 22, "target_metric": "val_accuracy",
  "parameter_importance": [
    {"param": "learning_rate", "correlation": 0.71, "rank": 1, "p_value": 0.0002, "warning": "co-varies with batch_size (|r|=0.74)"},
    {"param": "batch_size", "correlation": 0.68, "rank": 2, "p_value": 0.0005, "warning": "co-varies with learning_rate (|r|=0.74)"},
    {"param": "dropout", "correlation": 0.12, "rank": 3, "p_value": 0.59}
  ],
  "excluded_parameters": [{"param": "optimizer", "reason": "non_numeric"}],
  "caveat": "Correlation-based; unreliable with correlated hyperparameters or small sweeps. n=22.",
  "confidence": "high",
  "method": "pairwise Pearson correlation with target_metric; co-variance flagged where |corr| >= 0.7"
}
```

**Good answer:** "They're confounded with each other in this grid —
`learning_rate` and `batch_size` both rank highly (0.71 and 0.68
correlation with val_accuracy) but also correlate strongly with each
other (|r|=0.74), which the tool flags explicitly. That means I can't
tell you which one is actually driving the accuracy difference from this
sweep alone — they moved together across your grid. `dropout` ranks
low (0.12, not significant, p=0.59) but keep in mind this method only
catches linear relationships, so that doesn't rule out dropout mattering
in a non-monotonic way. `optimizer` was excluded because it's a
non-numeric setting (correlation isn't defined for it) — I'd need you to
tell me which optimizer values you used if that's part of what you're
investigating. Overall confidence is high (n=22, top p=0.0002), but that
confidence is about the correlation estimate being statistically real,
not about which of the two co-varying parameters is the true cause."

---

## Example 5: Simple retrieval chain

**User:** "What runs do I have in the mamfac project from the last week,
and give me the full details on whichever one has the highest reward."

**Tool calls:**
1. `list_runs(backend="wandb", project="mamfac", filters={created_after: <7 days ago>})`
2. From the returned lightweight summaries, you don't yet have
   `summary_metrics` — you need `get_run_summary` on each candidate (or
   ask the user to narrow it down if the list is long) to compare reward
   values, since `list_runs` deliberately omits metrics to stay cheap.
3. `get_run_summary(ref=<highest-reward run's ref>)` for the final answer.

This is a good example of **not** reaching for `compare_runs` — the user
wants one run's full detail, not a diff between named runs.

---

## Example 6: Multi-tool investigation, correctly scoped

**User:** "My CARM++ ablation looks off — reward barely changed and the
curve looks choppy. What's going on?"

This genuinely needs two tools, and the key discipline point is **not
letting one tool's finding bleed into a causal claim about the other**:

1. `audit_training_curve(ref=<ablation run>, metric="reward")` — check
   whether "choppy" is real (`high_frequency_oscillation`) and whether
   "barely changed" correlates with a `low_variance_plateau`.
2. `audit_ablation(baseline, ablation, claimed_variable=<whatever the
   user says they meant to change>)` — check whether the small reward
   delta might be explained by a confound rather than the claimed
   variable being genuinely low-impact.

**Good synthesis:** "Two separate things here. The curve itself does
show high-frequency oscillation (sign-flip ratio 0.81, confidence
medium) — so 'choppy' is real, not just your eye. Separately, the
ablation audit comes back `clean` at high confidence — the only config
difference from baseline is the claimed variable, so the small reward
delta likely reflects a genuinely small effect rather than a confound
hiding a larger one. I can't tell you from these two results alone
whether the oscillation itself is affecting the reliability of the
ablation comparison — if you want, I can check whether baseline shows
similar oscillation for comparison." This flags a real limit (the tools
didn't directly test curve stability's effect on the ablation
conclusion) instead of inventing a connection between the two findings.

---

# Part 2 — The reasoning engine with no MCP call

These examples involve no live W&B project — a paper, a pasted table, or
a described experiment. The reasoning discipline is identical to Part 1;
what changes is that you are the one doing the calibrated computation
(or explicitly declining to, when you can't), so you have to be the one
stating your method and confidence out loud.

## Example 7: Reviewing an ML paper's claims

**User:** "This paper claims their new attention variant beats standard
attention by 3.2 points on [benchmark]. Here's the results table and
methods section — is this a strong claim?" *(user pastes the table and
methods text)*

**Approach:** Identify the specific claim (3.2-point improvement on one
benchmark), then check what's actually offered for it: number of seeds,
whether the baseline is compute-matched, whether variance is reported,
whether the gain holds across more than one benchmark.

**Good answer:** "The claim itself is specific and checkable, which is
good — but the support for it is thinner than the headline number
suggests. Three things stand out: the table reports a single run per
configuration (no seeds or variance), so I can't tell whether 3.2 points
is outside normal run-to-run noise for this benchmark. The baseline and
the new method aren't stated to be parameter- or compute-matched — if
the new variant has more parameters, some of the gain could be capacity
rather than the attention mechanism itself. And the improvement is
reported on one benchmark; the methods section doesn't mention testing
generalization elsewhere. None of this means the result is wrong — it
means the paper as written supports a narrower claim ('we observed a
3.2-point gain in this one setup') than the one it's making ('our
method beats standard attention'). The ablation in Table 4 comparing
head count is a genuine strength — that part does isolate the claimed
mechanism cleanly."

**What NOT to do:** don't say "this is a weak paper" as a flat verdict,
and don't invent a p-value or seed count that isn't in the pasted text
to make the critique sound more quantitative than the evidence allows.

---

## Example 8: Writing reviewer feedback

**User:** "Write reviewer comments on the evaluation section I just
showed you."

**Good answer structure** (populated with the paper's actual specifics,
not generic placeholders):

> **Evaluation methodology:** The paper compares against three baselines
> but doesn't state whether hyperparameters for the baselines were tuned
> with the same budget as the proposed method — this matters because an
> undertuned baseline would inflate the reported gain. Recommend stating
> the tuning budget for each method explicitly.
>
> **Statistical validity:** Results are reported as single-run numbers
> across all five benchmarks. With no variance reported, Table 2's
> reported gains (ranging 0.8–3.2 points) can't be distinguished from
> run-to-run noise, especially the smaller ones. Recommend at least 3
> seeds per benchmark with reported standard deviation.
>
> **Strength:** The ablation in Section 5.2 does correctly isolate the
> claimed component — baseline and ablation configs differ only in the
> component under study, which is not always true of ablations in this
> literature.
>
> **Presentation (minor):** Figure 3's y-axis isn't labeled with units.

Notice this separates methodology concerns from result concerns from
presentation notes, gives a concrete fix for each substantive point, and
names a genuine strength rather than being uniformly critical.

---

## Example 9: Evaluating a statistical claim from pasted numbers

**User:** "My model got 84.3% accuracy vs the baseline's 82.1%. That's
better, right?"

**Approach:** This is a point-estimate comparison with no variance
information given — ask for what's missing if it's decision-relevant, or
answer conditionally on what's not yet known.

**Good answer:** "84.3% is a higher number than 82.1%, but whether
that's a *real* improvement depends on information you haven't given me
yet: how many runs/seeds went into each number, and what the run-to-run
variance looks like. A 2.2-point gap from single runs on a benchmark
with typical seed variance in that range could easily be noise rather
than a genuine improvement. If you can share results across a few seeds
for both, I can give you a real read on whether this gap is likely
signal or noise — right now the honest answer is 'higher, but I can't
say if it's meaningfully better.'"

**What NOT to do:** don't compute a made-up significance test on two
point estimates, and don't say "yes, better" without the caveat just
because it's the answer the user seems to want.

---

## Example 10: Checking reproducibility of a described experiment

**User:** "I'm trying to reproduce Experiment 2 from this paper — here's
what they say about setup." *(user pastes a paragraph: model size,
dataset, "standard data augmentation," "trained until convergence,"
no seed mentioned)*

**Good answer:** "A few things here are pinned down well enough to
reproduce directly: the model size and dataset are specific. Two things
are underspecified in a way that could matter: 'standard data
augmentation' doesn't say which specific transforms or their parameters,
and 'trained until convergence' doesn't give a concrete stopping
criterion (a fixed step count, a validation-loss threshold, or an early-
stopping patience would each give different results). No seed count is
mentioned either, so if your reproduction lands within a few points of
theirs, you likely won't be able to tell whether a gap is due to a real
discrepancy or to normal seed variance they didn't report. I'd flag the
augmentation and stopping criterion as the two gaps most likely to
actually move the result — worth trying the paper's official code or
contacting the authors for those two specifically before concluding your
reproduction disagrees with theirs."

---

## Example 11: Detecting contradictory evidence across two sources

**User:** "Paper A says method X improves robustness. Paper B tests the
same method and says it doesn't help. What's going on?"

**Approach:** Work through the "evaluating contradictory evidence"
checklist from `prompts.md` before concluding either paper is wrong.

**Good answer:** "Before concluding one of them is wrong, worth checking
whether they're measuring the same thing. A few possibilities, roughly
in order of how common this actually turns out to be: they may use
different robustness benchmarks or perturbation types (a method can help
against one attack/corruption type and not another) — check whether
Paper A and Paper B evaluate the same perturbation set. They may use
different base architectures or training scales, where an effect that
helps at one scale doesn't at another. Or one may simply have a smaller
sample of test conditions and be more susceptible to variance. If you
can tell me what benchmarks and architectures each paper actually used,
I can narrow this down further — right now, without that, the honest
answer is 'these look contradictory on the surface, but I can't tell you
which one is right, or whether they're actually both right under
different conditions, without more detail on their setups.'"

**What NOT to do:** don't default to "the more recent/more cited paper
is probably right" as a substitute for actually checking whether they
tested the same thing.

---

## Example 12: Interpreting confidence and uncertainty language

**User:** "The paper says 'our method significantly outperforms the
baseline (p < 0.05).' Does that mean it definitely works better?"

**Good answer:** "Not quite — 'p < 0.05' means that if the baseline and
the new method were truly equivalent, results this different (or more
extreme) would happen less than 5% of the time by chance, under this
test's assumptions. It doesn't tell you the size of the improvement, and
it doesn't mean there's a 95% (or any specific) probability the method
is actually better — that's a common misreading of what a p-value
represents. It also doesn't account for how many comparisons the paper
ran in total; if they tested many configurations and only report the
significant one, the true false-positive risk is higher than 5%. Worth
checking: what's the actual effect size (not just significance), and did
the paper pre-specify this comparison or report it after trying several?"

---

## Example 13: Analyzing an ablation study described in prose

**User:** "We removed the attention mechanism and accuracy dropped from
91% to 85%, so attention is clearly important, right?"

**Good answer:** "The direction is consistent with attention mattering,
but 'clearly important' is a bit stronger than a single before/after
number supports on its own. Two things worth checking before locking
that in: is this a single run on each side, or averaged across seeds —
a 6-point drop is large enough that it's probably not pure noise, but
without variance info I can't rule that out with certainty. And did
removing attention change anything else about the model (parameter
count, effective capacity, training dynamics) beyond the mechanism
itself — if removing attention also meaningfully shrinks the model, some
of that 6-point drop could be a capacity effect rather than an attention
effect specifically. If those two things check out — multiple seeds, and
a parameter-matched ablation — then yes, this is solid evidence attention
matters here. As described, I'd call it suggestive evidence with a
reasonably large effect, not yet a fully isolated causal claim."

---

## Example 14: Producing a publication-quality structured report

**User:** "Write up what we found from the hyperparameter sweep and the
training curve issue as a report I can share with my team."

**Approach:** Use the structured report format from `prompts.md`,
populated with the actual findings gathered earlier in the conversation
(e.g. an `audit_sweep` result and an `audit_training_curve` result),
keeping Evidence and Interpretation clearly separate.

**Good report shape:**

> ## Objective
> Determine which hyperparameters drove the accuracy gain in
> grid-sweep-1, and diagnose the reward discontinuity observed around
> step 40,000 in run xj29fk1a.
>
> ## Evidence
> - Sweep audit (n=22 usable runs, target_metric=val_accuracy):
>   learning_rate (r=0.71, p=0.0002) and batch_size (r=0.68, p=0.0005)
>   both rank highly and co-vary with each other (|r|=0.74). dropout
>   ranks low (r=0.12, p=0.59). optimizer excluded (non-numeric).
> - Training curve audit on xj29fk1a: sudden_jump signal at steps
>   40,120–40,140 (reward 0.84 → 0.11, z=7.3, confidence high). No other
>   signals fired.
>
> ## Interpretation
> learning_rate and batch_size appear jointly important, but the sweep
> can't attribute the effect to one versus the other since they moved
> together across the grid — a follow-up sweep varying them
> independently would be needed to disentangle this. dropout's low rank
> doesn't rule out a non-linear effect (Pearson correlation is
> linear-only). The reward discontinuity in xj29fk1a is a real,
> high-confidence level shift, not noise; its cause isn't established by
> this audit alone.
>
> ## Confidence and limitations
> Sweep ranking: high confidence in the correlations being statistically
> real, low confidence in disentangling the two co-varying parameters.
> Training curve finding: high confidence the shift occurred; no
> confidence attached to its cause.
>
> ## Recommended next steps
> Run a follow-up sweep holding batch_size fixed while varying
> learning_rate (and vice versa) to disentangle the two. Inspect logs
> around step 40,120–40,140 on xj29fk1a for a checkpoint restore or
> schedule change.

This keeps every number traceable to its source, keeps interpretation
visibly separate from evidence, and states differentiated confidence
levels per finding rather than one blanket confidence for the whole
report.
