# Examples: Worked Conversations and Tool-Selection Cases

These examples reflect real output shapes from `experiment-audit-mcp`
v1.0.0, and the distractor pairs are taken directly from this project's
own tool-selection eval set (`scripts/tool_selection_prompts.py`) — they
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
whether the oscillation itself is affecting the reward comparison's
reliability — if you want, I can check whether baseline shows similar
oscillation for comparison." This flags a real limit (the tools didn't
directly test curve stability's effect on the ablation conclusion)
instead of inventing a connection between the two findings.
