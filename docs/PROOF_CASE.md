# Proof case: a real confounded ablation, actually caught

This is not a synthetic narrative. Every number below comes from an
actually-trained model (`examples/run_experiment.py`, scikit-learn,
no fabricated data), and the verdict comes from calling
`experiment_audit_mcp.analysis.confound.audit_ablation()` directly —
the real, unmodified library function, not a reimplementation or a
mock. Reproduce it yourself:

```bash
pip install -e .
python examples/run_experiment.py   # trains the real models, writes real_results.json
python examples/run_audit.py        # runs the real audit_ablation() against them
```

## The setup

A researcher wants to test one claim: **does a "memory" feature
(a rolling-window statistic carried forward across samples) improve
classification accuracy?** `claimed_variable = "use_memory"`.

Setting up the "no memory" condition, they copy their config from a
different experiment template — one that also happened to use a
larger `batch_size`. This is not a contrived mistake; it's exactly how
config drift actually happens in real projects: nobody sits down and
decides to change `batch_size`, it just rides along in a copy-paste.

Two real runs, trained for real:

| run | use_memory | batch_size | accuracy |
|---|---|---|---|
| baseline | `True` | 32 | **0.969** |
| ablation | `False` | 512 | **0.942** |

## What a human skimming this would conclude

*"Removing the memory feature cost 2.7 accuracy points."*

That's the naive read, and it's what would go in a paper or a report
if nobody checked the configs line by line.

## What actually happened

A third, verification-only run (not something a real researcher would
normally run, but useful for checking the tool was right) isolates the
memory feature's true effect by holding `batch_size` fixed at 32:

| run | use_memory | batch_size | accuracy |
|---|---|---|---|
| control | `False` | 32 | 0.957 |

This decomposes the naive 2.7-point delta:

- **True effect of removing memory:** 0.969 − 0.957 = **1.2 points**
- **Effect of the accidental batch_size change alone:** 0.957 − 0.942 = **1.5 points**

More than half of the "memory effect" the naive comparison would have
reported was actually the unrelated batch_size confound.

## What `audit_ablation()` actually output

Run against only the two runs a real researcher would actually have
(baseline and ablation — not the control, which exists here purely to
verify the tool was right):

```json
{
  "verdict": "confounded",
  "confidence": "high",
  "differing_params": [
    { "param": "use_memory", "baseline_value": true, "ablation_value": false, "likely_intentional": true },
    { "param": "batch_size", "baseline_value": 32, "ablation_value": 512, "likely_intentional": false }
  ]
}
```

`batch_size` isn't on the allowlist (`seed`, `device`, run name/id
fields), so it's correctly flagged as an unaccounted-for difference,
and the verdict is `confounded` at `high` confidence — before anyone
needed to run the control condition to find out the confound was real
and sizeable.

## Why this matters

The tool didn't need the control run to catch this. It caught it from
the config diff alone, the moment the two runs were compared — exactly
the point at which a human, skimming a results table with two
accuracy numbers in it, has nothing to go on but "memory helped by
2.7 points" and no reason to suspect otherwise.

## Independently reproduced on 2026-07-11, exact match
E:\mcp project\experiment-audit-mcp\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (60) reached and the optimization hasn't converged yet.
  warnings.warn(
E:\mcp project\experiment-audit-mcp\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (60) reached and the optimization hasn't converged yet.
  warnings.warn(
E:\mcp project\experiment-audit-mcp\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (60) reached and the optimization hasn't converged yet.
  warnings.warn(
baseline  (use_memory=True,  batch_size=32): acc=0.9690
ablation  (use_memory=False, batch_size=512): acc=0.9420
control   (use_memory=False, batch_size=32): acc=0.9570
naive delta (baseline - ablation): +0.0270
true memory-only delta (baseline - control): +0.0120
confound's own contribution (control - ablation): +0.0150
=== REAL audit_ablation() OUTPUT (unmodified library function) ===
{
  "schema_version": 1,
  "verdict": "confounded",
  "confidence": "high",
  "differing_params": [
    {
      "param": "use_memory",
      "baseline_value": true,
      "ablation_value": false,
      "likely_intentional": true
    },
    {
      "param": "batch_size",
      "baseline_value": 32,
      "ablation_value": 512,
      "likely_intentional": false
    }
  ],
  "method": "full config diff against claimed_variable; params tagged intentional if name matches claimed_variable or is on the allowlist (seed, device, run name/id fields); see docs/audit-methods.md#ablation",
  "evidence": {
    "config_diff": [
      {
        "param": "use_memory",
        "values": {
          "wandb/demo/memory-ablation-demo/baseline-001": {
            "present": true,
            "value": true
          },
          "wandb/demo/memory-ablation-demo/ablation-002": {
            "present": true,
            "value": false
          }
        }
      },
      {
        "param": "batch_size",
        "values": {
          "wandb/demo/memory-ablation-demo/baseline-001": {
            "present": true,
            "value": 32
          },
          "wandb/demo/memory-ablation-demo/ablation-002": {
            "present": true,
            "value": 512
          }
        }
      }
    ],
    "metric_diff": [
      {
        "metric": "accuracy",
        "values": {
          "wandb/demo/memory-ablation-demo/baseline-001": 0.969,
          "wandb/demo/memory-ablation-demo/ablation-002": 0.942
        },
        "delta": 0.027000000000000024
      }
    ]
  }
}