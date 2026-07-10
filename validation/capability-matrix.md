# Capability Matrix — Per-Tool Validation Specification

For every capability Experiment Audit v1 exposes over MCP. This is the
index `validation-plan.md` §3 points to: one row group per tool, every
field the task brief requires (research question, hypothesis, expected
failure modes, ground truth, evaluation metrics, acceptable FP/FN rate,
required datasets, annotation procedure, reporting template) filled in
specifically for that tool — not a generic template restated eight times.
Where a field says "N/A — see note," that is a deliberate methodological
position (`validation-plan.md` §4 principle 5), not an omission.

Column definitions:
- **Category** — 1 = implementation correctness (already covered, see
  `research/04_benchmarks/benchmark-plan.md` §3.1 and §Part 4), 2 =
  calibration against human judgment (the actual gap this package
  closes), 3 = tool-selection accuracy (already built, not yet run).
- **Datasets / Annotation / Reporting** columns point into
  `datasets.md` and `annotation-guidelines.md` by section number rather
  than repeating their content.

---

## 1. `audit_ablation`

| Field | Specification |
|---|---|
| **Category** | 2 (calibration) — primary validation target, labeled first per priority ranking |
| **Research question** | Given a baseline/ablation run pair and a claimed variable, does the tool's `clean` / `confounded` / `uncertain` verdict match what an independent panel of ML researchers would conclude from the same raw run data? |
| **Hypothesis** | H1: on real Open RL Benchmark ablation pairs, `audit_ablation` achieves F1 >= panel-ceiling-minus-15pp on both `clean` and `confounded` verdict types. H2 (differentiation): `audit_ablation` outperforms an unaided-Claude baseline given the same raw config diff, consistent with AblationBench's ~38-45% unaided-agent figure (`research-progress.md` §5) suggesting LLM-alone ablation judgment is weak. Both are falsifiable; a null result on either is a valid, reportable outcome (`validation-plan.md` §7). |
| **Expected failure modes** | (a) Allowlist too narrow — a reproducibility-irrelevant param outside `ALLOWLIST_PARAMS` (e.g. a logging path, wandb tag, timestamp string) produces a false `confounded`. (b) Allowlist too permissive — a field literally named `seed` that is semantically *not* a model-init seed (e.g. an actively-studied data-shuffling seed) is wrongly waved through as intentional. (c) `uncertain` verdict on a pair where the panel would confidently say `clean` or `confounded` because zero config diff was logged, even though a real difference exists outside the tracked config. (d) Confidence mislabeled `high` on a case the panel finds genuinely contested. All four are the codebase's own documented, self-acknowledged limitations (`confound.py` module docstring; `research-progress.md` §3), not novel failure modes invented for this document. |
| **Ground truth** | Independent panel of >=3 ML researchers, blind to the tool's own verdict, labeling `clean` / `confounded` / `contested` per `annotation-guidelines.md` §3. |
| **Evaluation metrics** | Precision/recall/F1 per verdict type + full 3x3 confusion matrix; confidence calibration (high vs. panel-agreement rate); abstention correctness for `uncertain` (was abstention warranted per the panel); panel-ceiling F1 and unaided-Claude F1 on the same cases (`validation-plan.md` §8). |
| **Acceptable false-positive rate** | Tool says `confounded`, panel says `clean` — errs toward the design philosophy's stated caution ("refuse rather than mislead," `research-progress.md` §2). Target ceiling: **<= 20%** of panel-labeled `clean` cases. Elevated tolerance relative to the false-negative bound below because over-flagging costs researcher time/trust but does not let a bad comparison stand unflagged. |
| **Acceptable false-negative rate** | Tool says `clean`, panel says `confounded` — the more dangerous direction: a bad comparison stands unflagged, undermining the tool's entire trust premise. Target ceiling: **<= 5%** of panel-labeled `confounded` cases. This asymmetry is intentional (benchmark plan §3.6) and should not be averaged into one FP/FN number. |
| **Required datasets** | Real ablation-shaped pairs from Open RL Benchmark; hand-constructed pairs only for specific documented failure modes with no found real-data example; all allowlist-related adversarial trap cases (`datasets.md` §2, benchmark plan §3.8). |
| **Human annotation procedure** | `annotation-guidelines.md` §3 (ablation-specific rubric: realism / determinacy / label-confidence triad, adapted from SWE-bench Verified). |
| **Reporting template** | `validation-plan.md` §8, tool = `audit_ablation`. |

## 2. `audit_training_curve`

| Field | Specification |
|---|---|
| **Category** | 2 (calibration) — labeled second, per priority ranking |
| **Research question** | Given a metric history, do the four detected signals (`null_values`, `sudden_jump`, `low_variance_plateau`, `high_frequency_oscillation`) and their scores match what an independent panel would flag as pathological on the same curve — and, separately, does the panel agree a *clean* curve should produce an empty signal list? |
| **Hypothesis** | H1: `null_values` detection (deterministic presence check) achieves near-ceiling F1 — it is a factual check, not a heuristic, and should be the easiest of the four to validate. H2: the three threshold-based detectors (`sudden_jump`, `low_variance_plateau`, `high_frequency_oscillation`), whose fixed thresholds (z=4.0, CV=0.02, sign-flip ratio=0.7) were chosen at design time and never fit to real data (`research-progress.md` §3), show measurably lower calibration than `null_values` — a negative result (thresholds transfer fine) is equally informative and equally reportable. |
| **Expected failure modes** | (a) A scheduled, *intended* jump (LR-warmup restart, curriculum-stage transition) misread as `sudden_jump` — the tool cannot distinguish "the code did exactly what it was supposed to" from a genuine pathology, by design. (b) A genuine slow-onset pathology (per reward-hacking/specification-gaming literature, `related-work.md` §5) that crosses none of the four fixed thresholds — a true negative the panel would catch and the tool would miss entirely. (c) `sudden_jump` and `low_variance_plateau` co-firing at a plateau's edge — documented, intentional non-exclusivity (`divergence.py` module docstring) that could still register as a false positive to a panel member unaware it's a known, accepted limitation. (d) Metric-type misclassification (substring match on "loss"/"reward") — lower-stakes, does not affect detector behavior. |
| **Ground truth** | Independent panel, blind to tool output, labeling each curve for which of the four signal *shapes* a human would flag, plus explicit `not_pathological` labels for curves that resemble a signal but aren't (the negative cases currently entirely absent from `adversarial_cases.py`, per benchmark plan §3.3 item 2). |
| **Evaluation metrics** | Per-signal (not aggregated across the four) precision/recall/F1; confidence calibration (score-derived high/medium/low vs. panel agreement); false-positive rate on the "resembles but isn't" negative set specifically, reported separately from the general confusion matrix per signal, since this is the category existing tests cannot check. |
| **Acceptable false-positive rate** | Flags a signal on a curve the panel considers normal (e.g. scheduled jump read as pathological). Target ceiling: **<= 15% per signal type**. Tighter than `audit_ablation`'s FP ceiling because a false pathology flag is more likely to send a researcher chasing a non-existent bug. |
| **Acceptable false-negative rate** | Misses a pathology the panel would flag (including "below every fixed threshold" cases). Target ceiling: **<= 10% per signal type**, reported with an explicit caveat distinguishing "missed because below threshold" (an expected, documented scope limit) from "missed due to an implementation defect" (would warrant separate escalation). |
| **Required datasets** | Real training curves from Open RL Benchmark spanning both clean and pathological shapes; scheduled-jump and below-threshold adversarial trap cases (`datasets.md` §3, benchmark plan §3.8); a dedicated negative set of "resembles but isn't pathological" curves. |
| **Human annotation procedure** | `annotation-guidelines.md` §4 (per-signal rubric with explicit negative-case instructions). |
| **Reporting template** | `validation-plan.md` §8, tool = `audit_training_curve`, one section per signal name. |

## 3. `audit_sweep`

| Field | Specification |
|---|---|
| **Category** | 2 (calibration) — labeled third, per priority ranking; also the only tool with a non-panel reference method available (fANOVA) |
| **Research question** | Does the Pearson-correlation-based parameter ranking and covariance-warning output match what an independent panel would identify as the sweep's important/confounded parameters — and separately, how does it diverge from Optuna's fANOVA ranking on the same or a matched sweep? |
| **Hypothesis** | H1: on sweeps with a monotonic hyperparameter-to-metric relationship, `audit_sweep`'s ranking substantially agrees with both the panel and fANOVA. H2: on sweeps containing a non-monotonic hyperparameter (e.g. an interior-optimum learning rate), `audit_sweep` ranks that parameter falsely low relative to both the panel and fANOVA — the module's own docstring already predicts this; this is a test of a self-predicted failure mode, not a novel hypothesis. H3: the covariance warning correctly separates structural covariance (e.g. `total_steps = batch_size x num_epochs`) from accidental covariance, per panel judgment. |
| **Expected failure modes** | (a) Non-monotonic hyperparameter ranked falsely low (H2). (b) Structural covariance flagged identically to accidental covariance, losing a distinction a researcher would want to know apart. (c) Categorical/non-numeric parameters silently excluded — a documented design choice, so a "failure" here is a scope-mismatch test: does the panel consider the exclusion reasonable. (d) No multiple-comparisons correction — with enough parameters, some may show moderate correlation by chance; panel judgment on borderline-ranked parameters is the direct empirical test of whether this un-corrected design choice actually misleads in practice. |
| **Ground truth** | Independent panel judgment on parameter importance/covariance from raw sweep data (blind to tool output), plus fANOVA output on the same or a matched Optuna-run sweep as a *documented divergence characterization*, explicitly **not** a ground-truth oracle (fANOVA and Pearson-with-Fisher-z measure related but non-identical things). |
| **Evaluation metrics** | Rank correlation (e.g. Spearman) between tool ranking and panel ranking, and separately between tool ranking and fANOVA ranking, reported as two distinct numbers, never merged; precision/recall on the covariance-warning binary classification (structural vs. accidental, per panel judgment); confidence calibration against Fisher-z-derived confidence; `insufficient_samples`/empty-ranking abstention correctness. |
| **Acceptable false-positive rate** | Covariance warning on hyperparameters the panel considers independently meaningful. Target ceiling: **<= 20%** of panel-judged-independent parameter pairs. |
| **Acceptable false-negative rate** | Failing to warn on a real confound the panel would catch, or ranking a panel-important parameter outside the tool's own top band. Target ceiling: **<= 15%**, deliberately looser than `audit_ablation`'s false-negative bound — `audit_sweep`'s own documented scope (exploratory ranking, no correction applied, linear-only) sets a lower a priori expectation than `audit_ablation`'s deterministic allowlist logic. |
| **Required datasets** | Real sweeps from Open RL Benchmark with both numeric and categorical hyperparameters; a matched or parallel Optuna-run version of at least a subset of sweeps for the fANOVA comparison; the structural-vs-accidental-covariance trap case (`datasets.md` §4, benchmark plan §3.8). |
| **Human annotation procedure** | `annotation-guidelines.md` §5 (ranking + covariance-judgment rubric — the only one of the three requiring annotators to compare across multiple parameters at once rather than judge a single pair/curve). |
| **Reporting template** | `validation-plan.md` §8, tool = `audit_sweep`, with fANOVA divergence reported as a distinct sub-section, not folded into the panel-comparison numbers. |

## 4. Retrieval tools — `test_connection`, `list_runs`, `get_run_summary`, `get_metric_history`, `compare_runs`

| Field | Specification |
|---|---|
| **Category** | 1 (implementation correctness) — **not** a calibration target |
| **Research question** | Does each tool correctly translate a backend API response into the normalized MCP-facing shape (`RunRef`, `Run`, `MetricHistory`, `ComparisonResult`), including documented error paths? |
| **Hypothesis** | N/A — these are correctness assertions, not calibration hypotheses. A retrieval tool either matches its documented schema/error contract on a given backend response or it doesn't; there is no "would a human researcher agree with this retrieval" question to test, because there is no judgment being made. |
| **Expected failure modes** | Schema drift against a real backend's response shape not captured by `FakeBackend`'s fixtures (the primary risk `record_wandb_fixtures.py` — Phase 0, `validation-plan.md` §5 — exists to catch); pagination edge cases (`list_runs`'s `page_size`); partial-data flagging correctness (`data_completeness`) on a real, not synthetic, W&B project. |
| **Ground truth** | The real backend's actual response, captured via `record_wandb_fixtures.py` against a live W&B project — not a human panel. |
| **Evaluation metrics** | Pass/fail against recorded real-backend fixtures (regression-style), same category as the existing 233-test suite, extended with live-fixture cases once Phase 0 runs. |
| **Acceptable false-positive rate** | N/A — see note below. |
| **Acceptable false-negative rate** | N/A — see note below. |
| **Required datasets** | Real W&B project fixtures via `scripts/record_wandb_fixtures.py` (already written, blocked only on credentials — Phase 0). |
| **Human annotation procedure** | None required. Applying panel-based calibration scoring to a deterministic schema-translation function would misapply category-2 rigor to a category-1 problem — a genuine design mistake per benchmark plan §3.1, not extra caution. |
| **Reporting template** | Standard test-suite pass/fail reporting (existing CI, `pyproject.toml`/`ci.yml`), not the Section 8 template in `validation-plan.md`, which is judgment-tool-specific. |

**Note on N/A fields above:** FP/FN rates presuppose a judgment call that
can be right or wrong relative to a human's independent opinion.
Retrieval tools make no judgment call — they either correctly mirror the
backend's data or they don't, which is a bug, not a miscalibration.
Filling these cells with invented numbers would manufacture false rigor
(`validation-plan.md` §4 principle 5).

## 5. Cross-cutting — Tool-selection accuracy

| Field | Specification |
|---|---|
| **Category** | 3 (tool-selection accuracy) — already built, not yet run |
| **Research question** | Given a natural-language prompt, does an MCP client (Claude) invoke the correct `audit_*`/retrieval tool, including on the 4 distractor-pair prompts designed to be confusable? |
| **Hypothesis** | The existing 15-prompt, 4-distractor-pair design (`docs/tool-selection-eval.md`) achieves high accuracy on the unambiguous prompts and reveals, if anywhere, confusion specifically on the distractor pairs — this is a design question already answered by `scripts/tool_selection_prompts.py`; validation here means *running* the eval, not redesigning it. |
| **Expected failure modes** | Confusion between `audit_ablation` and `compare_runs` (superficially similar 2-run inputs); confusion between `audit_sweep` and `list_runs`/`compare_runs` when a prompt mentions "compare" without specifying sweep-scoped intent. |
| **Ground truth** | The prompt author's intended tool (already encoded in `scripts/tool_selection_prompts.py`), not a panel — a specification-conformance check, not a calibration-against-ambiguous-reality check. |
| **Evaluation metrics** | Per-prompt pass/fail, aggregated to accuracy overall and accuracy on distractor-pair prompts specifically (reported separately). |
| **Acceptable false-positive rate** | N/A in the FP/FN sense — a wrong tool call is a single wrong/right outcome per prompt, not a two-directional error. |
| **Acceptable false-negative rate** | N/A, same reason. Target: **>= 90% overall accuracy, >= 75% on distractor-pair prompts specifically**, as a recommended default bar consistent with this being an already-designed, not yet empirically validated eval — revise once real results exist. |
| **Required datasets** | None new — `scripts/tool_selection_prompts.py` already exists. |
| **Human annotation procedure** | None required. |
| **Reporting template** | `docs/tool-selection-eval.md`'s existing reporting convention (dated before/after entries) — Phase 0 work (`validation-plan.md` §5), reported alongside but structurally separate from the Section 8 per-tool calibration reports. |
