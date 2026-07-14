# W&B Fixture Recording Plan

Status: **plan only, nothing recorded yet.** This document was produced by
reading `wandb_backend.py`, `tests/test_wandb_backend.py`,
`tests/fixtures/README.md`, and `scripts/record_wandb_fixtures.py` end to
end. Every scenario below is tied to a specific assumption the code
already flags as unverified against a live project — nothing here is
invented; each row cites the exact line of reasoning in the source that
justifies it.

No fixture may be added to this list without a matching justification of
this shape. If a proposed scenario doesn't map to a real code path or a
real flagged assumption, it doesn't belong here — c.f. the adversarial
cases in `tests/fixtures/adversarial_cases.py`, which are deliberately
*not* recorded fixtures because they're synthetic shapes, not API-fidelity
questions.

## Required scenarios

| # | Scenario | Backend function(s) exercised | Why it exists / regression it prevents |
|---|----------|-------------------------------|------------------------------------------|
| 1 | `happy_path_run` | `get_run_summary`, `list_runs` → `_to_run` | Baseline check that a normal `finished` run's real attribute shape (id/name/project/entity/tags/state/created_at/config/summary_metrics) matches what `wandb_backend.py` assumes from the SDK's *documented* contract. The whole `_WandbRunLike` Protocol was written against docs, never against a live object — this is the sanity check underneath everything else. |
| 2 | `running_run_partial` | `_infer_data_completeness` | Confirms the real API actually reports state `"running"` (not e.g. `"live"`) for an in-progress run, so the `"partial"` branch of the data-completeness heuristic — explicitly flagged in code as the "central open question" of the module — fires on real data, not just the hand-built fake. |
| 3 | `crashed_run_with_nan` | `get_metric_history` → `_normalize_metric_value` | The single most speculative function in the file: it assumes W&B serializes a logged `NaN` as the literal string `"NaN"` in `scan_history()` records. This has never been checked against a real crashed run. If the real sentinel differs (e.g. `null`, `Infinity` spelled differently, or a bare Python `float('nan')` that breaks JSON), every downstream NaN-handling guarantee in spec §7 is unverified in production. |
| 4 | `empty_history_run` | `get_metric_history` | A run with summary metrics but zero logged steps (e.g. died before first `wandb.log()`, or only ever logged a final summary). Confirms `scan_history()` returns `[]`, not `None` or an error, so `MetricHistory(points=[])` is constructed safely rather than crashing. |
| 5 | `missing_metric_in_history` | `get_metric_history` (`scan_history(keys=[metric])`) | Calls `get_metric_history` for a metric name that was never logged on that run. Confirms whether the real API returns empty records, records with the key simply absent (relying on `_to_metric_point`'s `record.get(metric)` → `None` path), or raises — currently assumed, not observed. |
| 6 | `malformed_history_step` | `_to_metric_point` (`record.get("_step", 0)`) | A run whose history contains at least one record without a `_step` key (media-only log lines are a known source of this in real W&B projects). Confirms the `0` default doesn't silently misattribute a mid-run point to step 0. |
| 7 | `large_history_run` | `get_metric_history` | A run with well over 500 logged steps. `scan_history()` (not `history()`) is used specifically because the code comment claims it returns the *full unsampled* record set, unlike `history()`'s documented 500-point sampling. That claim has never been checked against a run that actually crosses the sampling threshold. |
| 8 | `multiple_runs_paginated_project` | `list_runs` (`paginator[offset:offset+per_page]`, `paginator.more`) | A project with more runs than one page. Confirms `wandb.apis.public.Runs`' slicing + `.more` contract really behaves the way the offset-cursor pagination logic assumes — flagged in the module docstring as unverified live-traffic behavior. |
| 9 | `run_not_found` | `get_run_summary`, `get_metric_history` → `WandbRunNotFoundError` | A request for a run ID that never existed or was deleted. Confirms the literal exception text/type `Api.run()` raises, so the `"not found" in str(exc).lower() or "404" in str(exc)` classification actually matches — right now it's a guess, and a mismatch means a 404 gets treated as a generic unclassified failure instead of the typed `WandbRunNotFoundError` the rest of the system expects. |
| 10 | `permission_denied_run` | `get_run_summary` error path | A run/project that exists but the configured API key has no read grant on (e.g. a private run under a different entity). `WandbRunNotFoundError`'s message already claims this case is indistinguishable from not-found ("or the API key lacks read access to it") — this scenario is what verifies that claim instead of just asserting it in a docstring. |
| 11 | `private_project_list_runs` | `list_runs` | `list_runs` against a project the key can't see. Confirms whether this raises or silently returns an empty page. Silently returning zero runs for a permissions failure would be a serious, hard-to-notice bug — an audit tool reporting "no runs found" when the real answer is "access denied" is exactly the kind of confidently-wrong failure spec §5 exists to prevent. |
| 12 | `sweep_with_runs` | `list_sweeps` → `_to_sweep` | A real sweep with `method` and `metric.name` set. Confirms `Sweep.config["method"]`, `config["metric"]["name"]`, and `Sweep.runs[i].id` match the documented shape `_to_sweep` relies on — Milestone 8 flagged this as unverified due to no live access at the time. |
| 13 | `sweep_legacy_missing_config` | `list_sweeps` defensive `.get()` fallback | An older/legacy sweep with no `method` or `metric` key in its config, if one exists in the account (otherwise best-effort — don't fabricate one). Confirms the defensive-default path (`"unsupported"` / `None`) is reachable against real malformed data, not just the hand-built fake in the test suite. |
| 14 | `auth_failure_invalid_key` | `test_connection` | A deliberately invalid `WANDB_API_KEY`. Confirms the real auth-failure exception text actually contains `"auth"` or `"api key"` (case-insensitive) so `test_connection` correctly reports `authenticated=False` with a clear message, rather than falling through to the generic "Could not verify W&B connection" branch. |
| 15 | `non_numeric_summary_metric` | `_to_summary_metrics` | A real run whose summary contains a non-numeric value (e.g. a logged image/table reference or checkpoint path string — common in real projects). Confirms these are silently dropped as intended, and that W&B's real media-reference shape (typically a nested dict) doesn't accidentally pass the `isinstance(value, (int, float))` check some other way. |

## Opportunistic (do not force)

| # | Scenario | Backend function(s) | Why it's opportunistic, not required |
|---|----------|---------------------|----------------------------------------|
| 16 | `rate_limited_429` | `_is_retryable` | Validates that a real 429/503 response's exception text actually contains one of the tokens `_is_retryable` checks for (`"429"`, `"rate limit"`, `"503"`, `"502"`, `"timeout"`). Deliberately forcing a real rate limit against your own account is disruptive and not worth doing on purpose — record this only if you hit one naturally while recording the others, or skip it and leave the heuristic flagged as unverified. |

## Explicitly out of scope

- **`sweep` variants beyond #12/#13** and **synthetic adversarial shapes**
  (3-run sweep, correlated hyperparameters, ablation pairs) — these live in
  `tests/fixtures/adversarial_cases.py` by design (see that file and the
  README's "Milestone 9 status" section). They test rule-engine *logic*
  against known shapes, not W&B API fidelity, so hand-built data is the
  right tool, not a recording.
- **`deleted_run`** as its own scenario — folded into `run_not_found`
  above; W&B's public API does not appear to distinguish "never existed"
  from "deleted" at the error-shape level this backend touches, so
  recording both would just duplicate the same verification.
