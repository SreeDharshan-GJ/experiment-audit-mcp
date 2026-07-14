"""Pure run-diffing logic used by compare_runs and audit_ablation.

Implemented in Milestone 5. See design-spec-v1.md §4.2 (compare_runs) and
§4.3 (explicit data flow: audit_ablation calls this as a library function
rather than duplicating diff logic).

**Architectural constraint (validated before this milestone began, per
the roadmap's pre-Milestone-5 checklist):** this module has no
dependency on FastMCP, MCP transport, `server.py`, `WandbBackend`, or
any request/response schema. It operates only on normalized `Run`
models (models.py, Milestone 1) and returns plain dataclasses with
their own `to_dict()` — the same pattern models.py already established.
Future tools (`audit_ablation`, and any later audit tool that needs a
config/metric diff) import `compare_runs` directly from here; nothing
about this module assumes it is being called from an MCP tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiment_audit.models import Run, RunRef


def _runref_key(ref: RunRef) -> str:
    """Stable, JSON-safe string key for a `RunRef`.

    `RunRef` is frozen/hashable (models.py) and is used as an internal
    dict key throughout this module, but JSON object keys must be
    strings — a `RunRef` can't be used as a dict key once serialized.
    This mirrors `models.py`'s `_runref_to_dict` (which handles `RunRef`
    in *value* position) applied to the key position instead.
    """
    return f"{ref.backend}/{ref.entity}/{ref.project}/{ref.run_id}"


@dataclass(frozen=True)
class ConfigValue:
    """One run's value for one config parameter, or its absence.

    `present=False` distinguishes "this run's config never had this
    key" from "this run explicitly set it to `None`" — since
    `Run.config: dict[str, Any]` (models.py, spec §2) permits arbitrary
    values, a bare `None` cannot double as an absence sentinel without
    creating exactly that ambiguity.
    """

    present: bool
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"present": self.present, "value": self.value if self.present else None}


@dataclass(frozen=True)
class ConfigDiffEntry:
    """One config parameter that differs (in value or presence) across
    the compared runs."""

    param: str
    values: dict[RunRef, ConfigValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "values": {_runref_key(ref): v.to_dict() for ref, v in self.values.items()},
        }


@dataclass(frozen=True)
class MetricDiffEntry:
    """One summary metric that differs (in value or presence) across
    the compared runs.

    `delta` generalizes the 2-run case (`value_a - value_b`) to N runs
    as `max(present values) - min(present values)` — for exactly two
    runs this is identical to the pairwise difference in magnitude;
    for N > 2 it reports the full spread, which is the only
    N-way-meaningful generalization of "how different are these
    numbers" without picking an arbitrary reference run. `None` when
    fewer than two runs actually logged this metric (a delta needs at
    least two values to compare).
    """

    metric: str
    values: dict[RunRef, float | None]
    delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "values": {_runref_key(ref): v for ref, v in self.values.items()},
            "delta": self.delta,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Full diff output of `compare_runs` — spec §4.2's
    `{config_diff, metric_diff}` shape."""

    config_diff: list[ConfigDiffEntry]
    metric_diff: list[MetricDiffEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_diff": [e.to_dict() for e in self.config_diff],
            "metric_diff": [e.to_dict() for e in self.metric_diff],
        }


class CompareRunsError(ValueError):
    """Raised for structurally invalid input to `compare_runs`.

    Deliberately a `ValueError` subclass, not a bespoke exception
    hierarchy — this is an input-contract violation by the *caller*
    (too few runs, duplicate refs), not a backend or data condition a
    `ToolError.error_type` (errors.py, spec §5) needs to model. The
    MCP tool layer (Milestone 4+ concern, see `server.py`) is
    responsible for translating this into whatever tool-boundary error
    shape it uses, exactly as it already does for backend exceptions.
    """


def compare_runs(runs: list[Run]) -> ComparisonResult:
    """Diff config and summary metrics across two or more runs.

    Pure and deterministic: no I/O, no dependency on how `runs` were
    fetched. Operates only on already-fetched `Run` objects
    (models.py) — the caller (an MCP tool, `audit_ablation`, or any
    other consumer) is responsible for resolving `RunRef`s into `Run`s
    beforehand via `ExperimentBackend.get_run_summary`.

    This is the shared diffing engine spec §4.3 requires
    `audit_ablation` (Milestone 7) to reuse directly rather than
    duplicate: `audit_ablation`'s 2-run case is exactly this function's
    general N-way path with N=2, which is why the N-way path — not a
    pairwise-only shortcut — is what's implemented and tested here
    (roadmap Milestone 5 completion criteria).

    Cross-project comparison is explicitly supported (spec §2): the
    runs being compared need not share a project. Each value in the
    output is keyed by its run's full `ref` (via `_runref_key`), so
    nothing is ambiguous even in a truncated context window.

    Only *differing* params/metrics appear in the output. A param or
    metric with an identical value present on every run carries no
    diagnostic signal, and including it would just be noise in an
    already information-dense diff.

    Args:
        runs: Two or more distinct `Run` objects to compare.

    Returns:
        A `ComparisonResult` listing every differing config parameter
        and summary metric.

    Raises:
        CompareRunsError: if fewer than 2 runs are given, or if any two
            runs share the same `ref` (comparing a run against itself
            is meaningless and is almost certainly an accidental
            duplicate argument, not a real ablation/comparison).
    """
    if len(runs) < 2:
        raise CompareRunsError(f"compare_runs requires at least 2 runs, got {len(runs)}.")

    refs = [run.ref for run in runs]
    if len(set(refs)) != len(refs):
        raise CompareRunsError(
            "compare_runs received duplicate RunRefs; every run being "
            "compared must be distinct."
        )

    return ComparisonResult(
        config_diff=_diff_config(runs),
        metric_diff=_diff_metrics(runs),
    )


def _diff_config(runs: list[Run]) -> list[ConfigDiffEntry]:
    all_params: dict[str, None] = {}
    for run in runs:
        for key in run.config:
            all_params.setdefault(key, None)

    entries: list[ConfigDiffEntry] = []
    for param in all_params:
        values: dict[RunRef, ConfigValue] = {}
        present_values: list[Any] = []
        for run in runs:
            if param in run.config:
                value = run.config[param]
                values[run.ref] = ConfigValue(present=True, value=value)
                present_values.append(value)
            else:
                values[run.ref] = ConfigValue(present=False)

        missing_somewhere = len(present_values) != len(runs)
        differs_in_value = len(_unique(present_values)) > 1
        if missing_somewhere or differs_in_value:
            entries.append(ConfigDiffEntry(param=param, values=values))
    return entries


def _diff_metrics(runs: list[Run]) -> list[MetricDiffEntry]:
    all_metrics: dict[str, None] = {}
    for run in runs:
        for key in run.summary_metrics:
            all_metrics.setdefault(key, None)

    entries: list[MetricDiffEntry] = []
    for metric in all_metrics:
        values: dict[RunRef, float | None] = {}
        present_values: list[float] = []
        for run in runs:
            value = run.summary_metrics.get(metric)
            values[run.ref] = value
            if value is not None:
                present_values.append(value)

        missing_somewhere = len(present_values) != len(runs)
        differs_in_value = len(_unique(present_values)) > 1
        if missing_somewhere or differs_in_value:
            delta = (
                max(present_values) - min(present_values)
                if len(present_values) >= 2
                else None
            )
            entries.append(MetricDiffEntry(metric=metric, values=values, delta=delta))
    return entries


def _unique(values: list[Any]) -> list[Any]:
    """De-duplicate while tolerating unhashable values.

    Config values (`dict[str, Any]`, spec §2) may themselves be lists
    or dicts, which can't go in a `set` — a plain linear scan avoids
    assuming every config value is hashable.
    """
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
