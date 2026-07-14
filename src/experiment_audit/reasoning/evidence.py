"""
Experiment Audit Scientific Reasoning Engine

Module: evidence

Defines `Evidence`, the canonical, backend-agnostic unit the reasoning
engine reasons *over*. Per research/07_reasoning_engine/evidence-model.md
("Everything becomes evidence"), every fact the engine can draw on —
config, metrics, curves, seeds, logs, hardware, dataset, code version,
and links to previous experiments — is represented uniformly here so
that later pipeline stages (observations.py, hypotheses.py, rules.py,
confidence.py, judgment.py, engine.py, recommendation.py — see
research/07_reasoning_engine/reasoning-engine.md's "Evidence Collection
-> Evidence Validation -> ..." pipeline) can operate on one shape
regardless of which stage of that pipeline produced or consumes it.

**Architectural constraint, mirrored from `analysis/*.py`:** this module
has no dependency on FastMCP, MCP transport, `server.py`, or any
backend implementation (`WandbBackend`, `FakeBackend`, ...). It operates
only on the normalized models already established in `models.py`
(`Run`, `RunRef`, `MetricHistory`, `MetricPoint`, `Sweep`) and returns
plain dataclasses with their own `to_dict()`, exactly the pattern
`models.py` and `analysis/comparison.py` already set. A caller (an MCP
tool, an `audit_*` heuristic, or a later reasoning-pipeline stage) is
responsible for resolving `RunRef`s into `Run`/`MetricHistory`/`Sweep`
objects via `ExperimentBackend` first, then building `Evidence` from
them here.

**Scope note:** this file is deliberately data-only. It has no opinion
about what a piece of evidence *means* — no scoring, no thresholds, no
verdicts. Per research/07_reasoning_engine/confidence-system.md
("Confidence is never guessed. Confidence is computed.") and
reasoning-engine.md's staged pipeline, interpretation belongs to later
stages (`observations.py` detects patterns in `Evidence`, `rules.py` and
`judgment.py` reason over those patterns, `confidence.py` scores the
result). This module only collects and exposes evidence, and does so
in a way that keeps every fact traceable back to its source run — per
evidence-model.md ("Evidence is never discarded. Reasoning is always
traceable back to evidence. Every judgment references evidence.").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from experiment_audit.models import MetricHistory, MetricPoint, Run, RunRef, Sweep


class EvidenceKind(StrEnum):
    """The evidence categories named in evidence-model.md, verbatim.

    A `str` subclass (not a bare `Enum`) so a kind serializes cleanly as
    its own value (`EvidenceKind.CONFIG == "config"`) without a manual
    `.value` lookup at every JSON-serialization call site — the same
    convenience `DataCompleteness` gets in `models.py` via a `Literal`,
    applied here via `Enum` instead because `EvidenceItem.kind` needs to
    be a closed, iterable set of categories (`items_by_kind`, below)
    rather than a bare string a caller could misspell.
    """

    CONFIG = "config"
    METRIC = "metric"
    CURVE = "curve"
    SEED = "seed"
    LOG = "log"
    HARDWARE = "hardware"
    DATASET = "dataset"
    CODE_VERSION = "code_version"
    PREVIOUS_EXPERIMENT = "previous_experiment"


@dataclass(frozen=True)
class EvidenceItem:
    """One atomic, provenance-bearing fact.

    `EvidenceItem` is the traceability unit evidence-model.md requires
    ("reasoning is always traceable back to evidence"): every fact an
    `Evidence` bundle exposes — a config value, a metric, a curve, a
    seed, a log line, a hardware detail — is recorded as one of these,
    tagged with the `RunRef` it came from, so a downstream judgment can
    cite exactly which item(s) it relied on rather than gesturing at
    "the evidence" in the abstract.

    Frozen because a fact, once observed, should not be mutated in
    place; correcting or superseding a fact means recording a new
    `EvidenceItem`, not editing an old one — consistent with "evidence
    is never discarded."

    Attributes:
        kind: Which evidence category this fact belongs to.
        key: The fact's name (a config parameter, a metric name, a
            hardware field such as `"gpu_type"`, etc). Free-form, since
            the real key vocabulary is backend- and experiment-specific.
        value: The fact's value. Left as `Any` deliberately — evidence
            is heterogeneous by nature (scalars, strings, lists of
            curve points, nested dicts); this module does not narrow or
            validate it, since doing so would start to encode
            reasoning-specific assumptions this file is scoped to avoid.
        source: The run this fact was collected from, if any. `None`
            is reserved for evidence that is not run-scoped (e.g. a
            fact about a `Sweep` as a whole).
        note: Optional free-text provenance detail (e.g. "from
            get_metric_history, step_range=(0, 1000)"). Never
            interpreted by this module; purely for a human or a later
            pipeline stage to read.
    """

    kind: EvidenceKind
    key: str
    value: Any
    source: RunRef | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "value": _json_safe(self.value),
            "source": _runref_to_dict(self.source) if self.source is not None else None,
            "note": self.note,
        }


@dataclass
class Evidence:
    """The full, backend-agnostic evidence body collected for one run.

    An `Evidence` bundle is built once a run's data has already been
    fetched through an `ExperimentBackend` (or a `FakeBackend` in
    tests) and normalized into `models.py` types; this class never
    performs I/O itself. It exists to give every later reasoning-engine
    stage one canonical shape to read from, with two properties held
    constant across all of them:

    1. Every fact is retained. Fields on this class are additive
       (`add_item`) — there is deliberately no method to remove an
       item, so a stage further down the pipeline can never silently
       lose a fact an earlier stage recorded (evidence-model.md:
       "Evidence is never discarded").
    2. Every fact is traceable. Constructing an `Evidence` bundle
       automatically materializes an `EvidenceItem` (see `items`) for
       every config value, summary metric, curve, seed, log line,
       hardware detail, dataset detail, and code version supplied —
       so a caller never has to remember to record provenance by hand.

    Attributes:
        ref: The run this evidence bundle is about. Every other
            run-scoped field on this class (`run`, `metric_histories`)
            must agree with this ref — enforced in `__post_init__`.
        run: The run's config, summary metrics, tags, and status, if
            fetched. `None` is allowed so evidence can be assembled
            incrementally (e.g. curves collected before the run summary
            is available).
        metric_histories: Full training curves, keyed by metric name.
            Corresponds to evidence-model.md's "Curves".
        sweep: The sweep this run belongs to, if any and if fetched.
        seeds: Every random seed known to apply to this run. A list
            (not a single `int`) because a bundle may aggregate seed
            evidence across repeated runs of the same configuration —
            reasoning-rules.md's Rule 002 ("only one random seed ->
            low statistical confidence") reads this field's length,
            not its content, to make that determination.
        hardware: Free-form hardware facts (e.g. `{"gpu_type": "A100",
            "gpu_count": 4}`). Backends today don't expose a
            structured hardware model (only `Run.config` /
            `Run.summary_metrics` are frozen in `models.py`), so this
            stays a plain dict rather than a bespoke type.
        dataset: Free-form dataset facts (e.g. `{"name": ..., "version":
            ..., "num_examples": ...}`), for the same reason as
            `hardware`.
        code_version: A git commit hash, tag, or other code-version
            identifier, if known.
        logs: Raw log lines or excerpts relevant to this run. Stored
            verbatim; this module does not parse or interpret them.
        previous_experiments: Other `Evidence` bundles this one should
            be reasoned about in the context of (evidence-model.md's
            "Previous Experiments"). Recursive by design: a prior
            experiment's evidence is exactly as valid a citation target
            as this run's own facts.
        collected_at: When this bundle was assembled, if the caller
            chooses to record it. Never set automatically — this
            module stays pure (no hidden `datetime.now()` calls),
            consistent with the determinism `analysis/comparison.py`
            documents for itself.
        items: Every `EvidenceItem` derived from the fields above, plus
            any appended later via `add_item`. Populated automatically
            in `__post_init__`; treat this as read-only from the
            outside except through `add_item`.
    """

    ref: RunRef
    run: Run | None = None
    metric_histories: dict[str, MetricHistory] = field(default_factory=dict)
    sweep: Sweep | None = None
    seeds: list[int] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    code_version: str | None = None
    logs: list[str] = field(default_factory=list)
    previous_experiments: list[Evidence] = field(default_factory=list)
    collected_at: datetime | None = None
    items: list[EvidenceItem] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.run is not None and self.run.ref != self.ref:
            raise ValueError(
                f"Evidence.ref {self.ref!r} does not match Evidence.run.ref "
                f"{self.run.ref!r} — a run's evidence must be scoped to its "
                "own run."
            )
        for name, history in self.metric_histories.items():
            if history.ref != self.ref:
                raise ValueError(
                    f"metric_histories[{name!r}].ref {history.ref!r} does not "
                    f"match Evidence.ref {self.ref!r}."
                )

        self.items = []
        if self.run is not None:
            for key, value in self.run.config.items():
                self._record(EvidenceKind.CONFIG, key, value)
            for name, value in self.run.summary_metrics.items():
                self._record(EvidenceKind.METRIC, name, value)
        for metric_name, history in self.metric_histories.items():
            self._record(
                EvidenceKind.CURVE,
                metric_name,
                [point.to_dict() for point in history.points],
                note=f"schema_version={history.schema_version}",
            )
        for seed in self.seeds:
            self._record(EvidenceKind.SEED, "seed", seed)
        for line in self.logs:
            self._record(EvidenceKind.LOG, "log", line)
        for key, value in self.hardware.items():
            self._record(EvidenceKind.HARDWARE, key, value)
        for key, value in self.dataset.items():
            self._record(EvidenceKind.DATASET, key, value)
        if self.code_version is not None:
            self._record(EvidenceKind.CODE_VERSION, "code_version", self.code_version)
        for prior in self.previous_experiments:
            self._record(
                EvidenceKind.PREVIOUS_EXPERIMENT,
                "run",
                _runref_to_dict(prior.ref),
                note="see Evidence.previous_experiments for the full bundle",
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        run: Run,
        *,
        metric_histories: dict[str, MetricHistory] | None = None,
        sweep: Sweep | None = None,
        seeds: list[int] | None = None,
        hardware: dict[str, Any] | None = None,
        dataset: dict[str, Any] | None = None,
        code_version: str | None = None,
        logs: list[str] | None = None,
        previous_experiments: list[Evidence] | None = None,
        collected_at: datetime | None = None,
    ) -> Evidence:
        """Build an `Evidence` bundle around an already-fetched `Run`.

        Convenience constructor for the common case (a caller has a
        `Run` from `ExperimentBackend.get_run_summary` and wants to
        wrap it, plus whatever else it has already fetched, in one
        `Evidence` bundle) so callers don't have to repeat `ref=run.ref`
        alongside `run=run` at every call site.
        """
        return cls(
            ref=run.ref,
            run=run,
            metric_histories=dict(metric_histories) if metric_histories else {},
            sweep=sweep,
            seeds=list(seeds) if seeds else [],
            hardware=dict(hardware) if hardware else {},
            dataset=dict(dataset) if dataset else {},
            code_version=code_version,
            logs=list(logs) if logs else [],
            previous_experiments=list(previous_experiments) if previous_experiments else [],
            collected_at=collected_at,
        )

    def add_item(self, item: EvidenceItem) -> None:
        """Append an ad hoc `EvidenceItem` not covered by the typed fields.

        Additive only — there is intentionally no corresponding
        `remove_item`. A later pipeline stage that concludes an item is
        wrong should record a superseding item (optionally with a
        `note` explaining why), not erase history.
        """
        self.items.append(item)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def get_config(self, key: str, default: Any = None) -> Any:
        """Return this run's config value for `key`, or `default` if unset
        or if no `run` was supplied.

        Returns `default` both when `key` is genuinely absent from the
        run's config and when `run` itself is `None`; callers that need
        to distinguish "unknown because no run" from "known-absent" have
        `self.run` (and `models.py`'s config dict) to check directly.
        """
        if self.run is None:
            return default
        return self.run.config.get(key, default)

    def has_config(self, key: str) -> bool:
        """Whether this run's config explicitly sets `key`."""
        return self.run is not None and key in self.run.config

    def config_keys(self) -> list[str]:
        """Every config parameter this run's config sets, if any."""
        if self.run is None:
            return []
        return list(self.run.config.keys())

    # ------------------------------------------------------------------
    # Summary metric helpers
    # ------------------------------------------------------------------

    def get_summary_metric(self, name: str, default: float | None = None) -> float | None:
        """Return this run's final/latest value for summary metric `name`,
        or `default` if unset or if no `run` was supplied.
        """
        if self.run is None:
            return default
        return self.run.summary_metrics.get(name, default)

    def has_summary_metric(self, name: str) -> bool:
        """Whether this run logged a summary value for metric `name`."""
        return self.run is not None and name in self.run.summary_metrics

    def summary_metrics(self) -> dict[str, float]:
        """A copy of every summary metric this run logged.

        A copy (not the underlying dict) so a caller mutating the
        result can't accidentally corrupt `self.run.summary_metrics`.
        """
        if self.run is None:
            return {}
        return dict(self.run.summary_metrics)

    # ------------------------------------------------------------------
    # Curve / metric-history helpers
    # ------------------------------------------------------------------

    def metric_names(self) -> list[str]:
        """Every metric this bundle has evidence for, from either summary
        metrics or full curves.
        """
        names: dict[str, None] = {}
        if self.run is not None:
            for name in self.run.summary_metrics:
                names.setdefault(name, None)
        for name in self.metric_histories:
            names.setdefault(name, None)
        return list(names)

    def has_metric_history(self, name: str) -> bool:
        """Whether this bundle has a full training curve for metric `name`."""
        return name in self.metric_histories

    def get_metric_history(self, name: str) -> MetricHistory | None:
        """Return the full training curve for metric `name`, if collected."""
        return self.metric_histories.get(name)

    def get_metric_points(self, name: str) -> list[MetricPoint]:
        """Return the `(step, value)` points for metric `name`'s curve, or
        an empty list if no curve was collected for it.
        """
        history = self.metric_histories.get(name)
        return list(history.points) if history is not None else []

    def latest_metric_value(self, name: str) -> float | None:
        """The most recent non-`None` value logged for metric `name`.

        Prefers the full curve (`metric_histories`) when available,
        since it reflects the actual last logged step rather than
        whatever a backend chose to report as the summary value.
        Falls back to the summary metric, then to `None` if neither is
        available. `None` points (logged NaN/null, per `models.py`
        §7) are skipped rather than treated as "no data" — a genuinely
        empty curve and a curve whose last points are `None` are
        different evidence and this method distinguishes them by
        walking backward past the trailing `None`s.
        """
        history = self.metric_histories.get(name)
        if history is not None:
            for point in reversed(history.points):
                if point.value is not None:
                    return point.value
        return self.get_summary_metric(name)

    # ------------------------------------------------------------------
    # Generic evidence-item access
    # ------------------------------------------------------------------

    def items_by_kind(self, kind: EvidenceKind) -> list[EvidenceItem]:
        """Every recorded `EvidenceItem` of a given `kind`, in the order
        it was recorded.
        """
        return [item for item in self.items if item.kind is kind]

    def items_by_key(self, key: str) -> list[EvidenceItem]:
        """Every recorded `EvidenceItem` with a given `key`, across all
        kinds. Useful when a caller knows the fact name but not which
        category it fell into.
        """
        return [item for item in self.items if item.key == key]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": _runref_to_dict(self.ref),
            "run": self.run.to_dict() if self.run is not None else None,
            "metric_histories": {
                name: history.to_dict() for name, history in self.metric_histories.items()
            },
            "sweep": self.sweep.to_dict() if self.sweep is not None else None,
            "seeds": list(self.seeds),
            "hardware": dict(self.hardware),
            "dataset": dict(self.dataset),
            "code_version": self.code_version,
            "logs": list(self.logs),
            "previous_experiments": [e.to_dict() for e in self.previous_experiments],
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
            "items": [item.to_dict() for item in self.items],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(
        self,
        kind: EvidenceKind,
        key: str,
        value: Any,
        note: str | None = None,
    ) -> None:
        self.items.append(EvidenceItem(kind=kind, key=key, value=value, source=self.ref, note=note))


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    """Local copy of `models.py`'s private `_runref_to_dict`.

    `models.py` deliberately doesn't export its underscore-prefixed
    helper for cross-module reuse; `analysis/comparison.py` follows the
    same convention with its own `_runref_key`, and this module does
    the same rather than reaching into another module's private name.
    """
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of an `EvidenceItem.value` into something
    JSON-serializable, for `to_dict()`.

    `EvidenceItem.value` is intentionally untyped (`Any`) since evidence
    is heterogeneous; this only normalizes the shapes this module
    itself produces (`RunRef`, anything with a `to_dict()`) and passes
    everything else through unchanged.
    """
    if isinstance(value, RunRef):
        return _runref_to_dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value
