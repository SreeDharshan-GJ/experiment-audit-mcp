"""
Experiment Audit Scientific Reasoning Engine

Module: hypotheses

Defines the Hypothesis Generation stage of the reasoning pipeline
(Evidence -> Observations -> **Hypotheses** -> Scientific Rules ->
Confidence -> Judgment -> Recommendation, per
research/07_reasoning_engine/reasoning-engine.md). This corresponds
to that document's "Hypothesis Generation" step: turning a flat set
of `Observation`s (observations.py) into `Hypothesis`es -- scientifically
plausible, falsifiable explanations for one or more of those
observations.

**Scope, strictly bounded.** A hypothesis is not a conclusion. This
module answers exactly one question: "what plausible explanation, if
any, connects these observations?" It never answers "is this actually
what happened," "how sure are we," or "what should be done." Per the
reasoning pipeline's staged design, those questions belong to later
modules:

- *is this true, and under what scientific rule* -> `rules.py`
- *what else might explain the same observations, or count against
  this one* -> the pipeline's "Contradiction Search" step (not this
  module -- see `Hypothesis.contradicting_observations` below)
- *how sure are we* -> `confidence.py` (per confidence-system.md:
  "Confidence is never guessed. Confidence is computed" -- deliberately
  not computed here, and deliberately absent from `Hypothesis` as a
  field)
- *what do we conclude* -> `judgment.py`
- *what should change* -> `recommendation.py`

Concretely, this means every `Hypothesis` produced here must be:

- **derived only from Observations** -- this module never inspects
  `Evidence`, a backend (W&B, MLflow), or MCP transport directly; its
  only input is `observations.py`'s `Observation` / `ObservationSet`.
  A hypothesis that cannot be traced back to specific observations
  does not belong here.
- **falsifiable** -- stated as a specific, checkable explanation (e.g.
  "possible overfitting"), never as a vague or unfalsifiable claim
  (e.g. "something may be wrong").
- **hedged, never asserted as fact** -- every `statement` is phrased
  with "possible", "consistent with", "may indicate", or "could
  suggest", and never with "is", "definitely", or "proved". A
  hypothesis that reads as a settled conclusion has leaked into
  `judgment.py`'s territory and does not belong here.
- **traceable** -- every `Hypothesis` carries the exact `Observation`s
  that produced it (`supporting_observations`) plus a flattened
  `evidence_trace` of the runs involved, so a later stage (or a human)
  can always answer "this hypothesis exists because of observations X,
  Y and Z" without re-deriving anything.

A statement such as "the model is overfitting" or "reduce the learning
rate" is a judgment or a recommendation, not a hypothesis, and does not
belong here. The corresponding *hypothesis* this module does produce is
the measured pairing this module reasons about: "'train_loss' decreased
while 'val_loss' increased" (two `Observation`s) becomes "possible
overfitting" (one `Hypothesis`, citing both). Deciding that this
hypothesis is *true*, and by how much, is `rules.py` and `confidence.py`'s
job, not this module's.

**Architectural constraint, mirrored from `evidence.py` and
`observations.py`:** this module has no dependency on FastMCP, MCP
transport, `server.py`, or any backend implementation (`WandbBackend`,
`FakeBackend`, MLflow, ...). It operates only on `observations.py`'s
`Observation` / `ObservationKind` and `models.py`'s `RunRef`, and
returns plain dataclasses with their own `to_dict()`, consistent with
every other module in this codebase.

**Internal architecture.** The scientific reasoning in this module is
split into two layers so it stays easy to extend once a Scientific
Rule Engine (`rules.py`) exists:

- *Detectors* (`_detect_*` functions, near the bottom of this file)
  are small, pure functions that only recognize a pattern in already-
  grouped `Observation`s and return the matching observations (plus
  any values a message needs), or `None` if the pattern is absent.
  They never construct a `Hypothesis`.
- *Orchestration* (`HypothesisGenerator`'s `generate_from_*` methods)
  groups raw `Observation`s into the shape a detector expects, calls
  the detector, and -- only if it fires -- hands the scientific
  statement/rationale/assumptions to `_build_hypothesis` to assemble
  the `Hypothesis`.

Each generated `Hypothesis` also carries an internal `rule_id` (the
`_RULE_*` constants below, e.g. `"H001"`) recording which specific
reasoning rule produced it. This is *not* part of the public/serialized
contract yet -- `to_dict()` does not emit it -- but it gives a future
`rules.py` a stable handle to connect a hypothesis back to the exact
rule that raised it, without having to re-infer that from `kind` alone
(several kinds, e.g. `POSSIBLE_INSUFFICIENT_REPRODUCIBILITY`, can be
reached by more than one rule).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experiment_audit_mcp.models import RunRef
from experiment_audit_mcp.reasoning.observations import Observation, ObservationKind, ObservationSet

# -- Thresholds -- pure constants, kept in sync by hand (the same caveat
# observations.py flags for its own thresholds). Each is referenced by
# name in the constructor of `HypothesisGenerator`, so a reader never has
# to guess which constant a given hypothesis used. -----------------------

_MIN_CONFOUND_PARAMS = 2
"""Minimum number of simultaneously-changed config parameters across a
compared group of runs for `POSSIBLE_CONFIGURATION_CONFOUND` to fire."""

_DATA_LEAKAGE_RATIO = 0.5
"""Max ratio of a validation-metric's last value to its paired
training-metric's last value for `POSSIBLE_DATA_LEAKAGE` to fire (e.g.
0.5 means the validation curve ended at less than half the training
curve's final value)."""

# -- Metric-name vocabulary -- used only to pair a "train" curve with its
# "val" counterpart for the same underlying quantity (e.g. 'train_loss'
# and 'val_loss'). This is a lexical convention check over metric *names*
# already present in an Observation, not an inference about what a metric
# measures. ---------------------------------------------------------------

_TRAIN_TOKENS = frozenset({"train", "training"})
_VAL_TOKENS = frozenset({"val", "valid", "validation", "eval", "evaluation"})
_METRIC_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HypothesisKind(StrEnum):
    """The closed set of hypothesis categories this module can produce.

    A `str` subclass, matching `ObservationKind`'s and `EvidenceKind`'s
    convention, so a kind serializes as its own value without a manual
    `.value` lookup at every call site.
    """

    POSSIBLE_OVERFITTING = "possible_overfitting"
    POSSIBLE_UNDERFITTING = "possible_underfitting"
    POSSIBLE_NUMERICAL_INSTABILITY = "possible_numerical_instability"
    POSSIBLE_OPTIMIZATION_STAGNATION = "possible_optimization_stagnation"
    POSSIBLE_DATA_LEAKAGE = "possible_data_leakage"
    POSSIBLE_HIGH_SEED_VARIANCE = "possible_high_seed_variance"
    POSSIBLE_MISSING_BASELINE = "possible_missing_baseline"
    POSSIBLE_CONFIGURATION_CONFOUND = "possible_configuration_confound"
    POSSIBLE_INSUFFICIENT_REPRODUCIBILITY = "possible_insufficient_reproducibility"
    POSSIBLE_INCOMPLETE_EXPERIMENT = "possible_incomplete_experiment"
    POSSIBLE_METRIC_LOGGING_ISSUE = "possible_metric_logging_issue"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One scientifically plausible, falsifiable explanation for one or
    more `Observation`s.

    Deliberately has no `confidence` or `verdict` field -- those belong
    to `confidence.py` and `judgment.py`. What this class does carry,
    so nothing downstream has to re-derive it, is exactly which
    observations produced it and what this module assumed along the
    way, per this module's docstring on traceability.

    Attributes:
        kind: Which category of hypothesis this is.
        statement: A single, hedged, human-readable sentence stating
            the explanation -- e.g. "'train_loss' decreasing while
            'val_loss' increases is consistent with overfitting."
            Always phrased with "possible", "consistent with", "may
            indicate", or "could suggest"; never with "is",
            "definitely", or "proved", per this module's docstring.
        supporting_observations: The `Observation`(s) that gave rise
            to this hypothesis. Always non-empty -- a hypothesis with
            no supporting observations would not be traceable to
            anything measured, and this module never generates one.
        contradicting_observations: `Observation`(s), if any, already
            on hand at generation time that appear to weigh against
            this hypothesis. Left empty by every generator in this
            module: per this module's docstring, searching for
            contradicting evidence is the reasoning pipeline's
            "Contradiction Search" step, which comes *after*
            Hypothesis Generation and is not this module's
            responsibility. The field exists so that later stage has
            somewhere to attach what it finds, without needing a
            different `Hypothesis` shape to do so.
        assumptions: Plain-language assumptions this hypothesis
            depends on beyond what the supporting observations
            literally state (e.g. that two differently-named metrics
            measure the same underlying quantity on different
            splits). Stating these explicitly is what keeps a hedged
            hypothesis falsifiable: a reader can check each
            assumption against the run, rather than the hypothesis
            resting on an unstated one.
        rationale: A short explanation of *why* the supporting
            observations are scientifically relevant to this
            hypothesis -- the general pattern being matched (e.g. "a
            widening train/validation gap ... is the standard
            observable signature of overfitting"), not a restatement
            of the observations themselves.
        evidence_trace: Every distinct `RunRef` covered by
            `supporting_observations`, in first-seen order. This
            module's analog of `Observation.subjects` -- a flattened,
            de-duplicated pointer back to which run(s) this hypothesis
            concerns, without requiring a reader to re-walk
            `supporting_observations` to find out.
        rule_id: Internal identifier (e.g. `"H001"`) of the specific
            `_detect_*` rule in this module that produced this
            hypothesis. Not yet part of the public/serialized contract
            -- `to_dict()` omits it -- and not intended to be read by
            callers outside this module today. It exists purely so a
            future `rules.py` / Scientific Rule Engine has a stable
            way to connect a `Hypothesis` back to the exact reasoning
            rule that raised it, since a single `kind` can sometimes
            be reached by more than one rule.
    """

    kind: HypothesisKind
    statement: str
    supporting_observations: tuple[Observation, ...]
    contradicting_observations: tuple[Observation, ...] = ()
    assumptions: tuple[str, ...] = ()
    rationale: str = ""
    evidence_trace: tuple[RunRef, ...] = ()
    rule_id: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "supporting_observations": [obs.to_dict() for obs in self.supporting_observations],
            "contradicting_observations": [
                obs.to_dict() for obs in self.contradicting_observations
            ],
            "assumptions": list(self.assumptions),
            "rationale": self.rationale,
            "evidence_trace": [_runref_to_dict(ref) for ref in self.evidence_trace],
        }


@dataclass(slots=True)
class HypothesisSet:
    """An ordered collection of `Hypothesis`es, with lookup helpers.

    Mutable and append-only by convention, mirroring `ObservationSet`
    in observations.py: `HypothesisGenerator` builds one of these
    incrementally, and a caller may append further hypotheses of its
    own (e.g. from a custom generator) without losing anything already
    collected. There is intentionally no `remove`, for the same "never
    discard a scientifically plausible explanation once raised" reason
    `ObservationSet` gives for its own append-only design.
    """

    hypotheses: list[Hypothesis] = field(default_factory=list)

    def add(self, hypothesis: Hypothesis) -> None:
        """Append a single hypothesis."""
        self.hypotheses.append(hypothesis)

    def extend(self, hypotheses: Iterable[Hypothesis]) -> None:
        """Append every hypothesis from `hypotheses`, in order."""
        self.hypotheses.extend(hypotheses)

    def by_kind(self, kind: HypothesisKind) -> list[Hypothesis]:
        """Every hypothesis of a given `kind`, in the order recorded."""
        return [hyp for hyp in self.hypotheses if hyp.kind is kind]

    def by_observation(self, observation: Observation) -> list[Hypothesis]:
        """Every hypothesis whose `supporting_observations` includes
        `observation` (by identity, matching how this module attaches
        the exact `Observation` objects it was given)."""
        return [
            hyp
            for hyp in self.hypotheses
            if any(obs is observation for obs in hyp.supporting_observations)
        ]

    def by_subject(self, ref: RunRef) -> list[Hypothesis]:
        """Every hypothesis whose `evidence_trace` includes `ref`."""
        return [hyp for hyp in self.hypotheses if ref in hyp.evidence_trace]

    def kinds(self) -> set[HypothesisKind]:
        """The distinct `HypothesisKind`s present in this set."""
        return {hyp.kind for hyp in self.hypotheses}

    def is_empty(self) -> bool:
        """Whether no hypotheses were recorded at all."""
        return not self.hypotheses

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [hyp.to_dict() for hyp in self.hypotheses]}

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self.hypotheses)

    def __bool__(self) -> bool:
        return bool(self.hypotheses)


# -- Internal rule identifiers -------------------------------------------
#
# One id per distinct reasoning rule (i.e. per `_detect_*` call site in
# `HypothesisGenerator`), not per `HypothesisKind` -- a kind such as
# `POSSIBLE_INSUFFICIENT_REPRODUCIBILITY` can be reached by more than one
# rule (missing provenance fields vs. a single recorded seed), and each
# path gets its own id so a future Scientific Rule Engine can tell them
# apart. Deliberately private and not part of this module's public
# surface: `Hypothesis.to_dict()` does not serialize `rule_id`.

_RULE_OVERFITTING = "H001"
_RULE_UNDERFITTING = "H002"
_RULE_NUMERICAL_INSTABILITY = "H003"
_RULE_OPTIMIZATION_STAGNATION = "H004"
_RULE_DATA_LEAKAGE = "H005"
_RULE_REPRODUCIBILITY_GAP = "H006"
_RULE_INCOMPLETE_EXPERIMENT = "H007"
_RULE_METRIC_LOGGING_ISSUE = "H008"
_RULE_MISSING_BASELINE = "H009"
_RULE_CONFIGURATION_CONFOUND = "H010"
_RULE_SINGLE_SEED = "H011"
_RULE_HIGH_SEED_VARIANCE = "H012"

# Observation kinds that `generate_from_missing_information` groups per
# run before deciding which (if any) missing-information hypothesis to
# raise for that run.
_MISSING_INFO_KINDS = frozenset(
    {
        ObservationKind.MISSING_SEED_INFORMATION,
        ObservationKind.MISSING_CODE_VERSION,
        ObservationKind.MISSING_HARDWARE_INFORMATION,
        ObservationKind.MISSING_DATASET_INFORMATION,
        ObservationKind.METRIC_MISSING,
        ObservationKind.MISSING_METRIC_HISTORY,
        ObservationKind.EMPTY_LOGS,
    }
)

# Observation kinds whose presence for a run may indicate the run's
# provenance is not fully reconstructable -- used by
# `generate_from_missing_information` to raise
# `POSSIBLE_INSUFFICIENT_REPRODUCIBILITY`.
_REPRODUCIBILITY_GAP_KINDS = (
    ObservationKind.MISSING_SEED_INFORMATION,
    ObservationKind.MISSING_CODE_VERSION,
    ObservationKind.MISSING_HARDWARE_INFORMATION,
    ObservationKind.MISSING_DATASET_INFORMATION,
)


# Per-metric observation entries as grouped by `generate_from_metric_patterns`
# before being handed to a detector: `{"decreasing": Observation, ...}`,
# keyed by `"decreasing"`, `"increasing"`, or `"plateau"`.
_MetricEntries = dict[str, Observation]


class HypothesisGenerator:
    """Converts `Observation`s into `Hypothesis`es. No confidence, no
    judgment, no recommendation.

    Stateless and pure: every method here is a deterministic function
    of its `Observation` argument(s) plus this instance's configured
    thresholds (constructor parameters, defaulted to the module-level
    constants above). Calling any method twice with the same inputs
    always returns equal results.

    Each helper method covers a related cluster of `HypothesisKind`s,
    named after the family of observations it reasons over:

    - `generate_from_metric_patterns` -- training-curve shape
      (overfitting, underfitting, numerical instability, optimization
      stagnation, data leakage).
    - `generate_from_missing_information` -- evidence-completeness gaps
      (insufficient reproducibility, incomplete experiment, metric
      logging issues).
    - `generate_from_configuration` -- cross-run configuration
      (configuration confound, missing baseline).
    - `generate_from_seed_information` -- randomness and run-to-run
      spread (high seed variance, insufficient reproducibility from a
      single seed).

    `generate` dispatches to all four automatically.

    Each of these methods is orchestration only: it groups the raw
    `Observation`s it was given into the shape a `_detect_*` function
    expects, calls that detector, and -- only if the detector reports a
    match -- turns the result into a `Hypothesis` via `_build_hypothesis`.
    The scientific pattern-matching itself lives in the module-level
    `_detect_*` functions below, so it can be read, tested, and reused
    independently of how observations happen to be grouped.
    """

    def __init__(
        self,
        *,
        min_confound_params: int = _MIN_CONFOUND_PARAMS,
        data_leakage_ratio: float = _DATA_LEAKAGE_RATIO,
    ) -> None:
        """Configure generation thresholds.

        Args:
            min_confound_params: Minimum number of simultaneously
                differing configuration parameters across a compared
                group of runs for `POSSIBLE_CONFIGURATION_CONFOUND` to
                fire. A single deliberately changed parameter (the
                normal shape of an ablation) does not, by itself,
                constitute a confound.
            data_leakage_ratio: Maximum ratio of a validation metric's
                last value to its paired training metric's last value
                for `POSSIBLE_DATA_LEAKAGE` to fire.
        """
        if min_confound_params < 2:
            raise ValueError(f"min_confound_params must be >= 2, got {min_confound_params}.")
        if not (0.0 < data_leakage_ratio <= 1.0):
            raise ValueError(f"data_leakage_ratio must be in (0.0, 1.0], got {data_leakage_ratio}.")
        self._min_confound_params = min_confound_params
        self._data_leakage_ratio = data_leakage_ratio

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self, observations: ObservationSet | Iterable[Observation]) -> HypothesisSet:
        """Run every generator over `observations` and collect the results.

        Args:
            observations: An `ObservationSet`, or any iterable of
                `Observation`s (e.g. a filtered slice of one), to
                reason over. Never `Evidence` -- this stage only ever
                sees what `observations.py` already extracted.

        Returns:
            A `HypothesisSet` with every hypothesis this generator's
            four helper methods could support from `observations`. An
            input with no matching patterns yields an empty
            `HypothesisSet` -- this method never manufactures a
            hypothesis where no observation supports one.
        """
        obs_list = list(observations)

        result = HypothesisSet()
        result.extend(self.generate_from_metric_patterns(obs_list))
        result.extend(self.generate_from_missing_information(obs_list))
        result.extend(self.generate_from_configuration(obs_list))
        result.extend(self.generate_from_seed_information(obs_list))
        return result

    # ------------------------------------------------------------------
    # Metric-pattern hypotheses
    # ------------------------------------------------------------------

    def generate_from_metric_patterns(
        self, observations: Sequence[Observation]
    ) -> list[Hypothesis]:
        """Hypothesize from training-curve shape: overfitting,
        underfitting, numerical instability, optimization stagnation,
        and data leakage.

        Only considers per-run, metric-specific observations (`NAN_DETECTED`,
        `TRAINING_PLATEAU_DETECTED`, `METRIC_INCREASING`,
        `METRIC_DECREASING`). Curve-shape observations for the same run
        whose metric names differ only by a train/validation naming
        convention (e.g. 'train_loss' vs. 'val_loss') are paired
        together, since several hypotheses here depend on comparing a
        training curve against its validation counterpart.

        This method only groups observations by run and by metric role
        (train/validation) and dispatches to the `_detect_overfitting`,
        `_detect_underfitting`, `_detect_data_leakage`,
        `_detect_metric_instability`, and `_detect_plateau` pattern
        detectors; it does not itself decide what counts as a match.

        Args:
            observations: The observations to reason over. Order does
                not matter; observations of other kinds, or with more
                than one subject, are ignored by this method.

        Returns:
            One `Hypothesis` per pattern matched. A `NAN_DETECTED`
            observation always yields a `POSSIBLE_NUMERICAL_INSTABILITY`
            hypothesis on its own; a `TRAINING_PLATEAU_DETECTED`
            observation yields `POSSIBLE_OPTIMIZATION_STAGNATION`
            unless it was already consumed by a paired underfitting
            hypothesis for the same run.
        """
        hypotheses: list[Hypothesis] = []
        consumed_plateaus: set[int] = set()

        nan_obs: list[Observation] = []
        plateau_obs: list[Observation] = []
        by_run: dict[RunRef, dict[str, _MetricEntries]] = {}

        for obs in observations:
            if len(obs.subjects) != 1:
                continue
            if obs.kind is ObservationKind.NAN_DETECTED:
                nan_obs.append(obs)
            elif obs.metric is None:
                continue
            elif obs.kind is ObservationKind.TRAINING_PLATEAU_DETECTED:
                plateau_obs.append(obs)
                by_run.setdefault(obs.subjects[0], {}).setdefault(obs.metric, {})["plateau"] = obs
            elif obs.kind is ObservationKind.METRIC_INCREASING:
                by_run.setdefault(obs.subjects[0], {}).setdefault(obs.metric, {})["increasing"] = (
                    obs
                )
            elif obs.kind is ObservationKind.METRIC_DECREASING:
                by_run.setdefault(obs.subjects[0], {}).setdefault(obs.metric, {})["decreasing"] = (
                    obs
                )

        for obs in nan_obs:
            detected = _detect_metric_instability(obs)
            if detected is None:
                continue
            hypotheses.append(
                _build_hypothesis(
                    kind=HypothesisKind.POSSIBLE_NUMERICAL_INSTABILITY,
                    rule_id=_RULE_NUMERICAL_INSTABILITY,
                    statement=(
                        f"Null/NaN values logged for metric '{detected.metric}' may "
                        "indicate numerical instability during training (e.g. exploding "
                        "gradients or an unstable loss computation)."
                    ),
                    supporting_observations=(detected,),
                    assumptions=(
                        "Assumes the NaN/null points reflect a genuine training-time "
                        "failure rather than an intentional gap in logging.",
                    ),
                    rationale=(
                        "Null/NaN values interrupting an otherwise numeric metric curve "
                        "are consistent with a known failure mode of gradient-based "
                        "training: divergence, overflow, or a bad optimizer step."
                    ),
                )
            )

        for run, metrics_for_run in by_run.items():
            grouped: dict[str, dict[str, tuple[str, _MetricEntries]]] = {}
            for metric_name, entries in metrics_for_run.items():
                parsed = _metric_role_and_base(metric_name)
                if parsed is None:
                    continue
                role, base = parsed
                grouped.setdefault(base, {})[role] = (metric_name, entries)

            for base, roles_map in grouped.items():
                train_info = roles_map.get("train")
                val_info = roles_map.get("val")
                if train_info is None or val_info is None:
                    continue
                train_name, train_entries = train_info
                val_name, val_entries = val_info

                overfitting_match = _detect_overfitting(train_entries, val_entries)
                if overfitting_match is not None:
                    train_dec, val_inc = overfitting_match
                    hypotheses.append(
                        _build_hypothesis(
                            kind=HypothesisKind.POSSIBLE_OVERFITTING,
                            rule_id=_RULE_OVERFITTING,
                            statement=(
                                f"'{train_name}' decreasing while '{val_name}' increases "
                                "is consistent with overfitting: the model may be fitting "
                                "the training data at the expense of generalization."
                            ),
                            supporting_observations=(train_dec, val_inc),
                            assumptions=(
                                f"Assumes '{train_name}' and '{val_name}' measure the same "
                                "underlying quantity on the training and validation "
                                "splits respectively.",
                            ),
                            rationale=(
                                "A widening gap between a training curve that keeps "
                                "improving and a validation curve of the same quantity "
                                "that worsens is the standard observable signature of "
                                "overfitting."
                            ),
                        )
                    )

                underfitting_match = _detect_underfitting(train_entries, val_entries)
                if underfitting_match is not None:
                    train_plateau, val_plateau = underfitting_match
                    hypotheses.append(
                        _build_hypothesis(
                            kind=HypothesisKind.POSSIBLE_UNDERFITTING,
                            rule_id=_RULE_UNDERFITTING,
                            statement=(
                                f"Both '{train_name}' and '{val_name}' were flat over the "
                                "same window, which may indicate underfitting: the model "
                                "could be failing to learn further from the data."
                            ),
                            supporting_observations=(train_plateau, val_plateau),
                            assumptions=(
                                f"Assumes '{train_name}' and '{val_name}' measure the same "
                                "underlying quantity on the training and validation "
                                "splits respectively.",
                            ),
                            rationale=(
                                "When the training and validation curves of the same "
                                "quantity stall together, a stopped-learning explanation "
                                "is more consistent with the pattern than the model "
                                "having already reached its own ceiling on both splits "
                                "simultaneously."
                            ),
                        )
                    )
                    consumed_plateaus.add(id(train_plateau))
                    consumed_plateaus.add(id(val_plateau))

                leakage_match = _detect_data_leakage(
                    train_entries, val_entries, self._data_leakage_ratio
                )
                if leakage_match is not None:
                    train_dec, val_dec, train_last, val_last = leakage_match
                    hypotheses.append(
                        _build_hypothesis(
                            kind=HypothesisKind.POSSIBLE_DATA_LEAKAGE,
                            rule_id=_RULE_DATA_LEAKAGE,
                            statement=(
                                f"'{val_name}' ended substantially lower than "
                                f"'{train_name}' ({val_last:.6g} vs {train_last:.6g}), "
                                "which could suggest data leakage between the "
                                "training and validation splits."
                            ),
                            supporting_observations=(train_dec, val_dec),
                            assumptions=(
                                f"Assumes '{train_name}' and '{val_name}' measure the "
                                "same underlying quantity on the training and "
                                "validation splits respectively.",
                                "Assumes the validation split is not expected to be "
                                "easier than the training split by construction.",
                            ),
                            rationale=(
                                "A validation metric that ends up meaningfully better "
                                "than its training counterpart runs counter to the "
                                "usual expectation that a model fits training data at "
                                "least as well as held-out data, a pattern that could "
                                "suggest leaked or overlapping data between the two "
                                "splits."
                            ),
                        )
                    )

        for obs in plateau_obs:
            detected = _detect_plateau(obs, consumed_plateaus)
            if detected is None:
                continue
            hypotheses.append(
                _build_hypothesis(
                    kind=HypothesisKind.POSSIBLE_OPTIMIZATION_STAGNATION,
                    rule_id=_RULE_OPTIMIZATION_STAGNATION,
                    statement=(
                        f"Metric '{detected.metric}' stayed flat for an extended window, "
                        "which may indicate optimization stagnation (e.g. a learning "
                        "rate too small to make further progress, or a saturated "
                        "objective)."
                    ),
                    supporting_observations=(detected,),
                    assumptions=(
                        "Assumes the flat window reflects an optimizer that has stopped "
                        "making progress, rather than a metric that has already "
                        "converged to its true optimum.",
                    ),
                    rationale=(
                        "A sustained, low-variance window in a training curve is "
                        "consistent with the optimizer no longer finding improving "
                        "steps; a learning rate or schedule that has become too small "
                        "relative to the remaining loss landscape is a common "
                        "explanation for that pattern."
                    ),
                )
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Missing-information hypotheses
    # ------------------------------------------------------------------

    def generate_from_missing_information(
        self, observations: Sequence[Observation]
    ) -> list[Hypothesis]:
        """Hypothesize from evidence-completeness gaps: insufficient
        reproducibility, incomplete experiment, and metric logging
        issues.

        This method only groups observations by run and dispatches to
        the `_detect_reproducibility_gap`, `_detect_incomplete_experiment`,
        and `_detect_metric_logging_issue` pattern detectors; it does
        not itself decide what counts as a match.

        Args:
            observations: The observations to reason over. Only
                single-subject observations whose kind is one of
                `_MISSING_INFO_KINDS` are considered; everything else
                is ignored by this method.

        Returns:
            Up to two `Hypothesis`es per run that has any qualifying
            gap: one `POSSIBLE_INSUFFICIENT_REPRODUCIBILITY` if any
            provenance field (seed, code version, hardware, dataset)
            is missing, and one of `POSSIBLE_INCOMPLETE_EXPERIMENT` or
            `POSSIBLE_METRIC_LOGGING_ISSUE` if metrics and/or logs are
            missing. Empty if no run in `observations` has any
            qualifying gap.
        """
        hypotheses: list[Hypothesis] = []
        by_run: dict[RunRef, dict[ObservationKind, Observation]] = {}
        for obs in observations:
            if len(obs.subjects) == 1 and obs.kind in _MISSING_INFO_KINDS:
                by_run.setdefault(obs.subjects[0], {})[obs.kind] = obs

        for kinds_present in by_run.values():
            repro_match = _detect_reproducibility_gap(kinds_present)
            if repro_match is not None:
                labels = ", ".join(
                    k.value.replace("_", " ")
                    for k in _REPRODUCIBILITY_GAP_KINDS
                    if k in kinds_present
                )
                hypotheses.append(
                    _build_hypothesis(
                        kind=HypothesisKind.POSSIBLE_INSUFFICIENT_REPRODUCIBILITY,
                        rule_id=_RULE_REPRODUCIBILITY_GAP,
                        statement=(
                            f"This run is missing recorded {labels}, which may indicate "
                            "it cannot be reproduced from the evidence on hand alone."
                        ),
                        supporting_observations=repro_match,
                        assumptions=(
                            "Assumes the missing fields are genuinely absent from the "
                            "run's record, rather than simply not requested from the "
                            "backend.",
                        ),
                        rationale=(
                            "Reproducing a run requires knowing what code, hardware, "
                            "data, and randomness produced it; the absence of any of "
                            "these leaves a gap an independent rerun could not close."
                        ),
                    )
                )

            incomplete_match = _detect_incomplete_experiment(kinds_present)
            if incomplete_match is not None:
                hypotheses.append(
                    _build_hypothesis(
                        kind=HypothesisKind.POSSIBLE_INCOMPLETE_EXPERIMENT,
                        rule_id=_RULE_INCOMPLETE_EXPERIMENT,
                        statement=(
                            "No metrics and no log lines were recorded for this run, "
                            "which may indicate the run never completed (or never "
                            "started) rather than a logging gap in an otherwise-finished "
                            "run."
                        ),
                        supporting_observations=incomplete_match,
                        assumptions=(
                            "Assumes the absence of metrics and logs reflects the run's "
                            "own state, rather than a retrieval gap at query time.",
                        ),
                        rationale=(
                            "A run with neither metrics nor logs is missing both "
                            "channels through which training progress would normally "
                            "surface, which is more consistent with an incomplete run "
                            "than a selective logging omission."
                        ),
                    )
                )
            else:
                logging_match = _detect_metric_logging_issue(kinds_present)
                if logging_match is not None:
                    hypotheses.append(
                        _build_hypothesis(
                            kind=HypothesisKind.POSSIBLE_METRIC_LOGGING_ISSUE,
                            rule_id=_RULE_METRIC_LOGGING_ISSUE,
                            statement=(
                                "Metric data is partially or fully absent for this run, "
                                "which could suggest a gap in metric logging rather than "
                                "a property of the run itself."
                            ),
                            supporting_observations=logging_match,
                            assumptions=(
                                "Assumes metric logging was intended to cover the "
                                "missing metric(s), rather than them being "
                                "intentionally untracked.",
                            ),
                            rationale=(
                                "Summary values with no corresponding curve, or no "
                                "metrics at all alongside otherwise-present logs, are "
                                "consistent with an instrumentation or "
                                "logging-configuration issue as much as with a "
                                "substantive property of the run."
                            ),
                        )
                    )

        return hypotheses

    # ------------------------------------------------------------------
    # Configuration hypotheses
    # ------------------------------------------------------------------

    def generate_from_configuration(self, observations: Sequence[Observation]) -> list[Hypothesis]:
        """Hypothesize from cross-run configuration: configuration
        confound and missing baseline.

        This method only groups observations (by comparison group, for
        configuration changes) and dispatches to the
        `_detect_missing_baseline` and `_detect_configuration_confound`
        pattern detectors; it does not itself decide what counts as a
        match.

        Args:
            observations: The observations to reason over. Only
                `CONFIGURATION_CHANGED` and `MISSING_BASELINE`
                observations are considered; everything else is
                ignored by this method.

        Returns:
            One `POSSIBLE_MISSING_BASELINE` hypothesis per
            `MISSING_BASELINE` observation, plus one
            `POSSIBLE_CONFIGURATION_CONFOUND` hypothesis per group of
            runs (identified by the exact set of `RunRef`s being
            compared) whose simultaneously-differing parameter count
            meets `min_confound_params`. Empty if no observation of
            either kind is present, or no group clears the threshold.
        """
        hypotheses: list[Hypothesis] = []

        for obs in observations:
            detected = _detect_missing_baseline(obs)
            if detected is None:
                continue
            hypotheses.append(
                _build_hypothesis(
                    kind=HypothesisKind.POSSIBLE_MISSING_BASELINE,
                    rule_id=_RULE_MISSING_BASELINE,
                    statement=(
                        "None of the compared runs carry an explicit baseline "
                        "marker, which may indicate this comparison lacks a "
                        "reference point to judge the other runs against."
                    ),
                    supporting_observations=(detected,),
                    assumptions=(
                        "Assumes a baseline, if one exists among these runs, would "
                        "have been marked with a recognized tag or config field.",
                    ),
                    rationale=(
                        "Without an identifiable baseline, a relative claim about "
                        "which compared run is better lacks a fixed point of "
                        "comparison."
                    ),
                )
            )

        groups: dict[frozenset[RunRef], list[Observation]] = {}
        for obs in observations:
            if obs.kind is ObservationKind.CONFIGURATION_CHANGED:
                groups.setdefault(frozenset(obs.subjects), []).append(obs)

        for group_obs in groups.values():
            confound_match = _detect_configuration_confound(group_obs, self._min_confound_params)
            if confound_match is None:
                continue
            params = sorted({str(o.measurements.get("parameter", "?")) for o in confound_match})
            hypotheses.append(
                _build_hypothesis(
                    kind=HypothesisKind.POSSIBLE_CONFIGURATION_CONFOUND,
                    rule_id=_RULE_CONFIGURATION_CONFOUND,
                    statement=(
                        f"{len(params)} configuration parameters ({', '.join(params)}) "
                        "differ simultaneously across the compared runs, which could "
                        "suggest any observed difference in outcome is confounded "
                        "rather than attributable to a single change."
                    ),
                    supporting_observations=confound_match,
                    assumptions=(
                        "Assumes the comparison is intended to isolate the effect of a "
                        "single changed variable (e.g. an ablation), rather than being "
                        "a deliberate multi-factor sweep.",
                    ),
                    rationale=(
                        "Attributing an outcome difference to one parameter requires "
                        "that parameter to be the only thing that changed; multiple "
                        "simultaneous changes leave the effect of any single one "
                        "unidentified."
                    ),
                )
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Seed-information hypotheses
    # ------------------------------------------------------------------

    def generate_from_seed_information(
        self, observations: Sequence[Observation]
    ) -> list[Hypothesis]:
        """Hypothesize from randomness and run-to-run spread: high
        seed variance, and insufficient reproducibility from a single
        recorded seed.

        This method dispatches directly to the `_detect_single_seed_gap`
        and `_detect_seed_variance` pattern detectors, one observation
        at a time; it does not itself decide what counts as a match.

        Args:
            observations: The observations to reason over. Only
                `SINGLE_RANDOM_SEED` and `LARGE_VARIANCE_BETWEEN_RUNS`
                observations are considered; everything else is
                ignored by this method.

        Returns:
            One `POSSIBLE_INSUFFICIENT_REPRODUCIBILITY` hypothesis per
            `SINGLE_RANDOM_SEED` observation, and one
            `POSSIBLE_HIGH_SEED_VARIANCE` hypothesis per
            `LARGE_VARIANCE_BETWEEN_RUNS` observation. Empty if neither
            kind is present in `observations`.
        """
        hypotheses: list[Hypothesis] = []

        for obs in observations:
            single_seed_match = _detect_single_seed_gap(obs)
            if single_seed_match is not None:
                hypotheses.append(
                    _build_hypothesis(
                        kind=HypothesisKind.POSSIBLE_INSUFFICIENT_REPRODUCIBILITY,
                        rule_id=_RULE_SINGLE_SEED,
                        statement=(
                            "Only one random seed was recorded, which may indicate "
                            "this run's outcome has not been checked for stability "
                            "across seeds."
                        ),
                        supporting_observations=(single_seed_match,),
                        assumptions=(
                            "Assumes the single-seed result is being treated as "
                            "representative, rather than as a deliberately scoped "
                            "single trial.",
                        ),
                        rationale=(
                            "A single seed cannot distinguish a real effect from "
                            "ordinary run-to-run variance, which limits how much the "
                            "result alone can support."
                        ),
                    )
                )
                continue

            variance_match = _detect_seed_variance(obs)
            if variance_match is not None:
                hypotheses.append(
                    _build_hypothesis(
                        kind=HypothesisKind.POSSIBLE_HIGH_SEED_VARIANCE,
                        rule_id=_RULE_HIGH_SEED_VARIANCE,
                        statement=(
                            f"Metric '{variance_match.metric}' varies widely across the "
                            "compared runs, which could suggest high sensitivity to "
                            "random seed or other uncontrolled sources of variance."
                        ),
                        supporting_observations=(variance_match,),
                        assumptions=(
                            "Assumes the compared runs otherwise share the same "
                            "configuration, so the spread is not itself explained by a "
                            "configuration difference.",
                        ),
                        rationale=(
                            "A wide spread in a summary metric across otherwise-"
                            "comparable runs is consistent with the training process "
                            "being sensitive to randomness (seed, data order, "
                            "nondeterministic kernels) rather than fully determined by "
                            "configuration."
                        ),
                    )
                )

        return hypotheses


# ------------------------------------------------------------------
# Pattern detectors
#
# Pure functions: given already-grouped `Observation`s, each either
# returns the observations (and any values a message needs) that make
# up the matched pattern, or `None` if the pattern does not hold. None
# of these construct a `Hypothesis` -- that is `_build_hypothesis`'s
# job, driven by the orchestration methods on `HypothesisGenerator`
# above. Keeping detection this narrow is what will let a future
# Scientific Rule Engine call, test, or replace one rule at a time.
# ------------------------------------------------------------------


def _detect_overfitting(
    train_entries: _MetricEntries, val_entries: _MetricEntries
) -> tuple[Observation, Observation] | None:
    """Detects a decreasing training curve paired with an increasing
    validation curve of the same underlying quantity.

    Returns:
        `(train_decreasing, val_increasing)` if both are present,
        else `None`.
    """
    train_dec = train_entries.get("decreasing")
    val_inc = val_entries.get("increasing")
    if train_dec is not None and val_inc is not None:
        return train_dec, val_inc
    return None


def _detect_underfitting(
    train_entries: _MetricEntries, val_entries: _MetricEntries
) -> tuple[Observation, Observation] | None:
    """Detects a training curve and its validation counterpart both
    plateauing over the same window.

    Returns:
        `(train_plateau, val_plateau)` if both are present, else
        `None`.
    """
    train_plateau = train_entries.get("plateau")
    val_plateau = val_entries.get("plateau")
    if train_plateau is not None and val_plateau is not None:
        return train_plateau, val_plateau
    return None


def _detect_data_leakage(
    train_entries: _MetricEntries, val_entries: _MetricEntries, ratio: float
) -> tuple[Observation, Observation, float, float] | None:
    """Detects a validation curve ending substantially below its
    training counterpart -- lower than `ratio` times the training
    curve's final value.

    Returns:
        `(train_decreasing, val_decreasing, train_last_value,
        val_last_value)` if both curves are decreasing and the
        validation curve's last value is suspiciously low relative to
        training's, else `None`.
    """
    train_dec = train_entries.get("decreasing")
    val_dec = val_entries.get("decreasing")
    if train_dec is None or val_dec is None:
        return None
    train_last = train_dec.measurements.get("last_value")
    val_last = val_dec.measurements.get("last_value")
    if (
        isinstance(train_last, (int, float))
        and isinstance(val_last, (int, float))
        and train_last > 0
        and val_last <= train_last * ratio
    ):
        return train_dec, val_dec, train_last, val_last
    return None


def _detect_metric_instability(obs: Observation) -> Observation | None:
    """Detects a null/NaN observation on a metric curve.

    Returns:
        `obs` if it is a `NAN_DETECTED` observation, else `None`.
    """
    if obs.kind is ObservationKind.NAN_DETECTED:
        return obs
    return None


def _detect_plateau(obs: Observation, consumed: set[int]) -> Observation | None:
    """Detects a stray plateau not already explained by a paired
    `POSSIBLE_UNDERFITTING` hypothesis for the same run.

    Args:
        obs: A `TRAINING_PLATEAU_DETECTED` observation.
        consumed: `id()`s of plateau observations already attached to
            an underfitting hypothesis, and therefore not eligible to
            independently suggest optimization stagnation.

    Returns:
        `obs` if it has not already been consumed, else `None`.
    """
    if id(obs) in consumed:
        return None
    return obs


def _detect_reproducibility_gap(
    kinds_present: dict[ObservationKind, Observation],
) -> tuple[Observation, ...] | None:
    """Detects any missing provenance field (seed, code version,
    hardware, dataset) recorded for a run.

    Returns:
        The observations for every missing provenance field present in
        `kinds_present`, in `_REPRODUCIBILITY_GAP_KINDS` order, or
        `None` if none are present.
    """
    repro_kinds = [k for k in _REPRODUCIBILITY_GAP_KINDS if k in kinds_present]
    if not repro_kinds:
        return None
    return tuple(kinds_present[k] for k in repro_kinds)


def _detect_incomplete_experiment(
    kinds_present: dict[ObservationKind, Observation],
) -> tuple[Observation, Observation] | None:
    """Detects a run with neither metrics nor logs recorded.

    Returns:
        `(metric_missing, empty_logs)` if both are present in
        `kinds_present`, else `None`.
    """
    if (
        ObservationKind.METRIC_MISSING in kinds_present
        and ObservationKind.EMPTY_LOGS in kinds_present
    ):
        return (
            kinds_present[ObservationKind.METRIC_MISSING],
            kinds_present[ObservationKind.EMPTY_LOGS],
        )
    return None


def _detect_metric_logging_issue(
    kinds_present: dict[ObservationKind, Observation],
) -> tuple[Observation, ...] | None:
    """Detects missing or summary-only metric data for a run.

    Returns:
        The `METRIC_MISSING` and/or `MISSING_METRIC_HISTORY`
        observations present in `kinds_present`, or `None` if neither
        is present.
    """
    if (
        ObservationKind.METRIC_MISSING in kinds_present
        or ObservationKind.MISSING_METRIC_HISTORY in kinds_present
    ):
        return tuple(
            kinds_present[k]
            for k in (ObservationKind.METRIC_MISSING, ObservationKind.MISSING_METRIC_HISTORY)
            if k in kinds_present
        )
    return None


def _detect_missing_baseline(obs: Observation) -> Observation | None:
    """Detects a `MISSING_BASELINE` observation.

    Returns:
        `obs` if it is a `MISSING_BASELINE` observation, else `None`.
    """
    if obs.kind is ObservationKind.MISSING_BASELINE:
        return obs
    return None


def _detect_configuration_confound(
    group_obs: Sequence[Observation], min_confound_params: int
) -> tuple[Observation, ...] | None:
    """Detects a group of runs with at least `min_confound_params`
    configuration parameters differing simultaneously.

    Returns:
        `group_obs` as a tuple if it meets `min_confound_params`, else
        `None`.
    """
    if len(group_obs) < min_confound_params:
        return None
    return tuple(group_obs)


def _detect_single_seed_gap(obs: Observation) -> Observation | None:
    """Detects a `SINGLE_RANDOM_SEED` observation.

    Returns:
        `obs` if it is a `SINGLE_RANDOM_SEED` observation, else
        `None`.
    """
    if obs.kind is ObservationKind.SINGLE_RANDOM_SEED:
        return obs
    return None


def _detect_seed_variance(obs: Observation) -> Observation | None:
    """Detects a `LARGE_VARIANCE_BETWEEN_RUNS` observation.

    Returns:
        `obs` if it is a `LARGE_VARIANCE_BETWEEN_RUNS` observation,
        else `None`.
    """
    if obs.kind is ObservationKind.LARGE_VARIANCE_BETWEEN_RUNS:
        return obs
    return None


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------


def _build_hypothesis(
    *,
    kind: HypothesisKind,
    rule_id: str,
    statement: str,
    supporting_observations: Sequence[Observation],
    assumptions: tuple[str, ...] = (),
    rationale: str = "",
    contradicting_observations: tuple[Observation, ...] = (),
) -> Hypothesis:
    """Assembles a `Hypothesis` from a detector's match plus its
    scientific write-up, filling in `evidence_trace` automatically.

    Every `generate_from_*` method funnels its `Hypothesis`
    construction through this one place, so `evidence_trace` is always
    derived consistently from `supporting_observations` and every
    hypothesis is stamped with the internal `rule_id` of the rule that
    produced it, without each call site repeating that bookkeeping.

    Args:
        kind: The `HypothesisKind` this hypothesis represents.
        rule_id: The internal `_RULE_*` identifier of the detector
            that produced this hypothesis.
        statement: The hedged, human-readable explanation.
        supporting_observations: The observations that support this
            hypothesis; also used to derive `evidence_trace`.
        assumptions: Plain-language assumptions this hypothesis
            depends on, beyond what `supporting_observations` states.
        rationale: Why the supporting observations are scientifically
            relevant to this hypothesis.
        contradicting_observations: Left empty by every current
            caller; see `Hypothesis.contradicting_observations`.

    Returns:
        A fully-populated `Hypothesis`.
    """
    supporting = tuple(supporting_observations)
    return Hypothesis(
        kind=kind,
        statement=statement,
        supporting_observations=supporting,
        contradicting_observations=contradicting_observations,
        assumptions=assumptions,
        rationale=rationale,
        evidence_trace=_trace(supporting),
        rule_id=rule_id,
    )


def _metric_role_and_base(metric_name: str) -> tuple[str, str] | None:
    """Classify `metric_name` as a "train" or "val" curve, if it lexically
    matches that convention, and return `(role, base)`.

    `base` is `metric_name` with the matched token removed (e.g.
    `"train_loss"` -> `("train", "loss")`, `"val_loss"` ->
    `("val", "loss")`), so two differently-scoped metric names for the
    same underlying quantity can be paired by comparing `base`.

    A purely lexical check over the metric *name itself* (already
    present on an `Observation`) -- not an inference about what the
    metric measures, and not a lookup against `Evidence`.

    Returns:
        `(role, base)` where `role` is `"train"` or `"val"`, or `None`
        if `metric_name` contains no recognized train/validation
        token.
    """
    tokens = _METRIC_TOKEN_RE.findall(metric_name.lower())
    for index, token in enumerate(tokens):
        if token in _TRAIN_TOKENS:
            return "train", "_".join(tokens[:index] + tokens[index + 1 :])
        if token in _VAL_TOKENS:
            return "val", "_".join(tokens[:index] + tokens[index + 1 :])
    return None


def _trace(observations: Sequence[Observation]) -> tuple[RunRef, ...]:
    """Every distinct `RunRef` across `observations`' `subjects`, in
    first-seen order. Builds `Hypothesis.evidence_trace`."""
    seen: dict[RunRef, None] = {}
    for obs in observations:
        for ref in obs.subjects:
            seen.setdefault(ref, None)
    return tuple(seen.keys())


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    """Local copy of `models.py`'s private `_runref_to_dict`, used by
    `Hypothesis.to_dict()` to serialize `evidence_trace`. Same
    "each module keeps its own private copy" convention `observations.py`
    documents for its own copy of this helper."""
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }
