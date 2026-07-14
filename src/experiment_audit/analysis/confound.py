"""Ablation-validity judgment logic for audit_ablation (Milestone 7).

Implements design-spec-v1.md §4.2 (`audit_ablation`) and reuses
`analysis/comparison.py`'s `compare_runs` directly, per §4.3's explicit
data-flow rule ("`audit_ablation` ... calls `compare_runs` internally,
then applies the intentional-param allowlist") and the roadmap's
Milestone 5 completion criteria ("`audit_ablation` in Milestone 7 will
call this as a library function, not duplicate the diffing logic").

**Architectural constraint, validated before this milestone began** (see
the Milestone 7 architecture review delivered alongside this file): this
module has no dependency on FastMCP, MCP transport, `server.py`, or
`WandbBackend`. It operates only on the normalized `Run` model
(models.py, Milestone 1) and `analysis.comparison`'s pure
`compare_runs`/`ComparisonResult` (Milestone 5), and returns plain frozen
dataclasses with their own `to_dict()` — exactly the pattern
`analysis/comparison.py` and `analysis/divergence.py` already established.
The MCP tool in `server.py` is a thin wrapper: it resolves `baseline` and
`ablation` `RunRef`s into `Run`s via `ExperimentBackend.get_run_summary`
and passes them to `audit_ablation` here — this module never fetches data
itself and never imports anything backend- or transport-shaped.

Exact allowlist membership and verdict/confidence rules are documented in
`docs/audit-methods.md#ablation` and referenced (not repeated) in the
`method` field of every result and in the MCP tool's schema description,
per spec §6. The constants below are the single source of truth;
docs/audit-methods.md restates them for human readers and must be kept in
sync by hand if they change here — there is no generation step tying the
two together (the same caveat `analysis/divergence.py` already flags for
its own thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from experiment_audit.analysis.comparison import compare_runs
from experiment_audit.models import Run

Verdict = Literal["clean", "confounded", "uncertain"]
Confidence = Literal["high", "medium", "low"]

SCHEMA_VERSION = 1
_METHOD_BASE = (
    "full config diff against claimed_variable; params tagged intentional "
    "if name matches claimed_variable or is on the allowlist "
    "(seed, device, run name/id fields); see docs/audit-methods.md#ablation"
)
_PARTIAL_DATA_SUFFIX = (
    " (confidence downgraded to low: partial data on at least one run, "
    "per design-spec-v1.md §5)"
)

# -- Allowlist ---------------------------------------------------------
#
# Deliberately a *closed set of exact config-key names*, matched
# case-insensitively — not a substring/fuzzy match. Spec §4.2: "a
# conservative allowlist (seed, device, run name/id fields) rather than
# guessing." Fuzzy matching (e.g. "any param containing 'device'") would
# risk exactly the "confidently wrong" failure mode design principle #3
# exists to prevent: a real confound like `device_batch_size` would be
# silently exempted just because it contains the substring "device".
# Exact matching is conservative in the *safe* direction — it can miss a
# differently-named seed/device field (e.g. a project using
# `random_seed` instead of `seed`), which would then correctly count
# against the verdict as an unaccounted difference rather than being
# wrongly waved through. This is a known, documented limitation (see
# docs/audit-methods.md#ablation), not an oversight: refusing to
# recognize an intentional param is far less damaging than silently
# hiding a real confound.
ALLOWLIST_PARAMS: frozenset[str] = frozenset(
    {"seed", "device", "run_name", "run_id", "name", "id"}
)


@dataclass(frozen=True)
class DifferingParam:
    """One config parameter that differs between baseline and ablation,
    with its `likely_intentional` classification — spec §4.2's
    `differing_params` entry shape."""

    param: str
    baseline_value: Any
    ablation_value: Any
    likely_intentional: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "baseline_value": self.baseline_value,
            "ablation_value": self.ablation_value,
            "likely_intentional": self.likely_intentional,
        }


@dataclass(frozen=True)
class AblationAudit:
    """Full result of `audit_ablation` — spec §4.2's exact shape:
    `{verdict, confidence, differing_params, method, evidence}`."""

    schema_version: int
    verdict: Verdict
    confidence: Confidence
    differing_params: list[DifferingParam]
    method: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "differing_params": [p.to_dict() for p in self.differing_params],
            "method": self.method,
            "evidence": self.evidence,
        }


def _is_allowlisted(param: str) -> bool:
    return param.lower() in ALLOWLIST_PARAMS


def audit_ablation(baseline: Run, ablation: Run, claimed_variable: str) -> AblationAudit:
    """Judge whether an ablation pair is a clean test of `claimed_variable`.

    Pure and deterministic: no I/O, no dependency on how `baseline`/
    `ablation` were fetched. The MCP tool (`server.py`) is responsible
    for calling `ExperimentBackend.get_run_summary` twice and passing the
    results here — this function never fetches data itself, mirroring
    `audit_training_curve`'s (Milestone 6) data-flow discipline.

    Reuses `compare_runs` (Milestone 5) directly for the underlying diff
    rather than re-deriving it: `audit_ablation`'s 2-run case is exactly
    that function's general N-way path with N=2 (see
    `analysis/comparison.py`'s module docstring). `evidence` in the
    returned `AblationAudit` is the full `compare_runs`-style diff (spec
    §4.2), i.e. both `config_diff` and `metric_diff` — not just the
    config parameters used for the verdict — so a reader can see the
    metric outcome (e.g. did the claimed variable actually move the
    target metric) alongside the confound judgment itself.

    Verdict logic:
        - No config parameters differ at all between the two runs →
          `"uncertain"`. This is deliberately distinct from `"clean"`:
          `"clean"` asserts *this is a validated test of
          claimed_variable*, which cannot be asserted if nothing in the
          config changed — including `claimed_variable` itself. Calling
          an unchanged-config pair "clean" would be exactly the
          confidently-wrong failure mode design principle #3 exists to
          prevent.
        - At least one config parameter differs, and every differing
          parameter is either `claimed_variable` itself or on
          `ALLOWLIST_PARAMS` → `"clean"`.
        - At least one differing parameter is neither `claimed_variable`
          nor on `ALLOWLIST_PARAMS` → `"confounded"`.

    Confidence:
        - `"low"` whenever `baseline.data_completeness == "partial"` or
          `ablation.data_completeness == "partial"` — automatic downgrade
          per spec §5, with the reason appended to `method` so it's
          visible without inspecting `evidence` separately. This check
          takes priority over the verdict-based confidence below.
        - `"low"` for an `"uncertain"` verdict (the tool cannot even
          confirm an ablation of `claimed_variable` was performed).
        - `"high"` for `"clean"`/`"confounded"` otherwise: the verdict
          here is a direct, deterministic consequence of the config diff
          and the fixed allowlist, not a score-thresholded estimate the
          way `audit_training_curve`'s three heuristic signals are — so
          there is no intermediate `"medium"` case to derive.

    Args:
        baseline: The reference run.
        ablation: The run being compared against `baseline`.
        claimed_variable: The config parameter the caller asserts is the
            (sole) intentional difference between the two runs.

    Returns:
        An `AblationAudit` at `schema_version: 1`.

    Raises:
        CompareRunsError: propagated unchanged from `compare_runs` if
            `baseline` and `ablation` share the same `RunRef` (comparing
            a run against itself is not a valid ablation pair). The MCP
            tool layer translates this the same way it already does for
            the `compare_runs` tool.
    """
    comparison = compare_runs([baseline, ablation])

    differing_params = [
        DifferingParam(
            param=entry.param,
            baseline_value=entry.values[baseline.ref].value,
            ablation_value=entry.values[ablation.ref].value,
            likely_intentional=(
                entry.param == claimed_variable or _is_allowlisted(entry.param)
            ),
        )
        for entry in comparison.config_diff
    ]

    if not differing_params:
        verdict: Verdict = "uncertain"
    elif all(p.likely_intentional for p in differing_params):
        verdict = "clean"
    else:
        verdict = "confounded"

    partial_data = (
        baseline.data_completeness == "partial" or ablation.data_completeness == "partial"
    )
    if partial_data:
        confidence: Confidence = "low"
        method = _METHOD_BASE + _PARTIAL_DATA_SUFFIX
    elif verdict == "uncertain":
        confidence = "low"
        method = _METHOD_BASE
    else:
        confidence = "high"
        method = _METHOD_BASE

    return AblationAudit(
        schema_version=SCHEMA_VERSION,
        verdict=verdict,
        confidence=confidence,
        differing_params=differing_params,
        method=method,
        evidence=comparison.to_dict(),
    )
