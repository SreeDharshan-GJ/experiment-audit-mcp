"""Regression tests for two confirmed serialization bugs found in
production audit #5 of the reporting/serialization layer:

1. `Claim.to_dict()` (claims.py) returned `list(self.evidence_trace)`
   verbatim -- leaving raw `RunRef` (models.py) instances in the
   `"evidence_trace"` field instead of the JSON-safe structure this
   method's own docstring ("Best-effort JSON-safe serialization")
   promises, and every other analogous field in this codebase
   (`evidence.py`'s `EvidenceItem.source`, `models.py`'s own
   `Run.ref` / `MetricPoint.ref` / `Sweep.run_refs`) actually
   produces. `RunRef` has no `to_dict()` method of its own (by
   design -- `models.py` keeps a private `_runref_to_dict` for
   internal reuse instead), so this silently degraded a structured
   `RunRef` into an opaque `str(...)` repr the moment anything
   downstream (e.g. `scientific_report.ScientificReport.to_json()`)
   serialized it via `json.dumps(..., default=str)`.

2. `Contradiction.to_dict()` (contradictions.py) called
   `run_ref.to_dict()` directly on every entry of `self.run_refs` --
   but `RunRef` has no `to_dict()` method (see above), so this raised
   `AttributeError: 'RunRef' object has no attribute 'to_dict'`
   whenever a `Contradiction` with `run_refs` populated was
   serialized. Because `ScientificReport.to_dict()` /
   `pipeline.ScientificReport.to_dict()` both call
   `contradiction.to_dict()` for every entry of
   `RuleContext.detected_contradictions`, this crashed the entire
   reporting layer's `to_dict()` / `to_json()` output for any report
   whose `RuleContext` carried even one contradiction with `run_refs`
   set -- not a rare shape: `run_refs` is a documented, ordinary field
   of `Contradiction`.
"""

from __future__ import annotations

from experiment_audit_mcp.models import RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, Scope
from experiment_audit_mcp.reasoning.contradictions import Contradiction, ContradictionCategory


def _run_ref(run_id: str) -> RunRef:
    return RunRef(backend="wandb", entity="test-team", project="proj", run_id=run_id)


def test_claim_to_dict_serializes_evidence_trace_as_json_safe_dicts() -> None:
    """Regression test for bug #1.

    Before the fix, `Claim.to_dict()["evidence_trace"]` contained raw
    `RunRef` instances -- not JSON-safe, and inconsistent with every
    other `RunRef`-bearing field in this codebase, which always
    serializes to a plain `{"backend": ..., "entity": ..., "project":
    ..., "run_id": ...}` mapping.
    """
    ref = _run_ref("run-001")
    claim = Claim(
        id="C1",
        subject="s",
        statement="st",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(ref,),
    )

    serialized = claim.to_dict()

    assert serialized["evidence_trace"] == [
        {"backend": "wandb", "entity": "test-team", "project": "proj", "run_id": "run-001"}
    ]
    # Precondition guard: the bug only manifests when the field is not
    # already a plain dict -- confirm no raw RunRef instance leaked
    # through.
    assert not any(isinstance(entry, RunRef) for entry in serialized["evidence_trace"])


def test_contradiction_to_dict_does_not_crash_when_run_refs_populated() -> None:
    """Regression test for bug #2.

    Before the fix, this call raised `AttributeError: 'RunRef' object
    has no attribute 'to_dict'` unconditionally whenever `run_refs`
    was non-empty.
    """
    ref = _run_ref("run-001")
    claim_a = Claim(
        id="A", subject="s", statement="st-a", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    claim_b = Claim(
        id="B", subject="s", statement="st-b", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    contradiction = Contradiction(
        id="K1",
        categories=(ContradictionCategory.CLAIM,),
        claims=(claim_a, claim_b),
        run_refs=(ref,),
    )

    serialized = contradiction.to_dict()  # must not raise AttributeError

    assert serialized["run_refs"] == [
        {"backend": "wandb", "entity": "test-team", "project": "proj", "run_id": "run-001"}
    ]


def test_contradiction_to_dict_with_multiple_run_refs() -> None:
    """`run_refs` order and content are preserved, each converted
    independently -- not just the single-element case above.
    """
    ref_a = _run_ref("run-a")
    ref_b = _run_ref("run-b")
    claim_a = Claim(
        id="A", subject="s", statement="st-a", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    claim_b = Claim(
        id="B", subject="s", statement="st-b", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    contradiction = Contradiction(
        id="K2",
        categories=(ContradictionCategory.EVIDENCE,),
        claims=(claim_a, claim_b),
        run_refs=(ref_a, ref_b),
    )

    serialized = contradiction.to_dict()

    assert [entry["run_id"] for entry in serialized["run_refs"]] == ["run-a", "run-b"]
