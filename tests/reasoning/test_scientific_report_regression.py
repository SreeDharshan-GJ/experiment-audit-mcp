"""Regression tests for confirmed bugs found in production audit #5 of
`scientific_report.py`, the reporting layer's public output type:

1. **Markdown table row injection.** `to_markdown()`'s Claims and
   Scientific Judgment tables interpolate caller-supplied free text
   (`Claim.id`, `Claim.subject`) directly into `| ... | ... |` rows
   with no escaping. Neither `claims.py`'s `Claim.__post_init__` nor
   this module restricts those fields beyond "non-empty string," so a
   claim whose `subject` or `id` contains a literal `|` or an embedded
   newline silently splits into extra columns or extra rows,
   corrupting the rendered table -- exactly the GitHub Actions check
   summary / PR comment rendering this module's own docstring names as
   a primary consumer of `to_markdown()`.

2. **End-to-end crash via `Contradiction.run_refs`.** Before
   `contradictions.py`'s `Contradiction.to_dict()` was fixed (see
   `test_claims_contradictions_serialization_regression.py`), any
   `ScientificReport` whose underlying `RuleContext.detected_contradictions`
   contained a `Contradiction` with `run_refs` populated crashed on
   `to_dict()` / `to_json()` with `AttributeError: 'RunRef' object has
   no attribute 'to_dict'`. This test exercises that failure through
   the full reporting layer's public API, not just the unit in
   isolation, since `to_dict()` / `to_json()` are what CLI, MCP, and
   web-dashboard callers actually depend on per this module's own
   documented architecture.
"""

from __future__ import annotations

import json

from experiment_audit.models import RunRef
from experiment_audit.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit.reasoning.contradictions import Contradiction, ContradictionCategory
from experiment_audit.reasoning.hypotheses import HypothesisSet
from experiment_audit.reasoning.observations import ObservationSet
from experiment_audit.reasoning.pipeline import ScientificReasoningPipeline
from experiment_audit.reasoning.rules import RuleContext
from experiment_audit.reasoning.scientific_report import ScientificReport


def _run_ref(run_id: str) -> RunRef:
    return RunRef(backend="wandb", entity="test-team", project="proj", run_id=run_id)


def _build_report(context: RuleContext) -> ScientificReport:
    pipeline_report = ScientificReasoningPipeline().execute(context)
    return ScientificReport.from_pipeline_report(pipeline_report)


def test_to_markdown_claims_table_survives_pipe_and_newline_in_subject() -> None:
    """Regression test for bug #1 (Claims table).

    Before the fix, a claim subject containing `|` and a newline
    produced a Claims table row that visually split into extra
    columns/rows, corrupting every subsequent row in the same table.
    After the fix, the entire logical row -- including the pipe and
    the collapsed newline -- must appear on a single markdown table
    line, and the table must still have exactly one header separator
    row plus one data row.
    """
    claim = Claim(
        id="C1",
        subject="Model | injected row break\nSecond line",
        statement="Model beats baseline.",
        category=ClaimCategory.COMPARISON,
        scope=Scope(),
    )
    context = RuleContext(
        evidence=[],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim]),
    )
    report = _build_report(context)

    markdown = report.to_markdown()
    lines = markdown.splitlines()

    claims_header_index = lines.index("## Claims")
    table_lines = [
        line
        for line in lines[claims_header_index : claims_header_index + 10]
        if line.startswith("|")
    ]

    # Exactly one header row, one separator row, one data row -- the
    # claim's own `|` and newline must never manufacture extra rows.
    assert len(table_lines) == 3, (
        f"Expected exactly 3 markdown table lines (header, separator, one "
        f"data row) for a single claim; got {len(table_lines)}: {table_lines!r}"
    )

    data_row = table_lines[2]
    assert data_row.count("\n") == 0
    # The literal pipe from the claim's subject must be escaped, not
    # treated as a column delimiter.
    assert "Model \\| injected row break Second line" in data_row
    # The row must still have exactly 5 columns (ID, Subject, Category,
    # Stage, Strength) -- 6 delimiter pipes, plus the one escaped
    # (`\|`) literal pipe from the claim's own subject.
    assert data_row.count("|") == 7


def test_to_markdown_judgment_table_survives_pipe_in_claim_id() -> None:
    """Regression test for bug #1 (Scientific Judgment table).

    `JudgmentRule`'s reported judgment values are a closed enum, but
    the claim id used as the table's row key is caller-supplied free
    text and was not escaped before the fix.
    """
    ref = _run_ref("run-001")
    claim = Claim(
        id="C|1",
        subject="s",
        statement="Model achieves 0.9 accuracy on Benchmark-B.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(ref,),
    )
    context = RuleContext(
        evidence=[],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim]),
    )
    report = _build_report(context)
    assert report.judgments(), (
        "Test fixture must reach JudgmentRule for this test to be meaningful."
    )

    markdown = report.to_markdown()
    lines = markdown.splitlines()
    judgment_header_index = lines.index("## Scientific Judgment")
    table_lines = [
        line
        for line in lines[judgment_header_index : judgment_header_index + 10]
        if line.startswith("|")
    ]

    assert len(table_lines) == 3
    assert "C\\|1" in table_lines[2]


def test_report_to_dict_does_not_crash_when_detected_contradiction_has_run_refs() -> None:
    """Regression test for bug #2, at the `ScientificReport.to_dict()`
    layer (not just `Contradiction.to_dict()` in isolation).
    """
    ref = _run_ref("run-001")
    claim_a = Claim(
        id="A", subject="s", statement="st-a", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    claim_b = Claim(
        id="B", subject="s", statement="st-b", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    known_contradiction = Contradiction(
        id="K1",
        categories=(ContradictionCategory.CLAIM,),
        claims=(claim_a, claim_b),
        run_refs=(ref,),
    )
    context = RuleContext(
        evidence=[],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim_a]),
        detected_contradictions=(known_contradiction,),
    )
    report = _build_report(context)

    serialized = report.to_dict()  # must not raise AttributeError
    detected = serialized["sections"]["contradictions"]["detected"]
    assert detected and detected[0]["run_refs"] == [
        {"backend": "wandb", "entity": "test-team", "project": "proj", "run_id": "run-001"}
    ]


def test_report_to_json_does_not_crash_when_detected_contradiction_has_run_refs() -> None:
    """Same as above, through the full `to_json()` path, which is what
    an MCP tool or CLI caller actually invokes.
    """
    ref = _run_ref("run-001")
    claim_a = Claim(
        id="A", subject="s", statement="st-a", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    claim_b = Claim(
        id="B", subject="s", statement="st-b", category=ClaimCategory.PERFORMANCE, scope=Scope()
    )
    known_contradiction = Contradiction(
        id="K1",
        categories=(ContradictionCategory.CLAIM,),
        claims=(claim_a, claim_b),
        run_refs=(ref,),
    )
    context = RuleContext(
        evidence=[],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim_a]),
        detected_contradictions=(known_contradiction,),
    )
    report = _build_report(context)

    text = report.to_json()  # must not raise AttributeError
    parsed = json.loads(text)
    assert parsed["sections"]["contradictions"]["detected"][0]["run_refs"][0]["run_id"] == "run-001"
