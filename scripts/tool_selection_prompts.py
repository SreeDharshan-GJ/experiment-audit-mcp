"""Fixed prompt set for the Milestone 9 tool-selection eval.

Per design-spec-v1.md §7 point 3 and the roadmap's Milestone 9 entry:
unit tests can't verify that an MCP client actually *invokes* the right
tool from a natural-language prompt — that's a property of the tool's
name and description, evaluated by a real client, not of the Python
function underneath it. This module is the single source of truth for
that fixed prompt set: `scripts/tool_selection_eval.py` runs it against
a live client, and `docs/tool-selection-eval.md` is generated from (and
should stay in sync with) the same list rather than a hand-copied one.

15 prompts, covering all eight tools at least once. The four `audit_*`
prompts each have a paired **distractor** prompt deliberately worded to
tempt a nearby but wrong tool (e.g. `compare_runs` instead of
`audit_ablation`, `get_metric_history` instead of `audit_training_curve`)
— per spec §4.1, telling retrieval/diffing apart from judgment by name
alone is the whole point of the naming convention, so the eval should
actually stress that boundary, not just confirm each tool answers to its
own most obvious phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSelectionPrompt:
    prompt: str
    expected_tool: str
    rationale: str


TOOL_SELECTION_PROMPTS: list[ToolSelectionPrompt] = [
    ToolSelectionPrompt(
        prompt="Can you check that my W&B credentials are working before we start?",
        expected_tool="test_connection",
        rationale="Direct match to the tool's stated purpose (spec §6: fail fast).",
    ),
    ToolSelectionPrompt(
        prompt="What runs are in the mamfac project from the last week?",
        expected_tool="list_runs",
        rationale="Plain listing/filtering request, no judgment implied.",
    ),
    ToolSelectionPrompt(
        prompt="Give me the full config and final metrics for run xj29fk1a.",
        expected_tool="get_run_summary",
        rationale="Explicit single-run detail request, not a curve or a comparison.",
    ),
    ToolSelectionPrompt(
        prompt="Pull the loss curve for run xj29fk1a across the whole run.",
        expected_tool="get_metric_history",
        rationale=(
            "Asks for the raw curve itself, not a judgment about it — the "
            "distractor pair below asks the reward version of this."
        ),
    ),
    ToolSelectionPrompt(
        prompt="Why did my reward crash partway through training?",
        expected_tool="audit_training_curve",
        rationale=(
            "Asks for an interpretation ('why did it crash'), not the raw "
            "series — should route to the judgment tool, not "
            "get_metric_history."
        ),
    ),
    ToolSelectionPrompt(
        prompt="What do the reward values actually look like over time for this run?",
        expected_tool="get_metric_history",
        rationale=(
            "Distractor for audit_training_curve: phrased around reward "
            "like the prompt above, but asks to *see* the series, not "
            "judge it — should not trigger the heuristic tool."
        ),
    ),
    ToolSelectionPrompt(
        prompt="What's different between run A and run B — config and metrics?",
        expected_tool="compare_runs",
        rationale="Pure diff request across two runs, no claimed-variable framing.",
    ),
    ToolSelectionPrompt(
        prompt="Did I mess up this ablation? I only meant to change the learning rate.",
        expected_tool="audit_ablation",
        rationale="Canonical audit_ablation phrasing from the roadmap's own example.",
    ),
    ToolSelectionPrompt(
        prompt="I changed the learning rate for my ablation run — what else differs from baseline?",
        expected_tool="compare_runs",
        rationale=(
            "Distractor for audit_ablation: mentions 'ablation' but asks a "
            "flat diff question ('what else differs'), not a verdict on "
            "whether the change was isolated — a careful client should "
            "still be able to reach for the cheaper compare_runs here, "
            "though audit_ablation answering it too is not wrong. This "
            "case exists to probe over-triggering, not to assert a single "
            "correct answer as sharply as the others."
        ),
    ),
    ToolSelectionPrompt(
        prompt=(
            "Was my seed-only rerun actually a clean ablation, or did "
            "something else change too?"
        ),
        expected_tool="audit_ablation",
        rationale="Explicit verdict request ('actually a clean ablation') on a claimed variable.",
    ),
    ToolSelectionPrompt(
        prompt="Which hyperparameter mattered most in my sweep?",
        expected_tool="audit_sweep",
        rationale="Canonical audit_sweep phrasing from the roadmap's own example.",
    ),
    ToolSelectionPrompt(
        prompt="List every run in my learning-rate sweep so I can eyeball them myself.",
        expected_tool="list_runs",
        rationale=(
            "Distractor for audit_sweep: mentions 'sweep' but explicitly "
            "asks for a raw listing, not an importance ranking."
        ),
    ),
    ToolSelectionPrompt(
        prompt=(
            "Is learning_rate and batch_size confounded in my grid sweep, "
            "or independently important?"
        ),
        expected_tool="audit_sweep",
        rationale="Names the co-variance concern audit_sweep specifically handles.",
    ),
    ToolSelectionPrompt(
        prompt="My run's data looks incomplete — can you tell me if it finished ingesting?",
        expected_tool="get_run_summary",
        rationale=(
            "data_completeness lives on the Run object returned by "
            "get_run_summary; no audit_* tool exists purely to answer "
            "this, so a good client should fetch the run, not guess."
        ),
    ),
    ToolSelectionPrompt(
        prompt="Has my training run flatlined? The loss hasn't moved in a while.",
        expected_tool="audit_training_curve",
        rationale="'Flatlined' maps directly to the low_variance_plateau signal.",
    ),
]
