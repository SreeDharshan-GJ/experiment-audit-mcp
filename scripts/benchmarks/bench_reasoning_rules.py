"""Audit #8 (performance) benchmark: reasoning-rules scalability.

Measures wall-clock time for `MissingEvidenceRule`, `ScopeRule`,
`ContradictionRule`, and `ConfidenceRule` -- the four rules audit #8
found repeatedly re-scanning `RuleContext.evidence_items()` (and, for
`ContradictionRule`, repeatedly comparing every pair of claims) once
per claim -- against synthetic `RuleContext`s of increasing size.

Usage:
    python scripts/benchmarks/bench_reasoning_rules.py [--sizes N N N ...]

Run this against an unmodified checkout to capture a "before" baseline,
and against the patched checkout to capture "after" numbers; the
`scripts/benchmarks/gen_bench_data.py` fixture generator and this
script's own logic are unaffected by the fix (they only exercise the
rules' public `evaluate()` entry points), so the same script is valid
for both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_bench_data import build_context  # noqa: E402

from experiment_audit_mcp.reasoning.scientific_rules import (  # noqa: E402
    ConfidenceRule,
    ContradictionRule,
    MissingEvidenceRule,
    ScopeRule,
)


def _time_rule(rule, context, repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        rule.evaluate(context)
    return (time.perf_counter() - start) / repeats


def run(sizes: list[int]) -> None:
    rules = {
        "MissingEvidenceRule (R001)": MissingEvidenceRule(),
        "ScopeRule (R002)": ScopeRule(),
        "ContradictionRule (R003)": ContradictionRule(),
        "ConfidenceRule (R004)": ConfidenceRule(),
    }

    header = f"{'claims':>8} | " + " | ".join(f"{name:>26}" for name in rules)
    print(header)
    print("-" * len(header))

    for size in sizes:
        context = build_context(size, evidence_per_claim=3, num_subjects=max(1, size // 20))
        timings = []
        for name, rule in rules.items():
            elapsed = _time_rule(rule, context)
            timings.append(f"{elapsed * 1000:>23.1f} ms")
        print(f"{size:>8} | " + " | ".join(timings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 300, 1000, 3000],
        help="claim-set sizes to benchmark (default: 100 300 1000 3000)",
    )
    args = parser.parse_args()
    run(args.sizes)
