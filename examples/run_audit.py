"""Feed the real experiment above through experiment-audit-mcp's actual,
unmodified audit_ablation() -- the real function from the real package,
not a reimplementation.

Run `pip install -e .` from the repo root first so the package import
below resolves against your local source, not any installed PyPI copy.
"""

import json
from datetime import UTC, datetime

from experiment_audit_mcp.analysis.confound import audit_ablation
from experiment_audit_mcp.models import Run, RunRef

results = json.load(open("real_results.json"))

now = datetime.now(UTC)

baseline = Run(
    ref=RunRef(
        backend="wandb", entity="demo", project="memory-ablation-demo", run_id="baseline-001"
    ),
    name="baseline-use-memory-true",
    tags=["baseline"],
    status="finished",
    created_at=now,
    config={"use_memory": True, "batch_size": 32, "hidden_size": 32, "seed": 42},
    summary_metrics={"accuracy": results["baseline_acc"]},
    data_completeness="complete",
)

ablation = Run(
    ref=RunRef(
        backend="wandb", entity="demo", project="memory-ablation-demo", run_id="ablation-002"
    ),
    name="ablation-use-memory-false",
    tags=["ablation"],
    status="finished",
    created_at=now,
    # This is the config a researcher actually wrote down for this run --
    # batch_size=512 got copied in from a different template and nobody
    # noticed, because the only INTENDED change was use_memory.
    config={"use_memory": False, "batch_size": 512, "hidden_size": 32, "seed": 42},
    summary_metrics={"accuracy": results["ablation_acc"]},
    data_completeness="complete",
)

audit = audit_ablation(baseline, ablation, claimed_variable="use_memory")

print("=== REAL audit_ablation() OUTPUT (unmodified library function) ===")
print(json.dumps(audit.to_dict(), indent=2))
