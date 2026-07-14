"""Diagnostic: bypass the MCP server and WandbBackend entirely, call the
raw wandb SDK fresh, and see how many runs come back. This isolates
whether the list_runs gap is a real W&B API behavior or something
specific to the long-lived MCP server process / our wrapper code.

Run locally:
    python examples/diagnose_list_runs_gap.py
"""

import wandb

ENTITY = "dashh0558-gooning-uni"
PROJECT = "experiment-audit-"

api = wandb.Api()

print("=== Raw api.runs(), default args (what WandbBackend.list_runs calls) ===")
runs = list(api.runs(path=f"{ENTITY}/{PROJECT}"))
print(f"Count: {len(runs)}")
for r in runs:
    print(f"  {r.id}  name={r.name!r}  tags={r.tags}  state={r.state}  sweep={r.sweep_name}")

print()
print("=== Raw api.runs(), include_sweeps=True, larger per_page ===")
runs2 = list(api.runs(path=f"{ENTITY}/{PROJECT}", include_sweeps=True, per_page=100))
print(f"Count: {len(runs2)}")

print()
print("=== Sweep object directly ===")
sweep = api.sweep(f"{ENTITY}/{PROJECT}/c5340zfr")
sweep_runs = list(sweep.runs)
print(f"Sweep run count: {len(sweep_runs)}")
for r in sweep_runs:
    print(f"  {r.id}  name={r.name!r}  state={r.state}")
