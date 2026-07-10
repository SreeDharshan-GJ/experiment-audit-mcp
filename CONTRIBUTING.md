# Contributing to experiment-audit-mcp

Thanks for considering a contribution. Before opening a PR, please read
this in full — the project's biggest source of accidental churn would be
a well-intentioned change that quietly breaks the frozen v1 contract.

## The v1 design is frozen

`docs/design-spec-v1.md` is marked **frozen for implementation**. That
means:

- Existing tool names, argument shapes, and return schemas
  (`test_connection`, `list_runs`, `get_run_summary`,
  `get_metric_history`, `compare_runs`, `audit_training_curve`,
  `audit_ablation`, `audit_sweep`) do not change without an explicit,
  logged design decision.
- `models.py`'s dataclasses (`RunRef`, `Run`, `MetricPoint`,
  `MetricHistory`, `Sweep`, `Page[T]`) do not change field names or
  types without the same process.
- The `ExperimentBackend` ABC (`backends/base.py`) does not gain or lose
  abstract methods without the same process.

If your change requires touching any of the above, don't silently patch
it. Instead, follow the pattern already established twice in this
project (see Revision 1 and Revision 2 in `docs/design-spec-v1.md`'s
Revision Log):

1. State the flaw precisely — what's wrong, and why the two closest
   workarounds you considered were rejected.
2. Propose the fix as a new, additive Revision entry in the design spec.
3. Get it reviewed and approved before writing implementation code
   against it.
4. Update every downstream file the revision touches (the spec, the
   roadmap, the affected modules and their tests) in the same PR — see
   Revision 2's "Downstream updates" list for the level of thoroughness
   expected.

Purely additive changes (a new optional parameter with a safe default, a
new non-breaking field) are the easiest case and don't require a full
revision write-up — but still flag them clearly in your PR description.

## New `audit_*` tools

If you're proposing a new judgment tool for a future version, it must
implement the mandatory `method` / `confidence` / `evidence` output
fields from the start (`docs/design-spec-v1.md` §8) — this is a
documented contribution requirement, not a style preference. A tool that
returns a bare verdict without evidence will not be merged regardless of
how useful the underlying heuristic is.

## Development setup

```bash
git clone https://github.com/<your-username>/experiment-audit-mcp.git
cd experiment-audit-mcp
pip install -e ".[dev]"
pytest
ruff check .
```

Every existing tool's logic is tested against `FakeBackend`
(`backends/fake_backend.py`), an in-memory test double that can inject
adversarial states (tiny sweeps, NaN mid-curve, correlated
hyperparameters, partial data) on demand. You should not need live W&B
credentials to develop or test new `analysis/` logic — only
`backends/wandb_backend.py` itself needs real API contact, and even that
is tested against recorded/synthetic fixtures, not live calls, per
`docs/design-spec-v1.md` §7.

## Testing expectations

- New pure logic (a new signal detector, a new analysis function) gets
  unit tests against hand-constructed inputs first — fast, isolated,
  no backend involved.
- New MCP-facing behavior gets at least one integration test invoking
  the tool through the MCP protocol layer, not just calling the Python
  function directly (see `tests/test_adversarial_mcp_layer.py` for the
  pattern).
- Any adversarial case you fix should get a named fixture or fake-backend
  builder, not just an assertion buried in an unrelated test — see
  `tests/fixtures/adversarial_cases.py` for the existing convention.
- Run `pytest` and `ruff check .` locally before opening a PR; CI
  (`.github/workflows/ci.yml`) runs both on every push and PR against
  Python 3.11 and 3.12.

## Reporting a design flaw without a fix in hand

If you spot a mismatch between the frozen spec and the actual
implementation (or between the spec and reality — e.g. a real backend
API doesn't map cleanly onto an abstraction the spec assumes), please
open an issue describing the flaw even if you don't have a proposed fix.
This project's own build history treats "stop and report" as a valid,
expected outcome of implementation work — see the Revision Log in
`docs/design-spec-v1.md` for two examples of exactly that happening.
