# Launch write-up (Milestone 10 deliverable)

Per `docs/implementation-roadmap-v1.md`'s Milestone 10 and
`docs/design-spec-v1.md` §10: "Launch wedge: a 'Show-HN'-style post
framed around the confounded-ablation problem specifically, not a
generic feature announcement." Draft below, ready to post once the
package is actually published and the repository is public.

---

**Show HN: experiment-audit-mcp — an MCP server that catches confounded
ablations before you write them up**

I kept making the same mistake in my own RL experiments: change one
variable to test an ablation, only to later notice — usually while
writing the paper — that something else had also silently changed
between the two runs. Batch size. A seed that wasn't actually fixed.
Once, an entire optimizer swap I'd forgotten about. The "effect" I was
about to report wasn't the effect of the thing I claimed to be testing.

This is a boring, mechanical thing to check — diff two configs, see
what differs, decide whether the differences are innocuous (seed,
device) or not. It's exactly the kind of task an LLM is bad at doing
reliably by eye across dozens of hyperparameters, and exactly the kind
of task a few lines of deterministic code are good at. So I built an MCP
server around that idea, wired up to W&B.

`audit_ablation(baseline, ablation, claimed_variable)` fetches both
runs' full config, diffs them, and returns `clean` / `confounded` /
`uncertain` — with every differing parameter listed and tagged
`likely_intentional` or not, never a bare verdict. Two sibling tools do
the same "show your work" thing for training curves
(`audit_training_curve`: scored signals for NaNs, sudden jumps,
plateaus, oscillation — not a fixed label) and hyperparameter sweeps
(`audit_sweep`: correlation-based importance ranking that refuses to
answer below 10 runs and flags hyperparameters that moved together, so
you don't credit one for the other's effect).

The design principle underneath all three: retrieval and judgment are
structurally separate, and every judgment tool shows its evidence. An
agent (or a human skimming the tool list) can tell from the name alone
whether a result is a deterministic fact (`get_*`/`list_*`), a diff
(`compare_*`), or a heuristic (`audit_*`) — and every `audit_*` result
carries `method`, `confidence`, and `evidence`, enforced at the schema
level, not by convention.

It's W&B-only for now (MLflow is prototyped at the interface level but
not shipped), open source, MIT-licensed, and doesn't send your data
anywhere except your own W&B endpoint.

Repo: https://github.com/&lt;your-username&gt;/experiment-audit-mcp
Install: `pip install experiment-audit-mcp`

Happy to answer questions about the design, especially anywhere the
correlation-based sweep ranking's linear-only limitation might bite
people in practice — that's the part I'd most like scrutiny on.
