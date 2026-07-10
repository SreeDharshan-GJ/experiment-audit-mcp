# Bibliography

Full synthesis and per-entry analysis in `related-work.md`. This is a flat
reference list only.

## Reproducibility
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.
  (2018). Deep Reinforcement Learning that Matters. *AAAI 2018*.
  https://arxiv.org/abs/1709.06560
- Islam, R., Henderson, P., Gomrokchi, M., Precup, D. (2017). Reproducibility
  of benchmarked deep reinforcement learning tasks for continuous control.
  *ICML 2017 Reproducibility in ML Workshop*.

## Statistical significance / multiple comparisons
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The Hitchhiker's
  Guide to Testing Statistical Significance in NLP. *ACL 2018*.
  https://aclanthology.org/P18-1128/
- Dror, R., Baumer, G., Bogomolov, M., Reichart, R. (2017). Replicability
  Analysis for NLP: Testing Significance with Multiple Datasets.
  https://arxiv.org/abs/1709.09500
- Dror, R., Peled-Cohen, L., Shlomov, S., Reichart, R. (2020). Statistical
  Significance Testing for Natural Language Processing. *Synthesis Lectures
  on Human Language Technologies*, Morgan & Claypool.

## Hyperparameter importance
- Hutter, F., Hoos, H., Leyton-Brown, K. (2014). An Efficient Approach for
  Assessing Hyperparameter Importance. *ICML 2014*.
  http://proceedings.mlr.press/v32/hutter14.html
- Optuna documentation: `optuna.importance.FanovaImportanceEvaluator`.
  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html

## Ablation methodology
- Lipton, Z., Steinhardt, J. (2018/2019). Troubling Trends in Machine
  Learning Scholarship. *ACM Queue* 17(1). https://arxiv.org/abs/1807.03341
- Biderman, S., Scheirer, W. (2020). Pitfalls in Machine Learning Research:
  Reexamining the Development Cycle. https://arxiv.org/abs/2011.02832
- Fostiropoulos, I. et al. (2023). ABLATOR: Robust Horizontal-Scaling of
  Machine Learning Ablation Experiments. *MLR Press*.
- Abramovich, T., Chechik, G. (2025, rev. 2026). AblationBench: Evaluating
  Automated Planning of Ablations in Empirical AI Research.
  https://arxiv.org/abs/2507.08038

## Reward hacking / training-curve pathology
- Multiple 2025-2026 preprints on reward-hacking / specification-gaming
  detection via learned classifiers over RL episode features (ROC-AUC based
  evaluation); see related-work.md §5 for framing and caveats. Citation set
  is fast-moving in this subfield — re-search before relying on any single
  paper as canonical.

## Product / retrieval-layer competitors (not academic literature)
- W&B official MCP server: https://github.com/wandb/wandb-mcp-server,
  https://docs.wandb.ai/platform/mcp-server
