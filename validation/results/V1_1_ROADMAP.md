# Version 1.1 Roadmap

## High Priority

### Feature

Oscillation Detection

Reason

Missing from audit_training_curve.

Possible approaches

- Rolling variance
- Peak detection
- FFT
- Autocorrelation

---

### Feature

Grouped Sweep Analysis

Current limitation

Requires native W&B Sweep.

Support

- Tags
- Run groups
- Filters

---

### Feature

Nonlinear Hyperparameter Importance

Current method

Linear Pearson correlation.

Future

- Spearman
- Mutual Information
- Random Forest Feature Importance
- SHAP-based ranking

---

## Validation Driven Development

Every Version 1.1 improvement originates from validation evidence rather than feature brainstorming.

This maintains an evidence-driven development process.
