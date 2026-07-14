"""
A genuinely messy, real sweep -- not hand-picked to be clean.

Uses a real W&B sweep (wandb.sweep + wandb.agent, not just tagged runs --
this is what makes list_sweeps/audit_sweep actually exercised, since
those need a real sweep object). 12 real runs, random search over
learning_rate_init, hidden_layer_size, and batch_size. The
learning_rate_init range is intentionally wide enough to include the
unstable regime, so at least one or two runs are expected to genuinely
diverge (real NaN loss from real numerical instability, not simulated) --
that's exactly the kind of pathology audit_training_curve exists to
catch, and it isn't scripted to happen on a particular run; it emerges
from the actual random search.

Each run logs REAL per-epoch history (loss + accuracy every epoch via
manual partial_fit, not one final number), so audit_training_curve has
an actual curve to analyze, and a real final accuracy for audit_sweep
to rank hyperparameter importance against.

Run locally:
    python examples/log_real_sweep.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import wandb

PROJECT = "experiment-audit-"
N_RUNS = 12
N_EPOCHS = 25

sweep_config = {
    "method": "random",
    "metric": {"name": "final_accuracy", "goal": "maximize"},
    "parameters": {
        # Intentionally wide -- includes both reasonable and unstable
        # values, same way a real researcher's first sweep often does
        # before they've learned the safe range.
        "learning_rate_init": {"distribution": "log_uniform_values", "min": 1e-4, "max": 2.0},
        "hidden_layer_size": {"values": [16, 32, 64, 128]},
        "batch_size": {"values": [16, 32, 64, 128]},
    },
}


def train_one_run():
    run = wandb.init(project=PROJECT)
    cfg = run.config

    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    classes = np.unique(y_train)

    clf = MLPClassifier(
        hidden_layer_sizes=(cfg.hidden_layer_size,),
        learning_rate_init=cfg.learning_rate_init,
        batch_size=cfg.batch_size,
        max_iter=1,
        warm_start=True,
        random_state=0,
    )

    diverged = False
    crashed = False
    for epoch in range(N_EPOCHS):
        try:
            clf.partial_fit(X_train, y_train, classes=classes)
        except ValueError:
            # Real sklearn behavior: sufficiently unstable learning rates
            # raise here ("non-finite parameter weights") rather than
            # silently producing NaN. This is a genuine crash mid-run --
            # exactly the "partial data, run died" scenario real sweeps
            # produce, not something staged.
            wandb.log({"epoch": epoch, "train_loss": float("nan"), "accuracy": float("nan")})
            crashed = True
            break

        try:
            proba = clf.predict_proba(X_train)
            loss = log_loss(y_train, proba, labels=classes)
        except Exception:
            loss = float("nan")

        if np.isnan(loss) or np.isinf(loss):
            diverged = True

        acc = clf.score(X_test, y_test) if not diverged else float("nan")
        wandb.log({"epoch": epoch, "train_loss": loss, "accuracy": acc})

        if diverged:
            # Real early stop on real divergence -- some runs in a real
            # sweep just die partway through. This is exactly that.
            break

    final_acc = clf.score(X_test, y_test) if not (diverged or crashed) else float("nan")
    wandb.summary["final_accuracy"] = final_acc
    wandb.summary["diverged"] = diverged
    wandb.summary["crashed"] = crashed
    run.finish(exit_code=1 if crashed else 0)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    print(f"Real sweep created: {sweep_id}")
    wandb.agent(sweep_id, function=train_one_run, count=N_RUNS)
    print(f"Done. {N_RUNS} real runs logged to sweep {sweep_id}.")
    print(f"View at: https://wandb.ai/{wandb.Api().default_entity}/{PROJECT}/sweeps/{sweep_id}")
