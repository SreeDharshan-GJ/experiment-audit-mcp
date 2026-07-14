"""
Logs the real memory-ablation proof-case experiment to a REAL, LIVE W&B
project -- actual wandb.init()/wandb.log() calls, not local JSON. This
is what closes the last untested gap: a real W&B project with real runs
in it, for the MCP server to actually query.

Run this locally (requires WANDB_API_KEY to be set, or you'll be
prompted to log in):

    pip install wandb scikit-learn numpy
    python examples/log_to_real_wandb.py

After it finishes, the runs will be visible at:
    https://wandb.ai/<your-entity>/experiment-audit-

Then, in Claude Desktop, ask it to use experiment-audit-mcp to list runs
in the "experiment-audit-" project.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import wandb

PROJECT = "experiment-audit-"  # matches your real W&B project slug
RNG_SEED = 42


def make_dataset():
    X, y = make_classification(
        n_samples=4000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=RNG_SEED,
    )
    return train_test_split(X, y, test_size=0.25, random_state=RNG_SEED)


def add_memory_feature(X, window=5):
    roll = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        lo = max(0, i - window)
        roll[i, 0] = X[lo : i + 1, :3].mean() if i > 0 else X[i, :3].mean()
    return np.hstack([X, roll])


def run_and_log(name, tags, config, X_train, X_test, y_train, y_test):
    """Trains for real and logs the real result to a real W&B run."""
    wandb.init(project=PROJECT, name=name, tags=tags, config=config, reinit=True)
    try:
        if config["use_memory"]:
            Xtr, Xte = add_memory_feature(X_train), add_memory_feature(X_test)
        else:
            Xtr, Xte = X_train, X_test
        clf = MLPClassifier(
            hidden_layer_sizes=(32,),
            batch_size=config["batch_size"],
            max_iter=60,
            random_state=config["seed"],
        )
        clf.fit(Xtr, y_train)
        acc = clf.score(Xte, y_test)
        wandb.log({"accuracy": acc})
        print(f"{name}: acc={acc:.4f}  (config={config})")
        return acc
    finally:
        wandb.finish()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_dataset()

    # Baseline: use_memory=True, batch_size=32
    run_and_log(
        "baseline-use-memory-true",
        ["baseline"],
        {"use_memory": True, "batch_size": 32, "hidden_size": 32, "seed": RNG_SEED},
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Ablation: use_memory=False -- batch_size accidentally also changed
    # to 512 (the confound). This is the pair the tool should flag.
    run_and_log(
        "ablation-use-memory-false",
        ["ablation"],
        {"use_memory": False, "batch_size": 512, "hidden_size": 32, "seed": RNG_SEED},
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Control: use_memory=False, batch_size=32 -- isolates the true
    # memory-only effect, for verification purposes.
    run_and_log(
        "control-use-memory-false-clean",
        ["control"],
        {"use_memory": False, "batch_size": 32, "hidden_size": 32, "seed": RNG_SEED},
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print()
    print(f"Done. View at: https://wandb.ai/{wandb.Api().default_entity}/{PROJECT}")
