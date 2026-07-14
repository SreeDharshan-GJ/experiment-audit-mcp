"""
Real (not fabricated) toy experiment demonstrating a confounded ablation,
built to feed through experiment-audit-mcp's actual audit_ablation()
function -- not a synthetic narrative, actual sklearn training runs with
actual randomness and actual numbers.

Scenario: a researcher wants to test "does a memory feature help
accuracy?" (claimed_variable = "use_memory"). While setting up the
"no memory" condition, they copy a config template that also has a
different batch_size in it -- a completely realistic, easy-to-make
mistake. The memory feature is a genuine engineered feature (rolling
window statistics), so it has a real, measurable effect -- but so does
the accidental batch_size change, and a human skimming the final
accuracy numbers has no way to disentangle the two.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
    """A genuine engineered feature: rolling mean over a sliding window
    of prior rows, simulating a 'memory' mechanism that legitimately
    carries information forward."""
    roll = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        lo = max(0, i - window)
        roll[i, 0] = X[lo : i + 1, :3].mean() if i > 0 else X[i, :3].mean()
    return np.hstack([X, roll])


def run_condition(X_train, X_test, y_train, y_test, *, use_memory, batch_size, seed):
    if use_memory:
        Xtr, Xte = add_memory_feature(X_train), add_memory_feature(X_test)
    else:
        Xtr, Xte = X_train, X_test
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        batch_size=batch_size,
        max_iter=60,
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    acc = clf.score(Xte, y_test)
    return acc


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_dataset()

    # Baseline: use_memory=True, batch_size=32 (the project default)
    baseline_acc = run_condition(
        X_train,
        X_test,
        y_train,
        y_test,
        use_memory=True,
        batch_size=32,
        seed=RNG_SEED,
    )

    # "Ablation": use_memory=False -- but batch_size accidentally also
    # changed to 512 because the config was copied from a different
    # template. This is the confound.
    ablation_acc = run_condition(
        X_train,
        X_test,
        y_train,
        y_test,
        use_memory=False,
        batch_size=512,
        seed=RNG_SEED,
    )

    # Control: use_memory=False, batch_size=32 -- isolates what the
    # memory feature's TRUE effect is, with the confound removed.
    # A researcher wouldn't normally run this -- it's how we verify
    # the tool was right to flag the confound.
    control_acc = run_condition(
        X_train,
        X_test,
        y_train,
        y_test,
        use_memory=False,
        batch_size=32,
        seed=RNG_SEED,
    )

    print(f"baseline  (use_memory=True,  batch_size=32): acc={baseline_acc:.4f}")
    print(f"ablation  (use_memory=False, batch_size=512): acc={ablation_acc:.4f}")
    print(f"control   (use_memory=False, batch_size=32): acc={control_acc:.4f}")
    print()
    print(f"naive delta (baseline - ablation): {baseline_acc - ablation_acc:+.4f}")
    print(f"true memory-only delta (baseline - control): {baseline_acc - control_acc:+.4f}")
    print(f"confound's own contribution (control - ablation): {control_acc - ablation_acc:+.4f}")

    import json

    json.dump(
        {"baseline_acc": baseline_acc, "ablation_acc": ablation_acc, "control_acc": control_acc},
        open("real_results.json", "w"),
        indent=2,
    )
