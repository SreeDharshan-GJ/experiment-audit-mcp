"""
generate_wandb_testbed.py

Creates a small, realistic Weights & Biases project with several runs
purpose-built to exercise every tool in the experiment-audit MCP server:

  list_runs             -> just needs >1 run in the project
  get_run_summary        -> needs varied configs + final summary metrics
  get_metric_history     -> needs full step-by-step metric logging
  compare_runs            -> needs 2+ runs with overlapping + differing config
  audit_training_curve    -> needs runs with BOTH clean and pathological curves
                             (NaNs, plateaus, oscillation, level shifts)
  audit_ablation          -> needs a matched pair that differs in exactly
                             one config flag ("use_memory")
  audit_sweep             -> needs a proper hyperparameter sweep where one
                             hyperparameter has a real, learnable effect on
                             the outcome metric and others are close to noise

Usage:
    pip install wandb --break-system-packages
    wandb login
    python generate_wandb_testbed.py --entity YOUR_WANDB_ENTITY

All runs live in one project: "experiment-audit-testbed" (override with --project).
"""

import argparse
import math
import random

import wandb

PROJECT_DEFAULT = "experiment-audit-testbed"


def simulate_training(
    run,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    use_memory: bool,
    pathology: str | None = None,
    seed: int = 0,
):
    """
    Logs a synthetic but realistic loss/accuracy curve to `run`, shaped by
    hyperparameters and an optional injected pathology.

    Hyperparameter effects (roughly, so audit_sweep has real signal):
      - lr: too high -> noisier/unstable convergence, too low -> slower
      - hidden_dim: bigger -> lower loss floor, small diminishing returns
      - batch_size: mostly noise-reduction, weak effect on final loss
      - use_memory: meaningfully lowers loss floor and boosts accuracy
        (this is the ablation variable)
    """
    rng = random.Random(seed)

    # Base convergence rate shaped by lr (too high -> worse late-stage loss)
    lr_penalty = 0.0 if lr <= 0.003 else (lr - 0.003) * 15
    convergence_rate = max(0.02, 0.12 - lr_penalty * 0.02)

    # Capacity effect: bigger hidden_dim -> lower floor, log-diminishing
    capacity_bonus = math.log2(max(hidden_dim, 8) / 64) * 0.03

    # Memory effect: the ablation variable, meaningful and consistent
    memory_bonus = 0.08 if use_memory else 0.0

    # Batch size: weak noise-reduction effect only
    noise_scale = 0.05 * (32 / max(batch_size, 8)) ** 0.5

    loss_floor = max(0.05, 0.55 - capacity_bonus - memory_bonus)
    acc_ceiling = min(0.97, 0.62 + capacity_bonus * 1.5 + memory_bonus * 1.8)

    nan_step = rng.randint(steps // 3, steps // 2) if pathology == "nan_spike" else None
    shift_step = rng.randint(steps // 2, int(steps * 0.7)) if pathology == "level_shift" else None
    plateau_start = int(steps * 0.3) if pathology == "plateau" else None

    loss = 1.5
    acc = 0.10
    for step in range(steps):
        # --- base dynamics ---
        target_loss = loss_floor + (1.5 - loss_floor) * math.exp(-convergence_rate * step)
        loss += (target_loss - loss) * 0.3
        loss += rng.gauss(0, noise_scale)

        target_acc = acc_ceiling - (acc_ceiling - 0.10) * math.exp(-convergence_rate * step)
        acc += (target_acc - acc) * 0.3
        acc += rng.gauss(0, 0.01)

        # --- pathology injection ---
        if pathology == "oscillation":
            loss += 0.35 * math.sin(step * 1.7) * math.exp(-step / steps)
        if pathology == "plateau" and step >= plateau_start:
            # freeze near a mediocre value regardless of "progress"
            loss = loss_floor + 0.28
            acc = acc_ceiling - 0.30
        if pathology == "nan_spike" and step == nan_step:
            loss = float("nan")
        if pathology == "level_shift" and step >= shift_step:
            loss += 0.4  # sudden, unexplained jump that never recovers

        loss = max(loss, 0.0) if not math.isnan(loss) else loss
        acc = min(max(acc, 0.0), 1.0)

        run.log({"train/loss": loss, "train/accuracy": acc}, step=step)

    final_loss = loss if not math.isnan(loss) else float("nan")
    run.summary["final_loss"] = final_loss
    run.summary["final_accuracy"] = acc
    run.summary["best_accuracy"] = acc_ceiling if pathology != "plateau" else acc


def make_run(entity, project, name, tags, config, pathology=None, seed=0, steps=60):
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        tags=tags,
        config=config,
        reinit=True,
    )
    simulate_training(
        run,
        steps=steps,
        lr=config["lr"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        use_memory=config["use_memory"],
        pathology=pathology,
        seed=seed,
    )
    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a W&B testbed project for experiment-audit",
    )
    parser.add_argument("--entity", required=True, help="Your W&B entity (username or team)")
    parser.add_argument("--project", default=PROJECT_DEFAULT, help="W&B project name")
    args = parser.parse_args()

    base_config = dict(lr=0.001, batch_size=32, hidden_dim=128, use_memory=True)

    # 1) Clean baseline runs (for list_runs, get_run_summary, get_metric_history)
    make_run(args.entity, args.project, "baseline", ["baseline"], dict(base_config), seed=1)

    # 2) Ablation pair — differs ONLY in use_memory (for audit_ablation)
    make_run(
        args.entity,
        args.project,
        "ablation_memory_on",
        ["ablation"],
        dict(base_config, use_memory=True),
        seed=2,
    )
    make_run(
        args.entity,
        args.project,
        "ablation_memory_off",
        ["ablation"],
        dict(base_config, use_memory=False),
        seed=2,
    )

    # 3) Hyperparameter sweep — lr has a real effect, batch_size/hidden_dim
    #    contribute smaller effects, giving audit_sweep something to rank
    sweep_grid = [
        dict(lr=0.0005, batch_size=32, hidden_dim=128, use_memory=True),
        dict(lr=0.001, batch_size=32, hidden_dim=128, use_memory=True),
        dict(lr=0.003, batch_size=32, hidden_dim=128, use_memory=True),
        dict(lr=0.01, batch_size=32, hidden_dim=128, use_memory=True),  # too high -> worse
        dict(lr=0.001, batch_size=16, hidden_dim=128, use_memory=True),
        dict(lr=0.001, batch_size=128, hidden_dim=128, use_memory=True),
        dict(lr=0.001, batch_size=32, hidden_dim=64, use_memory=True),
        dict(lr=0.001, batch_size=32, hidden_dim=256, use_memory=True),
    ]
    for i, cfg in enumerate(sweep_grid):
        make_run(
            args.entity,
            args.project,
            f"sweep_{i:02d}",
            ["sweep"],
            cfg,
            seed=10 + i,
        )

    # 4) Pathological runs (for audit_training_curve)
    make_run(
        args.entity,
        args.project,
        "pathology_nan",
        ["pathology"],
        dict(base_config),
        pathology="nan_spike",
        seed=20,
    )
    make_run(
        args.entity,
        args.project,
        "pathology_plateau",
        ["pathology"],
        dict(base_config),
        pathology="plateau",
        seed=21,
    )
    make_run(
        args.entity,
        args.project,
        "pathology_oscillation",
        ["pathology"],
        dict(base_config),
        pathology="oscillation",
        seed=22,
    )
    make_run(
        args.entity,
        args.project,
        "pathology_level_shift",
        ["pathology"],
        dict(base_config),
        pathology="level_shift",
        seed=23,
    )

    print(f"\nDone. Created 14 runs in entity='{args.entity}', project='{args.project}'.")
    print("Suggested first checks with experiment-audit:")
    print("  - list_runs on this project")
    print("  - get_run_summary / get_metric_history on 'baseline'")
    print("  - compare_runs on ['baseline', 'ablation_memory_on', 'ablation_memory_off']")
    print("  - audit_ablation on the ablation_memory_on / ablation_memory_off pair")
    print("  - audit_sweep on the 8 'sweep_*' runs")
    print("  - audit_training_curve on each 'pathology_*' run's train/loss metric")


if __name__ == "__main__":
    main()
