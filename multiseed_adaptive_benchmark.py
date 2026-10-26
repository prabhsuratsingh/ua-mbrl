"""Multi-seed benchmark for fixed and pilot-adaptive ensemble MPPI."""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, GenericMPPI
from robustness_mismatch_experiment import (
    DT,
    DEVICE,
    EVAL_STEPS,
    HORIZON,
    NOMINAL_G,
    NOMINAL_M,
    U_MAX,
    angle_error,
    collect_data,
    pendulum_cost,
    pendulum_step,
    train_models,
)

SEEDS = (0, 1)
EPISODES_PER_SEED = 1
LOW_K = 100
HIGH_K = 300
BENCHMARK_STEPS = 60


def make_states(seed):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            [rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)]
            for _ in range(EPISODES_PER_SEED)
        ],
        dtype=np.float32,
    )


def make_controller(model, method):
    common = {
        "dynamics": model,
        "action_dim": 1,
        "horizon": HORIZON,
        "action_low": [-5.0],
        "action_high": [5.0],
        "target": [np.pi, 0.0],
        "cost_fn": pendulum_cost,
        "dt": DT,
        "temperature": 1.0,
        "noise_sigma": 1.0,
        "device": DEVICE,
    }
    if method == "adaptive":
        return AdaptiveGenericMPPI(
            **common,
            min_rollouts=LOW_K,
            max_rollouts=HIGH_K,
            uncertainty_threshold=0.006,
            pilot_rollouts=25,
            pilot_horizon=5,
        )
    return GenericMPPI(
        **common,
        rollouts=LOW_K if method == "fixed_low" else HIGH_K,
    )


def run_episode(controller, x0, condition, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    rollout_count = 0
    errors = []

    for _ in range(BENCHMARK_STEPS):
        raw_action = controller.command(x)
        action = float(np.asarray(raw_action).reshape(-1)[0])
        rollout_count += getattr(controller, "last_rollouts", controller.rollouts)
        if isinstance(controller, AdaptiveGenericMPPI):
            rollout_count += (
                controller.pilot_rollouts
                * controller.pilot_horizon
            )
        prediction = controller.dynamics.model_step(x, action)
        next_x = pendulum_step(x, action, **condition)
        error = prediction - next_x
        error[0] = angle_error(prediction[0], next_x[0])
        errors.append(error)
        x = next_x

    return {
        "success": abs(angle_error(x[0])) < 0.1 and abs(x[1]) < 0.5,
        "theta_error": abs(angle_error(x[0])),
        "omega_error": abs(x[1]),
        "effective_rollouts": rollout_count,
        "model_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
    }


def evaluate_method(model, method, states, condition, seed):
    results = [
        run_episode(
            make_controller(model, method),
            state,
            condition,
            seed * 100 + index,
        )
        for index, state in enumerate(states)
    ]
    return {
        "success": np.mean([item["success"] for item in results]),
        "theta": np.mean([item["theta_error"] for item in results]),
        "omega": np.mean([item["omega_error"] for item in results]),
        "rollouts": np.mean([item["effective_rollouts"] for item in results]),
        "rmse": np.mean([item["model_rmse"] for item in results]),
    }


def summarize(records):
    values = np.asarray(records, dtype=np.float64)
    mean = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, 1.96 * standard_error


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    print(f"Seeds: {SEEDS}, episodes per seed: {EPISODES_PER_SEED}")
    print("Training one nominal bootstrap ensemble...")
    _, ensemble = train_models(collect_data())

    conditions = {
        "nominal": {},
        "mass+20%": {"mass": 1.2},
        "gravity-15%": {"gravity": NOMINAL_G * 0.85},
    }
    methods = ("fixed_low", "fixed_high", "adaptive")
    all_records = {
        (condition_name, method): []
        for condition_name in conditions
        for method in methods
    }

    for seed in SEEDS:
        states = make_states(seed)
        for condition_name, condition in conditions.items():
            for method in methods:
                metrics = evaluate_method(
                    ensemble,
                    method,
                    states,
                    condition,
                    seed,
                )
                all_records[(condition_name, method)].append(metrics)
                print(
                    f"seed={seed} condition={condition_name:<10} "
                    f"method={method:<10} "
                    f"success={metrics['success']:.2f} "
                    f"K={metrics['rollouts']:.1f}"
                )

    print("\n95% confidence intervals across seeds")
    print("condition    method       success       theta        omega        K")
    print("-" * 76)
    for condition_name in conditions:
        for method in methods:
            records = all_records[(condition_name, method)]
            matrix = np.asarray([
                [
                    item["success"],
                    item["theta"],
                    item["omega"],
                    item["rollouts"],
                ]
                for item in records
            ])
            mean, interval = summarize(matrix)
            print(
                f"{condition_name:<12} {method:<12} "
                f"{mean[0]:.3f} +/- {interval[0]:.3f} "
                f"{mean[1]:.3f} +/- {interval[1]:.3f} "
                f"{mean[2]:.3f} +/- {interval[2]:.3f} "
                f"{mean[3]:.1f} +/- {interval[3]:.1f}"
            )
