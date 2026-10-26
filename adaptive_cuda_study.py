"""Paired CUDA study for fixed versus adaptive generic MPPI."""

import numpy as np
import torch

import mujoco_benchmark as benchmark

SEEDS = (7, 17, 27)
DEVICE = torch.device("cuda")


def run_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = benchmark.TwoLinkMuJoCo()
    data = benchmark.collect_task_directed_data(
        environment,
        transitions=2000,
        seed=seed,
    )
    states = benchmark.make_states(seed + 1000)[:1]
    normalizer = benchmark.make_normalizer(data, DEVICE)
    torch.manual_seed(seed)
    ensemble = benchmark.train_ensemble(data, DEVICE, normalizer)

    calibrator = benchmark.make_controller(
        ensemble,
        DEVICE,
        adaptive=True,
        pilot_rollouts=1,
        pilot_horizon=1,
    )
    torch.manual_seed(seed + 5000)
    threshold = calibrator.calibrate_threshold(
        benchmark.make_states(seed + 2000),
    )

    torch.manual_seed(seed + 9000)
    fixed = benchmark.evaluate(
        ensemble,
        DEVICE,
        states,
        adaptive=False,
        threshold=threshold,
    )
    torch.manual_seed(seed + 9000)
    adaptive = benchmark.evaluate(
        ensemble,
        DEVICE,
        states,
        adaptive=True,
        threshold=threshold,
        pilot_rollouts=1,
        pilot_horizon=1,
    )
    return threshold, fixed, adaptive


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * values.std(ddof=1) / np.sqrt(len(values))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this study")

    fixed_results = []
    adaptive_results = []
    print("seed threshold fixed_success adaptive_success fixed_ms adaptive_ms fixed_K adaptive_K")

    for seed in SEEDS:
        threshold, fixed, adaptive = run_seed(seed)
        fixed_results.append(fixed)
        adaptive_results.append(adaptive)
        print(
            f"{seed:4d} {threshold:.6f} "
            f"{fixed['success_rate']:.3f} "
            f"{adaptive['success_rate']:.3f} "
            f"{fixed['latency_ms']:.3f} "
            f"{adaptive['latency_ms']:.3f} "
            f"{fixed['effective_rollouts']:.1f} "
            f"{adaptive['effective_rollouts']:.1f}"
        )

    print("\n95% confidence intervals across paired seeds")
    for name, results in [("fixed", fixed_results), ("adaptive", adaptive_results)]:
        success_mean, success_ci = summarize([
            result["success_rate"] for result in results
        ])
        latency_mean, latency_ci = summarize([
            result["latency_ms"] for result in results
        ])
        rollout_mean, rollout_ci = summarize([
            result["effective_rollouts"] for result in results
        ])
        print(
            f"{name:<9} success={100 * success_mean:.1f}% +/- {100 * success_ci:.1f}% "
            f"latency={latency_mean:.3f} +/- {latency_ci:.3f} ms "
            f"effective_K={rollout_mean:.1f} +/- {rollout_ci:.1f}"
        )
