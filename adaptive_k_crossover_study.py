"""Adaptive-K crossover study.

Prior work in this project (README.md, research_status.tex) found that
pilot-gated adaptive rollout allocation does not pay off on GPU at K_max=300:
pilot-rollout overhead exceeds the savings from skipping full-K rollouts,
and this held up even under a paired 3-seed follow-up. That conclusion was
never tested across a wider range of K_max, and the 3-seed sample size was
explicitly flagged as too small to trust.

This script re-runs the fixed-vs-adaptive comparison at a much larger seed
count (affordable now via generic_mbrl.SeedBatchedGenericMPPI, which batches
independent seeds as an extra tensor dimension) across a sweep of K_max, to
find out whether there is a rollout-budget regime where adaptive-K actually
wins on latency, or whether the negative result holds everywhere tested.

Fixed-K conditions use the seed-batched controller (all S seeds advanced in
one batched call). Adaptive-K needs a different rollout count per seed
depending on that seed's own pilot uncertainty, which does not fit the
seed-batched controller's dense-tensor design (see generic_mbrl.py's
SeedBatchedGenericMPPI docstring), so adaptive runs as a sequential
per-seed loop -- this is also what makes the reported latency comparison
fair: it is "cost to advance all S seeds by one control step" under each
approach, not a per-call microbenchmark.
"""

import time

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, SeedBatchedGenericMPPI
from robustness_mismatch_experiment import (
    DEVICE,
    DT,
    U_MAX,
    U_MIN,
    angle_error,
    collect_data,
    make_ensemble,
    pendulum_cost,
    pendulum_step,
    train_models,
)

HORIZON = 20
K_MIN = 100
K_MAX_VALUES = (100, 300, 1000, 3000, 8000)
PILOT_ROLLOUTS = 25
PILOT_HORIZON = 5
NUM_SEEDS = 20
EVAL_STEPS = 30
CALIBRATION_STATES_SEED = 900
INITIAL_STATES_SEED = 800


def make_initial_states(seed, count):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [[rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)] for _ in range(count)],
        dtype=np.float32,
    )


def calibrate_threshold(ensemble, k_max, calibration_states):
    calibrator = AdaptiveGenericMPPI(
        dynamics=ensemble,
        action_dim=1,
        horizon=HORIZON,
        action_low=[U_MIN],
        action_high=[U_MAX],
        target=[np.pi, 0.0],
        cost_fn=pendulum_cost,
        dt=DT,
        temperature=1.0,
        noise_sigma=1.0,
        min_rollouts=K_MIN,
        max_rollouts=k_max,
        uncertainty_threshold=0.0,
        pilot_rollouts=PILOT_ROLLOUTS,
        pilot_horizon=PILOT_HORIZON,
        device=DEVICE,
    )
    return calibrator.calibrate_threshold(calibration_states)


def run_fixed_high_batched(ensemble, initial_states, k_max, steps, seed):
    num_seeds = len(initial_states)
    controller = SeedBatchedGenericMPPI(
        dynamics=ensemble,
        num_seeds=num_seeds,
        action_dim=1,
        horizon=HORIZON,
        rollouts=k_max,
        action_low=[U_MIN],
        action_high=[U_MAX],
        target=[np.pi, 0.0],
        cost_fn=pendulum_cost,
        dt=DT,
        temperature=1.0,
        noise_sigma=1.0,
        device=DEVICE,
    )
    controller.reset()
    torch.manual_seed(seed)
    x = np.asarray(initial_states, dtype=np.float32)
    step_latencies = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = controller.command(x)
        torch.cuda.synchronize()
        step_latencies.append(time.perf_counter() - start)
        u = np.asarray(actions).reshape(num_seeds)
        x = np.stack([pendulum_step(x[s], float(u[s])) for s in range(num_seeds)])

    theta_err = np.abs(np.asarray([angle_error(x[s, 0]) for s in range(num_seeds)]))
    omega_err = np.abs(x[:, 1])
    success = (theta_err < 0.1) & (omega_err < 0.5)
    return {
        "success": float(success.mean()),
        "total_latency_ms": float(np.mean(step_latencies) * 1000),
        "per_seed_latency_ms": float(np.mean(step_latencies) * 1000 / num_seeds),
        "effective_rollouts": float(k_max),
    }


def run_adaptive_sequential(ensemble, initial_states, k_max, threshold, steps, seed):
    num_seeds = len(initial_states)
    successes = []
    step_latencies_per_seed = []
    rollout_counts = []

    for s in range(num_seeds):
        np.random.seed(seed + s)
        torch.manual_seed(seed + s)
        controller = AdaptiveGenericMPPI(
            dynamics=ensemble,
            action_dim=1,
            horizon=HORIZON,
            action_low=[U_MIN],
            action_high=[U_MAX],
            target=[np.pi, 0.0],
            cost_fn=pendulum_cost,
            dt=DT,
            temperature=1.0,
            noise_sigma=1.0,
            min_rollouts=K_MIN,
            max_rollouts=k_max,
            uncertainty_threshold=threshold,
            pilot_rollouts=PILOT_ROLLOUTS,
            pilot_horizon=PILOT_HORIZON,
            device=DEVICE,
        )
        controller.reset()
        x = np.asarray(initial_states[s], dtype=np.float32)
        seed_latencies = []
        seed_rollouts = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = controller.command(x)
            torch.cuda.synchronize()
            seed_latencies.append(time.perf_counter() - start)
            seed_rollouts.append(controller.last_rollouts + PILOT_ROLLOUTS * PILOT_HORIZON)
            u = float(np.asarray(action).reshape(-1)[0])
            x = pendulum_step(x, u)

        theta_err = abs(angle_error(x[0]))
        omega_err = abs(x[1])
        successes.append(theta_err < 0.1 and omega_err < 0.5)
        step_latencies_per_seed.append(np.mean(seed_latencies))
        rollout_counts.append(np.mean(seed_rollouts))

    return {
        "success": float(np.mean(successes)),
        "total_latency_ms": float(np.sum(step_latencies_per_seed) * 1000),
        "per_seed_latency_ms": float(np.mean(step_latencies_per_seed) * 1000),
        "effective_rollouts": float(np.mean(rollout_counts)),
    }


def summarize_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    print(f"Seeds: {NUM_SEEDS}, eval steps: {EVAL_STEPS}, K_min: {K_MIN}, K_max sweep: {K_MAX_VALUES}")

    print("\nTraining one nominal bootstrap ensemble...")
    _, ensemble = train_models(collect_data())

    calibration_states = make_initial_states(CALIBRATION_STATES_SEED, 40)
    initial_states = make_initial_states(INITIAL_STATES_SEED, NUM_SEEDS)

    print("\n" + "=" * 100)
    print(f"{'K_max':>7} {'method':<10} {'success':>9} {'total_ms':>10} {'per_seed_ms':>12} {'eff_K':>9}")
    print("-" * 100)

    rows = []
    for k_max in K_MAX_VALUES:
        threshold = calibrate_threshold(ensemble, k_max, calibration_states)

        fixed = run_fixed_high_batched(ensemble, initial_states, k_max, EVAL_STEPS, seed=1000)
        adaptive = run_adaptive_sequential(ensemble, initial_states, k_max, threshold, EVAL_STEPS, seed=2000)

        for name, result in (("fixed_high", fixed), ("adaptive", adaptive)):
            print(
                f"{k_max:7d} {name:<10} {result['success']:9.3f} "
                f"{result['total_latency_ms']:10.2f} {result['per_seed_latency_ms']:12.3f} "
                f"{result['effective_rollouts']:9.1f}"
            )
        speedup = fixed["total_latency_ms"] / adaptive["total_latency_ms"]
        winner = "ADAPTIVE faster" if speedup > 1.0 else "fixed_high faster"
        print(f"{'':7} total-latency ratio fixed/adaptive = {speedup:.3f}x -> {winner}")
        print("-" * 100)
        rows.append((k_max, fixed, adaptive, speedup))

    print("\nSummary: total-latency speedup (fixed_high_ms / adaptive_ms) vs K_max")
    print(f"{'K_max':>7} {'speedup':>9} {'fixed_success':>14} {'adaptive_success':>17}")
    for k_max, fixed, adaptive, speedup in rows:
        print(f"{k_max:7d} {speedup:9.3f} {fixed['success']:14.3f} {adaptive['success']:17.3f}")
