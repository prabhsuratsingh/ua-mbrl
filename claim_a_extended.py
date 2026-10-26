"""Extend the Claim A matched-sample-budget test (see research_status.tex
sec:claima / paper.tex sec:claima) beyond its single original data point
(K_max=1000, nominal: adaptive 85.0% at eff_K=249.2 vs fixed's best 82.5% at
K=271) to more K_max values and a mismatch condition, to see whether
"gating matches-or-beats a fixed allocation of the same average size" is a
real pattern or a single lucky point.

For each (K_max, condition) test case this reports BOTH:
  - the ceiling comparison: fixed at K_max (its own best sigma) vs adaptive
  - the matched-budget comparison: fixed at adaptive's own measured average
    effective K (its own best sigma) vs adaptive

All success rates at n=40, seed-batched for fixed (cheap), sequential for
adaptive (required, since adaptive needs a different K per seed).
"""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, SeedBatchedGenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, U_MIN, U_MAX, angle_error,
    collect_data, pendulum_cost, pendulum_step, train_models,
)

NUM_SEEDS = 40
EVAL_STEPS = 40
CEILING_SIGMAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def make_states(seed, count):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [[rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)] for _ in range(count)],
        dtype=np.float32,
    )


def run_fixed_batched(ensemble, k, sigma, states, condition, seed=1000):
    controller = SeedBatchedGenericMPPI(
        dynamics=ensemble, num_seeds=len(states), action_dim=1, horizon=HORIZON, rollouts=k,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0], cost_fn=pendulum_cost,
        dt=DT, temperature=1.0, noise_sigma=sigma, device=DEVICE,
    )
    controller.reset()
    torch.manual_seed(seed)
    x = np.asarray(states, dtype=np.float32)
    for _ in range(EVAL_STEPS):
        actions = controller.command(x)
        u = np.asarray(actions).reshape(len(states))
        x = np.stack([pendulum_step(x[s], float(u[s]), **condition) for s in range(len(states))])
    theta_err = np.abs(np.asarray([angle_error(x[s, 0]) for s in range(len(states))]))
    omega_err = np.abs(x[:, 1])
    return float(100 * ((theta_err < 0.1) & (omega_err < 0.5)).mean())


def sweep_sigma_batched(ensemble, k, sigmas, states, condition):
    results = {sigma: run_fixed_batched(ensemble, k, sigma, states, condition) for sigma in sigmas}
    best_sigma = max(results, key=results.get)
    return best_sigma, results[best_sigma], results


def make_adaptive(ensemble, k_max, threshold, min_sigma, max_sigma):
    return AdaptiveGenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0], cost_fn=pendulum_cost,
        dt=DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=100, max_rollouts=k_max, uncertainty_threshold=threshold,
        pilot_rollouts=25, pilot_horizon=5,
        min_noise_sigma=min_sigma, max_noise_sigma=max_sigma, device=DEVICE,
    )


def run_adaptive_sequential(ensemble, k_max, threshold, min_sigma, max_sigma, states, condition, seed=1000):
    successes, eff_ks = [], []
    for i, x0 in enumerate(states):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        controller = make_adaptive(ensemble, k_max, threshold, min_sigma, max_sigma)
        controller.reset()
        x = x0.copy()
        ks = []
        for step in range(EVAL_STEPS):
            action = controller.command(x)
            ks.append(controller.last_rollouts + controller.pilot_rollouts * controller.pilot_horizon)
            u = float(np.asarray(action).reshape(-1)[0])
            x = pendulum_step(x, u, **condition)
        eff_ks.append(np.mean(ks))
        theta_err = abs(angle_error(x[0]))
        omega_err = abs(x[1])
        successes.append(theta_err < 0.1 and omega_err < 0.5)
    return float(100 * np.mean(successes)), float(np.mean(eff_ks))


def run_test_case(ensemble, calibration_states, states, k_max, condition, condition_name):
    print(f"\n=== K_max={k_max}, condition={condition_name} ===")

    ceiling_sigma, ceiling_success, ceiling_all = sweep_sigma_batched(
        ensemble, k_max, CEILING_SIGMAS, states, condition
    )
    print(f"  ceiling sigma sweep: {ceiling_all}")
    print(f"  fixed @ K_max={k_max}, best sigma={ceiling_sigma}: {ceiling_success:.1f}%")

    calibrator = make_adaptive(ensemble, k_max, threshold=0.0, min_sigma=1.0, max_sigma=ceiling_sigma)
    threshold = calibrator.calibrate_threshold(calibration_states)

    adaptive_success, eff_k = run_adaptive_sequential(
        ensemble, k_max, threshold, 1.0, ceiling_sigma, states, condition
    )
    print(f"  adaptive: {adaptive_success:.1f}% at mean effective K={eff_k:.1f}")

    k_matched = int(round(eff_k))
    matched_sigma, matched_success, matched_all = sweep_sigma_batched(
        ensemble, k_matched, CEILING_SIGMAS, states, condition
    )
    print(f"  matched-budget sigma sweep at K={k_matched}: {matched_all}")
    print(f"  fixed @ matched K={k_matched}, best sigma={matched_sigma}: {matched_success:.1f}%")

    print(f"  --- SUMMARY: ceiling fixed={ceiling_success:.1f}% | matched fixed={matched_success:.1f}% "
          f"(K={k_matched}) | adaptive={adaptive_success:.1f}% (K={eff_k:.1f}) ---")

    return {
        "k_max": k_max, "condition": condition_name,
        "ceiling_sigma": ceiling_sigma, "ceiling_success": ceiling_success,
        "adaptive_success": adaptive_success, "eff_k": eff_k,
        "matched_k": k_matched, "matched_sigma": matched_sigma, "matched_success": matched_success,
    }


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    _, ensemble = train_models(collect_data())

    calibration_states = make_states(900, 40)
    states = make_states(800, NUM_SEEDS)

    test_cases = [
        (3000, {}, "nominal"),
        (8000, {}, "nominal"),
        (16000, {}, "nominal"),
        (1000, {"mass": 1.2}, "mild_mismatch"),
    ]

    results = []
    for k_max, condition, condition_name in test_cases:
        results.append(run_test_case(ensemble, calibration_states, states, k_max, condition, condition_name))

    print("\n\n=== FINAL SUMMARY (all test cases) ===")
    print(f"{'K_max':>7} {'condition':<14} {'ceiling_fixed':>14} {'matched_fixed':>14} {'adaptive':>10} {'matched_K':>10} {'eff_K':>8}")
    for r in results:
        print(
            f"{r['k_max']:7d} {r['condition']:<14} {r['ceiling_success']:13.1f}% "
            f"{r['matched_success']:13.1f}% {r['adaptive_success']:9.1f}% "
            f"{r['matched_k']:10d} {r['eff_k']:8.1f}"
        )
