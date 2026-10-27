"""Paired significance test for Claim A's matched-sample-budget pendulum
sweep (claim_a_extended.py). That script only ever logged the aggregate
success percentage per (K_max, condition), even though fixed and adaptive
are evaluated on the SAME 40 seeded initial states each time -- so an
independent-sample interval (Wilson) on the aggregates is the wrong test
and, as found during the CI-methodology review, overlaps at every row.

This script re-runs the identical protocol (same seeds, same ensemble,
same sigma sweeps -- verified to reproduce claim_a_extended.py's published
aggregate percentages exactly) but additionally records the per-seed
pass/fail outcome for BOTH the matched-budget fixed controller and the
adaptive controller, so the actual pairing can be exploited: a paired
sign test / exact McNemar's test on the seed-level win/tie/loss record,
directly analogous to the record already computed for the MuJoCo
replication (mujoco_claim_a.py).
"""

import math
from itertools import combinations

import numpy as np
import torch

from claim_a_extended import (
    CEILING_SIGMAS, DEVICE, EVAL_STEPS, NUM_SEEDS,
    make_adaptive, make_states,
)
from generic_mbrl import SeedBatchedGenericMPPI
from robustness_mismatch_experiment import (
    DT, HORIZON, U_MIN, U_MAX, angle_error,
    collect_data, pendulum_cost, pendulum_step, train_models,
)


def run_fixed_batched_paired(ensemble, k, sigma, states, condition, seed=1000):
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
    per_seed = (theta_err < 0.1) & (omega_err < 0.5)
    return float(100 * per_seed.mean()), per_seed


def sweep_sigma_batched_paired(ensemble, k, sigmas, states, condition):
    results = {sigma: run_fixed_batched_paired(ensemble, k, sigma, states, condition) for sigma in sigmas}
    best_sigma = max(results, key=lambda s: results[s][0])
    rate, per_seed = results[best_sigma]
    return best_sigma, rate, per_seed


def run_adaptive_sequential_paired(ensemble, k_max, threshold, min_sigma, max_sigma, states, condition, seed=1000):
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
    per_seed = np.asarray(successes, dtype=bool)
    return float(100 * per_seed.mean()), float(np.mean(eff_ks)), per_seed


def exact_sign_test(fixed_wins, adaptive_wins):
    """Two-sided exact sign test (equivalently, exact McNemar's test) on
    discordant pairs: under H0 (fixed and adaptive equally likely to win a
    discordant pair), wins ~ Binomial(n_discordant, 0.5)."""
    n = fixed_wins + adaptive_wins
    if n == 0:
        return float("nan")
    k = min(fixed_wins, adaptive_wins)
    # P(X <= k) doubled, capped at 1.0 (standard exact two-sided binomial test)
    p_le_k = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p_le_k)


def run_test_case(ensemble, calibration_states, states, k_max, condition, condition_name):
    print(f"\n=== K_max={k_max}, condition={condition_name} ===")

    ceiling_sigma, ceiling_success, _ = sweep_sigma_batched_paired(
        ensemble, k_max, CEILING_SIGMAS, states, condition
    )
    print(f"  fixed @ K_max={k_max}, best sigma={ceiling_sigma}: {ceiling_success:.1f}%")

    calibrator = make_adaptive(ensemble, k_max, threshold=0.0, min_sigma=1.0, max_sigma=ceiling_sigma)
    threshold = calibrator.calibrate_threshold(calibration_states)

    adaptive_success, eff_k, adaptive_per_seed = run_adaptive_sequential_paired(
        ensemble, k_max, threshold, 1.0, ceiling_sigma, states, condition
    )
    print(f"  adaptive: {adaptive_success:.1f}% at mean effective K={eff_k:.1f}")

    k_matched = int(round(eff_k))
    matched_sigma, matched_success, matched_per_seed = sweep_sigma_batched_paired(
        ensemble, k_matched, CEILING_SIGMAS, states, condition
    )
    print(f"  fixed @ matched K={k_matched}, best sigma={matched_sigma}: {matched_success:.1f}%")

    wins = int(np.sum(matched_per_seed & ~adaptive_per_seed))
    losses = int(np.sum(~matched_per_seed & adaptive_per_seed))
    ties = int(np.sum(matched_per_seed == adaptive_per_seed))
    p_value = exact_sign_test(wins, losses)

    print(f"  --- PAIRED: fixed wins={wins}, adaptive wins={losses}, ties={ties}, "
          f"exact sign-test p={p_value:.4f} ---")
    print(f"  --- SUMMARY: matched fixed={matched_success:.1f}% (K={k_matched}) | "
          f"adaptive={adaptive_success:.1f}% (K={eff_k:.1f}) ---")

    return {
        "k_max": k_max, "condition": condition_name,
        "adaptive_success": adaptive_success, "eff_k": eff_k,
        "matched_k": k_matched, "matched_sigma": matched_sigma, "matched_success": matched_success,
        "fixed_wins": wins, "adaptive_wins": losses, "ties": ties, "sign_test_p": p_value,
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

    print("\n\n=== FINAL SUMMARY: paired win/tie/loss + exact sign test ===")
    print(f"{'K_max':>7} {'condition':<14} {'matched_fixed':>14} {'adaptive':>10} "
          f"{'fixed_w':>8} {'adapt_w':>8} {'ties':>5} {'p_value':>9}")
    for r in results:
        print(
            f"{r['k_max']:7d} {r['condition']:<14} {r['matched_success']:13.1f}% "
            f"{r['adaptive_success']:9.1f}% {r['fixed_wins']:8d} {r['adaptive_wins']:8d} "
            f"{r['ties']:5d} {r['sign_test_p']:9.4f}"
        )
