"""Re-derive the adaptive-K crossover's quality comparison (originally
Table confound in research_status.tex / paper.tex) with exploration noise
retuned per K, correcting the confound this project's own K-sweeps uncovered
(a fixed noise_sigma=1.0 across every K_max sweep, an increasingly severe
under-exploration handicap as K grows -- see the "more compute hurts,
resolved" finding).

Best noise_sigma per K_max was found via a preliminary sweep (nominal
condition, n=40): K=100 -> 1.0, K=1000 -> 5.0, K=8000 -> 3.0. This script
reuses those values (not re-swept here) to re-run the ORIGINAL confound
check's exact protocol (matched RNG, sequential controllers, same seed base,
K_MIN=100) at each K_max, comparing the original (uniform sigma=1.0) result
against the retuned result.

fixed-high uses its own best sigma. adaptive uses min_noise_sigma=1.0
(K_min=100's own best) and max_noise_sigma=K_max's best, via the new
AdaptiveGenericMPPI(min_noise_sigma=..., max_noise_sigma=...) support --
the pilot itself keeps noise_sigma=1.0 fixed so the calibrated threshold's
meaning is unchanged.
"""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, GenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, U_MIN, U_MAX, angle_error,
    collect_data, pendulum_cost, pendulum_step, train_models,
)
from adaptive_k_crossover_study import K_MIN, NUM_SEEDS, EVAL_STEPS, make_initial_states

BEST_SIGMA = {100: 1.0, 1000: 5.0, 8000: 3.0}


def make_fixed(ensemble, k, noise_sigma):
    return GenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON, rollouts=k,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
        cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=noise_sigma, device=DEVICE,
    )


def make_adaptive(ensemble, k_max, threshold, min_noise_sigma, max_noise_sigma):
    return AdaptiveGenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
        cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=K_MIN, max_rollouts=k_max, uncertainty_threshold=threshold,
        pilot_rollouts=25, pilot_horizon=5,
        min_noise_sigma=min_noise_sigma, max_noise_sigma=max_noise_sigma,
        device=DEVICE,
    )


def run_success(make_controller, states, seed):
    successes = []
    for i, x0 in enumerate(states):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        controller = make_controller()
        controller.reset()
        x = x0.copy()
        for _ in range(EVAL_STEPS):
            action = controller.command(x)
            u = float(np.asarray(action).reshape(-1)[0])
            x = pendulum_step(x, u)
        theta_err = abs(angle_error(x[0]))
        omega_err = abs(x[1])
        successes.append(theta_err < 0.1 and omega_err < 0.5)
    return float(np.mean(successes))


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    _, ensemble = train_models(collect_data())

    calibration_states = make_initial_states(900, 40)
    states = make_initial_states(800, NUM_SEEDS)

    print(f"{'K_max':>7} {'method':<12} {'sigma=1.0 (orig)':>18} {'sigma retuned':>15} {'retuned sigma used':>20}")
    print("-" * 80)
    for k_max in (100, 1000, 8000):
        best_sigma = BEST_SIGMA[k_max]

        # Original (uniform sigma=1.0)
        calibrator_orig = make_adaptive(ensemble, k_max, threshold=0.0, min_noise_sigma=1.0, max_noise_sigma=1.0)
        threshold_orig = calibrator_orig.calibrate_threshold(calibration_states)
        fixed_orig = run_success(lambda: make_fixed(ensemble, k_max, 1.0), states, seed=1000)
        adaptive_orig = run_success(
            lambda: make_adaptive(ensemble, k_max, threshold_orig, 1.0, 1.0), states, seed=1000
        )

        # Retuned
        calibrator_retuned = make_adaptive(ensemble, k_max, threshold=0.0, min_noise_sigma=1.0, max_noise_sigma=best_sigma)
        threshold_retuned = calibrator_retuned.calibrate_threshold(calibration_states)
        fixed_retuned = run_success(lambda: make_fixed(ensemble, k_max, best_sigma), states, seed=1000)
        adaptive_retuned = run_success(
            lambda: make_adaptive(ensemble, k_max, threshold_retuned, 1.0, best_sigma), states, seed=1000
        )

        print(f"{k_max:7d} {'fixed-high':<12} {100*fixed_orig:17.1f}% {100*fixed_retuned:14.1f}% {best_sigma:20.1f}")
        print(f"{k_max:7d} {'adaptive':<12} {100*adaptive_orig:17.1f}% {100*adaptive_retuned:14.1f}% {best_sigma:20.1f}")
        print("-" * 80)
