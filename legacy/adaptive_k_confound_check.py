"""Controlled follow-up to adaptive_k_crossover_study.py.

That study found adaptive-K's success rate consistently HIGHER than
fixed-high's across every K_max, which is not mechanistically expected
(adaptive uses fewer rollouts on average, so should do no better than
fixed-high at matched K). The two conditions used different RNG paths:
fixed-high via SeedBatchedGenericMPPI (one shared advancing stream across
all seeds), adaptive via a sequential loop re-seeding torch.manual_seed
per seed. This script removes that confound by running BOTH conditions
sequentially with identical per-seed seeding, to check whether the success
gap survives.
"""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, GenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, U_MAX, U_MIN, angle_error, collect_data, pendulum_cost,
    pendulum_step, train_models,
)
from adaptive_k_crossover_study import (
    HORIZON, K_MIN, PILOT_HORIZON, PILOT_ROLLOUTS, calibrate_threshold,
    make_initial_states,
)

K_MAX_VALUES = (100, 1000, 8000)
NUM_SEEDS = 20
EVAL_STEPS = 30


def run_fixed_high_sequential(ensemble, initial_states, k_max, steps, seed):
    """Same per-seed re-seeding scheme as run_adaptive_sequential, so the
    only difference between this and adaptive is the rollout-count policy."""
    num_seeds = len(initial_states)
    successes = []
    for s in range(num_seeds):
        np.random.seed(seed + s)
        torch.manual_seed(seed + s)
        controller = GenericMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON, rollouts=k_max,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
            cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
            device=DEVICE,
        )
        controller.reset()
        x = np.asarray(initial_states[s], dtype=np.float32)
        for _ in range(steps):
            action = controller.command(x)
            u = float(np.asarray(action).reshape(-1)[0])
            x = pendulum_step(x, u)
        theta_err = abs(angle_error(x[0]))
        omega_err = abs(x[1])
        successes.append(theta_err < 0.1 and omega_err < 0.5)
    return float(np.mean(successes))


def run_adaptive_sequential_success_only(ensemble, initial_states, k_max, threshold, steps, seed):
    num_seeds = len(initial_states)
    successes = []
    for s in range(num_seeds):
        np.random.seed(seed + s)
        torch.manual_seed(seed + s)
        controller = AdaptiveGenericMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
            cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
            min_rollouts=K_MIN, max_rollouts=k_max, uncertainty_threshold=threshold,
            pilot_rollouts=PILOT_ROLLOUTS, pilot_horizon=PILOT_HORIZON, device=DEVICE,
        )
        controller.reset()
        x = np.asarray(initial_states[s], dtype=np.float32)
        for _ in range(steps):
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
    initial_states = make_initial_states(800, NUM_SEEDS)

    print(f"\n{'K_max':>7} {'fixed_high (seq, matched RNG)':>30} {'adaptive':>10}  {'gap':>6}")
    for k_max in K_MAX_VALUES:
        threshold = calibrate_threshold(ensemble, k_max, calibration_states)
        # Same seed base for both conditions: each seed index s starts both
        # controllers from an identical torch RNG state (torch.manual_seed
        # (1000+s)) -- streams necessarily diverge after step 1 since
        # different K consumes a different amount of randomness, but the
        # confound under test (different seed *offsets*/RNG *paths* between
        # conditions) is removed.
        fixed_success = run_fixed_high_sequential(ensemble, initial_states, k_max, EVAL_STEPS, seed=1000)
        adaptive_success = run_adaptive_sequential_success_only(
            ensemble, initial_states, k_max, threshold, EVAL_STEPS, seed=1000
        )
        print(f"{k_max:7d} {fixed_success:30.3f} {adaptive_success:10.3f}  {adaptive_success - fixed_success:+.3f}")
