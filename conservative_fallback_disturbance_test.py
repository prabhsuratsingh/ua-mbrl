"""Follow-up to conservative_fallback_experiment.py.

That experiment found the fallback never triggers under dynamics-parameter
mismatch (mass x3, gravity -60%) -- diagnosed directly: pilot uncertainty
under that mismatch is statistically indistinguishable from nominal, because
the pendulum still visits the same (theta, omega, action) input region the
ensemble was trained on; only the TRUE mapping changed, and all ensemble
members, trained on the same nominal data, extrapolate confidently and
identically wrong rather than disagreeing. Ensemble disagreement is an
epistemic-coverage signal, not a "the physics changed" signal.

This script tests the threat model the OOD gate was actually calibrated
for (ood_severity_sweep.py): genuine state/action-space OOD. A large
one-time external disturbance kicks omega well outside the training range
(training omega range was [-2,2]; the severity sweep's "severe" shift used
omega in [4,6]) partway through an otherwise-nominal trajectory, and we check
whether the fallback (a) actually triggers, and (b) reduces the worst-case
velocity excursion afterward compared to running adaptive control alone.
"""

import numpy as np
import torch

from conservative_fallback_experiment import (
    FALLBACK_GAIN, make_calibrator, make_states, FallbackAdaptiveMPPI,
)
from generic_mbrl import AdaptiveGenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, ROLLOUTS, U_MIN, U_MAX, angle_error,
    collect_data, pendulum_cost, pendulum_step, train_models,
)

EVAL_STEPS = 40
DISTURBANCE_STEP = 10
DISTURBANCE_OMEGA = 5.0  # matches ood_severity_sweep.py's "severe" omega range (4-6)
NUM_SEEDS = 15


def run_episode_with_disturbance(controller, x0, seed, track_fallback=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    max_abs_omega_after_disturbance = 0.0
    fallback_steps = 0
    for step in range(EVAL_STEPS):
        if step == DISTURBANCE_STEP:
            x[1] = DISTURBANCE_OMEGA  # inject genuine state-space OOD
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        if track_fallback and getattr(controller, "fallback_triggered_last", False):
            fallback_steps += 1
        x = pendulum_step(x, u)
        if step >= DISTURBANCE_STEP:
            max_abs_omega_after_disturbance = max(max_abs_omega_after_disturbance, abs(x[1]))
    theta_err = abs(angle_error(x[0]))
    omega_err = abs(x[1])
    return {
        "success": theta_err < 0.1 and omega_err < 0.5,
        "max_abs_omega_after": max_abs_omega_after_disturbance,
        "final_abs_omega": abs(x[1]),
        "fallback_fraction": fallback_steps / EVAL_STEPS,
    }


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    _, ensemble = train_models(collect_data())

    calibrator = make_calibrator(ensemble)
    calibration_states = make_states(900, 40)
    mild_threshold = calibrator.calibrate_threshold(calibration_states)
    severe_threshold = mild_threshold * 3.5

    states = make_states(800, NUM_SEEDS)

    def make_adaptive():
        return AdaptiveGenericMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
            cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
            min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS,
            uncertainty_threshold=mild_threshold,
            pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
        )

    def make_fallback():
        return FallbackAdaptiveMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
            cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
            min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS,
            uncertainty_threshold=mild_threshold, severe_threshold=severe_threshold,
            pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
        )

    print(f"Disturbance: omega set to {DISTURBANCE_OMEGA} at step {DISTURBANCE_STEP} (training range was [-2,2])")
    print(f"mild={mild_threshold:.6f} severe={severe_threshold:.6f}\n")

    for name, factory in (("adaptive (no fallback)", make_adaptive), ("+ fallback", make_fallback)):
        results = [
            run_episode_with_disturbance(factory(), x0, 1000 + i, track_fallback=True)
            for i, x0 in enumerate(states)
        ]
        success = np.mean([r["success"] for r in results])
        max_omega_mean = np.mean([r["max_abs_omega_after"] for r in results])
        max_omega_worst = np.max([r["max_abs_omega_after"] for r in results])
        fallback_frac = np.mean([r["fallback_fraction"] for r in results])
        print(
            f"{name:<24} success={success:.3f}  "
            f"mean|omega|max_after_disturbance={max_omega_mean:.3f}  "
            f"worst={max_omega_worst:.3f}  "
            f"fallback_triggered={100*fallback_frac:.1f}% of steps"
        )
