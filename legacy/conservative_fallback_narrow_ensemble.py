"""Conservative fallback, retried against the SAME ensemble/training recipe
that ood_severity_sweep.py actually validated.

conservative_fallback_disturbance_test.py found the fallback never triggers,
even under a genuine state-space disturbance (omega=5, matching the severity
sweep's "severe" range) -- diagnosed directly: single-step ensemble
disagreement for the robustness_mismatch_experiment ensemble (broader
training-data collection) *decreases* with omega magnitude, the opposite of
the monotonic increase found for ood_uncertainty_experiment's narrower-trained
ensemble. The dose-response curve validated in ood_severity_sweep.py is a
property of THAT specific ensemble/training recipe, not of "ensemble
disagreement" in general -- so the fallback needs to be built on the ensemble
whose behavior was actually characterized, not a differently-trained one.

This script trains via ood_uncertainty_experiment.collect_narrow_data (the
same recipe as the validated severity curve), builds MPPI/adaptive/fallback
controllers around it, and re-runs the state-space-disturbance test.
"""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI
from ood_uncertainty_experiment import collect_narrow_data, train_models as train_narrow_models
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, ROLLOUTS, U_MIN, U_MAX, angle_error, pendulum_cost, pendulum_step,
)
from conservative_fallback_experiment import FallbackAdaptiveMPPI, make_states

EVAL_STEPS = 40
DISTURBANCE_STEP = 10
DISTURBANCE_OMEGA = 5.0
NUM_SEEDS = 15
SEVERE_MULTIPLIER = 3.5


def run_episode_with_disturbance(controller, x0, seed, track_fallback=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    max_abs_omega_after = 0.0
    fallback_steps = 0
    for step in range(EVAL_STEPS):
        if step == DISTURBANCE_STEP:
            x[1] = DISTURBANCE_OMEGA
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        if track_fallback and getattr(controller, "fallback_triggered_last", False):
            fallback_steps += 1
        x = pendulum_step(x, u)
        if step >= DISTURBANCE_STEP:
            max_abs_omega_after = max(max_abs_omega_after, abs(x[1]))
    theta_err = abs(angle_error(x[0]))
    omega_err = abs(x[1])
    return {
        "success": theta_err < 0.1 and omega_err < 0.5,
        "max_abs_omega_after": max_abs_omega_after,
        "fallback_fraction": fallback_steps / EVAL_STEPS,
    }


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    narrow_data = collect_narrow_data(seed=0)
    _, ensemble = train_narrow_models(narrow_data)

    # Single-step disagreement sanity check first: does this ensemble show
    # the expected monotonic increase with omega magnitude?
    print("\nSingle-step disagreement vs omega (sanity check):")
    for omega in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        n = 100
        theta = np.random.uniform(-np.pi, np.pi, n).astype(np.float32)
        om = np.full(n, omega, dtype=np.float32)
        u = np.random.uniform(-2.0, 2.0, n).astype(np.float32)
        x = torch.as_tensor(np.stack([theta, om], axis=1), device=DEVICE)
        uu = torch.as_tensor(u.reshape(-1, 1), device=DEVICE)
        with torch.no_grad():
            _, var, _ = ensemble.forward_mean_var(x, uu)
            unc = var.sum(dim=1)
        print(f"  omega={omega:4.1f}  mean_uncertainty={unc.mean().item():.6f}")

    calibrator = AdaptiveGenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0], cost_fn=pendulum_cost,
        dt=DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS, uncertainty_threshold=0.0,
        pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
    )
    calibration_states = make_states(900, 40)
    # calibration states should match the narrow ensemble's training range
    calibration_states = np.clip(calibration_states, [-np.pi, -1.0], [np.pi, 1.0]).astype(np.float32)
    mild_threshold = calibrator.calibrate_threshold(calibration_states)
    severe_threshold = mild_threshold * SEVERE_MULTIPLIER
    print(f"\nmild={mild_threshold:.6f} severe={severe_threshold:.6f}")

    states = make_states(800, NUM_SEEDS)
    states = np.clip(states, [-np.pi, -1.0], [np.pi, 1.0]).astype(np.float32)

    def make_adaptive():
        return AdaptiveGenericMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0], cost_fn=pendulum_cost,
            dt=DT, temperature=1.0, noise_sigma=1.0,
            min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS, uncertainty_threshold=mild_threshold,
            pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
        )

    def make_fallback():
        return FallbackAdaptiveMPPI(
            dynamics=ensemble, action_dim=1, horizon=HORIZON,
            action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0], cost_fn=pendulum_cost,
            dt=DT, temperature=1.0, noise_sigma=1.0,
            min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS, uncertainty_threshold=mild_threshold,
            severe_threshold=severe_threshold, pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
        )

    print(f"\nDisturbance: omega set to {DISTURBANCE_OMEGA} at step {DISTURBANCE_STEP} (narrow training range was [-1,1])\n")
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
            f"mean|omega|max_after={max_omega_mean:.3f}  worst={max_omega_worst:.3f}  "
            f"fallback_triggered={100*fallback_frac:.1f}% of steps"
        )
