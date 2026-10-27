"""Conservative fallback controller, gated by a *severe*-OOD threshold.

research_status.tex's next-steps list said: "Add a conservative fallback
controller only after the OOD gate is validated; otherwise the gate should
increase compute without claiming safety guarantees." The continuous OOD
severity sweep (ood_severity_sweep.py) validated exactly that: detection is
near chance below shift~1.5, but reliable (AUC>=0.98) at shift>=3.5, where the
uncertainty ratio versus in-distribution was consistently ~3-4x. This script
uses that empirical ratio to set a second, higher "severe" threshold (on top
of the existing "mild" compute-gating threshold), and physically-motivated
fallback action -- when severe OOD is flagged, stop trusting the learned
ensemble's action and apply a model-free velocity-damping "brake" instead of
letting the (possibly-wrong-under-severe-mismatch) MPPI plan drive the system.

This trades task success for a bounded-motion safety property under genuinely
severe test-time model mismatch, which is the property the existing adaptive
controller (only gates *compute*, not *trust*) does not have.
"""

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, NOMINAL_G, ROLLOUTS, U_MAX, U_MIN,
    angle_error, collect_data, pendulum_cost, pendulum_step, train_models,
)

SEVERE_MULTIPLIER = 3.5  # grounded in ood_severity_sweep.py's observed ~3-4x
                          # uncertainty ratio at reliably-detected (AUC>=0.98) shift
FALLBACK_GAIN = 2.0
EVAL_STEPS = 40
NUM_SEEDS = 15
CALIBRATION_SEED = 900


class FallbackAdaptiveMPPI(AdaptiveGenericMPPI):
    """AdaptiveGenericMPPI plus a second, higher threshold that overrides the
    chosen action with a model-free damping law instead of trusting the
    ensemble's plan."""

    def __init__(self, *args, severe_threshold=None, fallback_gain=FALLBACK_GAIN, **kwargs):
        super().__init__(*args, **kwargs)
        self.severe_threshold = severe_threshold
        self.fallback_gain = fallback_gain
        self.fallback_triggered_last = False

    @torch.no_grad()
    def command(self, x0):
        action = super().command(x0)
        if self.severe_threshold is not None and self.last_pilot_uncertainty >= self.severe_threshold:
            self.fallback_triggered_last = True
            state = np.asarray(x0, dtype=np.float32).reshape(-1)
            omega = float(state[1])
            fallback_u = float(
                np.clip(-self.fallback_gain * omega, U_MIN, U_MAX)
            )
            return np.asarray([fallback_u], dtype=np.float32)
        self.fallback_triggered_last = False
        return action


def make_states(seed, count):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [[rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)] for _ in range(count)],
        dtype=np.float32,
    )


def make_calibrator(ensemble):
    return AdaptiveGenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
        cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=ROLLOUTS, max_rollouts=3 * ROLLOUTS, uncertainty_threshold=0.0,
        pilot_rollouts=25, pilot_horizon=5, device=DEVICE,
    )


def run_episode(controller, x0, condition, seed, track_fallback=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    max_abs_omega = abs(x[1])
    fallback_steps = 0
    for _ in range(EVAL_STEPS):
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        if track_fallback and getattr(controller, "fallback_triggered_last", False):
            fallback_steps += 1
        x = pendulum_step(x, u, **condition)
        max_abs_omega = max(max_abs_omega, abs(x[1]))
    theta_err = abs(angle_error(x[0]))
    omega_err = abs(x[1])
    return {
        "success": theta_err < 0.1 and omega_err < 0.5,
        "max_abs_omega": max_abs_omega,
        "fallback_fraction": fallback_steps / EVAL_STEPS,
    }


def evaluate(make_controller, states, condition, seed_base):
    results = [
        run_episode(make_controller(), state, condition, seed_base + i, track_fallback=True)
        for i, state in enumerate(states)
    ]
    return {
        "success": float(np.mean([r["success"] for r in results])),
        "max_abs_omega_mean": float(np.mean([r["max_abs_omega"] for r in results])),
        "max_abs_omega_worst": float(np.max([r["max_abs_omega"] for r in results])),
        "fallback_fraction": float(np.mean([r["fallback_fraction"] for r in results])),
    }


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    _, ensemble = train_models(collect_data())

    calibrator = make_calibrator(ensemble)
    calibration_states = make_states(CALIBRATION_SEED, 40)
    mild_threshold = calibrator.calibrate_threshold(calibration_states)
    severe_threshold = mild_threshold * SEVERE_MULTIPLIER
    print(f"mild (compute-gate) threshold : {mild_threshold:.6f}")
    print(f"severe (fallback) threshold   : {severe_threshold:.6f}  (x{SEVERE_MULTIPLIER})")

    states = make_states(800, NUM_SEEDS)

    conditions = {
        "nominal": {},
        "mild_mismatch (mass+20%)": {"mass": 1.2},
        "severe_mismatch (mass x3, gravity-60%)": {"mass": 3.0, "gravity": NOMINAL_G * 0.4},
    }

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

    print(f"\n{'condition':<40} {'method':<10} {'success':>8} {'mean|omega|max':>15} {'worst|omega|max':>16} {'fallback%':>10}")
    print("-" * 105)
    for condition_name, condition in conditions.items():
        adaptive_result = evaluate(make_adaptive, states, condition, seed_base=1000)
        fallback_result = evaluate(make_fallback, states, condition, seed_base=1000)
        for name, result in (("adaptive (no fallback)", adaptive_result), ("+ fallback", fallback_result)):
            print(
                f"{condition_name:<40} {name:<10} {result['success']:8.3f} "
                f"{result['max_abs_omega_mean']:15.3f} {result['max_abs_omega_worst']:16.3f} "
                f"{100*result['fallback_fraction']:9.1f}%"
            )
        print("-" * 105)
