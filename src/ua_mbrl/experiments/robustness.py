"""Learned-model MPPI under mass / gravity mismatch, all models trained on
nominal dynamics (robustness_mismatch_experiment.py's main)."""

import numpy as np
import torch

from ua_mbrl.core.mppi import AdaptiveGenericMPPI
from ua_mbrl.envs import pendulum as pend

EVAL_EPISODES = 3


def add_arguments(parser):
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)


def run_episode(controller, x0, true_params, seed, eval_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    errors = []
    uncertainties = []
    rollout_counts = []

    for _ in range(eval_steps):
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        effective_rollouts = getattr(controller, "last_rollouts", controller.rollouts)
        if isinstance(controller, AdaptiveGenericMPPI):
            effective_rollouts += controller.pilot_rollouts * controller.pilot_horizon
        rollout_counts.append(effective_rollouts)
        if hasattr(controller.dynamics, "model_step_with_uncertainty"):
            model_next, uncertainty = controller.dynamics.model_step_with_uncertainty(x, u)
            uncertainties.append(uncertainty)
        else:
            model_next = controller.dynamics.model_step(x, u)
            uncertainties.append(0.0)

        true_next = pend.pendulum_step(x, u, **true_params)
        error = model_next - true_next
        error[0] = pend.angle_error(model_next[0], true_next[0])
        errors.append(error)
        x = true_next

    return {
        "success": pend.is_success(x),
        "theta_error": abs(pend.angle_error(x[0])),
        "omega_error": abs(x[1]),
        "model_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "mean_rollouts": float(np.mean(rollout_counts)),
    }


def evaluate(controller_factory, states, true_params, seed_offset, eval_steps):
    # The factory is called before run_episode seeds -- as the original did.
    results = [
        run_episode(controller_factory(), x0, true_params, seed_offset + episode, eval_steps)
        for episode, x0 in enumerate(states)
    ]
    return {key: float(np.mean([r[key_src] for r in results])) for key, key_src in (
        ("success_rate", "success"), ("theta_error", "theta_error"), ("omega_error", "omega_error"),
        ("model_rmse", "model_rmse"), ("uncertainty", "mean_uncertainty"), ("rollouts", "mean_rollouts"),
    )}


def run(args, run_dir):
    single, ensemble = pend.train_models(pend.collect_data())
    calibration_states = pend.make_states(101, args.episodes)
    states = pend.make_states(100, args.episodes)

    calibration_controller = pend.make_adaptive(ensemble, 3 * pend.ROLLOUTS, threshold=0.006)
    torch.manual_seed(101)  # no effect on the threshold (pilot generator already seeded); kept for parity
    threshold = calibration_controller.calibrate_threshold(calibration_states)
    print(f"Calibrated adaptive uncertainty threshold: {threshold:.6f}")

    conditions = {
        "nominal": {},
        "mass +20%": {"mass": 1.2},
        "gravity -15%": {"gravity": 9.81 * 0.85},
    }
    methods = {
        "nominal analytic": lambda: pend.make_fixed(pend.AnalyticalNominalModel()),
        "single learned": lambda: pend.make_fixed(single),
        "ensemble": lambda: pend.make_fixed(ensemble),
        "ensemble K=300": lambda: pend.make_fixed(ensemble, rollouts=3 * pend.ROLLOUTS),
        "ensemble beta=0.1": lambda: pend.make_fixed(ensemble, beta=0.1),
        "adaptive ensemble": lambda: pend.make_adaptive(ensemble, 3 * pend.ROLLOUTS, threshold=threshold),
    }
    for condition, true_params in conditions.items():
        for method, factory in methods.items():
            result = evaluate(factory, states, true_params, 1000, args.eval_steps)
            run_dir.append({"condition": condition, "method": method, **result})
            print(
                f"{condition:<18} {method:<18} "
                f"success={100 * result['success_rate']:5.1f}% "
                f"theta={result['theta_error']:.4f} omega={result['omega_error']:.4f} "
                f"model_rmse={result['model_rmse']:.4f} unc={result['uncertainty']:.5f} "
                f"K={result['rollouts']:.1f}"
            )
