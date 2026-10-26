"""Claim A (statistics): at matched sample budget, does uncertainty-gated K_t
behave like a better estimator, and does its extra compute land where it
actually helps -- not just "does it use fewer rollouts on average"?

This directly tests the mechanism the adaptive-K crossover study (latency)
and the confound check (aggregate success rate) did not: whether the gate's
own uncertainty signal correlates with states where extra rollouts buy more
effective samples, and whether gated-K preserves task performance in harder
conditions relative to controllers matched on average budget.

Two parts, kept separate because naively comparing per-step diagnostics
across independently-run closed-loop episodes is invalid once trajectories
diverge (different controllers choose different actions -> different next
states):

Part 1 -- ESS-vs-uncertainty mechanism check, along ONE reference trajectory
per condition (states are fixed; we ask "what would K_min/K_max each achieve
here", not "what did some other run's step 7 happen to look like").

Part 2 -- episode-level, matched-RNG (same seed base for every controller,
per the earlier RNG-confound lesson) success-rate comparison, stratified by
KNOWN ground-truth condition difficulty (not by the model's own signal, to
avoid comparing controllers across trajectories that have already diverged).
"""

import copy

import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, GenericMPPI
from robustness_mismatch_experiment import (
    DEVICE, DT, HORIZON, NOMINAL_G, ROLLOUTS, U_MIN, U_MAX,
    angle_error, collect_data, pendulum_cost, pendulum_step, train_models,
)

K_MIN = ROLLOUTS          # 100
K_MAX = 3 * ROLLOUTS      # 300
PILOT_ROLLOUTS = 25
PILOT_HORIZON = 5
EVAL_STEPS = 40
NUM_SEEDS = 15
DISTURBANCE_STEP = 10
DISTURBANCE_OMEGA = 5.0

CONDITIONS = {
    "nominal": {},
    "mild_mismatch": {"mass": 1.2},
    "severe_mismatch": {"mass": 3.0, "gravity": NOMINAL_G * 0.4},
}


def make_states(seed, count):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [[rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)] for _ in range(count)],
        dtype=np.float32,
    )


def make_adaptive(ensemble, threshold):
    return AdaptiveGenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
        cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=K_MIN, max_rollouts=K_MAX, uncertainty_threshold=threshold,
        pilot_rollouts=PILOT_ROLLOUTS, pilot_horizon=PILOT_HORIZON, device=DEVICE,
    )


def make_fixed(ensemble, k):
    return GenericMPPI(
        dynamics=ensemble, action_dim=1, horizon=HORIZON, rollouts=k,
        action_low=[U_MIN], action_high=[U_MAX], target=[np.pi, 0.0],
        cost_fn=pendulum_cost, dt=DT, temperature=1.0, noise_sigma=1.0, device=DEVICE,
    )


def shadow_ess(ensemble, nominal_snapshot, x, k, seed):
    """ESS that a K-rollout MPPI call would achieve RIGHT NOW at state x,
    using the same nominal control sequence as the reference controller --
    a one-shot, throwaway evaluation that does not affect the reference
    trajectory."""
    torch.manual_seed(seed)
    shadow = make_fixed(ensemble, k)
    shadow.nominal = nominal_snapshot.clone()
    shadow.command(x)
    return shadow.last_ess


def run_reference_trajectory(ensemble, threshold, condition_name, condition, x0, seed, disturb):
    """One episode run by the adaptive controller; at each step, log the
    gate's own uncertainty/K choice plus shadow ESS(K_min) and ESS(K_max)
    at that exact state."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller = make_adaptive(ensemble, threshold)
    controller.reset()
    x = x0.copy()
    rows = []
    for step in range(EVAL_STEPS):
        if disturb and step == DISTURBANCE_STEP:
            x[1] = DISTURBANCE_OMEGA
        nominal_snapshot = controller.nominal.clone()
        action = controller.command(x)
        eta = controller.last_pilot_uncertainty
        k_used = controller.last_rollouts
        ess_min = shadow_ess(ensemble, nominal_snapshot, x, K_MIN, seed=90000 + step)
        ess_max = shadow_ess(ensemble, nominal_snapshot, x, K_MAX, seed=90000 + step)
        rows.append({
            "condition": condition_name, "step": step, "eta": eta,
            "k_used": k_used, "ess_min": ess_min, "ess_max": ess_max,
        })
        u = float(np.asarray(action).reshape(-1)[0])
        x = pendulum_step(x, u, **condition)
    return rows


def run_episode(make_controller, x0, condition, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller = make_controller()
    controller.reset()
    x = x0.copy()
    for step in range(EVAL_STEPS):
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        x = pendulum_step(x, u, **condition)
    theta_err = abs(angle_error(x[0]))
    omega_err = abs(x[1])
    return theta_err < 0.1 and omega_err < 0.5


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    _, ensemble = train_models(collect_data())

    calibrator = make_adaptive(ensemble, threshold=0.0)
    calibration_states = make_states(900, 40)
    threshold = calibrator.calibrate_threshold(calibration_states)
    print(f"mild threshold: {threshold:.6f}")

    # ---------------- Part 1: ESS-vs-uncertainty mechanism check ----------------
    print("\n=== Part 1: ESS vs pilot uncertainty (one reference trajectory per condition) ===")
    ref_states = make_states(700, 1)
    all_rows = []
    for condition_name, condition in CONDITIONS.items():
        rows = run_reference_trajectory(
            ensemble, threshold, condition_name, condition, ref_states[0], seed=500, disturb=False
        )
        all_rows.extend(rows)
    disturb_rows = run_reference_trajectory(
        ensemble, threshold, "state_disturbance", {}, ref_states[0], seed=500, disturb=True
    )
    all_rows.extend(disturb_rows)

    etas = np.array([r["eta"] for r in all_rows])
    ess_min = np.array([r["ess_min"] for r in all_rows])
    ess_max = np.array([r["ess_max"] for r in all_rows])
    k_used = np.array([r["k_used"] for r in all_rows])

    terciles = np.quantile(etas, [1 / 3, 2 / 3])
    bucket_names = ["low", "mid", "high"]
    bucket_idx = np.digitize(etas, terciles)

    print(f"{'bucket':>6} {'n':>4} {'eta_range':>18} {'ESS(K_min)':>12} {'ESS(K_max)':>12} {'gain(max-min)':>14} {'frac_K_max':>11}")
    for b in range(3):
        mask = bucket_idx == b
        if mask.sum() == 0:
            continue
        eta_lo, eta_hi = etas[mask].min(), etas[mask].max()
        gain = ess_max[mask].mean() - ess_min[mask].mean()
        frac_max = np.mean(k_used[mask] >= K_MAX)
        print(
            f"{bucket_names[b]:>6} {mask.sum():4d} [{eta_lo:.5f},{eta_hi:.5f}] "
            f"{ess_min[mask].mean():12.2f} {ess_max[mask].mean():12.2f} "
            f"{gain:14.2f} {frac_max:11.2f}"
        )

    # ---------------- Part 2: matched-RNG, condition-stratified success ----------------
    print("\n=== Part 2: episode success by condition, matched RNG across controllers ===")
    states = make_states(800, NUM_SEEDS)
    all_conditions = dict(CONDITIONS)
    all_conditions["state_disturbance"] = {}  # handled specially below

    results = {name: {c: [] for c in all_conditions} for name in ("fixed_low", "fixed_high", "adaptive")}
    effective_k = {name: {c: [] for c in all_conditions} for name in ("fixed_low", "fixed_high", "adaptive")}

    for condition_name, condition in all_conditions.items():
        for i in range(NUM_SEEDS):
            seed = 1000 + i
            x0 = states[i]

            def run_with(make_ctrl, disturb=(condition_name == "state_disturbance")):
                np.random.seed(seed)
                torch.manual_seed(seed)
                controller = make_ctrl()
                controller.reset()
                x = x0.copy()
                rollout_total = 0
                for step in range(EVAL_STEPS):
                    if disturb and step == DISTURBANCE_STEP:
                        x[1] = DISTURBANCE_OMEGA
                    action = controller.command(x)
                    rollout_total += getattr(controller, "last_rollouts", controller.rollouts)
                    if isinstance(controller, AdaptiveGenericMPPI):
                        rollout_total += PILOT_ROLLOUTS * PILOT_HORIZON
                    u = float(np.asarray(action).reshape(-1)[0])
                    x = pendulum_step(x, u, **condition)
                theta_err = abs(angle_error(x[0]))
                omega_err = abs(x[1])
                success = theta_err < 0.1 and omega_err < 0.5
                return success, rollout_total / EVAL_STEPS

            for name, factory in (
                ("fixed_low", lambda: make_fixed(ensemble, K_MIN)),
                ("fixed_high", lambda: make_fixed(ensemble, K_MAX)),
                ("adaptive", lambda: make_adaptive(ensemble, threshold)),
            ):
                success, eff_k = run_with(factory)
                results[name][condition_name].append(success)
                effective_k[name][condition_name].append(eff_k)

    print(f"{'condition':<18} {'method':<10} {'success (95% CI)':>20} {'mean eff. K':>13}")
    print("-" * 65)
    for condition_name in all_conditions:
        for name in ("fixed_low", "fixed_high", "adaptive"):
            s_mean, s_ci = summarize(results[name][condition_name])
            k_mean, _ = summarize(effective_k[name][condition_name])
            print(f"{condition_name:<18} {name:<10} {100*s_mean:8.1f}% +/- {100*s_ci:5.1f}%   {k_mean:13.1f}")
        print("-" * 65)
