"""Closed-loop pendulum evaluation runners shared by the experiments.

RNG protocol (kept identical to the original scripts so seeded results
reproduce exactly):

* seed-batched fixed-K: build + reset the controller, *then*
  ``torch.manual_seed(seed)`` once; all S seeds share one advancing stream.
* sequential (fixed or adaptive): per seed index i, ``np.random.seed(seed+i)``
  and ``torch.manual_seed(seed+i)``, then build the controller, then reset.
"""

import time

import numpy as np
import torch

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.utils.device import synchronize


def run_fixed_batched(model, states, rollouts, steps, seed=1000, noise_sigma=1.0, condition=None, timed=False):
    """All seeds advanced together by one SeedBatchedGenericMPPI.

    Returns per-seed success, final states and (if timed) per-step wall-clock
    latency for advancing all S seeds by one control step.
    """
    condition = condition or {}
    num_seeds = len(states)
    controller = pend.make_seed_batched(model, num_seeds, rollouts, noise_sigma=noise_sigma)
    controller.reset()
    torch.manual_seed(seed)
    x = np.asarray(states, dtype=np.float32)
    step_latencies = []
    for _ in range(steps):
        if timed:
            synchronize()
            start = time.perf_counter()
        actions = controller.command(x)
        if timed:
            synchronize()
            step_latencies.append(time.perf_counter() - start)
        u = np.asarray(actions).reshape(num_seeds)
        x = np.stack([pend.pendulum_step(x[s], float(u[s]), **condition) for s in range(num_seeds)])
    per_seed = pend.batch_success(x)
    return {"per_seed": per_seed, "final_states": x, "step_latencies": step_latencies}


def run_sequential(
    make_controller, states, steps, seed=1000, condition=None, timed=False,
    disturbance=None, construct_before_seed=False,
):
    """One controller per seed, run one after another with per-seed seeding.

    ``make_controller()`` builds a fresh controller (fixed or adaptive).
    ``disturbance=(step, omega)`` overwrites omega at that step (state-space
    kick). ``construct_before_seed`` reproduces the fallback/robustness
    scripts, which built each controller *before* seeding (that shifts which
    global CPU draw seeds an adaptive controller's pilot generator).

    Returns per-seed success, per-seed |omega| traces (initial state then
    after every step), fallback-trigger counts and, for adaptive controllers,
    the mean effective rollouts per step (main + pilot) and escalation rate.
    """
    condition = condition or {}
    successes, eff_ks, escalation, mean_latencies, final_states = [], [], [], [], []
    omega_traces, fallback_steps = [], []
    for i, x0 in enumerate(states):
        if construct_before_seed:
            controller = make_controller()
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        if not construct_before_seed:
            controller = make_controller()
        controller.reset()
        x = np.array(x0, dtype=np.float32, copy=True)
        ks, escalated, latencies = [], [], []
        omega_trace = [abs(x[1])]
        fallbacks = 0
        for step in range(steps):
            if disturbance is not None and step == disturbance[0]:
                x[1] = disturbance[1]
            if timed:
                synchronize()
                start = time.perf_counter()
            action = controller.command(x)
            if timed:
                synchronize()
                latencies.append(time.perf_counter() - start)
            if hasattr(controller, "pilot_rollouts"):
                ks.append(controller.last_rollouts + controller.pilot_rollouts * controller.pilot_horizon)
                escalated.append(controller.last_rollouts == controller.max_rollouts)
            else:
                ks.append(controller.rollouts)
            if getattr(controller, "fallback_triggered_last", False):
                fallbacks += 1
            u = float(np.asarray(action).reshape(-1)[0])
            x = pend.pendulum_step(x, u, **condition)
            omega_trace.append(abs(x[1]))
        successes.append(pend.is_success(x))
        eff_ks.append(np.mean(ks))
        escalation.append(float(np.mean(escalated)) if escalated else 0.0)
        if timed:
            mean_latencies.append(np.mean(latencies))
        final_states.append(x)
        omega_traces.append(omega_trace)
        fallback_steps.append(fallbacks)
    return {
        "per_seed": np.asarray(successes, dtype=bool),
        "eff_k_per_seed": eff_ks,
        "eff_k": float(np.mean(eff_ks)),
        "escalation_rate": float(np.mean(escalation)),
        "mean_step_latency_per_seed": mean_latencies,
        "final_states": np.asarray(final_states),
        "omega_traces": omega_traces,
        "fallback_steps": fallback_steps,
    }


def run_adaptive_sequential(
    model, states, k_max, threshold, steps, seed=1000, condition=None,
    k_min=pend.ROLLOUTS, min_sigma=1.0, max_sigma=1.0, timed=False,
):
    return run_sequential(
        lambda: pend.make_adaptive(
            model, k_max, threshold, k_min=k_min, min_sigma=min_sigma, max_sigma=max_sigma
        ),
        states, steps, seed=seed, condition=condition, timed=timed,
    )


def run_fixed_sequential(model, states, rollouts, steps, seed=1000, noise_sigma=1.0, condition=None):
    return run_sequential(
        lambda: pend.make_fixed(model, rollouts=rollouts, noise_sigma=noise_sigma),
        states, steps, seed=seed, condition=condition,
    )


def percent(per_seed):
    return float(100 * np.asarray(per_seed).mean())


def sweep_sigma_batched(model, rollouts, sigmas, states, steps, condition=None, seed=1000):
    """Seed-batched success at each sigma; best = first sigma attaining the max.

    Returns (best_sigma, best_percent, {sigma: percent}, best_per_seed).
    """
    runs = {
        sigma: run_fixed_batched(model, states, rollouts, steps, seed=seed, noise_sigma=sigma, condition=condition)
        for sigma in sigmas
    }
    rates = {sigma: percent(r["per_seed"]) for sigma, r in runs.items()}
    best_sigma = max(rates, key=rates.get)
    return best_sigma, rates[best_sigma], rates, runs[best_sigma]["per_seed"]


def paired_record(fixed_per_seed, adaptive_per_seed):
    """Seed-level win/tie/loss record of fixed vs adaptive."""
    fixed_per_seed = np.asarray(fixed_per_seed, dtype=bool)
    adaptive_per_seed = np.asarray(adaptive_per_seed, dtype=bool)
    return {
        "fixed_wins": int(np.sum(fixed_per_seed & ~adaptive_per_seed)),
        "adaptive_wins": int(np.sum(~fixed_per_seed & adaptive_per_seed)),
        "ties": int(np.sum(fixed_per_seed == adaptive_per_seed)),
    }
