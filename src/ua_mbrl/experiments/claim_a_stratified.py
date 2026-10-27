"""Does the gate's extra compute land where extra rollouts help?

Part 1 -- ESS vs pilot uncertainty along ONE reference trajectory per
condition: at each state the adaptive controller visits, a throwaway
"shadow" K_min / K_max MPPI call (same nominal plan) measures the effective
sample size each budget would achieve there. Steps are bucketed into
terciles of pilot uncertainty.

Part 2 -- episode success for fixed-low / fixed-high / adaptive under
matched per-seed RNG, stratified by known condition difficulty.

(claim_a_stratified_experiment.py)
"""

import numpy as np
import torch

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval import pendulum as runners
from ua_mbrl.utils.stats import summarize_z

DISTURBANCE_STEP = 10
DISTURBANCE_OMEGA = 5.0


def add_arguments(parser):
    parser.add_argument("--k-min", type=int, default=pend.ROLLOUTS)
    parser.add_argument("--k-max", type=int, default=3 * pend.ROLLOUTS)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--skip-part1", action="store_true")


def shadow_ess(ensemble, nominal_snapshot, x, k, seed):
    """ESS a K-rollout MPPI call would achieve right now at state x with the
    reference controller's nominal plan; does not affect the reference run
    (other than reseeding torch's global RNG, as the original did)."""
    torch.manual_seed(seed)
    shadow = pend.make_fixed(ensemble, rollouts=k)
    shadow.nominal = nominal_snapshot.clone()
    shadow.command(x)
    return shadow.last_ess


def run_reference_trajectory(ensemble, threshold, condition_name, condition, x0, seed, disturb, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller = pend.make_adaptive(ensemble, args.k_max, threshold, k_min=args.k_min)
    controller.reset()
    x = x0.copy()
    rows = []
    for step in range(args.eval_steps):
        if disturb and step == DISTURBANCE_STEP:
            x[1] = DISTURBANCE_OMEGA
        nominal_snapshot = controller.nominal.clone()
        action = controller.command(x)
        eta = controller.last_pilot_uncertainty
        k_used = controller.last_rollouts
        ess_min = shadow_ess(ensemble, nominal_snapshot, x, args.k_min, seed=90000 + step)
        ess_max = shadow_ess(ensemble, nominal_snapshot, x, args.k_max, seed=90000 + step)
        rows.append({
            "condition": condition_name, "step": step, "eta": eta,
            "k_used": k_used, "ess_min": ess_min, "ess_max": ess_max,
        })
        u = float(np.asarray(action).reshape(-1)[0])
        x = pend.pendulum_step(x, u, **condition)
    return rows


def run(args, run_dir):
    ensemble = pend.train_default_ensemble()
    threshold = pend.calibrate_threshold(
        ensemble, args.k_max, pend.make_states(pend.CALIBRATION_STATES_SEED, 40), k_min=args.k_min
    )
    print(f"mild threshold: {threshold:.6f}")

    if not args.skip_part1:
        print("\n=== Part 1: ESS vs pilot uncertainty (one reference trajectory per condition) ===")
        ref_state = pend.make_states(700, 1)[0]
        rows = []
        for condition_name, condition in pend.CONDITIONS.items():
            rows.extend(run_reference_trajectory(
                ensemble, threshold, condition_name, condition, ref_state, 500, False, args
            ))
        rows.extend(run_reference_trajectory(
            ensemble, threshold, "state_disturbance", {}, ref_state, 500, True, args
        ))
        for row in rows:
            run_dir.append({"part": 1, **row})

        etas = np.array([r["eta"] for r in rows])
        ess_min = np.array([r["ess_min"] for r in rows])
        ess_max = np.array([r["ess_max"] for r in rows])
        k_used = np.array([r["k_used"] for r in rows])
        bucket_idx = np.digitize(etas, np.quantile(etas, [1 / 3, 2 / 3]))
        print(f"{'bucket':>6} {'n':>4} {'eta_range':>18} {'ESS(K_min)':>12} {'ESS(K_max)':>12} "
              f"{'gain(max-min)':>14} {'frac_K_max':>11}")
        for b, name in enumerate(("low", "mid", "high")):
            mask = bucket_idx == b
            if mask.sum() == 0:
                continue
            print(
                f"{name:>6} {mask.sum():4d} [{etas[mask].min():.5f},{etas[mask].max():.5f}] "
                f"{ess_min[mask].mean():12.2f} {ess_max[mask].mean():12.2f} "
                f"{ess_max[mask].mean() - ess_min[mask].mean():14.2f} {np.mean(k_used[mask] >= args.k_max):11.2f}"
            )

    print("\n=== Part 2: episode success by condition, matched RNG across controllers ===")
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)
    conditions = dict(pend.CONDITIONS)
    conditions["state_disturbance"] = {}
    factories = {
        "fixed_low": lambda: pend.make_fixed(ensemble, rollouts=args.k_min),
        "fixed_high": lambda: pend.make_fixed(ensemble, rollouts=args.k_max),
        "adaptive": lambda: pend.make_adaptive(ensemble, args.k_max, threshold, k_min=args.k_min),
    }
    print(f"{'condition':<18} {'method':<10} {'success (95% CI)':>20} {'mean eff. K':>13}")
    print("-" * 65)
    for condition_name, condition in conditions.items():
        disturbance = (DISTURBANCE_STEP, DISTURBANCE_OMEGA) if condition_name == "state_disturbance" else None
        for name, factory in factories.items():
            result = runners.run_sequential(
                factory, states, args.eval_steps, seed=1000, condition=condition, disturbance=disturbance
            )
            s_mean, s_ci = summarize_z(result["per_seed"])
            k_mean, _ = summarize_z(result["eff_k_per_seed"])
            run_dir.append({
                "part": 2, "condition": condition_name, "method": name,
                "per_seed": result["per_seed"], "eff_k_per_seed": result["eff_k_per_seed"],
            })
            print(f"{condition_name:<18} {name:<10} {100*s_mean:8.1f}% +/- {100*s_ci:5.1f}%   {k_mean:13.1f}")
        print("-" * 65)
