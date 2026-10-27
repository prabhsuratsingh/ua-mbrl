"""Conservative fallback: a second, severe-OOD threshold (3.5x the mild
compute-gating threshold, from the OOD severity sweep's ~3-4x uncertainty
ratio at reliable detection) that replaces the MPPI action with a model-free
damping law.

Variants (each trains its own ensemble, as the original scripts did):
  mismatch     -- dynamics-parameter mismatch (conservative_fallback_experiment.py)
  disturbance  -- omega kicked to 5 at step 10 (conservative_fallback_disturbance_test.py)
  narrow       -- same kick, narrow-data ensemble from the OOD sweep
                  (conservative_fallback_narrow_ensemble.py)

The original scripts built each controller *before* seeding its episode;
``construct_before_seed=True`` preserves that.
"""

import numpy as np
import torch

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval import pendulum as runners
from ua_mbrl.utils.device import DEVICE

SEVERE_MULTIPLIER = 3.5
FALLBACK_GAIN = 2.0
DISTURBANCE_STEP = 10
DISTURBANCE_OMEGA = 5.0
VARIANTS = ("mismatch", "disturbance", "narrow")


def add_arguments(parser):
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--eval-steps", type=int, default=40)


def _factories(ensemble, mild, severe):
    k_min, k_max = pend.ROLLOUTS, 3 * pend.ROLLOUTS
    return {
        "adaptive": lambda: pend.make_adaptive(ensemble, k_max, mild, k_min=k_min),
        "fallback": lambda: pend.make_fallback(
            ensemble, mild, severe, k_min=k_min, k_max=k_max, fallback_gain=FALLBACK_GAIN
        ),
    }


def _summarize(result, eval_steps, after_step=None):
    traces = result["omega_traces"]
    if after_step is None:
        peaks = [max(trace) for trace in traces]
    else:
        # trace[0] is the initial state; trace[s + 1] is the state after step s.
        peaks = [max(trace[after_step + 1:]) for trace in traces]
    return {
        "success": float(np.mean(result["per_seed"])),
        "max_abs_omega_mean": float(np.mean(peaks)),
        "max_abs_omega_worst": float(np.max(peaks)),
        "fallback_fraction": float(np.mean([f / eval_steps for f in result["fallback_steps"]])),
        "per_seed": result["per_seed"],
        "peak_abs_omega_per_seed": peaks,
    }


def run_mismatch(args, run_dir):
    ensemble = pend.train_default_ensemble()
    mild = pend.calibrate_threshold(ensemble, 3 * pend.ROLLOUTS, pend.make_states(pend.CALIBRATION_STATES_SEED, 40))
    severe = mild * SEVERE_MULTIPLIER
    print(f"mild (compute-gate) threshold : {mild:.6f}")
    print(f"severe (fallback) threshold   : {severe:.6f}  (x{SEVERE_MULTIPLIER})")
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)
    conditions = {
        "nominal": {},
        "mild_mismatch (mass+20%)": {"mass": 1.2},
        "severe_mismatch (mass x3, gravity-60%)": {"mass": 3.0, "gravity": pend.NOMINAL_G * 0.4},
    }
    print(f"\n{'condition':<40} {'method':<10} {'success':>8} {'mean|omega|max':>15} {'worst|omega|max':>16} {'fallback%':>10}")
    print("-" * 105)
    for condition_name, condition in conditions.items():
        for name, factory in _factories(ensemble, mild, severe).items():
            result = runners.run_sequential(
                factory, states, args.eval_steps, seed=1000, condition=condition, construct_before_seed=True
            )
            summary = _summarize(result, args.eval_steps)
            run_dir.append({"variant": "mismatch", "condition": condition_name, "method": name,
                            "mild_threshold": mild, "severe_threshold": severe, **summary})
            print(
                f"{condition_name:<40} {name:<10} {summary['success']:8.3f} "
                f"{summary['max_abs_omega_mean']:15.3f} {summary['max_abs_omega_worst']:16.3f} "
                f"{100*summary['fallback_fraction']:9.1f}%"
            )
        print("-" * 105)


def _run_disturbance(variant, ensemble, mild, severe, states, args, run_dir):
    print(f"mild={mild:.6f} severe={severe:.6f}")
    print(f"Disturbance: omega set to {DISTURBANCE_OMEGA} at step {DISTURBANCE_STEP}\n")
    for name, factory in _factories(ensemble, mild, severe).items():
        result = runners.run_sequential(
            factory, states, args.eval_steps, seed=1000,
            disturbance=(DISTURBANCE_STEP, DISTURBANCE_OMEGA), construct_before_seed=True,
        )
        summary = _summarize(result, args.eval_steps, after_step=DISTURBANCE_STEP)
        run_dir.append({"variant": variant, "method": name, "mild_threshold": mild,
                        "severe_threshold": severe, **summary})
        print(
            f"{name:<10} success={summary['success']:.3f}  "
            f"mean|omega|max_after_disturbance={summary['max_abs_omega_mean']:.3f}  "
            f"worst={summary['max_abs_omega_worst']:.3f}  "
            f"fallback_triggered={100*summary['fallback_fraction']:.1f}% of steps"
        )


def run_disturbance(args, run_dir):
    ensemble = pend.train_default_ensemble()
    mild = pend.calibrate_threshold(ensemble, 3 * pend.ROLLOUTS, pend.make_states(pend.CALIBRATION_STATES_SEED, 40))
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)
    _run_disturbance("disturbance", ensemble, mild, mild * SEVERE_MULTIPLIER, states, args, run_dir)


def run_narrow(args, run_dir):
    _, ensemble = pend.train_models(pend.collect_narrow_data(seed=0), train_fraction=None)

    # Single-step disagreement vs omega for this ensemble (in-range actions).
    print("\nSingle-step disagreement vs omega (sanity check):")
    rng = np.random.default_rng(0)
    for omega in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        n = 100
        theta = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        u = rng.uniform(-2.0, 2.0, n).astype(np.float32)
        x = torch.as_tensor(np.stack([theta, np.full(n, omega, dtype=np.float32)], axis=1), device=DEVICE)
        with torch.no_grad():
            _, variance, _ = ensemble.forward_mean_var(x, torch.as_tensor(u.reshape(-1, 1), device=DEVICE))
        mean_uncertainty = variance.sum(dim=1).mean().item()
        run_dir.append({"variant": "narrow", "sanity_omega": omega, "mean_uncertainty": mean_uncertainty})
        print(f"  omega={omega:4.1f}  mean_uncertainty={mean_uncertainty:.6f}")

    # Calibration and evaluation states clipped to the narrow training range.
    bounds = ([-np.pi, -1.0], [np.pi, 1.0])
    calibration_states = np.clip(pend.make_states(pend.CALIBRATION_STATES_SEED, 40), *bounds).astype(np.float32)
    mild = pend.calibrate_threshold(ensemble, 3 * pend.ROLLOUTS, calibration_states)
    states = np.clip(pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds), *bounds).astype(np.float32)
    _run_disturbance("narrow", ensemble, mild, mild * SEVERE_MULTIPLIER, states, args, run_dir)


def run(args, run_dir):
    variants = {"mismatch": run_mismatch, "disturbance": run_disturbance, "narrow": run_narrow}
    for variant in args.variants:
        print(f"\n######## fallback: {variant} ########")
        variants[variant](args, run_dir)
