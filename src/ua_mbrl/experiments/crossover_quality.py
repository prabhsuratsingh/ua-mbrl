"""Matched-ceiling quality comparison of fixed-high vs adaptive-K.

``confound_check``: both controllers run sequentially with identical per-seed
seeding, sigma=1.0 throughout (adaptive_k_confound_check.py). Its published
adaptive column (tab:confound) predates the pilot-RNG isolation fix and is
not reproducible with the current controller.

``crossover_quality``: the same protocol at uniform sigma=1.0 vs sigma
retuned per K_max, with the adaptive controller switching to the retuned
sigma on escalation (crossover_quality_retuned.py, tab:confoundfixed).
"""

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval import pendulum as runners

# Best fixed-K sigma per K from the n=40 nominal sweep (tab:sigmasweep).
BEST_SIGMA = {100: 1.0, 1000: 5.0, 8000: 3.0}


def _common_arguments(parser):
    parser.add_argument("--k-values", type=int, nargs="+", default=[100, 1000, 8000])
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=30)


def add_confound_arguments(parser):
    _common_arguments(parser)


def add_arguments(parser):
    _common_arguments(parser)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=None,
        help="retuned sigma per K value (same order as --k-values); default: BEST_SIGMA",
    )


def _setup(args):
    ensemble = pend.train_default_ensemble()
    calibration_states = pend.make_states(pend.CALIBRATION_STATES_SEED, 40)
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)
    return ensemble, calibration_states, states


def _compare(ensemble, calibration_states, states, k_max, sigma, eval_steps):
    threshold = pend.calibrate_threshold(ensemble, k_max, calibration_states, min_sigma=1.0, max_sigma=sigma)
    fixed = runners.run_fixed_sequential(ensemble, states, k_max, eval_steps, seed=1000, noise_sigma=sigma)
    adaptive = runners.run_adaptive_sequential(
        ensemble, states, k_max, threshold, eval_steps, seed=1000, min_sigma=1.0, max_sigma=sigma
    )
    return {
        "threshold": threshold,
        "fixed_success": float(fixed["per_seed"].mean()),
        "adaptive_success": float(adaptive["per_seed"].mean()),
        "fixed_per_seed": fixed["per_seed"],
        "adaptive_per_seed": adaptive["per_seed"],
        "adaptive_eff_k": adaptive["eff_k"],
        "adaptive_escalation_rate": adaptive["escalation_rate"],
        **runners.paired_record(fixed["per_seed"], adaptive["per_seed"]),
    }


def run_confound(args, run_dir):
    ensemble, calibration_states, states = _setup(args)
    print(f"\n{'K_max':>7} {'fixed_high (seq, matched RNG)':>30} {'adaptive':>10}  {'gap':>6}")
    for k_max in args.k_values:
        result = _compare(ensemble, calibration_states, states, k_max, 1.0, args.eval_steps)
        run_dir.append({"k_max": k_max, "sigma": 1.0, **result})
        gap = result["adaptive_success"] - result["fixed_success"]
        print(f"{k_max:7d} {result['fixed_success']:30.3f} {result['adaptive_success']:10.3f}  {gap:+.3f}")


def run(args, run_dir):
    ensemble, calibration_states, states = _setup(args)
    sigmas = args.sigmas or [BEST_SIGMA[k] for k in args.k_values]
    print(f"{'K_max':>7} {'method':<12} {'sigma=1.0 (orig)':>18} {'sigma retuned':>15} {'retuned sigma used':>20}")
    print("-" * 80)
    for k_max, best_sigma in zip(args.k_values, sigmas):
        original = _compare(ensemble, calibration_states, states, k_max, 1.0, args.eval_steps)
        retuned = _compare(ensemble, calibration_states, states, k_max, best_sigma, args.eval_steps)
        run_dir.append({"k_max": k_max, "retuned_sigma": best_sigma, "original": original, "retuned": retuned})
        for method in ("fixed", "adaptive"):
            label = "fixed-high" if method == "fixed" else "adaptive"
            print(
                f"{k_max:7d} {label:<12} {100 * original[f'{method}_success']:17.1f}% "
                f"{100 * retuned[f'{method}_success']:14.1f}% {best_sigma:20.1f}"
            )
        print("-" * 80)
