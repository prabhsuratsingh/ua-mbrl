"""Claim A: matched sample budget, not matched ceiling.

For each (K_max, condition) case:
  1. sweep sigma for a seed-batched fixed controller at K_max (ceiling);
  2. run adaptive-K (K_min=100, sigma 1.0 -> ceiling sigma on escalation);
  3. sweep sigma for a fixed controller at round(adaptive's mean effective K);
  4. paired win/tie/loss on the shared seeded states + exact sign test.

Merges claim_a_extended.py (aggregates) and claim_a_paired_test.py (paired
record); both ran the identical protocol.
"""

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval import pendulum as runners
from ua_mbrl.utils.stats import exact_sign_test, wilson_interval

CEILING_SIGMAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
DEFAULT_CASES = ["3000:nominal", "8000:nominal", "16000:nominal", "1000:mild_mismatch"]


def add_arguments(parser):
    parser.add_argument(
        "--cases", nargs="+", default=DEFAULT_CASES,
        help="K_max:condition pairs; conditions: " + ", ".join(pend.CONDITIONS),
    )
    parser.add_argument("--num-seeds", type=int, default=40)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(CEILING_SIGMAS))
    parser.add_argument("--k-min", type=int, default=pend.ROLLOUTS)


def run_case(ensemble, calibration_states, states, k_max, condition_name, args):
    condition = pend.CONDITIONS[condition_name]
    print(f"\n=== K_max={k_max}, condition={condition_name} ===")

    ceiling_sigma, ceiling_success, ceiling_all, _ = runners.sweep_sigma_batched(
        ensemble, k_max, args.sigmas, states, args.eval_steps, condition
    )
    print(f"  ceiling sigma sweep: {ceiling_all}")
    print(f"  fixed @ K_max={k_max}, best sigma={ceiling_sigma}: {ceiling_success:.1f}%")

    threshold = pend.calibrate_threshold(
        ensemble, k_max, calibration_states, k_min=args.k_min, min_sigma=1.0, max_sigma=ceiling_sigma
    )
    adaptive = runners.run_adaptive_sequential(
        ensemble, states, k_max, threshold, args.eval_steps, seed=1000, condition=condition,
        k_min=args.k_min, min_sigma=1.0, max_sigma=ceiling_sigma,
    )
    adaptive_success = runners.percent(adaptive["per_seed"])
    print(f"  adaptive: {adaptive_success:.1f}% at mean effective K={adaptive['eff_k']:.1f} "
          f"(escalation rate {100 * adaptive['escalation_rate']:.2f}%)")

    k_matched = int(round(adaptive["eff_k"]))
    matched_sigma, matched_success, matched_all, matched_per_seed = runners.sweep_sigma_batched(
        ensemble, k_matched, args.sigmas, states, args.eval_steps, condition
    )
    print(f"  matched-budget sigma sweep at K={k_matched}: {matched_all}")
    print(f"  fixed @ matched K={k_matched}, best sigma={matched_sigma}: {matched_success:.1f}%")

    paired = runners.paired_record(matched_per_seed, adaptive["per_seed"])
    p_value = exact_sign_test(paired["fixed_wins"], paired["adaptive_wins"])
    n = len(states)
    print(f"  --- PAIRED: fixed wins={paired['fixed_wins']}, adaptive wins={paired['adaptive_wins']}, "
          f"ties={paired['ties']}, exact sign-test p={p_value:.4f} ---")

    return {
        "k_max": k_max, "condition": condition_name, "n": n,
        "ceiling_sigma": ceiling_sigma, "ceiling_success": ceiling_success, "ceiling_sweep": ceiling_all,
        "threshold": threshold,
        "adaptive_success": adaptive_success, "eff_k": adaptive["eff_k"],
        "escalation_rate": adaptive["escalation_rate"],
        "matched_k": k_matched, "matched_sigma": matched_sigma, "matched_success": matched_success,
        "matched_sweep": matched_all,
        **paired, "sign_test_p": p_value,
        "matched_wilson": wilson_interval(int(matched_per_seed.sum()), n),
        "adaptive_wilson": wilson_interval(int(adaptive["per_seed"].sum()), n),
        "matched_per_seed": matched_per_seed, "adaptive_per_seed": adaptive["per_seed"],
        "adaptive_eff_k_per_seed": adaptive["eff_k_per_seed"],
    }


def run(args, run_dir):
    ensemble = pend.train_default_ensemble()
    calibration_states = pend.make_states(pend.CALIBRATION_STATES_SEED, 40)
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)

    results = []
    for case in args.cases:
        k_max, condition_name = case.split(":")
        record = run_case(ensemble, calibration_states, states, int(k_max), condition_name, args)
        run_dir.append(record)
        results.append(record)

    print("\n\n=== FINAL SUMMARY ===")
    print(f"{'K_max':>7} {'condition':<14} {'ceiling':>8} {'matched':>8} {'adaptive':>9} {'matched_K':>9} "
          f"{'eff_K':>7} {'f_w':>4} {'a_w':>4} {'tie':>4} {'p':>7}")
    for r in results:
        print(
            f"{r['k_max']:7d} {r['condition']:<14} {r['ceiling_success']:7.1f}% {r['matched_success']:7.1f}% "
            f"{r['adaptive_success']:8.1f}% {r['matched_k']:9d} {r['eff_k']:7.1f} "
            f"{r['fixed_wins']:4d} {r['adaptive_wins']:4d} {r['ties']:4d} {r['sign_test_p']:7.4f}"
        )
    return results
