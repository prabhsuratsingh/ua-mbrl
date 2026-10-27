"""MuJoCo matched-sample-budget Claim A: for each seed, a fixed controller at
round(that seed's adaptive mean effective K) with its own best sigma, vs that
seed's adaptive success (mujoco_claim_a.py).

Adaptive results come from a ``mujoco_regulation --sigma 0.2`` run (records
indexed by seed, last record wins). Note the sigma sweep does not reseed
between sigma values, so each value's result depends on sweep order.
"""

import torch

from ua_mbrl.envs import two_link as arm
from ua_mbrl.experiments.mujoco_regulation import train_seed
from ua_mbrl.utils.device import DEVICE
from ua_mbrl.utils.results import PUBLISHED_DIR, read_jsonl
from ua_mbrl.utils.stats import exact_sign_test, summarize_t

SIGMAS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
DEFAULT_ADAPTIVE_RESULTS = PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl"


def add_arguments(parser):
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: every seed in --adaptive-results")
    parser.add_argument("--adaptive-results", default=str(DEFAULT_ADAPTIVE_RESULTS),
                        help="records.jsonl from a mujoco_regulation run (sigma 0.2)")
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    parser.add_argument("--episodes", type=int, default=arm.EPISODES)
    parser.add_argument("--steps", type=int, default=arm.STEPS)


def run_one_seed(seed, device, original_record, args):
    train_data, normalizer = train_seed(seed, device)
    states = arm.make_states(seed + 1000, args.episodes)
    torch.manual_seed(seed)
    ensemble = arm.train_ensemble(train_data, device, normalizer)

    eff_k = original_record["adaptive"]["effective_rollouts"]
    k_matched = int(round(eff_k))
    sigma_results = {
        sigma: arm.evaluate_fixed_at_k(ensemble, device, states, k_matched, sigma, steps=args.steps)
        for sigma in args.sigmas
    }
    best_sigma = max(sigma_results, key=sigma_results.get)
    return {
        "seed": seed, "k_matched": k_matched, "eff_k": eff_k,
        "adaptive_success": original_record["adaptive"]["success_rate"],
        "ensemble_success_k300": original_record["ensemble"]["success_rate"],
        "fixed_matched_sigma_sweep": sigma_results,
        "fixed_matched_best_sigma": best_sigma,
        "fixed_matched_best_success": sigma_results[best_sigma],
    }


def run(args, run_dir):
    original = {r["seed"]: r for r in read_jsonl(args.adaptive_results)}
    if not original:
        raise SystemExit(f"no adaptive results found in {args.adaptive_results}")
    seeds = args.seeds or sorted(original)
    print(f"Device: {DEVICE}, seeds: {seeds}, adaptive results: {args.adaptive_results}")
    records = []
    for seed in seeds:
        record = run_one_seed(seed, DEVICE, original[seed], args)
        run_dir.append(record)
        records.append(record)
        print(
            f"seed={seed:5d} K_matched={record['k_matched']:4d} "
            f"fixed_matched_best={record['fixed_matched_best_success']:.2f} "
            f"(sigma={record['fixed_matched_best_sigma']}) adaptive={record['adaptive_success']:.2f}"
        )
    fixed = [r["fixed_matched_best_success"] for r in records]
    adaptive = [r["adaptive_success"] for r in records]
    wins = sum(f > a for f, a in zip(fixed, adaptive))
    losses = sum(f < a for f, a in zip(fixed, adaptive))
    f_mean, f_ci = summarize_t(fixed)
    a_mean, a_ci = summarize_t(adaptive)
    print(f"\nfixed @ matched budget: {100 * f_mean:.1f}% +/- {100 * f_ci:.1f}%  "
          f"adaptive: {100 * a_mean:.1f}% +/- {100 * a_ci:.1f}%  (n={len(records)}, t-CI)")
    print(f"paired: fixed wins {wins}, ties {len(records) - wins - losses}, losses {losses}; "
          f"exact sign test p={exact_sign_test(wins, losses):.3f}")
    return records
