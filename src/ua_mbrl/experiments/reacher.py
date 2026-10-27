"""Reacher-style task: single / ensemble / adaptive per seed
(mujoco_reacher_experiment.py, tab:mujoco reaching)."""

import numpy as np
import torch

from ua_mbrl.envs import reacher
from ua_mbrl.utils.device import DEVICE
from ua_mbrl.utils.stats import summarize_t

METHODS = ("single", "ensemble", "adaptive")


def add_arguments(parser):
    parser.add_argument("--seeds", type=int, nargs="+", default=list(reacher.SEEDS))
    parser.add_argument("--sigma", type=float, default=reacher.NOISE_SIGMA)
    parser.add_argument("--rollouts", type=int, default=reacher.ROLLOUTS, help="fixed K and adaptive K_max")
    parser.add_argument("--episodes", type=int, default=reacher.EPISODES)
    parser.add_argument("--steps", type=int, default=reacher.STEPS)


def run_one_seed(seed, device, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_environment = reacher.ReacherMuJoCo()
    train_data = reacher.collect_task_directed_data(train_environment, transitions=2000, seed=seed)
    episode_seeds = reacher.make_episode_seeds(seed + 1000, args.episodes)
    normalizer = reacher.make_normalizer(train_data, device)

    torch.manual_seed(seed)
    single = reacher.train_single(train_data, device, normalizer)
    torch.manual_seed(seed)
    ensemble = reacher.train_ensemble(train_data, device, normalizer)

    calibrator = reacher.make_controller(
        ensemble, device, np.zeros(2, dtype=np.float32), adaptive=True,
        noise_sigma=args.sigma, rollouts=args.rollouts,
    )
    torch.manual_seed(seed + 5000)  # not the threshold: seeds the CPU stream later adaptive controllers draw pilot seeds from
    calibration_states = [reacher.ReacherMuJoCo().reset(seed=s) for s in reacher.make_episode_seeds(seed + 2000, args.episodes)]
    threshold = calibrator.calibrate_threshold(calibration_states)

    record = {"seed": seed, "sigma": args.sigma, "rollouts": args.rollouts, "threshold": threshold}
    for name, model, is_adaptive in [("single", single, False), ("ensemble", ensemble, False), ("adaptive", ensemble, True)]:
        record[name] = reacher.evaluate(
            model, device, episode_seeds, is_adaptive, threshold,
            noise_sigma=args.sigma, rollouts=args.rollouts, steps=args.steps,
        )
    return record


def run(args, run_dir):
    print(f"Device: {DEVICE}, sigma={args.sigma}, K={args.rollouts}, seeds: {args.seeds}")
    records = []
    for seed in args.seeds:
        record = run_one_seed(seed, DEVICE, args)
        run_dir.append(record)
        records.append(record)
        print(f"seed={seed:4d}  " + "  ".join(f"{m}={record[m]['success_rate']:.2f}" for m in METHODS))

    print(f"\n{'method':<10} {'success (95% t-CI)':>20} {'final distance':>20} {'latency_ms':>18}")
    for m in METHODS:
        s = summarize_t([r[m]["success_rate"] for r in records])
        d = summarize_t([r[m]["final_distance"] for r in records])
        lat = summarize_t([r[m]["latency_ms"] for r in records])
        print(f"{m:<10} {100 * s[0]:7.1f}% +/- {100 * s[1]:5.1f}% {d[0]:8.3f} +/- {d[1]:.3f} "
              f"{lat[0]:8.2f} +/- {lat[1]:.2f}")
    return records
