"""MuJoCo two-link regulation: PD / single / ensemble / adaptive per seed.

``--sigma 1.0`` with seeds 7..147 reproduces tab:mujoco regulation
(mujoco_multiseed_study.py); ``--sigma 0.2`` with seeds 7..77 reproduces the
retuned paragraph (mujoco_quality_retuned.py). One record per seed.
"""

import time

import numpy as np
import torch

from ua_mbrl.core.dynamics import GenericEnsembleDynamics
from ua_mbrl.core.mppi import GenericMPPI
from ua_mbrl.envs import two_link as arm
from ua_mbrl.utils.device import DEVICE
from ua_mbrl.utils.stats import summarize_t

DEFAULT_SEEDS = list(range(7, 148, 10))  # 15 seeds: 7, 17, ..., 147
METHODS = ("single", "ensemble", "adaptive")


def add_arguments(parser):
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--sigma", type=float, default=1.0, help="MPPI exploration noise (pilot uses the same)")
    parser.add_argument("--episodes", type=int, default=arm.EPISODES)
    parser.add_argument("--steps", type=int, default=arm.STEPS)


def train_seed(seed, device):
    """Data, states and models for one seed, seeded exactly as the original."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_environment = arm.TwoLinkMuJoCo()
    train_data = arm.collect_task_directed_data(train_environment, transitions=2000, seed=seed)
    normalizer = arm.make_normalizer(train_data, device)
    return train_data, normalizer


def run_one_seed(seed, device, args):
    train_data, normalizer = train_seed(seed, device)
    states = arm.make_states(seed + 1000, args.episodes)
    pd_success = arm.evaluate_pd(states, steps=args.steps)

    torch.manual_seed(seed)
    single = arm.train_single(train_data, device, normalizer)
    torch.manual_seed(seed)
    ensemble = arm.train_ensemble(train_data, device, normalizer)
    calibrator = arm.make_controller(ensemble, device, adaptive=True, noise_sigma=args.sigma)
    torch.manual_seed(seed + 5000)  # not the threshold: seeds the CPU stream later adaptive controllers draw pilot seeds from
    threshold = calibrator.calibrate_threshold(arm.make_states(seed + 2000, args.episodes))

    record = {"seed": seed, "device": str(device), "sigma": args.sigma, "pd_success": pd_success, "threshold": threshold}
    for name, model, is_adaptive in [("single", single, False), ("ensemble", ensemble, False), ("adaptive", ensemble, True)]:
        record[name] = arm.evaluate(
            model, device, states, is_adaptive, threshold, noise_sigma=args.sigma, steps=args.steps
        )
    return record


def print_summary(records):
    print(f"\n{'method':<10} {'success (95% t-CI)':>22} {'latency_ms':>20} {'effective_K':>18}")
    print(f"{'pd':<10} {100 * np.mean([r['pd_success'] for r in records]):8.1f}%")
    for name in METHODS:
        s = summarize_t([r[name]["success_rate"] for r in records])
        lat = summarize_t([r[name]["latency_ms"] for r in records])
        k = summarize_t([r[name]["effective_rollouts"] for r in records])
        print(f"{name:<10} {100 * s[0]:8.1f}% +/- {100 * s[1]:5.1f}% "
              f"{lat[0]:9.2f} +/- {lat[1]:6.2f} {k[0]:8.1f} +/- {k[1]:6.1f}")


def run(args, run_dir):
    print(f"Device: {DEVICE}, noise_sigma={args.sigma}, seeds: {args.seeds}")
    records = []
    for seed in args.seeds:
        start = time.perf_counter()
        record = run_one_seed(seed, DEVICE, args)
        run_dir.append(record)
        records.append(record)
        print(
            f"seed={seed:5d} ({time.perf_counter() - start:5.1f}s) pd={record['pd_success']:.2f} "
            + " ".join(f"{name}={record[name]['success_rate']:.2f}" for name in METHODS)
        )
    print_summary(records)
    return records


def add_smoke_arguments(parser):
    pass


def run_smoke(args, run_dir):
    environment = arm.TwoLinkMuJoCo()
    data = arm.collect_random_data(environment)
    ensemble = GenericEnsembleDynamics(
        state_dim=environment.state_dim, action_dim=environment.action_dim,
        ensemble_size=5, hidden_dim=128, device=DEVICE,
    )
    ensemble.train_ensemble(data, epochs=50, seed=7, verbose=False)
    controller = GenericMPPI(
        dynamics=ensemble, action_dim=environment.action_dim, horizon=20, rollouts=500,
        action_low=environment.action_low, action_high=environment.action_high,
        target=np.zeros(environment.state_dim, dtype=np.float32), cost_fn=arm.arm_cost,
        dt=arm.DT, device=DEVICE,
    )
    state = environment.reset(seed=11)
    for step in range(10):
        action = controller.command(state)
        state = environment.step(action)
        run_dir.append({"step": step, "action": action, "state_norm": float(np.linalg.norm(state))})
        print(f"step={step:02d} action={np.round(action, 3)} state_norm={np.linalg.norm(state):.4f}")
    print("MuJoCo generic MPPI smoke test passed")
