"""Re-derive the MuJoCo single/ensemble/adaptive quality comparison
(mujoco_multiseed_study.py, 15 seeds) with exploration noise retuned.

A quick sweep (single seed) found MuJoCo's regulation task -- unlike the
pendulum's near-underactuated swing-up -- needs SMALLER noise than the
default sigma=1.0 (100% success at sigma in [0.1,0.3] vs 66.7% at
sigma=1.0; sigma=0.4 already drops back to 66.7%): this is a small-correction
task from a near-origin start, not one needing large decisive torques, so
the same "noise scale must match K and task" lesson applies in the opposite
direction here. sigma=0.2 is used throughout below as a representative
retuned value.

Appends one JSON line per seed to mujoco_quality_retuned_results.jsonl so it
can run in chunks; run without arguments to use the default seed list.
"""

import json
import os
import sys

import numpy as np
import torch

import mujoco_benchmark as mb

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "mujoco_quality_retuned_results.jsonl")
NOISE_SIGMA = 0.2


def make_controller_retuned(model, device, adaptive=False, threshold=0.0, pilot_rollouts=8, pilot_horizon=2):
    from generic_mbrl import AdaptiveGenericMPPI, GenericMPPI
    controller_type = AdaptiveGenericMPPI if adaptive else GenericMPPI
    adaptive_args = {
        "min_rollouts": 100, "max_rollouts": mb.ROLLOUTS, "uncertainty_threshold": threshold,
        "pilot_rollouts": pilot_rollouts, "pilot_horizon": pilot_horizon,
        "min_noise_sigma": NOISE_SIGMA, "max_noise_sigma": NOISE_SIGMA,
    } if adaptive else {}
    controller_kwargs = {
        "dynamics": model, "action_dim": mb.ACTION_DIM, "horizon": mb.HORIZON,
        "action_low": -5.0 * np.ones(mb.ACTION_DIM), "action_high": 5.0 * np.ones(mb.ACTION_DIM),
        "target": np.zeros(mb.STATE_DIM, dtype=np.float32), "cost_fn": mb.arm_cost, "dt": 0.02,
        "noise_sigma": NOISE_SIGMA, "device": device,
    }
    if not adaptive:
        controller_kwargs["rollouts"] = mb.ROLLOUTS
    return controller_type(**controller_kwargs, **adaptive_args)


def evaluate_retuned(model, device, states, adaptive=False, threshold=0.0, pilot_rollouts=8, pilot_horizon=2):
    import time
    environment = mb.TwoLinkMuJoCo()
    controller = make_controller_retuned(model, device, adaptive, threshold, pilot_rollouts, pilot_horizon)
    successes = []
    latencies = []
    effective_rollouts = []
    for episode, initial_state in enumerate(states):
        environment.reset(seed=episode)
        environment.data.qpos[:] = initial_state[:2]
        environment.data.qvel[:] = initial_state[2:]
        import mujoco
        mujoco.mj_forward(environment.model, environment.data)
        state = environment.state()
        controller.reset()
        for _ in range(mb.STEPS):
            command_start = time.perf_counter()
            action = controller.command(state)
            latencies.append(time.perf_counter() - command_start)
            effective = getattr(controller, "last_rollouts", controller.rollouts)
            if adaptive:
                effective += controller.pilot_rollouts * controller.pilot_horizon
            effective_rollouts.append(effective)
            state = environment.step(action)
        successes.append(float(np.linalg.norm(state) < 0.5))
    return {
        "success_rate": float(np.mean(successes)),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
        "effective_rollouts": float(np.mean(effective_rollouts)),
    }


def run_one_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_environment = mb.TwoLinkMuJoCo()
    train_data = mb.collect_task_directed_data(train_environment, transitions=2000, seed=seed)
    states = mb.make_states(seed + 1000)
    normalizer = mb.make_normalizer(train_data, device)

    torch.manual_seed(seed)
    single = mb.train_single(train_data, device, normalizer)
    torch.manual_seed(seed)
    ensemble = mb.train_ensemble(train_data, device, normalizer)
    calibrator = make_controller_retuned(ensemble, device, adaptive=True)
    torch.manual_seed(seed + 5000)
    threshold = calibrator.calibrate_threshold(mb.make_states(seed + 2000))

    record = {"seed": seed, "threshold": threshold}
    for name, model, is_adaptive in [
        ("single", single, False),
        ("ensemble", ensemble, False),
        ("adaptive", ensemble, True),
    ]:
        result = evaluate_retuned(model, device, states, is_adaptive, threshold)
        record[name] = result
    return record


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [7, 17, 27, 37, 47, 57, 67, 77]
    device = mb.DEVICE
    print(f"Device: {device}, noise_sigma={NOISE_SIGMA}, seeds this chunk: {seeds}")

    with open(RESULTS_PATH, "a") as f:
        for seed in seeds:
            record = run_one_seed(seed, device)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"seed={seed:5d} single={record['single']['success_rate']:.2f} "
                f"ensemble={record['ensemble']['success_rate']:.2f} "
                f"adaptive={record['adaptive']['success_rate']:.2f}"
            )
    print(f"\nAppended {len(seeds)} seed(s) to {RESULTS_PATH}")
