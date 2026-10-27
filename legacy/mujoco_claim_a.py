"""MuJoCo's own matched-sample-budget Claim A test (see pendulum's
claim_a_extended.py and the resulting reversal in research_status.tex
sec:claima / paper.tex sec:claima).

For each seed, train the ensemble exactly as in mujoco_quality_retuned.py
(sigma=0.2 throughout), take that seed's own already-measured adaptive
effective rollout count (mujoco_quality_retuned_results.jsonl), sweep sigma
for a FIXED controller at that SAME matched budget on the SAME states, and
compare against the already-known adaptive success for that seed.

Appends one line per seed to mujoco_claim_a_results.jsonl.
"""

import json
import os
import sys

import numpy as np
import torch

import mujoco_benchmark as mb
import mujoco_quality_retuned as mqr

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "mujoco_claim_a_results.jsonl")
SIGMAS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def load_original_results():
    path = os.path.join(os.path.dirname(__file__), "mujoco_quality_retuned_results.jsonl")
    return {r["seed"]: r for r in (json.loads(l) for l in open(path))}


def run_fixed_at_k(model, device, states, k, sigma):
    from generic_mbrl import GenericMPPI
    import mujoco
    import time

    environment = mb.TwoLinkMuJoCo()
    successes = []
    for episode, initial_state in enumerate(states):
        environment.reset(seed=episode)
        environment.data.qpos[:] = initial_state[:2]
        environment.data.qvel[:] = initial_state[2:]
        mujoco.mj_forward(environment.model, environment.data)
        state = environment.state()
        controller = GenericMPPI(
            dynamics=model, action_dim=mb.ACTION_DIM, horizon=mb.HORIZON, rollouts=k,
            action_low=-5.0 * np.ones(mb.ACTION_DIM), action_high=5.0 * np.ones(mb.ACTION_DIM),
            target=np.zeros(mb.STATE_DIM, dtype=np.float32), cost_fn=mb.arm_cost, dt=0.02,
            noise_sigma=sigma, device=device,
        )
        controller.reset()
        for _ in range(mb.STEPS):
            action = controller.command(state)
            state = environment.step(action)
        successes.append(float(np.linalg.norm(state) < 0.5))
    return float(np.mean(successes))


def run_one_seed(seed, device, original_record):
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_environment = mb.TwoLinkMuJoCo()
    train_data = mb.collect_task_directed_data(train_environment, transitions=2000, seed=seed)
    states = mb.make_states(seed + 1000)
    normalizer = mb.make_normalizer(train_data, device)

    torch.manual_seed(seed)
    ensemble = mb.train_ensemble(train_data, device, normalizer)

    eff_k = original_record["adaptive"]["effective_rollouts"]
    k_matched = int(round(eff_k))
    adaptive_success = original_record["adaptive"]["success_rate"]
    ensemble_success = original_record["ensemble"]["success_rate"]

    sigma_results = {}
    for sigma in SIGMAS:
        sigma_results[sigma] = run_fixed_at_k(ensemble, device, states, k_matched, sigma)
    best_sigma = max(sigma_results, key=sigma_results.get)
    best_success = sigma_results[best_sigma]

    return {
        "seed": seed, "k_matched": k_matched, "eff_k": eff_k,
        "adaptive_success": adaptive_success, "ensemble_success_k300": ensemble_success,
        "fixed_matched_sigma_sweep": sigma_results,
        "fixed_matched_best_sigma": best_sigma, "fixed_matched_best_success": best_success,
    }


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [7, 17, 27, 37]
    device = mb.DEVICE
    original = load_original_results()
    print(f"Device: {device}, seeds: {seeds}")

    with open(RESULTS_PATH, "a") as f:
        for seed in seeds:
            record = run_one_seed(seed, device, original[seed])
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"seed={seed:5d} K_matched={record['k_matched']:4d} "
                f"fixed_matched_best={record['fixed_matched_best_success']:.2f} "
                f"(sigma={record['fixed_matched_best_sigma']}) "
                f"adaptive={record['adaptive_success']:.2f}"
            )
    print(f"\nAppended {len(seeds)} seed(s) to {RESULTS_PATH}")
