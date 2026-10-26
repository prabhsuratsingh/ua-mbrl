"""Higher-power re-run of mujoco_benchmark.py's single/ensemble/adaptive
comparison. research_status.tex and README.md both flag the original n=3
seed comparison as too small/high-variance to trust. This script reuses the
exact same pipeline (mujoco_benchmark.py's functions, unchanged) at a larger
seed count, appending one JSON line per completed seed to a results file so
it can be run in several chunks (each ~100s/seed) and aggregated afterward.

Usage: python mujoco_multiseed_study.py <seed1> <seed2> ...
Appends to mujoco_multiseed_results.jsonl (created if absent).
"""

import json
import os
import sys
import time

import numpy as np
import torch

import mujoco_benchmark as mb

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "mujoco_multiseed_results.jsonl")


def run_one_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_environment = mb.TwoLinkMuJoCo()
    train_data = mb.collect_task_directed_data(train_environment, transitions=2000, seed=seed)
    states = mb.make_states(seed + 1000)
    normalizer = mb.make_normalizer(train_data, device)

    pd_success = mb.evaluate_pd(states)

    torch.manual_seed(seed)
    single = mb.train_single(train_data, device, normalizer)
    torch.manual_seed(seed)
    ensemble = mb.train_ensemble(train_data, device, normalizer)
    calibrator = mb.make_controller(ensemble, device, adaptive=True)
    torch.manual_seed(seed + 5000)
    threshold = calibrator.calibrate_threshold(mb.make_states(seed + 2000))

    record = {"seed": seed, "device": str(device), "pd_success": pd_success, "threshold": threshold}
    for name, model, is_adaptive in [
        ("single", single, False),
        ("ensemble", ensemble, False),
        ("adaptive", ensemble, True),
    ]:
        result = mb.evaluate(model, device, states, is_adaptive, threshold)
        record[name] = result
    return record


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]]
    if not seeds:
        raise SystemExit("usage: python mujoco_multiseed_study.py <seed1> <seed2> ...")

    device = mb.DEVICE
    print(f"Device: {device}, seeds this chunk: {seeds}")

    with open(RESULTS_PATH, "a") as f:
        for seed in seeds:
            start = time.perf_counter()
            record = run_one_seed(seed, device)
            elapsed = time.perf_counter() - start
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"seed={seed:5d} ({elapsed:5.1f}s) "
                f"pd={record['pd_success']:.2f} "
                f"single={record['single']['success_rate']:.2f} "
                f"ensemble={record['ensemble']['success_rate']:.2f} "
                f"adaptive={record['adaptive']['success_rate']:.2f}"
            )
    print(f"\nAppended {len(seeds)} seed(s) to {RESULTS_PATH}")
