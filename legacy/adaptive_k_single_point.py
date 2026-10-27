"""Measure the fixed-high vs adaptive latency ratio at ONE K_max value, in a
fresh process/CUDA context.

adaptive_k_crossover_fine.py swept several K_max values sequentially within
one process and got an inconsistent, discontinuous result (K=16000 measured
at 854.71ms in one run, 12931.50ms in another run reaching it via a different
sequence of prior K values) -- almost certainly CUDA allocator
fragmentation/caching building up across iterations within a single process,
not a real hardware ceiling. Running one K_max per fresh process eliminates
that confound.

Usage: python adaptive_k_single_point.py <k_max>
Appends one line to adaptive_k_fine_results.jsonl.
"""

import json
import os
import sys

from adaptive_k_crossover_study import (
    NUM_SEEDS,
    EVAL_STEPS,
    calibrate_threshold,
    make_initial_states,
    run_adaptive_sequential,
    run_fixed_high_batched,
)
from robustness_mismatch_experiment import collect_data, train_models

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "adaptive_k_fine_results.jsonl")


if __name__ == "__main__":
    k_max = int(sys.argv[1])

    _, ensemble = train_models(collect_data())
    calibration_states = make_initial_states(900, 40)
    initial_states = make_initial_states(800, NUM_SEEDS)

    threshold = calibrate_threshold(ensemble, k_max, calibration_states)
    fixed = run_fixed_high_batched(ensemble, initial_states, k_max, EVAL_STEPS, seed=1000)
    adaptive = run_adaptive_sequential(ensemble, initial_states, k_max, threshold, EVAL_STEPS, seed=2000)
    ratio = fixed["total_latency_ms"] / adaptive["total_latency_ms"]

    record = {"k_max": k_max, "fixed_ms": fixed["total_latency_ms"], "adaptive_ms": adaptive["total_latency_ms"], "ratio": ratio}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"K_max={k_max:6d}  fixed={fixed['total_latency_ms']:9.2f}ms  adaptive={adaptive['total_latency_ms']:9.2f}ms  ratio={ratio:.3f}")
