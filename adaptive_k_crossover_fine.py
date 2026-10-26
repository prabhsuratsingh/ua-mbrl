"""Fine localization of the adaptive-K/fixed-high crossover point.

adaptive_k_crossover_study.py located the crossover between K_max=12,000
(ratio 0.83) and K_max=16,000 (ratio 1.11). This script sweeps K_max in
1,000-rollout steps across that interval, with 3 repeated timing trials per
point (latency is a systems measurement, not confounded by RNG realization
-- only success-rate comparisons need matched seeding, per
adaptive_k_confound_check.py), to narrow the boundary.
"""

import numpy as np
import torch

from adaptive_k_crossover_study import (
    NUM_SEEDS,
    EVAL_STEPS,
    calibrate_threshold,
    make_initial_states,
    run_adaptive_sequential,
    run_fixed_high_batched,
)
from robustness_mismatch_experiment import collect_data, train_models

K_MAX_FINE = (12000, 13000, 14000, 15000, 16000)
REPEATS = 2


if __name__ == "__main__":
    _, ensemble = train_models(collect_data())
    calibration_states = make_initial_states(900, 40)
    initial_states = make_initial_states(800, NUM_SEEDS)

    print(f"{'K_max':>7} {'fixed_ms':>10} {'adaptive_ms':>12} {'ratio':>8} {'ratio_ci':>9}")
    print("-" * 60)
    rows = []
    for k_max in K_MAX_FINE:
        threshold = calibrate_threshold(ensemble, k_max, calibration_states)
        ratios = []
        fixed_ms_list = []
        adaptive_ms_list = []
        for repeat in range(REPEATS):
            fixed = run_fixed_high_batched(
                ensemble, initial_states, k_max, EVAL_STEPS, seed=1000 + repeat
            )
            adaptive = run_adaptive_sequential(
                ensemble, initial_states, k_max, threshold, EVAL_STEPS, seed=2000 + repeat
            )
            ratios.append(fixed["total_latency_ms"] / adaptive["total_latency_ms"])
            fixed_ms_list.append(fixed["total_latency_ms"])
            adaptive_ms_list.append(adaptive["total_latency_ms"])

        ratio_mean = np.mean(ratios)
        ratio_ci = 1.96 * np.std(ratios, ddof=1) / np.sqrt(len(ratios)) if len(ratios) > 1 else 0.0
        print(
            f"{k_max:7d} {np.mean(fixed_ms_list):10.2f} {np.mean(adaptive_ms_list):12.2f} "
            f"{ratio_mean:8.3f} {ratio_ci:9.3f}"
        )
        rows.append((k_max, ratio_mean, ratio_ci))

    print("\nCrossover (ratio=1.0) location:")
    for i in range(len(rows) - 1):
        k0, r0, _ = rows[i]
        k1, r1, _ = rows[i + 1]
        if r0 < 1.0 <= r1:
            # linear interpolation between the two bracketing points
            frac = (1.0 - r0) / (r1 - r0)
            k_cross = k0 + frac * (k1 - k0)
            print(f"Between K={k0} (ratio={r0:.3f}) and K={k1} (ratio={r1:.3f})")
            print(f"Linear-interpolated crossover: K_max ~= {k_cross:.0f}")
