"""Adaptive-K wall-clock crossover.

Fixed-high runs seed-batched (all S seeds advanced by one batched call);
adaptive-K runs as a sequential per-seed loop, since it needs a different
rollout count per seed. Latency is "cost to advance all S seeds by one
control step" under each approach.

Each K_max should be measured in a fresh process: sweeping several K_max in
one process was found to produce discontinuous latencies from CUDA allocator
fragmentation (e.g. K=16000 at 854.71 ms in one run, 12931.50 ms in another).
``crossover_sweep`` therefore launches one ``crossover_point`` subprocess per
K_max by default (``--mode inprocess`` reproduces the original single-process
sweep, adaptive_k_crossover_study.py / adaptive_k_crossover_fine.py).
"""

import json
import subprocess
import sys

import numpy as np

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval import pendulum as runners
from ua_mbrl.utils.results import read_jsonl

DEFAULT_K_VALUES = (100, 300, 1000, 3000, 8000, 12000, 13000, 14000, 15000, 16000, 20000, 26000)


def _common_arguments(parser):
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--k-min", type=int, default=pend.ROLLOUTS)


def add_point_arguments(parser):
    parser.add_argument("--k-max", type=int, default=16000)
    parser.add_argument("--repeat", type=int, default=0, help="seed offset: fixed 1000+r, adaptive 2000+r")
    _common_arguments(parser)


def add_sweep_arguments(parser):
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--mode", choices=("fresh", "inprocess"), default="fresh")
    parser.add_argument("--repeats", type=int, default=1)
    _common_arguments(parser)


def measure(ensemble, calibration_states, states, k_max, threshold, eval_steps, repeat=0, k_min=pend.ROLLOUTS):
    fixed = runners.run_fixed_batched(ensemble, states, k_max, eval_steps, seed=1000 + repeat, timed=True)
    adaptive = runners.run_adaptive_sequential(
        ensemble, states, k_max, threshold, eval_steps, seed=2000 + repeat, k_min=k_min, timed=True
    )
    fixed_ms = float(np.mean(fixed["step_latencies"]) * 1000)
    adaptive_ms = float(np.sum(adaptive["mean_step_latency_per_seed"]) * 1000)
    return {
        "k_max": k_max,
        "repeat": repeat,
        "fixed_ms": fixed_ms,
        "adaptive_ms": adaptive_ms,
        "ratio": fixed_ms / adaptive_ms,
        "threshold": threshold,
        "fixed_success": float(fixed["per_seed"].mean()),
        "adaptive_success": float(adaptive["per_seed"].mean()),
        "adaptive_eff_k": adaptive["eff_k"],
        "adaptive_escalation_rate": adaptive["escalation_rate"],
        "fixed_per_seed_latency_ms": fixed_ms / len(states),
        "adaptive_per_seed_latency_ms": float(np.mean(adaptive["mean_step_latency_per_seed"]) * 1000),
    }


def _setup(args):
    ensemble = pend.train_default_ensemble()
    calibration_states = pend.make_states(pend.CALIBRATION_STATES_SEED, 40)
    states = pend.make_states(pend.EVAL_STATES_SEED, args.num_seeds)
    return ensemble, calibration_states, states


def run_point(args, run_dir):
    ensemble, calibration_states, states = _setup(args)
    threshold = pend.calibrate_threshold(ensemble, args.k_max, calibration_states, k_min=args.k_min)
    record = measure(
        ensemble, calibration_states, states, args.k_max, threshold, args.eval_steps, args.repeat, args.k_min
    )
    run_dir.append(record)
    print(
        f"K_max={args.k_max:6d}  fixed={record['fixed_ms']:9.2f}ms  "
        f"adaptive={record['adaptive_ms']:9.2f}ms  ratio={record['ratio']:.3f}"
    )
    return record


def run_sweep(args, run_dir):
    if args.mode == "inprocess":
        ensemble, calibration_states, states = _setup(args)
        for k_max in args.k_values:
            threshold = pend.calibrate_threshold(ensemble, k_max, calibration_states, k_min=args.k_min)
            for repeat in range(args.repeats):
                record = measure(
                    ensemble, calibration_states, states, k_max, threshold, args.eval_steps, repeat, args.k_min
                )
                run_dir.append(record)
                print(f"K_max={k_max:6d} r={repeat} ratio={record['ratio']:.3f} "
                      f"fixed_success={record['fixed_success']:.3f} adaptive_success={record['adaptive_success']:.3f}")
    else:
        for k_max in args.k_values:
            for repeat in range(args.repeats):
                point_dir = run_dir.path / "points" / f"k{k_max}_r{repeat}"
                command = [
                    sys.executable, "-m", "ua_mbrl", "run", "crossover_point",
                    "--k-max", str(k_max), "--repeat", str(repeat),
                    "--num-seeds", str(args.num_seeds), "--eval-steps", str(args.eval_steps),
                    "--k-min", str(args.k_min), "--out-dir", str(point_dir),
                ]
                subprocess.run(command, check=True)
                for record in read_jsonl(point_dir / "records.jsonl"):
                    run_dir.append(record)

    analysis = analyze_crossover(run_dir.records())
    print_analysis(analysis)
    (run_dir.path / "analysis.json").write_text(json.dumps(analysis, indent=2))
    return analysis


def analyze_crossover(records, regression_min_k=1000):
    """Locate the fixed/adaptive latency crossover two ways.

    * interpolation: linear interpolation of the (repeat-averaged) ratio
      between the first bracketing pair of K_max values with ratio crossing 1;
    * regression: OLS of fixed_ms on K_max for K_max >= regression_min_k,
      intersected with the mean adaptive_ms over the same points (adaptive
      latency is near-constant in K_max), i.e. eq:fit / eq:kstar.
    """
    by_k = {}
    for r in records:
        by_k.setdefault(int(r["k_max"]), []).append(r)
    ks = sorted(by_k)
    ratio = {k: float(np.mean([r["ratio"] for r in by_k[k]])) for k in ks}
    fixed = {k: float(np.mean([r["fixed_ms"] for r in by_k[k]])) for k in ks}
    adaptive = {k: float(np.mean([r["adaptive_ms"] for r in by_k[k]])) for k in ks}

    interpolated = None
    for k0, k1 in zip(ks, ks[1:]):
        if ratio[k0] < 1.0 <= ratio[k1]:
            interpolated = k0 + (1.0 - ratio[k0]) / (ratio[k1] - ratio[k0]) * (k1 - k0)
            break

    fit_ks = [k for k in ks if k >= regression_min_k]
    regression = None
    if len(fit_ks) >= 2:
        x = np.asarray(fit_ks, dtype=np.float64)
        y = np.asarray([fixed[k] for k in fit_ks])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        r_squared = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
        mean_adaptive = float(np.mean([adaptive[k] for k in fit_ks]))
        regression = {
            "slope_ms_per_rollout": float(slope),
            "intercept_ms": float(intercept),
            "r_squared": float(r_squared),
            "mean_adaptive_ms": mean_adaptive,
            "k_star": float((mean_adaptive - intercept) / slope),
            "n_points": len(fit_ks),
        }
    return {
        "k_values": ks,
        "ratio": ratio,
        "fixed_ms": fixed,
        "adaptive_ms": adaptive,
        "interpolated_crossover": interpolated,
        "regression": regression,
    }


def print_analysis(analysis):
    print(f"\n{'K_max':>7} {'fixed_ms':>10} {'adaptive_ms':>12} {'ratio':>7}")
    for k in analysis["k_values"]:
        print(f"{k:7d} {analysis['fixed_ms'][k]:10.2f} {analysis['adaptive_ms'][k]:12.2f} {analysis['ratio'][k]:7.3f}")
    if analysis["interpolated_crossover"] is not None:
        print(f"Interpolated crossover: K_max ~= {analysis['interpolated_crossover']:.0f}")
    else:
        print("Interpolated crossover: ratio never crosses 1.0 in the measured range")
    reg = analysis["regression"]
    if reg:
        print(
            f"Regression: fixed_ms = {reg['intercept_ms']:.2f} + {reg['slope_ms_per_rollout']:.5f} K "
            f"(R^2={reg['r_squared']:.3f}, n={reg['n_points']}); K* = {reg['k_star']:.0f}"
        )
