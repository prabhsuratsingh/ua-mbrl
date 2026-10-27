"""# UA-MBRL

Uncertainty-aware GPU model-based MPPI experiments for a learned pendulum.

## Current Research Result

Bootstrap ensembles outperform a single learned dynamics model for MPPI control
and are more robust to mass and gravity mismatch. Directly adding ensemble
variance to the MPPI cost has not yet improved control reliably.

The strongest validated contribution is **GPU-efficient uncertainty-aware
MPPI**. It combines bootstrap ensemble dynamics, severe-OOD detection, and
batched ensemble inference. Severe OOD states/actions produced approximately
a 4x increase in ensemble disagreement and ROC AUC 1.0, while mild OOD
produced only AUC 0.63. Disagreement is therefore treated as a severe-shift
detector, not universally calibrated uncertainty or a raw risk penalty.

The adaptive controller now calibrates its threshold from held-out
in-distribution pilot rollouts at the 95th percentile. This keeps nominal
compute near the low rollout budget, but current screening shows that it can
miss moderate dynamics shifts. That limitation is part of the research
hypothesis: the gate is intended for severe OOD detection and must be tested
with explicit detection coverage, false-alarm rate, and effective compute.
Calibration states and evaluation states are kept separate, and effective
compute includes both pilot and full-planning rollouts.

## Experiments

```powershell
uv run python experiment_ensemble_uncertainty.py
uv run python validation_ensemble_mppi.py
uv run python robustness_mismatch_experiment.py
uv run python ood_uncertainty_experiment.py
```

MuJoCo smoke test:

```powershell
uv run python mujoco_generic_experiment.py
```

The main experimental modules are `ensemble_dynamics.py`, `ensemble_mppi.py`,
and `ood_uncertainty_experiment.py`.

## General Simulation API

The dimension-agnostic primitives are in `generic_mbrl.py`. They support
arbitrary state and action dimensions and accept simulator-specific hooks for:

- feature encoding: `feature_fn(x, u)`
- residual targets: `residual_fn(x_next, x)`
- state reconstruction: `transition_fn(x, delta)`
- task cost: `cost_fn(x, u, target)`

This is the intended path to MuJoCo and other simulators. A simulator adapter
should own observation/action conversion, quaternion or angle residuals,
termination handling, and task cost. The learned ensemble and MPPI controller
should remain unchanged across environments.

The generic path has been smoke-tested with a 6-dimensional state and
2-dimensional action vector.

It has also been validated against a MuJoCo two-link arm with a 4-dimensional
`qpos/qvel` observation and 2-dimensional torque action. The adapter collected
1,000 simulator transitions, trained a five-model ensemble, and executed
20-step GPU MPPI with 500 rollouts.

The first comparative MuJoCo benchmark exposed an important limitation. With
2,000 random transitions and 50 control steps, the single model achieved 0%
success on both CPU and CUDA. The ensemble achieved 20% on CPU and 0% on
CUDA. This confirms the generic execution path, but not yet MuJoCo control
quality. The next required algorithmic step is state/action normalization and
task-directed data coverage before making cross-simulator performance claims.

State/action normalization, task-directed replay, and a stabilized MuJoCo
actuation model are now included. The corrected benchmark has a `100%` PD
reference success rate. Learned control reached `40%` single-model and `60%`
ensemble success on CPU, and `0%` single-model and `40%` ensemble success on
CUDA. This is the first valid non-pendulum control result, but device and seed
variance require a multi-seed benchmark before publication claims.

The completed three-seed follow-up reports CUDA success of `33.3% +/- 0%` for
the single model and `55.6% +/- 21.8%` for the ensemble. CPU success was
`55.6% +/- 21.8%` for both. Mean CUDA command latency was `121.8 ms` for the
single model and `344.9 ms` for the ensemble. The ensemble improvement is
promising on CUDA, but it carries a substantial latency cost and is not yet a
universal cross-device result.

The ensemble inference path is now vectorized across members and caches its
stacked parameters. In the same paired benchmark, this preserved CUDA ensemble
success at `55.6% +/- 21.8%` while reducing mean CUDA command latency to
`94.5 ms`, approximately a `3.6x` reduction from the uncached path. The exact
prediction equivalence check reports zero maximum delta difference.

A controlled loop-versus-batched CUDA ablation at `E=5`, `K=1000`, and `N=10`
reduced latency from `23.65 ms` to `12.37 ms` and increased throughput from
`2.11M` to `4.04M` model evaluations per second. The CPU path did not improve
consistently, so this is specifically a GPU ensemble-inference contribution.

The generic calibrated adaptive-$K$ benchmark produced a mixed but useful
result. On CPU, adaptive MPPI reached `66.7% +/- 37.7%` success versus
`55.6% +/- 57.6%` for fixed ensemble MPPI, while latency fell from `139.3 ms`
to `96.7 ms`. On CUDA, both methods reached `55.6%` success, but adaptive
latency increased from `95.9 ms` to `114.1 ms` because pilot overhead exceeded
the saved full rollouts. This identifies the next systems target: fuse or
reduce pilot inference on GPU before claiming adaptive compute savings.

Adaptive-$K$ experiments are retained as future work. A focused single-seed
CUDA ablation was encouraging, but the paired multi-seed study did not show a
reliable latency benefit.

The paired three-seed follow-up did not validate a universal adaptive-$K$
advantage. With the `8x2` pilot, fixed and adaptive success were both
`33.3% +/- 65.3%`; adaptive effective rollout work fell to `265.3 +/- 42.2`,
but latency increased from `74.4 +/- 3.4 ms` to `88.5 +/- 9.3 ms`. A minimal
`1x1` pilot reduced work further but produced `0%` success in the same screen.
Adaptive-$K$ is therefore retained as future work, not the primary claim.

## GPU Scaling

Run the generic systems benchmark with MuJoCo-like dimensions:

```powershell
uv run python gpu_mppi_benchmark.py --state-dim 13 --action-dim 4
```

The benchmark reports command latency, total model evaluations, and model
evaluations per second for different ensemble sizes and rollout counts. In a
CUDA smoke run with `E=5`, `K=1000`, and `N=15`, the generic controller
processed approximately 1.72 million model evaluations per second, compared
with approximately 0.87 million on CPU. These numbers are hardware-specific;
the benchmark is the reproducible result, not the absolute throughput claim.
