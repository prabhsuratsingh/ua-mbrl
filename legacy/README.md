# Legacy scripts (frozen)

The original flat experiment scripts, kept unchanged. They are not part of the
package. `tests/test_equivalence.py` imports them, to check that the refactored
`ua_mbrl` package produces exactly the same numbers. Use
`uv run ua-mbrl run <experiment>` for new runs. `uv run ua-mbrl list` maps each
experiment to the paper result it produces.

Not ported (not used by the paper): `mppi_pendulum_learning*.py`,
`ensemble_dynamics.py`, `ensemble_mppi.py`, `experiment_ensemble_uncertainty.py`,
`validation_ensemble_mppi.py`, `multiseed_adaptive_benchmark.py`,
`adaptive_cuda_study.py`, `adaptive_k_crossover_fine.py` (superseded by
`crossover_sweep --mode inprocess --repeats 2`).
