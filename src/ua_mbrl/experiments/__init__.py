"""Experiment registry.

Each experiment is a function ``run(args, run_dir)`` plus an
``add_arguments(parser)`` hook, registered here under a CLI name. Run one
with ``ua-mbrl run <name> [options]``; list them with ``ua-mbrl list``.
"""

import argparse
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    module: str
    summary: str
    paper: str
    run: str = "run"
    add_arguments: str = "add_arguments"

    def load(self):
        module = importlib.import_module(f"ua_mbrl.experiments.{self.module}")
        return getattr(module, self.add_arguments), getattr(module, self.run)


REGISTRY = {
    e.name: e
    for e in [
        Experiment(
            "crossover_point", "crossover", "Fixed-high vs adaptive latency at ONE K_max (run in a fresh process)",
            "tab:crossover, fig2, eq:kstar (one point)",
            run="run_point", add_arguments="add_point_arguments",
        ),
        Experiment(
            "crossover_sweep", "crossover", "Latency crossover over a K_max list, one fresh process per K_max (or in-process)",
            "tab:crossover, fig2, eq:fit/eq:kstar",
            run="run_sweep", add_arguments="add_sweep_arguments",
        ),
        Experiment(
            "confound_check", "crossover_quality", "Fixed-high vs adaptive success, both sequential with matched RNG (sigma=1.0)",
            "tab:confound (post pilot-RNG fix)",
            run="run_confound", add_arguments="add_confound_arguments",
        ),
        Experiment(
            "crossover_quality", "crossover_quality", "Matched-ceiling quality comparison, uniform sigma=1.0 vs per-K retuned sigma",
            "tab:confoundfixed",
        ),
        Experiment(
            "claim_a", "claim_a", "Matched-sample-budget Claim A: ceiling/matched sigma sweeps, adaptive, paired sign test",
            "sec:claima tables (aggregate, paired, Wilson)",
        ),
        Experiment(
            "claim_a_stratified", "claim_a_stratified", "ESS vs pilot-uncertainty terciles + condition-stratified success",
            "sec:morekhurts first observation (n=15)",
        ),
        Experiment(
            "fallback", "fallback", "Severe-OOD fallback controller under mismatch / state disturbance / narrow ensemble",
            "sec:fallback",
        ),
        Experiment(
            "ood_severity", "ood", "OOD detection AUC/TPR/FPR vs shift magnitude",
            "fig1, sec:ood",
            run="run_severity", add_arguments="add_severity_arguments",
        ),
        Experiment(
            "ood_two_point", "ood", "Two-point (mild/severe) OOD uncertainty experiment",
            "README (not in paper)",
            run="run_two_point", add_arguments="add_two_point_arguments",
        ),
        Experiment(
            "robustness", "robustness", "Single/ensemble/adaptive MPPI under mass/gravity mismatch (3 episodes)",
            "sec:seedbug context (not tabulated)",
        ),
        Experiment(
            "mujoco_regulation", "mujoco_regulation", "MuJoCo two-link regulation: PD/single/ensemble/adaptive per seed",
            "tab:mujoco regulation (sigma=1.0), sec:mujoco retuned paragraph (sigma=0.2)",
        ),
        Experiment(
            "mujoco_claim_a", "mujoco_claim_a", "MuJoCo matched-budget fixed sigma sweep vs adaptive, per seed",
            "sec:claima MuJoCo replication",
        ),
        Experiment(
            "reacher", "reacher", "MuJoCo reacher: single/ensemble/adaptive per seed",
            "tab:mujoco reaching, fig4",
        ),
        Experiment(
            "gpu_benchmark", "gpu_benchmark", "Ensemble MPPI latency: looped vs batched, CPU vs CUDA",
            "tab:gpuspeedups row 1",
        ),
        Experiment(
            "mujoco_smoke", "mujoco_regulation", "10-step MuJoCo adapter smoke test",
            "none (smoke test)",
            run="run_smoke", add_arguments="add_smoke_arguments",
        ),
    ]
}


def build_args(name, **overrides):
    """Default argparse Namespace for an experiment, with keyword overrides
    (used by tests to run experiments at reduced scale)."""
    add_arguments, _ = REGISTRY[name].load()
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"{name} has no argument {key!r}")
        setattr(args, key, value)
    return args
