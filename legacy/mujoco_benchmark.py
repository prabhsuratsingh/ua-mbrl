"""Benchmark single-model versus ensemble GPU MPPI in MuJoCo."""

import time

import mujoco
import numpy as np
import torch

from generic_mbrl import (
    AffineInputNormalizer,
    AdaptiveGenericMPPI,
    GenericEnsembleDynamics,
    GenericDynamicsModel,
    GenericMPPI,
)
from mujoco_generic_experiment import (
    ACTION_DIM,
    DEVICE,
    STATE_DIM,
    TwoLinkMuJoCo,
    arm_cost,
)

TRAIN_EPOCHS = 75
EPISODES = 3
SEEDS = (7, 17, 27)
STEPS = 75
HORIZON = 50
ROLLOUTS = 300


def make_normalizer(data, device):
    return AffineInputNormalizer(
        np.mean(data["x"], axis=0),
        np.std(data["x"], axis=0),
        np.mean(data["u"], axis=0),
        np.std(data["u"], axis=0),
        device,
    )


def collect_task_directed_data(environment, transitions=2000, seed=7):
    """Collect stabilizing demonstrations plus exploration noise."""
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    state = environment.reset(seed)
    for step in range(transitions):
        if step % 100 == 0:
            state = environment.reset(seed + step)
        position = state[:2]
        velocity = state[2:]
        action = -3.0 * position - 1.0 * velocity
        action += rng.normal(0.0, 0.5, size=ACTION_DIM)
        action = np.clip(action, -5.0, 5.0).astype(np.float32)
        next_state = environment.step(action)
        states.append(state.copy())
        actions.append(action.copy())
        next_states.append(next_state.copy())
        state = next_state
    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


def train_single(data, device, normalizer):
    model = GenericDynamicsModel(
        STATE_DIM,
        ACTION_DIM,
        hidden_dim=128,
        input_normalizer=normalizer,
        device=device,
    )
    model.train_model(data, epochs=TRAIN_EPOCHS, verbose=False)
    return model


def train_ensemble(data, device, normalizer):
    model = GenericEnsembleDynamics(
        STATE_DIM,
        ACTION_DIM,
        ensemble_size=5,
        hidden_dim=128,
        input_normalizer=normalizer,
        device=device,
    )
    model.train_ensemble(data, epochs=TRAIN_EPOCHS, seed=7, verbose=False)
    return model


def make_controller(
    model,
    device,
    adaptive=False,
    threshold=0.0,
    pilot_rollouts=8,
    pilot_horizon=2,
):
    controller_type = AdaptiveGenericMPPI if adaptive else GenericMPPI
    adaptive_args = {
        "min_rollouts": 100,
        "max_rollouts": ROLLOUTS,
        "uncertainty_threshold": threshold,
        "pilot_rollouts": pilot_rollouts,
        "pilot_horizon": pilot_horizon,
    } if adaptive else {}
    controller_kwargs = {
        "dynamics": model,
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
        "action_low": -5.0 * np.ones(ACTION_DIM),
        "action_high": 5.0 * np.ones(ACTION_DIM),
        "target": np.zeros(STATE_DIM, dtype=np.float32),
        "cost_fn": arm_cost,
        "dt": 0.02,
        "device": device,
    }
    if not adaptive:
        controller_kwargs["rollouts"] = ROLLOUTS
    return controller_type(
        **controller_kwargs,
        **adaptive_args,
    )


def evaluate(
    model,
    device,
    states,
    adaptive=False,
    threshold=0.0,
    pilot_rollouts=8,
    pilot_horizon=2,
):
    environment = TwoLinkMuJoCo()
    controller = make_controller(
        model,
        device,
        adaptive,
        threshold,
        pilot_rollouts,
        pilot_horizon,
    )
    successes = []
    latencies = []
    effective_rollouts = []

    for episode, initial_state in enumerate(states):
        environment.reset(seed=episode)
        environment.data.qpos[:] = initial_state[:2]
        environment.data.qvel[:] = initial_state[2:]
        mujoco.mj_forward(environment.model, environment.data)
        state = environment.state()
        controller.reset()

        start = time.perf_counter()
        for _ in range(STEPS):
            command_start = time.perf_counter()
            action = controller.command(state)
            latencies.append(time.perf_counter() - command_start)
            effective = getattr(controller, "last_rollouts", controller.rollouts)
            if adaptive:
                effective += controller.pilot_rollouts * controller.pilot_horizon
            effective_rollouts.append(effective)
            state = environment.step(action)
        _ = time.perf_counter() - start
        successes.append(float(np.linalg.norm(state) < 0.5))

    return {
        "success_rate": float(np.mean(successes)),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
        "p95_latency_ms": 1000.0 * float(np.percentile(latencies, 95)),
        "effective_rollouts": float(np.mean(effective_rollouts)),
    }


def evaluate_pd(states):
    environment = TwoLinkMuJoCo()
    successes = []
    for episode, initial_state in enumerate(states):
        environment.reset(seed=episode)
        environment.data.qpos[:] = initial_state[:2]
        environment.data.qvel[:] = initial_state[2:]
        mujoco.mj_forward(environment.model, environment.data)
        state = environment.state()
        for _ in range(STEPS):
            action = np.clip(
                -3.0 * state[:2] - 1.0 * state[2:],
                -5.0,
                5.0,
            )
            state = environment.step(action)
        successes.append(float(np.linalg.norm(state) < 0.5))
    return float(np.mean(successes))


def make_states(seed):
    rng = np.random.default_rng(seed)
    return np.asarray([
        np.concatenate([
            rng.uniform(-0.5, 0.5, size=2),
            rng.uniform(-0.2, 0.2, size=2),
        ])
        for _ in range(EPISODES)
    ], dtype=np.float32)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)
    if len(values) < 2:
        return mean, 0.0
    interval = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, interval


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    for device in [torch.device("cpu"), DEVICE]:
        print(f"\nTraining/evaluating on {device}")
        records = {"pd": [], "single": [], "ensemble": [], "adaptive": []}
        latencies = {"single": [], "ensemble": [], "adaptive": []}
        compute = {"single": [], "ensemble": [], "adaptive": []}
        for seed in SEEDS:
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_environment = TwoLinkMuJoCo()
            train_data = collect_task_directed_data(
                train_environment,
                transitions=2000,
                seed=seed,
            )
            states = make_states(seed + 1000)
            records["pd"].append(evaluate_pd(states))
            normalizer = make_normalizer(train_data, device)

            torch.manual_seed(seed)
            single = train_single(train_data, device, normalizer)
            torch.manual_seed(seed)
            ensemble = train_ensemble(train_data, device, normalizer)
            calibrator = make_controller(ensemble, device, adaptive=True)
            torch.manual_seed(seed + 5000)
            threshold = calibrator.calibrate_threshold(make_states(seed + 2000))

            for name, model, is_adaptive in [
                ("single", single, False),
                ("ensemble", ensemble, False),
                ("adaptive", ensemble, True),
            ]:
                result = evaluate(model, device, states, is_adaptive, threshold)
                records[name].append(result["success_rate"])
                latencies[name].append(result["latency_ms"])
                compute[name].append(result["effective_rollouts"])

        print("method     success_95ci    latency_ms_95ci    effective_K")
        for name in ("pd", "single", "ensemble", "adaptive"):
            success_mean, success_ci = summarize(records[name])
            if name == "pd":
                print(f"{name:<10} {100 * success_mean:.1f}% +/- {100 * success_ci:.1f}%")
                continue
            latency_mean, latency_ci = summarize(latencies[name])
            compute_mean, compute_ci = summarize(compute[name])
            print(
                f"{name:<10} {100 * success_mean:.1f}% +/- {100 * success_ci:.1f}% "
                f"{latency_mean:.3f} +/- {latency_ci:.3f} ms "
                f"{compute_mean:.0f} +/- {compute_ci:.0f}"
            )
