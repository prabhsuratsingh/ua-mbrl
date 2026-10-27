"""Reacher-style task on the existing two-link MuJoCo arm.

mujoco_benchmark.py's TwoLinkMuJoCo task was "regulate qpos/qvel to zero" --
not yet a standard non-trivial benchmark (research_status.tex's next-steps
list explicitly asked for Reacher or Hopper). The existing arm is already a
gravity-free planar 2-link manipulator, which is exactly Gymnasium Reacher's
shape; it was only missing a target and an end-effector distance cost. This
adds both, reusing the same environment/model/training infrastructure as
mujoco_benchmark.py (same ensemble, GenericMPPI, seeding conventions) so the
single-vs-ensemble-vs-adaptive comparison directly extends to a real reaching
task instead of only regulation-to-origin.
"""

import time

import mujoco
import numpy as np
import torch

from generic_mbrl import AdaptiveGenericMPPI, GenericDynamicsModel, GenericEnsembleDynamics, GenericMPPI
from mujoco_generic_experiment import ACTION_DIM, DEVICE, STATE_DIM, MODEL_XML

LINK1_LENGTH = 0.5
LINK2_LENGTH = 0.5
TARGET_RADIUS_RANGE = (0.15, 0.35)  # offset from the arm's own start position
HORIZON = 30
ROLLOUTS = 300
STEPS = 150
NOISE_SIGMA = 1.5
SUCCESS_THRESHOLD = 0.15  # tuned to this arm's achievable precision, not an arbitrary 0.1
EPISODES = 6
SEEDS = tuple(range(7, 7 + 10 * 10, 10))[:10]  # 7,17,...,97 (10 seeds)
TRAIN_EPOCHS = 75
JACOBIAN_KP = 60.0
JACOBIAN_KD = 5.0


class ReacherMuJoCo:
    """TwoLinkMuJoCo plus a sampled Cartesian target the arm must reach."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_low = np.full(ACTION_DIM, -5.0, dtype=np.float32)
        self.action_high = np.full(ACTION_DIM, 5.0, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = rng.uniform(-0.2, 0.2, size=2)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Target is a small offset from wherever the arm actually starts (near
        # full extension, ee at Cartesian distance ~1.0 from the shoulder) --
        # NOT sampled near the origin, which would require drastic
        # reconfiguration regardless of the "radius" chosen and made every
        # episode equally (and needlessly) hard.
        start_ee = self.end_effector()
        radius = rng.uniform(*TARGET_RADIUS_RANGE)
        angle = rng.uniform(-np.pi, np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        self.target = start_ee + offset
        return self.state()

    def state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]).astype(np.float32)

    def end_effector(self):
        theta1, theta2 = self.data.qpos[0], self.data.qpos[1]
        x = LINK1_LENGTH * np.cos(theta1) + LINK2_LENGTH * np.cos(theta1 + theta2)
        y = LINK1_LENGTH * np.sin(theta1) + LINK2_LENGTH * np.sin(theta1 + theta2)
        return np.array([x, y], dtype=np.float32)

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        mujoco.mj_step(self.model, self.data)
        return self.state()


def forward_kinematics_torch(theta1, theta2):
    x = LINK1_LENGTH * torch.cos(theta1) + LINK2_LENGTH * torch.cos(theta1 + theta2)
    y = LINK1_LENGTH * torch.sin(theta1) + LINK2_LENGTH * torch.sin(theta1 + theta2)
    return x, y


def reacher_cost(x, u, target):
    """target: [target_x, target_y, 0, 0] -- only the first two entries are used."""
    theta1 = x[:, 0]
    theta2 = x[:, 1]
    ee_x, ee_y = forward_kinematics_torch(theta1, theta2)
    distance_sq = (ee_x - target[0]).square() + (ee_y - target[1]).square()
    velocity_penalty = 0.01 * x[:, 2:4].square().sum(dim=1)
    control_penalty = 0.01 * u.square().sum(dim=1)
    return distance_sq + velocity_penalty + control_penalty


def _jacobian(theta1, theta2):
    dex_dt1 = -LINK1_LENGTH * np.sin(theta1) - LINK2_LENGTH * np.sin(theta1 + theta2)
    dex_dt2 = -LINK2_LENGTH * np.sin(theta1 + theta2)
    dey_dt1 = LINK1_LENGTH * np.cos(theta1) + LINK2_LENGTH * np.cos(theta1 + theta2)
    dey_dt2 = LINK2_LENGTH * np.cos(theta1 + theta2)
    return np.array([[dex_dt1, dex_dt2], [dey_dt1, dey_dt2]])


def collect_task_directed_data(environment, transitions=2000, seed=7):
    """Jacobian-transpose reaching toward randomly resampled targets, plus
    exploration noise -- gives the dynamics model coverage of actual reaching
    motions across the workspace, not just origin-stabilization data."""
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    state = environment.reset(seed)
    for step in range(transitions):
        if step % 100 == 0:
            state = environment.reset(seed + step)
        theta1, theta2 = state[0], state[1]
        qvel = state[2:4]
        ee_x = LINK1_LENGTH * np.cos(theta1) + LINK2_LENGTH * np.cos(theta1 + theta2)
        ee_y = LINK1_LENGTH * np.sin(theta1) + LINK2_LENGTH * np.sin(theta1 + theta2)
        error = environment.target - np.array([ee_x, ee_y])
        jac = _jacobian(theta1, theta2)
        action = jac.T @ error * JACOBIAN_KP - JACOBIAN_KD * qvel
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


def make_normalizer(data, device):
    from generic_mbrl import AffineInputNormalizer
    return AffineInputNormalizer(
        np.mean(data["x"], axis=0), np.std(data["x"], axis=0),
        np.mean(data["u"], axis=0), np.std(data["u"], axis=0), device,
    )


def train_single(data, device, normalizer):
    model = GenericDynamicsModel(STATE_DIM, ACTION_DIM, hidden_dim=128, input_normalizer=normalizer, device=device)
    model.train_model(data, epochs=TRAIN_EPOCHS, verbose=False)
    return model


def train_ensemble(data, device, normalizer):
    model = GenericEnsembleDynamics(STATE_DIM, ACTION_DIM, ensemble_size=5, hidden_dim=128, input_normalizer=normalizer, device=device)
    model.train_ensemble(data, epochs=TRAIN_EPOCHS, seed=7, verbose=False)
    return model


def make_controller(model, device, target, adaptive=False, threshold=0.0, pilot_rollouts=8, pilot_horizon=2):
    controller_type = AdaptiveGenericMPPI if adaptive else GenericMPPI
    adaptive_args = {
        "min_rollouts": 100, "max_rollouts": ROLLOUTS, "uncertainty_threshold": threshold,
        "pilot_rollouts": pilot_rollouts, "pilot_horizon": pilot_horizon,
    } if adaptive else {}
    controller_kwargs = {
        "dynamics": model, "action_dim": ACTION_DIM, "horizon": HORIZON,
        "action_low": -5.0 * np.ones(ACTION_DIM), "action_high": 5.0 * np.ones(ACTION_DIM),
        "target": np.array([target[0], target[1], 0.0, 0.0], dtype=np.float32),
        "cost_fn": reacher_cost, "dt": 0.02, "noise_sigma": NOISE_SIGMA, "device": device,
    }
    if not adaptive:
        controller_kwargs["rollouts"] = ROLLOUTS
    return controller_type(**controller_kwargs, **adaptive_args)


def evaluate(model, device, episode_seeds, adaptive=False, threshold=0.0, pilot_rollouts=8, pilot_horizon=2):
    environment = ReacherMuJoCo()
    successes = []
    latencies = []
    final_distances = []
    for episode_seed in episode_seeds:
        state = environment.reset(seed=episode_seed)
        controller = make_controller(model, device, environment.target, adaptive, threshold, pilot_rollouts, pilot_horizon)
        controller.reset()
        for _ in range(STEPS):
            command_start = time.perf_counter()
            action = controller.command(state)
            latencies.append(time.perf_counter() - command_start)
            state = environment.step(action)
        distance = float(np.linalg.norm(environment.end_effector() - environment.target))
        final_distances.append(distance)
        successes.append(float(distance < SUCCESS_THRESHOLD))
    return {
        "success_rate": float(np.mean(successes)),
        "final_distance": float(np.mean(final_distances)),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
    }


def make_episode_seeds(seed):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 1_000_000)) for _ in range(EPISODES)]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


if __name__ == "__main__":
    import sys

    print(f"CUDA available: {torch.cuda.is_available()}, device: {DEVICE}")
    records = {"single": [], "ensemble": [], "adaptive": []}
    latencies = {"single": [], "ensemble": [], "adaptive": []}
    distances = {"single": [], "ensemble": [], "adaptive": []}

    seeds_to_run = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else list(SEEDS)
    for seed in seeds_to_run:
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_environment = ReacherMuJoCo()
        train_data = collect_task_directed_data(train_environment, transitions=2000, seed=seed)
        episode_seeds = make_episode_seeds(seed + 1000)
        normalizer = make_normalizer(train_data, DEVICE)

        torch.manual_seed(seed)
        single = train_single(train_data, DEVICE, normalizer)
        torch.manual_seed(seed)
        ensemble = train_ensemble(train_data, DEVICE, normalizer)

        calibrator = make_controller(ensemble, DEVICE, np.zeros(2, dtype=np.float32), adaptive=True)
        torch.manual_seed(seed + 5000)
        calib_seeds = make_episode_seeds(seed + 2000)
        calib_states = []
        for s in calib_seeds:
            env = ReacherMuJoCo()
            calib_states.append(env.reset(seed=s))
        threshold = calibrator.calibrate_threshold(calib_states)

        seed_record = {"seed": seed}
        for name, model, is_adaptive in [("single", single, False), ("ensemble", ensemble, False), ("adaptive", ensemble, True)]:
            result = evaluate(model, DEVICE, episode_seeds, is_adaptive, threshold)
            records[name].append(result["success_rate"])
            latencies[name].append(result["latency_ms"])
            distances[name].append(result["final_distance"])
            seed_record[name] = result
        print(
            f"seed={seed:4d}  single={records['single'][-1]:.2f}  "
            f"ensemble={records['ensemble'][-1]:.2f}  adaptive={records['adaptive'][-1]:.2f}"
        )

        import json
        import os
        results_path = os.path.join(os.path.dirname(__file__), "mujoco_reacher_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps(seed_record) + "\n")

    print(f"\nAppended {len(seeds_to_run)} seed(s) to mujoco_reacher_results.jsonl")
