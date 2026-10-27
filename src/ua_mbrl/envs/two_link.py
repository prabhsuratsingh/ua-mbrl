"""Two-link planar MuJoCo arm, regulation-to-origin task.

Merges mujoco_generic_experiment.py (adapter, XML), mujoco_benchmark.py
(data, training, controllers, evaluation) and the per-script copies of the
controller/evaluation code in mujoco_quality_retuned.py / mujoco_claim_a.py.
"""

import time

import mujoco
import numpy as np

from ua_mbrl.core.dynamics import GenericDynamicsModel, GenericEnsembleDynamics
from ua_mbrl.core.mppi import AdaptiveGenericMPPI, GenericMPPI
from ua_mbrl.core.normalizer import AffineInputNormalizer

DT = 0.02
STATE_DIM = 4
ACTION_DIM = 2

TRAIN_EPOCHS = 75
EPISODES = 3
STEPS = 75
HORIZON = 50
ROLLOUTS = 300
K_MIN = 100
SUCCESS_RADIUS = 0.5

MODEL_XML = """
<mujoco model="two_link_arm">
  <option timestep="0.02" gravity="0 0 0"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
    <joint name="joint1" type="hinge" axis="0 0 1" damping="1" armature="1"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.06" density="1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="1" armature="1"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" density="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="motor2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


class TwoLinkMuJoCo:
    """Adapter exposing a flat qpos/qvel state and bounded torque action."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_low = np.full(ACTION_DIM, -5.0, dtype=np.float32)
        self.action_high = np.full(ACTION_DIM, 5.0, dtype=np.float32)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = rng.uniform(-0.2, 0.2, size=2)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_state(self, state, episode=0):
        """Reset (seeded by episode index, as the original evaluation did) and
        overwrite qpos/qvel with `state`."""
        self.reset(seed=episode)
        self.data.qpos[:] = state[:2]
        self.data.qvel[:] = state[2:]
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
        ]).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = np.clip(
            action,
            self.action_low,
            self.action_high,
        )
        mujoco.mj_step(self.model, self.data)
        return self.state()


def arm_cost(x, u, target):
    state_error = x - target
    return state_error.square().sum(dim=1) + 0.01 * u.square().sum(dim=1)


def _as_dataset(states, actions, next_states):
    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


def collect_random_data(environment, transitions=1000, seed=0):
    """Uniform random torques in [-1, 1] (smoke test only)."""
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    state = environment.reset(seed)
    for _ in range(transitions):
        action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        next_state = environment.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state
    return _as_dataset(states, actions, next_states)


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
    return _as_dataset(states, actions, next_states)


def make_normalizer(data, device):
    return AffineInputNormalizer(
        np.mean(data["x"], axis=0),
        np.std(data["x"], axis=0),
        np.mean(data["u"], axis=0),
        np.std(data["u"], axis=0),
        device,
    )


def train_single(data, device, normalizer, epochs=TRAIN_EPOCHS):
    model = GenericDynamicsModel(
        STATE_DIM,
        ACTION_DIM,
        hidden_dim=128,
        input_normalizer=normalizer,
        device=device,
    )
    model.train_model(data, epochs=epochs, verbose=False)
    return model


def train_ensemble(data, device, normalizer, epochs=TRAIN_EPOCHS, bootstrap_seed=7):
    # bootstrap_seed=7 regardless of the outer seed is what every published
    # MuJoCo result used.
    model = GenericEnsembleDynamics(
        STATE_DIM,
        ACTION_DIM,
        ensemble_size=5,
        hidden_dim=128,
        input_normalizer=normalizer,
        device=device,
    )
    model.train_ensemble(data, epochs=epochs, seed=bootstrap_seed, verbose=False)
    return model


def make_controller(
    model,
    device,
    adaptive=False,
    threshold=0.0,
    pilot_rollouts=8,
    pilot_horizon=2,
    noise_sigma=1.0,
    rollouts=ROLLOUTS,
    k_min=K_MIN,
):
    common = dict(
        dynamics=model,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        action_low=-5.0 * np.ones(ACTION_DIM),
        action_high=5.0 * np.ones(ACTION_DIM),
        target=np.zeros(STATE_DIM, dtype=np.float32),
        cost_fn=arm_cost,
        dt=0.02,
        noise_sigma=noise_sigma,
        device=device,
    )
    if adaptive:
        return AdaptiveGenericMPPI(
            **common,
            min_rollouts=k_min,
            max_rollouts=rollouts,
            uncertainty_threshold=threshold,
            pilot_rollouts=pilot_rollouts,
            pilot_horizon=pilot_horizon,
            min_noise_sigma=noise_sigma,
            max_noise_sigma=noise_sigma,
        )
    return GenericMPPI(**common, rollouts=rollouts)


def is_success(state):
    return float(np.linalg.norm(state) < SUCCESS_RADIUS)


def evaluate(
    model,
    device,
    states,
    adaptive=False,
    threshold=0.0,
    pilot_rollouts=8,
    pilot_horizon=2,
    noise_sigma=1.0,
    steps=STEPS,
):
    """One controller reused (reset) across episodes; no per-episode reseeding,
    so methods evaluated in sequence share torch's global stream."""
    environment = TwoLinkMuJoCo()
    controller = make_controller(
        model, device, adaptive, threshold, pilot_rollouts, pilot_horizon, noise_sigma=noise_sigma
    )
    successes = []
    latencies = []
    effective_rollouts = []

    for episode, initial_state in enumerate(states):
        state = environment.set_state(initial_state, episode)
        controller.reset()
        for _ in range(steps):
            command_start = time.perf_counter()
            action = controller.command(state)
            latencies.append(time.perf_counter() - command_start)
            effective = getattr(controller, "last_rollouts", controller.rollouts)
            if adaptive:
                effective += controller.pilot_rollouts * controller.pilot_horizon
            effective_rollouts.append(effective)
            state = environment.step(action)
        successes.append(is_success(state))

    return {
        "success_rate": float(np.mean(successes)),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
        "p95_latency_ms": 1000.0 * float(np.percentile(latencies, 95)),
        "effective_rollouts": float(np.mean(effective_rollouts)),
        "per_episode_success": successes,
    }


def evaluate_fixed_at_k(model, device, states, rollouts, noise_sigma, steps=STEPS):
    """Fixed-K success with a fresh controller per episode (matched-budget test)."""
    environment = TwoLinkMuJoCo()
    successes = []
    for episode, initial_state in enumerate(states):
        state = environment.set_state(initial_state, episode)
        controller = make_controller(model, device, noise_sigma=noise_sigma, rollouts=rollouts)
        controller.reset()
        for _ in range(steps):
            action = controller.command(state)
            state = environment.step(action)
        successes.append(is_success(state))
    return float(np.mean(successes))


def evaluate_pd(states, steps=STEPS):
    environment = TwoLinkMuJoCo()
    successes = []
    for episode, initial_state in enumerate(states):
        state = environment.set_state(initial_state, episode)
        for _ in range(steps):
            action = np.clip(
                -3.0 * state[:2] - 1.0 * state[2:],
                -5.0,
                5.0,
            )
            state = environment.step(action)
        successes.append(is_success(state))
    return float(np.mean(successes))


def make_states(seed, episodes=EPISODES):
    rng = np.random.default_rng(seed)
    return np.asarray([
        np.concatenate([
            rng.uniform(-0.5, 0.5, size=2),
            rng.uniform(-0.2, 0.2, size=2),
        ])
        for _ in range(episodes)
    ], dtype=np.float32)
