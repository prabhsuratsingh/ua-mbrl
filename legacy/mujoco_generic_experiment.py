"""Minimal MuJoCo adapter smoke test for generic ensemble MPPI."""

import numpy as np
import mujoco
import torch

from generic_mbrl import GenericEnsembleDynamics, GenericMPPI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.02
STATE_DIM = 4
ACTION_DIM = 2

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


def collect_data(environment, transitions=1000, seed=0):
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
    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


def arm_cost(x, u, target):
    state_error = x - target
    return state_error.square().sum(dim=1) + 0.01 * u.square().sum(dim=1)


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    environment = TwoLinkMuJoCo()
    data = collect_data(environment)
    print(
        f"MuJoCo state_dim={environment.state_dim} "
        f"action_dim={environment.action_dim} "
        f"transitions={len(data['x'])}"
    )

    ensemble = GenericEnsembleDynamics(
        state_dim=environment.state_dim,
        action_dim=environment.action_dim,
        ensemble_size=5,
        hidden_dim=128,
        device=DEVICE,
    )
    ensemble.train_ensemble(data, epochs=50, seed=7, verbose=False)

    controller = GenericMPPI(
        dynamics=ensemble,
        action_dim=environment.action_dim,
        horizon=20,
        rollouts=500,
        action_low=environment.action_low,
        action_high=environment.action_high,
        target=np.zeros(environment.state_dim, dtype=np.float32),
        cost_fn=arm_cost,
        dt=DT,
        device=DEVICE,
    )

    state = environment.reset(seed=11)
    for step in range(10):
        action = controller.command(state)
        state = environment.step(action)
        print(
            f"step={step:02d} action={np.round(action, 3)} "
            f"state_norm={np.linalg.norm(state):.4f}"
        )

    print("MuJoCo generic MPPI smoke test passed")
