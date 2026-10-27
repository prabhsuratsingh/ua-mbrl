"""Torque-limited pendulum swing-up: true dynamics, learned-model hooks,
data collection, model training and controller factories.

Everything here was previously spread over robustness_mismatch_experiment.py,
ood_uncertainty_experiment.py and the per-experiment scripts that each kept
their own copy of make_states / make_adaptive / the success test.
"""

import numpy as np
import torch

from ua_mbrl.core.dynamics import GenericDynamicsModel, GenericEnsembleDynamics
from ua_mbrl.core.fallback import FallbackAdaptiveMPPI
from ua_mbrl.core.mppi import AdaptiveGenericMPPI, GenericMPPI, SeedBatchedGenericMPPI
from ua_mbrl.utils.device import DEVICE

DT = 0.1
NOMINAL_G = 9.81
NOMINAL_M = 1.0
L = 1.0
U_MIN = -5.0
U_MAX = 5.0
TARGET = [np.pi, 0.0]

INITIAL_EPISODES = 20
INITIAL_STEPS = 100
ENSEMBLE_SIZE = 5
TRAIN_EPOCHS = 100
HORIZON = 20
ROLLOUTS = 100
# Matches the hidden width of the retired ensemble_dynamics.DynamicsModel,
# preserved so the ported models are architecturally identical.
HIDDEN_DIM = 64

# Narrow-distribution training recipe used by the OOD experiments.
NARROW_EPISODES = 20
NARROW_STEPS = 60
OOD_TEST_SAMPLES = 400

# Seed bases shared by the pendulum control experiments.
CALIBRATION_STATES_SEED = 900
EVAL_STATES_SEED = 800

CONDITIONS = {
    "nominal": {},
    "mild_mismatch": {"mass": 1.2},
    "severe_mismatch": {"mass": 3.0, "gravity": NOMINAL_G * 0.4},
}


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_error(theta, target=np.pi):
    return wrap_angle(theta - target)


def pendulum_features(x, u):
    """[sin(theta), cos(theta), omega, u] -- matches the retired DynamicsModel.predict."""
    theta = x[:, 0:1]
    omega = x[:, 1:2]
    return torch.cat([torch.sin(theta), torch.cos(theta), omega, u], dim=1)


def pendulum_transition(x, delta):
    """Apply delta then wrap theta -- matches the retired DynamicsModel.forward."""
    theta_next = torch.remainder(x[:, 0:1] + delta[:, 0:1] + np.pi, 2 * np.pi) - np.pi
    omega_next = x[:, 1:2] + delta[:, 1:2]
    return torch.cat([theta_next, omega_next], dim=1)


def pendulum_residual(x_next, x):
    """Angle-wrapped delta_theta + raw delta_omega training target."""
    delta_theta = torch.remainder(x_next[:, 0:1] - x[:, 0:1] + np.pi, 2 * np.pi) - np.pi
    delta_omega = x_next[:, 1:2] - x[:, 1:2]
    return torch.cat([delta_theta, delta_omega], dim=1)


def pendulum_cost(x, u, target):
    """Angle-wrapped theta error -- MPPI cost must not treat theta as linear."""
    theta_error = torch.remainder(x[:, 0] - target[0] + np.pi, 2 * np.pi) - np.pi
    omega_error = x[:, 1] - target[1]
    return theta_error.square() + omega_error.square() + 0.01 * u.square().sum(dim=1)


def pendulum_step(x, u, gravity=NOMINAL_G, mass=NOMINAL_M, torque_bias=0.0):
    theta, omega = x
    omega_dot = (
        -(gravity / L) * np.sin(theta)
        + (u + torque_bias) / (mass * L**2)
    )
    return np.array(
        [wrap_angle(theta + omega * DT), omega + omega_dot * DT],
        dtype=np.float32,
    )


def is_success(x):
    """Upright and nearly still: |theta - pi| < 0.1 and |omega| < 0.5."""
    return abs(angle_error(x[0])) < 0.1 and abs(x[1]) < 0.5


def batch_success(x):
    """Vectorized is_success over a [S, 2] array of final states."""
    theta_err = np.abs(np.asarray([angle_error(x[s, 0]) for s in range(len(x))]))
    omega_err = np.abs(x[:, 1])
    return (theta_err < 0.1) & (omega_err < 0.5)


def make_states(seed, count):
    """`count` initial states, theta ~ U(-pi, pi), omega ~ U(-2, 2)."""
    rng = np.random.default_rng(seed)
    return np.asarray(
        [[rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)] for _ in range(count)],
        dtype=np.float32,
    )


def _as_dataset(states, actions, next_states):
    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


def collect_data(num_episodes=INITIAL_EPISODES, steps=INITIAL_STEPS, seed=0):
    """Uniform-random-torque rollouts over the full state space."""
    rng = np.random.default_rng(seed)
    states = []
    actions = []
    next_states = []

    for _ in range(num_episodes):
        x = np.array(
            [rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)],
            dtype=np.float32,
        )
        for _ in range(steps):
            u = rng.uniform(U_MIN, U_MAX)
            x_next = pendulum_step(x, u)
            states.append(x.copy())
            actions.append([u])
            next_states.append(x_next.copy())
            x = x_next

    return _as_dataset(states, actions, next_states)


def collect_narrow_data(seed=0):
    """Narrow training distribution (omega in [-1, 1], u in [-2, 2]) used by
    the OOD-detection experiments."""
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    for _ in range(NARROW_EPISODES):
        x = np.array(
            [rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)],
            dtype=np.float32,
        )
        for _ in range(NARROW_STEPS):
            u = rng.uniform(-2.0, 2.0)
            x_next = pendulum_step(x, u)
            states.append(x.copy())
            actions.append([u])
            next_states.append(x_next.copy())
            x = x_next
    return _as_dataset(states, actions, next_states)


def make_test_data(seed, omega_range, action_range, samples=OOD_TEST_SAMPLES):
    """Independent one-step transitions with omega and u drawn from the given ranges."""
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    for _ in range(samples):
        x = np.array(
            [rng.uniform(-np.pi, np.pi), rng.uniform(*omega_range)],
            dtype=np.float32,
        )
        u = rng.uniform(*action_range)
        states.append(x)
        actions.append([u])
        next_states.append(pendulum_step(x, u))
    return _as_dataset(states, actions, next_states)


class AnalyticalNominalModel:
    """Nominal analytical model used by the model-based baseline."""

    def __init__(self, device=DEVICE):
        self.device = torch.device(device)

    def __call__(self, x, u):
        theta = x[:, 0]
        omega = x[:, 1]
        action = u[:, 0]
        omega_next = omega + (
            -(NOMINAL_G / L) * torch.sin(theta)
            + action / (NOMINAL_M * L**2)
        ) * DT
        theta_next = torch.remainder(
            theta + omega * DT + np.pi,
            2 * np.pi,
        ) - np.pi
        return torch.stack([theta_next, omega_next], dim=1)

    def model_step(self, x, u):
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, 2)
        u_tensor = torch.as_tensor([u], dtype=torch.float32, device=self.device).reshape(1, 1)
        with torch.no_grad():
            prediction = self(x_tensor, u_tensor)
        return prediction.squeeze(0).cpu().numpy()


def make_single_model(device=DEVICE):
    return GenericDynamicsModel(
        state_dim=2,
        action_dim=1,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )


def make_ensemble(ensemble_size=ENSEMBLE_SIZE, device=DEVICE):
    return GenericEnsembleDynamics(
        state_dim=2,
        action_dim=1,
        ensemble_size=ensemble_size,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )


def train_models(data, train_fraction=0.8, epochs=TRAIN_EPOCHS, single_seed=1, ensemble_seed=2):
    """Train one single model and one bootstrap ensemble.

    The control experiments train on the first 80% of `data`; the OOD
    experiments pass train_fraction=None to train on all of it. Seeds 1/2
    (and bootstrap seed 2) are the values every published result used.
    """
    if train_fraction is None:
        train_data = data
    else:
        split = int(train_fraction * len(data["x"]))
        train_data = {
            "x": data["x"][:split],
            "u": data["u"][:split],
            "x_next": data["x_next"][:split],
        }

    torch.manual_seed(single_seed)
    single = make_single_model()
    single.train_model(train_data, epochs=epochs, verbose=False)

    torch.manual_seed(ensemble_seed)
    ensemble = make_ensemble()
    ensemble.train_ensemble(
        train_data,
        epochs=epochs,
        seed=ensemble_seed,
        verbose=False,
    )
    return single, ensemble


def train_default_ensemble():
    """The single nominal ensemble every pendulum control experiment uses."""
    _, ensemble = train_models(collect_data())
    return ensemble


def _controller_kwargs(model, horizon):
    return dict(
        dynamics=model,
        action_dim=1,
        horizon=horizon,
        action_low=[U_MIN],
        action_high=[U_MAX],
        target=TARGET,
        cost_fn=pendulum_cost,
        dt=DT,
        temperature=1.0,
        device=DEVICE,
    )


def make_fixed(model, rollouts=ROLLOUTS, noise_sigma=1.0, beta=0.0, horizon=HORIZON):
    return GenericMPPI(
        **_controller_kwargs(model, horizon),
        rollouts=rollouts,
        noise_sigma=noise_sigma,
        use_uncertainty=beta > 0.0,
        uncertainty_weight=beta,
    )


def make_adaptive(
    model,
    k_max,
    threshold=0.0,
    k_min=ROLLOUTS,
    min_sigma=1.0,
    max_sigma=1.0,
    pilot_rollouts=25,
    pilot_horizon=5,
    horizon=HORIZON,
):
    """Pilot-gated adaptive-K controller (K_min / K_max, pilot 25 x 5 by default)."""
    return AdaptiveGenericMPPI(
        **_controller_kwargs(model, horizon),
        noise_sigma=1.0,
        min_rollouts=k_min,
        max_rollouts=k_max,
        uncertainty_threshold=threshold,
        pilot_rollouts=pilot_rollouts,
        pilot_horizon=pilot_horizon,
        min_noise_sigma=min_sigma,
        max_noise_sigma=max_sigma,
    )


def make_fallback(model, threshold, severe_threshold, k_min=ROLLOUTS, k_max=3 * ROLLOUTS, fallback_gain=2.0):
    return FallbackAdaptiveMPPI(
        **_controller_kwargs(model, HORIZON),
        noise_sigma=1.0,
        min_rollouts=k_min,
        max_rollouts=k_max,
        uncertainty_threshold=threshold,
        severe_threshold=severe_threshold,
        fallback_gain=fallback_gain,
        fallback_bounds=(U_MIN, U_MAX),
        velocity_index=1,
        pilot_rollouts=25,
        pilot_horizon=5,
    )


def make_seed_batched(model, num_seeds, rollouts, noise_sigma=1.0, horizon=HORIZON):
    return SeedBatchedGenericMPPI(
        dynamics=model,
        num_seeds=num_seeds,
        action_dim=1,
        horizon=horizon,
        rollouts=rollouts,
        action_low=[U_MIN],
        action_high=[U_MAX],
        target=TARGET,
        cost_fn=pendulum_cost,
        dt=DT,
        temperature=1.0,
        noise_sigma=noise_sigma,
        device=DEVICE,
    )


def calibrate_threshold(model, k_max, calibration_states, k_min=ROLLOUTS, min_sigma=1.0, max_sigma=1.0):
    """95th-percentile pilot uncertainty over held-out in-distribution states.

    Note: constructing the calibrator consumes one draw from torch's global
    CPU generator (to seed its private pilot generator), and the threshold
    depends on that draw -- call order matters for exact reproduction.
    """
    calibrator = make_adaptive(
        model, k_max, threshold=0.0, k_min=k_min, min_sigma=min_sigma, max_sigma=max_sigma
    )
    return calibrator.calibrate_threshold(calibration_states)
