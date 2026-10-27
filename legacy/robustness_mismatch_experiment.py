"""
Robustness experiment for learned-model MPPI.

Train all learned models on nominal dynamics, then evaluate against
mass, gravity, and additive torque mismatches without retraining.
"""

import numpy as np
import torch

from generic_mbrl import (
    AdaptiveGenericMPPI,
    GenericDynamicsModel,
    GenericEnsembleDynamics,
    GenericMPPI,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DT = 0.1
NOMINAL_G = 9.81
NOMINAL_M = 1.0
L = 1.0
U_MIN = -5.0
U_MAX = 5.0

INITIAL_EPISODES = 20
INITIAL_STEPS = 100
ENSEMBLE_SIZE = 5
TRAIN_EPOCHS = 100
EVAL_EPISODES = 3
EVAL_STEPS = 25
HORIZON = 20
ROLLOUTS = 100
# Matches the hidden width of the retired ensemble_dynamics.DynamicsModel,
# preserved so the ported models are architecturally identical.
HIDDEN_DIM = 64


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


def collect_data(num_episodes=INITIAL_EPISODES, steps=INITIAL_STEPS, seed=0):
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

    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


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


def make_single_model():
    return GenericDynamicsModel(
        state_dim=2,
        action_dim=1,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )


def make_ensemble():
    return GenericEnsembleDynamics(
        state_dim=2,
        action_dim=1,
        ensemble_size=ENSEMBLE_SIZE,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )


def train_models(data):
    split = int(0.8 * len(data["x"]))
    train_data = {
        "x": data["x"][:split],
        "u": data["u"][:split],
        "x_next": data["x_next"][:split],
    }

    torch.manual_seed(1)
    single = make_single_model()
    single.train_model(train_data, epochs=TRAIN_EPOCHS, verbose=False)

    torch.manual_seed(2)
    ensemble = make_ensemble()
    ensemble.train_ensemble(
        train_data,
        epochs=TRAIN_EPOCHS,
        seed=2,
        verbose=False,
    )
    return single, ensemble


def initial_states(seed=100):
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            [rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)]
            for _ in range(EVAL_EPISODES)
        ],
        dtype=np.float32,
    )


def run_episode(controller, x0, true_params, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()
    x = x0.copy()
    errors = []
    uncertainties = []
    rollout_counts = []

    for _ in range(EVAL_STEPS):
        action = controller.command(x)
        u = float(np.asarray(action).reshape(-1)[0])
        effective_rollouts = getattr(
            controller,
            "last_rollouts",
            controller.rollouts,
        )
        if isinstance(controller, AdaptiveGenericMPPI):
            effective_rollouts += (
                controller.pilot_rollouts
                * controller.pilot_horizon
            )
        rollout_counts.append(effective_rollouts)
        if hasattr(controller.dynamics, "model_step_with_uncertainty"):
            model_next, uncertainty = controller.dynamics.model_step_with_uncertainty(x, u)
            uncertainties.append(uncertainty)
        else:
            model_next = controller.dynamics.model_step(x, u)
            uncertainties.append(0.0)

        true_next = pendulum_step(x, u, **true_params)
        error = model_next - true_next
        error[0] = angle_error(model_next[0], true_next[0])
        errors.append(error)
        x = true_next

    theta_error = abs(angle_error(x[0]))
    omega_error = abs(x[1])
    return {
        "success": theta_error < 0.1 and omega_error < 0.5,
        "theta_error": theta_error,
        "omega_error": omega_error,
        "model_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "mean_rollouts": float(np.mean(rollout_counts)),
    }


def evaluate(controller_factory, states, true_params, seed_offset):
    results = [
        run_episode(
            controller_factory(),
            x0,
            true_params,
            seed_offset + episode,
        )
        for episode, x0 in enumerate(states)
    ]
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "theta_error": float(np.mean([r["theta_error"] for r in results])),
        "omega_error": float(np.mean([r["omega_error"] for r in results])),
        "model_rmse": float(np.mean([r["model_rmse"] for r in results])),
        "uncertainty": float(np.mean([r["mean_uncertainty"] for r in results])),
        "rollouts": float(np.mean([r["mean_rollouts"] for r in results])),
    }


def controller_factory(model, beta=0.0, rollouts=ROLLOUTS):
    def make_controller():
        return GenericMPPI(
            dynamics=model,
            action_dim=1,
            horizon=HORIZON,
            rollouts=rollouts,
            action_low=[U_MIN],
            action_high=[U_MAX],
            target=[np.pi, 0.0],
            cost_fn=pendulum_cost,
            dt=DT,
            temperature=1.0,
            noise_sigma=1.0,
            use_uncertainty=beta > 0.0,
            uncertainty_weight=beta,
            device=DEVICE,
        )
    return make_controller


def adaptive_controller_factory(model, threshold=0.006):
    def make_controller():
        return AdaptiveGenericMPPI(
            dynamics=model,
            action_dim=1,
            horizon=HORIZON,
            action_low=[U_MIN],
            action_high=[U_MAX],
            target=[np.pi, 0.0],
            cost_fn=pendulum_cost,
            dt=DT,
            temperature=1.0,
            noise_sigma=1.0,
            min_rollouts=ROLLOUTS,
            max_rollouts=3 * ROLLOUTS,
            uncertainty_threshold=threshold,
            pilot_rollouts=25,
            pilot_horizon=5,
            device=DEVICE,
        )
    return make_controller


def print_result(condition, method, result):
    print(
        f"{condition:<18} {method:<18} "
        f"success={100 * result['success_rate']:5.1f}% "
        f"theta={result['theta_error']:.4f} "
        f"omega={result['omega_error']:.4f} "
        f"model_rmse={result['model_rmse']:.4f} "
        f"unc={result['uncertainty']:.5f} "
        f"K={result['rollouts']:.1f}"
    )


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    print("Collecting nominal data and training models...")
    data = collect_data()
    single, ensemble = train_models(data)
    calibration_states = initial_states(seed=101)
    states = initial_states(seed=100)

    calibration_controller = adaptive_controller_factory(ensemble)()
    torch.manual_seed(101)
    calibrated_threshold = calibration_controller.calibrate_threshold(
        calibration_states,
    )
    print(
        f"Calibrated adaptive uncertainty threshold: "
        f"{calibrated_threshold:.6f}"
    )

    conditions = {
        "nominal": {},
        "mass +20%": {"mass": 1.2},
        "gravity -15%": {"gravity": 9.81 * 0.85},
    }

    methods = {
        "nominal analytic": controller_factory(AnalyticalNominalModel()),
        "single learned": controller_factory(single),
        "ensemble": controller_factory(ensemble),
        "ensemble K=300": controller_factory(ensemble, rollouts=3 * ROLLOUTS),
        "ensemble beta=0.1": controller_factory(ensemble, beta=0.1),
        "adaptive ensemble": adaptive_controller_factory(
            ensemble,
            threshold=calibrated_threshold,
        ),
    }

    print("\nRobustness results")
    print("=" * 130)
    print("condition          method             success theta  omega  model_rmse unc")
    print("-" * 130)

    for condition, true_params in conditions.items():
        for method, factory in methods.items():
            result = evaluate(
                factory,
                states,
                true_params,
                seed_offset=1000,
            )
            print_result(condition, method, result)
