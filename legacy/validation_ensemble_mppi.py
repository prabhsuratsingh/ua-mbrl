"""
Ensemble MPPI Validation Experiment
------------------------------------

Research Question:
  Does ensemble-based MPPI perform comparably to single-model MPPI?
  Can we maintain performance while estimating uncertainty?

Methodology:
  1. Collect initial random dataset
  2. Train single DNN model
  3. Train ensemble of E=5 models
  4. Run MPPI control with both on true pendulum
  5. Compare: success rate, control effort, online model error
  6. Validate ensemble uncertainty correlation during control

Expected Outcome:
  - Ensemble mean predictions ≈ single model (same test error)
  - Ensemble provides uncertainty estimates
  - MPPI control with ensemble shows feasibility
  - Ready for uncertainty-aware cost term (Phase 3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(__file__))

from generic_mbrl import GenericDynamicsModel, GenericEnsembleDynamics, GenericMPPI
from robustness_mismatch_experiment import (
    pendulum_cost,
    pendulum_features,
    pendulum_residual,
    pendulum_transition,
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Matches the hidden width of the retired ensemble_dynamics.DynamicsModel.
HIDDEN_DIM = 64

# Environment parameters
DT = 0.1
G = 9.81
L = 1.0
M = 1.0
U_MIN = -5.0
U_MAX = 5.0

# Training config
INITIAL_EPISODES = 20
INITIAL_STEPS = 100
ENSEMBLE_SIZE = 5
TRAIN_EPOCHS = 100

# MPPI config
MPPI_HORIZON = 30
MPPI_ROLLOUTS = 1000
MPPI_LAMBDA = 1.0
MPPI_NOISE_SIGMA = 1.0
CONTROL_WEIGHT = 0.01

# Evaluation config
EVAL_EPISODES = 10
EVAL_STEPS = 60

SEED_DATA = 0
SEED_INITIAL_STATES = 100


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_error(theta, target=np.pi):
    return wrap_angle(theta - target)


# ============================================================
# TRUE ENVIRONMENT
# ============================================================

def environment_dynamics(
    x,
    g=G,
    l=L,
    u=0.0,
    m=M,
    dt=DT,
):
    """Analytical pendulum dynamics."""
    theta, omega = x

    theta_dot = omega
    omega_dot = -(g / l) * np.sin(theta) + u / (m * l**2)

    theta_next = theta + theta_dot * dt
    omega_next = omega + omega_dot * dt

    return np.array(
        [wrap_angle(theta_next), omega_next],
        dtype=np.float32,
    )


class AnalyticalDynamicsModel:
    """Torch batch wrapper for the true dynamics baseline."""

    def __init__(self, device=DEVICE):
        self.device = torch.device(device)

    def __call__(self, x, u):
        theta = x[:, 0]
        omega = x[:, 1]
        action = u[:, 0]

        omega_next = omega + (
            -(G / L) * torch.sin(theta)
            + action / (M * L**2)
        ) * DT
        theta_next = torch.remainder(
            theta + omega * DT + np.pi,
            2 * np.pi,
        ) - np.pi

        return torch.stack([theta_next, omega_next], dim=1)

    def model_step(self, x, u):
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2)
        u_tensor = torch.as_tensor(
            [u],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1)

        with torch.no_grad():
            prediction = self(x_tensor, u_tensor)

        return prediction.squeeze(0).cpu().numpy()


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim=2, action_dim=1):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, x, u, x_next):
        self.states[self.ptr] = x
        self.actions[self.ptr] = u
        self.next_states[self.ptr] = x_next
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_all(self):
        return {
            "x": self.states[:self.size],
            "u": self.actions[:self.size],
            "x_next": self.next_states[:self.size],
        }


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_initial_data(
    buffer,
    num_episodes=INITIAL_EPISODES,
    steps_per_episode=INITIAL_STEPS,
    action_scale=U_MAX,
    seed=SEED_DATA,
):
    """Collect random transitions."""
    np.random.seed(seed)

    for _ in range(num_episodes):
        x = np.array(
            [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-2.0, 2.0),
            ],
            dtype=np.float32,
        )

        for _ in range(steps_per_episode):
            u = np.random.uniform(-action_scale, action_scale)
            x_next = environment_dynamics(x, u=u)
            buffer.add(
                x,
                np.array([u], dtype=np.float32),
                x_next,
            )
            x = x_next


# ============================================================
# MPPI ROLLOUT
# ============================================================

def run_mppi(
    controller,
    x0,
    num_steps=EVAL_STEPS,
    seed=0,
):
    """
    Run receding-horizon MPPI on true environment.

    Returns:
        x_history: [num_steps+1, 2]
        u_history: [num_steps]
        model_errors: [num_steps, 2]
        uncertainties: [num_steps] or None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller.reset()

    x = np.array(x0, dtype=np.float32)
    x_history = [x.copy()]
    u_history = []
    model_errors = []
    uncertainties = []

    for _ in range(num_steps):
        # Plan with learned model
        raw_action = controller.command(x)
        u0 = float(np.asarray(raw_action).reshape(-1)[0])

        # Get model's prediction (for error tracking)
        if hasattr(controller.dynamics, "model_step_with_uncertainty"):
            x_model_next, uncertainty = controller.dynamics.model_step_with_uncertainty(x, u0)
            uncertainties.append(uncertainty)
        else:
            x_model_next = controller.dynamics.model_step(x, u0)
            uncertainties.append(0.0)

        # Apply to true environment
        x_next = environment_dynamics(x, u=u0)

        # Measure error
        error = x_model_next - x_next
        error[0] = angle_error(x_model_next[0], x_next[0])
        model_errors.append(error)

        x = x_next
        x_history.append(x.copy())
        u_history.append(u0)

    return (
        np.array(x_history),
        np.array(u_history),
        np.array(model_errors),
        np.array(uncertainties),
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_controller(
    controller,
    initial_states,
    num_steps=EVAL_STEPS,
    seed=100,
):
    """Evaluate controller over multiple initial states."""
    results = []

    for episode, x0 in enumerate(initial_states):
        x_history, u_history, model_errors, uncertainties = run_mppi(
            controller=controller,
            x0=x0,
            num_steps=num_steps,
            seed=seed + episode,
        )

        theta_final = x_history[-1, 0]
        omega_final = x_history[-1, 1]
        theta_err = angle_error(theta_final)

        success = (
            abs(theta_err) < 0.1
            and abs(omega_final) < 0.5
        )

        result = {
            "episode": episode + 1,
            "theta_error": theta_err,
            "final_omega": omega_final,
            "mean_abs_u": np.mean(np.abs(u_history)),
            "max_abs_u": np.max(np.abs(u_history)),
            "model_error_magnitude": np.sqrt(
                np.mean(model_errors ** 2)
            ),
            "success": success,
            "trajectory": x_history,
            "controls": u_history,
            "model_errors": model_errors,
            "uncertainties": uncertainties,
        }

        results.append(result)

        print(
            f"Episode {episode + 1:2d}/{len(initial_states)}: "
            f"theta_err={theta_err: .3f} | "
            f"omega={omega_final: .3f} | "
            f"success={success}"
        )

    return results


# ============================================================
# ANALYSIS
# ============================================================

def summarize_results(results, title):
    """Print summary statistics."""
    successes = np.array([r["success"] for r in results])
    theta_errors = np.array([abs(r["theta_error"]) for r in results])
    omega_errors = np.array([abs(r["final_omega"]) for r in results])
    mean_controls = np.array([r["mean_abs_u"] for r in results])

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(
        f"Success rate: "
        f"{100 * np.mean(successes):.1f}% "
        f"({np.sum(successes)}/{len(results)})"
    )
    print(f"Mean |theta error|: {np.mean(theta_errors):.4f}")
    print(f"Mean |omega|: {np.mean(omega_errors):.4f}")
    print(f"Mean |u|: {np.mean(mean_controls):.4f}")
    print(f"Max |u|: {np.max([r['max_abs_u'] for r in results]):.4f}")

    return {
        "success_rate": np.mean(successes),
        "mean_theta_error": np.mean(theta_errors),
        "mean_omega_error": np.mean(omega_errors),
        "mean_control": np.mean(mean_controls),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_comparison(
    single_results,
    ensemble_results,
    filename="runs/ensemble_mppi_comparison.png",
):
    """Compare trajectories from single vs ensemble."""
    fig, axes = plt.subplots(
        EVAL_EPISODES, 3,
        figsize=(14, 3 * EVAL_EPISODES),
    )

    if EVAL_EPISODES == 1:
        axes = axes.reshape(1, -1)

    for ep in range(EVAL_EPISODES):
        # Theta trajectory
        axes[ep, 0].plot(
            single_results[ep]["trajectory"][:, 0],
            label="single model",
            alpha=0.7,
        )
        axes[ep, 0].plot(
            ensemble_results[ep]["trajectory"][:, 0],
            label="ensemble",
            alpha=0.7,
            linestyle="--",
        )
        axes[ep, 0].axhline(np.pi, linestyle=":", alpha=0.5)
        axes[ep, 0].set_ylabel(r"$\theta$ [rad]")
        axes[ep, 0].set_title(f"Episode {ep+1}: Angle")
        axes[ep, 0].legend()
        axes[ep, 0].grid(True, alpha=0.3)

        # Omega trajectory
        axes[ep, 1].plot(
            single_results[ep]["trajectory"][:, 1],
            label="single model",
            alpha=0.7,
        )
        axes[ep, 1].plot(
            ensemble_results[ep]["trajectory"][:, 1],
            label="ensemble",
            alpha=0.7,
            linestyle="--",
        )
        axes[ep, 1].axhline(0.0, linestyle=":", alpha=0.5)
        axes[ep, 1].set_ylabel(r"$\omega$ [rad/s]")
        axes[ep, 1].set_title(f"Episode {ep+1}: Velocity")
        axes[ep, 1].legend()
        axes[ep, 1].grid(True, alpha=0.3)

        # Uncertainty
        axes[ep, 2].plot(
            ensemble_results[ep]["uncertainties"],
            label="ensemble uncertainty",
            alpha=0.7,
            color="purple",
        )
        axes[ep, 2].set_ylabel("Uncertainty")
        axes[ep, 2].set_title(f"Episode {ep+1}: Predicted Uncertainty")
        axes[ep, 2].legend()
        axes[ep, 2].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")
    axes[-1, 2].set_xlabel("Time step")

    plt.tight_layout()
    os.makedirs("runs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {filename}")
    plt.close()


def plot_summary_comparison(
    single_stats,
    ensemble_stats,
    filename="runs/ensemble_summary_stats.png",
):
    """Bar chart comparing key metrics."""
    metrics = [
        ("Success Rate", "success_rate"),
        ("Theta Error", "mean_theta_error"),
        ("Control Effort", "mean_control"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x_pos = np.arange(2)
    width = 0.35

    for idx, (title, key) in enumerate(metrics):
        values = [
            single_stats[key],
            ensemble_stats[key],
        ]

        bars = axes[idx].bar(
            x_pos,
            values,
            width,
            label=[f"Single", f"Ensemble"],
        )

        axes[idx].set_ylabel(title)
        axes[idx].set_title(f"{title} Comparison")
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(["Single", "Ensemble"])
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs("runs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.close()


# ============================================================
# MAIN VALIDATION
# ============================================================

if __name__ == "__main__":

    print(f"Compute device: {DEVICE}")

    # --------------------------------------------------------
    # 1. Data collection
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: COLLECTING DATA")
    print("=" * 60)

    set_seed(SEED_DATA)

    buffer = ReplayBuffer(capacity=10000)
    collect_initial_data(buffer, INITIAL_EPISODES, INITIAL_STEPS)

    data = buffer.get_all()

    train_episodes = int(0.8 * INITIAL_EPISODES)
    split = train_episodes * INITIAL_STEPS

    train_data = {
        "x": data["x"][:split],
        "u": data["u"][:split],
        "x_next": data["x_next"][:split],
    }

    print(f"Collected {len(data['x'])} transitions")
    print(f"Train: {len(train_data['x'])}, Test: {len(data['x']) - len(train_data['x'])}")

    # --------------------------------------------------------
    # 2. Train single model
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING SINGLE MODEL")
    print("=" * 60)

    set_seed(1)

    single_model = GenericDynamicsModel(
        state_dim=2,
        action_dim=1,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )

    single_model.train_model(train_data, epochs=TRAIN_EPOCHS, verbose=False)

    print("Single model training complete")

    # --------------------------------------------------------
    # 3. Train ensemble
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: TRAINING ENSEMBLE ({ENSEMBLE_SIZE} models)")
    print("=" * 60)

    set_seed(2)

    ensemble = GenericEnsembleDynamics(
        state_dim=2,
        action_dim=1,
        ensemble_size=ENSEMBLE_SIZE,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )

    ensemble.train_ensemble(train_data, epochs=TRAIN_EPOCHS, verbose=False)

    print("Ensemble training complete")

    # --------------------------------------------------------
    # 4. Setup initial states
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: SETUP EVALUATION")
    print("=" * 60)

    rng = np.random.default_rng(SEED_INITIAL_STATES)
    initial_states = np.array(
        [
            [
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-2.0, 2.0),
            ]
            for _ in range(EVAL_EPISODES)
        ],
        dtype=np.float32,
    )

    print(f"Created {len(initial_states)} initial states for evaluation")

    # --------------------------------------------------------
    # 5. Analytical-model MPPI ceiling
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: ANALYTICAL-MODEL MPPI")
    print("=" * 60)

    def make_mppi_controller(model, use_uncertainty=False, uncertainty_weight=0.0):
        return GenericMPPI(
            dynamics=model,
            action_dim=1,
            horizon=MPPI_HORIZON,
            rollouts=MPPI_ROLLOUTS,
            action_low=[U_MIN],
            action_high=[U_MAX],
            target=[np.pi, 0.0],
            cost_fn=pendulum_cost,
            dt=DT,
            temperature=MPPI_LAMBDA,
            noise_sigma=MPPI_NOISE_SIGMA,
            use_uncertainty=use_uncertainty,
            uncertainty_weight=uncertainty_weight,
            device=DEVICE,
        )

    analytical_controller = make_mppi_controller(AnalyticalDynamicsModel(device=DEVICE))

    analytical_results = evaluate_controller(
        analytical_controller,
        initial_states,
        num_steps=EVAL_STEPS,
        seed=100,
    )
    analytical_stats = summarize_results(
        analytical_results,
        "ANALYTICAL MODEL MPPI",
    )

    # --------------------------------------------------------
    # 6. MPPI with single model
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: SINGLE-MODEL MPPI")
    print("=" * 60)

    single_controller = make_mppi_controller(single_model)

    single_results = evaluate_controller(
        single_controller,
        initial_states,
        num_steps=EVAL_STEPS,
        seed=100,
    )

    single_stats = summarize_results(single_results, "SINGLE MODEL MPPI")

    # --------------------------------------------------------
    # 7. MPPI with ensemble
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: ENSEMBLE MPPI")
    print("=" * 60)

    ensemble_controller = make_mppi_controller(ensemble)  # Not using uncertainty yet (Phase 3)

    ensemble_results = evaluate_controller(
        ensemble_controller,
        initial_states,
        num_steps=EVAL_STEPS,
        seed=100,
    )

    ensemble_stats = summarize_results(ensemble_results, "ENSEMBLE MPPI (no uncertainty)")

    # --------------------------------------------------------
    # 8. Uncertainty-aware MPPI beta sweep
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: UNCERTAINTY-AWARE MPPI BETA SWEEP")
    print("=" * 60)

    uncertainty_stats = {}
    for beta in (0.1, 1.0, 10.0):
        uncertainty_controller = make_mppi_controller(
            ensemble, use_uncertainty=True, uncertainty_weight=beta
        )

        beta_results = evaluate_controller(
            uncertainty_controller,
            initial_states,
            num_steps=EVAL_STEPS,
            seed=100,
        )
        uncertainty_stats[beta] = summarize_results(
            beta_results,
            f"ENSEMBLE MPPI (uncertainty beta={beta})",
        )

    # --------------------------------------------------------
    # 9. Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: ANALYSIS & VISUALIZATION")
    print("=" * 60)

    print("\nKey Comparison:")
    print(
        f"  Analytical model success rate: "
        f"{100*analytical_stats['success_rate']:.1f}%"
    )
    print(
        f"  Single model success rate: "
        f"{100*single_stats['success_rate']:.1f}%"
    )
    print(
        f"  Ensemble success rate:     "
        f"{100*ensemble_stats['success_rate']:.1f}%"
    )

    # Check if ensemble is providing uncertainty
    avg_uncertainty = np.mean([
        np.mean(r["uncertainties"])
        for r in ensemble_results
    ])
    print(f"  Ensemble average uncertainty: {avg_uncertainty:.6f}")

    # Visualizations
    plot_comparison(single_results, ensemble_results)
    plot_summary_comparison(single_stats, ensemble_stats)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print("\nNext Steps:")
    print("  1. Verify ensemble uncertainty correlates with errors")
    print("  2. Implement uncertainty-aware cost term")
    print("  3. Test adaptive K based on uncertainty")
    print("  4. Evaluate robustness to model mismatch")
