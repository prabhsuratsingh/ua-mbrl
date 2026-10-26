"""
Research Experiment: Ensemble Uncertainty vs Model Error
---------------------------------------------------------

Research Question:
  Does ensemble disagreement actually predict where our learned 
  MPPI dynamics model is going to fail?

Methodology:
  1. Collect initial random dataset from true environment
  2. Train single DNN model
  3. Train ensemble of E=5 DNN models
  4. Evaluate one-step prediction error on test set
  5. Compare ensemble disagreement (variance) with actual errors
  6. Analyze correlation structure
  7. Visualize: uncertainty vs error, success/failure regions

Expected Outcome:
  If ensemble disagreement correlates strongly with actual model error,
  then we have foundation for uncertainty-aware MPPI.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn

# Import from main scripts
# Note: Adjust imports based on your project structure
import sys
sys.path.insert(0, os.path.dirname(__file__))

from generic_mbrl import GenericDynamicsModel, GenericEnsembleDynamics
from robustness_mismatch_experiment import (
    pendulum_features,
    pendulum_residual,
    pendulum_transition,
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Matches the hidden width of the retired ensemble_dynamics.DynamicsModel.
HIDDEN_DIM = 64

# Environment parameters (pendulum)
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
SEED_DATA = 0
SEED_INITIAL_STATES = 100


# ============================================================
# UTILITIES (same as main script)
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


def environment_dynamics_batch(x, u):
    """Vectorized analytical dynamics."""
    theta = x[:, 0]
    omega = x[:, 1]

    theta_dot = omega
    omega_dot = -(G / L) * torch.sin(theta) + u[:, 0] / (M * L**2)

    theta_next = theta + theta_dot * DT
    omega_next = omega + omega_dot * DT

    theta_next = torch.remainder(
        theta_next + np.pi,
        2 * np.pi,
    ) - np.pi

    return torch.stack(
        [theta_next, omega_next],
        dim=1,
    )


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim=2, action_dim=1):
        self.capacity = capacity
        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
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

    def __len__(self):
        return self.size


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
# EVALUATION
# ============================================================

def evaluate_single_model(model, test_data):
    """One-step evaluation for single model."""
    predictions = np.array(
        [
            model.model_step(x, u[0])
            for x, u in zip(
                test_data["x"],
                test_data["u"],
            )
        ]
    )

    errors = predictions - test_data["x_next"]

    # Handle angle periodicity
    errors[:, 0] = np.array(
        [
            angle_error(p, t)
            for p, t in zip(
                predictions[:, 0],
                test_data["x_next"][:, 0],
            )
        ]
    )

    theta_mse = np.mean(errors[:, 0] ** 2)
    omega_mse = np.mean(errors[:, 1] ** 2)
    total_mse = np.mean(errors ** 2)

    return {
        "predictions": predictions,
        "errors": errors,
        "theta_mse": theta_mse,
        "omega_mse": omega_mse,
        "total_mse": total_mse,
        "abs_errors": np.abs(errors),
    }


def ensemble_disagreement_diagnostics(ensemble, test_data):
    """Pendulum-specific diagnostic: ensemble predictions vs true next state,
    with angle-wrapped error. (generic_mbrl.py stays dimension-agnostic, so
    this lives here rather than on GenericEnsembleDynamics.)"""
    x = torch.as_tensor(test_data["x"], dtype=torch.float32, device=DEVICE)
    u = torch.as_tensor(test_data["u"], dtype=torch.float32, device=DEVICE)
    x_next_true = torch.as_tensor(test_data["x_next"], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        mean, var, std = ensemble.forward_mean_var(x, u)
        uncertainty = ensemble.predict_uncertainty(x, u)

    error = mean - x_next_true
    error[:, 0] = torch.remainder(error[:, 0] + np.pi, 2 * np.pi) - np.pi
    abs_error = torch.abs(error)
    error_magnitude = torch.sqrt(torch.sum(error ** 2, dim=1))

    return {
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
        "uncertainty": uncertainty.cpu().numpy(),
        "error": error.cpu().numpy(),
        "abs_error": abs_error.cpu().numpy(),
        "error_magnitude": error_magnitude.cpu().numpy(),
        "x_true": x_next_true.cpu().numpy(),
    }


def evaluate_ensemble_model(ensemble, test_data):
    """One-step evaluation for ensemble."""
    # Evaluate ensemble disagreement
    diag = ensemble_disagreement_diagnostics(ensemble, test_data)

    # Also get mean predictions
    predictions = diag["mean"]

    # Compute errors
    errors = diag["error"]
    abs_errors = diag["abs_error"]

    theta_mse = np.mean(errors[:, 0] ** 2)
    omega_mse = np.mean(errors[:, 1] ** 2)
    total_mse = np.mean(errors ** 2)

    return {
        "mean_predictions": predictions,
        "errors": errors,
        "abs_errors": abs_errors,
        "std": diag["std"],
        "uncertainty": diag["uncertainty"],
        "error_magnitude": diag["error_magnitude"],
        "theta_mse": theta_mse,
        "omega_mse": omega_mse,
        "total_mse": total_mse,
    }


# ============================================================
# ANALYSIS & VISUALIZATION
# ============================================================

def analyze_uncertainty_error_correlation(
    single_model_results,
    ensemble_results,
):
    """
    Primary research analysis:
    Does ensemble disagreement correlate with prediction error?
    """
    uncertainty = ensemble_results["uncertainty"]
    error_magnitude = ensemble_results["error_magnitude"]

    # Correlation analysis
    pearson_r, pearson_p = pearsonr(
        uncertainty,
        error_magnitude,
    )
    spearman_r, spearman_p = spearmanr(
        uncertainty,
        error_magnitude,
    )

    print("\n" + "=" * 60)
    print("UNCERTAINTY vs ERROR CORRELATION")
    print("=" * 60)
    print(
        f"Pearson r:  {pearson_r:.4f} "
        f"(p={pearson_p:.2e})"
    )
    print(
        f"Spearman r: {spearman_r:.4f} "
        f"(p={spearman_p:.2e})"
    )

    print("\nInterpretation:")
    if abs(pearson_r) > 0.7:
        print(
            "  ✓ STRONG correlation: "
            "uncertainty is a good predictor of error"
        )
    elif abs(pearson_r) > 0.5:
        print(
            "  ~ MODERATE correlation: "
            "uncertainty has some predictive value"
        )
    else:
        print(
            "  ✗ WEAK correlation: "
            "uncertainty does not predict error well"
        )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def analyze_uncertainty_calibration(ensemble_results, num_bins=5):
    """Measure whether uncertainty ranking identifies high-error samples."""
    uncertainty = ensemble_results["uncertainty"]
    error_magnitude = ensemble_results["error_magnitude"]

    order = np.argsort(uncertainty)
    bins = np.array_split(order, num_bins)
    bin_errors = np.array([
        np.mean(error_magnitude[indices])
        for indices in bins
    ])
    bin_uncertainties = np.array([
        np.mean(uncertainty[indices])
        for indices in bins
    ])

    top_count = max(1, len(order) // 5)
    top_indices = order[-top_count:]
    error_order = np.argsort(error_magnitude)
    worst_indices = set(error_order[-top_count:])
    top_error_capture = len(
        set(top_indices).intersection(worst_indices)
    ) / top_count

    print("\n" + "=" * 60)
    print("UNCERTAINTY CALIBRATION")
    print("=" * 60)
    print("Bin | mean uncertainty | mean error")
    for index, (mean_uncertainty, mean_error) in enumerate(
        zip(bin_uncertainties, bin_errors),
        start=1,
    ):
        print(
            f" {index:2d} | {mean_uncertainty:16.6f} | "
            f"{mean_error:10.6f}"
        )
    print(
        f"Top-20% uncertainty captures "
        f"{100 * top_error_capture:.1f}% of top-20% errors"
    )

    return {
        "bin_uncertainty": bin_uncertainties,
        "bin_error": bin_errors,
        "top_error_capture": top_error_capture,
    }


def plot_uncertainty_vs_error(
    single_model_results,
    ensemble_results,
    filename="runs/uncertainty_vs_error.png",
):
    """Scatter plot: ensemble uncertainty vs prediction error."""
    uncertainty = ensemble_results["uncertainty"]
    error_magnitude = ensemble_results["error_magnitude"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        uncertainty,
        error_magnitude,
        alpha=0.5,
        s=20,
        label="test samples",
    )

    # Add trend line
    z = np.polyfit(uncertainty, error_magnitude, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(
        uncertainty.min(),
        uncertainty.max(),
        100,
    )
    ax.plot(
        x_trend,
        p(x_trend),
        "r--",
        linewidth=2,
        label="trend",
    )

    ax.set_xlabel("Ensemble Uncertainty (variance)")
    ax.set_ylabel("Prediction Error Magnitude")
    ax.set_title(
        "Does Ensemble Disagreement Predict Model Failure?"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs("runs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {filename}")
    plt.close()


def plot_uncertainty_distribution(
    ensemble_results,
    filename="runs/uncertainty_distribution.png",
):
    """Distribution of ensemble uncertainty."""
    uncertainty = ensemble_results["uncertainty"]
    error_magnitude = ensemble_results["error_magnitude"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Uncertainty distribution
    axes[0].hist(uncertainty, bins=30, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Ensemble Uncertainty")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Uncertainty Distribution")
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    axes[1].hist(error_magnitude, bins=30, alpha=0.7, edgecolor="black", color="orange")
    axes[1].set_xlabel("Prediction Error Magnitude")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("runs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.close()


def plot_ensemble_std_vs_error(
    ensemble_results,
    filename="runs/ensemble_std_vs_error.png",
):
    """Plot per-dimension standard deviation vs error."""
    std = ensemble_results["std"]
    errors = ensemble_results["errors"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Theta: std vs error
    axes[0].scatter(
        std[:, 0],
        np.abs(errors[:, 0]),
        alpha=0.5,
        s=20,
    )
    axes[0].set_xlabel("Ensemble Std Dev (theta)")
    axes[0].set_ylabel("Absolute Error (theta)")
    axes[0].set_title("Theta: Uncertainty vs Error")
    axes[0].grid(True, alpha=0.3)

    # Omega: std vs error
    axes[1].scatter(
        std[:, 1],
        np.abs(errors[:, 1]),
        alpha=0.5,
        s=20,
        color="orange",
    )
    axes[1].set_xlabel("Ensemble Std Dev (omega)")
    axes[1].set_ylabel("Absolute Error (omega)")
    axes[1].set_title("Omega: Uncertainty vs Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("runs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.close()


def compare_single_vs_ensemble(
    single_results,
    ensemble_results,
):
    """Compare single model vs ensemble."""
    print("\n" + "=" * 60)
    print("SINGLE MODEL vs ENSEMBLE")
    print("=" * 60)

    print(f"\nSingle Model:")
    print(f"  Theta MSE:   {single_results['theta_mse']:.6f}")
    print(f"  Omega MSE:   {single_results['omega_mse']:.6f}")
    print(f"  Total MSE:   {single_results['total_mse']:.6f}")

    print(f"\nEnsemble Mean (should be similar):")
    print(f"  Theta MSE:   {ensemble_results['theta_mse']:.6f}")
    print(f"  Omega MSE:   {ensemble_results['omega_mse']:.6f}")
    print(f"  Total MSE:   {ensemble_results['total_mse']:.6f}")

    print(f"\nEnsemble Uncertainty (variance):")
    print(
        f"  Mean uncertainty: "
        f"{np.mean(ensemble_results['uncertainty']):.6f}"
    )
    print(
        f"  Std uncertainty:  "
        f"{np.std(ensemble_results['uncertainty']):.6f}"
    )
    print(
        f"  Max uncertainty:  "
        f"{np.max(ensemble_results['uncertainty']):.6f}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    print(f"Compute device: {DEVICE}")

    # --------------------------------------------------------
    # 1. Data collection
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: COLLECTING INITIAL DATA")
    print("=" * 60)

    set_seed(SEED_DATA)

    buffer = ReplayBuffer(capacity=10000)
    collect_initial_data(
        buffer,
        num_episodes=INITIAL_EPISODES,
        steps_per_episode=INITIAL_STEPS,
    )

    data = buffer.get_all()

    # Train/test split
    TRAIN_EPISODES = int(0.8 * INITIAL_EPISODES)
    split = TRAIN_EPISODES * INITIAL_STEPS

    train_data = {
        "x": data["x"][:split],
        "u": data["u"][:split],
        "x_next": data["x_next"][:split],
    }

    test_data = {
        "x": data["x"][split:],
        "u": data["u"][split:],
        "x_next": data["x_next"][split:],
    }

    print(f"Total samples: {len(data['x'])}")
    print(f"Train samples: {len(train_data['x'])}")
    print(f"Test samples: {len(test_data['x'])}")

    print(f"\nData ranges:")
    print(
        f"  theta: [{data['x'][:, 0].min():.2f}, "
        f"{data['x'][:, 0].max():.2f}]"
    )
    print(
        f"  omega: [{data['x'][:, 1].min():.2f}, "
        f"{data['x'][:, 1].max():.2f}]"
    )
    print(
        f"  action: [{data['u'].min():.2f}, "
        f"{data['u'].max():.2f}]"
    )

    # --------------------------------------------------------
    # 2. Train single DNN model
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING SINGLE DYNAMICS MODEL")
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

    single_model.train_model(train_data, epochs=TRAIN_EPOCHS, verbose=True)

    # --------------------------------------------------------
    # 3. Train ensemble model
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

    ensemble.train_ensemble(
        train_data,
        epochs=TRAIN_EPOCHS,
        verbose=True,
    )

    # --------------------------------------------------------
    # 4. Evaluate single model
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATING SINGLE MODEL")
    print("=" * 60)

    single_results = evaluate_single_model(single_model, test_data)

    print(f"Theta MSE:   {single_results['theta_mse']:.6f}")
    print(f"Omega MSE:   {single_results['omega_mse']:.6f}")
    print(f"Total MSE:   {single_results['total_mse']:.6f}")

    # --------------------------------------------------------
    # 5. Evaluate ensemble model
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATING ENSEMBLE MODEL")
    print("=" * 60)

    ensemble_results = evaluate_ensemble_model(ensemble, test_data)

    print(f"Mean Theta MSE:   {ensemble_results['theta_mse']:.6f}")
    print(f"Mean Omega MSE:   {ensemble_results['omega_mse']:.6f}")
    print(f"Mean Total MSE:   {ensemble_results['total_mse']:.6f}")

    # --------------------------------------------------------
    # 6. Compare and analyze
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: ANALYSIS")
    print("=" * 60)

    compare_single_vs_ensemble(single_results, ensemble_results)

    correlation_stats = analyze_uncertainty_error_correlation(
        single_results,
        ensemble_results,
    )

    calibration_stats = analyze_uncertainty_calibration(
        ensemble_results,
    )

    # --------------------------------------------------------
    # 7. Visualizations
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_uncertainty_vs_error(
        single_results,
        ensemble_results,
    )

    plot_uncertainty_distribution(
        ensemble_results,
    )

    plot_ensemble_std_vs_error(
        ensemble_results,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print("\nKey Research Finding:")
    print(
        f"  Correlation between ensemble uncertainty and error: "
        f"r={correlation_stats['pearson_r']:.4f}"
    )
    print(
        f"  Top-20% error capture: "
        f"{100 * calibration_stats['top_error_capture']:.1f}%"
    )

    if abs(correlation_stats['pearson_r']) > 0.7:
        print(
            "  ✓ SUCCESS: Ensemble can predict model failure regions"
        )
        print("  → Proceed to uncertainty-aware MPPI")
    else:
        print(
            "  ~ MODERATE: Consider alternative uncertainty estimators"
        )
        print("  → Investigate epistemic vs aleatoric uncertainty")
