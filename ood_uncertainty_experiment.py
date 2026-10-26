"""OOD uncertainty experiment for the pendulum dynamics ensemble.

Train on a narrow state/action distribution, then evaluate on held-out
in-distribution samples and deliberately out-of-distribution samples.
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from generic_mbrl import GenericDynamicsModel, GenericEnsembleDynamics
from robustness_mismatch_experiment import (
    angle_error,
    pendulum_features,
    pendulum_residual,
    pendulum_step,
    pendulum_transition,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EPISODES = 20
TRAIN_STEPS = 60
ENSEMBLE_SIZE = 5
EPOCHS = 100
TEST_SAMPLES = 400
# Matches the hidden width of the retired ensemble_dynamics.DynamicsModel.
HIDDEN_DIM = 64


def collect_narrow_data(seed=0):
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    for _ in range(TRAIN_EPISODES):
        x = np.array(
            [rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)],
            dtype=np.float32,
        )
        for _ in range(TRAIN_STEPS):
            u = rng.uniform(-2.0, 2.0)
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


def make_test_data(seed, omega_range, action_range):
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    for _ in range(TEST_SAMPLES):
        x = np.array(
            [rng.uniform(-np.pi, np.pi), rng.uniform(*omega_range)],
            dtype=np.float32,
        )
        u = rng.uniform(*action_range)
        states.append(x)
        actions.append([u])
        next_states.append(pendulum_step(x, u))
    return {
        "x": np.asarray(states, dtype=np.float32),
        "u": np.asarray(actions, dtype=np.float32),
        "x_next": np.asarray(next_states, dtype=np.float32),
    }


def train_models(data):
    torch.manual_seed(1)
    single = GenericDynamicsModel(
        state_dim=2,
        action_dim=1,
        feature_fn=pendulum_features,
        transition_fn=pendulum_transition,
        residual_fn=pendulum_residual,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    single.train_model(data, epochs=EPOCHS, verbose=False)

    torch.manual_seed(2)
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
    ensemble.train_ensemble(data, epochs=EPOCHS, seed=2, verbose=False)
    return single, ensemble


def evaluate(single, ensemble, data):
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=DEVICE)
    u = torch.as_tensor(data["u"], dtype=torch.float32, device=DEVICE)
    true = data["x_next"]

    with torch.no_grad():
        mean, variance, _ = ensemble.forward_mean_var(x, u)
        uncertainty = torch.sum(variance, dim=1).cpu().numpy()
        ensemble_prediction = mean.cpu().numpy()
        predictions = single(x, u).cpu().numpy()

    ensemble_error = ensemble_prediction - true
    single_error = predictions - true
    ensemble_error[:, 0] = np.asarray([
        angle_error(prediction[0], target[0])
        for prediction, target in zip(ensemble_prediction, true)
    ])
    single_error[:, 0] = np.asarray([
        angle_error(prediction[0], target[0])
        for prediction, target in zip(predictions, true)
    ])

    error_magnitude = np.linalg.norm(ensemble_error, axis=1)
    single_magnitude = np.linalg.norm(single_error, axis=1)
    rank_correlation = spearmanr(uncertainty, error_magnitude).statistic
    top_count = max(1, len(uncertainty) // 5)
    uncertain = set(np.argsort(uncertainty)[-top_count:])
    worst = set(np.argsort(error_magnitude)[-top_count:])

    return {
        "ensemble_rmse": float(np.sqrt(np.mean(ensemble_error ** 2))),
        "single_rmse": float(np.sqrt(np.mean(single_error ** 2))),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "median_uncertainty": float(np.median(uncertainty)),
        "spearman": float(rank_correlation),
        "top_error_capture": len(uncertain & worst) / top_count,
        "uncertainty": uncertainty,
        "error": error_magnitude,
        "single_error": single_magnitude,
    }


def print_result(name, result):
    print(
        f"{name:<16} single_rmse={result['single_rmse']:.5f} "
        f"ensemble_rmse={result['ensemble_rmse']:.5f} "
        f"mean_unc={result['mean_uncertainty']:.5f} "
        f"median_unc={result['median_uncertainty']:.5f} "
        f"spearman={result['spearman']:.3f} "
        f"top20_capture={100 * result['top_error_capture']:.1f}%"
    )


def binary_auc(scores, labels):
    """Compute ROC AUC without adding another dependency."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def evaluate_ood_detection(in_result, ood_result):
    """Evaluate uncertainty as a distribution-shift detector."""
    scores = np.concatenate([
        in_result["uncertainty"],
        ood_result["uncertainty"],
    ])
    labels = np.concatenate([
        np.zeros(len(in_result["uncertainty"])),
        np.ones(len(ood_result["uncertainty"])),
    ])
    threshold = np.percentile(in_result["uncertainty"], 95)
    predicted_ood = scores >= threshold
    true_positive_rate = np.mean(predicted_ood[labels == 1])
    false_positive_rate = np.mean(predicted_ood[labels == 0])
    return {
        "auc": binary_auc(scores, labels),
        "threshold": threshold,
        "tpr": float(true_positive_rate),
        "fpr": float(false_positive_rate),
    }


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    train_data = collect_narrow_data()
    single, ensemble = train_models(train_data)

    in_distribution = make_test_data(
        seed=10,
        omega_range=(-1.0, 1.0),
        action_range=(-2.0, 2.0),
    )
    mild_ood = make_test_data(
        seed=20,
        omega_range=(2.0, 3.0),
        action_range=(2.0, 3.0),
    )
    severe_ood = make_test_data(
        seed=20,
        omega_range=(4.0, 6.0),
        action_range=(4.0, 5.0),
    )

    in_result = evaluate(single, ensemble, in_distribution)
    mild_result = evaluate(single, ensemble, mild_ood)
    severe_result = evaluate(single, ensemble, severe_ood)

    print("\nOOD uncertainty results")
    print("=" * 120)
    print_result("in-distribution", in_result)
    print_result("mild OOD", mild_result)
    print_result("severe OOD", severe_result)

    for name, result in [("mild OOD", mild_result), ("severe OOD", severe_result)]:
        detection = evaluate_ood_detection(in_result, result)
        print(f"\n{name} detection")
        print(f"uncertainty ratio: {result['mean_uncertainty'] / in_result['mean_uncertainty']:.2f}x")
        print(f"ensemble RMSE ratio: {result['ensemble_rmse'] / in_result['ensemble_rmse']:.2f}x")
        print(f"ROC AUC: {detection['auc']:.3f}")
        print(
            f"95%-ID threshold={detection['threshold']:.5f} | "
            f"TPR={100 * detection['tpr']:.1f}% | "
            f"FPR={100 * detection['fpr']:.1f}%"
        )
