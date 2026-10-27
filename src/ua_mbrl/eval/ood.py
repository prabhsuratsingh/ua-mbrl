"""One-step prediction error / ensemble disagreement and OOD-detection metrics."""

import numpy as np
import torch
from scipy.stats import spearmanr

from ua_mbrl.envs.pendulum import angle_error
from ua_mbrl.utils.device import DEVICE
from ua_mbrl.utils.stats import binary_auc


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
