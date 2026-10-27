"""Statistics used across experiments and the paper's tables."""

import math

import numpy as np
from scipy import stats as scipy_stats


def summarize_z(values):
    """Mean and normal-approximation 95% half-width (1.96 * SE) -- what the
    experiment scripts print."""
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


def summarize_t(values, confidence=0.95):
    """Mean and Student-t half-width -- the interval used in the paper's tables."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    t = scipy_stats.t.ppf(0.5 + confidence / 2, len(values) - 1)
    return mean, float(t * np.std(values, ddof=1) / np.sqrt(len(values)))


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion, as (low, high)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half, center + half


def exact_sign_test(fixed_wins, adaptive_wins):
    """Two-sided exact sign test (equivalently, exact McNemar's test) on
    discordant pairs: under H0 (fixed and adaptive equally likely to win a
    discordant pair), wins ~ Binomial(n_discordant, 0.5)."""
    n = fixed_wins + adaptive_wins
    if n == 0:
        return float("nan")
    k = min(fixed_wins, adaptive_wins)
    # P(X <= k) doubled, capped at 1.0 (standard exact two-sided binomial test)
    p_le_k = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p_le_k)


def binary_auc(scores, labels):
    """ROC AUC by exhaustive pairwise comparison (ties count one half)."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )
