import numpy as np
import pytest

from ua_mbrl.utils.stats import binary_auc, exact_sign_test, summarize_t, summarize_z, wilson_interval


@pytest.mark.parametrize(
    "fixed_wins, adaptive_wins, published_p",
    [(7, 3, 0.344), (7, 1, 0.070), (7, 0, 0.016), (14, 4, 0.031), (3, 0, 0.25)],
)
def test_sign_test_matches_paper(fixed_wins, adaptive_wins, published_p):
    assert exact_sign_test(fixed_wins, adaptive_wins) == pytest.approx(published_p, abs=5e-4)


def test_sign_test_no_discordant_pairs():
    assert np.isnan(exact_sign_test(0, 0))


@pytest.mark.parametrize(
    "successes, published",
    # sec:morekfixed and the sec:claima Wilson table, n=40 (the paper's
    # rounding differs by up to 0.1 point, e.g. 76.95 printed as 77.0).
    [(36, (77.0, 96.0)), (33, (68.1, 91.3)), (39, (87.1, 99.6)), (40, (91.2, 100.0)),
     (30, (59.8, 85.8)), (20, (35.2, 64.8))],
)
def test_wilson_matches_paper(successes, published):
    low, high = wilson_interval(successes, 40)
    assert 100 * low == pytest.approx(published[0], abs=0.1)
    assert 100 * high == pytest.approx(published[1], abs=0.1)


def test_t_interval_is_wider_than_z():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert summarize_t(values)[1] > summarize_z(values)[1]


def test_binary_auc():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert binary_auc(scores, labels) == 1.0
    assert binary_auc(scores, 1 - labels) == 0.0
