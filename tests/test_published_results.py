"""The paper's MuJoCo and crossover numbers, recomputed from the stored raw
results (results/published/) with the package's own statistics/analysis."""

import pytest

from ua_mbrl.experiments.crossover import analyze_crossover
from ua_mbrl.utils.results import PUBLISHED_DIR, read_jsonl
from ua_mbrl.utils.stats import exact_sign_test, summarize_t


def _pct(values):
    mean, ci = summarize_t(values)
    return round(100 * mean, 1), round(100 * ci, 1)


def test_tab_mujoco_regulation():
    records = read_jsonl(PUBLISHED_DIR / "mujoco_multiseed_results.jsonl")
    assert len(records) == 15
    assert _pct([r["single"]["success_rate"] for r in records]) == (42.2, 19.1)
    assert _pct([r["ensemble"]["success_rate"] for r in records]) == (55.6, 16.6)
    assert _pct([r["adaptive"]["success_rate"] for r in records]) == (62.2, 13.7)
    k_mean, k_ci = summarize_t([r["adaptive"]["effective_rollouts"] for r in records])
    assert (round(k_mean, 1), round(k_ci, 1)) == (205.7, 33.8)


def test_tab_mujoco_reaching():
    records = read_jsonl(PUBLISHED_DIR / "mujoco_reacher_results.jsonl")
    assert len(records) == 10
    assert _pct([r["single"]["success_rate"] for r in records]) == (16.7, 11.2)
    assert _pct([r["ensemble"]["success_rate"] for r in records]) == (33.3, 15.9)
    distance = summarize_t([r["single"]["final_distance"] for r in records])
    assert (round(distance[0], 3), round(distance[1], 3)) == (0.344, 0.086)
    distance = summarize_t([r["ensemble"]["final_distance"] for r in records])
    assert (round(distance[0], 3), round(distance[1], 3)) == (0.193, 0.030)


def test_mujoco_retuned_paragraph():
    records = read_jsonl(PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl")
    assert len(records) == 8
    assert _pct([r["single"]["success_rate"] for r in records]) == (45.8, 33.1)
    assert _pct([r["ensemble"]["success_rate"] for r in records]) == (70.8, 23.3)
    assert _pct([r["adaptive"]["success_rate"] for r in records]) == (70.8, 31.4)


def test_mujoco_claim_a_replication():
    records = read_jsonl(PUBLISHED_DIR / "mujoco_claim_a_results.jsonl")
    fixed = [r["fixed_matched_best_success"] for r in records]
    adaptive = [r["adaptive_success"] for r in records]
    assert _pct(fixed) == (83.3, 21.1)
    assert _pct(adaptive) == (70.8, 31.4)
    wins = sum(f > a for f, a in zip(fixed, adaptive))
    losses = sum(f < a for f, a in zip(fixed, adaptive))
    assert (wins, losses) == (3, 0)
    assert exact_sign_test(wins, losses) == 0.25


def test_crossover_location():
    analysis = analyze_crossover(read_jsonl(PUBLISHED_DIR / "adaptive_k_fine_results.jsonl"))
    # Local interpolation between K=15000 and K=16000 (paper: ~15,600).
    assert 15000 < analysis["interpolated_crossover"] < 16000
    assert round(analysis["interpolated_crossover"], -2) == 15600
    # Global affine regression (paper eq:kstar: K* ~ 14,773, R^2 = 0.998).
    regression = analysis["regression"]
    assert round(regression["k_star"]) == pytest.approx(14773, abs=1)
    assert round(regression["r_squared"], 3) == 0.998


def test_fig2_values_match_published_results():
    from ua_mbrl import figures
    import inspect

    published = {r["k_max"]: r["ratio"] for r in read_jsonl(PUBLISHED_DIR / "adaptive_k_fine_results.jsonl")}
    source = inspect.getsource(figures.fig2_crossover)
    for k_max, ratio in published.items():
        assert f"{ratio:.4f}" in source, f"fig2 ratio for K={k_max} not {ratio:.4f}"
