"""Elo 拟合。"""

import numpy as np
import pytest

from cornerstone.elo import elo_from_score_rate, fit_elo, score_rate_from_elo


def test_score_rate_roundtrip():
    for diff in [-800, -400, -100, 0, 100, 400, 800]:
        assert elo_from_score_rate(score_rate_from_elo(diff)) == pytest.approx(diff, abs=1e-6)
    assert score_rate_from_elo(0) == pytest.approx(0.5)
    assert elo_from_score_rate(0.5) == pytest.approx(0.0)


def test_fit_recovers_known_ratings():
    true_elo = np.array([0.0, 150.0, 320.0, 500.0, 760.0])
    n = len(true_elo)
    games_per_pair = 20000

    scores = np.zeros((n, n))
    counts = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rate = score_rate_from_elo(true_elo[i] - true_elo[j])
            scores[i, j] = rate * games_per_pair
            counts[i, j] = games_per_pair

    est = fit_elo(scores, counts, anchor=0, prior=0.0)
    assert np.allclose(est, true_elo, atol=5.0), est


def test_fit_handles_a_perfect_agent():
    """全胜时极大似然解是无穷大，先验必须把它拉回有限值。"""
    n = 3
    counts = np.full((n, n), 100.0)
    np.fill_diagonal(counts, 0.0)
    scores = np.zeros((n, n))
    # 2 全胜，1 赢 0
    scores[2, 0] = scores[2, 1] = 100.0
    scores[1, 0] = 100.0
    scores[0, 1] = scores[0, 2] = scores[1, 2] = 0.0

    est = fit_elo(scores, counts, anchor=0, prior=1.0)
    assert np.all(np.isfinite(est))
    assert est[2] > est[1] > est[0]


def test_anchor_is_respected():
    counts = np.array([[0.0, 100.0], [100.0, 0.0]])
    scores = np.array([[0.0, 25.0], [75.0, 0.0]])
    for anchor, value in [(0, 0.0), (1, 0.0), (0, 1500.0)]:
        est = fit_elo(scores, counts, anchor=anchor, anchor_elo=value)
        assert est[anchor] == pytest.approx(value)
    # 25% 得分率约等于落后 191 Elo
    est = fit_elo(scores, counts, anchor=0, prior=0.0)
    assert est[1] - est[0] == pytest.approx(elo_from_score_rate(0.75), abs=1.0)


def test_draws_count_as_half():
    counts = np.array([[0.0, 100.0], [100.0, 0.0]])
    all_draws = np.array([[0.0, 50.0], [50.0, 0.0]])
    est = fit_elo(all_draws, counts, anchor=0, prior=0.0)
    assert est[1] - est[0] == pytest.approx(0.0, abs=1e-6)
