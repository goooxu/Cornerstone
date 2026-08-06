"""从两两对局结果拟合 Elo。

用 Bradley-Terry 模型加 MM（minorization-maximization）迭代求极大似然解，
和局按半胜计。相比「直接由胜率换算」，这样能把整个阶梯的信息联合起来，
中间档位的估计会稳得多。

带一点先验（默认对每个对手额外加 1 局和棋）：否则某个智能体全胜时
Bradley-Terry 的极大似然解是无穷大。
"""

from __future__ import annotations

import numpy as np

ELO_SCALE = 400.0


def elo_from_score_rate(rate: float) -> float:
    """由单一对局的得分率换算 Elo 差。得分率含和局（和局算 0.5）。"""
    rate = min(max(rate, 1e-9), 1 - 1e-9)
    return -ELO_SCALE * np.log10(1.0 / rate - 1.0)


def score_rate_from_elo(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE))


def fit_elo(
    scores: np.ndarray,
    games: np.ndarray,
    anchor: int = 0,
    anchor_elo: float = 0.0,
    prior: float = 1.0,
    iters: int = 10000,
    tol: float = 1e-12,
) -> np.ndarray:
    """拟合 Elo。

    scores[i, j]: i 对 j 拿到的分数（胜 1、和 0.5、负 0）
    games[i, j]:  i 与 j 的总对局数（应与 games[j, i] 相同）
    anchor:       把哪个下标固定为 anchor_elo
    prior:        对每一对额外加多少局和棋，用于正则化
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    games = np.asarray(games, dtype=np.float64).copy()
    n = scores.shape[0]

    if prior > 0:
        off = ~np.eye(n, dtype=bool)
        games[off] += prior
        scores[off] += prior / 2.0

    p = np.ones(n, dtype=np.float64)
    total_score = scores.sum(axis=1)

    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            m = games[i] > 0
            denom[i] = np.sum(games[i][m] / (p[i] + p[m]))
        new_p = np.where(denom > 0, total_score / np.maximum(denom, 1e-300), p)
        new_p = np.maximum(new_p, 1e-300)
        new_p /= np.exp(np.mean(np.log(new_p)))  # 归一化，避免整体漂移
        if np.max(np.abs(np.log(new_p) - np.log(p))) < tol:
            p = new_p
            break
        p = new_p

    elo = ELO_SCALE * np.log10(p)
    return elo - elo[anchor] + anchor_elo


def elo_stderr(scores: np.ndarray, games: np.ndarray) -> np.ndarray:
    """每个智能体 Elo 的粗略标准误。

    只按「总得分服从二项分布」估计，忽略对手强度的不确定性，
    所以是个下界，用来判断阶梯档位之间是否分得开足够了。
    """
    scores = np.asarray(scores, dtype=np.float64)
    games = np.asarray(games, dtype=np.float64)
    n = scores.shape[0]
    out = np.zeros(n)
    for i in range(n):
        g = games[i].sum()
        if g <= 0:
            out[i] = np.inf
            continue
        rate = scores[i].sum() / g
        rate = min(max(rate, 1e-6), 1 - 1e-6)
        se_rate = np.sqrt(rate * (1 - rate) / g)
        # d(Elo)/d(rate) = 400 / (ln10 * rate * (1-rate))
        out[i] = ELO_SCALE * se_rate / (np.log(10) * rate * (1 - rate))
    return out
