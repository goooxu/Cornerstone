"""对局评测框架。

要点：
 - 先后手逐局交换（`play_match` 里偶数局 A 执先），Blokus Duo 有约 4 个百分点的
   先手优势，不交换的话结果没有意义
 - 开局若干手双方均匀随机，把确定性智能体之间的重复对局打散
 - 和局记 0.5 分
 - Elo 用整个阶梯的两两结果联合拟合，锚定在 random = 0
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import _engine as E
from .elo import elo_stderr, fit_elo

# 纯规则构造的基线阶梯。名字即 CLI 里可用的标识。
BASELINES: dict[str, E.AgentConfig] = {
    "random": E.AgentConfig(kind=E.AgentKind.Random),
    "greedy-area": E.AgentConfig(kind=E.AgentKind.GreedyArea),
    # 权重由 tools/tune_mobility.py 扫出，见 docs/03-基线与评测.md
    "greedy-mobility": E.AgentConfig(
        kind=E.AgentKind.GreedyMobility, w_size=4.0, w_own_anchors=0.5, w_opp_anchors=20.0
    ),
    "flat-mcts-256": E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=256),
    "flat-mcts-1k": E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=1024),
    "flat-mcts-4k": E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=4096),
    "flat-mcts-16k": E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=16384),
}

DEFAULT_LADDER = [
    "random",
    "greedy-area",
    "greedy-mobility",
    "flat-mcts-256",
    "flat-mcts-1k",
    "flat-mcts-4k",
]


@dataclass
class PairResult:
    a: str
    b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int
    score_a: float          # 含和局的得分（胜 1 和 0.5）
    mean_squares_a: float
    mean_squares_b: float
    mean_plies: float
    a_wins_as_first: int
    a_wins_as_second: int

    @property
    def score_rate_a(self) -> float:
        return self.score_a / self.games if self.games else 0.0


@dataclass
class ArenaResult:
    names: list[str]
    pairs: list[PairResult]
    scores: np.ndarray = field(repr=False)   # scores[i, j]
    games: np.ndarray = field(repr=False)
    elo: np.ndarray = field(repr=False)
    elo_se: np.ndarray = field(repr=False)

    def table(self) -> str:
        order = np.argsort(-self.elo)
        w = max(len(n) for n in self.names) + 2
        lines = [f"{'智能体':<{w}}{'Elo':>9}{'±':>7}{'对局':>8}{'得分率':>9}"]
        lines.append("-" * (w + 33))
        for i in order:
            g = self.games[i].sum()
            rate = self.scores[i].sum() / g if g else 0.0
            lines.append(
                f"{self.names[i]:<{w}}{self.elo[i]:>9.1f}{self.elo_se[i]:>7.1f}"
                f"{int(g):>8}{rate:>9.3f}"
            )
        return "\n".join(lines)

    def pair_table(self) -> str:
        w = max(len(p.a) for p in self.pairs) + 1
        lines = [
            f"{'A':<{w}}{'B':<{w}}{'局数':>6}{'A胜':>6}{'和':>5}{'B胜':>6}"
            f"{'A得分率':>10}{'先手胜':>8}{'后手胜':>8}{'平均手数':>10}"
        ]
        lines.append("-" * (2 * w + 59))
        for p in self.pairs:
            lines.append(
                f"{p.a:<{w}}{p.b:<{w}}{p.games:>6}{p.wins_a:>6}{p.draws:>5}{p.wins_b:>6}"
                f"{p.score_rate_a:>10.3f}{p.a_wins_as_first:>8}{p.a_wins_as_second:>8}"
                f"{p.mean_plies:>10.1f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "elo": self.elo.tolist(),
            "elo_se": self.elo_se.tolist(),
            "scores": self.scores.tolist(),
            "games": self.games.tolist(),
            "pairs": [vars(p) for p in self.pairs],
        }


def play_pair(
    name_a: str,
    name_b: str,
    cfg_a: E.AgentConfig,
    cfg_b: E.AgentConfig,
    games: int,
    seed: int,
    threads: int,
    opening_plies: int,
) -> PairResult:
    if games % 2:
        raise ValueError("对局数必须是偶数，否则先后手分配不平衡")
    r = E.play_match(cfg_a, cfg_b, games, seed, threads, opening_plies)
    return PairResult(
        a=name_a,
        b=name_b,
        games=r["games"],
        wins_a=r["wins_a"],
        wins_b=r["wins_b"],
        draws=r["draws"],
        score_a=r["wins_a"] + 0.5 * r["draws"],
        mean_squares_a=r["score_a"] / r["games"],
        mean_squares_b=r["score_b"] / r["games"],
        mean_plies=r["plies"] / r["games"],
        a_wins_as_first=r["a_wins_as_first"],
        a_wins_as_second=r["a_wins_as_second"],
    )


def round_robin(
    names: list[str],
    agents: dict[str, E.AgentConfig] | None = None,
    games: int = 200,
    seed: int = 1,
    threads: int = 1,
    opening_plies: int = 4,
    anchor: str = "random",
    progress=None,
) -> ArenaResult:
    """两两循环赛，返回结果与拟合出的 Elo。"""
    reg = dict(BASELINES)
    if agents:
        reg.update(agents)
    missing = [n for n in names if n not in reg]
    if missing:
        raise KeyError(f"未知的智能体: {missing}")

    n = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    scores = np.zeros((n, n))
    counts = np.zeros((n, n))
    pairs: list[PairResult] = []

    for k, (a, b) in enumerate(itertools.combinations(names, 2)):
        if progress:
            progress(a, b)
        pr = play_pair(a, b, reg[a], reg[b], games, seed + 1000 * k, threads, opening_plies)
        pairs.append(pr)
        i, j = idx[a], idx[b]
        scores[i, j] += pr.score_a
        scores[j, i] += pr.games - pr.score_a
        counts[i, j] += pr.games
        counts[j, i] += pr.games

    anchor_i = idx.get(anchor, 0)
    elo = fit_elo(scores, counts, anchor=anchor_i, anchor_elo=0.0)
    return ArenaResult(
        names=list(names),
        pairs=pairs,
        scores=scores,
        games=counts,
        elo=elo,
        elo_se=elo_stderr(scores, counts),
    )
