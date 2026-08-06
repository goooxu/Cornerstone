"""把网络放到规则基线阶梯上量强度。

复用自博弈的批量通路：网络方用 Gumbel-AZ 搜索，对手用 C++ 基线直接选点，
先后手逐局交换、开局随机若干手。得分率换算成相对该基线的 Elo 差，
再加上基线在阶梯里的绝对 Elo，就得到网络的绝对强度。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _engine as E
from .arena import BASELINES
from .elo import elo_from_score_rate
from .selfplay import SelfPlayDriver

# tools/run_arena.py 实测的阶梯，锚定 random = 0
BASELINE_ELO = {
    "random": 0.0,
    "greedy-area": 448.4,
    "flat-mcts-256": 537.0,
    "flat-mcts-1k": 733.2,
    "greedy-mobility": 836.7,
    "flat-mcts-4k": 991.6,
}


@dataclass
class EvalResult:
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float
    elo_diff: float
    elo_abs: float | None
    mean_plies: float
    mean_squares_net: float
    mean_squares_opp: float
    wins_as_first: int
    wins_as_second: int

    def __str__(self) -> str:
        abs_s = f"，绝对 Elo≈{self.elo_abs:.0f}" if self.elo_abs is not None else ""
        return (f"vs {self.opponent}: {self.games} 局 "
                f"{self.wins}胜/{self.draws}和/{self.losses}负 "
                f"得分率 {self.score_rate:.3f}（Elo 差 {self.elo_diff:+.0f}{abs_s}）")


@torch.no_grad()
def evaluate_vs_baseline(
    model,
    device,
    opponent: str = "greedy-area",
    games: int = 200,
    simulations: int = 64,
    parallel_games: int = 128,
    opening_plies: int = 4,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    max_seconds: float | None = None,
    compile_model: bool = False,
) -> EvalResult:
    if opponent not in BASELINES:
        raise KeyError(f"未知基线 {opponent}，可选 {list(BASELINES)}")

    mcts = E.MctsConfig(
        simulations=simulations,
        max_considered=16,
        temperature_plies=0,     # 评测时不加采样噪声，要看确定性棋力
    )
    ev = E.EvalConfig(enabled=True, opponent=BASELINES[opponent], opening_plies=opening_plies)
    driver = SelfPlayDriver(model, device, num_games=min(parallel_games, games),
                            mcts=mcts, seed=seed, eval_cfg=ev, dtype=dtype,
                            compile_model=compile_model)

    recs, _ = driver.run(games, max_seconds=max_seconds)
    if not recs:
        raise RuntimeError("一局都没跑完，检查 max_seconds 或 parallel_games")

    wins = draws = losses = 0
    plies = sq_net = sq_opp = 0
    w_first = w_second = 0
    for r in recs:
        p = r["net_player"]
        res = r["result0"] if p == 0 else -r["result0"]
        if res > 0:
            wins += 1
            if p == 0:
                w_first += 1
            else:
                w_second += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1
        plies += len(r["actions"])
        sq_net += r["score0"] if p == 0 else r["score1"]
        sq_opp += r["score1"] if p == 0 else r["score0"]

    n = len(recs)
    rate = (wins + 0.5 * draws) / n
    diff = elo_from_score_rate(rate)
    base = BASELINE_ELO.get(opponent)
    return EvalResult(
        opponent=opponent, games=n, wins=wins, draws=draws, losses=losses,
        score_rate=rate, elo_diff=diff,
        elo_abs=(base + diff) if base is not None else None,
        mean_plies=plies / n, mean_squares_net=sq_net / n, mean_squares_opp=sq_opp / n,
        wins_as_first=w_first, wins_as_second=w_second,
    )


def evaluate_ladder(model, device, opponents=("random", "greedy-area", "greedy-mobility"),
                    games: int = 200, **kw) -> list[EvalResult]:
    return [evaluate_vs_baseline(model, device, opponent=o, games=games, **kw)
            for o in opponents]
