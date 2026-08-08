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
    engine_threads: int = 16,
) -> EvalResult:
    """网络 vs 规则基线。

    `engine_threads` 一定要给够。这个参数是补上去的 —— 之前这里根本没有它，
    于是 `SelfPlayDriver` 吃默认值 1，在 144 核的机器上**单线程**跑树搜索。
    对手是 `flat-mcts-4k` 这种每手 4096 次 rollout 的基线时，
    200 局要跑 25 分钟；BF16 那条正式训练里 50 次周期评测因此吃掉了
    21 小时墙钟，比训练加自博弈加起来还多。
    """
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
                            compile_model=compile_model, engine_threads=engine_threads)

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


@torch.no_grad()
def evaluate_vs_network(
    model_a,
    model_b,
    device,
    games: int = 200,
    simulations: int = 64,
    parallel_games: int = 128,
    opening_plies: int = 4,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    engine_threads: int = 16,
    label: str = "对手网络",
) -> EvalResult:
    """网络 vs 网络。

    网络强过全部规则基线之后（对 greedy-area 打到 200:0 就是这种情况），
    规则阶梯量不出强度了 —— 得分率钉在 1.0，换算出来的 Elo 只是钳位产生的假数。
    这时唯一还能继续量的办法是和自己的历史 checkpoint 打。

    两方各自建树、各用各的网络。`prepare()` 会告诉我们每个待评估局面归谁算，
    按标记分组前向再合并即可。
    """
    import contextlib

    import numpy as np

    from . import _engine as E
    from .model import ACTIONS, BOARD, PLANES, SCALARS

    dev = torch.device(device)
    parallel = max(2, min(parallel_games, games))
    mcts = E.MctsConfig(simulations=simulations, max_considered=16, temperature_plies=0)
    ev = E.EvalConfig(enabled=True, net_opponent=True, opening_plies=opening_plies)
    eng = E.SelfPlayEngine(parallel, mcts, seed, ev, max(1, engine_threads))

    planes = np.zeros((parallel, PLANES, BOARD, BOARD), dtype=np.float32)
    scalars = np.zeros((parallel, SCALARS), dtype=np.float32)
    which = np.zeros(parallel, dtype=np.int8)
    logits = np.zeros((parallel, ACTIONS), dtype=np.float32)
    wdl = np.zeros((parallel, 3), dtype=np.float32)

    models = (model_a.eval(), model_b.eval())
    recs: list[dict] = []
    while len(recs) < games:
        n = eng.prepare(planes, scalars, which)
        if n == 0:
            recs.extend(eng.advance())
            continue
        for tag, model in enumerate(models):
            idx = np.flatnonzero(which[:n] == tag)
            if idx.size == 0:
                continue
            # 当前 CUDA 设备必须与张量设备一致，否则 TE 的 FP8 GEMM 会启动失败
            with (torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()):
                p = torch.from_numpy(planes[idx]).to(device)
                s = torch.from_numpy(scalars[idx]).to(device)
                with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
                    pol, w, _ = model(p, s)
            logits[idx] = pol.float().cpu().numpy()
            wdl[idx] = w.float().softmax(dim=-1).cpu().numpy()
        eng.feed(logits[:n], wdl[:n])

    wins = draws = losses = plies = sq_a = sq_b = w_first = w_second = 0
    for r in recs:
        p = r["net_player"]
        res = r["result0"] if p == 0 else -r["result0"]
        if res > 0:
            wins += 1
            w_first += (p == 0)
            w_second += (p == 1)
        elif res < 0:
            losses += 1
        else:
            draws += 1
        plies += len(r["actions"])
        sq_a += r["score0"] if p == 0 else r["score1"]
        sq_b += r["score1"] if p == 0 else r["score0"]

    n = len(recs)
    rate = (wins + 0.5 * draws) / n
    return EvalResult(
        opponent=label, games=n, wins=wins, draws=draws, losses=losses,
        score_rate=rate, elo_diff=elo_from_score_rate(rate), elo_abs=None,
        mean_plies=plies / n, mean_squares_net=sq_a / n, mean_squares_opp=sq_b / n,
        wins_as_first=w_first, wins_as_second=w_second,
    )
