#!/usr/bin/env python3
"""把**网络和规则基线放进同一场循环赛**，联合拟合 Elo。

    python3 tools/net_arena.py \
        --nets ../runs/ab-bf16/ckpt/step00000630.pt ../runs/ab-bf16/ckpt/step00200230.pt \
        --rules random greedy-area greedy-mobility flat-mcts-4k \
        --games 400 --engine-threads 128 --out ../runs/arena/bf16_arena.json

为什么需要这个：网络强过全部规则基线之后，`evaluate_vs_baseline` 的得分率会钉在
1.0，换算出来的 Elo 只是钳位假数（`elo_from_score_rate` 把胜率钳到 1−1e-9，
200:0 就变成 +3600）。而 `run_arena.py` 只跑规则之间，`compare_nets.py --curve`
又把所有 checkpoint 都对最弱的那个比，同样全部饱和。

正确做法是**让每条边尽量不饱和**，再靠连通图把尺度传过去：强网络之间互相比，
早期网络去和规则阶梯搭桥，规则之间自己跑一遍锚住零点。Bradley-Terry 会把
整张图的信息联合起来，比单条边换算稳得多。

`--pairs` 控制打哪些对：

    all     全对全（默认）
    bridge  网络之间全打；网络 vs 规则只打「桥接网络」（--bridge 指定）；规则之间全打

饱和的边（比如最终模型对 random）打了也没信息量，只是白烧时间。
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.arena import BASELINES, play_pair          # noqa: E402
from cornerstone.elo import fit_elo                         # noqa: E402


def step_of(path: str) -> int:
    import re
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


class Participant:
    """一个参赛者：要么是规则基线，要么是一份 checkpoint。"""

    def __init__(self, spec: str, device: str):
        self.spec = spec
        if spec in BASELINES:
            self.kind = "rule"
            self.name = spec
            self.cfg = BASELINES[spec]
            self.model = None
        else:
            from cornerstone.model import load_checkpoint
            self.kind = "net"
            self.model, st = load_checkpoint(spec, device)
            self.step = step_of(spec)
            self.name = f"net@{self.step}" if self.step >= 0 else os.path.basename(spec)
            self.cfg = None

    @property
    def is_net(self) -> bool:
        return self.kind == "net"


def play(a: Participant, b: Participant, games: int, sims: int, seed: int,
         device: str, threads: int, opening_plies: int) -> tuple[float, int]:
    """打一对，返回 (a 的得分, 总局数)。得分含和局（和局 0.5）。

    三种组合各走各的路径。**最容易写错的是方向** —— `evaluate_vs_baseline`
    报的是**网络方**的胜负，网络坐 b 位时必须翻过来。这个项目在别处已经因为
    「传参顺序和打印的名字对不上」把结论读反过一次（见 docs/08），所以这里
    把方向写死在一个函数里，并且有单测盯着。
    """
    from cornerstone.evaluate import evaluate_vs_baseline, evaluate_vs_network

    if a.is_net and b.is_net:
        r = evaluate_vs_network(a.model, b.model, device, games=games,
                                simulations=sims, parallel_games=min(128, games),
                                seed=seed, engine_threads=threads,
                                opening_plies=opening_plies)
        return r.wins + 0.5 * r.draws, r.games

    if a.is_net or b.is_net:
        net, rule = (a, b) if a.is_net else (b, a)
        r = evaluate_vs_baseline(net.model, device, opponent=rule.name, games=games,
                                 simulations=sims, parallel_games=min(128, games),
                                 seed=seed, engine_threads=threads,
                                 opening_plies=opening_plies)
        net_score = r.wins + 0.5 * r.draws
        # 网络是 a 就直接用；网络是 b 的话，a 的得分是「总局数减去网络的得分」
        return (net_score if a.is_net else r.games - net_score), r.games

    r = play_pair(a.name, b.name, a.cfg, b.cfg, games, seed, threads, opening_plies)
    return r.score_a, r.games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="*", default=[], help="checkpoint 路径")
    ap.add_argument("--rules", nargs="*", default=[], help=f"可选: {' '.join(BASELINES)}")
    ap.add_argument("--bridge", nargs="*", default=[],
                    help="--pairs bridge 时，哪些 checkpoint 去和规则基线搭桥")
    ap.add_argument("--pairs", choices=["all", "bridge"], default="all")
    ap.add_argument("--net-max-dist", type=int, default=0,
                    help="网络之间只打里程碑距离 <= N 的对（0 = 不限）")
    ap.add_argument("--bootstrap", type=int, default=200, help="自举次数，用来算 ±")
    ap.add_argument("--games", type=int, default=400, help="每对的对局数（必须是偶数）")
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--engine-threads", type=int, default=os.cpu_count() or 16)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor", default="random", help="Elo 零点锚定在谁身上")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.games % 2:
        raise SystemExit("对局数必须是偶数，否则先后手分配不平衡")
    for r in args.rules:
        if r not in BASELINES:
            raise SystemExit(f"未知基线 {r}，可选 {list(BASELINES)}")

    print(f"载入 {len(args.nets)} 份 checkpoint …", flush=True)
    parts = [Participant(s, args.device) for s in args.rules + args.nets]
    idx = {p.name: i for i, p in enumerate(parts)}
    if args.anchor not in idx:
        raise SystemExit(f"锚点 {args.anchor} 不在参赛者里")

    bridge = {step_of(b) for b in args.bridge}
    # 网络按步数排序后的名次，用来限制「只打相邻的几档」
    net_rank = {p.name: r for r, p in
                enumerate(sorted((q for q in parts if q.is_net), key=lambda q: q.step))}

    def wanted(a: Participant, b: Participant) -> bool:
        if a.is_net and b.is_net:
            # 相隔太远的两档必然是碾压，打了没信息量；而且**纯链式**（只打相邻）
            # 会让每段的偏差线性累加，所以要留 2、4 档的斜拉边做三角校正。
            if args.net_max_dist <= 0:
                return True
            return abs(net_rank[a.name] - net_rank[b.name]) <= args.net_max_dist
        if not a.is_net and not b.is_net:
            return True                      # 规则之间全打，锚住零点
        if args.pairs == "all":
            return True
        net = a if a.is_net else b
        return net.step in bridge            # 混合对只打桥接的那几个

    todo = [(i, j) for i, j in itertools.combinations(range(len(parts)), 2)
            if wanted(parts[i], parts[j])]
    print(f"{len(parts)} 个参赛者，{len(todo)} 对 × {args.games} 局，"
          f"{args.simulations} 次模拟，{args.engine_threads} 引擎线程", flush=True)

    n = len(parts)
    scores = np.zeros((n, n))
    games = np.zeros((n, n))
    rows = []
    t0 = time.perf_counter()
    for k, (i, j) in enumerate(todo):
        a, b = parts[i], parts[j]
        t = time.perf_counter()
        sa, g = play(a, b, args.games, args.simulations, args.seed + 1000 * k,
                     args.device, args.engine_threads, args.opening_plies)
        scores[i, j] += sa
        scores[j, i] += g - sa
        games[i, j] += g
        games[j, i] += g
        rate = sa / max(1, g)
        print(f"  [{k+1:>3}/{len(todo)}] {a.name} vs {b.name}: "
              f"{rate:.3f}  ({time.perf_counter()-t:.1f}s)", flush=True)
        rows.append({"a": a.name, "b": b.name, "games": g,
                     "score_a": sa, "rate_a": rate})

    anchor_i = idx[args.anchor]
    elo = fit_elo(scores, games, anchor=anchor_i)

    # ± 用参数自举，不用 elo_stderr。后者是按「总得分率服从二项分布」估的，
    # 依赖每个参赛者碰到的对手组合 —— 而这里各人打的对子并不一样
    # （网络只打相邻档、桥接只打几对），那个 ± 跨行不可比。
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        s2 = np.zeros_like(scores)
        for i, j in itertools.combinations(range(n), 2):
            g = games[i, j]
            if g <= 0:
                continue
            w = rng.binomial(int(round(g)), min(max(scores[i, j] / g, 0.0), 1.0))
            s2[i, j], s2[j, i] = w, g - w
        boot.append(fit_elo(s2, games, anchor=anchor_i))
    err = np.std(np.array(boot), axis=0) if boot else np.zeros(n)
    order = np.argsort(-elo)

    # 方向自检：早期网络必须打不过 flat-mcts-4k，最终网络必须稳赢它。
    # 整张表上下颠倒是这类工具最典型也最难发现的错法（见 docs/08），
    # 所以宁可在这里硬崩，也不要输出一张看着合理的反表。
    def _rate(x: str, y: str):
        for r in rows:
            if r["a"] == x and r["b"] == y:
                return r["rate_a"]
            if r["a"] == y and r["b"] == x:
                return 1.0 - r["rate_a"]
        return None
    nets_sorted = sorted((p for p in parts if p.is_net), key=lambda q: q.step)
    if nets_sorted and "flat-mcts-4k" in idx:
        lo = _rate(nets_sorted[0].name, "flat-mcts-4k")
        hi = _rate(nets_sorted[-1].name, "flat-mcts-4k")
        if lo is not None and lo > 0.5:
            raise SystemExit(f"方向自检失败：最早的网络 {nets_sorted[0].name} "
                             f"对 flat-mcts-4k 得分率 {lo:.3f}，不该 > 0.5")
        if hi is not None and hi < 0.9:
            raise SystemExit(f"方向自检失败：最终网络 {nets_sorted[-1].name} "
                             f"对 flat-mcts-4k 得分率 {hi:.3f}，不该 < 0.9")

    print(f"\n{len(todo)} 对打完，用时 {time.perf_counter()-t0:.1f}s\n")
    print(f"{'参赛者':<22}{'Elo':>9}{'±':>7}{'总得分率':>10}{'局数':>8}")
    print("-" * 58)
    for i in order:
        tot = games[i].sum()
        print(f"{parts[i].name:<22}{elo[i]:>9.1f}{err[i]:>7.1f}"
              f"{scores[i].sum()/max(1,tot):>10.3f}{int(tot):>8}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "participants": [{"name": p.name, "kind": p.kind, "spec": p.spec,
                                  "elo": float(elo[i]), "stderr": float(err[i]),
                                  "games": int(games[i].sum()),
                                  "score_rate": float(scores[i].sum() / max(1, games[i].sum()))}
                                 for i, p in enumerate(parts)],
                "pairs": rows,
                "config": {"games": args.games, "simulations": args.simulations,
                           "opening_plies": args.opening_plies, "anchor": args.anchor,
                           "seed": args.seed, "pairs_mode": args.pairs},
            }, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
