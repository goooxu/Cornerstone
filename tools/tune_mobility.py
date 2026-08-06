#!/usr/bin/env python3
"""给 greedy-mobility 挑权重。

评分是 w_size*棋子格数 + w_own*己方角点前沿 - w_opp*对方角点前沿。
三项的量纲差得远（格数 1~5，前沿数常有 10~30），随手取 1:1:1 会让格数项被淹没，
所以扫一遍。用固定对手（greedy-area）打，比得分率。
"""

import argparse
import itertools
import os

from cornerstone import _engine as E
from cornerstone.elo import elo_from_score_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w-size", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--w-own", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--w-opp", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    opponent = E.AgentConfig(kind=E.AgentKind.GreedyArea)

    grid = [
        (ws, wo, wp)
        for ws, wo, wp in itertools.product(args.w_size, args.w_own, args.w_opp)
        if not (wo == 0.0 and wp == 0.0)   # 两项都是 0 就退化成 greedy-area 了
    ]

    print(f"对手 greedy-area，每组 {args.games} 局，{args.threads} 线程")
    print(f"{'w_size':>7}{'w_own':>7}{'w_opp':>7}{'得分率':>9}{'Elo差':>9}")
    print("-" * 39)

    rows = []
    for ws, wo, wp in grid:
        cfg = E.AgentConfig(kind=E.AgentKind.GreedyMobility,
                            w_size=ws, w_own_anchors=wo, w_opp_anchors=wp)
        r = E.play_match(cfg, opponent, args.games, args.seed, args.threads, args.opening_plies)
        rate = (r["wins_a"] + 0.5 * r["draws"]) / r["games"]
        rows.append((rate, ws, wo, wp))
        print(f"{ws:>7.1f}{wo:>7.1f}{wp:>7.1f}{rate:>9.3f}{elo_from_score_rate(rate):>9.1f}")

    rows.sort(reverse=True)
    print()
    print("最好的几组：")
    for rate, ws, wo, wp in rows[:5]:
        print(f"  w_size={ws} w_own={wo} w_opp={wp} -> 得分率 {rate:.3f}"
              f"（Elo 差 {elo_from_score_rate(rate):+.1f}）")


if __name__ == "__main__":
    main()
