#!/usr/bin/env python3
"""两档之间的 Elo 差值有多确定 —— 直接自举差值，而不是把两个 ± 相加。

    python3 tools/arena_diff.py ../runs/arena_v2_bf16.json net@120227 net@150227 ...

为什么要单独算：`net_arena` 表里的 ± 是**绝对刻度**的误差，被「整条曲线相对
`random` 一起平移」的不确定度主导（那份不确定度来自网络与规则阶梯之间那条很弱的
边）。比较同一场里的两档时这份平移会整体抵消，差值的误差要小得多 ——
把两个 ±33 相加会得出「谁也分不出谁」，而实测差值误差只有 ±10 量级。

第一个名字是基准，其余每个都与它比。
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.elo import fit_elo                         # noqa: E402
from tools.arena_best import load                           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("names", nargs="+", help="第一个是基准，其余与它比")
    ap.add_argument("--anchor", default="random")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    names, idx, S, G, _ = load(args.json)
    n = len(names)
    for nm in args.names:
        if nm not in idx:
            raise SystemExit(f"{nm} 不在这场里；有 {', '.join(names)}")
    anchor = idx[args.anchor] if args.anchor in idx else 0
    elo = fit_elo(S, G, anchor=anchor)

    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.boot, n))
    for b in range(args.boot):
        s2 = np.zeros_like(S)
        for i, j in itertools.combinations(range(n), 2):
            g = G[i, j]
            if g <= 0:
                continue
            rate = min(max(S[i, j] / g, 0.0), 1.0)
            npair = max(1, int(round(g / 2)))
            sa = rng.binomial(npair, rate) / npair * g
            s2[i, j], s2[j, i] = sa, g - sa
        boot[b] = fit_elo(s2, G, anchor=anchor)

    base = idx[args.names[0]]
    print(f"基准 {args.names[0]}（Elo {elo[base]:.1f}）")
    print(f"{'对手':<16}{'Δ Elo':>9}{'±':>7}{'P(基准更强)':>14}{'直接头对头':>12}")
    print("-" * 60)
    for nm in args.names[1:]:
        j = idx[nm]
        d = boot[:, base] - boot[:, j]
        h2h = S[base, j] / G[base, j] if G[base, j] else float("nan")
        print(f"{nm:<16}{elo[base]-elo[j]:>+9.1f}{d.std():>7.1f}"
              f"{100*(d > 0).mean():>13.1f}%{h2h:>12.3f}")


if __name__ == "__main__":
    main()
