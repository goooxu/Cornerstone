#!/usr/bin/env python3
"""把 arena JSON 重新拟合成 Elo 表，可以指定基准，也可以把多场合并起来一起拟合。

    python3 tools/arena_table.py ../runs/arena_v4-fp8-anneal.json
    python3 tools/arena_table.py ../runs/arena_v4-fp8-anneal.json --ref net@111391
    python3 tools/arena_table.py ../runs/arena_v4-cross*.json --ref bf16-111k.pt

**合并的前提是同名即同模型。** 多场之间靠参赛者名对齐，所以跨场比较时
一定要用同一套软链名（见 `runs/_arena/`），不能靠 `net@步数` —— 不同的跑
在同一步数上是**不同的模型**，`net@111389` 在两条轨迹上各有一个，合并会把
它们悄悄当成同一个。名字里带上跑的来源（`fork-` / `main-` / `bf16-`）就不会。

为什么需要它：`net_arena.py` 打印的 Elo 表把**第一个参赛者**当 0 点，
而跨场比较时想看的往往是「相对某个特定模型」。基准换了，整张表平移，
**名次和差值不变** —— 但读的人会因为符号写反而把结论读反。这个项目已经
因为符号读反过一次（把「111k 比 144k 低 29 Elo」说成「144k 比 111k 低」），
所以这里强制把基准打在表头上，并且**同时打印直接对局的得分率**：
得分率是原始观测，不依赖任何拟合，读反了一眼能看出来。

`±` 用**对级**自举：400 局 = 200 对（同一个开局、执先方相反），
重采样的单位是对而不是局。单格标准误约 ±0.035 得分率 ≈ ±10 Elo，
所以**一格一格地读差值是读不出结论的**，见 docs/08。
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.elo import fit_elo                          # noqa: E402


def load(paths: list[str]):
    """读一场或多场，按参赛者名合并成一张对局矩阵。"""
    names: list[str] = []
    idx: dict[str, int] = {}
    pairs = []
    for path in paths:
        d = json.load(open(path))
        for p in d["participants"]:
            n = p["name"] if isinstance(p, dict) else p
            if n not in idx:
                idx[n] = len(names)
                names.append(n)
        pairs.extend(d["pairs"])

    n = len(names)
    S = np.zeros((n, n))
    G = np.zeros((n, n))
    for p in pairs:
        a, b = idx[p["a"]], idx[p["b"]]
        S[a, b] += p["score_a"]
        S[b, a] += p["games"] - p["score_a"]
        G[a, b] += p["games"]
        G[b, a] += p["games"]
    return pairs, names, idx, S, G


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="一场或多场 arena JSON（多场按参赛者名合并）")
    ap.add_argument("--ref", help="基准参赛者名（默认取第一个）")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    pairs, names, idx, S, G = load(args.path)
    if args.ref and args.ref not in idx:
        raise SystemExit(f"没有这个参赛者：{args.ref}\n可选：{' '.join(names)}")
    anchor = idx[args.ref] if args.ref else 0

    elo = fit_elo(S, G, anchor=anchor)

    # 对级自举：每对独立重采样它那 games/2 对的胜负，再整体重拟合
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(args.bootstrap):
        Sb = np.zeros_like(S)
        for p in pairs:
            a, b = idx[p["a"]], idx[p["b"]]
            npair = p["games"] // 2
            mean = p["score_a"] / p["games"]
            # 一对的得分 ∈ {0, 0.5, 1} × 2 局；用对级二项近似重采样
            sa = rng.binomial(npair, mean) * 2.0
            Sb[a, b] += sa
            Sb[b, a] += p["games"] - sa
        draws.append(fit_elo(Sb, G, anchor=anchor))
    se = np.std(np.array(draws), axis=0)

    print(" + ".join(os.path.basename(p) for p in args.path) + f"   基准 = {names[anchor]}")
    print(f"{'参赛者':<18}{'Elo':>9}{'±':>8}{'局数':>8}   对基准的直接得分率")
    for i in np.argsort(-elo):
        if i == anchor:
            direct = "—"
        elif G[i, anchor] > 0:
            direct = f"{S[i, anchor] / G[i, anchor]:.3f}  ({int(G[i, anchor])} 局)"
        else:
            direct = "（未直接交手）"
        print(f"{names[i]:<18}{elo[i]:>+9.1f}{se[i]:>8.1f}{int(G[i].sum()):>8}   {direct}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
