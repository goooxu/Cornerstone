#!/usr/bin/env python3
"""两组参赛者之间的**跨组**总账：得分率、95% 区间、换算 Elo、P(B 更强)。

    python3 tools/cross_arm.py ../runs/cross_ab.json \
        --a net@120227 net@100227 net@110227 net@130227 \
        --b net@110234 net@100234 net@80234  net@130234

用途是 A/B 的正题 —— 两条训练腿各取几档放进同一场单循环之后，
**只有跨组的那些对**回答「哪条腿更强」；组内的对只是把各自的内部次序也
放进同一次拟合，让跨组差值不被单档噪声带偏。

为什么不直接读 `net_arena` 表里的 Elo：那是 8 个参赛者的联合拟合，
一条腿多带一个弱档就会把它自己的均值拖下去。跨组得分率不受这个影响 ——
它是 4×4 个格子的直接对局，每个格子权重相同。

误差按**对（成对开局）**重采样，不按局：一对里两局用同一个开局、只是执先方
相反，纯策略下强相关，按 400 局独立二项试验算会高估精度。
"""

import argparse
import json
import math
import random


def elo(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / p - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--a", nargs="+", required=True, help="第一组的参赛者名")
    ap.add_argument("--b", nargs="+", required=True, help="第二组的参赛者名")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    blob = json.load(open(args.json))
    A, B = set(args.a), set(args.b)
    if A & B:
        raise SystemExit(f"两组不能有交集：{sorted(A & B)}")
    known = {p["name"] for p in blob["participants"]}
    missing = (A | B) - known
    if missing:
        raise SystemExit(f"这些参赛者不在 {args.json} 里：{sorted(missing)}")

    # 统一成「A 侧得分」，方向由名字决定而不是由它在 pairs 里的位置决定
    cross = []
    for r in blob["pairs"]:
        if r["a"] in A and r["b"] in B:
            cross.append((r["a"], r["b"], r["score_a"], r["games"]))
        elif r["a"] in B and r["b"] in A:
            cross.append((r["b"], r["a"], r["games"] - r["score_a"], r["games"]))
    want = len(A) * len(B)
    if len(cross) != want:
        raise SystemExit(f"跨组只找到 {len(cross)} 对，应有 {want} 对 —— "
                         f"这场单循环没把两组全对全打完")

    tot_s = sum(c[2] for c in cross)
    tot_g = sum(c[3] for c in cross)
    rate = tot_s / tot_g
    a_wins = sum(1 for c in cross if c[2] / c[3] > 0.5)

    print(f"跨组 {len(cross)} 对 / {int(tot_g)} 局")
    print(f"A 侧得分率 {rate:.4f}   ->  A {elo(rate):+.1f} Elo")
    print(f"A 侧赢下的对：{a_wins}/{len(cross)}")

    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.boot):
        s = g = 0.0
        for _a, _b, sa, gg in cross:
            npair = max(1, int(round(gg / 2)))
            p = sa / gg
            k = sum(1 for _ in range(npair) if rng.random() < p)
            s += k / npair * gg
            g += gg
        boot.append(s / g)
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print(f"95% 区间   得分率 [{lo:.4f}, {hi:.4f}]   Elo [{elo(lo):+.1f}, {elo(hi):+.1f}]")
    print(f"P(B 更强) = {sum(1 for x in boot if x < 0.5) / len(boot) * 100:.1f}%")

    print("\n交叉表（A 侧得分率）")
    d = {(a, b): sa / gg for a, b, sa, gg in cross}
    cols, rows = list(args.b), list(args.a)
    short = lambda s: s.replace("net@", "")                       # noqa: E731
    print("| A \\ B | " + " | ".join(short(c) for c in cols) + " | 行均 |")
    print("|---" * (len(cols) + 2) + "|")
    for r in rows:
        vs = [d[(r, c)] for c in cols]
        print(f"| {short(r)} | " + " | ".join(f"{v:.3f}" for v in vs)
              + f" | {sum(vs) / len(vs):.3f} |")
    print("| 列均 | " + " | ".join(
        f"{sum(d[(r, c)] for r in rows) / len(rows):.3f}" for c in cols) + " | |")


if __name__ == "__main__":
    main()
