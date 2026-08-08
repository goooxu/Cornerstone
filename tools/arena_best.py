#!/usr/bin/env python3
"""从 net_arena 的输出里回答「哪一档最强」，并给出这个结论有多确定。

    python3 tools/arena_best.py ../runs/arena/bf16_purepolicy.json

单看 Elo 表的第一行是不够的：一堆参赛者的误差棒互相重叠时，「第一名」很可能只是
噪声抽签的结果。这里用**自举出的最大值分布**回答 —— 每次重采样后重新拟合，
数一数每个参赛者当第一的次数，得到 `P(是最强)`。

重采样按**对**做，不按局：开局是成对的（同一开局双方各执先一次），
同一对里的两局强相关，按局重采样会高估精度。
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


def load(path):
    d = json.load(open(path))
    names = [p["name"] for p in d["participants"]] if "participants" in d else d["names"]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    S, G = np.zeros((n, n)), np.zeros((n, n))
    for p in d["pairs"]:
        i, j = idx[p["a"]], idx[p["b"]]
        S[i, j] += p["score_a"]; S[j, i] += p["games"] - p["score_a"]
        G[i, j] += p["games"];   G[j, i] += p["games"]
    return names, idx, S, G, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--anchor", default="random")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=8, help="列出前几名的头对头")
    ap.add_argument("--plot", default=None, help="把增长曲线画到这个路径")
    args = ap.parse_args()

    names, idx, S, G, _ = load(args.json)
    n = len(names)
    anchor = idx[args.anchor]
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

    err = boot.std(axis=0)
    # 只在网络之间评「最强」——规则基线不是候选
    nets = [i for i, nm in enumerate(names) if nm.startswith("net@")]
    win = np.zeros(n)
    champs = np.array(nets)[boot[:, nets].argmax(axis=1)]
    for c in champs:
        win[c] += 1
    win /= args.boot

    order = sorted(nets, key=lambda i: -elo[i])
    print(f"{'checkpoint':<16}{'Elo':>9}{'±':>7}{'P(是最强)':>12}")
    print("-" * 46)
    for i in order:
        star = "  ←最高" if i == order[0] else ""
        print(f"{names[i]:<16}{elo[i]:>9.1f}{err[i]:>7.1f}{100*win[i]:>10.1f}%{star}")

    print()
    tot = sum(win[i] for i in order[:args.top])
    print(f"前 {args.top} 名合计 P(是最强) = {100*tot:.1f}%"
          f" —— 单独任何一档的把握都不超过 {100*max(win[i] for i in order):.1f}%")

    print()
    print(f"前 {args.top} 名之间的头对头（行对列的得分率）：")
    top = order[:args.top]
    hdr = "".join(f"{names[j].replace('net@',''):>9}" for j in top)
    print(f"{'':<14}{hdr}")
    for i in top:
        cells = ""
        for j in top:
            cells += "        ·" if i == j else f"{S[i, j]/max(1, G[i, j]):>9.3f}"
        print(f"{names[i]:<14}{cells}")

    # 成对差值的误差远小于各自的 ±：两档网络共享同一批对手，绝对刻度上那份
    # 「整条曲线一起平移」的不确定度（主要来自通往 random 的那条弱边）会抵消掉。
    dsd = float(np.median([(boot[:, i] - boot[:, j]).std()
                           for i, j in itertools.combinations(nets, 2)]))
    print()
    print(f"两档之间比较的误差：±{dsd:.1f}（而不是表里的 ±{err[nets].mean():.0f}）—— "
          f"后者被「整条曲线相对 random 一起平移」的不确定度主导，比较档位时会抵消。")

    print()
    print("规则基线（同一次拟合，供定标）：")
    for i in sorted((i for i in range(n) if i not in nets), key=lambda i: -elo[i]):
        print(f"  {names[i]:<18}{elo[i]:>8.1f}  ±{err[i]:.1f}")

    if args.plot:
        plot(names, idx, nets, elo, dsd, win, args.plot)


def plot(names, idx, nets, elo, dsd, win, out: str) -> None:
    """棋力随训练步数的变化。标注一律用英文 —— 容器里没有中文字体。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = np.array([int(names[i].split("@")[1]) for i in nets], float)
    y = np.array([elo[i] for i in nets])
    o = np.argsort(step)
    step, y = step[o], y[o]
    ns = [nets[k] for k in o]
    peak = int(np.argmax(y))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.fill_between(step, y - dsd, y + dsd, color="#2f7fd1", alpha=0.18, lw=0)
    ax.plot(step, y, "-o", color="#1b3a5c", ms=4.5, lw=1.8, label="checkpoint (pure policy)")
    ax.plot(step[peak], y[peak], "o", ms=11, mfc="none", mec="#c0392b", mew=2.2)
    ax.annotate(f"peak  step {int(step[peak]):,}\n{y[peak]:.0f} Elo   P(best)={100*win[ns[peak]]:.0f}%",
                xy=(step[peak], y[peak]), xytext=(step[peak] - 4000, y[peak] - 210),
                fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    ax.annotate(f"final  step {int(step[-1]):,}\n{y[-1]:.0f} Elo  ({y[-1]-y[peak]:+.0f})",
                xy=(step[-1], y[-1]), xytext=(step[-1] - 30000, y[-1] - 250),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    for nm, col in (("greedy-mobility", "#2f9e6f"), ("flat-mcts-1k", "#e8892b")):
        if nm not in idx:
            continue
        val = elo[idx[nm]]
        ax.axhline(val, color=col, ls=":", lw=1.2)
        # 标签靠左：右下角是图例，压上去会看不见
        ax.text(step[0], val + 16, nm, ha="left", fontsize=8.5, color=col)
    ax.set_xlabel("training step")
    ax.set_ylabel("Elo  (random = 0)")
    # 「前 4 万步占多少」按实测算 —— 旧口径那个 93% 是带搜索那次的数，不能沿用
    i40 = int(np.argmin(np.abs(step - 40230)))
    frac = 100.0 * (y[i40] - y[0]) / (y[peak] - y[0])
    ax.set_title(f"Playing strength without search — {frac:.0f}% of the gain lands by step 40k, "
                 f"then it goes flat", fontsize=10.5, pad=8)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
