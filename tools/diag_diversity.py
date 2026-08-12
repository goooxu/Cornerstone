#!/usr/bin/env python3
"""D2：量自博弈的**有效多样性**。

    python3 tools/diag_diversity.py ../runs/v2-bf16/ckpt/step000{10227,80227}.pt \
        ../runs/v2-bf16/ckpt/step00150227.pt --games 512

为什么需要：replay buffer 名义上压着 300 万个局面（约 49 轮历史），但如果自博弈
的棋局互相高度雷同，**有效样本量远小于名义值** —— 网络就是在一个退化数据集上
反复刷，于是 loss 一路降、棋力在退。这个脚本直接量那个"有效"。

**不能用 replay 快照来做这件事**：快照只留最新一份（约覆盖训练末段），
拿不到早期基线，而这里要看的正是"多样性随训练怎么变"。所以现取现跑。

口径与训练时的自博弈完全一致（64 模拟、temperature_plies=12、无随机开局），
不是评测口径 —— 评测那条路径有 opening_plies=2 的强制随机开局，会把结论洗掉。
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone import _engine as E                        # noqa: E402
from cornerstone.model import load_checkpoint               # noqa: E402
from cornerstone.selfplay import SelfPlayDriver             # noqa: E402


def first_divergence(a: np.ndarray, b: np.ndarray) -> int:
    """两局棋在第几手第一次分岔（都走完还一样就返回较短的长度）。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if ne.size else n


def stats(recs: list[dict], rng: np.random.Generator, n_pairs: int = 20000) -> dict:
    seqs = [np.asarray(r["actions"], dtype=np.int32) for r in recs]
    n = len(seqs)
    plies = np.array([len(s) for s in seqs])

    def uniq_prefix(k: int) -> int:
        return len({tuple(s[:k].tolist()) for s in seqs if len(s) >= k})

    # 随机抽对算首分歧手数，n 大时全对全太贵（512 局就是 13 万对，还能算；
    # 但接口要能用在更大的 n 上，所以一律抽样）
    m = min(n_pairs, n * (n - 1) // 2)
    ii = rng.integers(0, n, size=m)
    jj = rng.integers(0, n, size=m)
    keep = ii != jj
    div = np.array([first_divergence(seqs[i], seqs[j])
                    for i, j in zip(ii[keep], jj[keep])])

    res = np.array([r["result0"] for r in recs])
    return {
        "games": n,
        "mean_plies": float(plies.mean()),
        "uniq_first_move": uniq_prefix(1),
        "uniq_first_2": uniq_prefix(2),
        "uniq_first_4": uniq_prefix(4),
        "uniq_first_8": uniq_prefix(8),
        "uniq_games": len({tuple(s.tolist()) for s in seqs}),
        # 首分歧手数：越小说明棋局越早分开（多样性好），越大说明前 N 手都一样
        "diverge_p10": float(np.percentile(div, 10)),
        "diverge_median": float(np.median(div)),
        "diverge_p90": float(np.percentile(div, 90)),
        "identical_pair_rate": float((div >= plies.mean()).mean()),
        "p0_win_rate": float((res > 0).mean()),
        "draw_rate": float((res == 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=1024)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--temperature-plies", type=int, default=12)
    ap.add_argument("--random-opening-prob", type=float, default=0.0)
    ap.add_argument("--random-opening-max-plies", type=int, default=0)
    ap.add_argument("--engine-threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mcts = E.MctsConfig(simulations=args.simulations, max_considered=16,
                        temperature_plies=args.temperature_plies,
                        random_opening_prob=args.random_opening_prob,
                        random_opening_max_plies=args.random_opening_max_plies)

    rows = []
    for path in args.ckpts:
        model, step = load_checkpoint(path, args.device)
        drv = SelfPlayDriver(model, args.device, num_games=args.parallel, mcts=mcts,
                             seed=args.seed, compile_model=True,
                             engine_threads=args.engine_threads)
        drv.warmup()
        recs, sp = drv.run(args.games)
        s = stats(recs, rng)
        s["step"] = step
        s["ckpt"] = os.path.basename(path)
        s["evals_per_s"] = round(sp.evals_per_s)
        rows.append(s)
        print(f"{s['ckpt']}  跑了 {s['games']} 局，{sp.seconds:.1f}s", flush=True)

    cols = [("step", "步数", 8), ("uniq_first_move", "唯一首手", 10),
            ("uniq_first_2", "唯一前2手", 11), ("uniq_first_4", "唯一前4手", 11),
            ("uniq_first_8", "唯一前8手", 11), ("uniq_games", "唯一整局", 10),
            ("diverge_median", "首分歧中位", 12), ("diverge_p90", "首分歧P90", 11),
            ("p0_win_rate", "先手胜率", 10), ("draw_rate", "和局率", 9),
            ("mean_plies", "平均手数", 10)]
    print()
    print("".join(f"{h:>{w}}" for _, h, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for r in rows:
        line = ""
        for k, _, w in cols:
            v = r[k]
            line += f"{v:>{w}.3f}" if isinstance(v, float) else f"{v:>{w}}"
        print(line)

    print()
    print(f"参照：一局 {rows[-1]['mean_plies']:.0f} 手、共 {args.games} 局；"
          f"首手的合法着法有 414 种。「唯一前4手」若从几百掉到十几，"
          f"就是「buffer 名义 300 万局面、有效多样性小得多」的直接证据。")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "rows": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
