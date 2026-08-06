#!/usr/bin/env python3
"""跑基线阶梯的循环赛，输出 Elo 表。

    python3 tools/run_arena.py --games 400 --threads 128
    python3 tools/run_arena.py --agents random greedy-area --games 1000
"""

import argparse
import json
import os
import sys
import time

from cornerstone.arena import BASELINES, DEFAULT_LADDER, round_robin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=DEFAULT_LADDER,
                    help=f"可选: {' '.join(BASELINES)}")
    ap.add_argument("--games", type=int, default=200, help="每对的对局数（必须是偶数）")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--opening-plies", type=int, default=4, help="开局随机手数")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor", default="random", help="Elo 零点锚定在谁身上")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    t0 = time.perf_counter()

    def progress(a, b):
        print(f"  {a} vs {b} ...", flush=True)

    print(f"循环赛：{len(args.agents)} 个智能体，每对 {args.games} 局，"
          f"{args.threads} 线程，开局随机 {args.opening_plies} 手")
    res = round_robin(
        args.agents,
        games=args.games,
        seed=args.seed,
        threads=args.threads,
        opening_plies=args.opening_plies,
        anchor=args.anchor if args.anchor in args.agents else args.agents[0],
        progress=progress,
    )
    dt = time.perf_counter() - t0

    print()
    print(res.pair_table())
    print()
    print(res.table())
    print()
    total = int(res.games.sum() // 2)
    print(f"共 {total} 局，耗时 {dt:.1f}s（{total / dt:.0f} 局/s）")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
