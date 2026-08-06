#!/usr/bin/env python3
"""引擎吞吐基准。

随机对局全程在 C++ 里跑（GIL 已释放），所以多线程数字是真实的多核吞吐，
可以直接用来估自博弈的 CPU 侧上限。
"""

import argparse
import time

import cornerstone as cs


def bench(n_games: int, threads: int, seed: int = 1) -> dict:
    t0 = time.perf_counter()
    st = cs.random_playouts(n_games, seed, threads)
    dt = time.perf_counter() - t0
    return {
        "threads": threads,
        "seconds": dt,
        "games_per_s": st["games"] / dt,
        "movegens_per_s": st["movegens"] / dt,
        "plies_per_game": st["plies"] / st["games"],
        "movegens_per_game": st["movegens"] / st["games"],
        "win_rate_p0": st["wins0"] / st["games"],
        "draw_rate": st["draws"] / st["games"],
        "mean_score": (st["score0"] / st["games"], st["score1"] / st["games"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1, help="最大线程数（会按 1,2,4,... 扫上去）")
    ap.add_argument("--games", type=int, default=20000, help="单线程档位的对局数")
    args = ap.parse_args()

    levels = []
    t = 1
    while t < args.threads:
        levels.append(t)
        t *= 2
    levels.append(args.threads)

    print(f"{'线程':>6} {'对局/s':>12} {'着法生成/s':>14} {'耗时(s)':>9}")
    base = None
    for t in levels:
        # 线程多时按比例加大对局数，保证每档至少跑够时间
        r = bench(args.games * max(1, t // 2), t)
        if base is None:
            base = r["games_per_s"]
        print(f"{t:>6} {r['games_per_s']:>12,.0f} {r['movegens_per_s']:>14,.0f} {r['seconds']:>9.2f}")
        last = r

    print()
    print(f"平均每局 {last['plies_per_game']:.1f} 手，{last['movegens_per_game']:.1f} 次着法生成")
    print(f"随机对局先手胜率 {last['win_rate_p0']:.3f}，和局率 {last['draw_rate']:.3f}")
    print(f"平均占格数 先手 {last['mean_score'][0]:.1f} / 后手 {last['mean_score'][1]:.1f}")
    if base and len(levels) > 1:
        print(f"{levels[-1]} 线程相对单线程加速比 {last['games_per_s'] / base:.1f}x")


if __name__ == "__main__":
    main()
