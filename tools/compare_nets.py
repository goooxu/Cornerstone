#!/usr/bin/env python3
"""两个 checkpoint 直接对下，量它们之间的 Elo 差。

网络强过全部规则基线之后，规则阶梯就量不出强度了 —— 得分率钉在 1.0，
换算出来的 Elo 只是钳位产生的假数。这时只能让网络和自己的历史版本打。

    # 最新 vs 某个历史点
    python3 tools/compare_nets.py runs/bf16/ckpt/step00084000.pt runs/bf16/ckpt/step00040000.pt

    # 一条曲线：拿最早的 checkpoint 当锚点，逐个往后比
    python3 tools/compare_nets.py --curve runs/bf16/ckpt

    # 跨实验：FP8 vs BF16（对照实验的最终答案）
    python3 tools/compare_nets.py runs/fp8/ckpt/step00048000.pt runs/bf16/ckpt/step00048000.pt
"""

import argparse
import glob
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.evaluate import evaluate_vs_network   # noqa: E402
from cornerstone.model import load_checkpoint as load   # noqa: E402


def step_of(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="两个 checkpoint；--curve 模式下给一个目录")
    ap.add_argument("--curve", default=None, help="checkpoint 目录，逐个与最早的比")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--engine-threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.curve:
        cks = sorted(glob.glob(os.path.join(args.curve, "step*.pt")), key=step_of)
        if len(cks) < 2:
            raise SystemExit(f"{args.curve} 里少于两个 checkpoint")
        base, base_step = load(cks[0], args.device)
        print(f"锚点：{os.path.basename(cks[0])}（step {base_step}）")
        print(f"{'step':>10}{'得分率':>10}{'相对锚点 Elo':>14}")
        for ck in cks[1:]:
            m, st = load(ck, args.device)
            r = evaluate_vs_network(m, base, args.device, games=args.games,
                                    simulations=args.simulations,
                                    parallel_games=args.parallel,
                                    engine_threads=args.engine_threads,
                                    label=os.path.basename(cks[0]))
            print(f"{st:>10}{r.score_rate:>10.3f}{r.elo_diff:>+14.0f}")
            del m
            torch.cuda.empty_cache()
        return

    if len(args.paths) != 2:
        raise SystemExit("需要两个 checkpoint 路径，或用 --curve <目录>")
    a, sa = load(args.paths[0], args.device)
    b, sb = load(args.paths[1], args.device)
    r = evaluate_vs_network(a, b, args.device, games=args.games,
                            simulations=args.simulations, parallel_games=args.parallel,
                            engine_threads=args.engine_threads,
                            label=os.path.basename(args.paths[1]))
    print(f"A = {os.path.basename(args.paths[0])} (step {sa})")
    print(f"B = {os.path.basename(args.paths[1])} (step {sb})")
    print(f"{r.games} 局：A {r.wins}胜 / {r.draws}和 / {r.losses}负")
    print(f"A 得分率 {r.score_rate:.3f}，Elo 差 {r.elo_diff:+.0f}")
    print(f"A 先手赢 {r.wins_as_first} 局，后手赢 {r.wins_as_second} 局")
    print(f"平均手数 {r.mean_plies:.1f}，平均占格 A {r.mean_squares_net:.1f} / B {r.mean_squares_opp:.1f}")


if __name__ == "__main__":
    main()
