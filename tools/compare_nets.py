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
    # 输出里直接印名字，而不是让人去记「第一个位置参数叫 A」。
    # 不加这个就出过事：调用方按 (B, A) 的顺序传参、又在自己那边
    # 打印「A=BF16」，和本脚本印的 A 正好相反 —— 两行都在同一屏输出里，
    # 结论看着是「BF16 领先」，实际是反的。见 --name-a / --name-b 的用处。
    ap.add_argument("--name-a", default=None, help="第一个 checkpoint 的显示名")
    ap.add_argument("--name-b", default=None, help="第二个 checkpoint 的显示名")
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
    na = args.name_a or os.path.basename(args.paths[0])
    nb = args.name_b or os.path.basename(args.paths[1])
    print(f"{na} = {os.path.basename(args.paths[0])} (step {sa})")
    print(f"{nb} = {os.path.basename(args.paths[1])} (step {sb})")
    print(f"{r.games} 局：{na} {r.wins}胜 / {r.draws}和 / {r.losses}负")
    print(f"{na} 得分率 {r.score_rate:.3f}，相对 {nb} 的 Elo 差 {r.elo_diff:+.0f}")
    print(f"{na} 先手赢 {r.wins_as_first} 局，后手赢 {r.wins_as_second} 局")
    print(f"平均手数 {r.mean_plies:.1f}，平均占格 "
          f"{na} {r.mean_squares_net:.1f} / {nb} {r.mean_squares_opp:.1f}")
    winner = na if r.score_rate > 0.5 else (nb if r.score_rate < 0.5 else "平手")
    print(f"→ 领先方：{winner}")


if __name__ == "__main__":
    main()
