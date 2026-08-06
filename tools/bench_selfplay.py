#!/usr/bin/env python3
"""自博弈吞吐基准。

关心的是「并行局数 -> 批大小 -> GPU 利用率」这条链。C++ 侧的着法生成早就证明
不是瓶颈（128 线程 1030 万次/s），所以这里量的其实是：批攒得够不够大、
每次网络调用的固定开销摊薄了没有。
"""

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone import _engine as E          # noqa: E402
from cornerstone.model import CornerNet, ModelConfig   # noqa: E402
from cornerstone.selfplay import SelfPlayDriver        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--parallel", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--seconds", type=float, default=25.0, help="每档跑多久")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    model = CornerNet(ModelConfig(dim=args.dim, blocks=args.blocks)).cuda().eval()
    print(f"模型 {model.num_params()/1e6:.1f}M 参数，{args.simulations} 次模拟/手")
    print(f"{'并行局数':>9}{'局/s':>10}{'评估/s':>12}{'批均':>8}{'GPU占比':>9}{'每局评估':>10}")
    print("-" * 60)

    for p in args.parallel:
        driver = SelfPlayDriver(model, "cuda", num_games=p,
                                mcts=E.MctsConfig(simulations=args.simulations), seed=1)
        driver.run(10**9, max_seconds=args.seconds * 0.3)          # 预热
        _, st = driver.run(10**9, max_seconds=args.seconds)
        if st.games == 0:
            print(f"{p:>9}  该档在给定时间内一局都没走完，加大 --seconds")
            continue
        print(f"{p:>9}{st.games_per_s:>10.1f}{st.evals_per_s:>12,.0f}"
              f"{st.mean_batch:>8.0f}{st.gpu_seconds/st.seconds:>9.2f}"
              f"{st.evals/st.games:>10.0f}")


if __name__ == "__main__":
    main()
