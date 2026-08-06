#!/usr/bin/env python3
"""自博弈吞吐基准。

主指标是**每秒网络评估次数**，不是每秒对局数：一局要 27 手 × 模拟数 次评估，
短时间窗口里几乎没有对局能走完，"局/s" 会被未完成的对局严重低估。

C++ 侧的着法生成早就证明不是瓶颈（128 线程 1030 万次/s），所以这里量的是
GPU 侧：批攒得够不够大、每次前向的固定开销摊薄了没有。
"""

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone import _engine as E                   # noqa: E402
from cornerstone.model import CornerNet, ModelConfig   # noqa: E402
from cornerstone.selfplay import SelfPlayDriver        # noqa: E402

PLIES_PER_GAME = 27.1     # 随机对局实测均值，用来把评估数换算成等效对局数


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--parallel", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--engine-threads", type=int, default=1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    model = CornerNet(ModelConfig(dim=args.dim, blocks=args.blocks, fp8=args.fp8)).cuda().eval()
    print(f"模型 {model.num_params()/1e6:.1f}M 参数 | {args.simulations} 次模拟/手 | "
          f"compile={args.compile} fp8={args.fp8} 引擎线程={args.engine_threads}")
    print(f"{'并行局数':>9}{'评估/s':>12}{'等效局/s':>11}{'批均':>8}{'GPU占比':>9}{'前向ms':>9}")
    print("-" * 58)

    for p in args.parallel:
        driver = SelfPlayDriver(model, "cuda", num_games=p,
                                mcts=E.MctsConfig(simulations=args.simulations), seed=1,
                                compile_model=args.compile,
                                engine_threads=args.engine_threads)
        driver.run(10**9, max_seconds=max(8.0, args.seconds * 0.5))     # 预热（含编译）
        _, st = driver.run(10**9, max_seconds=args.seconds)
        if st.nn_calls == 0:
            print(f"{p:>9}  时间窗口里一次评估都没做，加大 --seconds")
            continue
        eq_games = st.evals_per_s / (PLIES_PER_GAME * args.simulations)
        fwd_ms = st.gpu_seconds / st.nn_calls * 1000
        print(f"{p:>9}{st.evals_per_s:>12,.0f}{eq_games:>11.1f}{st.mean_batch:>8.0f}"
              f"{st.gpu_seconds/st.seconds:>9.2f}{fwd_ms:>9.2f}")


if __name__ == "__main__":
    main()
