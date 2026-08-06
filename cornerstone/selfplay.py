"""自博弈驱动。

循环形状很简单：
    prepare()  收集所有待评估叶子的特征（C++，已释放 GIL）
    模型前向   一次算完整批（GPU）
    feed()     回填先验与价值，展开叶子并回传
    advance()  本手模拟做完，各局落子

批量大小来自「同时开多少局」，不是单局内的并行叶子数 ——
所以既不需要 virtual loss，也没有多线程回调 Python 的 GIL 争用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _engine as E
from .model import ACTIONS, BOARD, CornerNet, PLANES, SCALARS


@dataclass
class SelfPlayStats:
    games: int = 0
    evals: int = 0
    nn_calls: int = 0
    seconds: float = 0.0
    gpu_seconds: float = 0.0

    @property
    def games_per_s(self) -> float:
        return self.games / self.seconds if self.seconds else 0.0

    @property
    def evals_per_s(self) -> float:
        return self.evals / self.seconds if self.seconds else 0.0

    @property
    def mean_batch(self) -> float:
        return self.evals / self.nn_calls if self.nn_calls else 0.0


class SelfPlayDriver:
    def __init__(
        self,
        model: CornerNet,
        device: torch.device | str,
        num_games: int = 512,
        mcts: E.MctsConfig | None = None,
        seed: int = 0,
        eval_cfg: E.EvalConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.mcts = mcts or E.MctsConfig()
        self.engine = E.SelfPlayEngine(num_games, self.mcts, seed,
                                       eval_cfg or E.EvalConfig())

        # 主机侧缓冲复用，避免每轮申请几十 MB
        self.planes = np.empty((num_games, PLANES, BOARD, BOARD), dtype=np.float32)
        self.scalars = np.empty((num_games, SCALARS), dtype=np.float32)
        pin = self.device.type == "cuda"
        self.planes_t = torch.from_numpy(self.planes)
        self.scalars_t = torch.from_numpy(self.scalars)
        self.logit_host = torch.empty((num_games, ACTIONS), dtype=torch.float32, pin_memory=pin)
        self.wdl_host = torch.empty((num_games, 3), dtype=torch.float32, pin_memory=pin)
        self.logit_np = self.logit_host.numpy()
        self.wdl_np = self.wdl_host.numpy()

    @torch.no_grad()
    def _evaluate(self, n: int) -> None:
        # 走 autocast 而不是手工转 dtype：权重保持 fp32 主副本，
        # 与训练路径完全一致，否则推理和训练看到的是两个不同的模型
        p = self.planes_t[:n].to(self.device, non_blocking=True)
        s = self.scalars_t[:n].to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=self.dtype,
                            enabled=self.device.type == "cuda"):
            pol, wdl, _ = self.model(p, s)
        self.logit_host[:n].copy_(pol.float(), non_blocking=True)
        self.wdl_host[:n].copy_(wdl.float().softmax(dim=-1), non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, target_games: int, max_seconds: float | None = None,
            on_games=None) -> tuple[list[dict], SelfPlayStats]:
        """跑到收集够 target_games 局（或超时）。返回 (对局列表, 统计)。"""
        was_training = self.model.training
        self.model.eval()
        out: list[dict] = []
        st = SelfPlayStats()
        t0 = time.perf_counter()
        try:
            while len(out) < target_games:
                if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                    break
                n = self.engine.prepare(self.planes, self.scalars)
                if n == 0:
                    done = self.engine.advance()
                    if done:
                        out.extend(done)
                        st.games += len(done)
                        if on_games:
                            on_games(done)
                    continue
                g0 = time.perf_counter()
                self._evaluate(n)
                st.gpu_seconds += time.perf_counter() - g0
                self.engine.feed(self.logit_np[:n], self.wdl_np[:n])
                st.evals += n
                st.nn_calls += 1
        finally:
            if was_training:
                self.model.train()
        st.seconds = time.perf_counter() - t0
        return out, st
