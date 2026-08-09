"""多 GPU 自博弈。

每张卡一份模型副本 + 一个独立的 SelfPlayDriver，用 Python 线程并发跑。
线程能真正并行是因为两段热点都放开了 GIL：

- C++ 引擎的 prepare/feed/advance 全程 `gil_scoped_release`
- torch 的 CUDA 算子在下发内核时也不持有 GIL

训练权重每轮同步一次到各副本。自博弈用的是「上一轮的权重」，这在 AlphaZero
类算法里本来就是常态（异步自博弈更是滞后好几轮），不影响正确性。
"""

from __future__ import annotations

import threading
import time

import torch

from . import _engine as E
from .model import CornerNet
from .selfplay import SelfPlayDriver, SelfPlayStats


class MultiGpuSelfPlay:
    def __init__(
        self,
        model: CornerNet,
        devices: list[str] | list[torch.device],
        num_games: int,
        mcts: E.MctsConfig,
        seed: int = 0,
        compile_model: bool = True,
        engine_threads: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.source = model
        self.devices = [torch.device(d) for d in devices]
        self.replicas: list[CornerNet] = []
        self.drivers: list[SelfPlayDriver] = []

        per_gpu = max(1, num_games // len(self.devices))
        for i, dev in enumerate(self.devices):
            if dev == next(model.parameters()).device:
                rep = model
            else:
                # 必须在**目标设备的上下文里**构造。FP8 模式下 te.Linear 的
                # MXFP8 权重会在当时的当前设备上分配，先在 cuda:0 建好再 .to(dev)
                # 搬不过去（量化张量的数据与缩放是分开的几块），
                # 运行时表现为 cuDNN kernel 里的非法访存，且报错位置离根因很远。
                with torch.cuda.device(dev):
                    rep = CornerNet(model.cfg).to(dev)
            rep.eval()
            self.replicas.append(rep)
            self.drivers.append(SelfPlayDriver(
                rep, dev, num_games=per_gpu, mcts=mcts, seed=seed + 1000 * i,
                compile_model=compile_model, engine_threads=engine_threads, dtype=dtype))

        # 副本是空初始化的，先同步一次权重再预热
        self.sync_weights()

        # 串行预热，把各卡的编译先做完。torch.compile 的编译期有全局状态，
        # 放到工作线程里并发编译会直接报错（表现为 Dynamo 内部的 weakref 异常）。
        for d in self.drivers:
            d.warmup()

    @torch.no_grad()
    def sync_weights(self) -> None:
        """把训练权重推给各卡的副本。

        FP8 参数不能直接跨卡 copy_ / load_state_dict，但原因**不是**「量化张量搬不过去」——
        实测 `.to(dev)` 会把 MXFP8Tensor 的四块数据（行/列两套 E4M3 数据 + 各自的
        E8M0 块缩放）全部正确搬走。真正的原因是 TE 按**当前 CUDA 设备**取 cuBLAS 句柄
        （见 docs/06 第五条）：跨卡 copy_ 天然要同时碰两张卡上的张量，无论把当前设备
        设成哪一边，另一边都不匹配，于是非法访存。
        走「反量化 -> 传普通张量 -> 重量化」就没这个问题：每个 TE 操作都待在自己那张卡上，
        中间搬运的是不带 TE 语义的普通张量。慢一点但正确。
        这里的重量化用确定性舍入 —— 它只是把主权重复制过去，不是优化器更新，
        不需要随机舍入的无偏性。
        """
        try:
            from .fp8 import is_quantized, write_weight_
        except ImportError:            # 没装 transformer_engine 时不可能有量化参数
            def is_quantized(_t):
                return False
            write_weight_ = None

        src_params = dict(self.source.named_parameters())
        src_buffers = dict(self.source.named_buffers())
        for rep, dev in zip(self.replicas, self.devices):
            if rep is self.source:
                continue
            for name, p in rep.named_parameters():
                s = src_params[name]
                if is_quantized(s) or is_quantized(p):
                    write_weight_(p, s.dequantize().to(dev).float(), stochastic=False)
                else:
                    p.data.copy_(s.data.to(dev, non_blocking=True))
            for name, b in rep.named_buffers():
                if name in src_buffers:
                    b.data.copy_(src_buffers[name].data.to(dev, non_blocking=True))

    def run(self, target_games: int, max_seconds: float | None = None,
            should_stop=None):
        """各卡并发跑，合并结果。target_games 会平均摊到各卡上。"""
        per = max(1, target_games // len(self.drivers))
        results: list[list[dict]] = [[] for _ in self.drivers]
        stats: list[SelfPlayStats] = [SelfPlayStats() for _ in self.drivers]
        errors: list[BaseException | None] = [None] * len(self.drivers)

        def work(i: int) -> None:
            try:
                results[i], stats[i] = self.drivers[i].run(per, max_seconds=max_seconds,
                                                          should_stop=should_stop)
            except BaseException as e:                      # noqa: BLE001
                errors[i] = e

        t0 = time.perf_counter()
        threads = [threading.Thread(target=work, args=(i,), daemon=True)
                   for i in range(len(self.drivers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for e in errors:
            if e is not None:
                raise e

        merged = [g for r in results for g in r]
        total = SelfPlayStats()
        total.games = sum(s.games for s in stats)
        total.evals = sum(s.evals for s in stats)
        total.nn_calls = sum(s.nn_calls for s in stats)
        total.gpu_seconds = sum(s.gpu_seconds for s in stats)
        total.seconds = time.perf_counter() - t0            # 墙钟，不是各线程之和
        return merged, total


def visible_devices(spec: str = "") -> list[str]:
    """把 "cuda:1,cuda:2" 之类的字符串解析成设备列表；空串表示用全部可见 GPU。"""
    if spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]
