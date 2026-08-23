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

import contextlib
import time
from dataclasses import dataclass

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


def compile_for_inference(model: CornerNet):
    """编译推理前向。**低精度只能按 block 编译，不能整模型编译。**

    整模型编译会把 `CornerNet.trunk` 里的 `_fp8_scope()`（TE 的量化 autocast
    上下文）一起包进图里，而 TE 的量化状态是**全局**的。多卡多线程下各驱动
    并发跑起编译后的图，就会 **SIGSEGV** —— 崩在第一轮自博弈里，两次自检都还是
    「FP8 计算已启用」。三个因素缺一都不炸（单卡不炸、不编译不炸、BF16 不炸），
    所以孤立的单卡基准完全测不到。**FP4 一样按 block 编** —— 单卡 fullgraph 能过
    不代表多卡安全，全局状态那个风险是原样继承的，没有理由在这里赌一把。

    按 block 编译把那个上下文留在 eager，实测速度几乎没差
    （batch=1024 前向：按 block 15.97 ms，整模型 16.05 ms，eager 37.49 ms），
    而双卡驱动能稳定跑。

    BF16 没有这个问题，整模型编译更快（13.46 vs 15.50 ms），所以走整模型。

    **编法不做成可配置的** —— 每组各走各最快的那种。代价是受控 A/B 里
    两组的算子融合不同，严格说多了一个变量；但它比 FP8 量化本身的影响小
    （compile 换 eager 改 4.3% 的 argmax，FP8 量化改 9.2%），
    换来的是自博弈 2.4 倍。这是有意的取舍。

    幂等：重复调用不会叠加编译（driver 0 与训练循环共用同一个模型对象）。
    """
    if not getattr(model.cfg, "quantized", False):
        return torch.compile(model, dynamic=False)
    if not getattr(model, "_blocks_compiled", False):
        for blk in model.blocks:
            blk.compile(dynamic=False)      # 原地，不动 state_dict 的键
        model._blocks_compiled = True
    return model


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
        compile_model: bool = False,
        fixed_batch: bool = True,
        engine_threads: int = 1,
    ):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.mcts = mcts or E.MctsConfig()
        # **可达度平面由模型的 in_planes 驱动，不是独立旋钮。**
        # 算它要两次全量走法生成，而自博弈每个待评估叶子都调一次 features()，
        # 实测慢 27%（101.5 -> 74.3 局/s）—— 所以只在模型真的要吃它时才算。
        #
        # 写在这里而不是让调用方传，是为了让「模型要 11 个平面而引擎只产 9 个」
        # 这类错配**不可能发生**：那种错不报错，只是让平面 9/10 恒为 0，
        # 表现成「这个模型莫名其妙变弱了」。
        self.mcts.with_mobility = getattr(model.cfg, "in_planes", 9) > 9
        # 各局的树完全独立，engine_threads>1 时把树搜索摊到多核上。
        # 单线程时 144 核里只用得上一个 —— 而着法生成与树操作正是 CPU 侧的主要开销。
        self.engine = E.SelfPlayEngine(num_games, self.mcts, seed,
                                       eval_cfg or E.EvalConfig(),
                                       max(1, min(engine_threads, num_games)))

        # 每次都按固定批跑。prepare() 返回的 n 是变的，若照 n 切片喂进去，
        # torch.compile 会为每个出现过的 n 重新编译一次（单次编译约 60s）。
        # 实测批均已占并行局数的九成以上，补齐这点浪费远小于重编译的代价；
        # 顺带也满足了 MXFP8 对 batch 是 8 的倍数的要求。
        self.fixed_batch = fixed_batch
        # 用 zeros 而不是 empty：补齐部分的输出虽然被丢弃，但 empty 可能是 NaN，
        # 一旦 NaN 通过 RMSNorm 之类的规约算子传染到整批就查不出来了
        self.planes = np.zeros((num_games, PLANES, BOARD, BOARD), dtype=np.float32)
        self.scalars = np.zeros((num_games, SCALARS), dtype=np.float32)
        self.fwd = compile_for_inference(model) if compile_model else model
        pin = self.device.type == "cuda"
        self.planes_t = torch.from_numpy(self.planes)
        self.scalars_t = torch.from_numpy(self.scalars)
        self.logit_host = torch.empty((num_games, ACTIONS), dtype=torch.float32, pin_memory=pin)
        self.wdl_host = torch.empty((num_games, 3), dtype=torch.float32, pin_memory=pin)
        self.logit_np = self.logit_host.numpy()
        self.wdl_np = self.wdl_host.numpy()

    def _device_ctx(self):
        """把当前 CUDA 设备设成本驱动所在的卡。

        FP8 必须这么做：TransformerEngine 的 cuBLAS 句柄按**当前设备**取，
        张量在 cuda:1 而当前设备是 cuda:0 时会报
        `cublas_gemm: the function failed to launch on the GPU`，
        而且报错点常常飘到注意力的 cuDNN kernel 上，离根因很远。
        `torch.cuda.device` 是线程局部的，所以多卡多线程各设各的互不影响。
        """
        return (torch.cuda.device(self.device) if self.device.type == "cuda"
                else contextlib.nullcontext())

    @torch.no_grad()
    def _evaluate(self, n: int) -> None:
        # 推理用的就是训练那份计算权重（bf16），fp32 主权重只服务优化器 ——
        # 否则推理和训练看到的是差一次舍入的两个模型。autocast 在这里管的是
        # 激活：把 fp32 的输入和 RMSNorm 的输出接到 bf16 的算子上
        m = self.planes.shape[0] if self.fixed_batch else n
        with self._device_ctx():
            p = self.planes_t[:m].to(self.device, non_blocking=True)
            s = self.scalars_t[:m].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.dtype,
                                enabled=self.device.type == "cuda"):
                pol, wdl, _ = self.fwd(p, s)
        self.logit_host[:n].copy_(pol[:n].float(), non_blocking=True)
        self.wdl_host[:n].copy_(wdl[:n].float().softmax(dim=-1), non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def warmup(self) -> None:
        """跑一次前向把 torch.compile 的编译触发掉。

        必须在起线程**之前**串行做完：Dynamo 编译期有全局状态，
        多个线程同时编译同一个模型类会直接报错。
        """
        self._evaluate(1)

    def sync_weights(self) -> None:
        """单卡时驱动直接持有训练用的那个模型对象，无需同步。
        接口和 WorkerPool 保持一致，调用方不用分支。"""

    def run(self, target_games: int, max_seconds: float | None = None,
            on_games=None, should_stop=None) -> tuple[list[dict], SelfPlayStats]:
        """跑到收集够 target_games 局（或超时、或 should_stop() 为真）。

        should_stop 用于响应 SIGTERM：一轮自博弈可能要几分钟，
        不能等它跑完才收尾，否则会被强杀，丢掉未落盘的进度。
        """
        was_training = self.model.training
        self.model.eval()
        out: list[dict] = []
        st = SelfPlayStats()
        t0 = time.perf_counter()
        try:
            while len(out) < target_games:
                if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                    break
                if should_stop is not None and should_stop():
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
