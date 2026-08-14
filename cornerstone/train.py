"""训练循环。

同步循环：自博弈一批 -> 塞进 replay -> 训若干步。
M5 会把自博弈和训练拆成独立进程各占各的 GPU；但先把「能学起来」这件事坐实，
异步化是性能问题，不是正确性问题。

开发机单次会话有时长上限，所以 checkpoint 按「每 N 步」和「每 T 秒」双触发落盘，
replay 热数据写本地盘、快照回写工作目录，`resume()` 能从任意一次落盘处接着跑。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import _engine as E
from .losses import total_loss
from .model import CornerNet, ModelConfig, load_weights
from .optim import MasterWeightAdamW
from .replay import ReplayBuffer
from .selfplay import SelfPlayDriver, compile_for_inference
from .pool import WorkerPool


@dataclass
class TrainConfig:
    exp: str = "bf16"
    run_dir: str = ""          # 空则用 <repo>/../runs/<exp>
    hot_dir: str = ""          # 空则用 /tmp/cornerstone/<exp>

    # 模型
    dim: int = 256
    blocks: int = 16
    attn_every: int = 4
    # 主干 GEMM 的计算精度：bf16 | fp8(MXFP8) | fp4(NVFP4)。
    # **直接换掉了老的 `fp8: bool`**（不像 ModelConfig 要留兼容别名）——
    # TrainConfig 不需要读老 checkpoint，让老命令行 `--fp8 true` 硬报错才是对的：
    # 静默把它当成一个未知参数忽略，等于开了一组本该是 fp8 的 bf16 训练。
    precision: str = "bf16"

    # 自博弈
    parallel_games: int = 8192    # 多卡时按卡均分（4 卡 -> 每卡 2048，实测该点最优）
    # torch.compile 不做成开关：一律开，每组各走各最快的编法
    # （BF16 整模型 13.46 ms、FP8 按 block 15.97 ms，eager 是 37 ms）。
    # CPU 上自动跳过。
    # C++ 侧树搜索**每张卡**用多少 CPU 线程。名字里带 per_gpu 是因为这里踩过坑：
    # 它原先叫 engine_threads、语义是"总数、多卡均分"，而 `docs/07` 里那个
    # "32 线程最优"是**单卡**基准测的（单卡下总数就等于每卡）。照抄成多卡的
    # 每卡值就是 128 总数 —— 实测最差的一档。
    #
    # 8 是 4 卡实测的最优点：`parallel_games`/`feed` 每次调用都现建现销
    # std::thread，线程一多 spawn 开销就压过并行收益。每卡 8 给 180,554 评估/s，
    # 每卡 32 只有 99,815（差 1.81×）。曲线在 6~8 之间是平的，两侧都掉。
    # 注意最优点绑在**每卡并行局数**上（这里 1024），`parallel_games` 一改就要重测。
    engine_threads_per_gpu: int = 8
    selfplay_devices: str = ""    # 逗号分隔，空则用全部可见 GPU
    simulations: int = 64
    max_considered: int = 16
    temperature_plies: int = 12
    games_per_iter: int = 2048

    # 自博弈的**状态分布**旋钮。注意 temperature_plies 不是 ——
    # 它只在「SH 胜者（带 Gumbel）」和「改进策略 argmax」之间二选一，
    # 先验一尖锐两者就选同一手（实测把它从 12 延到 30，先手胜率不降反升）。
    #
    # 治的是开局塌缩。实测（tools/diag_diversity.py，v2-bf16）：训练到第 8 万步时
    # 512 局自博弈的**首手全部相同**，而首手有 414 种合法着法；15 万步时只剩
    # 5 种前两手、15 种前四手，45% 的对局与别的对局逐手重复。
    # replay 名义压着 300 万个局面（约 49 轮），但这 49 轮走的是同一个漏斗。
    #
    # 开着（0.5 / 6）是默认，因为它**要么更好要么持平，且零成本**：
    #   * 唯一首手 1 → 283，唯一整局 55% → 99%
    #   * 自博弈先手胜率 0.887 → 0.739，而该档镜像自战的**客观**值是 0.725 ——
    #     多出来的那 0.16 全是漏斗，一注入就还原
    #   * 同步数带搜索头对头：第 2~6 万步领先 +92 ~ +149 Elo
    #   * 墙钟 4.52h vs 4.42h
    # **但它不抬天花板**：15 万步的终点与不开时打平（16 对 / 6,400 局带搜索，
    # 不开那侧 +11.3 Elo，95% 区间 [−0.7, +23.5] 跨零）。见 docs/08。
    random_opening_prob: float = 0.5
    random_opening_max_plies: int = 6

    # 以下三个此前只在 C++ 里有默认值、没暴露出来。不改默认值，只是让它们可调：
    #   c_visit / c_scale —— sigma(q) = (c_visit + max_N) * c_scale，本项目约 50~66。
    #     它是 Gumbel-AZ 里 q 相对先验的放大倍数，也是 policy churn 的放大器
    #     （实测相邻两档 checkpoint 之间 25~39% 的局面最优着法整个换了，而 loss 不动）
    #   value_from_score —— >0 时把终局占格差按此权重混进价值目标。
    #     占格差是连续量，胜负塌缩（先手胜率 0.965）之后它仍有分辨率
    c_visit: float = 50.0
    c_scale: float = 1.0
    value_from_score: float = 0.0


    # 训练
    batch_size: int = 1024
    steps_per_iter: int = 400
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    total_steps: int = 200_000

    # 学习率日程。**`total_steps` 只管「这一次跑到哪停」，不再决定曲线形状** ——
    # 形状由 `lr_horizon_steps` 定。少了这层解耦，把 total_steps 从 11.1 万改到
    # 25 万的那一刻，退火窗口整体右移，已经退到 2e-4 的学习率会跳回 2e-3，
    # 那不是「加训」而是「做了一次 warm restart 再加训」。
    #
    #   cosine  余弦退火到 lr*min_lr_ratio（历史默认，horizon=0 时逐位不变）
    #   wsd     warmup → stable 恒 lr → 末段线性退到 lr*min_lr_ratio
    #
    # WSD 的意义是「预算不必预先知道」：stable 段与总预算无关，想收工时才花
    # 最后约 10% 退火；而且 stable 段的 checkpoint 可以反复分叉出不同的终点。
    lr_schedule: str = "cosine"     # cosine | wsd
    lr_horizon_steps: int = 0       # 曲线的横轴终点；0 = 跟随 total_steps
    lr_decay_steps: int = 0         # wsd 末段退火步数；0 = horizon 的 10%
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    w_value: float = 1.0
    w_score: float = 0.25
    augment: bool = True
    loader_threads: int = 32

    # replay
    replay_capacity: int = 3_000_000
    min_positions: int = 20_000     # 攒够这么多局面才开始训练
    max_epochs_per_iter: float = 4.0  # 单轮最多把 replay 过几遍，防止小 buffer 上过拟合

    # checkpoint
    # 开发机每 8 小时过期一次（容器被回收，训练进程随之消失且来不及优雅收尾），
    # 所以丢失量由这两个阈值决定。取 400 = 一轮的步数，即每轮都落盘。
    ckpt_every_steps: int = 400
    ckpt_every_seconds: float = 240.0
    keep_last: int = 3
    milestone_every_steps: int = 10_000   # 里程碑 checkpoint 永久保留
    snapshot_every_iters: int = 20
    fp8_check_every_iters: int = 10   # 低精度会静默降级，只能靠反复测

    seed: int = 1
    device: str = "cuda"

    def resolve(self, repo_root: str) -> "TrainConfig":
        if not self.run_dir:
            self.run_dir = os.path.join(os.path.dirname(repo_root), "runs", self.exp)
        if not self.hot_dir:
            self.hot_dir = os.path.join("/tmp", "cornerstone", self.exp)
        # 这里曾经强制关掉 FP8 那组的 compile。**现在不需要了** ——
        # 真正的限制是「FP8 不能整模型编译」，而不是「FP8 不能编译」，
        # 按 block 编译既安全又几乎一样快（见 selfplay.compile_for_inference）。
        return self


class _Sampler:
    """给工作池的批次来源。`peek()` 让它先看一批以确定共享缓冲的形状与 dtype。"""

    def __init__(self, tr: "Trainer"):
        self.tr = tr
        self._first = None

    def _draw(self) -> dict:
        c = self.tr.cfg
        return self.tr.buffer.sample(c.batch_size, self.tr.rng,
                                     threads=c.loader_threads, augment=c.augment)

    def peek(self) -> dict:
        if self._first is None:
            self._first = self._draw()
        return self._first

    def __call__(self) -> dict:
        if self._first is not None:
            b, self._first = self._first, None
            return b
        return self._draw()


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)

        # 这三行的顺序是硬的：模型在 fp32 下初始化 -> 优化器取走 fp32 master ->
        # 参数降到计算权重精度。写反的话 master 是从 bf16 值回填的，
        # 等于一开始就丢一半精度，而训练看不出任何异常。
        self.model = CornerNet(ModelConfig(
            dim=cfg.dim, blocks=cfg.blocks, attn_every=cfg.attn_every,
            precision=cfg.precision,
        )).to(self.device)
        self.opt = self._make_optimizer()
        self.model.to_param_dtype()
        self._compile_hot_modules()
        self.pool = None
        self._stable_saved = False      # WSD：stable 终点的永久档是否已落
        self._snap_bucket = -1          # 已经存过快照的最大里程碑桶
        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.step = 0
        self.iteration = 0
        # 全生命周期产生过的对局数。不能用 buffer.total_games_seen 代替 ——
        # 那是进程内计数器，续训时会被快照重新播种，看起来像「从头训了」。
        self.games_played = 0
        self.last_ckpt_time = time.time()
        self.last_ckpt_step = 0
        self.history: list[dict] = []

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(cfg.hot_dir, exist_ok=True)

    # ---- 路径 ----
    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.cfg.run_dir, "ckpt")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.cfg.run_dir, "logs")

    @property
    def snapshot_dir(self) -> str:
        return os.path.join(self.cfg.run_dir, "replay_snapshot")

    def _compile_hot_modules(self) -> None:
        """只编译 SwiGLU —— 训练步唯一值得编译的地方。

        训练步此前完全没走过 `torch.compile`（它只包了自博弈的前向）。
        而按模块拆开看，**SwiGLU 占前向 74%，其中 78% 是 `silu(g)*v` 这一个逐元素算子**：
        `h.chunk(2, -1)` 产生的是非连续视图，PyTorch 于是退回非向量化的逐元素核，
        实测只跑到 1.28 TB/s，而连续张量能到 3.73。

        为什么是「只编译 SwiGLU」而不是别的两种做法（都实测过）：

        - **整模型 compile 只值 1.13×**。带 autograd 时前向必须为反向保存中间量，
          融合空间比推理路径小得多；只编译热点反而让 inductor 在前反向都能融合掉
          那个逐元素算子，实测 **1.38×**（FP8 1.34×）。
        - **把 SwiGLU 拆成 gate/val 两个 Linear** 也能绕开非连续视图（1.24×），
          但要改 checkpoint 键名，而且 FP8 下等于把同一个输入量化两次、
          跑两个更小的 GEMM —— 自博弈的编译前向反而慢 15%。

        必须用 `nn.Module.compile()` 原地编译，**不能写 `blk.mlp = torch.compile(blk.mlp)`**：
        后者返回 `OptimizedModule`，会把 state_dict 的键变成 `..._orig_mod...`，
        当场破坏 checkpoint 契约（实测多出 32 个键）。

        自博弈那条路径会再把整个模型 compile 一次，形成嵌套 —— 实测中性
        （BF16 0.997×、FP8 1.000×）。
        """
        if self.device.type != "cuda":
            return
        if self.model.cfg.quantized:
            # 低精度有更强的约束：只能按 block 编译，不能整模型编译，否则多卡自博弈
            # 会段错误。这里直接走那条路径 —— 自博弈驱动复用同一个模型对象，
            # 而它是幂等的，不会叠加编译。
            compile_for_inference(self.model)
        else:
            for blk in self.model.blocks:
                blk.mlp.compile(dynamic=False)

    def _make_optimizer(self) -> MasterWeightAdamW:
        """两组用同一个优化器 —— FP8 与否只影响 GEMM，不影响主权重与更新。"""
        # Norm / bias / 位置嵌入不做权重衰减
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith("pos") else decay).append((name, p))
        groups = [{"params": [p for _, p in decay], "names": [n for n, _ in decay],
                   "weight_decay": self.cfg.weight_decay},
                  {"params": [p for _, p in no_decay], "names": [n for n, _ in no_decay],
                   "weight_decay": 0.0}]
        return MasterWeightAdamW(groups, lr=self.cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    def lr_at(self, step: int) -> float:
        """纯函数，全仓只有这一份（工作进程只是转发父进程算好的标量）。

        `horizon` 而不是 `total_steps` 决定形状 —— 见 TrainConfig.lr_horizon_steps。
        """
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        horizon = c.lr_horizon_steps or c.total_steps
        if c.lr_schedule == "wsd":
            decay = c.lr_decay_steps or max(1, horizon // 10)
            start = max(c.warmup_steps, horizon - decay)
            if step < start:
                return c.lr                                    # stable
            t = min(1.0, (step - start) / max(1, horizon - start))
            return c.lr * (1 - (1 - c.min_lr_ratio) * t)       # 线性退到底
        t = min(1.0, (step - c.warmup_steps) / max(1, horizon - c.warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * cos)

    def decay_start_step(self) -> int:
        """WSD 的 stable 段在哪一步结束（非 wsd 时返回 0）。"""
        c = self.cfg
        if c.lr_schedule != "wsd":
            return 0
        horizon = c.lr_horizon_steps or c.total_steps
        return max(c.warmup_steps, horizon - (c.lr_decay_steps or max(1, horizon // 10)))

    def verify_fp8_compute(self, tag: str = "") -> bool | None:
        """跑一次前向，确认量化层**确实在用本组的精度计算**，并把结论写进日志。

        TE 在「该量化却没量化」时是静默的：模型照常训练，只是低精度名存实亡。
        与其去追一次性的告警，不如把「低精度是否真的在算」做成一个**可反复测量**
        的性质：启动时、建完驱动后、以及每个自检周期各查一次。
        探针还会分辨精度本身（fp4 被降级成 fp8 也算失败），见 `fp8.gemm_active`。

        返回 None 有两种含义，调用方都据此**不写** `fp8_active` 字段：
        「这条跑本来就是 bf16」，以及「TE 换了内部 API，查不到」。
        BF16 对照组以前在这里返回 True，写进 metrics 就成了 `fp8_active: true` ——
        照着日志看会得出「对照组也在跑 FP8」的结论，而这恰好是整个 A/B
        唯一要区分的那个变量。把「查不到」记成 False 是同一个错误的反方向。
        """
        if not self.model.cfg.quantized:
            return None
        prec = self.model.cfg.precision
        from .fp8 import gemm_active
        with torch.cuda.device(self.device), torch.autocast("cuda", dtype=torch.bfloat16):
            ok = gemm_active(
                self.model, prec,
                torch.zeros(8, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=self.device),
                torch.zeros(8, E.NUM_SCALARS, device=self.device))
        where = f"（{tag}）" if tag else ""
        verdict = {True: "已启用", False: f"未启用 —— GEMM 没走 {prec.upper()}！",
                   None: "查不到（TE 内部 API 变了，探针需要更新）"}[ok]
        print(f"[自检]{where} {prec.upper()} 计算{verdict}", flush=True)
        return ok

    # ---- 自博弈 ----
    def make_driver(self):
        """单卡返回 SelfPlayDriver，多卡返回 WorkerPool，两者接口一致。

        循环是同步的（自博弈与训练轮流跑），但**两个阶段都用全部 GPU**：
        自博弈各卡各跑自己的那批局，训练走 DDP（每卡 batch_size/N）。
        早先只有自博弈是多卡的，训练阶段另外三张卡全闲着。
        """
        from .devices import visible_devices
        c = self.cfg
        mcts = E.MctsConfig(
            simulations=c.simulations,
            max_considered=c.max_considered,
            temperature_plies=c.temperature_plies,
            c_visit=c.c_visit,
            c_scale=c.c_scale,
            value_from_score=c.value_from_score,
            random_opening_prob=c.random_opening_prob,
            random_opening_max_plies=c.random_opening_max_plies,
        )
        seed = int(self.rng.integers(1 << 30))
        devices = visible_devices(c.selfplay_devices)
        if len(devices) <= 1:
            self.pool = None
            return SelfPlayDriver(self.model, self.device, num_games=c.parallel_games,
                                  mcts=mcts, seed=seed, compile_model=True,
                                  engine_threads=c.engine_threads_per_gpu)
        # 多卡走每卡一个进程的工作池：自博弈与训练都在里面做。
        # 注意此后**权重的真身在工作进程**，self.model 只是镜像（见 WorkerPool.sync_weights）
        self.pool = WorkerPool(self.model, devices, c, mcts, seed)
        return self.pool

    def steps_for_iteration(self) -> int:
        """按 replay 里现有的数据量给本轮的训练步数限流。

        续训后如果 replay 快照丢了（或刚开跑），buffer 只有几万个局面，
        照 steps_per_iter x batch_size 训下去等于在同一批数据上过好多遍，
        loss 会掉得很好看，模型却在过拟合。实测这种情况下 loss 从 4.64 直接
        掉到 3.39 —— 数字变好，其实是坏了。
        """
        c = self.cfg
        cap = max(1, int(len(self.buffer) * c.max_epochs_per_iter / c.batch_size))
        return min(c.steps_per_iter, cap)

    # ---- 训练 ----
    def train_steps(self, n: int, should_stop=None) -> dict:
        c = self.cfg
        if getattr(self, "pool", None) is not None:
            t0 = time.perf_counter()
            rows = self.pool.train_steps(n, _Sampler(self),
                                         lambda i: self.lr_at(self.step + i),
                                         should_stop=should_stop)
            self.step += len(rows)
            if not rows:
                return {"lr": self.lr_at(self.step), "train_steps_per_s": 0.0}
            out = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            out["lr"] = self.lr_at(self.step)
            out["train_steps_per_s"] = len(rows) / (time.perf_counter() - t0)
            return out
        self.model.train()
        agg: dict[str, float] = {}
        t0 = time.perf_counter()
        done = 0
        for _ in range(n):
            if should_stop is not None and should_stop():
                break
            batch_np = self.buffer.sample(c.batch_size, self.rng,
                                          threads=c.loader_threads, augment=c.augment)
            batch = {k: torch.from_numpy(v).to(self.device, non_blocking=True)
                     for k, v in batch_np.items()}

            lr = self.lr_at(self.step)
            for gp in self.opt.param_groups:
                gp["lr"] = lr

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                out = self.model(batch["planes"], batch["scalars"])
                # 损失在 fp32 下算：策略是 17836 类的 log_softmax，BF16 精度不够
                out = (out[0].float(), out[1].float(), out[2].float())
                loss, parts = total_loss(out, batch, c.w_value, c.w_score)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            # 裁剪在 fp32 主权重的梯度上做，由 step 内部完成 ——
            # 累加永远要比乘法精度高，而 bf16 梯度直接求范数会丢有效数字
            gnorm = self.opt.step(grad_clip=c.grad_clip)
            self.step += 1

            parts["grad_norm"] = gnorm.detach()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            done += 1

        if done == 0:
            return {"lr": self.lr_at(self.step), "train_steps_per_s": 0.0}
        out = {k: v / done for k, v in agg.items()}
        out["lr"] = self.lr_at(self.step)
        out["train_steps_per_s"] = done / (time.perf_counter() - t0)
        return out

    # ---- checkpoint ----
    def save_checkpoint(self, tag: str | None = None, update_latest: bool = True) -> str:
        """`update_latest=False` 用于 WSD 的 stable 旁档：它是一份**分叉点**，
        不是这条跑的进度。让 `latest` 指过去的话，下次恢复会从它读起 ——
        本身状态是对的，但恢复出来的进程会把 `_stable_saved` 重置成 False，
        于是下一轮又往 `stable.pt` 上写一份**已经在退火段里**的权重，
        分叉点就悄悄漂走了。"""
        name = tag or f"step{self.step:08d}"
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        tmp = path + ".tmp"
        # `"model"` 存的是 **fp32 主权重**，不是 bf16 计算权重：主权重才是真值，
        # 计算权重精确等于 master.bfloat16()，随时可重建。这样存还有两个好处：
        # 键名/dtype/结构与历史 checkpoint 完全一致，所有读者不用改；
        # 而且 master 只有一份 —— torch 的优化器从不序列化 params。
        meta = {
            "step": self.step,
            "iteration": self.iteration,
            "games_played": self.games_played,
            "opt_format": "master-adamw-v1",
            "config": asdict(self.cfg),
            "model_config": asdict(self.model.cfg),
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        if getattr(self, "pool", None) is not None:
            # 权重与优化器状态都在工作进程里，由 rank 0 自己落盘 ——
            # 把 170 MB 的优化器状态传回父进程再写，纯属绕远
            self.pool.save_checkpoint(path, meta)
            self.last_ckpt_time = time.time()
            self.last_ckpt_step = self.step
            self._prune_checkpoints()
            if update_latest:
                with open(os.path.join(self.ckpt_dir, "latest"), "w") as f:
                    f.write(os.path.basename(path))
            return path

        # 单卡这条路径以前自己又拼了一份字段表，和上面的 meta 是**两处真值** ——
        # 加门控的簿记时只改了一处，于是单卡存档静默地少了冠军字段。
        # 现在两条路径共用 meta，新增字段不会再漏。
        model_sd = self.model.state_dict()          # 参数 + buffers + TE 的 _extra_state
        model_sd.update(self.opt.master_state_dict())
        blob = dict(meta)
        blob["model"] = model_sd
        blob["optimizer"] = self.opt.state_dict()
        torch.save(blob, tmp)
        os.replace(tmp, path)     # 原子替换，避免半截文件被当成有效 checkpoint
        self.last_ckpt_time = time.time()
        self.last_ckpt_step = self.step
        self._prune_checkpoints()
        if update_latest:
            with open(os.path.join(self.ckpt_dir, "latest"), "w") as f:
                f.write(os.path.basename(path))
        return path

    def _prune_checkpoints(self) -> None:
        """保留最近 K 份 + 里程碑份。

        只保留最近 K 份是不够的：网络强过全部规则基线之后，量 Elo 的唯一办法
        是和自己的历史版本对下（见 tools/compare_nets.py）。历史被删光就没法回溯了。
        """
        keep = self.cfg.keep_last
        every = max(1, self.cfg.milestone_every_steps)
        steps = []
        for f in os.listdir(self.ckpt_dir):
            if f.startswith("step") and f.endswith(".pt"):
                try:
                    steps.append(int(f[4:-3]))
                except ValueError:
                    pass
        steps.sort()
        if len(steps) <= keep:
            return

        # 里程碑不能用「step 是 every 的整数倍」判断：早期按数据量限流会让步数
        # 计数错位，实际落盘点是 19832、23832 这种数，永远命中不了整数倍 ——
        # 结果一个里程碑都留不下来，而没有对齐的历史 checkpoint，
        # 「在同一步数处头对头」就做不了（上一次对照实验就是栽在这上面）。
        # 改成：每跨过一个 every 区间，保留该区间里的第一个 checkpoint。
        milestones, seen = set(), set()
        for st in steps:
            bucket = st // every
            if bucket not in seen:
                seen.add(bucket)
                milestones.add(st)

        for st in steps[:-keep]:
            if st in milestones:
                continue
            try:
                os.remove(os.path.join(self.ckpt_dir, f"step{st:08d}.pt"))
            except OSError:
                pass

    def maybe_checkpoint(self) -> str | None:
        """按「距上次落盘的步数 / 秒数」触发，**不要**用 step 对间隔取模。

        取模那种写法（`step % every == 0`）看着等价，实际永远不会成立：
        早期按数据量限流会让步数错位成 19832 这种数，再也回不到整数倍上，
        于是只剩时间触发在起作用。开发机每 8 小时过期一次，
        丢多少进度完全取决于这个触发器，不能让它悄悄失效。
        """
        c = self.cfg
        due = (self.step - self.last_ckpt_step >= c.ckpt_every_steps) or \
              (time.time() - self.last_ckpt_time >= c.ckpt_every_seconds)
        return self.save_checkpoint() if due else None

    def load_checkpoint(self, path: str) -> None:
        """续训入口，**只吃训练档**（`runs/<exp>/ckpt/*.pt`）。

        与 `model.load_checkpoint`（只吃发布包）是两条互不相通的路。
        喂错时在这里就报出来，而不是让它掉进 `KeyError: 'optimizer'` ——
        那个报错离「你拿了个推理包」这个真相太远。
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        from .export import is_release
        if is_release(blob):
            raise ValueError(
                f"{path} 是推理发布包，没有优化器状态与 RNG，不能用来续训。\n"
                f"续训要用 runs/<exp>/ckpt/ 下的训练档；发布包只供评测与试玩。")
        if getattr(self, "pool", None) is not None:
            self.pool.load_checkpoint(path)         # 各 rank 各自读，不经父进程转发
            self.pool.pull_weights()                # 父进程的镜像也跟上
        else:
            # 先把 buffers 之类装好（参数会被下面的 master 覆盖成同一份值）
            load_weights(self.model, blob["model"])
        # master 与 bf16 参数由这一次调用一起同步，不给「顺序写反」留空间
            self.opt.load_state_dict(blob["optimizer"], master=blob["model"])
        self.step = blob["step"]
        self.last_ckpt_step = self.step
        self.iteration = blob["iteration"]
        self.games_played = blob.get("games_played", 0)
        self.rng.bit_generator.state = blob["rng"]
        torch.set_rng_state(blob["torch_rng"].cpu())

    def resume(self) -> bool:
        latest = os.path.join(self.ckpt_dir, "latest")
        if not os.path.exists(latest):
            return False
        with open(latest) as f:
            name = f.read().strip()
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            return False
        self.load_checkpoint(path)
        snap = os.path.join(self.snapshot_dir, "replay.npz")
        if os.path.exists(snap):
            try:
                self.buffer.load_shard(snap)
            except Exception as e:                      # noqa: BLE001
                # 快照读不出来（比如上次被强杀写坏了）不该拖垮整次续训：
                # 模型和优化器状态已经恢复了，replay 大不了重新攒。
                print(f"[续训] replay 快照损坏，忽略并从空 buffer 重新攒："
                      f"{type(e).__name__}: {e}")
        return True

    def save_snapshot(self, name: str = "replay.npz") -> None:
        # 热数据在本地盘，快照写工作目录 —— 换机器后靠它恢复
        self.buffer.save_shard(os.path.join(self.snapshot_dir, name))

    def maybe_snapshot_milestone(self) -> str | None:
        """每跨过一个里程碑桶，额外存一份**带步数的** replay 快照。

        默认只有一份 `replay.npz`、每次覆盖。那对续训够用（只需要最后一份），
        但**做不了「从第 N 万步分叉」这件事** —— 分叉要的是那一刻的 replay，
        而不是跑完时的。拿后者去分叉，退火段吃的是更晚、更强的模型产的数据，
        等于给分叉点开了后门。

        代价是每份约 450 MB。只在里程碑上存（默认每 1 万步），
        所以一条 9 万步的跑多占约 4 GB —— 换来的是分叉点可复现。
        """
        every = max(1, self.cfg.milestone_every_steps)
        bucket = self.step // every
        if bucket <= self._snap_bucket:
            return None
        self._snap_bucket = bucket
        name = f"step{self.step:08d}.npz"
        self.save_snapshot(name)
        return name

    def maybe_save_stable(self) -> bool:
        """WSD 跨进 decay 段的那一刻，额外留一份**永久**的 stable 档 + replay 快照。

        这是 WSD「stable 段可反复分叉」这个卖点的落地条件，而默认机制留不住它：

        * replay 快照只有一份、每次 `save_snapshot()` 覆盖同一个文件
        * 里程碑保留的是「每个 milestone_every_steps 桶里的**第一份**」，
          stable 终点（比如 100,000）落在桶 10 里，而桶 10 早被 100,0xx 之前的
          某一份占了 —— 所以 stable 那一步的档不会被保留

        没有这两份东西，将来想「从 stable 续到 25 万」就只能从零重跑。
        """
        if self.cfg.lr_schedule != "wsd" or self._stable_saved:
            return False
        if self.step < self.decay_start_step():
            return False
        path = os.path.join(self.ckpt_dir, "stable.pt")
        # 磁盘上已经有了就认它 —— 进程内的 `_stable_saved` 在每次恢复时归零，
        # 光靠它会让恢复后的第一轮又覆盖一次，而那时已经在退火段里了。
        if os.path.exists(path):
            self._stable_saved = True
            return False
        self.save_checkpoint(tag="stable", update_latest=False)
        self.save_snapshot("stable.npz")
        self._stable_saved = True
        print(f"[WSD] 已跨入退火段（step {self.step}），"
              f"stable 档与快照已永久保留：{path}", flush=True)
        return True

    # ---- 日志 ----
    def log(self, record: dict) -> None:
        record = {"step": self.step, "iteration": self.iteration,
                  "wall": time.time(), **record}
        self.history.append(record)
        with open(os.path.join(self.log_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
