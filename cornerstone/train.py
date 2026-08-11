"""训练循环。

M3 阶段是同步的：自博弈一批 -> 塞进 replay -> 训若干步 -> 周期性评测。
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
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import _engine as E
from .evaluate import evaluate_vs_baseline
from .losses import total_loss
from .model import CornerNet, ModelConfig, load_weights
from .optim import MasterWeightAdamW
from .replay import ReplayBuffer
from .selfplay import SelfPlayDriver


@dataclass
class TrainConfig:
    exp: str = "bf16"
    run_dir: str = ""          # 空则用 <repo>/../runs/<exp>
    hot_dir: str = ""          # 空则用 /tmp/cornerstone/<exp>

    # 模型
    dim: int = 256
    blocks: int = 16
    attn_every: int = 4
    fp8: bool = False

    # 自博弈
    parallel_games: int = 8192    # 多卡时按卡均分（4 卡 -> 每卡 2048，实测该点最优）
    compile_model: bool = True    # torch.compile 实测 2.4-2.5x，首次编译约 60s
    engine_threads: int = 128     # C++ 侧树搜索的总线程数，多卡时按卡均分
    selfplay_devices: str = ""    # 逗号分隔，空则用全部可见 GPU
    simulations: int = 64
    max_considered: int = 16
    temperature_plies: int = 12
    games_per_iter: int = 2048

    # 训练
    batch_size: int = 1024
    steps_per_iter: int = 400
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    total_steps: int = 200_000
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

    # checkpoint / 评测
    # 开发机每 8 小时过期一次（容器被回收，训练进程随之消失且来不及优雅收尾），
    # 所以丢失量由这两个阈值决定。取 400 = 一轮的步数，即每轮都落盘。
    ckpt_every_steps: int = 400
    ckpt_every_seconds: float = 240.0
    keep_last: int = 3
    milestone_every_steps: int = 10_000   # 里程碑 checkpoint 永久保留
    snapshot_every_iters: int = 20
    eval_every_iters: int = 10
    fp8_check_every_iters: int = 10   # FP8 静默降级只能靠反复测，不能挂在评测上
    eval_games: int = 200
    eval_opponent: str = "greedy-area"
    eval_simulations: int = 64

    seed: int = 1
    device: str = "cuda"

    def resolve(self, repo_root: str) -> "TrainConfig":
        if not self.run_dir:
            self.run_dir = os.path.join(os.path.dirname(repo_root), "runs", self.exp)
        if not self.hot_dir:
            self.hot_dir = os.path.join("/tmp", "cornerstone", self.exp)
        if self.fp8 and self.compile_model:
            # TE 的 FP8 自定义算子和 Dynamo 不兼容：先是 graph break 告警，
            # 随后编译出来的图会撞 CUDA 非法访存。两者只能二选一。
            # 代价是 FP8 跑拿不到 compile 的约 1.8x —— 这也算 FP8 的隐性成本之一。
            print("[配置] fp8 与 torch.compile 不兼容，本次自动关闭 compile")
            self.compile_model = False
        return self


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
            dim=cfg.dim, blocks=cfg.blocks, attn_every=cfg.attn_every, fp8=cfg.fp8
        )).to(self.device)
        self.opt = self._make_optimizer()
        self.model.to_param_dtype()
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

    def _make_optimizer(self) -> MasterWeightAdamW:
        """两条腿用同一个优化器 —— FP8 与否只影响 GEMM，不影响主权重与更新。"""
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
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        t = min(1.0, (step - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * cos)

    def verify_fp8_compute(self, tag: str = "") -> bool | None:
        """跑一次前向，确认 FP8 层**确实在用 FP8 计算**，并把结论写进日志。

        TE 在「该量化却没量化」时是静默的：模型照常训练，只是 FP8 名存实亡。
        与其去追一次性的告警，不如把「FP8 是否真的在算」做成一个**可反复测量**
        的性质：启动时、建完驱动后、以及每个自检周期各查一次。

        返回 None 有两种含义，调用方都据此**不写** `fp8_active` 字段：
        「这条跑本来就没开 FP8」，以及「TE 换了内部 API，查不到」。
        BF16 对照组以前在这里返回 True，写进 metrics 就成了 `fp8_active: true` ——
        照着日志看会得出「对照组也在跑 FP8」的结论，而这恰好是整个 A/B
        唯一要区分的那个变量。把「查不到」记成 False 是同一个错误的反方向。
        """
        if not self.cfg.fp8:
            return None
        from .fp8 import fp8_gemm_active
        with torch.cuda.device(self.device), torch.autocast("cuda", dtype=torch.bfloat16):
            ok = fp8_gemm_active(
                self.model,
                torch.zeros(8, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=self.device),
                torch.zeros(8, E.NUM_SCALARS, device=self.device))
        where = f"（{tag}）" if tag else ""
        verdict = {True: "已启用", False: "未启用 —— GEMM 没走 FP8！",
                   None: "查不到（TE 内部 API 变了，探针需要更新）"}[ok]
        print(f"[自检]{where} FP8 计算{verdict}", flush=True)
        return ok

    # ---- 自博弈 ----
    def make_driver(self):
        """单卡返回 SelfPlayDriver，多卡返回 MultiGpuSelfPlay，两者接口一致。

        循环是同步的（自博弈与训练轮流跑），所以自博弈阶段把**全部** GPU 都用上，
        训练阶段再回到主卡 —— 没有哪张卡会闲着。
        """
        from .multigpu import MultiGpuSelfPlay, visible_devices
        c = self.cfg
        mcts = E.MctsConfig(
            simulations=c.simulations,
            max_considered=c.max_considered,
            temperature_plies=c.temperature_plies,
        )
        seed = int(self.rng.integers(1 << 30))
        devices = visible_devices(c.selfplay_devices)
        if len(devices) <= 1:
            return SelfPlayDriver(self.model, self.device, num_games=c.parallel_games,
                                  mcts=mcts, seed=seed, compile_model=c.compile_model,
                                  engine_threads=c.engine_threads)
        return MultiGpuSelfPlay(self.model, devices, num_games=c.parallel_games,
                                mcts=mcts, seed=seed, compile_model=c.compile_model,
                                engine_threads=max(1, c.engine_threads // len(devices)))

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
    def save_checkpoint(self, tag: str | None = None) -> str:
        name = tag or f"step{self.step:08d}"
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        tmp = path + ".tmp"
        # `"model"` 存的是 **fp32 主权重**，不是 bf16 计算权重：主权重才是真值，
        # 计算权重精确等于 master.bfloat16()，随时可重建。这样存还有两个好处：
        # 键名/dtype/结构与历史 checkpoint 完全一致，所有读者不用改；
        # 而且 master 只有一份 —— torch 的优化器从不序列化 params。
        model_sd = self.model.state_dict()          # 参数 + buffers + TE 的 _extra_state
        model_sd.update(self.opt.master_state_dict())
        torch.save({
            "model": model_sd,
            "optimizer": self.opt.state_dict(),
            # 显式标记优化器状态的格式，比事后嗅探键名可靠
            "opt_format": "master-adamw-v1",
            "step": self.step,
            "iteration": self.iteration,
            "games_played": self.games_played,
            "config": asdict(self.cfg),
            "model_config": asdict(self.model.cfg),
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }, tmp)
        os.replace(tmp, path)     # 原子替换，避免半截文件被当成有效 checkpoint
        self.last_ckpt_time = time.time()
        self.last_ckpt_step = self.step
        self._prune_checkpoints()
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
        blob = torch.load(path, map_location="cpu", weights_only=False)
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

    def save_snapshot(self) -> None:
        # 热数据在本地盘，快照写工作目录 —— 换机器后靠它恢复
        self.buffer.save_shard(os.path.join(self.snapshot_dir, "replay.npz"))

    # ---- 日志 ----
    def log(self, record: dict) -> None:
        record = {"step": self.step, "iteration": self.iteration,
                  "wall": time.time(), **record}
        self.history.append(record)
        with open(os.path.join(self.log_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
