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
from .model import CornerNet, ModelConfig
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
    stochastic_rounding: bool = True   # FP8 主权重下关掉它是对照实验用的

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
    ckpt_every_steps: int = 2000
    ckpt_every_seconds: float = 600.0
    keep_last: int = 3
    milestone_every_steps: int = 10_000   # 里程碑 checkpoint 永久保留
    snapshot_every_iters: int = 20
    eval_every_iters: int = 10
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

        self.model = CornerNet(ModelConfig(
            dim=cfg.dim, blocks=cfg.blocks, attn_every=cfg.attn_every, fp8=cfg.fp8
        )).to(self.device)

        self.opt = self._make_optimizer()
        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.step = 0
        self.iteration = 0
        # 全生命周期产生过的对局数。不能用 buffer.total_games_seen 代替 ——
        # 那是进程内计数器，续训时会被快照重新播种，看起来像「从头训了」。
        self.games_played = 0
        self.last_ckpt_time = time.time()
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

    def _make_optimizer(self) -> torch.optim.Optimizer:
        # Norm / bias / 位置嵌入不做权重衰减
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith("pos") else decay).append(p)
        groups = [{"params": decay, "weight_decay": self.cfg.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        if self.cfg.fp8:
            # 主权重就是 FP8，没有高精度副本，更新必须走随机舍入
            from .fp8 import Fp8AdamW
            return Fp8AdamW(groups, lr=self.cfg.lr, betas=(0.9, 0.95), eps=1e-8,
                            stochastic_rounding=self.cfg.stochastic_rounding)
        return torch.optim.AdamW(groups, lr=self.cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        t = min(1.0, (step - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * cos)

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
        if c.fp8 and len(devices) > 1:
            # TE 的 FP8 状态（cuBLAS 工作区、句柄）是**进程级且绑定单设备**的：
            # 同一进程里在第二张卡上做 FP8 GEMM 会直接 "failed to launch on the GPU"。
            # 正确的多卡 FP8 做法是每卡一个独立进程，属于后续工作。
            print(f"[配置] FP8 模式下自博弈只能单卡，忽略 {devices[1:]}")
            devices = [str(self.device)]
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
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            self.opt.step()
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
        # 量化参数反量化后再存：MXFP8 张量跨设备/跨进程加载会失败，
        # 而 FP8 只是训练期主权重的格式，不必也不该是序列化格式
        model_sd = self.model.state_dict()
        if self.cfg.fp8:
            from .fp8 import dequantized_state_dict
            model_sd = dequantized_state_dict(self.model)
        torch.save({
            "model": model_sd,
            "optimizer": self.opt.state_dict(),
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
        files = sorted(f for f in os.listdir(self.ckpt_dir)
                       if f.startswith("step") and f.endswith(".pt"))
        if len(files) <= keep:
            return
        for f in files[:-keep]:
            try:
                step = int(f[4:-3])
            except ValueError:
                continue
            if step % every == 0:
                continue                      # 里程碑，留着
            try:
                os.remove(os.path.join(self.ckpt_dir, f))
            except OSError:
                pass

    def maybe_checkpoint(self) -> str | None:
        c = self.cfg
        due = (self.step > 0 and self.step % c.ckpt_every_steps == 0) or \
              (time.time() - self.last_ckpt_time >= c.ckpt_every_seconds)
        return self.save_checkpoint() if due else None

    def load_checkpoint(self, path: str) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if self.cfg.fp8:
            from .fp8 import load_state_dict_into
            load_state_dict_into(self.model, blob["model"])
        else:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in blob["model"].items()})
        self.opt.load_state_dict(blob["optimizer"])
        self.step = blob["step"]
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
