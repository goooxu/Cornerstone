"""每卡一个进程的工作池：自博弈 + 数据并行训练。

## 为什么是进程而不是线程

同进程多线程下 FP8 完全不扩展 —— 实测 4 卡 39k 评估/s，而 1 卡就有 35k，
加卡等于白加，4 卡甚至比 2 卡慢。换成每卡一个进程后线性扩展回来：

    FP8   1卡 43,659  2卡 82,809  4卡 172,557 评估/s   （3.95×）
    BF16  4卡线程 164,470  ->  4卡进程 187,856        （1.14×）

排查时排除了三个更省事的解释，记下来免得再走一遍：**不是编译粒度**
（按 block 47k vs 整个 block 循环编成一个区 38k，改了更差）、**不是
`fp8_autocast` 上下文的进出开销**（提到驱动层仍 1.01×）、**也不是 dynamo 的
重编译上限**（日志里确实报 `hit config.recompile_limit (8)`，因为 16 个
PolyBlock 共享一个代码对象；但抬到 256 之后吞吐一点没变）。剩下的解释是
同进程内的争用（GIL 与 TE 的全局 FP8 状态）。

## 为什么训练也放进来

修掉自博弈之后训练成了大头（实测墙钟 自博弈 36% / 训练 57%），而训练时
另外三张卡全闲着。数据并行在这里是**数学等价**的，不是近似：三个损失分量
都以 `.mean()` 收尾，所以「4 份 256 的梯度取平均」严格等于「1 份 1024 的梯度」。
梯度归约用 bf16 `all_reduce`，实测代价与取舍见 `_allreduce_grads`。
实测单卡步时 batch 1024 是 73.2 ms、batch 256 是 30.4 ms，4 卡 DDP 约 2.4×
（不到 4× 是因为模型太瘦，batch 256 就触到核函数启动的地板了 ——
128 和 256 的步时完全一样）。

## 结构

父进程只做协调：持有 replay buffer、采样、拼 metrics、管学习率与 checkpoint 元数据。
N 个工作进程对等，各占一张卡，自博弈与训练都在里面做。

三条容易写错的地方：

1. **工作进程必须常驻**。`torch.compile` 首次编译几十秒，每轮重启的话编译比
   干活还久。
2. **跨进程的 CUDA 操作没有隐式顺序**。父进程写完权重缓冲必须 `synchronize`
   再发命令，否则工作进程可能读到写了一半的权重 —— 这种错不报错，
   只会让自博弈悄悄用错版本的网络。
3. **集合通信里任一 rank 出错，其余会卡在 all_reduce 上死等**。所以工作进程
   捕获异常后要立刻把回溯发回父进程，父进程等回包时同时盯着进程存活。
"""

from __future__ import annotations

import os
import queue
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import _engine as E
from .losses import total_loss
from .model import CornerNet, ModelConfig
from .optim import MasterWeightAdamW
from .selfplay import SelfPlayDriver, SelfPlayStats

_READY_TIMEOUT = 900.0      # 首次编译 + 预热，四卡并行实测不到 2 分钟
_SLOTS = 2                  # 批次双缓冲：父进程填一个，工作进程正在训另一个


def _fields(obj) -> dict:
    """pybind11 的配置对象不可 pickle，只能把字段拆出来跨进程传。"""
    return {a: getattr(obj, a) for a in dir(obj) if not a.startswith("_")}


def _rebuild(kind, fields: dict):
    o = kind()
    for k, v in fields.items():
        setattr(o, k, v)
    return o


def _layout(model: CornerNet):
    """(参数名, 形状, 偏移, 元素数)。模型没有 buffer，只需覆盖参数。"""
    out, off = [], 0
    for n, p in model.named_parameters():
        out.append((n, tuple(p.shape), off, p.numel()))
        off += p.numel()
    return out


@torch.no_grad()
def _flat_to_model(model: CornerNet, flat: torch.Tensor, layout) -> None:
    """整块拷到本卡再散开。

    逐参数跨设备拷是 48 次小传输、启动开销主导；整块一次 57 MiB 走 NVLink
    只要零点几毫秒，之后的散开都是卡内拷贝。
    """
    dev = next(model.parameters()).device
    local = flat.to(dev, non_blocking=True) if flat.device != dev else flat
    params = dict(model.named_parameters())
    for name, shape, off, n in layout:
        params[name].copy_(local[off:off + n].view(shape))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def _model_to_flat(model: CornerNet, flat: torch.Tensor, layout,
                   masters: dict | None = None) -> None:
    params = dict(model.named_parameters())
    for name, _shape, off, n in layout:
        src = masters[name] if masters is not None else params[name].detach()
        flat[off:off + n].copy_(src.reshape(-1).to(flat.device))
    if flat.device.type == "cuda":
        torch.cuda.synchronize(flat.device)


@torch.no_grad()
def _allreduce_grads(params, buf: torch.Tensor, world: int) -> None:
    """梯度归约：打平成一块 bf16，一次 `all_reduce`。

    打平是必须的：48 次小集合会被延迟主导；合成一块之后 14.2M 参数 bf16 是 28 MB，
    在 NVLink 上约 0.2 ms，相对 40 ms 的步时可忽略。

    **归约就在 bf16 里做**，这是一处有意的偏离 —— 速查表 BF16 行写的是
    「梯度 bf16（规约常 fp32）」。NCCL（这里是 2.30.7）没有暴露归约精度的控制：
    329 个环境变量里没有任何与累加精度相关的，`ReduceOp` 也只有
    SUM/AVG/PRODUCT/MIN/MAX 那几个，语义就是按传入张量的 dtype 累加。
    要 fp32 求和只能改用 `all_gather` 再本地相加，流量翻一倍。

    实测代价（梯度范数相对单进程整批的相对差）：

        fp32 all_reduce                5.8e-05
        bf16 通信 + 本地 fp32 求和      2.4e-04
        **bf16 all_reduce（本实现）     7.9e-04**

    都远小于 bf16 梯度自身的相对分辨率 2^-8 ≈ 3.9e-03。而且真正吃精度的地方
    没有让步：优化器仍在 fp32 上更新（fp32 master + fp32 动量）。

    ⚠️ 这个结论**绑在 world=4 上** —— bf16 累加的误差随卡数增长，
    卡数明显变大时要重新评估。
    """
    off = 0
    for p in params:
        n = p.numel()
        if p.grad is None:
            buf[off:off + n].zero_()
        else:
            buf[off:off + n].copy_(p.grad.reshape(-1))
        off += n
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    buf.div_(world)
    off = 0
    for p in params:
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(buf[off:off + n].view_as(p.grad))
        off += n


def _make_optimizer(model: CornerNet, weight_decay: float, lr: float) -> MasterWeightAdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("pos") else decay).append((name, p))
    groups = [{"params": [p for _, p in decay], "names": [n for n, _ in decay],
               "weight_decay": weight_decay},
              {"params": [p for _, p in no_decay], "names": [n for n, _ in no_decay],
               "weight_decay": 0.0}]
    return MasterWeightAdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def _worker(rank: int, world: int, dev: str, spec: dict, flat: torch.Tensor, layout,
            req, rep, stop_ev) -> None:
    """常驻工作进程。命令：slots / selfplay / train / hash / pull / save / load / stop。

    `hash` 守的是「各 rank 的 learner 逐位相同」—— DDP 的核心不变量。
    它破了不会报错，只表现为四张卡在训练四个略微不同的网络。
    """
    model = opt = drv = None
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(spec["port"])
        torch.manual_seed(spec["seed"] + 1000 * rank)
        d = torch.device(dev)
        torch.cuda.set_device(d)

        # 构造顺序是硬的：fp32 初始化 -> 优化器取走 fp32 master -> 降到计算权重
        with torch.cuda.device(d):
            model = CornerNet(ModelConfig(**spec["model_cfg"])).to(d)
        opt = _make_optimizer(model, spec["weight_decay"], spec["lr"])
        model.to_param_dtype()
        _flat_to_model(model, flat, layout)          # rank 0 的初值广播给所有 rank
        opt.load_master_state_dict(
            {n: p.detach().float() for n, p in model.named_parameters()}, partial=True)

        if world > 1:
            dist.init_process_group("nccl", rank=rank, world_size=world)
        # 归约缓冲用梯度自身的 dtype（bf16），不上抬 —— 见 _allreduce_grads
        gbuf = torch.zeros(sum(x[3] for x in layout),
                           dtype=next(model.parameters()).dtype, device=d)
        params = [p for _, p in model.named_parameters()]

        # 训练步的热点编译。**这在工作池路径上漏了很久**：
        # `Trainer._compile_hot_modules()` 只作用于父进程里那个镜像模型，
        # 而真正在训练的是工作进程自己 new 出来的这一份。
        # `docs/07` 实测只编译 SwiGLU 能让训练步快 1.38×，而训练占墙钟约 41% ——
        # 漏掉等于白丢十几个百分点，且没有任何症状。
        if d.type == "cuda":
            if spec["model_cfg"].get("precision", "bf16") != "bf16":
                # 低精度只能按 block 编译（整模型会把 TE 的全局量化上下文编进图里，
                # 多卡并发跑就段错误）。下面建驱动时还会调一次，那个是幂等的。
                from .selfplay import compile_for_inference
                compile_for_inference(model)
            else:
                for blk in model.blocks:
                    blk.mlp.compile(dynamic=False)   # 原地，不动 state_dict 的键

        drv = SelfPlayDriver(model, d, num_games=spec["games_per_gpu"],
                             mcts=_rebuild(E.MctsConfig, spec["mcts"]),
                             seed=spec["seed"] + 1000 * rank,
                             compile_model=True, engine_threads=spec["engine_threads"])
        drv.warmup()
        rep.put(("ready", None))
        slots = None

        while True:
            msg = req.get()
            cmd = msg[0]
            if cmd == "stop":
                return

            if cmd == "slots":                        # 父进程把批次缓冲发过来（只发一次）
                slots = msg[1]
                rep.put(("ok", None))

            elif cmd == "selfplay":
                _, target, max_seconds, reload_w = msg
                if reload_w:
                    _flat_to_model(model, flat, layout)
                recs, st = drv.run(target, max_seconds=max_seconds,
                                   should_stop=stop_ev.is_set)
                rep.put(("ok", (recs, (st.games, st.evals, st.nn_calls,
                                       st.gpu_seconds, st.seconds))))

            elif cmd == "train":
                _, slot, lr = msg
                for gp in opt.param_groups:
                    gp["lr"] = lr
                batch = {k: v.to(d, non_blocking=True) for k, v in slots[slot].items()}
                model.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # **多卡走的是这一份训练步，不是 train.py 里那份。**
                    # 加归属头时只改了那边，结果 own-bf16 起跑后指标里根本没有
                    # owner 项 —— 模型建对了（model_cfg 带着 owner_head），
                    # 只是从没被要求输出，训练照跑、loss 照降，完全看不出来。
                    ow = bool(spec.get("owner_head"))
                    out = model(batch["planes"], batch["scalars"], with_owner=ow)
                    out = tuple(t.float() for t in out if t is not None)
                    loss, parts = total_loss(out, batch, spec["w_value"], spec["w_score"],
                                             spec.get("w_owner", 0.25))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if world > 1:
                    _allreduce_grads(params, gbuf, world)
                gnorm = opt.step(grad_clip=spec["grad_clip"])
                parts["grad_norm"] = gnorm
                rep.put(("ok", {k: float(v) for k, v in parts.items()}))

            elif cmd == "hash":                       # 守「各 rank 的 learner 逐位相同」
                import hashlib
                h = hashlib.blake2b(digest_size=16)
                for name, p in sorted(model.named_parameters()):
                    h.update(name.encode())
                    h.update(p.detach().float().cpu().numpy().tobytes())
                rep.put(("ok", h.hexdigest()))

            elif cmd == "pull":                       # rank 0 把主权重写回共享缓冲
                if rank == 0:
                    _model_to_flat(model, flat, layout, opt.master_state_dict())
                rep.put(("ok", None))

            elif cmd == "save":
                _, path, meta = msg
                if rank == 0:
                    sd = model.state_dict()
                    sd.update(opt.master_state_dict())
                    blob = dict(meta)
                    blob["model"] = {k: v.cpu() for k, v in sd.items()}
                    blob["optimizer"] = opt.state_dict()
                    tmp = path + ".tmp"
                    torch.save(blob, tmp)
                    os.replace(tmp, path)
                rep.put(("ok", None))

            elif cmd == "load":
                _, blob_path = msg
                blob = torch.load(blob_path, map_location="cpu", weights_only=False)
                from .model import load_weights
                load_weights(model, blob["model"])
                opt.load_state_dict(blob["optimizer"], master=blob["model"])
                rep.put(("ok", None))
    except BaseException:                              # noqa: BLE001
        # 必须回传回溯：否则父进程只看到「工作进程没了」，而其余 rank 还会
        # 卡在 all_reduce 上死等
        try:
            rep.put(("err", traceback.format_exc()))
        except Exception:                              # noqa: BLE001
            pass
    finally:
        try:
            if world > 1 and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:                              # noqa: BLE001
            pass


class WorkerPool:
    """N 个对等工作进程，各占一张卡，自博弈与训练都在里面做。"""

    def __init__(self, model: CornerNet, devices, cfg, mcts: E.MctsConfig,
                 seed: int, port: int = 0):
        self.source = model
        self.devices = [str(d) for d in devices]
        self.world = len(self.devices)
        self.cfg = cfg
        self.layout = _layout(model)
        total = sum(x[3] for x in self.layout)

        src = next(model.parameters()).device
        if src.type == "cuda":
            with torch.cuda.device(src):
                self.flat = torch.empty(total, dtype=torch.float32, device=src)
        else:
            self.flat = torch.empty(total, dtype=torch.float32).share_memory_()
        _model_to_flat(model, self.flat, self.layout)

        ctx = mp.get_context("spawn")
        self.stop_ev = ctx.Event()
        self.reqs = [ctx.Queue() for _ in range(self.world)]
        self.reps = [ctx.Queue() for _ in range(self.world)]
        spec = dict(
            model_cfg=dict(model.cfg.__dict__), mcts=_fields(mcts), seed=seed,
            games_per_gpu=max(1, cfg.parallel_games // self.world),
            engine_threads=cfg.engine_threads_per_gpu, weight_decay=cfg.weight_decay,
            lr=cfg.lr, grad_clip=cfg.grad_clip, w_value=cfg.w_value, w_score=cfg.w_score,
            owner_head=cfg.owner_head, w_owner=cfg.w_owner,
            port=port or (29500 + (os.getpid() % 2000)),
        )
        self.procs = []
        for i, dev in enumerate(self.devices):
            p = ctx.Process(target=_worker,
                            args=(i, self.world, dev, spec, self.flat, self.layout,
                                  self.reqs[i], self.reps[i], self.stop_ev),
                            daemon=True)
            p.start()
            self.procs.append(p)
        for i in range(self.world):
            kind, payload = self._recv(i, _READY_TIMEOUT)
            if kind != "ready":
                self.close()
                raise RuntimeError(f"工作进程 {i}（{self.devices[i]}）启动失败：\n{payload}")
        self._pending_reload = False
        self._slots = None

    # ---- 内部 ----

    def _recv(self, i: int, timeout: float):
        end = time.perf_counter() + timeout
        while True:
            try:
                return self.reps[i].get(timeout=0.5)
            except queue.Empty:
                if not self.procs[i].is_alive():
                    raise RuntimeError(f"工作进程 {i}（{self.devices[i]}）意外退出，"
                                       f"exitcode={self.procs[i].exitcode}")
                if time.perf_counter() > end:
                    raise TimeoutError(f"等工作进程 {i} 超时（{timeout:.0f}s）")

    def _bcast(self, msg, timeout: float = 3600.0):
        for q in self.reqs:
            q.put(msg)
        out = []
        for i in range(self.world):
            kind, payload = self._recv(i, timeout)
            if kind == "err":
                raise RuntimeError(f"工作进程 {i} 出错：\n{payload}")
            out.append(payload)
        return out

    def _ensure_slots(self, batch: dict) -> None:
        """按第一批的形状与 dtype 建共享固定内存，只建一次。"""
        if self._slots is not None:
            return
        per = self.cfg.batch_size // self.world
        self._slots = []
        for _ in range(self.world):
            per_worker = []
            for _s in range(_SLOTS):
                d = {}
                for k, v in batch.items():
                    # 形状与 dtype 照第一批的来，不硬编码 —— replay 的输出里
                    # 有 uint8/int32/int64/float32 四种，写死迟早对不上
                    t = torch.empty((per,) + v.shape[1:], dtype=torch.from_numpy(v[:1]).dtype)
                    d[k] = t.share_memory_()
                per_worker.append(d)
            self._slots.append(per_worker)
        for i, q in enumerate(self.reqs):
            q.put(("slots", self._slots[i]))
        for i in range(self.world):
            self._recv(i, 120.0)

    # ---- 对外 ----

    def sync_weights(self) -> None:
        """**空操作**，接口留着是为了和 `SelfPlayDriver` 一致。

        这是用工作池之后一个语义反转，写错就会毁掉训练：训练也在工作进程里做，
        所以**权重的真身在工作进程**，父进程的 `self.source` 只是个镜像。
        再从父进程往下推就是拿陈旧权重覆盖掉刚训出来的结果 —— 而且不报错，
        只表现为「怎么训都不涨棋力」。

        工作进程自博弈时直接用自己刚更新过的权重，本来就不需要同步。
        父进程要看权重（落 checkpoint 之外的用途）走 `pull_weights()`。
        """

    def weight_hashes(self) -> list[str]:
        """各 rank 的 learner 权重哈希。**全都相同**是本地晋升成立的前提。"""
        for q in self.reqs:
            q.put(("hash",))
        out = []
        for i in range(self.world):
            kind, payload = self._recv(i, 300.0)
            if kind == "err":
                raise RuntimeError(f"工作进程 {i} 取哈希出错：\n{payload}")
            out.append(payload)
        return out

    def run(self, target_games: int, max_seconds: float | None = None,
            should_stop=None) -> tuple[list[dict], SelfPlayStats]:
        per = max(1, target_games // self.world)
        t0 = time.perf_counter()
        for q in self.reqs:
            q.put(("selfplay", per, max_seconds, self._pending_reload))
        self._pending_reload = False
        out, agg = [], SelfPlayStats()
        pending = set(range(self.world))
        while pending:
            if should_stop is not None and should_stop():
                self.stop_ev.set()
            for i in sorted(pending):
                try:
                    kind, payload = self.reps[i].get(timeout=0.2)
                except queue.Empty:
                    if not self.procs[i].is_alive():
                        raise RuntimeError(f"工作进程 {i} 意外退出，"
                                           f"exitcode={self.procs[i].exitcode}")
                    continue
                if kind == "err":
                    raise RuntimeError(f"工作进程 {i} 出错：\n{payload}")
                recs, (g, ev, nc, gs, _s) = payload
                out.extend(recs)
                agg.games += g
                agg.evals += ev
                agg.nn_calls += nc
                agg.gpu_seconds += gs
                pending.discard(i)
        self.stop_ev.clear()
        agg.seconds = time.perf_counter() - t0
        return out, agg

    def _fill(self, batch: dict, slot: int) -> None:
        per = self.cfg.batch_size // self.world
        for i in range(self.world):
            dst = self._slots[i][slot]
            for k, v in batch.items():
                dst[k].copy_(torch.from_numpy(v[i * per:(i + 1) * per]))

    def _send_train(self, slot: int, lr: float) -> None:
        for q in self.reqs:
            q.put(("train", slot, lr))

    def _collect_train(self) -> dict:
        parts = []
        for i in range(self.world):
            kind, payload = self._recv(i, 600.0)
            if kind == "err":
                raise RuntimeError(f"工作进程 {i} 出错：\n{payload}")
            parts.append(payload)
        # 各 rank 的 parts 是各自 1/N 切片上的均值，取平均就是整批的均值
        return {k: sum(p[k] for p in parts) / self.world for k in parts[0]}

    def train_steps(self, n: int, sample_fn, lr_fn, should_stop=None) -> list[dict]:
        """流水线式训练：填下一批与工作进程算当前批**重叠**。

        父进程每步要把 24.9 MiB 的批切成 N 份拷进共享缓冲，实测约 10 ms；
        同步填充时这 10 ms 全暴露在关键路径上（步时 40.5 ms，纯 GPU 才 30.4）。
        双缓冲之后它藏进工作进程的计算里。
        """
        if n <= 0:
            return []
        self._ensure_slots(sample_fn.peek())
        out, slot = [], 0
        self._fill(sample_fn(), slot)
        self._send_train(slot, lr_fn(0))
        for i in range(n):
            nxt = (slot + 1) % _SLOTS
            more = i + 1 < n and not (should_stop is not None and should_stop())
            if more:
                self._fill(sample_fn(), nxt)      # 工作进程正在算上一批
            out.append(self._collect_train())
            if not more:
                break
            self._send_train(nxt, lr_fn(i + 1))
            slot = nxt
        return out

    def pull_weights(self) -> None:
        """把 rank 0 的 fp32 主权重取回父进程的镜像模型（落 checkpoint 用）。"""
        self._bcast(("pull",))
        _flat_to_model(self.source, self.flat, self.layout)

    def save_checkpoint(self, path: str, meta: dict) -> None:
        self._bcast(("save", path, meta))

    def load_checkpoint(self, path: str) -> None:
        self._bcast(("load", path))

    def close(self) -> None:
        for q in getattr(self, "reqs", []):
            try:
                q.put(("stop",))
            except Exception:                          # noqa: BLE001
                pass
        for p in getattr(self, "procs", []):
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        self.procs = []

    def __del__(self):
        try:
            self.close()
        except Exception:                              # noqa: BLE001
            pass
