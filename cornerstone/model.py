"""CornerNet —— 自研网络结构。

设计围绕两件事：

1. **FP8 GEMM 是主体**。深度可分离卷积负责空间归纳偏置但只占 <2% 的 FLOPs，
   通道混合的两个大 GEMM 才是计算主体 —— M4 里把它们换成 FP8 就能吃到
   Blackwell 张量核的加速。所以主干刻意做成「便宜的空间算子 + 昂贵的 Linear」。

2. **零拷贝的两种视图**。张量全程保持 ``(B, 196, D)`` 连续布局。
   一个连续的 ``(B, 196, D)`` 张量在物理上**就是** channels_last 的 ``(B, D, 14, 14)``，
   所以 ``x.view(B,14,14,D).permute(0,3,1,2)`` 是纯视图、零拷贝，
   卷积走 cuDNN 的 channels_last 快路径，回来同样零拷贝。
   如果不注意这一点，每个 block 要来回两次 200MB 级别的转置。

与方案里最初设想的差别：库存信息没有做成额外 token，而是投影成一个向量广播加到每个格上。
信息等价，但省掉了「196 个格 token + 2 个特殊 token」带来的形状特例。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _engine as E

BOARD = E.BOARD_N
CELLS = E.NUM_CELLS
PLANES = E.NUM_PLANES
SCALARS = E.NUM_SCALARS
ORI = E.NUM_ORI
ACTIONS = E.NUM_ACTIONS


@dataclass
class ModelConfig:
    dim: int = 256
    blocks: int = 16
    mlp_ratio: int = 4
    attn_every: int = 4          # 每隔几个 block 插一层全局注意力，0 表示不插
    heads: int = 8
    dw_kernel: int = 5           # 深度可分离卷积核大小
    fp8: bool = False            # M4 打开：把 MLP/注意力的 Linear 换成 FP8
    fp8_first_last_bf16: bool = True   # 首尾 block 保持高精度

    @property
    def hidden(self) -> int:
        return self.dim * self.mlp_ratio


def make_linear(cfg: ModelConfig, in_f: int, out_f: int, bias: bool = False,
                force_bf16: bool = False) -> nn.Module:
    """MLP / 注意力投影用的线性层。

    M4 会在这里返回 transformer_engine 的 FP8 Linear；现在统一是 nn.Linear，
    但结构上已经把切换点收敛到这一个函数里。
    """
    if cfg.fp8 and not force_bf16:
        from .fp8 import fp8_linear
        return fp8_linear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class SwiGLU(nn.Module):
    """通道混合。两个 Linear 占了全网络绝大部分 FLOPs。"""

    def __init__(self, cfg: ModelConfig, force_bf16: bool = False):
        super().__init__()
        d, h = cfg.dim, cfg.hidden
        self.up = make_linear(cfg, d, 2 * h, force_bf16=force_bf16)
        self.down = make_linear(cfg, h, d, force_bf16=force_bf16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class DepthwiseSpatial(nn.Module):
    """空间混合：深度可分离卷积。参数与 FLOPs 都极小，但提供关键的局部归纳偏置。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        k = cfg.dw_kernel
        self.conv = nn.Conv2d(cfg.dim, cfg.dim, k, padding=k // 2, groups=cfg.dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 196, D) 连续 —— 物理布局即 channels_last 的 (B, D, 14, 14)
        b, n, d = x.shape
        xc = x.view(b, BOARD, BOARD, d).permute(0, 3, 1, 2)   # 视图，零拷贝
        y = self.conv(xc)
        return y.permute(0, 2, 3, 1).reshape(b, n, d)          # 视图，零拷贝


class Attention(nn.Module):
    """196 个格 token 之间的全局注意力。

    用来做「哪片区域还够得着」这类判断 —— 纯局部卷积要堆很多层才能传播这种信息。
    """

    def __init__(self, cfg: ModelConfig, force_bf16: bool = False):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.qkv = make_linear(cfg, cfg.dim, 3 * cfg.dim, force_bf16=force_bf16)
        self.proj = make_linear(cfg, cfg.dim, cfg.dim, force_bf16=force_bf16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(b, n, d))


class PolyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, with_attn: bool, force_bf16: bool = False):
        super().__init__()
        self.norm_sp = nn.RMSNorm(cfg.dim)
        self.spatial = DepthwiseSpatial(cfg)
        self.norm_mlp = nn.RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg, force_bf16=force_bf16)
        self.attn = None
        if with_attn:
            self.norm_attn = nn.RMSNorm(cfg.dim)
            self.attn = Attention(cfg, force_bf16=force_bf16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial(self.norm_sp(x))
        if self.attn is not None:
            x = x + self.attn(self.norm_attn(x))
        return x + self.mlp(self.norm_mlp(x))


class CornerNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.dim

        self.stem = nn.Conv2d(PLANES, d, 3, padding=1, bias=True)
        self.pos = nn.Parameter(torch.zeros(1, CELLS, d))
        # 双方剩余棋子 + 占格数 -> 广播到每个格
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALARS, d), nn.SiLU(), nn.Linear(d, d)
        )

        last = cfg.blocks - 1
        self.blocks = nn.ModuleList([
            PolyBlock(
                cfg,
                with_attn=(cfg.attn_every > 0 and (i + 1) % cfg.attn_every == 0),
                # 首尾 block 对精度最敏感，FP8 时保持 BF16
                force_bf16=(cfg.fp8_first_last_bf16 and i in (0, last)),
            )
            for i in range(cfg.blocks)
        ])
        self.norm_out = nn.RMSNorm(d)

        # 策略头：每个格给出 91 个朝向的 logit。动作编号 = ori*196 + 格号
        self.policy = nn.Linear(d, ORI)
        # 价值头：胜/和/负三分类。和局是本项目里的真实结果，不能用 tanh 标量糊过去
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 3))
        # 辅助头：终局占格数差，信号比稀疏的三分类结果密集得多
        self.score = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos, std=0.02)
        # 策略头零初始化 -> 训练一开始策略就是均匀分布，不会给 MCTS 一个随机的强先验
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

    def trunk(self, planes: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = planes.shape[0]
        x = self.stem(planes.to(memory_format=torch.channels_last))
        x = x.permute(0, 2, 3, 1).reshape(b, CELLS, self.cfg.dim)
        x = x + self.pos + self.scalar_mlp(scalars).unsqueeze(1)
        # FP8 的作用域只包住主干：stem/头部/损失都留在高精度
        with self._fp8_scope():
            for blk in self.blocks:
                x = blk(x)
        return self.norm_out(x)

    def _fp8_scope(self):
        if not self.cfg.fp8:
            return contextlib.nullcontext()
        from .fp8 import fp8_autocast
        return fp8_autocast()

    def _device_scope(self):
        """FP8 下把当前 CUDA 设备设成本模型所在的卡。

        TransformerEngine 按**当前设备**取 cuBLAS 句柄/上下文。当前设备与张量设备
        不一致时，它的表现是**两种**，都很难查：
          - 有时静默退回非量化计算，只发一条 UserWarning（FP8 就名存实亡了）
          - 有时直接 `CUDA error: an illegal memory access was encountered`，
            而且报错点常飘到别的算子上

        所以护栏放在 forward 里，而不是逐个调用点去打补丁 ——
        训练步、自博弈、评测、Web 推理、checkpoint 读写有五六条路径，
        漏掉任何一条都会以上面两种形态之一炸出来。
        `torch.cuda.device` 是线程局部的，多卡多线程各设各的互不干扰。
        """
        if not self.cfg.fp8:
            return contextlib.nullcontext()
        dev = self.pos.device
        return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor):
        """返回 (policy_logits[B, 17836], wdl_logits[B, 3], score_diff[B])。

        policy_logits 未做合法性 mask —— mask 由调用方施加（训练和推理的 mask 来源不同）。
        """
        with self._device_scope():
            return self._forward(planes, scalars)

    def _forward(self, planes: torch.Tensor, scalars: torch.Tensor):
        b0 = planes.shape[0]
        if self.cfg.fp8:
            # MXFP8 要求 GEMM 的两维都是 32 的倍数，token 维是 B*196 且 196%32==4，
            # 所以 B 必须是 8 的倍数。自博弈的批大小是变的，这里补齐再切回来。
            from .fp8 import pad_to_mxfp8
            bp = pad_to_mxfp8(b0)
            if bp != b0:
                planes = torch.cat([planes, planes[-1:].expand(bp - b0, -1, -1, -1)], 0)
                scalars = torch.cat([scalars, scalars[-1:].expand(bp - b0, -1)], 0)

        h = self.trunk(planes, scalars)
        b = h.shape[0]

        # (B, 196, 91) -> (B, 91, 196) -> 展平成 ori*196+cell
        pol = self.policy(h).transpose(1, 2).reshape(b, ACTIONS)

        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        wdl, sc = self.value(pooled), self.score(pooled).squeeze(-1)
        if b != b0:
            pol, wdl, sc = pol[:b0], wdl[:b0], sc[:b0]      # 去掉补齐用的样本
        return pol, wdl, sc

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_checkpoint(path: str, device="cuda") -> tuple["CornerNet", object]:
    """从 checkpoint 建模型。FP8 与非 FP8 的 checkpoint 都能读。

    统一走这里，别在各处自己 torch.load + load_state_dict —— FP8 模型的参数是
    MXFP8 张量，必须在目标设备的上下文里构造、并走重量化路径装载，
    否则会得到 `cublas_gemm: failed to launch on the GPU` 这种离根因很远的错。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mc = blob.get("model_config") or {}
    cfg = ModelConfig(**{k: v for k, v in mc.items()
                         if k in ModelConfig.__dataclass_fields__})
    dev = torch.device(device)
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
    with ctx:
        model = CornerNet(cfg)
        if cfg.fp8:
            from .fp8 import load_state_dict_into
            model = model.to(dev)
            load_state_dict_into(model, blob["model"])
        else:
            model.load_state_dict(blob["model"])
            model = model.to(dev)
    return model.eval(), blob.get("step", "?")


def mask_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """把非法着法的 logit 压到 -inf。legal 是 0/1 或 bool 张量，形状 [B, 17836]。"""
    return logits.masked_fill(~legal.bool(), float("-inf"))
