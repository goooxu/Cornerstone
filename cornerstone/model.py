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
    # 主干 GEMM 的计算精度：bf16 | fp8(MXFP8) | fp4(NVFP4)。三者的**参数存储完全
    # 相同**，差别只在 te.Linear 前向时用哪个量化配方。
    precision: str = "bf16"
    # 兼容别名。**不能删** —— `load_checkpoint` 对未知键做静默过滤，而
    # `te.Linear` 与 `nn.Linear` 的 state_dict 键名都是 `weight`、`_extra_state`
    # 又被 `load_weights` 剥掉：删了它，`runs/v2-fp8` 那些只写了 `fp8: true` 的老
    # checkpoint 会**静默降级成 BF16 模型**，不抛任何异常。新代码一律读 `precision`。
    fp8: bool | None = None
    fp8_first_last_bf16: bool = True   # 首尾 block 不走低精度
    # 计算权重的精度。**一切推理都用它**，fp32 主权重只服务优化器。
    # 老 checkpoint 的 model_config 里没有这个字段，加载时走的也是这个默认值。
    param_dtype: str = "bf16"

    def __post_init__(self):
        # 只有老 blob（有 fp8、无 precision）才走这条映射；新配置里 precision 已经
        # 显式给了，此时不能让 fp8=False 把它顶回 bf16。
        if self.fp8 is not None and self.precision == "bf16":
            self.precision = "fp8" if self.fp8 else "bf16"
        self.fp8 = (self.precision == "fp8")       # 回填，asdict 出来的 blob 仍带它
        # `tools/train.py` 的 CLI 是从 dataclass 自动生成的，没有 choices 校验 ——
        # `--precision fp16` 这种手误会安静地建出一个错配置，这句 assert 是唯一防线。
        assert self.precision in ("bf16", "fp8", "fp4"), f"未知精度 {self.precision!r}"

    @property
    def quantized(self) -> bool:
        """GEMM 是否走低精度。判「要不要 autocast / 补齐 batch」一律用它。"""
        return self.precision in ("fp8", "fp4")

    @property
    def hidden(self) -> int:
        return self.dim * self.mlp_ratio

    @property
    def torch_param_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.param_dtype]


def make_linear(cfg: ModelConfig, in_f: int, out_f: int, bias: bool = False,
                force_bf16: bool = False) -> nn.Module:
    """MLP / 注意力投影用的线性层 —— 低精度与否的唯一切换点。

    **各组的初始权重必须逐位相同**，否则 A/B 里就混进了一个隐藏变量。
    `te.Linear` 默认用 `normal(0, 0.023)` 初始化，而 `nn.Linear` 用 kaiming_uniform，
    分布本身就不一样；更麻烦的是两者消耗的 RNG 流长度不同，会让**后面所有层**
    （包括不含 TE 的卷积和位置嵌入）跟着错位。实测同一 seed 下 126 个参数张量里
    有 75 个不同。所以这里无条件先建一个 fp32 的 nn.Linear 当初始化参考，
    FP8 路径把它的权重拷进去 —— RNG 消耗和初值就都对齐了。

    te.Linear 也以 fp32 构造：全模型统一在 fp32 下初始化，优化器取走 fp32 master
    之后再由 `CornerNet.to_param_dtype()` 整体降到 bf16。
    """
    ref = nn.Linear(in_f, out_f, bias=bias)
    if not (cfg.quantized and not force_bf16):
        return ref
    from .fp8 import quant_linear
    lin = quant_linear(cfg.precision, in_f, out_f, bias=bias, params_dtype=torch.float32)
    with torch.no_grad():
        lin.weight.copy_(ref.weight)
        if bias:
            lin.bias.copy_(ref.bias)
    return lin


def match_dtype(x: torch.Tensor, lin: nn.Module) -> torch.Tensor:
    """把输入折回线性层权重的 dtype，再交给它。

    **为什么必须有这一步**：`nn.RMSNorm` 在 `torch.autocast` 下按 fp32 策略执行，
    输出是 **fp32**，而它的下游正是 SwiGLU 与 Attention 的投影。
    BF16 与 MXFP8 两组对此无所谓（autocast 会在进 GEMM 前把它 cast 回 bf16），
    **但 NVFP4 会直接报错**：

        RHT is only supported for bfloat16 input, got dtype enum value 4

    （4 = kFloat32。随机 Hadamard 变换是 FP4 收敛的必需件，见 fp8.py，
    不能为了绕开它去关 `disable_rht`。）报错点在 `blocks.1.attn.qkv` ——
    第一个不被首尾 bf16 规则豁免的 te.Linear。

    实测的影响（dim=64/blocks=4 的小模型，同 seed 同输入）：

    * **BF16 组逐位不变** —— autocast 本来就会做同一个 cast，只是做得更晚。
    * **FP8 组会有极小的变化**（wdl 最大绝对差 4.4e-3、score 5.9e-3）：
      MXFP8 从 fp32 量化和从 bf16 量化落到的块不完全一样。

    第二条是**有意接受的**：折回之后三组喂给 GEMM 的输入 dtype 完全一致，
    差别只剩量化格式本身 —— 这正是这一轮对照要隔离的那个变量。
    不折的话，FP8 从 fp32 量化而 FP4 只能从 bf16 量化，反倒多出一个变量。
    （代价是 v4-fp8 与 v2-fp8 在这一处不再逐位可比；本轮换了 WSD 日程，
    两者本来就不是同一个基线。）
    """
    w = getattr(lin, "weight", None)
    return x if w is None or x.dtype is w.dtype else x.to(w.dtype)


class SwiGLU(nn.Module):
    """通道混合。两个 Linear 占了全网络绝大部分 FLOPs。"""

    def __init__(self, cfg: ModelConfig, force_bf16: bool = False):
        super().__init__()
        d, h = cfg.dim, cfg.hidden
        self.up = make_linear(cfg, d, 2 * h, force_bf16=force_bf16)
        self.down = make_linear(cfg, h, d, force_bf16=force_bf16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(match_dtype(x, self.up)).chunk(2, dim=-1)
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
        qkv = self.qkv(match_dtype(x, self.qkv))
        qkv = qkv.view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(match_dtype(o, self.proj))


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
                # 首尾 block 对精度最敏感，FP8 时不走 FP8 GEMM
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

    def to_param_dtype(self) -> "CornerNet":
        """把参数降到 `cfg.param_dtype`（计算权重）。

        **必须在优化器取走 fp32 master 之后调用** —— 反过来的话 master 是从
        bf16 值回填的，等于一开始就丢掉一半精度，而且训练看不出任何异常。

        这里依赖 `nn.Module._apply` 的默认行为：它做的是 `param.data = param.data.to()`，
        **Parameter 对象本身不变**，所以优化器按对象持有的引用仍然有效。
        真要哪天不成立，表现是「训练不报错、权重永远不动」——
        `tests/test_optim.py` 里那条 `is` 断言就是守这个的。
        """
        return self.to(self.cfg.torch_param_dtype)

    def trunk(self, planes: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = planes.shape[0]
        x = self.stem(planes.to(memory_format=torch.channels_last))
        x = x.permute(0, 2, 3, 1).reshape(b, CELLS, self.cfg.dim)
        x = x + self.pos + self.scalar_mlp(scalars).unsqueeze(1)
        # 量化作用域只包住主干：stem/头部/损失都留在高精度
        with self._fp8_scope():
            for blk in self.blocks:
                x = blk(x)
        return self.norm_out(x)

    def _fp8_scope(self):
        if not self.cfg.quantized:
            return contextlib.nullcontext()
        from .fp8 import quant_autocast
        return quant_autocast(self.cfg.precision)

    def _device_scope(self):
        """低精度下把当前 CUDA 设备设成本模型所在的卡。

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
        if not self.cfg.quantized:
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
        # 参数是 bf16 而输入是 fp32 时，没有 autocast 的路径会直接报
        # `Input type (float) and bias type (c10::BFloat16) should be the same`。
        # 全仓 4 个推理点都包了 autocast，但它们的 `enabled=` 都挂着
        # `device.type == "cuda"` —— CPU 上 autocast 是关的，而 web 有真实的
        # CPU 回退路径，单测也直接在 CPU 上调 forward。护栏放在唯一入口，
        # 比指望每条调用路径都记得转 dtype 可靠。
        dt = self.pos.dtype
        if not torch.is_autocast_enabled() and planes.dtype != dt:
            planes, scalars = planes.to(dt), scalars.to(dt)

        b0 = planes.shape[0]
        if self.cfg.quantized:
            # MXFP8 与 NVFP4 都要求 GEMM 的两维是 32 的倍数，token 维是 B*196 且
            # 196%32==4，所以 B 必须是 8 的倍数。自博弈的批大小是变的，补齐再切回来。
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


def load_weights(model: "CornerNet", sd: dict) -> None:
    """装载权重，顺手剥掉 TE 的 `*._extra_state`。

    那些键是 FP8 的元数据（amax 历史等），既不是 parameter 也不是 buffer，
    下一次前向会自己重建；而老 checkpoint 里那份来自已经废弃的量化权重路径，
    装进来只会是误导。dtype 由 `load_state_dict` 自动转成参数的 dtype，
    所以老的 fp32 checkpoint 直接就落到 bf16 计算权重上。
    """
    clean = {k: v for k, v in sd.items() if not k.endswith("_extra_state")}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if unexpected:
        raise KeyError(f"checkpoint 里有模型上不存在的键: {list(unexpected)[:5]}")
    stray = [k for k in missing if not k.endswith("_extra_state")]
    if stray:
        raise KeyError(f"checkpoint 缺少这些键: {stray[:5]}")


def load_checkpoint(path: str, device="cuda") -> tuple["CornerNet", object]:
    """**读发布包**建模型做推理（`runs/<exp>/model/*.pt`）。

    名字沿用 `load_checkpoint` 是因为四个工具和 web 都在调它，但它现在
    **只吃发布包，不吃训练档** —— 拿到训练档会明确报错，不会将就着读。

    这条分家是有意的（速查表第六章）：训练档带着 AdamW 动量、RNG、步数簿记，
    那些东西只有续训用得上；推理只该拿到权重。通吃两种格式的加载器意味着
    「拿训练档当发布包用」这件事永远不会被发现，而两者在低精度下**不是**
    同一组数值 —— 发布包里是已经量化好的权重，训练档里是 fp32 master。

    统一走这里，别在各处自己 torch.load —— 量化模型必须在目标设备的上下文里
    构造（TE 按当前设备取 cuBLAS 句柄），否则会得到
    `cublas_gemm: failed to launch on the GPU` 这种离根因很远的错。
    """
    from .export import load_release
    return load_release(path, device)


def mask_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """把非法着法的 logit 压到 -inf。legal 是 0/1 或 bool 张量，形状 [B, 17836]。"""
    return logits.masked_fill(~legal.bool(), float("-inf"))
