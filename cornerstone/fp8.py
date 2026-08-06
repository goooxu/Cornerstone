"""真·FP8 主权重（MXFP8）与随机舍入更新。

「模型主权重精度是 FP8」在这里是字面意思：权重张量本身就是 MXFP8
（E4M3 数据 + 每 32 个元素一个 E8M0 缩放），Blackwell 的张量核直接消费这个格式。
**没有高精度 master 副本** —— 优化器每一步都是「反量化 -> fp32 更新 -> 量化回去」。

为什么必须是随机舍入
    E4M3 只有 3 位尾数，相对分辨率约 2^-4。舍入到最近值时，凡是小于半个 ulp 的更新
    都会被系统性地丢弃 —— 学习率一降，几乎所有更新都归零，训练直接停住。
    随机舍入以正比于距离的概率向上/向下取整，期望值恰好是真值：
    偏差变成零均值噪声，而噪声可以靠大 batch 压，偏差压不掉。

一个反直觉的实测结论
    MXFP8 主权重**并不省显存**。TE 同时保留行布局与列布局两套数据
    （反向的两个 GEMM 需要不同的连续维，而块缩放的 FP8 没法便宜地转置），
    每个参数是 2*(1 + 1/32) ≈ 2.06 字节，比 BF16 的 2 字节还略多。
    FP8 在本项目里买的是吞吐，不是显存 —— 文档里不要写反。
"""

from __future__ import annotations

import contextlib

import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# E4M3：1 符号 + 4 指数 + 3 尾数
E4M3_MANTISSA_BITS = 3
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9
MX_BLOCK = 32          # MXFP8 的缩放块大小，硬件规定


def mxfp8_recipe() -> recipe.Recipe:
    return recipe.MXFP8BlockScaling()


def is_available() -> tuple[bool, str]:
    ok = te.is_mxfp8_available()
    return (ok, "") if isinstance(ok, bool) else ok


def pad_to_mxfp8(batch: int) -> int:
    """把批大小补齐到 MXFP8 能接受的值。

    MXFP8 要求参与 GEMM 的**两个维度都是 32 的倍数**。主干的 token 维是 `B*196`，
    而 196 mod 32 == 4，所以 `4*(B mod 8) == 0`，即 **B 必须是 8 的倍数**。
    自博弈里 prepare() 返回的批大小随局面数变化，送进 FP8 前向之前必须补齐。
    """
    return ((batch + 7) // 8) * 8


# TE 2.17 把这两个 API 改了名，旧名还能用但会告警。优先用新名，保持向后兼容。
_quantized_init = getattr(te, "quantized_model_init", None) or te.fp8_model_init
_autocast = getattr(te, "autocast", None) or te.fp8_autocast


def fp8_linear(in_features: int, out_features: int, bias: bool = False,
               params_dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """创建权重**直接以 MXFP8 存储**的线性层。

    这个上下文管理器是关键：不加它的话 TE 会保留高精度权重、每步临时量化，
    那就是普通的 FP8 混合精度，而不是 FP8 主权重了。
    """
    with _quantized_init(enabled=True, recipe=mxfp8_recipe()):
        return te.Linear(in_features, out_features, bias=bias, params_dtype=params_dtype)


def fp8_autocast():
    try:
        return _autocast(enabled=True, recipe=mxfp8_recipe())
    except TypeError:                       # 旧签名用的是 fp8_recipe=
        return _autocast(enabled=True, fp8_recipe=mxfp8_recipe())


def is_quantized(t: torch.Tensor) -> bool:
    return isinstance(t, te.MXFP8Tensor) or hasattr(t, "_rowwise_data")


# ---- 随机舍入 ----

def _block_amax(x: torch.Tensor, block: int = MX_BLOCK) -> torch.Tensor:
    """沿最后一维按 block 分块求 amax，返回逐元素展开后的结果。"""
    n = x.shape[-1]
    pad = (-n) % block
    flat = x.abs()
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(*flat.shape[:-1], -1, block)
    amax = blocks.amax(dim=-1, keepdim=True)
    return amax.expand_as(blocks).reshape(*flat.shape)[..., :n]


def quantization_ulp(x: torch.Tensor) -> torch.Tensor:
    """x 落到 MXFP8 网格上时的绝对量化间隔。

    因为块缩放是 2 的幂，绝对 ulp 与块缩放无关，等于 2^(floor(log2|x|) - 3)；
    只有掉进 e4m3 次正规区的元素才受块缩放影响，那里 ulp 固定为 scale * 2^-9。
    """
    amax = _block_amax(x)
    # TE 的块缩放取 2 的幂，使块内最大值刚好落在 e4m3 的表示上限附近
    scale = torch.exp2(torch.ceil(torch.log2((amax / E4M3_MAX).clamp_min(1e-38))))
    normal_ulp = torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(1e-38)))
                            - E4M3_MANTISSA_BITS)
    return torch.maximum(normal_ulp, scale * E4M3_MIN_SUBNORMAL)


def stochastic_round(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """在量化间隔尺度上叠加零均值抖动，使后续的「舍入到最近」在期望上无偏。

    这就是随机舍入的标准实现（dither-then-round）：对 [q, q+ulp) 区间内的值 v，
    加上 U(-ulp/2, ulp/2) 的噪声后取最近值，得到 q+ulp 的概率恰好是 (v-q)/ulp。
    """
    ulp = quantization_ulp(x)
    noise = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator) - 0.5
    return x + noise * ulp


def write_weight_(param: torch.Tensor, value: torch.Tensor,
                  stochastic: bool = True, generator: torch.Generator | None = None) -> None:
    """把高精度的 value 写回 FP8 主权重。

    **必须用 fp32 送进 quantize_，中间不能过一道 bf16。** 这里踩过一个很隐蔽的坑：
    param.dtype 是 bfloat16，早先版本写成 `param.quantize_(src.to(param.dtype))`，
    结果随机舍入完全失效。原因是 bf16 在 0.1 附近的 ulp 是 4.9e-4，
    而抖动后「该向上进位」的那条窗口只有 ~1e-4 宽；更糟的是 bf16 的格点
    0.10546875 恰好就是相邻两个 e4m3 格点的中点，整条窗口被塌到中点上，
    再按 round-half-to-even 倒回原值。表现是权重几乎不动，但完全看不出哪里错了。
    """
    if not is_quantized(param):
        param.data.copy_(value)
        return
    src = stochastic_round(value, generator) if stochastic else value
    param.quantize_(src.float())


def read_weight(param: torch.Tensor) -> torch.Tensor:
    return param.dequantize().float() if is_quantized(param) else param.data.float()


class Fp8AdamW(torch.optim.Optimizer):
    """直接在 FP8 主权重上更新的 AdamW。

    每步：反量化 -> fp32 下做完整的 AdamW（含权重衰减）-> 随机舍入量化回 FP8。
    动量 m/v 用 BF16 存 —— 它们不是「权重」，约束不适用，而且模型只有几十 M 参数，
    这点显存无所谓；用 BF16 只是顺手。

    非量化参数（Norm 的仿射、偏置、嵌入、卷积核）走普通 AdamW 路径。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 stochastic_rounding: bool = True, state_dtype: torch.dtype = torch.bfloat16):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    shape = p.shape
                    state["m"] = torch.zeros(shape, dtype=self.state_dtype, device=p.device)
                    state["v"] = torch.zeros(shape, dtype=self.state_dtype, device=p.device)

                state["step"] += 1
                t = state["step"]
                g = p.grad.float()
                w = read_weight(p)

                m = state["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                v = state["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                state["m"].copy_(m)
                state["v"].copy_(v)

                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                if wd:
                    w = w * (1 - lr * wd)          # 解耦权重衰减，在 fp32 下做
                w = w - lr * mh / (vh.sqrt() + eps)

                write_weight_(p, w, stochastic=self.stochastic_rounding)

        return loss


@contextlib.contextmanager
def maybe_fp8_autocast(enabled: bool):
    if enabled:
        with fp8_autocast():
            yield
    else:
        yield
