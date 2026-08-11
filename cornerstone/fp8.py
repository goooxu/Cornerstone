"""FP8 混合精度：把主干的大 GEMM 换成 MXFP8 计算。

**"FP8 训练"指的是 GEMM 的计算精度，不是权重的存储精度。** 周边脚手架一律高精度：
主权重是 fp32（由 `optim.MasterWeightAdamW` 持有）、优化器状态 fp32、
张量核内部的部分和累加 fp32、损失 fp32。参数本身是 bf16 —— 它是"计算权重"，
进 GEMM 前再由 TE 量化成 MXFP8。

缩放用 MXFP8 块缩放：每 32 个连续元素共用一个 E8M0（2 的幂）缩放因子。
数据格式取 HYBRID —— **前向 E4M3（精度优先），反向 E5M2（范围优先）**，
因为梯度的动态范围比激活宽得多。

历史注记：这个文件早先实现的是"真·FP8 主权重"（`quantized_model_init` 让权重张量
本身就是 MXFP8，优化器每步"反量化 → fp32 更新 → 随机舍入量化回去"）。那条路
违反"主权重与优化器更新始终 fp32"这条不变原则，已经移除；随机舍入随之一并移除
（master 保着全精度，舍入到最近就够了）。教训见 docs/06。
"""

from __future__ import annotations

import contextlib

import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

MX_BLOCK = 32          # MXFP8 的缩放块大小，硬件规定


def mxfp8_recipe() -> recipe.Recipe:
    """前向 E4M3 / 反向 E5M2。

    `MXFP8BlockScaling` 默认是纯 E4M3。梯度的动态范围比激活宽，E4M3 只有 4 位指数，
    所以反向换成 E5M2 —— 这也是 TE 对 delayed scaling 的默认取法（HYBRID）。
    """
    return recipe.MXFP8BlockScaling(fp8_format=recipe.Format.HYBRID)


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


# TE 2.17 把 autocast 改了名，旧名还能用但会告警。优先用新名，保持向后兼容。
_autocast = getattr(te, "autocast", None) or te.fp8_autocast


def fp8_linear(in_features: int, out_features: int, bias: bool = False,
               params_dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """创建走 FP8 GEMM 的线性层。权重本身是 `params_dtype`（bf16），不是量化张量。

    **刻意不用 `te.quantized_model_init`** —— 那个上下文会让权重张量直接以 MXFP8 存储、
    没有高精度副本，正是本项目要避开的做法。这里走 TE 的标准路径：权重保持高精度，
    每次前向临时量化用于 GEMM。
    """
    return te.Linear(in_features, out_features, bias=bias, params_dtype=params_dtype)


def fp8_autocast():
    try:
        return _autocast(enabled=True, recipe=mxfp8_recipe())
    except TypeError:                       # 旧签名用的是 fp8_recipe=
        return _autocast(enabled=True, fp8_recipe=mxfp8_recipe())


def device_scope(t: torch.Tensor):
    """把当前 CUDA 设备设成张量所在的卡。

    TE 的 GEMM 按**当前设备**取 cuBLAS 句柄，不匹配时要么静默退回非量化计算，
    要么报 `Expected all tensors to be on the same device` / `illegal memory access`。
    这个坑在模型前向、自博弈推理两条路径上各踩过一次，所以护栏下沉到
    `CornerNet.forward`，而不是指望每个调用点都记得包一层。
    """
    dev = t.device
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


def _global_state():
    """TE 的 FP8 全局状态管理器。优先取非弃用路径。"""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:                      # 老版本 TE
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    return FP8GlobalStateManager


def fp8_gemm_active(model: torch.nn.Module, *inputs) -> bool | None:
    """跑一次前向，确认主干的 te.Linear **确实走了 MXFP8 计算**。

    为什么需要这个：TE 在"该量化却没量化"时的表现是静默的 —— 模型照常训练，
    只是 FP8 名存实亡，而这恰好是 A/B 唯一要区分的那个变量。旧版探针靠捕获
    `quantized weights without quantized compute` 这条 UserWarning，但那条警告
    只在"权重是量化张量而计算没量化"时才发；改成标准配置后权重不再是量化张量，
    它**永远不会出现**，探针就退化成恒真了。所以这里换成直接查 TE 的状态。

    返回 `None` 表示查不到（TE 换了内部 API）。**不能把查不到记成 False** ——
    那会让 metrics 里出现假的 `fp8_active: false`，比没有这个字段更误导。
    """
    tgt = next((m for m in model.modules() if isinstance(m, te.Linear)), None)
    if tgt is None:
        return False                         # 配置说开了 FP8，却一个 TE 层都没有
    try:
        G = _global_state()
        is_on = getattr(G, "is_fp8_enabled", None) or G.is_quantized_enabled
    except (ImportError, AttributeError):
        return None

    seen: dict = {}

    def hook(mod, args, out):
        on = bool(is_on())
        seen.update(scope=on,
                    # 模块自己对"这次前向要不要量化"的结论：抓的是
                    # "作用域开着但这一层降级了"。属性不存在时退化成不判它。
                    module=getattr(mod, "fp8", True),
                    rcp=G.get_fp8_recipe() if on else None)

    h = tgt.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        h.remove()

    if not seen:
        return None                          # 钩子没被调到，说明这层不在前向路径上
    return bool(seen["scope"] and seen["module"]
                and isinstance(seen["rcp"], recipe.MXFP8BlockScaling))
