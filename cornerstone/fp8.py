"""低精度混合训练：把主干的大 GEMM 换成 MXFP8 或 NVFP4 计算。

**"FP8/FP4 训练"指的是 GEMM 的计算精度，不是权重的存储精度。** 周边脚手架一律
高精度：主权重是 fp32（由 `optim.MasterWeightAdamW` 持有）、优化器状态 fp32、
张量核内部的部分和累加 fp32、损失 fp32。参数本身是 bf16 —— 它是"计算权重"，
进 GEMM 前再由 TE 量化。三种精度的**参数存储完全一致**，差别只在 GEMM 里。

两种量化配方：

* **MXFP8**（`precision="fp8"`）：每 32 个连续元素共用一个 E8M0（2 的幂）缩放因子。
  取 HYBRID —— **前向 E4M3（精度优先），反向 E5M2（范围优先）**，
  因为梯度的动态范围比激活宽得多。
* **NVFP4**（`precision="fp4"`）：数据是 E2M1（4 bit），每 16 个元素一个 **E4M3**
  缩放因子，再叠一个整张量的 fp32 因子 —— 两层缩放是 FP4 能收敛的前提，
  单层 E8M0 在 2 位尾数上完全不够。TE 的默认参数还带着另外三件必需品：
  输入与梯度过 16×16 随机 Hadamard 变换（把离群值摊到块内）、梯度做随机舍入
  （2 位尾数下确定性舍入会系统性地偏置梯度）、权重按 16×16 二维块量化。
  **这些 `disable_*` 开关一个都不要动** —— 默认值就是速查表要求的配置。

  微块是 16，但**整除要求仍然是 32**（与 MXFP8 相同），所以 `pad_to_mxfp8` 原样通用。

历史注记：这个文件早先实现的是"真·FP8 主权重"（`quantized_model_init` 让权重张量
本身就是 MXFP8，优化器每步"反量化 → fp32 更新 → 随机舍入量化回去"）。那条路
违反"主权重与优化器更新始终 fp32"这条不变原则，已经移除；随机舍入随之一并移除
（master 保着全精度，舍入到最近就够了）。教训见 docs/06。
"""

from __future__ import annotations

import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

MX_BLOCK = 32          # MXFP8 的缩放块大小，硬件规定
NVFP4_BLOCK = 16       # NVFP4 的微块大小（但整除要求仍是 32，见 pad_to_mxfp8）

PRECISIONS = ("bf16", "fp8", "fp4")


def check_precision(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"未知精度 {precision!r}，只接受 {PRECISIONS}")
    return precision


def recipe_for(precision: str) -> recipe.Recipe:
    """按精度取量化配方。

    fp8：`MXFP8BlockScaling` 默认是纯 E4M3。梯度的动态范围比激活宽，E4M3 只有
    4 位指数，所以反向换成 E5M2（HYBRID）。

    fp4：`NVFP4BlockScaling` 的**默认参数**已经是速查表要求的配置
    （两层缩放 + 随机 Hadamard + 梯度随机舍入 + 权重二维块量化），
    不传任何 `disable_*`。它没有 `fp8_format` 这个旋钮 —— 格式由配方本身定死。
    """
    check_precision(precision)
    if precision == "fp8":
        return recipe.MXFP8BlockScaling(fp8_format=recipe.Format.HYBRID)
    if precision == "fp4":
        return recipe.NVFP4BlockScaling()
    raise ValueError("bf16 没有量化配方")


def is_available(precision: str) -> tuple[bool, str]:
    """当前 GPU 是否支持该精度。返回 (可用, 原因)，与 TE 新旧两种返回形态兼容。"""
    check_precision(precision)
    if precision == "bf16":
        return (True, "")
    fn = te.is_mxfp8_available if precision == "fp8" else te.is_nvfp4_available
    ok = fn()
    return (ok, "") if isinstance(ok, bool) else ok


def pad_to_mxfp8(batch: int) -> int:
    """把批大小补齐到量化 GEMM 能接受的值。**fp8 与 fp4 的约束相同。**

    两种配方都要求参与 GEMM 的**两个维度都是 32 的倍数**（NVFP4 的微块虽然是 16，
    整除要求仍按 32 算）。主干的 token 维是 `B*196`，而 196 mod 32 == 4，
    所以 `4*(B mod 8) == 0`，即 **B 必须是 8 的倍数**。
    自博弈里 prepare() 返回的批大小随局面数变化，送进量化前向之前必须补齐。
    """
    return ((batch + 7) // 8) * 8


# TE 2.17 把 autocast 改了名，旧名还能用但会告警。优先用新名，保持向后兼容。
_autocast = getattr(te, "autocast", None) or te.fp8_autocast


def quant_linear(precision: str, in_features: int, out_features: int, bias: bool = False,
                 params_dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """创建走量化 GEMM 的线性层。权重本身是 `params_dtype`（bf16），不是量化张量。

    fp8 与 fp4 用的是**同一个** `te.Linear` —— 精度由前向时生效的 autocast 配方决定，
    不由层的类型决定。所以 `precision` 在这里只用于校验；写成参数是为了让调用点
    读起来是"按精度建层"，与 `recipe_for` 对称。

    **刻意不用 `te.quantized_model_init`** —— 那个上下文会让权重张量直接以量化格式
    存储、没有高精度副本，正是本项目要避开的做法。这里走 TE 的标准路径：
    权重保持高精度，每次前向临时量化用于 GEMM。
    """
    check_precision(precision)
    if precision == "bf16":
        raise ValueError("bf16 不该走到量化层")
    return te.Linear(in_features, out_features, bias=bias, params_dtype=params_dtype)


def quant_autocast(precision: str):
    rcp = recipe_for(precision)
    try:
        return _autocast(enabled=True, recipe=rcp)
    except TypeError:                       # 旧签名用的是 fp8_recipe=
        return _autocast(enabled=True, fp8_recipe=rcp)


def _global_state():
    """TE 的量化全局状态管理器。优先取非弃用路径。"""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:                      # 老版本 TE
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    return FP8GlobalStateManager


def _recipe_matches(rcp, precision: str) -> bool:
    """配方是不是**这个**精度。

    用 TE 基类自带的谓词（`rcp.mxfp8()` / `rcp.nvfp4()`）而不是 isinstance ——
    谓词是 TE 的公开语义，isinstance 会在它哪天换实现类时静默变成恒假。
    """
    pred = getattr(rcp, {"fp8": "mxfp8", "fp4": "nvfp4"}[precision], None)
    return bool(pred()) if callable(pred) else False


def gemm_active(model: torch.nn.Module, precision: str, *inputs) -> bool | None:
    """跑一次前向，确认主干的 te.Linear **确实走了该精度的量化计算**。

    为什么需要这个：TE 在"该量化却没量化"时的表现是静默的 —— 模型照常训练，
    只是低精度名存实亡，而这恰好是 A/B 唯一要区分的那个变量。旧版探针靠捕获
    `quantized weights without quantized compute` 这条 UserWarning，但那条警告
    只在"权重是量化张量而计算没量化"时才发；改成标准配置后权重不再是量化张量，
    它**永远不会出现**，探针就退化成恒真了。所以这里换成直接查 TE 的状态。

    除了作用域与配方类型，还多查一条**更强的证据**：模块自己挂着的量化器里
    出现了对应精度的 `Quantizer`。配方类型只说明"打算怎么量化"，量化器是
    TE 真正建出来给这一层用的东西 —— fp4 若被降级成 fp8，配方类型这一关看不出来。

    返回 `None` 表示查不到（TE 换了内部 API）。**不能把查不到记成 False** ——
    那会让 metrics 里出现假的 `fp8_active: false`，比没有这个字段更误导。
    """
    check_precision(precision)
    if precision == "bf16":
        return False
    tgt = next((m for m in model.modules() if isinstance(m, te.Linear)), None)
    if tgt is None:
        return False                         # 配置说开了量化，却一个 TE 层都没有
    try:
        G = _global_state()
        is_on = getattr(G, "is_fp8_enabled", None) or G.is_quantized_enabled
    except (ImportError, AttributeError):
        return None

    want = "NVFP4Quantizer" if precision == "fp4" else "MXFP8Quantizer"
    seen: dict = {}

    def hook(mod, args, out):
        on = bool(is_on())
        # 模块的量化器列表。属性不存在（TE 换了实现）时**不判它** ——
        # 判成 False 会把"查不到"报成"掉了"。
        qs = getattr(mod, "quantizers", None)
        if isinstance(qs, dict):
            qs = [q for v in qs.values() for q in (v if isinstance(v, (list, tuple)) else [v])]
        names = {type(q).__name__ for q in qs} if qs else None
        seen.update(scope=on,
                    # 模块自己对"这次前向要不要量化"的结论：抓的是
                    # "作用域开着但这一层降级了"。属性不存在时退化成不判它。
                    module=getattr(mod, "fp8", True),
                    rcp=G.get_fp8_recipe() if on else None,
                    quantizer=(want in names) if names else None)

    h = tgt.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        h.remove()

    if not seen:
        return None                          # 钩子没被调到，说明这层不在前向路径上
    ok = bool(seen["scope"] and seen["module"] and _recipe_matches(seen["rcp"], precision))
    if ok and seen["quantizer"] is False:
        return False                         # 配方对得上，量化器却不是这个精度 —— 降级了
    return ok
