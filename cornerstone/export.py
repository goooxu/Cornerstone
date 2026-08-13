"""训练档 → 发布包。

《低精度训练精度速查表》第六章：**只有权重（+ config）交给推理，其余全部丢弃**，
fp32 主权重是推理权重的唯一来源，低精度训练直接发布量化权重 + 分块缩放因子
（"继承训练量化方案"，推理引擎原样加载，量化误差与训练一致）。

于是本项目有两种产物，**各自只有一个消费方**：

| | 内容 | 谁读 |
|---|---|---|
| `runs/<exp>/ckpt/*.pt` | fp32 master + AdamW 动量/二阶矩 + RNG + 步数簿记 | **只有续训** |
| `runs/<exp>/model/*.pt` | 只有权重（低精度跑是量化权重 + 缩放因子） | 其余一切：评测、web、诊断 |

两边都显式检查、喂错就报错，见 `model.load_checkpoint` 与
`train.Trainer.load_checkpoint`。**不做双格式通吃的加载器** —— 通吃意味着
「拿训练档当发布包用」这件事永远不会被发现。

## 导出不改变数值，一分都不改

MXFP8 与 NVFP4 都是**块缩放**：缩放因子是权重张量自身的确定性函数
（每 32 / 每 16 个元素取块内最大值），不需要校准数据，也没有跨步累积的状态。
所以「预量化存盘」与「每次前向现算」给出逐位相同的结果 —— 实测确认，
`tests/test_export.py::test_exported_forward_is_bit_identical` 钉着这一条。

导出买到的是三件事，**都与精度无关**：一个真正的部署产物、体积
（fp4 包约 11 MB，训练档 162.5 MiB）、以及推理快 1.1~1.2×
（省掉每次前向重扫权重求块最大值）。

## 量化的输入是 bf16 计算权重，不是 fp32 master

链条是 `fp32 master → bf16 计算权重 → 量化`，中间那道 bf16 **不能省**：
训练与评测时 `te.Linear` 量化的输入就是 bf16 计算权重，直接拿 master 去量化
会得到另一组量化值，发布出去的模型就与阶梯上量到的那个不是同一个 ——
而"误差与训练一致"正是这条路线的全部价值。

## 只存 rowwise

量化张量同时带 rowwise 与 columnwise 两份数据，后者是**反向传播**用的。
推理不需要，而且留着它 **MXFP8 发布包会比 bf16 还大**（1.03×）。
丢掉之后 fp8 是 0.516×、fp4 是 0.281×，前向逐位不变。
"""

from __future__ import annotations

import os
from dataclasses import asdict

import torch

from .model import CornerNet, ModelConfig, load_weights

FORMAT = "cornerstone-model/1"

# TE 的 DType 枚举不能直接进 torch.save（pybind 对象），存名字再映射回去
_DTYPE_NAMES = ("kFloat8E4M3", "kFloat8E5M2", "kFloat4E2M1")


def is_release(blob: dict) -> bool:
    """这个 blob 是发布包还是训练档。训练档没有 `format` 键。"""
    return isinstance(blob, dict) and blob.get("format") == FORMAT


def _te():
    import transformer_engine.pytorch as te
    return te


def _quant_linears(model: CornerNet) -> list[tuple[str, torch.nn.Module]]:
    """要量化的层：主干里的 te.Linear。

    首尾 block 与所有非 GEMM 张量（stem、位置嵌入、标量 MLP、深度卷积、
    全部 RMSNorm、三个头）建的是 `nn.Linear`/`nn.Conv2d`，天然不在这个名单里 ——
    这正是速查表要的"部署时也保持混合精度 checkpoint"。判据用模块类型，
    不用名字：`make_linear` 改了规则这里自动跟上。
    """
    te = _te()
    return [(n, m) for n, m in model.named_modules() if isinstance(m, te.Linear)]


def _weight_quantizer(model: CornerNet, device: torch.device):
    """拿到 te.Linear **自己**在前向时用的那个权重量化器，改成只出 rowwise。

    两个坑（都实测过）：

    1. `lin.quantizers["scaling_fwd"]` 在第一次 `te.autocast` 前向之前是**空列表**，
       所以这里先跑一次前向把它建出来。
    2. 模块缓存的那个量化器是 `internal=True`，产出的是 `MXFP8TensorStorage`
       这种 **pybind 对象而不是 torch.Tensor 子类**，`nn.Parameter` 不收。
       必须 copy 一份再把 `internal` 关掉。

    不自己 `MXFP8Quantizer(...)` 现构造一个，是因为那样要手抄一串参数
    （E4M3 还是 E5M2、fp4 要不要二维块量化…），抄错了不会报错，
    只会得到一组"能跑但和训练不一样"的权重。从模块上取就不存在这个问题。
    """
    import copy
    te = _te()
    from . import _engine as E
    lin = next((m for _, m in _quant_linears(model)), None)
    if lin is None:
        raise RuntimeError("这个模型里一个 te.Linear 都没有，不该走到量化导出")

    # 触发一次前向，让 TE 把量化器建出来。批大小必须是 8 的倍数（见 fp8.pad_to_mxfp8）
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        model(torch.zeros(8, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=device),
              torch.zeros(8, E.NUM_SCALARS, device=device))

    fwd = lin.quantizers["scaling_fwd"]
    if not fwd:
        raise RuntimeError("前向之后量化器仍然是空的 —— TE 换实现了，导出逻辑要更新")
    q = copy.copy(fwd[1])              # 1 = GEMM 的权重那一路
    q.internal = False                 # 否则产出 *TensorStorage，不是 Tensor 子类
    q.set_usage(rowwise=True, columnwise=False)   # columnwise 是反向用的，推理不要
    return q


def _tensor_meta(qw) -> dict:
    """把一个量化权重拆成**普通张量 + 标量**，可以被 torch.save/safetensors 吃。

    刻意不存 TE 的张量子类：它的 state_dict 只有装了同版本 TE 的环境才读得出来，
    而发布包应该是自描述的。实测这套拆解/重建往返逐位相同。
    """
    out = {
        "data": qw._rowwise_data.detach().cpu(),
        "scale_inv": qw._rowwise_scale_inv.detach().cpu(),
        "shape": tuple(qw.shape),
    }
    # scale_inv 的形状被 TE 补齐到 (128, 4) 的倍数，**只能存不能算**
    amax = getattr(qw, "_amax_rowwise", None)
    if amax is not None:
        # NVFP4 的第二层缩放（整张量 fp32 因子）。它是承重的：
        # 传错 2 倍输出就 2 倍，传 None 直接报错
        out["amax"] = amax.detach().cpu()
    for attr in ("_fp8_dtype", "_fp4_dtype"):
        d = getattr(qw, attr, None)
        if d is None:
            continue
        # 用 `.name` 而不是 str()：后者给的是 `<DType.kFloat8E4M3: 7>`，
        # 带尖括号和枚举值，按后缀匹配会静默落空
        name = getattr(d, "name", None)
        if name not in _DTYPE_NAMES:
            raise ValueError(f"没见过的量化数据类型 {d!r}，导出逻辑要更新")
        out["dtype_name"] = name
    if "dtype_name" not in out:
        raise ValueError(f"{type(qw).__name__} 上找不到量化数据类型，导出逻辑要更新")
    return out


def build_release(ckpt_path: str, device: str = "cuda") -> dict:
    """读训练档，产出发布包的 blob（不落盘）。"""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if is_release(blob):
        raise ValueError(f"{ckpt_path} 已经是发布包了，不用再导出")
    if "model" not in blob:
        raise ValueError(f"{ckpt_path} 不像训练档：没有 model 字段")

    mc = blob.get("model_config") or {}
    cfg = ModelConfig(**{k: v for k, v in mc.items()
                         if k in ModelConfig.__dataclass_fields__})
    dev = torch.device(device)
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else _null()
    with ctx:
        model = CornerNet(cfg).to(dev)
        load_weights(model, blob["model"])
        # **这一步是关键**：降到 bf16 计算权重，也就是训练与评测时真正在用的那份。
        # 少了它就会拿 fp32 master 去量化，得到另一组值。
        model.to_param_dtype()
        model.eval()

        params = dict(model.named_parameters())
        quant: dict[str, dict] = {}
        if cfg.quantized:
            q = _weight_quantizer(model, dev)
            for name, lin in _quant_linears(model):
                key = f"{name}.weight"
                quant[key] = _tensor_meta(q(params[key].detach()))
                params.pop(key)
        bf16 = {k: v.detach().to(torch.bfloat16).cpu() for k, v in params.items()}

    import transformer_engine.pytorch as te
    return {
        "format": FORMAT,
        # 存储精度恒等于训练精度 —— 速查表修订版删掉了「先 cast 成 bf16 再发布」
        # 那一步，所以这里没有第二种选择，也就没有选错的可能
        "weight_precision": cfg.precision,
        "model_config": asdict(cfg),
        "step": blob.get("step", -1),
        "source": os.path.basename(ckpt_path),
        # 量化格式是绑在 TE 版本上的，换版本读不出来时至少知道该找谁
        "exported_by": {"te": getattr(te, "__version__", "?"),
                        "torch": torch.__version__},
        "bf16": bf16,
        "quant": quant,
    }


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def load_release(path: str, device: str = "cuda") -> tuple[CornerNet, object]:
    """读发布包，建一个**只能推理**的模型。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not is_release(blob):
        raise ValueError(
            f"{path} 是训练 checkpoint，不是发布包。\n"
            f"评测/试玩/诊断一律只读发布包（runs/<exp>/model/），"
            f"先用 tools/export_model.py 导出。")
    return build_model_from_release(blob, device)


def build_model_from_release(blob: dict, device: str = "cuda") -> tuple[CornerNet, object]:
    cfg = ModelConfig(**{k: v for k, v in blob["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    dev = torch.device(device)
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else _null()
    with ctx:
        model = CornerNet(cfg).to(dev)
        model.to_param_dtype()
        missing, unexpected = model.load_state_dict(blob["bf16"], strict=False)
        if unexpected:
            raise KeyError(f"发布包里有模型上不存在的键: {list(unexpected)[:5]}")
        want = {k for k in missing if not k.endswith("_extra_state")}
        if want != set(blob["quant"]):
            raise KeyError(
                f"发布包的量化权重对不上模型：缺 {sorted(want - set(blob['quant']))[:3]}，"
                f"多 {sorted(set(blob['quant']) - want)[:3]}")
        if blob["quant"]:
            _install_quantized(model, blob["quant"], dev)
    return model.eval(), blob.get("step", "?")


def _install_quantized(model: CornerNet, quant: dict, dev: torch.device) -> None:
    """把存下来的 uint8 数据 + 缩放因子重建成量化张量，装回 te.Linear。

    **不走 `te.quantized_model_init`**：那个上下文会让权重本身就是量化张量、
    没有 fp32 master —— 训练侧因此把它移除了（见 fp8.py 的历史注记）。
    那条反对意见只针对训练；但把它做成 `quant_linear` 的一个开关，迟早有人
    在训练侧打开。所以推理这条路自己重建，两边代码不共用。
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor

    mods = dict(model.named_modules())
    for key, rec in quant.items():
        mod = mods[key.rsplit(".", 1)[0]]
        dt = getattr(tex.DType, rec["dtype_name"])
        common = dict(shape=tuple(rec["shape"]), dtype=model.cfg.torch_param_dtype,
                      rowwise_data=rec["data"].to(dev),
                      rowwise_scale_inv=rec["scale_inv"].to(dev),
                      columnwise_data=None, columnwise_scale_inv=None,
                      # TE 在 GEMM 时自己 swizzle，存的是未 swizzle 的紧凑排布
                      with_gemm_swizzled_scales=False,
                      requires_grad=False, device=dev)
        if "amax" in rec:
            qz = NVFP4Quantizer(fp4_dtype=dt, rowwise=True, columnwise=False,
                                with_2d_quantization=True)
            t = NVFP4Tensor(fp4_dtype=dt, quantizer=qz,
                            amax_rowwise=rec["amax"].to(dev), amax_columnwise=None,
                            **common)
        else:
            qz = MXFP8Quantizer(fp8_dtype=dt, rowwise=True, columnwise=False)
            t = MXFP8Tensor(fp8_dtype=dt, quantizer=qz, **common)
        mod.weight = torch.nn.Parameter(t, requires_grad=False)


def write_release(ckpt_path: str, out_path: str, device: str = "cuda") -> int:
    """导出一份，返回字节数。原子替换，半截文件不会被当成有效发布包。"""
    blob = build_release(ckpt_path, device)
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(blob, tmp)
    os.replace(tmp, out_path)
    return os.path.getsize(out_path)
