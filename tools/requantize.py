#!/usr/bin/env python3
"""把一个 **BF16 发布包**重量化成 FP8 / FP4 的计算权重，产出新的发布包。

    python3 tools/requantize.py ../runs/_ruler/v4-bf16-111k.pt --precision fp4 \
        --out ../runs/_arena/ptq/bf16-ptq-fp4.pt

量的是**训练后量化（PTQ）**：拿 BF16 训出来的权重直接压成低精度去推理，
与「低精度训练（QAT）」对照 —— 后者训练全程的 GEMM 就在低精度上。
两者的推理开销完全相同，差别只在权重是怎么来的。

**为什么从发布包重量化等价于从 fp32 主权重走一遍**：
`export.write_release` 在量化之前先做 `model.to_param_dtype()`，也就是
fp32 master -> bf16 计算权重 -> 量化。BF16 发布包里存的正是那份 bf16 计算权重，
所以这里少走的只是一次「已经发生过」的 fp32->bf16 转换，数值上没有差别。
（这一点由 `--verify` 顺带钉住：重量化包的 bf16 部分必须与源包逐位相同。）

**量化器一律从模型上取，不自己构造** —— 理由见 `export._weight_quantizer`：
手抄一串参数（E4M3 还是 E5M2、要不要二维块量化…）抄错了不报错，
只会得到一组「能跑但和训练时不一样」的权重。
"""

import argparse
import os
import sys
from dataclasses import asdict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.export import (                       # noqa: E402
    FORMAT, _quant_linears, _tensor_meta, _weight_quantizer, is_release,
)
from cornerstone.model import CornerNet, ModelConfig, load_weights   # noqa: E402


def requantize(src_path: str, precision: str, out_path: str, device: str) -> dict:
    blob = torch.load(src_path, map_location="cpu", weights_only=False)
    if not is_release(blob):
        raise SystemExit(f"{src_path} 不是发布包")
    if blob.get("weight_precision") != "bf16":
        raise SystemExit(
            f"源包的精度是 {blob.get('weight_precision')}，不是 bf16。\n"
            f"PTQ 的定义就是「从 BF16 训练的权重出发」，从别的精度重量化没有意义。")
    if blob.get("quant"):
        raise SystemExit("源包里已经有量化权重 —— 这不该出现在 bf16 包里")

    cfg = ModelConfig(**{k: v for k, v in blob["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    cfg.precision = precision           # 唯一的改动
    cfg.fp8 = None                      # 老包可能带这个别名，别让它把 precision 顶回去
    cfg.__post_init__()

    dev = torch.device(device)
    with torch.cuda.device(dev) if dev.type == "cuda" else _null():
        model = CornerNet(cfg).to(dev)
        # 源包存的就是 bf16 计算权重，直接装载；量化在下面一步做
        load_weights(model, blob["bf16"])
        model.to_param_dtype().eval()

        params = dict(model.named_parameters())
        q = _weight_quantizer(model, dev)
        quant = {}
        for name, _ in _quant_linears(model):
            key = f"{name}.weight"
            quant[key] = _tensor_meta(q(params[key].detach()))
            params.pop(key)
        bf16 = {k: v.detach().to(torch.bfloat16).cpu() for k, v in params.items()}

    import transformer_engine.pytorch as te
    out = {
        "format": FORMAT,
        "weight_precision": precision,
        "model_config": asdict(cfg),
        "step": blob.get("step", -1),
        "source": os.path.basename(src_path),
        "exported_by": {"te": getattr(te, "__version__", "?"),
                        "torch": torch.__version__},
        "bf16": bf16,
        "quant": quant,
        # 留个记号：这是 PTQ 产物，不是低精度训练出来的。
        # 两者的 model_config 完全一样，没有这一条就分不出来。
        "requantized_from": os.path.basename(src_path),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(out, out_path)
    return out


def verify(src_path: str, out_path: str, device: str) -> None:
    """两条：未量化部分必须与源包逐位相同；重量化包能加载且前向有限。"""
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    out = torch.load(out_path, map_location="cpu", weights_only=False)
    for k, v in out["bf16"].items():
        if not torch.equal(v, src["bf16"][k].to(v.dtype)):
            raise SystemExit(f"未量化的 {k} 与源包不一致 —— 重量化动了不该动的东西")

    from cornerstone import _engine as E
    from cornerstone.model import load_checkpoint
    m, _ = load_checkpoint(out_path, device)
    with torch.no_grad(), torch.autocast(
            device, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        pol, wdl, sc = m(torch.zeros(8, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=device),
                         torch.zeros(8, E.NUM_SCALARS, device=device))
    for t, n in ((pol, "policy"), (wdl, "wdl"), (sc, "score")):
        if not torch.isfinite(t).all():
            raise SystemExit(f"{n} 输出里有非有限值")


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="BF16 发布包")
    ap.add_argument("--precision", required=True, choices=["fp8", "fp4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    requantize(a.src, a.precision, a.out, a.device)
    if not a.no_verify:
        verify(a.src, a.out, a.device)
    mb = os.path.getsize(a.out) / 2**20
    src_mb = os.path.getsize(a.src) / 2**20
    print(f"{a.src}  {src_mb:.1f} MB  --{a.precision}-->  {a.out}  {mb:.1f} MB  验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
