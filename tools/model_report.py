#!/usr/bin/env python3
"""逐层拆解 BF16 / FP8 / FP4 三种配置的结构差异。

回答的问题是：**换成低精度之后，模型到底哪里变了、哪里没变。**
参数分类是从实例化后的模型上现场读的（看模块类型是不是 TE 的），
不是照着代码猜的。

    python3 tools/model_report.py
    python3 tools/model_report.py --dim 512 --blocks 16
"""

import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.model import (  # noqa: E402
    ACTIONS, BOARD, CELLS, ORI, PLANES, SCALARS, Attention, CornerNet,
    DepthwiseSpatial, ModelConfig, SwiGLU,
)

MX_BLOCK = 32

PRECISIONS = ("bf16", "fp8", "fp4")
# 表头与配方名。三条腿的存储完全一致，差别只在这一列。
RECIPE = {"bf16": "—", "fp8": "MXFP8", "fp4": "NVFP4"}


def is_quant_layer(mod) -> bool:
    """这个模块的 GEMM 走不走低精度。看类型而不是看权重 ——
    权重是普通 bf16 张量，量化只发生在计算里，从存储上分辨不出来。"""
    return type(mod).__module__.startswith("transformer_engine")


def storage_bytes(t) -> int:
    return t.numel() * t.element_size()


def analytic_flops(cfg: ModelConfig) -> dict:
    """每个局面（196 个 token）的前向 FLOPs，按「走不走低精度」分开算。

    只统计乘加两次的矩阵/卷积运算；Norm、激活、softmax 之类的逐元素算子
    FLOPs 可忽略（但它们是访存大户，这一点在性能文档里另说）。
    """
    d, h, n = cfg.dim, cfg.hidden, CELLS
    last = cfg.blocks - 1

    gemm_quant = gemm_bf16 = spatial = other = 0

    other += 2 * 9 * PLANES * d * n                     # stem 3x3 卷积
    other += 2 * (SCALARS * d + d * d)                  # 标量 MLP
    other += 2 * n * d * ORI                            # 策略头
    other += 2 * (2 * d * d + d * 3 + 2 * d * d + d)    # 价值头 + 辅助头

    for i in range(cfg.blocks):
        quant_here = cfg.quantized and not (cfg.fp8_first_last_bf16 and i in (0, last))

        spatial += 2 * (cfg.dw_kernel ** 2) * d * n      # 深度可分离卷积

        mlp = 2 * n * d * (2 * h) + 2 * n * h * d        # SwiGLU 的两个 GEMM
        if quant_here:
            gemm_quant += mlp
        else:
            gemm_bf16 += mlp

        if cfg.attn_every > 0 and (i + 1) % cfg.attn_every == 0:
            proj = 2 * n * d * (3 * d) + 2 * n * d * d   # QKV + 输出投影
            if quant_here:
                gemm_quant += proj
            else:
                gemm_bf16 += proj
            other += 2 * 2 * n * n * d                   # QK^T 与 AV，始终高精度

    total = gemm_quant + gemm_bf16 + spatial + other
    return {"quant_gemm": gemm_quant, "bf16_gemm": gemm_bf16,
            "spatial": spatial, "other": other, "total": total}


def build(cfg: ModelConfig, device: str):
    dev = torch.device(device)
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else _Null()
    with ctx:
        return CornerNet(cfg).to(dev)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def classify(model: CornerNet) -> list[tuple[str, str, int, int, bool]]:
    """(名字, 归属模块类型, 元素数, 字节数, GEMM 是否走低精度)"""
    kind_of, quant_of = {}, {}
    for name, mod in model.named_modules():
        for pn, _ in mod.named_parameters(recurse=False):
            full = f"{name}.{pn}" if name else pn
            kind_of[full] = type(mod).__name__
            quant_of[full] = is_quant_layer(mod)
    return [(n, kind_of.get(n, "?"), p.numel(), storage_bytes(p), quant_of.get(n, False))
            for n, p in model.named_parameters()]


def report(cfg: ModelConfig, device: str) -> dict:
    model = build(cfg, device)
    rows = classify(model)
    quant_rows = [r for r in rows if r[4]]
    plain = [r for r in rows if not r[4]]
    fl = analytic_flops(cfg)
    return {
        "cfg": cfg, "model": model, "rows": rows,
        "n_tensors": len(rows), "n_quant": len(quant_rows),
        "p_quant": sum(r[2] for r in quant_rows), "p_plain": sum(r[2] for r in plain),
        "bytes": sum(r[3] for r in rows),
        "flops": fl,
        "quant_kinds": sorted({r[1] for r in quant_rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base = dict(dim=args.dim, blocks=args.blocks, attn_every=args.attn_every)
    rep = {p: report(ModelConfig(precision=p, **base), args.device) for p in PRECISIONS}
    cols = "".join(f"{p.upper() + ' 配置':>16}" for p in PRECISIONS)

    print(f"配置 dim={args.dim} blocks={args.blocks} attn_every={args.attn_every}\n")

    def row(label, fn, width=22):
        print(f"{label:<{width}}" + "".join(fn(rep[p]) for p in PRECISIONS))

    print("== 参数与训练期显存 ==")
    print("（三条腿的存储完全相同 —— 量化只发生在 GEMM 里，不是存储格式）")
    print(f"{'':<22}{cols}")
    row("参数张量总数", lambda r: f"{r['n_tensors']:>16,}")
    row("其中 GEMM 走低精度", lambda r: f"{r['n_quant']:>16,}", 20)
    row("参数量（走低精度）", lambda r: f"{r['p_quant']:>16,}", 20)
    row("参数量（走高精度）", lambda r: f"{r['p_plain']:>16,}", 20)
    n_param = rep["bf16"]["p_quant"] + rep["bf16"]["p_plain"]
    row("参数总量", lambda r: f"{r['p_quant'] + r['p_plain']:>16,}", 24)
    row("计算权重 bf16 (MB)", lambda r: f"{r['bytes']/1e6:>16.1f}", 20)
    print(f"{'fp32 主权重 (MB)':<21}" + f"{n_param*4/1e6:>16.1f}" * 3)
    print(f"{'AdamW m+v fp32 (MB)':<20}" + f"{n_param*8/1e6:>16.1f}" * 3)
    print(f"{'合计字节/参数':<22}" + f"{'16':>16}" * 3)
    for p in PRECISIONS[1:]:
        print(f"GEMM 走 {p.upper()} 的模块类型: {rep[p]['quant_kinds'] or '（无）'}")

    print("\n== 每局面前向 FLOPs（196 token）==")
    print(f"{'':<22}{cols}")
    for label, key in [("走低精度的 GEMM", "quant_gemm"), ("走高精度的 GEMM", "bf16_gemm"),
                       ("深度可分离卷积", "spatial"), ("其余（stem/注意力/头）", "other"),
                       ("合计", "total")]:
        row(label, lambda r, k=key: f"{r['flops'][k]/1e9:>15.2f}G", 20)
    print(f"{'低精度覆盖的 FLOPs 占比':<16}"
          + "".join(f"{rep[p]['flops']['quant_gemm']/rep[p]['flops']['total']*100:>15.1f}%"
                    for p in PRECISIONS))

    print("\n== 结构约束 ==")
    print("（FP4 的微块是 16，但整除要求与 MXFP8 一样是 32，所以批大小约束相同）")
    print(f"{'批大小要求':<22}{'任意':>16}{'8 的倍数':>16}{'8 的倍数':>16}")
    print(f"{'量化配方':<24}" + "".join(f"{RECIPE[p]:>16}" for p in PRECISIONS))
    print(f"{'优化器':<24}" + f"{'master-AdamW':>16}" * 3)
    print(f"{'主权重 / 动量':<22}" + f"{'fp32 / fp32':>16}" * 3)
    print(f"{'torch.compile':<21}{'整模型':>16}{'按 block':>16}{'按 block':>16}")
    print(f"{'同进程多卡':<22}" + f"{'可以':>16}" * 3)

    print("\n== 低精度配置里逐层的精度归属（fp8 与 fp4 的划分完全相同）==")
    cfg_b = rep["fp8"]["cfg"]
    last = cfg_b.blocks - 1
    for i in range(cfg_b.blocks):
        forced = cfg_b.fp8_first_last_bf16 and i in (0, last)
        has_attn = cfg_b.attn_every > 0 and (i + 1) % cfg_b.attn_every == 0
        mods = "SwiGLU" + ("+Attention" if has_attn else "")
        why = "（首尾 block 强制高精度）" if forced else ""
        dest = "BF16" if forced else "MXFP8 / NVFP4"
        print(f"  block {i:>2}: {mods:<18} -> {dest} {why}")
    print("  始终高精度：stem 卷积、位置嵌入、标量 MLP、各深度可分离卷积、"
          "全部 RMSNorm、注意力的 softmax 与 QK^T/AV、策略头、价值头、辅助头")


if __name__ == "__main__":
    main()
