#!/usr/bin/env python3
"""逐层拆解 BF16 与 FP8 两种配置的结构差异。

回答的问题是：**打开 fp8 之后，模型到底哪里变了、哪里没变。**
参数分类是从实例化后的模型上现场读的（看权重是不是 MXFP8 量化张量），
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


def is_quant(t) -> bool:
    return hasattr(t, "_rowwise_data")


def storage_bytes(t) -> int:
    """参数实际占多少字节。MXFP8 要算上行/列两套数据与两套块缩放。"""
    if is_quant(t):
        return sum(x.numel() for x in (t._rowwise_data, t._columnwise_data,
                                       t._rowwise_scale_inv, t._columnwise_scale_inv))
    return t.numel() * t.element_size()


def analytic_flops(cfg: ModelConfig) -> dict:
    """每个局面（196 个 token）的前向 FLOPs，按「走不走 FP8」分开算。

    只统计乘加两次的矩阵/卷积运算；Norm、激活、softmax 之类的逐元素算子
    FLOPs 可忽略（但它们是访存大户，这一点在性能文档里另说）。
    """
    d, h, n = cfg.dim, cfg.hidden, CELLS
    last = cfg.blocks - 1

    gemm_fp8 = gemm_bf16 = spatial = other = 0

    other += 2 * 9 * PLANES * d * n                     # stem 3x3 卷积
    other += 2 * (SCALARS * d + d * d)                  # 标量 MLP
    other += 2 * n * d * ORI                            # 策略头
    other += 2 * (2 * d * d + d * 3 + 2 * d * d + d)    # 价值头 + 辅助头

    for i in range(cfg.blocks):
        fp8_here = cfg.fp8 and not (cfg.fp8_first_last_bf16 and i in (0, last))
        bucket = "fp8" if fp8_here else "bf16"

        spatial += 2 * (cfg.dw_kernel ** 2) * d * n      # 深度可分离卷积

        mlp = 2 * n * d * (2 * h) + 2 * n * h * d        # SwiGLU 的两个 GEMM
        if bucket == "fp8":
            gemm_fp8 += mlp
        else:
            gemm_bf16 += mlp

        if cfg.attn_every > 0 and (i + 1) % cfg.attn_every == 0:
            proj = 2 * n * d * (3 * d) + 2 * n * d * d   # QKV + 输出投影
            if bucket == "fp8":
                gemm_fp8 += proj
            else:
                gemm_bf16 += proj
            other += 2 * 2 * n * n * d                   # QK^T 与 AV，始终高精度

    total = gemm_fp8 + gemm_bf16 + spatial + other
    return {"fp8_gemm": gemm_fp8, "bf16_gemm": gemm_bf16,
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
    """(名字, 归属模块类型, 元素数, 字节数, 是否量化)"""
    kind_of = {}
    for name, mod in model.named_modules():
        for pn, _ in mod.named_parameters(recurse=False):
            kind_of[f"{name}.{pn}" if name else pn] = type(mod).__name__
    return [(n, kind_of.get(n, "?"), p.numel(), storage_bytes(p), is_quant(p))
            for n, p in model.named_parameters()]


def report(cfg: ModelConfig, device: str) -> dict:
    model = build(cfg, device)
    rows = classify(model)
    quant = [r for r in rows if r[4]]
    plain = [r for r in rows if not r[4]]
    fl = analytic_flops(cfg)
    return {
        "cfg": cfg, "model": model, "rows": rows,
        "n_tensors": len(rows), "n_quant": len(quant),
        "p_quant": sum(r[2] for r in quant), "p_plain": sum(r[2] for r in plain),
        "b_quant": sum(r[3] for r in quant), "b_plain": sum(r[3] for r in plain),
        "flops": fl,
        "quant_kinds": sorted({r[1] for r in quant}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base = dict(dim=args.dim, blocks=args.blocks, attn_every=args.attn_every)
    a = report(ModelConfig(fp8=False, **base), args.device)
    b = report(ModelConfig(fp8=True, **base), args.device)

    print(f"配置 dim={args.dim} blocks={args.blocks} attn_every={args.attn_every}\n")

    print("== 参数张量 ==")
    print(f"{'':<22}{'BF16 配置':>16}{'FP8 配置':>16}")
    for label, key in [("参数张量总数", "n_tensors"), ("其中 MXFP8 存储", "n_quant")]:
        print(f"{label:<22}{a[key]:>16,}{b[key]:>16,}")
    print(f"{'参数量（量化部分）':<20}{a['p_quant']:>16,}{b['p_quant']:>16,}")
    print(f"{'参数量（高精度部分）':<19}{a['p_plain']:>16,}{b['p_plain']:>16,}")
    print(f"{'参数总量':<24}{a['p_quant']+a['p_plain']:>16,}{b['p_quant']+b['p_plain']:>16,}")
    print(f"{'权重占用 (MB)':<22}"
          f"{(a['b_quant']+a['b_plain'])/1e6:>16.1f}{(b['b_quant']+b['b_plain'])/1e6:>16.1f}")
    if b["p_quant"]:
        print(f"{'  量化部分字节/参数':<19}{'—':>16}"
              f"{b['b_quant']/b['p_quant']:>16.3f}")
    print(f"被量化的模块类型: {b['quant_kinds'] or '（无）'}")

    print("\n== 每局面前向 FLOPs（196 token）==")
    fa, fb = a["flops"], b["flops"]
    print(f"{'':<22}{'BF16 配置':>16}{'FP8 配置':>16}")
    for label, key in [("走 FP8 的 GEMM", "fp8_gemm"), ("走高精度的 GEMM", "bf16_gemm"),
                       ("深度可分离卷积", "spatial"), ("其余（stem/注意力/头）", "other"),
                       ("合计", "total")]:
        print(f"{label:<20}{fa[key]/1e9:>15.2f}G{fb[key]/1e9:>15.2f}G")
    print(f"{'FP8 覆盖的 FLOPs 占比':<18}{'0.0%':>16}"
          f"{fb['fp8_gemm']/fb['total']*100:>15.1f}%")

    print("\n== 结构约束 ==")
    print(f"{'批大小要求':<22}{'任意':>16}{'8 的倍数':>16}")
    print(f"{'优化器':<24}{'AdamW':>16}{'Fp8AdamW(随机舍入)':>16}")
    print(f"{'torch.compile':<21}{'可用(约 1.8x)':>16}{'不可用':>16}")
    print(f"{'同进程多卡':<22}{'可以':>16}{'不行':>16}")

    print("\n== FP8 配置里逐层的精度归属 ==")
    cfg_b = b["cfg"]
    last = cfg_b.blocks - 1
    for i in range(cfg_b.blocks):
        forced = cfg_b.fp8_first_last_bf16 and i in (0, last)
        has_attn = cfg_b.attn_every > 0 and (i + 1) % cfg_b.attn_every == 0
        mods = "SwiGLU" + ("+Attention" if has_attn else "")
        why = "（首尾 block 强制高精度）" if forced else ""
        print(f"  block {i:>2}: {mods:<18} -> {'BF16' if forced else 'MXFP8'} {why}")
    print("  始终高精度：stem 卷积、位置嵌入、标量 MLP、各深度可分离卷积、"
          "全部 RMSNorm、注意力的 softmax 与 QK^T/AV、策略头、价值头、辅助头")


if __name__ == "__main__":
    main()
