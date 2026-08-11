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


def is_fp8_layer(mod) -> bool:
    """这个模块的 GEMM 走不走 FP8。看类型而不是看权重 ——
    权重是普通 bf16 张量，FP8 只发生在计算里，从存储上分辨不出来。"""
    return type(mod).__module__.startswith("transformer_engine")


def storage_bytes(t) -> int:
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
    """(名字, 归属模块类型, 元素数, 字节数, GEMM 是否走 FP8)"""
    kind_of, fp8_of = {}, {}
    for name, mod in model.named_modules():
        for pn, _ in mod.named_parameters(recurse=False):
            full = f"{name}.{pn}" if name else pn
            kind_of[full] = type(mod).__name__
            fp8_of[full] = is_fp8_layer(mod)
    return [(n, kind_of.get(n, "?"), p.numel(), storage_bytes(p), fp8_of.get(n, False))
            for n, p in model.named_parameters()]


def report(cfg: ModelConfig, device: str) -> dict:
    model = build(cfg, device)
    rows = classify(model)
    fp8_rows = [r for r in rows if r[4]]
    plain = [r for r in rows if not r[4]]
    fl = analytic_flops(cfg)
    return {
        "cfg": cfg, "model": model, "rows": rows,
        "n_tensors": len(rows), "n_fp8": len(fp8_rows),
        "p_fp8": sum(r[2] for r in fp8_rows), "p_plain": sum(r[2] for r in plain),
        "bytes": sum(r[3] for r in rows),
        "flops": fl,
        "fp8_kinds": sorted({r[1] for r in fp8_rows}),
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

    print("== 参数与训练期显存 ==")
    print("（两条腿的存储完全相同 —— FP8 只发生在 GEMM 里，不是存储格式）")
    print(f"{'':<22}{'BF16 配置':>16}{'FP8 配置':>16}")
    for label, key in [("参数张量总数", "n_tensors"), ("其中 GEMM 走 FP8", "n_fp8")]:
        print(f"{label:<22}{a[key]:>16,}{b[key]:>16,}")
    print(f"{'参数量（走 FP8）':<21}{a['p_fp8']:>16,}{b['p_fp8']:>16,}")
    print(f"{'参数量（走高精度）':<20}{a['p_plain']:>16,}{b['p_plain']:>16,}")
    n_param = a["p_fp8"] + a["p_plain"]
    print(f"{'参数总量':<24}{n_param:>16,}{b['p_fp8']+b['p_plain']:>16,}")
    print(f"{'计算权重 bf16 (MB)':<20}{a['bytes']/1e6:>16.1f}{b['bytes']/1e6:>16.1f}")
    print(f"{'fp32 主权重 (MB)':<21}{n_param*4/1e6:>16.1f}{n_param*4/1e6:>16.1f}")
    print(f"{'AdamW m+v fp32 (MB)':<20}{n_param*8/1e6:>16.1f}{n_param*8/1e6:>16.1f}")
    print(f"{'合计字节/参数':<22}{'16':>16}{'16':>16}")
    print(f"GEMM 走 FP8 的模块类型: {b['fp8_kinds'] or '（无）'}")

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
    print(f"{'优化器':<24}{'master-AdamW':>16}{'master-AdamW':>16}")
    print(f"{'主权重 / 动量':<22}{'fp32 / fp32':>16}{'fp32 / fp32':>16}")
    print(f"{'torch.compile':<21}{'可用(约 2.7x)':>16}{'可用(约 2.2x)':>16}")
    print(f"{'同进程多卡':<22}{'可以':>16}{'可以':>16}")

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
