#!/usr/bin/env python3
"""把 metrics.jsonl 画成损失曲线。

    python3 tools/plot_metrics.py --exp ab-bf16 --out reports/图表/损失曲线.png

三块面板对应报告里那段结论：

    上   总损失在某一步触底后回升，而学习率还在往下退 —— 不是「没退火完」
    中   policy 与 policy_entropy 几乎完全重合，说明策略损失基本等于**目标本身的熵**，
         所以总损失的回升反映的是搜索目标变了，不是网络学坏了
    下   同一段时间里价值损失还在降、WDL 准确率还在升 —— 和总损失方向相反

**标注一律用英文**：容器里没有任何中文字体（`fc-list :lang=zh` 为空），
写中文只会渲染成一排豆腐块。图的中文说明放在报告正文里。
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rolling(y, k: int):
    """居中滑动平均，两端用逐渐变短的窗口，长度不变。"""
    import numpy as np
    y = np.asarray(y, dtype=float)
    if k <= 1 or len(y) < 3:
        return y
    out = np.empty_like(y)
    half = k // 2
    for i in range(len(y)):
        out[i] = y[max(0, i - half):min(len(y), i + half + 1)].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ab-bf16")
    ap.add_argument("--metrics", default=None, help="直接给 metrics.jsonl 路径")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=int, default=5, help="滑动平均窗口（0 = 不平滑）")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    src = args.metrics or os.path.join(REPO, "..", "runs", args.exp, "logs", "metrics.jsonl")
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if "loss" in r]          # 攒数据那几轮没有训练项
    if not rows:
        raise SystemExit(f"{src} 里没有带 loss 的记录")

    step = np.array([r["step"] for r in rows], dtype=float)
    get = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)  # noqa: E731
    total, pol, val = get("loss"), get("policy"), get("value")
    ent, acc, lr = get("policy_entropy"), get("wdl_acc"), get("lr")
    sm = lambda y: rolling(y, args.smooth)                                    # noqa: E731

    lo = int(np.nanargmin(total))
    C = {"total": "#1b3a5c", "pol": "#2f7fd1", "ent": "#e8892b",
         "val": "#2f9e6f", "acc": "#c0392b", "lr": "#9aa5b1"}

    fig, ax = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True,
                           gridspec_kw={"hspace": 0.16})

    # ── 上：总损失与策略项，叠学习率 ──────────────────────────────
    a = ax[0]
    a.plot(step, total, color=C["total"], alpha=0.22, lw=0.8)
    a.plot(step, sm(total), color=C["total"], lw=2.0, label="total loss")
    a.plot(step, sm(pol), color=C["pol"], lw=1.6, label="policy loss")
    a.plot(step, sm(val), color=C["val"], lw=1.4, label="value loss")
    a.axvline(step[lo], color="#888", ls="--", lw=1.0)
    rise = 100 * (total[-1] - total[lo]) / total[lo]
    a.annotate(f"minimum {total[lo]:.4f}\n@ step {int(step[lo]):,}",
               xy=(step[lo], total[lo]), xytext=(step[lo] - 62000, total[lo] + 0.95),
               fontsize=9, color="#333",
               arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    a.annotate(f"then +{rise:.0f}% to {total[-1]:.4f}",
               xy=(step[-1], total[-1]), xytext=(step[-1] - 46000, total[-1] + 1.15),
               fontsize=9, color="#333",
               arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    a.set_ylabel("loss")
    a.set_title("Total loss bottoms out, then rises — while the learning rate is still decaying",
                fontsize=10.5, pad=8)
    a.legend(loc="upper right", fontsize=9, framealpha=0.9)
    a.grid(alpha=0.18)
    a2 = a.twinx()
    a2.plot(step, lr, color=C["lr"], lw=1.0, ls=":", label="lr")
    a2.set_ylabel("learning rate", color=C["lr"], fontsize=9)
    a2.tick_params(axis="y", labelcolor=C["lr"], labelsize=8)

    # ── 中：policy 与目标熵几乎重合 ───────────────────────────────
    b = ax[1]
    b.plot(step, sm(pol), color=C["pol"], lw=2.4, label="policy loss")
    b.plot(step, sm(ent), color=C["ent"], lw=1.2, ls="--", label="policy target entropy")
    b.set_ylabel("nats")
    b.set_title("Policy loss ≈ entropy of the search target itself: the network already fits it",
                fontsize=10.5, pad=8)
    b.legend(loc="upper right", fontsize=9, framealpha=0.9)
    b.grid(alpha=0.18)
    b2 = b.twinx()
    d = pol - ent
    b2.plot(step, d, color="#b04a8a", lw=0.6, alpha=0.20)      # 原始值，只作底噪
    b2.plot(step, sm(d), color="#b04a8a", lw=1.5, alpha=0.95)
    b2.set_ylabel("policy − entropy (log)", color="#b04a8a", fontsize=9)
    b2.tick_params(axis="y", labelcolor="#b04a8a", labelsize=8)
    # 对数轴：早期尖峰有 1e-2，末段只有 1e-5，线性轴会把后者压成一条贴底的线
    b2.set_yscale("log")
    b2.set_ylim(1e-5, 3e-2)
    late = d[step > 20000]
    b2.text(0.028, 0.86,
            f"gap: median {np.median(d):.1e}   ≤ {late.max():.1e} after step 20k"
            f"   (loss itself is ~1.4)",
            transform=b.transAxes, fontsize=8.5, color="#b04a8a",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d8c0d0", alpha=0.92))

    # ── 下：价值项反着走 ─────────────────────────────────────────
    c = ax[2]
    c.plot(step, sm(val), color=C["val"], lw=2.0, label="value loss (left)")
    c.axvline(step[lo], color="#888", ls="--", lw=1.0)
    c.set_ylabel("value loss", color=C["val"])
    c.tick_params(axis="y", labelcolor=C["val"])
    c.set_xlabel("training step")
    c.set_title("Meanwhile the value head keeps improving — opposite direction to the total loss",
                fontsize=10.5, pad=8)
    c.grid(alpha=0.18)
    c2 = c.twinx()
    c2.plot(step, sm(acc), color=C["acc"], lw=1.8, label="WDL accuracy (right)")
    c2.set_ylabel("WDL accuracy", color=C["acc"])
    c2.tick_params(axis="y", labelcolor=C["acc"])
    h = c.get_legend_handles_labels()[0] + c2.get_legend_handles_labels()[0]
    c.legend(h, [x.get_label() for x in h], loc="center right", fontsize=9, framealpha=0.9)

    for a_ in ax:
        a_.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))

    out = args.out or os.path.join(REPO, "reports", "图表", "损失曲线.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"已写入 {out}（{len(rows)} 个数据点，最低点 step {int(step[lo]):,}）")


if __name__ == "__main__":
    main()
