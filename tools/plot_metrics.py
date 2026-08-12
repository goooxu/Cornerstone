#!/usr/bin/env python3
"""把 metrics.jsonl 画成图。

    python3 tools/plot_metrics.py --kind loss      --exp v2-bf16
    python3 tools/plot_metrics.py --kind timeline  --exp v2-bf16

`loss` 画损失曲线，`timeline` 画一轮的时间线（各阶段按真实秒数等比例）。

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
    ap.add_argument("--kind", choices=["loss", "timeline"], default="loss")
    ap.add_argument("--exp", default="v2-bf16")
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

    if args.kind == "timeline":
        out = args.out or os.path.join(REPO, "reports", "图表", "一轮的时间线.png")
        fig_timeline(rows, plt, np, out, args.dpi)
        return

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
    # 最低点落在最后 5% 里 = 这条腿的损失一路降到底，没有回升可标 ——
    # 两个标注会叠在同一个点上糊成一团，标题也就说反了。
    span = float(step[-1] - step[0])
    tail = bool(step[lo] >= step[0] + 0.95 * span)
    a.annotate(f"minimum {total[lo]:.4f}\n@ step {int(step[lo]):,}"
               + ("  (the last iteration)" if tail else ""),
               xy=(step[lo], total[lo]),
               xytext=(step[lo] - 0.34 * span, total[lo] + 0.95),
               fontsize=9, color="#333",
               arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    if not tail:
        a.annotate(f"then +{rise:.0f}% to {total[-1]:.4f}",
                   xy=(step[-1], total[-1]),
                   xytext=(step[-1] - 0.25 * span, total[-1] + 1.15),
                   fontsize=9, color="#333",
                   arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    a.set_ylabel("loss")
    a.set_title("Total loss keeps falling all the way to the end"
                if tail else
                "Total loss bottoms out, then rises — while the learning rate is still decaying",
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
    c.set_title("The value head improves monotonically throughout"
                if tail else
                "Meanwhile the value head keeps improving — opposite direction to the total loss",
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


def fig_timeline(rows, plt, np, out: str, dpi: int) -> None:
    """一轮的时间线：各阶段按真实秒数等比例画。

    读出来的是「自博弈和训练各占一轮的多少」—— 两者严格交替、不重叠，
    所以这张图也就是「哪一半值得先去优化」。

    横轴尺度、每轮步数、要不要画评测那一条，**全部由数据推**。
    周期性评测已经从训练循环里移除（规则基线量不了这个网络，见
    `evaluate.evaluate_vs_baseline` 的注释），新的 metrics 里没有评测轮，
    此时自动退化成一根三段的条子。历史 metrics 仍然读得了，会多画一条评测轮。
    """
    import matplotlib.ticker

    wall = np.array([r["wall"] for r in rows], float)
    dt = np.diff(wall)
    is_ev = np.array(["eval_score_rate" in r for r in rows])[1:]
    sp = np.array([r["selfplay_games"] / r["selfplay_games_per_s"] for r in rows])[1:]
    tr = np.array([r.get("planned_steps", 400) / r["train_steps_per_s"] for r in rows])[1:]
    other = dt - sp - tr
    # 每轮的训练步数也不写死：第 1 轮被数据量限流过，取众数才是「一轮训多少步」
    steps = [int(r.get("planned_steps", 400)) for r in rows[1:]]
    steps_per_iter = max(set(steps), key=steps.count)

    SP, TR = (float(np.median(x)) for x in (sp, tr))
    # 没有评测轮时 other[is_ev] 是空数组，np.median 会给 nan 并告警
    OT = float(np.median(other[~is_ev])) if (~is_ev).any() else float(np.median(other))
    EV = float(np.median(other[is_ev])) if is_ev.any() else 0.0
    n_ev, n_pl = int(is_ev.sum()), int((~is_ev).sum())

    C = {"sp": "#2f7fd1", "tr": "#2f9e6f", "ev": "#c0392b", "ot": "#9aa5b1"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.4),
                           gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.62})

    # ── 上：一轮的时间线，两种轮次共用一条时间轴 ──────────────────
    a = ax[0]
    plain = [("sync + I/O", OT, C["ot"]), ("self-play", SP, C["sp"]),
             ("train %d steps" % steps_per_iter, TR, C["tr"])]
    bars = [("typical iteration\n(×%d)" % n_pl, 0, plain)]
    if n_ev:
        bars.insert(0, ("eval iteration\n(every 10th, ×%d)" % n_ev, 1,
                        plain + [("evaluation", EV, C["ev"])]))
    # 横轴尺度必须由数据定，不能写死：移除周期性评测后一轮从 1500s 降到 40s 量级，
    # 写死的 1700 会把整根条子压成一根看不见的线。
    span = SP + TR + OT + EV
    XMAX = 1.30 * span
    seen = {}
    for label, y, segs in bars:
        x = 0.0
        for name, w, col in segs:
            h = a.barh(y, w, left=x, height=0.46, color=col, edgecolor="white", lw=0.8)
            seen.setdefault(name, h)
            # 只有够宽的段才塞得下字；窄段靠图例认色，别画成截断的半个词
            if w / XMAX > 0.09:
                a.text(x + w / 2, y, f"{name}\n{w:.0f}s", ha="center", va="center",
                       fontsize=9, color="white", fontweight="bold")
            x += w
        a.text(-0.017 * XMAX, y, label, ha="right", va="center", fontsize=9)
        a.text(x + 0.01 * XMAX, y, f"{x:.0f}s", ha="left", va="center", fontsize=9.5,
               fontweight="bold", color="#333")
    # 窄段的数字放到条子下面，图上就不会有半截词
    a.text(SP + TR + OT + 0.01 * XMAX, -0.34,
           f"self-play {SP:.0f}s  ·  train {TR:.0f}s  ·  sync + I/O {OT:.0f}s",
           ha="left", va="center", fontsize=8.5, color="#666")
    keys = [k for k in ("self-play", "train %d steps" % steps_per_iter,
                        "evaluation", "sync + I/O") if k in seen]
    a.legend([seen[k] for k in keys], keys, loc="upper right", fontsize=8.5,
             ncol=len(keys), frameon=False, bbox_to_anchor=(1.0, 1.22))
    a.set_yticks([])
    a.set_ylim(-0.62, 1.42 if n_ev else 0.62)
    a.set_xlim(-0.20 * XMAX, XMAX)
    # 时间轴不该出现负刻度，左边那段是留给行标签的
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    a.set_xticks([t for t in a.get_xticks() if 0 <= t <= XMAX])
    a.set_xlabel("seconds (to scale)")
    a.set_title(
        (f"One iteration = self-play → train {steps_per_iter} steps. "
         f"Every 10th also evaluates — that eval alone costs {EV/(SP+TR+OT):.0f}× the rest")
        if n_ev else
        (f"One iteration = self-play → train {steps_per_iter} steps, "
         f"strictly alternating ({span:.0f}s median)"),
        fontsize=10.5, pad=10)
    a.spines[["left", "right", "top"]].set_visible(False)
    a.grid(axis="x", alpha=0.18)

    # ── 下：全程墙钟怎么分掉的 ───────────────────────────────────
    b = ax[1]
    tot = float(wall[-1] - wall[0])
    parts = [("self-play", float(sp.sum()), C["sp"]), ("training", float(tr.sum()), C["tr"])]
    if n_ev:
        parts.append(("evaluation", float(other[is_ev].sum()), C["ev"]))
    parts.append(("other", tot - sum(p[1] for p in parts), C["ot"]))
    x = 0.0
    for name, v, col in parts:
        b.barh(0, v, left=x, height=0.5, color=col, edgecolor="white", lw=0.8)
        if v / tot > 0.04:
            b.text(x + v / 2, 0, f"{name}\n{v/3600:.1f}h  ({100*v/tot:.0f}%)",
                   ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        x += v
    b.set_yticks([])
    b.set_xlim(0, tot)
    b.set_xlabel("total wall clock")
    b.set_title(f"Where the {tot/3600:.1f} hours went", fontsize=10.5, pad=8)
    # 刻度间隔跟着总时长走：4 小时的跑用 6h 一格只会画出一个「0h」
    hstep = next(h for h in (1, 2, 3, 6, 12, 24) if tot / 3600 / h <= 8)
    b.set_xticks([i * 3600 for i in range(0, int(tot / 3600) + 1, hstep)])
    b.set_xticklabels([f"{i}h" for i in range(0, int(tot / 3600) + 1, hstep)])
    b.spines[["left", "right", "top"]].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"已写入 {out}（普通轮 {SP+TR+OT:.0f}s，评测轮 {SP+TR+OT+EV:.0f}s）")


if __name__ == "__main__":
    main()
