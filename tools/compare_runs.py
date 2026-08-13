#!/usr/bin/env python3
"""对比两条（或多条）训练曲线。

主要用途是 FP8 vs BF16 的对照实验 —— 没有这条对比，
「FP8 无损」就是没有依据的说法。

注意 loss 跨实验不可比（策略目标是网络自己的搜索结果，网络变强目标就变），
这里的曲线只用来看「训练有没有跑飞」，棋力结论一律以
`tools/net_arena.py` 的头对头为准。

    python3 tools/compare_runs.py bf16 fp8
    python3 tools/compare_runs.py bf16 fp8 --metric policy value policy_entropy_model
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(REPO), "runs")

DEFAULT_METRICS = ["loss", "policy", "value", "score", "policy_entropy_model",
                   "wdl_acc"]
# 只在历史 metrics 里存在：训练期的周期性评测已经移除（规则基线量不了这个网络，
# 见 docs/08）。留着是为了还能读 ab-* 那两条跑的日志；缺字段时自动跳过。
LEGACY_METRICS = ["eval_score_rate", "eval_elo_abs"]


def load(exp: str) -> list[dict]:
    path = os.path.join(RUNS, exp, "logs", "metrics.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# 改过名的字段：新名 -> 老名。v2-* 那批 metrics.jsonl 里还是老名，
# 缺了就回退，否则跨代对比里这一列整片是「—」。
RENAMED = {"policy_entropy_model": "policy_entropy"}


def tail_mean(rows, key, n=10):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals and key in RENAMED:
        old = RENAMED[key]
        vals = [r[old] for r in rows if old in r and r[old] is not None]
    return sum(vals[-n:]) / len(vals[-n:]) if vals else None


def sparkline(vals, width=32):
    """用文本画个粗略趋势，比一堆数字直观。"""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return ""
    step = max(1, len(vals) // width)
    pts = vals[::step][:width]
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-12:
        return "─" * len(pts)
    chars = "▁▂▃▄▅▆▇█"
    return "".join(chars[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="实验名（runs/<exp>）")
    ap.add_argument("--metric", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--tail", type=int, default=10, help="末尾多少条取平均")
    args = ap.parse_args()

    data = {e: load(e) for e in args.exps}

    print(f"{'指标':<20}" + "".join(f"{e:>16}" for e in args.exps))
    print("-" * (20 + 16 * len(args.exps)))
    for m in args.metric:
        vals = [tail_mean(data[e], m, args.tail) for e in args.exps]
        if all(v is None for v in vals):
            continue
        cells = "".join(f"{v:>16.4f}" if v is not None else f"{'—':>16}" for v in vals)
        print(f"{m:<20}{cells}")

    print()
    for e in args.exps:
        rows = data[e]
        steps = rows[-1].get("step", 0) if rows else 0
        print(f"{e}: {len(rows)} 轮，step {steps}")
        for m in ["loss", *LEGACY_METRICS]:
            s = sparkline([r.get(m) for r in rows])
            if s:
                print(f"  {m:<14} {s}")

    if len(args.exps) == 2:
        a, b = args.exps
        print()
        for m in ["loss", "policy", "value", *LEGACY_METRICS]:
            va, vb = tail_mean(data[a], m, args.tail), tail_mean(data[b], m, args.tail)
            if va is None or vb is None:
                continue
            diff = vb - va
            rel = diff / abs(va) * 100 if va else 0.0
            print(f"{m:<16} {b} 相对 {a}: {diff:+.4f} ({rel:+.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
