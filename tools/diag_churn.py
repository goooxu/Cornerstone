#!/usr/bin/env python3
"""D3：量 policy churn 与价值头退化，**分「分布内 / 分布外」两层**。

    python3 tools/diag_churn.py ../runs/v2-bf16/ckpt/step001{10227,20227,30227,40227,50227}.pt \
        --ref ../runs/v2-bf16/ckpt/step00150227.pt

## 要验的假设

Gumbel-AZ 的改进目标是 `softmax(log π + σ·q̂)`，而 `σ = (c_visit + max_n)·c_scale`
在本项目的配置下约 50~66。q̂ 是 1~4 次访问的叶子价值均值 —— 价值头在单个叶子上
的噪声乘上 σ 就是一个多 logit 的纯噪声。当候选着法的真实 q 差距降到 1/σ ≈ 0.012
以下，**目标就退化成「把价值噪声放大 60 倍重新排序」**，每轮教网络记住一组新的
随机重排。这就是 policy churn，它精确预言「标量指标全好、真实棋力退」。

## 判据

churn 假设预言：相邻档之间 loss 几乎不变，但

- **分布外局面（类 B）的 top-1 翻转率显著高于分布内（类 A）** —— 因为分布外
  局面没有梯度去维护，价值头在那里最噪
- 类 B 上末期档的 WDL 输出更接近常数（价值头塌成"猜先手赢"）
- 各档在类 B 上的价值输出两两相关度随训练下降 = 噪声在涨

如果两层的翻转率差不多，churn 假设就要让位给别的解释。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone import _engine as E                        # noqa: E402
from cornerstone.model import load_checkpoint               # noqa: E402
from cornerstone.selfplay import SelfPlayDriver             # noqa: E402


def random_boards(n: int, rng: np.random.Generator, lo: int, hi: int) -> list:
    """均匀随机走 k∈[lo,hi] 手得到的局面 —— 分布外的那一层。

    这些局面训练时几乎不会碰到（自博弈 12 手之后全是 argmax，且分布已经塌了），
    所以它们上面的策略/价值没有任何梯度在维护。评测用的却正是随机开局。
    """
    out = []
    while len(out) < n:
        b = E.Board()
        b.reset()
        k = int(rng.integers(lo, hi + 1))
        ok = True
        for _ in range(k):
            if b.terminal:
                ok = False
                break
            mv = b.legal_moves()
            if len(mv) == 0:
                ok = False
                break
            b.play(int(mv[rng.integers(len(mv))]))
        if ok and not b.terminal and b.legal_count() > 0:
            out.append(b)
    return out


def selfplay_boards(recs, n: int, rng: np.random.Generator, lo: int, hi: int) -> list:
    """把自博弈记录回放到某一手 —— 分布内的那一层。"""
    out = []
    for r in recs:
        acts = np.asarray(r["actions"], dtype=np.int32)
        if len(acts) <= lo + 1:
            continue
        k = int(rng.integers(lo, min(hi, len(acts) - 1) + 1))
        b = E.Board()
        b.reset()
        for a in acts[:k]:
            b.play(int(a))
        if not b.terminal and b.legal_count() > 0:
            out.append(b)
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def evaluate(model, boards, device: str, batch: int = 1024):
    """返回 (合法集上的 log_softmax [B,A], argmax [B], wdl 概率 [B,3])。"""
    planes, scalars = E.features_batch(boards)
    legal = np.stack([b.legal_mask() for b in boards]).astype(bool)
    n = len(boards)
    logp = np.empty((n, E.NUM_ACTIONS), dtype=np.float32)
    wdl = np.empty((n, 3), dtype=np.float32)
    lg = torch.from_numpy(legal)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        p = torch.from_numpy(planes[i:j]).to(device)
        s = torch.from_numpy(scalars[i:j]).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pol, w, _ = model(p, s)
        pol = pol.float().cpu()
        pol = pol.masked_fill(~lg[i:j], float("-inf"))
        logp[i:j] = torch.log_softmax(pol, dim=-1).numpy()
        wdl[i:j] = w.float().softmax(dim=-1).cpu().numpy()
    return logp, logp.argmax(axis=1), wdl


def kl(p_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    """KL(p‖q)，两边都是 log 概率；-inf（非法着法）的位置贡献 0。"""
    p = np.exp(p_log)
    d = np.where(np.isfinite(p_log) & np.isfinite(q_log), p * (p_log - q_log), 0.0)
    return d.sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="按步数升序给")
    ap.add_argument("--ref", required=True, help="用哪一档生成「分布内」局面")
    ap.add_argument("--positions", type=int, default=4096)
    ap.add_argument("--games", type=int, default=1024, help="生成分布内局面用的自博弈局数")
    ap.add_argument("--ply-lo", type=int, default=6)
    ap.add_argument("--ply-hi", type=int, default=20)
    ap.add_argument("--engine-threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = args.device

    print("① 生成分布内局面（用 ref 档自博弈）…", flush=True)
    ref, ref_step = load_checkpoint(args.ref, dev)
    mcts = E.MctsConfig(simulations=64, max_considered=16, temperature_plies=12)
    drv = SelfPlayDriver(ref, dev, num_games=1024, mcts=mcts, seed=args.seed,
                         compile_model=True, engine_threads=args.engine_threads)
    drv.warmup()
    recs, _ = drv.run(args.games)
    del drv, ref
    torch.cuda.empty_cache()

    A = selfplay_boards(recs, args.positions, rng, args.ply_lo, args.ply_hi)
    print(f"   分布内 {len(A)} 个局面（ref = step {ref_step}）", flush=True)

    print("② 生成分布外局面（均匀随机走子）…", flush=True)
    B = random_boards(args.positions, rng, args.ply_lo, args.ply_hi)
    print(f"   分布外 {len(B)} 个局面", flush=True)

    print("③ 逐档前向…", flush=True)
    res = {}
    steps = []
    for path in args.ckpts:
        m, step = load_checkpoint(path, dev)
        steps.append(step)
        res[step] = {kind: evaluate(m, bs, dev) for kind, bs in (("in", A), ("out", B))}
        del m
        torch.cuda.empty_cache()
        print(f"   step {step} 完成", flush=True)

    rows = []
    for a, b in zip(steps, steps[1:]):
        row = {"from": a, "to": b}
        for kind in ("in", "out"):
            lp_a, am_a, w_a = res[a][kind]
            lp_b, am_b, w_b = res[b][kind]
            row[f"flip_{kind}"] = float((am_a != am_b).mean())
            row[f"kl_{kind}"] = float(np.median(kl(lp_a, lp_b)))
        rows.append(row)

    print()
    print(f"{'相邻档':>18}{'翻转率(分布内)':>16}{'翻转率(分布外)':>16}"
          f"{'KL中位(内)':>13}{'KL中位(外)':>13}{'外/内':>8}")
    print("-" * 84)
    for r in rows:
        ratio = r["flip_out"] / max(1e-9, r["flip_in"])
        print(f"{r['from']:>8} → {r['to']:<8}{r['flip_in']:>16.3f}{r['flip_out']:>16.3f}"
              f"{r['kl_in']:>13.4f}{r['kl_out']:>13.4f}{ratio:>8.2f}")

    print()
    print(f"{'档':>10}{'P(先手胜)均值':>16}{'P(先手胜)标准差':>18}{'WDL熵':>10}   （分布外）")
    print("-" * 60)
    vals = {}
    for s in steps:
        w = res[s]["out"][2]
        ent = float((-w * np.log(np.clip(w, 1e-9, 1))).sum(axis=1).mean())
        vals[s] = w[:, 0] - w[:, 2]                       # P(胜) − P(负)，行棋方视角
        print(f"{s:>10}{w[:, 0].mean():>16.3f}{w[:, 0].std():>18.3f}{ent:>10.3f}")

    print()
    print("各档在分布外局面上的价值输出两两相关（低 = 价值噪声大）：")
    hdr = "".join(f"{s // 1000:>8}k" for s in steps)
    print(f"{'':<10}{hdr}")
    for i in steps:
        line = "".join(f"{np.corrcoef(vals[i], vals[j])[0, 1]:>9.3f}" for j in steps)
        print(f"{i // 1000:>8}k  {line}")

    print()
    print("判据：churn 假设预言「外/内」这一列显著 > 1 —— 也就是回退发生在"
          "没有梯度维护的分布外局面上，而评测恰恰用随机开局。")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": {k: v for k, v in vars(args).items()},
                       "steps": steps, "pairs": rows}, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
