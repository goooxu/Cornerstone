#!/usr/bin/env python3
"""训练入口。

    python3 tools/train.py --exp bf16 --dim 256 --blocks 16
    python3 tools/train.py --smoke              # 小配置跑通端到端

同一条命令重复执行即为续训：会自动从 runs/<exp>/ckpt/latest 恢复。
"""

import argparse
import os
import sys
import time
from dataclasses import fields

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.evaluate import evaluate_vs_baseline  # noqa: E402
from cornerstone.train import TrainConfig, Trainer      # noqa: E402


def build_config(argv=None) -> tuple[TrainConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小配置跑通端到端，几分钟出结果")
    ap.add_argument("--max-iters", type=int, default=0, help="跑多少轮后停（0 = 不限）")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--no-resume", action="store_true")
    # TrainConfig 的每个字段都暴露成命令行参数
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        val = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(val, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=val)
        else:
            ap.add_argument(flag, type=type(val), default=val)
    args = ap.parse_args(argv)

    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    if args.smoke:
        cfg.exp = cfg.exp if cfg.exp != "bf16" else "smoke"
        cfg.dim, cfg.blocks, cfg.attn_every = 96, 6, 3
        cfg.parallel_games, cfg.games_per_iter = 128, 128
        cfg.simulations, cfg.max_considered = 32, 8
        cfg.batch_size, cfg.steps_per_iter = 256, 60
        cfg.min_positions = 3000
        cfg.replay_capacity = 300_000
        cfg.warmup_steps, cfg.total_steps = 100, 4000
        cfg.eval_every_iters, cfg.eval_games = 4, 120
        cfg.ckpt_every_steps = 500
    return cfg.resolve(REPO), args


def main() -> int:
    cfg, args = build_config()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(cfg)
    print(f"实验 {cfg.exp} | 参数量 {trainer.model.num_params()/1e6:.1f}M | "
          f"设备 {cfg.device} | 输出 {cfg.run_dir}")
    if not args.no_resume and trainer.resume():
        print(f"已从 checkpoint 恢复：step={trainer.step} iter={trainer.iteration} "
              f"replay={len(trainer.buffer)} 局面")

    driver = trainer.make_driver()
    t_start = time.time()

    while True:
        if args.max_iters and trainer.iteration >= args.max_iters:
            break
        if args.max_minutes and (time.time() - t_start) / 60 >= args.max_minutes:
            print("到达时间上限，收尾")
            break
        if trainer.step >= cfg.total_steps:
            print("到达总步数上限")
            break

        recs, sp = driver.run(cfg.games_per_iter)
        trainer.buffer.add_records(recs)

        row = {
            "selfplay_games": sp.games,
            "selfplay_games_per_s": sp.games_per_s,
            "selfplay_evals_per_s": sp.evals_per_s,
            "selfplay_mean_batch": sp.mean_batch,
            "replay_positions": len(trainer.buffer),
            "replay_games": trainer.buffer.total_games_seen,
            "mean_plies": sum(len(r["actions"]) for r in recs) / max(1, len(recs)),
            "draw_rate": sum(1 for r in recs if r["result0"] == 0) / max(1, len(recs)),
            "p0_win_rate": sum(1 for r in recs if r["result0"] > 0) / max(1, len(recs)),
        }

        if len(trainer.buffer) >= cfg.min_positions:
            row.update(trainer.train_steps(cfg.steps_per_iter))

        trainer.iteration += 1

        if cfg.eval_every_iters and trainer.iteration % cfg.eval_every_iters == 0:
            res = evaluate_vs_baseline(
                trainer.model, trainer.device, opponent=cfg.eval_opponent,
                games=cfg.eval_games, simulations=cfg.eval_simulations,
                parallel_games=min(cfg.parallel_games, cfg.eval_games),
                seed=trainer.iteration,
            )
            print("  " + str(res))
            row.update({"eval_opponent": res.opponent, "eval_score_rate": res.score_rate,
                        "eval_elo_diff": res.elo_diff, "eval_elo_abs": res.elo_abs})

        trainer.log(row)
        loss_s = f" loss={row['loss']:.4f} pol={row['policy']:.4f} val={row['value']:.4f}" \
            if "loss" in row else " （攒数据中）"
        print(f"[iter {trainer.iteration:4d} step {trainer.step:7d}] "
              f"自博弈 {sp.games_per_s:.1f} 局/s 批均 {sp.mean_batch:.0f} | "
              f"replay {len(trainer.buffer):,}" + loss_s, flush=True)

        if trainer.maybe_checkpoint():
            pass
        if cfg.snapshot_every_iters and trainer.iteration % cfg.snapshot_every_iters == 0:
            trainer.save_snapshot()

    trainer.save_checkpoint()
    trainer.save_snapshot()
    print(f"已落盘 checkpoint 与 replay 快照到 {cfg.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
