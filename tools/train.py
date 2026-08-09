#!/usr/bin/env python3
"""训练入口。

    python3 tools/train.py --exp bf16 --dim 256 --blocks 16
    python3 tools/train.py --smoke              # 小配置跑通端到端

同一条命令重复执行即为续训：会自动从 runs/<exp>/ckpt/latest 恢复。
"""

import argparse
import os
import signal
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


# 开发机单次会话有时长上限，被回收时收到的是 SIGTERM。
# 一轮自博弈可能要几分钟，不能等它跑完才收尾，否则会被强杀、丢掉未落盘的进度。
# 这里把信号转成一个协作式的停止标志，自博弈与训练循环都会检查它。
_STOP = False


def _request_stop(signum, _frame):
    global _STOP
    if _STOP:                       # 第二次信号就别再等了
        print("再次收到信号，立即退出", flush=True)
        sys.exit(1)
    _STOP = True
    print(f"收到信号 {signum}，尽快收尾并落盘…", flush=True)


def should_stop() -> bool:
    return _STOP


def main() -> int:
    cfg, args = build_config()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(cfg)
    print(f"实验 {cfg.exp} | 参数量 {trainer.model.num_params()/1e6:.1f}M | "
          f"设备 {cfg.device} | 输出 {cfg.run_dir}")
    if not args.no_resume and trainer.resume():
        print(f"已从 checkpoint 恢复：step={trainer.step} iter={trainer.iteration} "
              f"replay={len(trainer.buffer)} 局面")

    trainer.verify_fp8_compute("启动")
    driver = trainer.make_driver()
    # 再查一次：构建自博弈驱动会创建副本、同步权重、做预热，
    # 这些路径都可能把 FP8 的状态搞坏。启动时那条找不到源头的
    # "quantized weights without quantized compute" 警告就出现在这一段。
    trainer.verify_fp8_compute("建驱动后")

    # 对手池：另起一个「主网络 vs 历史 checkpoint」的驱动。两方都建树、两方的手
    # 都记录，所以记录可以回放、能进 replay buffer；但只有主网络那一方的手参与训练。
    pool_driver = trainer.make_pool_driver() if cfg.pool_frac > 0 else None
    if pool_driver is not None:
        print(f"对手池已启用：{cfg.pool_frac:.0%} 的对局对手从最近 "
              f"{cfg.pool_window} 个里程碑里采样，每轮 {cfg.pool_opponents_per_iter} 个对手")
        # 池驱动会在每张卡上再建一份对手副本，正是容易把 FP8 计算通路碰坏的地方
        trainer.verify_fp8_compute("建对手池驱动后")

    t_start = time.time()

    while True:
        if _STOP:
            print("按请求停止")
            break
        if args.max_iters and trainer.iteration >= args.max_iters:
            break
        if args.max_minutes and (time.time() - t_start) / 60 >= args.max_minutes:
            print("到达时间上限，收尾")
            break
        if trainer.step >= cfg.total_steps:
            print("到达总步数上限")
            break

        driver.sync_weights()      # 把上一轮训好的权重推给各卡的副本

        # 池里还没有可用的里程碑时（训练最初的一万步）自动退化成纯自博弈
        opps = trainer.sample_pool_opponents(cfg.pool_opponents_per_iter) \
            if pool_driver is not None else []
        pool_games = int(cfg.games_per_iter * cfg.pool_frac) if opps else 0
        self_games = cfg.games_per_iter - pool_games

        recs, sp = driver.run(self_games, should_stop=should_stop)
        trainer.buffer.add_records(recs)
        trainer.games_played += len(recs)

        pool_rate, pool_n = None, 0
        if pool_games:
            pool_driver.sync_weights()
            per = max(2, pool_games // len(opps))
            wins = 0.0
            for st in opps:
                pool_driver.load_opponent(trainer.load_pool_state_dict(st))
                prec, psp = pool_driver.run(per, should_stop=should_stop)
                # 只训练主网络那一方的手
                trainer.buffer.add_records(prec, main_player_only=True)
                trainer.games_played += len(prec)
                for r in prec:
                    res = r["result0"] if r["net_player"] == 0 else -r["result0"]
                    wins += 1.0 if res > 0 else (0.5 if res == 0 else 0.0)
                pool_n += len(prec)
                sp.games += psp.games
                sp.evals += psp.evals
                sp.seconds += psp.seconds
            pool_rate = wins / max(1, pool_n)

        row = {
            "selfplay_games": sp.games,
            "selfplay_games_per_s": sp.games_per_s,
            "selfplay_evals_per_s": sp.evals_per_s,
            "selfplay_mean_batch": sp.mean_batch,
            "replay_positions": len(trainer.buffer),
            # buffer_games 是进程内计数（续训后从快照重新播种），
            # games_played 才是全生命周期的累计对局数
            "buffer_games": trainer.buffer.total_games_seen,
            "games_played": trainer.games_played,
            "mean_plies": sum(len(r["actions"]) for r in recs) / max(1, len(recs)),
            "draw_rate": sum(1 for r in recs if r["result0"] == 0) / max(1, len(recs)),
            "p0_win_rate": sum(1 for r in recs if r["result0"] > 0) / max(1, len(recs)),
        }
        if pool_rate is not None:
            # 主网络对池中对手的得分率。这是不会饱和的进度信号 —— 对手也在变强。
            row["pool_score_rate"] = pool_rate
            row["pool_games"] = pool_n
            row["pool_opponents"] = list(opps)

        if len(trainer.buffer) >= cfg.min_positions:
            steps = trainer.steps_for_iteration()
            row["planned_steps"] = steps
            row.update(trainer.train_steps(steps, should_stop=should_stop))

        trainer.iteration += 1

        # FP8 自检**不能挂在评测上**。原来两者在同一个 if 里，一旦把评测关掉
        # （eval_every_iters=0），自检也跟着没了 —— 而 FP8 是会静默降级的：
        # 只发一条 UserWarning，模型看着在训练，FP8 已经名存实亡（docs/06 第五条）。
        # 没开 FP8 的跑返回 None，这时**不写这个字段**：记成 false 会读作
        # 「FP8 掉了」，记成 true 更糟（对照组看着像实验组）。
        if cfg.fp8_check_every_iters and \
                trainer.iteration % cfg.fp8_check_every_iters == 0 and not _STOP:
            fp8_ok = trainer.verify_fp8_compute(f"iter {trainer.iteration}")
            if fp8_ok is not None:
                row["fp8_active"] = fp8_ok

        if cfg.eval_every_iters and trainer.iteration % cfg.eval_every_iters == 0 and not _STOP:
            res = evaluate_vs_baseline(
                trainer.model, trainer.device, opponent=cfg.eval_opponent,
                games=cfg.eval_games, simulations=cfg.eval_simulations,
                parallel_games=min(cfg.parallel_games, cfg.eval_games),
                seed=trainer.iteration,
                # 不传的话会退到单线程，评测能吃掉大半墙钟（见 evaluate_vs_baseline 的注释）
                engine_threads=cfg.engine_threads,
            )
            print("  " + str(res))
            row.update({"eval_opponent": res.opponent, "eval_score_rate": res.score_rate,
                        "eval_elo_diff": res.elo_diff, "eval_elo_abs": res.elo_abs})

        trainer.log(row)
        loss_s = f" loss={row['loss']:.4f} pol={row['policy']:.4f} val={row['value']:.4f}" \
            if "loss" in row else " （攒数据中）"
        pool_s = f" 池 {pool_rate:.3f}" if pool_rate is not None else ""
        print(f"[iter {trainer.iteration:4d} step {trainer.step:7d}] "
              f"自博弈 {sp.games_per_s:.1f} 局/s 批均 {sp.mean_batch:.0f} | "
              f"replay {len(trainer.buffer):,}" + loss_s + pool_s, flush=True)

        if trainer.maybe_checkpoint():
            pass
        if cfg.snapshot_every_iters and trainer.iteration % cfg.snapshot_every_iters == 0:
            trainer.save_snapshot()

    # 无论是正常结束还是被信号打断，都要落一份完整的 checkpoint + replay 快照
    trainer.save_checkpoint()
    trainer.save_snapshot()
    print(f"已落盘 checkpoint 与 replay 快照到 {cfg.run_dir}（step {trainer.step}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
