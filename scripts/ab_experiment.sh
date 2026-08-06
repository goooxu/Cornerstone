#!/usr/bin/env bash
# FP8 vs BF16 的受控对照实验。
#
#   bash scripts/ab_experiment.sh start
#   bash scripts/ab_experiment.sh stop
#   bash scripts/ab_experiment.sh status
#   bash scripts/ab_experiment.sh compare [步数]     # 在对齐步数处头对头
#
# 方案里写死了一条：**没有 BF16 对照，「FP8 无损」就是没有依据的说法。**
# 第一次做这个对照失败了，原因值得记下来：
#
#   1. 两条跑的 games_per_iter 不一样（1024 vs 2048），于是相同 step 下
#      FP8 只看到一半的新自博弈数据 —— 而数据量是 AlphaZero 类训练的主导因素
#   2. FP8 那条中途崩过 6 次，其中 2 次 replay 快照丢失、buffer 从零重建
#   3. 里程碑 checkpoint 保留是后加的，早期 checkpoint 已被删，
#      没有可对齐的步数做头对头
#
# 所以这次把变量锁死：
#
#   * 除 --fp8 外全部配置逐字相同，种子相同，都从零开始
#   * **两边都关掉 torch.compile** —— 它会改变算子融合与数值细节，
#     留着它 BF16 就多了一个变量。关掉后两边墙钟速度也变得可比
#   * 每 10000 步留一个永久里程碑 checkpoint，供后续头对头
#   * 各占两张卡，engine_threads 对半分
#
# 结论只认**头对头胜负**，不认 loss 曲线：策略目标是网络自己搜索出来的，
# 网络变强目标就变尖，跨实验比 loss 得不出棋力结论。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$(dirname "$REPO")/runs"

A_EXP="${A_EXP:-ab-bf16}"
B_EXP="${B_EXP:-ab-fp8}"

# 除 --fp8 与设备外，两边逐字相同
COMMON=(
  --dim 256 --blocks 16 --attn-every 4
  --parallel-games 4096 --games-per-iter 2048
  --simulations 64 --max-considered 16 --temperature-plies 12
  --batch-size 1024 --steps-per-iter 400
  --lr 0.002 --warmup-steps 500 --total-steps 200000
  --engine-threads 64
  --compile-model false          # 两边都关，消掉这个变量
  --milestone-every-steps 10000
  --eval-opponent flat-mcts-4k --eval-simulations 128 --eval-every-iters 10
  --seed 1
)

case "${1:-}" in
  start)
    for e in "$A_EXP" "$B_EXP"; do
      if [ -e "$RUNS/$e/ckpt/latest" ]; then
        echo "$RUNS/$e 已有 checkpoint —— 对照实验必须从零开始。" >&2
        echo "要重来请先手动清掉该目录。" >&2
        exit 1
      fi
    done
    bash "$REPO/scripts/train.sh" start "$A_EXP" \
      --fp8 false --device cuda:0 --selfplay-devices cuda:0,cuda:1 "${COMMON[@]}"
    bash "$REPO/scripts/train.sh" start "$B_EXP" \
      --fp8 true  --device cuda:2 --selfplay-devices cuda:2,cuda:3 "${COMMON[@]}"
    ;;
  stop)
    bash "$REPO/scripts/train.sh" stop "$A_EXP"
    bash "$REPO/scripts/train.sh" stop "$B_EXP"
    ;;
  status)
    for e in "$A_EXP" "$B_EXP"; do
      echo "=== $e ==="
      bash "$REPO/scripts/train.sh" status "$e" | head -3
    done
    ;;
  compare)
    STEP="${2:-}"
    if [ -z "$STEP" ]; then
      echo "用法: $0 compare <步数>，例如 $0 compare 10000" >&2
      exit 1
    fi
    # 找**最接近**目标步数的 checkpoint 而不是要求精确命中 ——
    # 落盘点受限流影响不会正好停在整数倍上（19832 而不是 20000）
    read -r A A_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$A_EXP/ckpt" "$STEP") \
      || { echo "$A_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    read -r B B_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$B_EXP/ckpt" "$STEP") \
      || { echo "$B_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    D=$(( A_STEP > B_STEP ? A_STEP - B_STEP : B_STEP - A_STEP ))
    echo "目标 step $STEP -> 实际 $A_EXP@$A_STEP vs $B_EXP@$B_STEP（相差 $D 步）"
    cd "$REPO"
    echo "在 step $STEP 处头对头（A=$A_EXP 为 BF16，B=$B_EXP 为 FP8）"
    python3 tools/compare_nets.py "$B" "$A" --games 400 --simulations 64 \
            --parallel 200 --device cuda:0
    ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
