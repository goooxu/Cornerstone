#!/usr/bin/env bash
# 带对手池的 FP8 vs BF16 受控对照。
#
#   bash scripts/pool_experiment.sh start-bf16
#   bash scripts/pool_experiment.sh start-fp8
#   bash scripts/pool_experiment.sh status
#   bash scripts/pool_experiment.sh pool          # 看两条腿的池得分率走势
#   bash scripts/pool_experiment.sh compare [步数]
#
# 与 ab_experiment.sh 的唯一区别是**自博弈的对手**：
# 纯自博弈时对手只有当前的自己，网络往哪儿漂对手就跟着漂，没有外部参照。
# 这里让一半的对局去打本次运行自己已落盘的历史里程碑 —— 每一轮都要重新赢过
# 自己的过去，漂弱了会立刻表现为 pool_score_rate 下降。
#
# 三条必须守住的约束（写在这里免得改坏）：
#
#   1. 池对局用 net_opponent + opening_plies=0：两方都建树、两方的手都记录，
#      序列才能从空盘回放、进得了 replay buffer。引擎里有断言兜着。
#   2. 只有主网络那一方的手参与训练。另一侧来自更弱的历史 checkpoint，
#      拿它的搜索结果当目标等于向弱教师学习。
#   3. games_per_iter 从 2048 提到 2730：池对局只有一半的手可训练，
#      不提的话每轮的可训练局面数会掉四分之一，就多了一个变量。
#
# 除 --fp8 与设备外两边逐字相同；结论只认头对头胜负，不认 loss 曲线。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$(dirname "$REPO")/runs"

BF16_EXP="${BF16_EXP:-pool-bf16}"
FP8_EXP="${FP8_EXP:-pool-fp8}"

# 两条腿各自用哪些 GPU。**卡数必须一致**，否则每卡的并行局数不同，就多了一个变量。
#   机器 1:  POOL_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/pool_experiment.sh start-bf16
#   机器 2:  POOL_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/pool_experiment.sh start-fp8
POOL_DEVICES="${POOL_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
ENGINE_THREADS="${ENGINE_THREADS:-128}"

# 超参只此一份。抄第二份的教训见 docs/06：守护脚本曾自己拼过一套启动参数，
# 只在开发机过期后自动恢复时生效，两处一旦不同步，实验会从某次恢复开始
# 悄悄变成另一个实验，日志上完全看不出来。
COMMON=(
  --dim 256 --blocks 16 --attn-every 4
  --parallel-games 4096 --games-per-iter 2730
  --simulations 64 --max-considered 16 --temperature-plies 12
  --batch-size 1024 --steps-per-iter 400
  --lr 0.002 --warmup-steps 500 --total-steps 200000
  --compile-model false          # 两边都关，消掉这个变量
  --milestone-every-steps 10000
  --pool-frac 0.5 --pool-window 8 --pool-opponents-per-iter 2
  --eval-every-iters 0           # 规则基线在训练走完 5% 时就被打穿了，
                                 # 真正的进度信号是 pool_score_rate
  --seed 1
)

# 重复执行即续训（train.sh start 会自动从 checkpoint 恢复）。
# **这里不能加「已有 checkpoint 就拒绝」的护栏** —— 守护恢复训练走的正是这条路，
# 加了护栏开发机一回收训练就被搁浅，而日志里只会看到「恢复失败，下一轮重试」。
# 要从零重来请先手动清掉 runs/<exp>。
start_one() {   # $1 = 实验名, $2 = fp8 true/false
  bash "$REPO/scripts/train.sh" start "$1" --fp8 "$2" \
    --device "${POOL_DEVICES%%,*}" --selfplay-devices "$POOL_DEVICES" \
    --engine-threads "$ENGINE_THREADS" "${COMMON[@]}"
}

case "${1:-}" in
  start-bf16) start_one "$BF16_EXP" false ;;
  start-fp8)  start_one "$FP8_EXP"  true  ;;
  stop)
    bash "$REPO/scripts/train.sh" stop "$BF16_EXP" || true
    bash "$REPO/scripts/train.sh" stop "$FP8_EXP"  || true
    ;;
  status)
    for e in "$BF16_EXP" "$FP8_EXP"; do
      echo "=== $e ==="
      bash "$REPO/scripts/train.sh" status "$e" 2>/dev/null | head -3 || echo "（未启动）"
    done
    ;;
  pool)
    # 池得分率的走势。持续贴近 1.0 说明对手太弱（窗口取得太靠后）；
    # 掉到 0.5 以下说明主网络正在被自己的过去打败 —— 那就是漂弱了。
    for e in "$BF16_EXP" "$FP8_EXP"; do
      f="$RUNS/$e/logs/metrics.jsonl"
      [ -f "$f" ] || continue
      echo "=== $e ==="
      python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$f')]
pr=[r for r in rows if 'pool_score_rate' in r]
if not pr: print('  还没有池数据（前一万步没有可用里程碑）'); sys.exit()
for r in pr[::max(1,len(pr)//12)]:
    print(f\"  step {r['step']:>7}  池得分率 {r['pool_score_rate']:.3f}  \"
          f\"先手胜率 {r['p0_win_rate']:.3f}  对手 {r.get('pool_opponents')}\")
"
    done
    ;;
  compare)
    STEP="${2:-}"
    if [ -z "$STEP" ]; then
      echo "用法: $0 compare <步数>" >&2; exit 1
    fi
    read -r A A_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$BF16_EXP/ckpt" "$STEP") \
      || { echo "$BF16_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    read -r B B_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$FP8_EXP/ckpt" "$STEP") \
      || { echo "$FP8_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    echo "目标 step $STEP -> 实际 $BF16_EXP@$A_STEP vs $FP8_EXP@$B_STEP"
    cd "$REPO"
    # 名字显式传给 compare_nets.py 印在每一行上，不靠位置约定 ——
    # 这个对照唯一要产出的结论，最容易被一个参数顺序悄悄翻转（见 docs/08）。
    python3 tools/compare_nets.py "$A" "$B" --games 400 --simulations 0 \
            --parallel 200 --device cuda:0 \
            --name-a "$BF16_EXP" --name-b "$FP8_EXP"
    ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
