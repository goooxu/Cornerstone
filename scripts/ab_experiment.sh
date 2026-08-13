#!/usr/bin/env bash
# BF16 / FP8(MXFP8) / FP4(NVFP4) 三组的受控对照实验。
#
#   bash scripts/ab_experiment.sh start        # 起 A+B（同机两组，各半数卡）
#   bash scripts/ab_experiment.sh start-c      # FP4 组，通常在第二台机器上单起
#   bash scripts/ab_experiment.sh stop
#   bash scripts/ab_experiment.sh status
#   bash scripts/ab_experiment.sh compare [步数]     # 在对齐步数处头对头
#
# 方案里写死了一条：**没有 BF16 对照，「FP8 无损」就是没有依据的说法。**
# FP4 组同理 —— 它的判据是「相对 BF16 组损失多少 Elo」，不是它自己的 loss 曲线。
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
#   * 除 --precision 外全部配置逐字相同，种子相同，都从零开始
#   * **torch.compile 两边都开**，各走各最快的编法（BF16 整模型、
#     FP8 只能按 block，整模型会 SIGSEGV，见 docs/06 第四条）。
#     **这是一个已知的不对称**：两边的算子融合不同，严格说多了一个变量。
#     接受它是因为影响比 FP8 量化本身小（compile 换 eager 改 4.3% 的 argmax，
#     FP8 量化改 9.2%），而换来的是自博弈 2.4 倍。读结论时心里有数即可。
#   * 每 10000 步留一个永久里程碑 checkpoint，供后续头对头
#   * 每卡的引擎线程数用 TrainConfig 的默认值（8），不再另外指定
#   * **两组的 COMMON 参数表是同一份**，没有任何按组分叉的旋钮 —— 见下面
#     那条注释：曾经为 B 组单独留了一个 B_PARALLEL，结果在一次自动恢复时
#     悄悄把并行局数减半，训练日志上看不出来。
#
# 结论只认**头对头胜负**，不认 loss 曲线：策略目标是网络自己搜索出来的，
# 网络变强目标就变尖，跨实验比 loss 得不出棋力结论。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$(dirname "$REPO")/runs"

A_EXP="${A_EXP:-v4-bf16}"
B_EXP="${B_EXP:-v4-fp8}"
C_EXP="${C_EXP:-v4-fp4}"

# 两组各自用哪些 GPU。默认挤在一台机器上各占两张卡；
# 有第二台机器时，在各自机器上分别 start 单组、各占四张卡：
#   机器 1:  A_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/ab_experiment.sh start-a
#   机器 2:  B_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/ab_experiment.sh start-b
# **两组的卡数必须一致**，否则每卡的并行局数不同，就多了一个变量。
A_DEVICES="${A_DEVICES:-cuda:0,cuda:1}"
B_DEVICES="${B_DEVICES:-cuda:2,cuda:3}"
# C 组（FP4）默认吃满一台机器 —— 它本来就要单独排一轮，见 docs/09 的编排。
C_DEVICES="${C_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

# 这里曾经有一个只作用于 B 组的 `B_PARALLEL=2048`（当时 FP8 自博弈在同进程
# 多线程下不扩展，用减半并行局数来对齐每卡负载）。**换成每卡一个工作进程之后
# FP8 线性扩展了（4 卡 3.95×），这个旋钮就没有理由再存在** —— 而它留下来的
# 那段时间里造成过一次真实的污染：训练中断后守护脚本自动恢复，恢复用的是
# 默认值，于是 B 组最后 15% 的并行局数悄悄从 4096 变成 2048。
# 日志上只有「自博弈慢了一半」这一个症状，配置本身没有任何提示。
#
# **教训：受控对照里不要留「只作用于一组」的默认值。** 它在手动启动时是显式的，
# 在自动恢复时是隐式的，而自动恢复恰恰是最没人盯着的时刻。

# 每组的额外旋钮，**按实验名分派、写死在脚本里**。
#
# 不用环境变量传：守护脚本自动恢复时只转交实验名与设备，任何靠 env 传进来的
# 旋钮都会在那一刻悄悄消失或退回默认值 —— v2-fp8 的 parallel_games 就是这么
# 在第 12.8 万步被从 4096 改成 2048 的，日志上只表现为「自博弈慢了一半」。
# 配置跟着实验名走，恢复出来的就一定还是同一个实验。
#
# 现在是空的（随机开局注入已成为 TrainConfig 的默认值，不需要再单独传）。
# 留着这个钩子是因为下一轮还要拿它挂 value_from_score / simulations 之类的臂。
extra_for() {
  case "$1" in
    *)  echo "" ;;
  esac
}

# 精度也按实验名分派，理由与 extra_for 完全相同：守护脚本自动恢复时只转交
# 实验名与设备。**跑名里没写精度就退回 bf16**，所以实验名必须带后缀。
# `watch_training.sh` 那边有一条对称的、拒绝拉起无法判定精度的实验的守卫。
precision_for() {
  case "$1" in
    *-fp4) echo "fp4" ;;
    *-fp8) echo "fp8" ;;
    *)     echo "bf16" ;;
  esac
}

# 三组逐字共用这一份，除 --precision、设备、extra_for 外没有任何差别。
#
# **WSD 的三个字段必须待在这里，不能走环境变量**：`save_checkpoint` 虽然存了
# `asdict(cfg)`，但 `load_checkpoint` 从不读它 —— 学习率曲线的形状 100% 由本次
# 命令行决定。守护恢复时若少传一个 `--lr-horizon-steps`，horizon 会退回
# `total_steps`，曲线形状不变但**下一次改 total_steps 时会整体右移**。
#
# 另：`watch_training.sh` 的 `total_steps()` 是 `grep --total-steps | head -1`
# 抠这个文件，所以 `--total-steps` 必须是这里出现的第一个（也是唯一一个）步数参数，
# 也不要往 extra_for 里塞任何 --total-steps。
#
# 交付点 A：warmup 500 -> stable 恒 2e-3 到 100,000 -> 线性退火 11,000 步
# 到 2e-4 @ 111,000。选 10 万做 stable 的依据是实测「92% 的棋力在 8 万步到手」，
# 之后每万步的增量（+2~+12 Elo）已落在测量误差 ±9.6 之内。
COMMON=(
  --dim 256 --blocks 16 --attn-every 4
  --parallel-games 4096 --games-per-iter 2048
  --simulations 64 --max-considered 16 --temperature-plies 12
  --batch-size 1024 --steps-per-iter 400
  --lr 0.002 --warmup-steps 500 --total-steps 111000
  --lr-schedule wsd --lr-horizon-steps 111000 --lr-decay-steps 11000
  --milestone-every-steps 10000
  --seed 1
)

# 三条 start-* 共用这一个函数：精度与配置都从实验名推，没有按组分叉的旋钮。
start_leg() {
  local exp="$1" devs="$2"
  bash "$REPO/scripts/train.sh" start "$exp" --precision "$(precision_for "$exp")" \
    --device "${devs%%,*}" --selfplay-devices "$devs" \
    "${COMMON[@]}" $(extra_for "$exp")
}

case "${1:-}" in
  start)
    for e in "$A_EXP" "$B_EXP"; do
      if [ -e "$RUNS/$e/ckpt/latest" ]; then
        echo "$RUNS/$e 已有 checkpoint —— 首次启动必须从零开始。" >&2
        echo "要重来请先手动清掉该目录；要续训请用 start-a / start-b。" >&2
        exit 1
      fi
    done
    "$0" start-a
    "$0" start-b
    ;;
  start-a) start_leg "$A_EXP" "$A_DEVICES" ;;
  start-b) start_leg "$B_EXP" "$B_DEVICES" ;;
  start-c) start_leg "$C_EXP" "$C_DEVICES" ;;
  stop)
    bash "$REPO/scripts/train.sh" stop "$A_EXP"
    bash "$REPO/scripts/train.sh" stop "$B_EXP"
    ;;
  stop-a) bash "$REPO/scripts/train.sh" stop "$A_EXP" ;;
  stop-b) bash "$REPO/scripts/train.sh" stop "$B_EXP" ;;
  stop-c) bash "$REPO/scripts/train.sh" stop "$C_EXP" ;;
  status)
    for e in "$A_EXP" "$B_EXP" "$C_EXP"; do
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
    read -r A A_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$A_EXP/model" "$STEP") \
      || { echo "$A_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    read -r B B_STEP < <(python3 "$REPO/tools/nearest_ckpt.py" "$RUNS/$B_EXP/model" "$STEP") \
      || { echo "$B_EXP 还没有可用的 checkpoint" >&2; exit 1; }
    D=$(( A_STEP > B_STEP ? A_STEP - B_STEP : B_STEP - A_STEP ))
    echo "目标 step $STEP -> 实际 $A_EXP@$A_STEP vs $B_EXP@$B_STEP（相差 $D 步）"
    cd "$REPO"
    # 名字直接透给 compare_nets.py 印在每一行上。
    # 这里曾经传 "$B" "$A"，同时自己 echo 一行「A=$A_EXP 为 BF16」——
    # 而 compare_nets.py 把**第一个**位置参数叫 A，也就是 $B（FP8）。
    # 两行相反的说法出现在同一屏里，得分率 0.600 会被读成「BF16 领先」，
    # 实际是 FP8 领先。**这是整个对照实验唯一要产出的那个结论**，
    # 偏偏最容易被一个位置参数的顺序悄悄翻转。现在不靠位置约定了。
    python3 tools/compare_nets.py "$B" "$A" --games 400 --simulations 64 \
            --parallel 200 --device cuda:0 \
            --name-a "$B_EXP" --name-b "$A_EXP"
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
