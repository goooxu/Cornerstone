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
D_EXP="${D_EXP:-v4-bf16-long}"

# 两组各自用哪些 GPU。默认挤在一台机器上各占两张卡；
# 有第二台机器时，在各自机器上分别 start 单组、各占四张卡：
#   机器 1:  A_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/ab_experiment.sh start-a
#   机器 2:  B_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/ab_experiment.sh start-b
# **两组的卡数必须一致**，否则每卡的并行局数不同，就多了一个变量。
A_DEVICES="${A_DEVICES:-cuda:0,cuda:1}"
B_DEVICES="${B_DEVICES:-cuda:2,cuda:3}"
# C 组（FP4）默认吃满一台机器 —— 它本来就要单独排一轮，见 docs/09 的编排。
C_DEVICES="${C_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
# D 组：从 stable 分叉点续训的加长跑，同样吃满一台机器
D_DEVICES="${D_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

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
    # 退火分叉只要终点那一份档 —— 不留里程碑、只留最近 1 份、不周期性存快照。
    # 27 个分叉全量保留约 46 GB，而磁盘只剩 100 GB 出头。
    *-a??|*-a???) echo "--keep-last 1 --milestone-every-steps 500000 --snapshot-every-iters 0" ;;
    *)  echo "" ;;
  esac
}

# 精度按实验名分派，理由与 extra_for 完全相同：守护脚本自动恢复时只转交
# 实验名与设备。
#
# **认不出来就报错退出，不猜。** 这里原先写的是 `*-fp4)` / `*-fp8)` / `*)`，
# 后缀锚定 —— 那样 `v4-fp8-long` 会落到 `*)` 上被当成 bf16，而且只在守护自动
# 恢复那一刻发生，日志上完全看不出来。改成不锚定后缀，并加一条兜底。
precision_for() {
  case "$1" in
    *fp4*)  echo "fp4" ;;
    *fp8*)  echo "fp8" ;;
    *bf16*) echo "bf16" ;;
    *)
      echo "实验名 $1 里没有 bf16/fp8/fp4，无法确定精度" >&2
      exit 1 ;;
  esac
}

# 步数预算也按实验名分派。同一个理由：守护恢复时只转交实验名。
#
# `-long` 后缀 = 从 WSD 的 stable 分叉点（`ckpt/stable.pt`）续训到 22 万步。
# 这正是 `lr_horizon_steps` 与 `total_steps` 解耦要支持的用法：horizon 一改，
# 退火点后移，原本已在退火段里的步数重新落回 stable 平顶，**不会**变成
# 一次 warm restart。
# 种子同样按实验名分派 —— 理由和上面两个一模一样。
# **这个尤其要紧**：守护恢复时若退回默认 seed 1，`v4-bf16-s2` 就变成了
# `v4-bf16` 的重复跑，而它存在的唯一理由就是换种子。日志上看不出来。
seed_for() {
  case "$1" in
    *-s2) echo 2 ;;
    *-s3) echo 3 ;;
    *)    echo 1 ;;
  esac
}

# 宽度也从实验名推，理由和精度一样：**能选就能选错**，而选错的表现是
# 「A/B 里混进了第二个变量」，没有任何症状。`v5-` 打头的一律 dim=384。
#
# 为什么加宽：v4 那轮把精度和步数两根轴都测到头了 —— 三个精度的峰值落在
# 14 Elo 之内（FP8 +13.5、BF16 0、FP4 -4.5），而每个精度练过自己的峰值都是
# 净损失（-25~-51）。两根轴都封顶，说明限制在容量。旁证：FP4 最早饱和（78k），
# 更像「容量先到顶」而不是「数值精度不够」。
#
# 为什么是 384 不是 512：512 的自博弈吞吐实测掉 2.1 倍，384 约掉 1.5 倍；
# 384 也仍然满足量化 GEMM 两维被 32 整除。
# 骨干形状。`qwen-` 打头的走 Qwen3-0.6B 形状的全注意力层（cornerstone/qwen_block.py）。
#
# **只借形状，不载任何预训练权重 —— 全部随机初始化。** 名字里带 Qwen 很容易被
# 后来的人读成「从 Qwen3-0.6B 微调来的」，而那正是本项目最防的那类误读：
# 不报错、跑得通、结论全错。
#
# 不叫 `v7-`：v4/v5/v6 是同一根 poly 骨干在改宽度和学习率，编号连着走是有含义的。
arch_for() {
  case "$1" in
    qwen-*) echo qwen ;;
    *)      echo poly ;;
  esac
}

dim_for() {
  case "$1" in
    qwen-*) echo 1024 ;;
    v6-*) echo 512 ;;
    v5-*) echo 384 ;;
    *)    echo 256 ;;
  esac
}

# 学习率同样按实验名分派 —— 而且**它必须跟着宽度走**。
#
# 这条是 v5 那轮踩出来的：dim 256->384 时我把 lr 留在 2e-3 没动，注释里还写着
# 「除 --precision 和 --dim 外逐字相同」当作严谨。实测同预算（111k 步、只差
# 宽度）加宽后**掉 77.0 ± 13.5 Elo**（1200 局，见 runs/arena_v5-width.json）。
#
# 「保持不变」对宽度不是中性的：按 µP / 宽度缩放，隐藏层的学习率大致该随宽度
# 反比缩小。384/256 = 1.5 倍宽 -> 2e-3 / 1.5 ≈ 1.3e-3。
# `-lr13` 这条跑就是去验证这个解释 —— 如果它把那 77 分追回来，说明加宽本身没错，
# 错的是没重调学习率；如果追不回来，那问题在数据量或别处。
#
# **不要后缀锚定。** 写成 `*-lr13)` 的话，从它分叉出来的 `v5-bf16-lr13-a111`
# 会落到兜底上、退火段悄悄用回 2e-3 —— 这个文件里 `precision_for` 已经栽过
# 同一个坑（`v4-fp8-long` 被当成 bf16），日志上完全看不出来。
#
# **已测出的规律：lr 大致按 1/宽度 缩放。** 基准 dim=256 -> 2e-3，于是
# 384 -> 1.3e-3、512 -> 1.0e-3。实测支撑（同预算、同精度，只差 lr）：
#
#   BF16 dim=384   lr 2e-3 -70.6 ± 10.7  ->  lr 1.3e-3 -24.2 ± 10.2   (+46.4)
#   FP4  dim=384   lr 2e-3 -158.0 ± 11.5 ->  lr 1.3e-3 +21.8 ± 11.2   (+179.8)
#
# FP4 的救回量是 BF16 的近 4 倍 —— NVFP4 数值余量最小，lr 偏高时它最先失稳。
#
# 仍然逐条写死而不按公式算：历史跑要能原样复现（`v5-bf16` 当时就是 dim=384
# 配 2e-3 跑的，那是个**错误**，但它的结论建立在那个配置上，脚本不能改写它）。
lr_for() {
  case "$1" in
    *-lr13*) echo 0.0013 ;;
    qwen-*)  echo 0.0005 ;;     # hidden=1024，按 1/宽度：2e-3 * 256/1024
    v6-*)    echo 0.001  ;;     # dim=512
    *)       echo 0.002  ;;     # dim=256 的基准；v5-* 不带 -lr13 的是未调那一版
  esac
}

# qwen 骨干的层数与头形。poly 用 COMMON 里的默认值，这里只给 qwen 那一组。
#
# **必须排在 `${COMMON[@]}` 之后展开**（见 start_leg）：COMMON 里有 `--blocks 16`，
# argparse 取最后一个同名参数。放在前面的话 28 会被静默顶成 16 ——
# 第一次起 qwen-bf16 就是这么建出了一个 256.3M 的模型（应为 445.1M），
# 而日志上除了参数量那一行没有任何异常。
shape_for() {
  case "$1" in
    qwen-*) echo "--blocks 28 --heads 16 --kv-heads 8 --head-dim 128 --intermediate 3072" ;;
    # `attn1-*` = 16 层**全部**带全局注意力（默认是每 4 层一次）。
    # dim 与层数都不动，只改注意力密度 —— 参数 14.2M -> 17.3M（注意力占比 7.4% -> 24.2%）。
    # 这是唯一没被单独测过的轴：qwen 那轮虽然是全注意力，但同时把参数翻了 31 倍、
    # 换了 Transformer 结构，三个变量捆在一起，读不出「全注意力有没有用」。
    attn1-*) echo "--attn-every 1" ;;
    # `own-*` = 打开逐格归属辅助头（预测终局时每个格归谁）。
    # 除这个头之外一切与尺子 v4-bf16-111k 逐字相同 —— 又一个单变量对照。
    own-*)   echo "--owner-head true" ;;
    *)      echo "" ;;
  esac
}

# `-aNNN` = 从 stable 段某一步分叉，在第 NNN 千步收工（退火占 horizon 的 10%，
# 与交付点 A 的 11k/111k 同比例）。用来回答「退火点设在哪一步最强」——
# WSD 的分叉点让这件事不用重训，每个只花十几分钟。
budget_for() {
  case "$1" in
    # `-x190` = 把 stable 段从第 10 万步延到 19 万，全程 2e-3 不退火
    # （horizon 22 万、退火 2 万 -> 退火点在 20 万，跑到 19 万停，始终在平顶）。
    # 每个里程碑存一份 replay 快照，供后面 `-aNNN` 分叉用。
    *-x190) echo "--total-steps 190000 --lr-horizon-steps 220000 --lr-decay-steps 20000" ;;
    # `-x100` = qwen-fp4 的长平顶。qwen-fp4 在 4 万步收工时对尺子还差 106.8，
    # 而 FP4 只花 9 小时（BF16 要 13.8）—— 值得看它练长了能到哪。
    # 退火点在 10.8 万，跑到 10 万停，全程都在平顶；每万步一份 replay 快照，
    # 供 `-a60` / `-a80` / `-a100` 分叉。**跑一条线拿三个预算的答案**，
    # 而且 6 万那个点到手的时间与专门跑 6 万几乎相同。
    *-x100) echo "--total-steps 100000 --lr-horizon-steps 120000 --lr-decay-steps 12000" ;;
    # 给上面那条长平顶配的分叉点（退火占 11%，与交付点 A 同比例）
    *-a60)  echo "--total-steps 60000 --lr-horizon-steps 60000 --lr-decay-steps 6600" ;;
    *-a80)  echo "--total-steps 80000 --lr-horizon-steps 80000 --lr-decay-steps 8800" ;;
    # `-e100` = 从零重训到 10 万步，**日程与 v4-bf16 的前 10 万步逐字相同**
    # （horizon 11.1 万、退火 1.1 万 -> 退火点在 10 万，跑到 10 万整段都在平顶）。
    # 目的只有一个：拿到 11.1 万步**以前**的分叉点 —— 原跑那些档在收割时删了，
    # 而 v4-bf16-s2 虽然档还在，却没有逐档 replay 快照。
    *-e100) echo "--total-steps 100000 --lr-horizon-steps 111000 --lr-decay-steps 11000" ;;
    # `-aNNN` = 从 stable 段第 (NNN-decay) 千步分叉，退火占 horizon 的 11%
    # （与交付点 A 的 11k/111k 同比例），在第 NNN 千步收工
    # a40 是给 qwen 那一轮配的同步数 poly 对照 —— qwen 骨干只跑 4 万步
    # （445M 参数、70 步/分，144k 要 34 小时），poly 要在同一步数上退火才可比。
    *-a40)  echo "--total-steps 40000 --lr-horizon-steps 40000 --lr-decay-steps 4400" ;;
    *-a67)  echo "--total-steps 67000 --lr-horizon-steps 67000 --lr-decay-steps 7000" ;;
    *-a78)  echo "--total-steps 78000 --lr-horizon-steps 78000 --lr-decay-steps 8000" ;;
    *-a89)  echo "--total-steps 89000 --lr-horizon-steps 89000 --lr-decay-steps 9000" ;;
    *-a100) echo "--total-steps 100000 --lr-horizon-steps 100000 --lr-decay-steps 10000" ;;
    # a111 = 从 e100 的第 10 万步分叉退火 —— **这条跑自己的交付点 A**。
    # 需要它是因为同 seed 重训并不复现原跑（多进程自博弈的 RNG 依赖调度时序、
    # CUDA 不确定、torch.compile），所以早期退火点只能和同一条跑内的 a111 比，
    # 不能直接跨到原跑的 111389 上。
    *-a111) echo "--total-steps 111000 --lr-horizon-steps 111000 --lr-decay-steps 11000" ;;
    *-a122) echo "--total-steps 122000 --lr-horizon-steps 122000 --lr-decay-steps 12000" ;;
    *-a133) echo "--total-steps 133000 --lr-horizon-steps 133000 --lr-decay-steps 13000" ;;
    *-a144) echo "--total-steps 144000 --lr-horizon-steps 144000 --lr-decay-steps 14000" ;;
    *-a155) echo "--total-steps 155000 --lr-horizon-steps 155000 --lr-decay-steps 15000" ;;
    *-a167) echo "--total-steps 167000 --lr-horizon-steps 167000 --lr-decay-steps 17000" ;;
    *-a178) echo "--total-steps 178000 --lr-horizon-steps 178000 --lr-decay-steps 18000" ;;
    *-a189) echo "--total-steps 189000 --lr-horizon-steps 189000 --lr-decay-steps 19000" ;;
    *-a200) echo "--total-steps 200000 --lr-horizon-steps 200000 --lr-decay-steps 20000" ;;
    *-a211) echo "--total-steps 211000 --lr-horizon-steps 211000 --lr-decay-steps 21000" ;;
    *-long) echo "--total-steps 220000 --lr-horizon-steps 220000 --lr-decay-steps 20000" ;;
    # `v5-*` = 加宽到 dim=384 的那一轮，跑到 14.4 万（退火 1.4 万，仍是 11%）。
    #
    # 不用 11.1 万的理由：v4 那轮实测判断点选错会让结论翻符号 —— FP8 在 111k 处
    # 是 -20.8，在 144k 处是 +13.5。加宽是纯加容量，峰值只会比 dim=256 更靠后
    # （v4 里有效容量最低的 FP4 饱和最早，78k），用 111k 量它等于低估。
    # 14.4 万覆盖了 v4 三个精度峰值的全部范围（10 万~14.4 万）。
    #
    # 不用 19 万的理由：吞吐掉 1.5 倍，19 万要 ~12 小时跨两三个 Slurm 作业；
    # 先花那么多赌一个可能在 11.1 万就见顶的模型不划算。里程碑快照留着，
    # 真在爬坡就从快照分叉续，不用重跑。
    #
    # **必须放在 `-aNNN` 之后**：否则 `v5-bf16-a111` 会先撞上这一条。
    v5-*|v6-*) echo "--total-steps 144000 --lr-horizon-steps 144000 --lr-decay-steps 14000" ;;
    # `qwen-*` = 全注意力骨干（445M）。**只跑 4 万步**，退火仍占 11%。
    #
    # 实测吞吐 18.1 局/s、70 步/分 —— 144k 要 34.3 小时、跨 5~6 个 Slurm 作业，
    # 而本轮已经在跨作业上出过三次事故（CUDA IPC、磁盘满、重复启动）。
    # 4 万步约 9.5 小时，且 v4 实测「92% 的棋力在 8 万步到手」，
    # 4 万步足以看出这根骨干救不救得回来；好就接着跑，不好就省下 25 小时。
    qwen-*) echo "--total-steps 40000 --lr-horizon-steps 40000 --lr-decay-steps 4400" ;;
    *)      echo "--total-steps 111000 --lr-horizon-steps 111000 --lr-decay-steps 11000" ;;
  esac
}

# 三组逐字共用这一份，除 --precision、设备、extra_for 外没有任何差别。
#
# **WSD 的三个字段必须待在这里，不能走环境变量**：`save_checkpoint` 虽然存了
# `asdict(cfg)`，但 `load_checkpoint` 从不读它 —— 学习率曲线的形状 100% 由本次
# 命令行决定。守护恢复时若少传一个 `--lr-horizon-steps`，horizon 会退回
# `total_steps`，曲线形状不变但**下一次改 total_steps 时会整体右移**。
#
# 步数预算不在这里，在 `budget_for()` —— 因为它要按实验名变（见上）。
# `watch_training.sh` 判「跑满没有」时**调用本脚本的 `budget` 子命令**取步数，
# 不自己解析文件：同一套规则写在两个地方，迟早对不上。
#
# 交付点 A：warmup 500 -> stable 恒 2e-3 到 100,000 -> 线性退火 11,000 步
# 到 2e-4 @ 111,000。选 10 万做 stable 的依据是实测「92% 的棋力在 8 万步到手」，
# 之后每万步的增量（+2~+12 Elo）已落在测量误差 ±9.6 之内。
COMMON=(
  --blocks 16 --attn-every 4
  --parallel-games 4096 --games-per-iter 2048
  --simulations 64 --max-considered 16 --temperature-plies 12
  --batch-size 1024 --steps-per-iter 400
  --warmup-steps 500 --lr-schedule wsd
  --milestone-every-steps 10000
)

# 三条 start-* 共用这一个函数：精度与配置都从实验名推，没有按组分叉的旋钮。
start_leg() {
  local exp="$1" devs="$2"
  bash "$REPO/scripts/train.sh" start "$exp" --precision "$(precision_for "$exp")" \
    --dim "$(dim_for "$exp")" --lr "$(lr_for "$exp")" \
    --arch "$(arch_for "$exp")" \
    --device "${devs%%,*}" --selfplay-devices "$devs" \
    "${COMMON[@]}" $(budget_for "$exp") --seed "$(seed_for "$exp")" \
    $(extra_for "$exp") $(shape_for "$exp")
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
  start-d) start_leg "$D_EXP" "$D_DEVICES" ;;
  # 守护脚本靠它取步数预算 —— 规则只有 budget_for 一处
  budget)
    [ $# -ge 2 ] || { echo "用法: $0 budget <实验名>" >&2; exit 1; }
    budget_for "$2" | sed -n 's/.*--total-steps \([0-9]\+\).*/\1/p' ;;
  stop)
    bash "$REPO/scripts/train.sh" stop "$A_EXP"
    bash "$REPO/scripts/train.sh" stop "$B_EXP"
    ;;
  stop-a) bash "$REPO/scripts/train.sh" stop "$A_EXP" ;;
  stop-b) bash "$REPO/scripts/train.sh" stop "$B_EXP" ;;
  stop-c) bash "$REPO/scripts/train.sh" stop "$C_EXP" ;;
  stop-d) bash "$REPO/scripts/train.sh" stop "$D_EXP" ;;
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
