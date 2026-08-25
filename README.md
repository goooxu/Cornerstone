# cornerstone

从零自博弈的 Blokus Duo 模型。名字取自 Blokus 的核心规则——棋子之间只能**角接触**（corner）。

## 项目约束

这四条是项目的出发点，不是可选项：

1. **不使用任何棋谱**。训练数据全部由引擎自博弈产生；评测基线也只能由规则本身构造（随机 / 贪心 / 纯 rollout MCTS），不能拿棋谱当测试集。
2. **不复用开源 Blokus AI 的模型结构**。网络结构自研。
3. **采用 FP8 混合精度训练**：GEMM 输入是 FP8（前向 E4M3 / 反向 E5M2），
   主权重 fp32、优化器状态 fp32、累加 fp32。
4. **压满 CPU 与 GPU**：CPU 跑 MCTS 与着法生成，GPU 跑批量推理与训练，两者流水重叠。

## 规则（本项目采用的版本）

- 棋盘 14×14，每方 21 枚 polyomino（1 单格 + 1 双格 + 2 三格 + 5 四格 + 12 五格），共 89 格
- 起始格 `(4,4)` 与 `(9,9)`（0-indexed），首手必须覆盖己方起始格
- 之后每手必须与己方棋子**角相邻**，且不得与己方棋子**边相邻**；与对方棋子怎么贴都行
- 一方无合法着法时停手，另一方继续；双方都无着法则终局
- **终局后占格数多者胜，相同为和局**——没有「全部落完 +15」「最后一枚单格 +20」之类的加分项

## 快速开始

编译与训练都在装有 GPU 的开发机上、在容器里进行。

```bash
bash scripts/devbox.sh up                       # 拉起常驻开发容器（幂等）
bash scripts/devbox.sh exec bash scripts/probe.sh    # 环境报告
bash scripts/devbox.sh exec bash scripts/build.sh    # 编译 C++ 引擎
bash scripts/devbox.sh exec python3 -m pytest tests/ # 规则单测
bash scripts/devbox.sh exec python3 tools/bench_engine.py --threads 128
```

引擎交叉比对默认跑 5 万局面，拉到百万级：

```bash
bash scripts/devbox.sh exec env CORNERSTONE_XCHECK=1000000 \
    python3 -m pytest tests/test_reference.py -q
```

## 目录

```
engine/       C++ 核心（位棋盘 / 着法生成 / 对称变换 / 随机对局）+ pybind11 绑定
cornerstone/  Python 包（编译产出的 _engine*.so 落在这里）
tools/        基准与辅助脚本
scripts/      容器管理、构建、探测
tests/        规则单测与交叉比对
docs/         设计与实验记录
```

模型、replay buffer、日志都写在 repo 之外的 `runs/`，不进 git。其中模型分两种，
**各自只有一个消费方**：

```
runs/<跑名>/ckpt/    训练档：fp32 主权重 + AdamW 动量 + RNG，162.5 MiB/份 —— 只有续训读
runs/<跑名>/model/   发布包：只有权重（低精度是量化权重 + 缩放因子）—— 评测/试玩/诊断读
```

训练结束后用 `tools/export_model.py harvest` 收割：全部导出成发布包，
`ckpt/` 里只留为继续训练而存在的那两份。导出**不改变数值**（逐位验证过），
细节见 [docs/06](docs/06-低精度训练.md) 的「训练产物 → 推理部署」。

## 训练与试玩

```bash
D="bash scripts/devbox.sh exec"

$D python3 tools/train.py --smoke              # 小配置端到端自检，几分钟出结果

# FP8 与 BF16 的受控对照（v2-bf16 / v2-fp8），各占两张卡
$D bash scripts/ab_experiment.sh start
$D bash scripts/ab_experiment.sh status
$D bash scripts/ab_experiment.sh compare 20000   # 在对齐步数处头对头

# 单独跑一条
$D bash scripts/train.sh start <exp> [参数...]   # 重跑同一条命令即续训
$D bash scripts/train.sh stop <exp>

$D python3 tools/run_arena.py --games 400 --threads 128   # 规则基线阶梯 Elo
$D python3 tools/model_report.py                          # 三种精度的结构差异

# 训练结束后收割：导出发布包，只留为继续训练而存在的训练档
$D python3 tools/export_model.py harvest ../runs/<exp>          # 默认只预演
$D python3 tools/export_model.py harvest ../runs/<exp> --yes
$D bash scripts/monitor.sh 15                             # CPU/GPU 利用率

# 棋力评测：把发布包和规则基线放进同一场单循环，联合拟合 Elo。
# --simulations 0 是纯策略（一次前向、落 argmax(prior)，不做搜索）——
# 想量模型本身就用它；带搜索测的是「网络 + 搜索」的合力。
$D python3 tools/net_arena.py --rules random greedy-area corner-min \
      greedy-mobility flat-mcts-1k --nets ../runs/<exp>/model/step*.pt \
      --games 400 --simulations 0 --engine-threads 128 --out ../runs/arena/x.json
$D python3 tools/arena_best.py ../runs/arena/x.json --plot reports/图表/曲线.png
      # 回答「哪一档最强」并给出把握（自举出的最大值分布），顺带画增长曲线
$D python3 tools/plot_metrics.py --kind loss     --exp <exp>   # 损失曲线
$D python3 tools/plot_metrics.py --kind timeline --exp <exp>   # 一轮的墙钟去向

$D bash scripts/web.sh start               # 试玩服务，浏览器开 <开发机>:8080
```

**换机器 / 会话到期**：训练档与 replay 快照都写在工作目录的 `runs/` 下，
新机器上先 `scripts/probe.sh` 确认环境，再 `scripts/build.sh` 重新编译引擎，
然后 `scripts/train.sh start <exp>` 就会从上次落盘处接着跑。

## 文档

- [docs/00-总览.md](docs/00-总览.md) —— 整体设计、关键决策与里程碑
- [docs/01-环境与构建.md](docs/01-环境与构建.md) —— 目标平台、容器、构建链路
- [docs/02-引擎设计.md](docs/02-引擎设计.md) —— 位棋盘、动作空间、着法生成、对称群
- [docs/03-基线与评测.md](docs/03-基线与评测.md) —— 规则基线阶梯、arena、Elo 拟合
- [docs/04-网络与自博弈训练.md](docs/04-网络与自博弈训练.md) —— CornerNet、Gumbel-AZ、replay
- [docs/05-web试玩工具.md](docs/05-web试玩工具.md) —— 试玩工具的接口与前端
- [docs/06-低精度训练.md](docs/06-低精度训练.md) —— 各项精度的分工、为什么主权重必须 fp32、踩过的坑
- [docs/07-性能调优.md](docs/07-性能调优.md) —— 瓶颈三次转移、单卡 4.2×、每卡一进程 + DDP
- [docs/08-已知问题与后续.md](docs/08-已知问题与后续.md) —— 当前限制、优先级、经验教训
- [docs/09-棋力提升：结构轴与数据轴.md](docs/09-棋力提升：结构轴与数据轴.md) —— 七个结构实验全败、数据量 +85、步数预算重标定
- [docs/附录-arena原始结果.md](docs/附录-arena原始结果.md) —— 44 场 arena 的逐对得分率与 Elo（`runs/` 清空前存下的凭据）
- [docs/data/](docs/data/) —— 文档引用的原始评测数据（搜索阶梯、循环赛、交叉臂、多样性、churn）
- [reports/BF16训练报告.md](reports/BF16训练报告.md) —— 首个完整模型的结构、训练方法与棋力评测
- [reports/FP8训练报告.md](reports/FP8训练报告.md) —— FP8 那组，以及与 BF16 的 A/B 结论
