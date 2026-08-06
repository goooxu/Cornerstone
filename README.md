# cornerstone

从零自博弈的 Blokus Duo 模型。名字取自 Blokus 的核心规则——棋子之间只能**角接触**（corner）。

## 项目约束

这四条是项目的出发点，不是可选项：

1. **不使用任何棋谱**。训练数据全部由引擎自博弈产生；评测基线也只能由规则本身构造（随机 / 贪心 / 纯 rollout MCTS），不能拿棋谱当测试集。
2. **不复用开源 Blokus AI 的模型结构**。网络结构自研。
3. **模型主权重是 FP8**，采用混合精度训练。
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

模型 checkpoint、replay buffer、日志都写在 repo 之外的 `runs/`，不进 git。

## 文档

- [docs/00-总览.md](docs/00-总览.md) —— 整体设计与里程碑
- [docs/01-环境与构建.md](docs/01-环境与构建.md) —— 目标平台、容器、构建链路
- [docs/02-引擎设计.md](docs/02-引擎设计.md) —— 位棋盘、动作空间、着法生成、对称群
