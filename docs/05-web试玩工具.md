# Web 试玩工具

```bash
bash scripts/devbox.sh exec bash scripts/web.sh start        # 自动挑最新的 checkpoint
bash scripts/devbox.sh exec bash scripts/web.sh start <ckpt> # 指定 checkpoint
bash scripts/devbox.sh exec bash scripts/web.sh status|stop|restart
```

常驻服务，绑 `0.0.0.0:8080`，内网直接访问，无鉴权。
没有 checkpoint 时 AI 自动退回规则基线，所以训练还没出结果也能先玩。

## 一条硬规则：合法性判断不在前端重写

后端把 `legal_actions` 一次性发给前端，前端只做「这个动作在不在集合里」的查表。
落子合法性、停手、终局、计分全部走 C++ 引擎——和训练、评测用的是同一份实现。

在 JS 里重写一遍规则是最容易埋雷的做法：两边的判定迟早会分叉，
而且分叉之后表现为「网页上能下的棋，模型认为非法」这种极难定位的现象。

## 后端

`web/server.py`，FastAPI + uvicorn。

| 接口 | 作用 |
|---|---|
| `GET /api/meta` | 棋子与全部朝向的格坐标、起始格、难度档、AI 后端名 |
| `POST /api/new` | 开新局，选执先/执后与难度 |
| `POST /api/move` | 人类落子（非法着法返回 400） |
| `POST /api/ai` | AI 走一步，附带分析 |
| `GET /api/analysis` | 对当前局面跑一次搜索，返回胜率/top 着法/热力图 |
| `POST /api/undo` | 悔棋，退到轮到人类为止 |

单局面搜索复用自博弈引擎：`SelfPlayEngine(num_games=1)` + `set_position(history)`
+ 跑完模拟 + `root_info()`，不调 `advance()`。没有为试玩工具单写一套搜索。

### 一个踩过的坑：Pydantic 模型必须放模块级

`server.py` 开头有 `from __future__ import annotations`，注解全变成字符串。
FastAPI 要靠端点函数的 `__globals__` 去解析这些字符串——
把请求模型定义在 `build_app()` 内部就解析不到，参数会被当成 **query** 而不是 body，
所有 POST 直接 422。模型定义在模块级才行。

## 前端

`web/static/`，单页 + Canvas，无构建步骤。

- 14×14 棋盘，起始格画圈，双方用不同颜色
- 棋子托盘：21 枚，用掉的置灰；选中后列出该枚棋子的**全部朝向缩略图**，点选即可
  （朝向已经去重，所以「旋转/翻转」本质就是在这个列表里循环；`R` 键下一形态、
  `F` 键跳到镜像组、`Esc` 取消选择）
- 悬停预览：棋子包围盒左上角落在悬停格，合法为绿、非法为红
- **当前朝向的全部合法锚点用小圆点标出**，不用靠猜
- AI 策略热力图：把每个合法着法的概率累加到它覆盖的格子上，看 AI 想往哪下
- 胜率条、top 着法列表（悬停高亮该着法的落点）
- 悔棋、AI 自动应手开关、对局导出为 JSON

## 难度与实测延迟

难度映射到 MCTS 模拟数（无网络时映射到规则基线阶梯）：

| 难度 | 模拟数 | 无网络时的基线 | 实测每手用时 |
|---|---:|---|---:|
| 简单 | 16 | greedy-area | 0.08 s |
| 普通 | 64 | greedy-mobility | 0.30 s |
| 困难 | 256 | flat-mcts-1k | 1.13 s |
| 极难 | 800 | flat-mcts-4k | 4.60 s |

延迟偏高的原因是**单局面搜索的批大小恒为 1**：每次模拟都要单独跑一次网络前向，
而单次前向约 10 ms 且与批大小几乎无关（见 [04](04-网络与自博弈训练.md) 末尾）。
自博弈靠「同时开几百局」攒批，单人试玩没有这个条件。

真要压下来，得在单棵树里做 virtual loss 并行展开若干叶子。对一个人机试玩工具来说
4.6 秒还在可接受范围，暂时不做——如果后面觉得慢，这是明确的优化点。
