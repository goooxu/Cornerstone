#!/usr/bin/env python3
"""Web 试玩工具后端。

    python3 web/server.py                          # 默认对手 greedy-mobility
    python3 web/server.py --model ../runs/v4-fp8/model/step00111391.pt

规则判定全部走 C++ 引擎（`cornerstone._engine`），和训练用的是同一份实现 ——
前端只负责画和收集点击，任何合法性判断都不在 JS 里重写，否则迟早两边对不上。

对手可以在界面上随时换，两类：

  * 规则基线 `rule:<名字>`（greedy-area / corner-min / greedy-mobility）
  * 网络 `net:<跑名>/<文件>`，外加 `net:<跑名>/latest` —— 后者**跟随训练**，
    每次用的时候才去解析，所以边训边打能一直对上最新的权重
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cornerstone as cs                       # noqa: E402
from cornerstone import _engine as E           # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RUNS_DIR = os.path.join(os.path.dirname(REPO), "runs")
# 网络后端只从发布包目录里找。训练档（ckpt/）带着 AdamW 动量与 RNG，
# 那是续训的东西；web 只做推理，读发布包。见 cornerstone/export.py
MODEL_SUBDIR = "model"
# 尺子目录：三个精度各自的峰值模型，刻度已量定，是下拉里默认该选的那几个。
RULER_DIR = "_ruler"
# 标签里带上 Elo，省得还要回去翻 README。基准是 BF16 峰值 = 0，
# 数字来自六场独立 arena 的合并拟合（关键格 23,200 局）。
RULER_LABELS = {
    "v4-bf16-111k.pt": "BF16 峰值 @111k（基准 0）",
    "v4-fp8-144k.pt":  "FP8 峰值 @144k（+13.5 ± 3.8）",
    "v4-fp4-100k.pt":  "FP4 峰值 @100k（−4.5 ± 6.8）",
}

# 界面上直接选模拟数，不再套「简单/普通/困难」这层名字 ——
# 两个座位可以各选各的，用来比「同一个网络多搜一倍值多少棋力」这种事，
# 名字反而挡着看不清实际预算。
# 0 = 纯策略：直接取网络先验的 argmax，不看搜索结果，确定性。
# 见 NetBrain._choose_by_policy —— 它**不能**用「把模拟数调到很小」来近似，
# 少量模拟反而是随机性最大的情形（改进策略里 sigma≈51 会让一次随机采样
# 到的 q 盖过整个 log 先验）。这一点起初判断错过，详见 docs/05。
PURE_POLICY = 0
# **界面上不提供纯策略**（API 仍然接受，见 validate_sims —— net_arena 和 diag
# 工具还要用它做研究口径）。
#
# 理由是它会给出**相反**的排名：同一批 checkpoint，纯策略口径下「训练越久越差」
# （终点比峰值低 18.9~22.2），带 64 次搜索重打则终点最强，直接头对头从 0.527
# 翻成 0.485（docs/08）。机制是后期的进步主要存进价值头，而纯策略每手只做一次
# 前向、落 argmax(prior)，把价值头整个丢掉。
#
# 摆在试玩界面里，等于请人用一把已知会读反的尺子去比较模型 —— 而界面上
# 看不出任何异常。研究要用就走命令行，那里有文档说明口径。
SIM_CHOICES = [64, 256, 800]
# 「连续对战」的可选局数。这只是界面上的一个选项：一局下完自动开下一局，
# 棋盘照常逐手显示，统计在前端累加 —— 服务端不需要知道有这回事。
SERIES_COUNTS = [1, 100, 200, 400]
DEFAULT_SIMS = 64
# 上限不是审美问题：单局面搜索的批大小恒为 1，模拟数直接线性折算成等待时间，
# 放开了就能让一个请求把服务占住好几分钟。
MAX_SIMS = 2000

# 界面上可选的规则基线。给这三个是因为它们各代表一种打法：
# 只看棋子大小 / 只堵对方 / 调好权重的综合版。强弱和设计依据见 docs/03。
RULE_BACKENDS = ["greedy-area", "corner-min", "greedy-mobility"]

DEFAULT_BACKEND = "rule:greedy-mobility"

# 随机开局的手数。**只在连打多局时注入**，由调用方传 `opening_seed` 决定 ——
# 不传就不注入。
#
# 为什么多局要有：自博弈的开局会塌到只剩一种首手（docs/08 实测：第 8 万步之后
# 512 局里每局都走同一手）。不注入的话，两个网络打 400 局，量到的是「这一条
# 特定开局线上谁强」，而不是整体棋力；纯策略档更是确定性的，400 局全同。
# 手数取 2，与 `play_pair` 的默认值一致 —— 三把尺子的 Elo 刻度就是这么量的。
#
# 为什么单局不注入：单局的用途是**看棋**，而「网络自己开什么」正是想看的东西
# （开局塌缩本身就是个观察点）；随机掉前两手反而把它盖住了。而且单局也谈不上
# 「双方各执先一次」—— 成对的理由只在多局时成立。
#
# 另外**人机对局一律不注入**：随机掉人的前两手没有意义。
OPENING_PLIES = 2


# --------------------------------------------------------------------- 后端发现

def asset_version() -> str:
    """前端静态资源的版本号：内容变了它就变。

    用 mtime 而不是内容哈希 —— 文件就两个、每次请求都要算，
    mtime 足够区分且不用读盘内容。
    """
    h = hashlib.md5()
    for name in ("app.js", "style.css", "index.html"):
        try:
            h.update(f"{name}:{os.path.getmtime(os.path.join(STATIC, name))};".encode())
        except OSError:
            h.update(f"{name}:missing;".encode())
    return h.hexdigest()[:10]


def _step_of(fname: str) -> int:
    m = re.search(r"step(\d+)", fname)
    return int(m.group(1)) if m else -1


def resolve_backend(backend: str) -> tuple[str, str]:
    """把后端 ID 解析成 (kind, 目标)。kind 是 'rule' 或 'net'。

    **网络后端指向 `runs/<跑名>/model/`，也就是发布包，不是训练档**。
    训练档（`ckpt/`）只有续训读，收割之后那里也只剩一两份了。

    `net:<跑名>/latest` 是**每次调用都重新解析**的，不缓存 ——
    正在训练的跑每隔几分钟就多一份发布包，缓存了就永远停在选中那一刻。
    """
    if backend.startswith("rule:"):
        name = backend[5:]
        if name not in RULE_BACKENDS:
            raise ValueError(f"未知的规则基线: {name}")
        return "rule", name

    if not backend.startswith("net:"):
        raise ValueError(f"无法识别的后端: {backend}")
    rel = backend[4:]
    run, _, fname = rel.partition("/")

    # 尺子（三个精度各自的峰值）平铺在 runs/_ruler/ 下，没有 model/ 子目录，
    # 文件名也刻意不带 `step`（否则 net_arena 会拿 step(\d+) 命名参赛者，
    # 和被测跑里同步数的档撞名后被悄悄合并成同一个）。所以单独一条路径。
    if run == RULER_DIR:
        path = os.path.join(RUNS_DIR, RULER_DIR, os.path.basename(fname))
        if not os.path.exists(path):
            raise ValueError(f"尺子不存在：{os.path.basename(fname)}")
        return "net", path

    model_dir = os.path.join(RUNS_DIR, run, MODEL_SUBDIR)
    if not os.path.isdir(model_dir):
        raise ValueError(f"找不到 {run} 的发布包目录（先跑 tools/export_model.py harvest）")

    if fname in ("latest", ""):
        # 发布包目录里没有 latest 指针文件（那是训练档的东西），直接取步数最大的那份
        found = sorted(glob.glob(os.path.join(model_dir, "step*.pt")), key=_step_of)
        if not found:
            raise ValueError(f"{run} 还没有发布包")
        return "net", found[-1]

    path = os.path.join(model_dir, os.path.basename(fname))
    if not os.path.exists(path):
        # 发布包会被轮换/收割，界面上列出来的那一刻还在、点下去可能就没了
        raise ValueError(f"发布包已不存在：{os.path.basename(fname)}")
    return "net", path


def discover_backends() -> list[dict]:
    """列出当前可选的对手。每次调用都重新扫盘，好让新导出的发布包能出现。

    **尺子排在最前面。** 实验跑攒到几十条之后，每条又有十几个里程碑，
    下拉里几百项，想找「当前最强」反而找不到。尺子是三个精度各自的峰值、
    刻度已经量定（见 `runs/_ruler/README.md`），是默认该选的那几个。
    """
    out: list[dict] = []

    for path in sorted(glob.glob(os.path.join(RUNS_DIR, RULER_DIR, "*.pt"))):
        name = os.path.basename(path)
        out.append({"id": f"net:{RULER_DIR}/{name}",
                    "label": RULER_LABELS.get(name, name[:-3]),
                    "group": "尺子（各精度峰值）", "kind": "net"})

    out += [{"id": f"rule:{n}", "label": n, "group": "规则基线", "kind": "rule"}
            for n in RULE_BACKENDS]

    for model_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*", MODEL_SUBDIR))):
        run = os.path.basename(os.path.dirname(model_dir))
        cks = sorted(glob.glob(os.path.join(model_dir, "step*.pt")), key=_step_of)
        if not cks:
            continue
        out.append({"id": f"net:{run}/latest",
                    "label": f"最新（跟随训练，当前 step {_step_of(os.path.basename(cks[-1])):,}）",
                    "group": run, "kind": "net"})
        for ck in reversed(cks):
            b = os.path.basename(ck)
            out.append({"id": f"net:{run}/{b}", "label": f"step {_step_of(b):,}",
                        "group": run, "kind": "net"})
    return out


# --------------------------------------------------------------------------- AI

class RuleBrain:
    """规则基线对手。无网络，也就没有分析面板。"""

    has_net = False

    def __init__(self, name: str):
        from cornerstone.arena import BASELINES
        self.name = name
        self.label = f"规则基线 {name}"
        self.cfg = BASELINES[name]

    def analyse(self, history: list[int], sims: int) -> None:
        return None

    def choose(self, board: cs.Board, history: list[int], sims: int) -> tuple[int, None]:
        # 同分随机打散：确定性地取第一个会让同一个基线每局走出完全一样的棋
        return E.select_move(board, self.cfg, int(time.time_ns() & 0xFFFFFFFF)), None


class NetBrain:
    """单个发布包的推理后端。"""

    has_net = True

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        from cornerstone.model import load_checkpoint

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model, step = load_checkpoint(path, device)
        self.torch = torch
        self.model = model
        self.device = torch.device(device)
        self.step = step
        self.path = path
        # 精度写进标签：v4-bf16 / v4-fp8 / v4-fp4 三组的发布包混在一个
        # 下拉里，光看 step 分不出是哪一条。这个值来自发布包自带的
        # model_config，不是从跑名猜的 —— 跑名可以随便起，模型配置不会骗人。
        self.precision = str(getattr(model.cfg, "precision", "bf16")).upper()
        self.label = (f"CornerNet {model.num_params()/1e6:.1f}M · {self.precision}"
                      f" · step {step}")
        self.engines: dict[int, E.SelfPlayEngine] = {}
        # SelfPlayEngine 是有状态的：set_position 之后要 prepare/feed 到底。
        # FastAPI 的同步 handler 跑在线程池里，两个人同时点「让 AI 走」
        # 就会踩同一棵树 —— 表现是着法乱掉或直接崩，且难复现。
        self.lock = threading.Lock()

    def _engine(self, sims: int) -> E.SelfPlayEngine:
        if sims not in self.engines:
            self.engines[sims] = E.SelfPlayEngine(
                1, E.MctsConfig(simulations=sims, max_considered=32, temperature_plies=0), 0)
        return self.engines[sims]

    def analyse(self, history: list[int], sims: int) -> dict | None:
        """跑一次搜索，返回根节点的改进策略与估值。"""
        torch = self.torch
        with self.lock:
            eng = self._engine(sims)
            eng.set_position(history)

            planes = np.zeros((1, cs.NUM_PLANES, cs.BOARD_N, cs.BOARD_N), dtype=np.float32)
            scal = np.zeros((1, cs.NUM_SCALARS), dtype=np.float32)
            guard = 0
            while True:
                guard += 1
                if guard > 100000:
                    break
                n = eng.prepare(planes, scal)
                if n == 0:
                    break
                with torch.no_grad(), torch.autocast(
                        self.device.type, dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda"):
                    pol, wdl, _ = self.model(
                        torch.from_numpy(planes[:n]).to(self.device),
                        torch.from_numpy(scal[:n]).to(self.device))
                eng.feed(pol.float().cpu().numpy(), wdl.float().softmax(-1).cpu().numpy())

            info = eng.root_info()
        return info if info["ready"] else None

    def choose(self, board: cs.Board, history: list[int], sims: int) -> tuple[int, dict | None]:
        if sims <= PURE_POLICY:
            return self._choose_by_policy(history)
        info = self.analyse(history, sims)
        if info is None:
            raise RuntimeError("搜索没有产出可用的根节点信息")
        best = int(np.argmax(info["probs"]))
        return int(info["actions"][best]), info

    def _choose_by_policy(self, history: list[int]) -> tuple[int, dict]:
        """纯策略：直接取网络先验的 argmax，完全不看搜索结果。

        `root.prior` 是网络 logits 在合法着法上的 softmax，节点展开时写一次、
        之后再不改动，既没有 Dirichlet 也没有 Gumbel —— 所以这条路是**确定性**的。

        对比之下，改进策略 `probs` 里有一项 `sigma * q`，`sigma ≈ 51`，
        少量模拟时那一个随机采样到的 q 会盖过整个 log 先验，反而最不稳定
        （见 docs/05）。所以「纯策略」不能靠把模拟数调小来近似，只能走这里。

        引擎至少要跑 1 次模拟才会展开根节点、才有 prior 可读；
        多出来的那次子节点评估不影响 prior，只是多约 20ms。
        """
        info = self.analyse(history, 1)
        if info is None:
            raise RuntimeError("网络没有给出可用的根节点先验")
        priors = np.asarray(info["priors"], dtype=np.float64)
        best = int(np.argmax(priors))
        shown = dict(info)
        # 面板显示策略本身；访问数在这条路上没有意义，清零免得被当成搜索结果
        shown["probs"] = priors
        shown["visits"] = np.zeros(len(priors), dtype=np.int64)
        return int(info["actions"][best]), shown


class BrainPool:
    """按需加载并缓存后端。网络按发布包路径缓存，LRU 淘汰。

    不缓存的话每走一步都要重新 torch.load 一份发布包（bf16 约 28 MB、
    fp8 约 17 MB、fp4 约 11 MB —— 收割之前这里是 162.5 MB 的训练档）；
    无上限地缓存又会在来回切模型时把显存吃光。
    """

    def __init__(self, device: str = "cuda", max_models: int = 3):
        self.device = device
        self.max_models = max_models
        self.rules: dict[str, RuleBrain] = {}
        self.nets: OrderedDict[str, NetBrain] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, backend: str):
        kind, target = resolve_backend(backend)
        if kind == "rule":
            with self.lock:
                if target not in self.rules:
                    self.rules[target] = RuleBrain(target)
                return self.rules[target]

        with self.lock:
            hit = self.nets.get(target)
            if hit is not None:
                self.nets.move_to_end(target)
                return hit
        # 加载放在锁外：读盘 + 建模型要几秒，占着锁会把所有请求堵住
        brain = NetBrain(target, self.device)
        with self.lock:
            if target in self.nets:                 # 有人抢先加载好了，用他那份
                self.nets.move_to_end(target)
                return self.nets[target]
            self.nets[target] = brain
            while len(self.nets) > self.max_models:
                _, victim = self.nets.popitem(last=False)
                del victim
            return brain


# ----------------------------------------------------------------------- 会话

@dataclass
class Session:
    """一局对局。

    两个座位各自绑一个后端，`None` 表示这个座位由人来下。
    「人机」和「AI 对战」因此不是两种模式，而是同一套东西的两种填法 ——
    少一个模式开关，就少一堆「模式与实际配置不一致」的状态。
    """

    board: cs.Board = field(default_factory=cs.Board)
    history: list[int] = field(default_factory=list)
    players: list[str | None] = field(default_factory=lambda: [None, DEFAULT_BACKEND])
    # 每个座位各自的模拟数。规则基线不搜索，这一项对它无效。
    sims: list[int] = field(default_factory=lambda: [DEFAULT_SIMS, DEFAULT_SIMS])
    created: float = field(default_factory=time.time)
    # 注入的随机开局有几手。**它不是任何人走的手**，所以：
    #   * 判「对局是否已开始」要跳过它（否则一出生配置就被锁死）
    #   * 悔棋不能退进去（那不是谁的失误，退了也没法重来同一个开局）
    opening_len: int = 0

    @property
    def human_player(self) -> int:
        """人坐哪个座位；两个座位都是 AI 时返回 -1。"""
        for i, p in enumerate(self.players):
            if p is None:
                return i
        return -1

    def replay(self, actions: list[int]) -> None:
        self.board = cs.Board()
        self.history = []
        for a in actions:
            self.board.play(int(a))
            self.history.append(int(a))


SESSIONS: dict[str, Session] = {}
MAX_SESSIONS = 500


def state_of(s: Session, analysis: dict | None = None) -> dict:
    b = s.board
    grid = [[-1] * cs.BOARD_N for _ in range(cs.BOARD_N)]
    for p in (0, 1):
        for r, c in b.occupancy_cells(p):
            grid[r][c] = p

    legal = b.legal_moves().tolist() if not b.terminal else []
    out = {
        "grid": grid,
        "current_player": b.current_player,
        "human_player": s.human_player,
        "terminal": bool(b.terminal),
        "scores": list(b.scores()),
        "ply": b.ply,
        "history": s.history,
        "legal_actions": legal,
        "remaining": [[bool(b.piece_remaining(p, i)) for i in range(cs.NUM_PIECES)]
                      for p in (0, 1)],
        "anchors": [b.anchor_cells(p) for p in (0, 1)],
        # result 相对人类；AI 对战时没有「人类」，就相对先手报，
        # 前端据 human_player == -1 换一种说法渲染
        "result": b.result_for(max(s.human_player, 0)) if b.terminal else None,
        "sims": list(s.sims),
        "players": list(s.players),
    }
    if analysis is not None:
        out["analysis"] = analysis
    return out


def analysis_payload(info: dict | None, board: cs.Board) -> dict | None:
    """把根节点信息整理成前端要的形状：胜率 + top 着法。"""
    if info is None:
        return None
    actions = np.asarray(info["actions"])
    probs = np.asarray(info["probs"])
    order = np.argsort(-probs)[:8]

    top = []
    for i in order:
        d = cs.decode_action(int(actions[i]))
        top.append({
            "action": int(actions[i]),
            "piece": d["piece_name"],
            "prob": float(probs[i]),
            "visits": int(info["visits"][i]),
        })
    return {
        "value": float(info["value"]),
        "win_rate": float((info["value"] + 1) / 2),
        "top_moves": top,
    }


# ------------------------------------------------------------------------ HTTP

try:
    from pydantic import BaseModel
except ImportError:                      # 只在没装 fastapi 的环境里导入本模块时才会走到
    BaseModel = object


# 这三个模型必须放在**模块级**：本文件开头有 from __future__ import annotations，
# 注解都成了字符串，FastAPI 要靠函数的 __globals__ 去解析。
# 定义在 build_app 内部的话解析不到，参数会被当成 query 而不是 body（422）。
def apply_random_opening(s: "Session", seed: int | None) -> None:
    """在空盘上走 OPENING_PLIES 手均匀随机的合法着法。

    种子决定这几手走什么，所以**同一个种子跑两遍会得到逐手相同的开局** ——
    连续对战靠这一点做成对：一对里两局传同一个种子、先后手互换，
    先手优势对双方各记一半。引擎侧的等价实现见 `engine/src/mcts.cpp`
    里 `paired_second` 那段（它是存/恢复 rng，效果一样）。

    走的是**均匀随机**而不是网络自己的先验：目的正是绕开自博弈那个
    「所有局都走同一手」的漏斗（docs/08）。
    """
    rng = random.Random(seed)
    for _ in range(OPENING_PLIES):
        if s.board.terminal:
            break
        mv = s.board.legal_moves().tolist()
        if not mv:
            break
        a = int(mv[rng.randrange(len(mv))])
        s.board.play(a)
        s.history.append(a)
    s.opening_len = len(s.history)


def new_game_for_test(players, opening_seed):
    """按 new_game 的规则造一局，返回被注入的开局着法。只给测试用。"""
    s = Session(players=list(players))
    if opening_seed is not None and all(p is not None for p in players):
        apply_random_opening(s, opening_seed)
    assert s.opening_len == len(s.history)
    return list(s.history)


class NewGame(BaseModel):
    # players[i] = 该座位的后端 ID，None 表示人来下。
    # 不给就退回 human_player + backend 这组旧参数。
    players: list[str | None] | None = None
    sims: list[int] | None = None
    human_player: int = 0
    backend: str | None = None
    # 随机开局的种子。**不传就不注入**（单局观战走这条）。
    # 连打时同一对的两局要传同一个 —— 于是开局逐手相同而先后手已翻，
    # 正是 arena 的成对口径（见 engine/src/mcts.cpp 里 paired_second 那段）。
    opening_seed: int | None = None


class MoveReq(BaseModel):
    sid: str
    action: int


class SidReq(BaseModel):
    sid: str


class BackendReq(BaseModel):
    sid: str
    players: list[str | None] | None = None
    sims: list[int] | None = None


def build_app(pool: BrainPool, default_backend: str = DEFAULT_BACKEND):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="cornerstone 试玩")

    def get(sid: str) -> Session:
        s = SESSIONS.get(sid)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请开新局")
        return s

    def load_brain(backend: str):
        """加载后端。失败要给出**能看懂**的 400，而不是 500。

        发布包会被收割/轮换，界面上列出来的那一刻还在、
        点下去可能就没了 —— 这不是 bug，是正常状态，得让用户知道换一个就行。
        """
        try:
            return pool.get(backend)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:                      # torch.load 失败等
            raise HTTPException(400, f"后端 {backend} 加载失败：{e}")

    def brain_to_move(s: Session):
        """当前该走棋的那个座位的后端。轮到人类则 400。"""
        backend = s.players[s.board.current_player]
        if backend is None:
            raise HTTPException(400, "现在轮到人类走，不该让 AI 落子")
        return load_brain(backend)

    def validate_players(players: list[str | None]) -> list[str | None]:
        if len(players) != 2:
            raise HTTPException(400, "players 必须是两项")
        for p in players:
            if p is None:
                continue
            try:
                resolve_backend(p)
            except ValueError as e:
                raise HTTPException(400, str(e))
        return list(players)

    def validate_sims(sims: list[int]) -> list[int]:
        if len(sims) != 2:
            raise HTTPException(400, "sims 必须是两项")
        out = []
        for v in sims:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"模拟数必须是整数：{v!r}")
            if not PURE_POLICY <= n <= MAX_SIMS:
                raise HTTPException(400, f"模拟数要在 {PURE_POLICY}..{MAX_SIMS} 之间"
                                         f"（0 表示纯策略），收到 {n}")
            out.append(n)
        return out

    def sims_for(s: Session, player: int) -> int:
        return s.sims[player]

    # 前端资源一律要求**每次回源校验**。
    #
    # 页面和脚本是分开缓存的，改代码时很容易出现「新的 index.html 配旧的
    # app.js」：旧脚本去绑一个已经不存在的按钮，抛
    # `Cannot set properties of null`，而这一抛整个模块就停了 ——
    # 连填充下拉框的启动代码都不会执行，表现却是「某个控件用不了」。
    # 实际就这么坑过一次，查了很久才发现是缓存。
    #
    # no-cache 不是「不缓存」，是「用之前先校验」。StaticFiles 会带
    # ETag/Last-Modified，没变就是一个 304，代价可以忽略。
    @app.middleware("http")
    async def revalidate_static(request, call_next):
        resp = await call_next(request)
        p = request.url.path
        if p == "/" or p.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/")
    def index():
        """把 index.html 里的静态资源链接加上版本号再发出去。

        光靠 no-cache 不够：那只在浏览器**真的发请求**时才起作用，
        它若认为手上那份还新鲜，可以连问都不问。而页面与脚本是分开缓存的，
        于是出现过「新 HTML 配旧 JS/CSS」——旧 JS 去绑不存在的元素直接
        把整个模块带停，旧 CSS 则让两个图标一起显示、暂停按钮藏不住。

        加了版本号之后，内容一变 URL 就变，浏览器手上那份**根本对不上**，
        没有「要不要复用」这个问题。index.html 自身仍是 no-cache。
        """
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        v = asset_version()
        for name in ("app.js", "style.css"):
            html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/api/meta")
    def meta():
        pieces = []
        for p in range(cs.NUM_PIECES):
            oris = cs.piece_orientations(p)
            pieces.append({
                "id": p,
                "name": cs.piece_names()[p],
                "size": cs.piece_sizes()[p],
                "orientations": [{"id": int(o), "cells": cs.orientation_info(o)["cells"],
                                  "h": cs.orientation_info(o)["height"],
                                  "w": cs.orientation_info(o)["width"]}
                                 for o in oris],
            })
        return {
            "board_n": cs.BOARD_N,
            "num_cells": cs.NUM_CELLS,
            "start_cells": [list(x) for x in cs.START_CELLS],
            "pieces": pieces,
            "sim_choices": SIM_CHOICES,
            "series_counts": SERIES_COUNTS,
            "default_sims": DEFAULT_SIMS,
            "max_sims": MAX_SIMS,
            "default_backend": default_backend,
        }

    @app.get("/api/backends")
    def backends():
        """每次都重新扫盘：训练在跑，发布包是会变多的。"""
        return {"backends": discover_backends(), "default": default_backend}

    @app.post("/api/backend")
    def set_backend(req: BackendReq):
        """改这一局的双方或模拟数。**只在开局前或终局后允许。**

        中途换引擎会让「这一局是谁对谁」变得没法陈述：棋盘上一半的手是
        A 走的、一半是 B 走的，最后那个比分不属于任何一对组合。
        这道拦截必须在服务端 —— 界面把下拉置灰只是提示，
        请求照样可以直接发过来。
        """
        s = get(req.sid)
        if (req.players is not None or req.sims is not None) \
                and len(s.history) > s.opening_len and not s.board.terminal:
            raise HTTPException(400, "对局进行中，不能改双方或模拟数；请先开新局")
        if req.players is not None:
            s.players = validate_players(req.players)
        if req.sims is not None:
            s.sims = validate_sims(req.sims)
        return {"state": state_of(s), "labels": seat_labels(s)}

    def seat_labels(s: Session) -> list[str]:
        out = []
        for p in s.players:
            if p is None:
                out.append("人类")
            else:
                try:
                    out.append(pool.get(p).label)
                except Exception:
                    out.append(p)          # 只是显示用，加载不了也别让整个请求挂掉
        return out

    @app.post("/api/new")
    def new_game(req: NewGame):
        if len(SESSIONS) >= MAX_SESSIONS:      # 简单的容量保护，删最旧的
            oldest = min(SESSIONS, key=lambda k: SESSIONS[k].created)
            SESSIONS.pop(oldest, None)
        if req.players is not None:
            players = validate_players(req.players)
        else:                                  # 旧参数形式：human_player + backend
            human = int(req.human_player) & 1
            backend = req.backend or default_backend
            players = [None, None]
            players[1 - human] = validate_players([backend, None])[0]
        sid = uuid.uuid4().hex[:16]
        sims = validate_sims(req.sims) if req.sims is not None \
            else [DEFAULT_SIMS, DEFAULT_SIMS]
        s = Session(players=players, sims=sims)
        # 不传种子 = 不注入（单局观战、人机对局）
        if req.opening_seed is not None and all(p is not None for p in players):
            apply_random_opening(s, req.opening_seed)
        SESSIONS[sid] = s
        return {"sid": sid, "state": state_of(s), "labels": seat_labels(s)}

    @app.get("/api/state")
    def state(sid: str):
        return state_of(get(sid))

    @app.post("/api/move")
    def move(req: MoveReq):
        s = get(req.sid)
        if s.board.terminal:
            raise HTTPException(400, "本局已经结束")
        if not s.board.is_legal(req.action):
            raise HTTPException(400, "非法着法")
        s.board.play(req.action)
        s.history.append(int(req.action))
        return state_of(s)

    @app.post("/api/ai")
    def ai_move(req: SidReq):
        s = get(req.sid)
        if s.board.terminal:
            raise HTTPException(400, "本局已经结束")
        mover = s.board.current_player
        b = brain_to_move(s)
        t0 = time.perf_counter()
        action, info = b.choose(s.board, s.history, sims_for(s, mover))
        if not s.board.is_legal(int(action)):
            # 后端选出非法着法说明局面与搜索树对不上，继续走会把棋盘弄脏
            raise HTTPException(500, f"后端 {s.players[mover]} 给出了非法着法 {action}")
        s.board.play(int(action))
        s.history.append(int(action))
        out = state_of(s, analysis_payload(info, s.board))
        out["ai_move"] = int(action)
        out["ai_player"] = mover
        out["ai_seconds"] = round(time.perf_counter() - t0, 3)
        out["ai_label"] = b.label
        out["labels"] = seat_labels(s)
        return out

    @app.post("/api/undo")
    def undo(req: SidReq):
        s = get(req.sid)
        hist = list(s.history)
        if s.human_player < 0:
            # AI 对战没有「轮到人类」这回事。照原逻辑找下去会一路 pop 到空棋盘，
            # 看起来像「悔棋把整局都撤了」。这里就退一手。
            if len(hist) > s.opening_len:
                hist.pop()
        else:
            # 退回到轮到人类且至少撤掉一手为止（注入的开局是地板，不退进去）
            while len(hist) > s.opening_len:
                hist.pop()
                probe = cs.Board()
                for a in hist:
                    probe.play(a)
                if probe.terminal:
                    continue
                if probe.current_player == s.human_player:
                    break
        s.replay(hist)
        return state_of(s)

    if os.path.isdir(STATIC):
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--checkpoint", dest="model", default=None,
                    help="默认对手用哪个**发布包**（runs/<跑名>/model/…）；"
                         "界面上仍可随时换。旧名 --checkpoint 仍可用")
    ap.add_argument("--backend", default=None,
                    help="默认后端 ID，如 rule:greedy-mobility 或 net:v2-fp8/latest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-models", type=int, default=3, help="同时缓存几个网络")
    args = ap.parse_args()

    import uvicorn
    default = args.backend
    if default is None and args.model:
        # 把 --model 折成后端 ID，好和界面上的选项对得上
        ck = os.path.abspath(args.model)
        run = os.path.basename(os.path.dirname(os.path.dirname(ck)))
        default = f"net:{run}/{os.path.basename(ck)}"
    if default is None:
        default = DEFAULT_BACKEND

    pool = BrainPool(args.device, max_models=args.max_models)
    try:
        resolve_backend(default)
        label = pool.get(default).label            # 启动时就加载，别让第一个用户等
    except Exception as e:
        print(f"默认后端 {default} 不可用（{e}），退回 {DEFAULT_BACKEND}")
        default = DEFAULT_BACKEND
        label = pool.get(default).label

    print(f"默认 AI 后端: {label}（{default}）")
    print(f"可选后端 {len(discover_backends())} 个，界面上可随时切换")
    print(f"监听 http://{args.host}:{args.port}")
    uvicorn.run(build_app(pool, default), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
