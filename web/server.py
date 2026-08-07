#!/usr/bin/env python3
"""Web 试玩工具后端。

    python3 web/server.py                          # 默认对手 greedy-mobility
    python3 web/server.py --checkpoint runs/ab-fp8/ckpt/step00031452.pt

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

# 界面上直接选模拟数，不再套「简单/普通/困难」这层名字 ——
# 两个座位可以各选各的，用来比「同一个网络多搜一倍值多少棋力」这种事，
# 名字反而挡着看不清实际预算。
# 0 = 纯策略：直接取网络先验的 argmax，不看搜索结果，确定性。
# 见 NetBrain._choose_by_policy —— 它**不能**用「把模拟数调到很小」来近似，
# 少量模拟反而是随机性最大的情形（改进策略里 sigma≈51 会让一次随机采样
# 到的 q 盖过整个 log 先验）。这一点起初判断错过，详见 docs/05。
PURE_POLICY = 0
SIM_CHOICES = [PURE_POLICY, 64, 256, 800]
DEFAULT_SIMS = 64
# 上限不是审美问题：单局面搜索的批大小恒为 1，模拟数直接线性折算成等待时间，
# 放开了就能让一个请求把服务占住好几分钟。
MAX_SIMS = 2000

# 界面上可选的规则基线。给这三个是因为它们各代表一种打法：
# 只看棋子大小 / 只堵对方 / 调好权重的综合版。强弱和设计依据见 docs/03。
RULE_BACKENDS = ["greedy-area", "corner-min", "greedy-mobility"]

DEFAULT_BACKEND = "rule:greedy-mobility"


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

    `net:<跑名>/latest` 是**每次调用都重新解析**的，不缓存 ——
    正在训练的跑每隔几分钟就落一份新 checkpoint，缓存了就永远停在选中那一刻。
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
    ckpt_dir = os.path.join(RUNS_DIR, run, "ckpt")
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"找不到训练跑 {run}")

    if fname in ("latest", ""):
        # 先信 latest 文件；它指向的那份可能已经被 _prune_checkpoints 删了，
        # 所以要回退到目录里实际存在的最大步数那份
        pointer = os.path.join(ckpt_dir, "latest")
        if os.path.exists(pointer):
            with open(pointer) as f:
                cand = os.path.join(ckpt_dir, f.read().strip())
            if os.path.exists(cand):
                return "net", cand
        found = sorted(glob.glob(os.path.join(ckpt_dir, "step*.pt")), key=_step_of)
        if not found:
            raise ValueError(f"{run} 还没有 checkpoint")
        return "net", found[-1]

    path = os.path.join(ckpt_dir, os.path.basename(fname))
    if not os.path.exists(path):
        # checkpoint 会被轮换删除，界面上列出来的那一刻还在、点下去可能就没了
        raise ValueError(f"checkpoint 已不存在（可能已被轮换删除）：{os.path.basename(fname)}")
    return "net", path


def discover_backends() -> list[dict]:
    """列出当前可选的对手。每次调用都重新扫盘，好让新落的 checkpoint 能出现。"""
    out = [{"id": f"rule:{n}", "label": n, "group": "规则基线", "kind": "rule"}
           for n in RULE_BACKENDS]

    for ckpt_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "ckpt"))):
        run = os.path.basename(os.path.dirname(ckpt_dir))
        cks = sorted(glob.glob(os.path.join(ckpt_dir, "step*.pt")), key=_step_of)
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
    """单个 checkpoint 的推理后端。"""

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
        self.label = f"CornerNet {model.num_params()/1e6:.1f}M (step {step})"
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
    """按需加载并缓存后端。网络按 checkpoint 路径缓存，LRU 淘汰。

    不缓存的话每走一步都要重新 torch.load 一份 100 MB 级的 checkpoint；
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


# ------------------------------------------------------------------- 多局对战

MATCH_GAMES = [100, 200, 400]
# 分块跑而不是一次性交给评测函数：评测函数本身不报进度也不能中途停，
# 分块之后既能显示「已打 N 局」，也能按停止键就停。
# 块要够大才摊得平批量并行的收益，又不能大到一块要跑好几分钟。
MATCH_CHUNK = 20


@dataclass
class MatchState:
    """一场多局对战的累计结果。A 恒指先手座位（seat 0）。"""

    total: int = 0
    played: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    plies: float = 0.0
    squares_a: float = 0.0
    squares_b: float = 0.0
    a_first_wins: int = 0
    a_second_wins: int = 0
    label_a: str = ""
    label_b: str = ""
    sims: int = 0
    running: bool = False
    error: str = ""

    def as_dict(self) -> dict:
        n = max(1, self.played)
        score_a = (self.wins_a + 0.5 * self.draws) / n
        from cornerstone.elo import elo_from_score_rate
        return {
            "total": self.total, "played": self.played, "running": self.running,
            "error": self.error, "label_a": self.label_a, "label_b": self.label_b,
            "sims": self.sims,
            "wins_a": self.wins_a, "wins_b": self.wins_b, "draws": self.draws,
            "score_a": round(score_a, 4) if self.played else None,
            "elo_diff": round(elo_from_score_rate(score_a), 1) if self.played else None,
            "mean_plies": round(self.plies / n, 1) if self.played else None,
            "mean_squares_a": round(self.squares_a / n, 1) if self.played else None,
            "mean_squares_b": round(self.squares_b / n, 1) if self.played else None,
            "a_first_wins": self.a_first_wins, "a_second_wins": self.a_second_wins,
        }


class MatchRunner:
    """后台跑多局对战。同一时刻只允许一场 —— GPU 就一块，排队没有意义。"""

    def __init__(self, pool: BrainPool):
        self.pool = pool
        self.lock = threading.Lock()
        self.state = MatchState()
        self.stop_flag = False
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return self.state.as_dict()

    def stop(self) -> None:
        self.stop_flag = True

    def start(self, players: list[str], sims: list[int], games: int) -> None:
        # 双方在开始时就**定死**，整场都用这两个对象，中途不再解析后端 ID。
        #
        # 两个原因，都不是理论问题：
        #  1. `net:<跑>/latest` 每次解析都可能变 —— 训练还在跑。分块之间重新解析
        #     的话，一场 400 局会横跨好几个 checkpoint，那个总比分
        #     不属于任何一对模型。
        #  2. checkpoint 会被 _prune_checkpoints 轮换删除。实测就撞上了：
        #     一场 100 局的网络对网络跑到 80 局时报
        #     「checkpoint 已不存在：step00133030.pt」。
        # 抓住对象引用还顺带绕开了 BrainPool 的 LRU 淘汰 —— 模型已在显存里，
        # 文件没了也不影响这一场打完。
        brains = [self.pool.get(p) for p in players]
        with self.lock:
            if self.state.running:
                raise ValueError("已有一场多局对战在跑，先停掉它")
            self.stop_flag = False
            self.state = MatchState(total=games, running=True, sims=max(sims),
                                    label_a=brains[0].label, label_b=brains[1].label)
        self.thread = threading.Thread(
            target=self._run, args=(brains, list(sims), games), daemon=True)
        self.thread.start()

    def _run(self, players: list, sims: list[int], games: int) -> None:
        try:
            seed = 1
            while True:
                with self.lock:
                    done = self.state.played
                if self.stop_flag or done >= games:
                    break
                # 每块必须是偶数：先后手要成对交换，奇数会让分配不平衡
                chunk = min(MATCH_CHUNK, games - done)
                chunk -= chunk % 2
                if chunk <= 0:
                    break
                self._one_chunk(players, sims, chunk, seed)
                seed += 1
        except Exception as e:                      # noqa: BLE001
            with self.lock:
                self.state.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.state.running = False

    def _one_chunk(self, players: list, sims: list[int], chunk: int, seed: int) -> None:
        a, b = players            # 已在 start() 里定死的 Brain 对象，不再按 ID 解析
        if a.has_net and b.has_net:
            from cornerstone.evaluate import evaluate_vs_network
            r = evaluate_vs_network(a.model, b.model, a.device, games=chunk,
                                    simulations=max(sims), parallel_games=min(64, chunk),
                                    seed=seed, engine_threads=16)
            self._accumulate(r.wins, r.losses, r.draws, r.mean_plies * chunk,
                             r.mean_squares_net * chunk, r.mean_squares_opp * chunk,
                             r.wins_as_first, r.wins_as_second, chunk)
        elif a.has_net or b.has_net:
            from cornerstone.evaluate import evaluate_vs_baseline
            net, rule = (a, b) if a.has_net else (b, a)
            r = evaluate_vs_baseline(net.model, net.device, opponent=rule.name, games=chunk,
                                     simulations=max(sims), parallel_games=min(64, chunk),
                                     seed=seed)
            if a.has_net:       # 网络就是 A
                self._accumulate(r.wins, r.losses, r.draws, r.mean_plies * chunk,
                                 r.mean_squares_net * chunk, r.mean_squares_opp * chunk,
                                 r.wins_as_first, r.wins_as_second, chunk)
            else:               # 网络是 B，胜负要翻过来
                self._accumulate(r.losses, r.wins, r.draws, r.mean_plies * chunk,
                                 r.mean_squares_opp * chunk, r.mean_squares_net * chunk,
                                 0, 0, chunk)
        else:
            from cornerstone.arena import play_pair
            r = play_pair(a.name, b.name, a.cfg, b.cfg, chunk, seed, 16, 4)
            self._accumulate(r.wins_a, r.wins_b, r.draws, r.mean_plies * chunk,
                             r.mean_squares_a * chunk, r.mean_squares_b * chunk,
                             r.a_wins_as_first, r.a_wins_as_second, chunk)

    def _accumulate(self, wa, wb, dr, plies, sqa, sqb, af, as_, chunk) -> None:
        with self.lock:
            s = self.state
            s.wins_a += wa; s.wins_b += wb; s.draws += dr
            s.plies += plies; s.squares_a += sqa; s.squares_b += sqb
            s.a_first_wins += af; s.a_second_wins += as_
            s.played += chunk


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
            "cells": d["cells"],
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
class NewGame(BaseModel):
    # players[i] = 该座位的后端 ID，None 表示人来下。
    # 不给就退回 human_player + backend 这组旧参数。
    players: list[str | None] | None = None
    sims: list[int] | None = None
    human_player: int = 0
    backend: str | None = None


class MoveReq(BaseModel):
    sid: str
    action: int


class SidReq(BaseModel):
    sid: str


class MatchReq(BaseModel):
    players: list[str | None]
    sims: list[int]
    games: int = 100


class BackendReq(BaseModel):
    sid: str
    players: list[str | None] | None = None
    sims: list[int] | None = None


def build_app(pool: BrainPool, default_backend: str = DEFAULT_BACKEND):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="cornerstone 试玩")
    matches = MatchRunner(pool)

    def get(sid: str) -> Session:
        s = SESSIONS.get(sid)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请开新局")
        return s

    def load_brain(backend: str):
        """加载后端。失败要给出**能看懂**的 400，而不是 500。

        checkpoint 会被训练侧轮换删除，界面上列出来的那一刻还在、
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
            "match_games": MATCH_GAMES,
            "default_sims": DEFAULT_SIMS,
            "max_sims": MAX_SIMS,
            "default_backend": default_backend,
        }

    @app.post("/api/match/start")
    def match_start(req: MatchReq):
        players = validate_players(req.players)
        sims = validate_sims(req.sims)
        if req.games not in MATCH_GAMES:
            raise HTTPException(400, f"局数只能是 {MATCH_GAMES} 之一")
        if any(p is None for p in players):
            raise HTTPException(400, "多局对战两方都必须是 AI")

        # 下面两条限制来自批量对局引擎，不是随手加的，报错要说清楚原因
        brains = [load_brain(p) for p in players]
        if any(b.has_net for b in brains) and any(
                s <= PURE_POLICY for s, b in zip(sims, brains) if b.has_net):
            raise HTTPException(400,
                                "多局对战暂不支持「纯策略」：批量对局由引擎自己选着法，"
                                "走的是改进策略，没有读先验的入口。请选 64/256/800。")
        if all(b.has_net for b in brains) and sims[0] != sims[1]:
            raise HTTPException(400,
                                f"网络对网络时两方要用相同的模拟数（收到 {sims[0]} 和 {sims[1]}）："
                                "批量对局引擎两方共用一份 MCTS 配置。")
        try:
            matches.start(players, sims, req.games)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return matches.snapshot()

    @app.get("/api/match/status")
    def match_status():
        return matches.snapshot()

    @app.post("/api/match/stop")
    def match_stop():
        matches.stop()
        return matches.snapshot()

    @app.get("/api/backends")
    def backends():
        """每次都重新扫盘：训练在跑，checkpoint 是会变多的。"""
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
                and s.board.ply > 0 and not s.board.terminal:
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
            if hist:
                hist.pop()
        else:
            # 退回到轮到人类且至少撤掉一手为止
            while hist:
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
    ap.add_argument("--checkpoint", default=None,
                    help="默认对手用哪个 checkpoint；界面上仍可随时换")
    ap.add_argument("--backend", default=None,
                    help="默认后端 ID，如 rule:greedy-mobility 或 net:ab-fp8/latest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-models", type=int, default=3, help="同时缓存几个网络")
    args = ap.parse_args()

    import uvicorn
    default = args.backend
    if default is None and args.checkpoint:
        # 把 --checkpoint 折成后端 ID，好和界面上的选项对得上
        ck = os.path.abspath(args.checkpoint)
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
