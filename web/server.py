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

# 难度只对网络后端有意义 —— 它就是搜索的模拟数。
# 规则基线是确定性的启发式，没有可调的「想多久」，选了它这一项就不起作用。
DIFFICULTIES = {"简单": 16, "普通": 64, "困难": 256, "极难": 800}

# 界面上可选的规则基线。给这三个是因为它们各代表一种打法：
# 只看棋子大小 / 只堵对方 / 调好权重的综合版。强弱和设计依据见 docs/03。
RULE_BACKENDS = ["greedy-area", "corner-min", "greedy-mobility"]

DEFAULT_BACKEND = "rule:greedy-mobility"


# --------------------------------------------------------------------- 后端发现

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
        info = self.analyse(history, sims)
        if info is None:
            raise RuntimeError("搜索没有产出可用的根节点信息")
        best = int(np.argmax(info["probs"]))
        return int(info["actions"][best]), info


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


# ----------------------------------------------------------------------- 会话

@dataclass
class Session:
    board: cs.Board = field(default_factory=cs.Board)
    history: list[int] = field(default_factory=list)
    human_player: int = 0
    difficulty: str = "普通"
    backend: str = DEFAULT_BACKEND
    created: float = field(default_factory=time.time)

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
        "result": b.result_for(s.human_player) if b.terminal else None,
        "difficulty": s.difficulty,
        "backend": s.backend,
    }
    if analysis is not None:
        out["analysis"] = analysis
    return out


def analysis_payload(info: dict | None, board: cs.Board) -> dict | None:
    """把根节点信息整理成前端要的形状：胜率、top 着法、落点热力图。"""
    if info is None:
        return None
    actions = np.asarray(info["actions"])
    probs = np.asarray(info["probs"])
    order = np.argsort(-probs)[:8]

    heat = np.zeros((cs.BOARD_N, cs.BOARD_N), dtype=np.float64)
    for a, p in zip(actions, probs):
        if p < 1e-4:
            continue
        for r, c in cs.decode_action(int(a))["cells"]:
            heat[r, c] += float(p)
    m = heat.max()
    if m > 0:
        heat /= m

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
        "heatmap": heat.round(4).tolist(),
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
    human_player: int = 0
    difficulty: str = "普通"
    backend: str | None = None


class MoveReq(BaseModel):
    sid: str
    action: int


class SidReq(BaseModel):
    sid: str


class BackendReq(BaseModel):
    sid: str
    backend: str | None = None
    difficulty: str | None = None


def build_app(pool: BrainPool, default_backend: str = DEFAULT_BACKEND):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="cornerstone 试玩")

    def get(sid: str) -> Session:
        s = SESSIONS.get(sid)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请开新局")
        return s

    def brain_for(s: Session):
        """取会话对应的后端。加载失败要给出**能看懂**的 400，而不是 500。

        checkpoint 会被训练侧轮换删除，界面上列出来的那一刻还在、
        点下去可能就没了 —— 这不是 bug，是正常状态，得让用户知道换一个就行。
        """
        try:
            return pool.get(s.backend)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:                      # torch.load 失败等
            raise HTTPException(400, f"后端 {s.backend} 加载失败：{e}")

    def sims_for(s: Session) -> int:
        return DIFFICULTIES.get(s.difficulty, DIFFICULTIES["普通"])

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

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
            "difficulties": list(DIFFICULTIES),
            "default_backend": default_backend,
        }

    @app.get("/api/backends")
    def backends():
        """每次都重新扫盘：训练在跑，checkpoint 是会变多的。"""
        return {"backends": discover_backends(), "default": default_backend}

    @app.post("/api/backend")
    def set_backend(req: BackendReq):
        """中途换对手。棋盘不动 —— 换个引擎接着下这一局正是试玩要干的事。"""
        s = get(req.sid)
        if req.backend is not None:
            try:
                resolve_backend(req.backend)
            except ValueError as e:
                raise HTTPException(400, str(e))
            s.backend = req.backend
        if req.difficulty is not None:
            s.difficulty = req.difficulty
        b = brain_for(s)
        return {"state": state_of(s), "label": b.label, "has_net": b.has_net}

    @app.post("/api/new")
    def new_game(req: NewGame):
        if len(SESSIONS) >= MAX_SESSIONS:      # 简单的容量保护，删最旧的
            oldest = min(SESSIONS, key=lambda k: SESSIONS[k].created)
            SESSIONS.pop(oldest, None)
        backend = req.backend or default_backend
        try:
            resolve_backend(backend)
        except ValueError as e:
            raise HTTPException(400, str(e))
        sid = uuid.uuid4().hex[:16]
        s = Session(human_player=int(req.human_player) & 1, difficulty=req.difficulty,
                    backend=backend)
        SESSIONS[sid] = s
        return {"sid": sid, "state": state_of(s)}

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
        b = brain_for(s)
        t0 = time.perf_counter()
        action, info = b.choose(s.board, s.history, sims_for(s))
        if not s.board.is_legal(int(action)):
            # 后端选出非法着法说明局面与搜索树对不上，继续走会把棋盘弄脏
            raise HTTPException(500, f"后端 {s.backend} 给出了非法着法 {action}")
        s.board.play(int(action))
        s.history.append(int(action))
        out = state_of(s, analysis_payload(info, s.board))
        out["ai_move"] = int(action)
        out["ai_seconds"] = round(time.perf_counter() - t0, 3)
        out["ai_label"] = b.label
        return out

    @app.get("/api/analysis")
    def analysis(sid: str):
        s = get(sid)
        if s.board.terminal:
            return {"analysis": None}
        b = brain_for(s)
        return {"analysis": analysis_payload(b.analyse(s.history, sims_for(s)), s.board),
                "has_net": b.has_net, "label": b.label}

    @app.post("/api/undo")
    def undo(req: SidReq):
        s = get(req.sid)
        # 退回到轮到人类且至少撤掉一手为止
        hist = list(s.history)
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
