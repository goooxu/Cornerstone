#!/usr/bin/env python3
"""Web 试玩工具后端。

    python3 web/server.py                          # 无 checkpoint 时用规则基线当对手
    python3 web/server.py --checkpoint runs/bf16/ckpt/step00010000.pt

规则判定全部走 C++ 引擎（`cornerstone._engine`），和训练用的是同一份实现 ——
前端只负责画和收集点击，任何合法性判断都不在 JS 里重写，否则迟早两边对不上。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cornerstone as cs                       # noqa: E402
from cornerstone import _engine as E           # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 难度 -> (是否用网络, 模拟数 / 基线名)
DIFFICULTIES = {
    "简单": dict(net_sims=16, baseline="greedy-area"),
    "普通": dict(net_sims=64, baseline="greedy-mobility"),
    "困难": dict(net_sims=256, baseline="flat-mcts-1k"),
    "极难": dict(net_sims=800, baseline="flat-mcts-4k"),
}


# --------------------------------------------------------------------------- AI

class Brain:
    """AI 后端。有 checkpoint 就用网络 + Gumbel-AZ，否则退回规则基线。"""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self.model = None
        self.device = None
        self.name = "规则基线"
        self.engines: dict[int, E.SelfPlayEngine] = {}
        if checkpoint:
            self._load(checkpoint, device)

    def _load(self, path: str, device: str) -> None:
        import torch
        from cornerstone.model import CornerNet, ModelConfig

        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = blob.get("model_config") or {}
        cfg = ModelConfig(**{k: v for k, v in mc.items()
                             if k in ModelConfig.__dataclass_fields__})
        model = CornerNet(cfg)
        model.load_state_dict(blob["model"])
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.torch = torch
        step = blob.get("step", "?")
        self.name = f"CornerNet {model.num_params()/1e6:.1f}M (step {step})"

    @property
    def has_net(self) -> bool:
        return self.model is not None

    def _engine(self, sims: int) -> E.SelfPlayEngine:
        if sims not in self.engines:
            self.engines[sims] = E.SelfPlayEngine(
                1, E.MctsConfig(simulations=sims, max_considered=32, temperature_plies=0), 0)
        return self.engines[sims]

    def analyse(self, history: list[int], sims: int) -> dict | None:
        """跑一次搜索，返回根节点的改进策略与估值。无网络时返回 None。"""
        if not self.has_net:
            return None
        torch = self.torch
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
        if not info["ready"]:
            return None
        return info

    def choose(self, board: cs.Board, history: list[int], difficulty: str) -> tuple[int, dict | None]:
        spec = DIFFICULTIES.get(difficulty, DIFFICULTIES["普通"])
        if self.has_net:
            info = self.analyse(history, spec["net_sims"])
            if info is not None:
                best = int(np.argmax(info["probs"]))
                return int(info["actions"][best]), info
        from cornerstone.arena import BASELINES
        cfg = BASELINES[spec["baseline"]]
        return E.select_move(board, cfg, int(time.time_ns() & 0xFFFFFFFF)), None


# ----------------------------------------------------------------------- 会话

@dataclass
class Session:
    board: cs.Board = field(default_factory=cs.Board)
    history: list[int] = field(default_factory=list)
    human_player: int = 0
    difficulty: str = "普通"
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


class MoveReq(BaseModel):
    sid: str
    action: int


class SidReq(BaseModel):
    sid: str


def build_app(brain: Brain):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="cornerstone 试玩")

    def get(sid: str) -> Session:
        s = SESSIONS.get(sid)
        if s is None:
            raise HTTPException(404, "会话不存在或已过期，请开新局")
        return s

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
            "ai": brain.name,
            "has_net": brain.has_net,
        }

    @app.post("/api/new")
    def new_game(req: NewGame):
        if len(SESSIONS) >= MAX_SESSIONS:      # 简单的容量保护，删最旧的
            oldest = min(SESSIONS, key=lambda k: SESSIONS[k].created)
            SESSIONS.pop(oldest, None)
        sid = uuid.uuid4().hex[:16]
        s = Session(human_player=int(req.human_player) & 1, difficulty=req.difficulty)
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
        t0 = time.perf_counter()
        action, info = brain.choose(s.board, s.history, s.difficulty)
        s.board.play(int(action))
        s.history.append(int(action))
        out = state_of(s, analysis_payload(info, s.board))
        out["ai_move"] = int(action)
        out["ai_seconds"] = round(time.perf_counter() - t0, 3)
        return out

    @app.get("/api/analysis")
    def analysis(sid: str):
        s = get(sid)
        if s.board.terminal:
            return {"analysis": None}
        spec = DIFFICULTIES.get(s.difficulty, DIFFICULTIES["普通"])
        info = brain.analyse(s.history, spec["net_sims"])
        return {"analysis": analysis_payload(info, s.board)}

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
    ap.add_argument("--checkpoint", default=None, help="CornerNet checkpoint；不给就用规则基线")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn
    brain = Brain(args.checkpoint, args.device)
    print(f"AI 后端: {brain.name}")
    print(f"监听 http://{args.host}:{args.port}")
    uvicorn.run(build_app(brain), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
