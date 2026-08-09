"""Replay buffer。

只存**着法序列 + 稀疏策略目标**，局面特征在取批时由 C++ 回放重建。
一局约 27 手，每手 32 个 (int32 动作, fp32 概率) ≈ 260 B，加上着法本身
一局大约 7 KB —— 而稠密策略目标是每手 17836 个 float = 71 KB/手。
差三个数量级，这是工作目录只剩百来 GB 时唯一可行的存法。
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import _engine as E

TOP_K = E.MAX_TOPK


@dataclass
class Game:
    actions: np.ndarray      # int32 [T]
    players: np.ndarray      # int8  [T]
    n_legal: np.ndarray      # int32 [T]
    n_top: np.ndarray        # uint8 [T]
    rest_prob: np.ndarray    # f32   [T]
    top_actions: np.ndarray  # int32 [T, K]
    top_probs: np.ndarray    # f32   [T, K]
    result0: int
    score0: int
    score1: int
    # 哪些手可以拿来训练。纯自博弈是全 True；对手池的对局里只有主网络那一方为 True ——
    # 旧 checkpoint 那侧的策略目标来自更弱的搜索，拿来训练等于向弱教师学习。
    # 注意**回放仍然需要全部的手**（要从空盘重建局面），所以这只是采样掩码。
    trainable: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_trainable(self) -> int:
        return len(self) if self.trainable is None else int(self.trainable.sum())

    def trainable_plies(self) -> np.ndarray:
        """可训练手的绝对 ply 下标。"""
        if self.trainable is None:
            return np.arange(len(self), dtype=np.int32)
        return np.flatnonzero(self.trainable).astype(np.int32)

    @staticmethod
    def from_record(d: dict, main_player_only: bool = False) -> "Game":
        players = np.asarray(d["players"], dtype=np.int8)
        trainable = None
        if main_player_only:
            trainable = players == np.int8(d["net_player"])
        return Game(
            trainable=trainable,
            actions=np.asarray(d["actions"], dtype=np.int32),
            players=np.asarray(d["players"], dtype=np.int8),
            n_legal=np.asarray(d["n_legal"], dtype=np.int32),
            n_top=np.asarray(d["n_top"], dtype=np.uint8),
            rest_prob=np.asarray(d["rest_prob"], dtype=np.float32),
            top_actions=np.asarray(d["top_actions"], dtype=np.int32),
            top_probs=np.asarray(d["top_probs"], dtype=np.float32),
            result0=int(d["result0"]),
            score0=int(d["score0"]),
            score1=int(d["score1"]),
        )


class ReplayBuffer:
    """按局面数限容的滚动窗口。"""

    def __init__(self, capacity_positions: int = 2_000_000):
        self.capacity = int(capacity_positions)
        self.games: deque[Game] = deque()
        self.n_positions = 0
        self._cum: np.ndarray | None = None
        self.total_games_seen = 0

    def __len__(self) -> int:
        return self.n_positions

    def add_records(self, records, main_player_only: bool = False) -> int:
        """main_player_only=True 用于对手池的对局：只有主网络那一方的手参与训练。"""
        added = 0
        for d in records:
            # 评测模式的记录只含网络方走的手，从空棋盘回放会得到非法序列。
            # 真让它混进来，build_batch 里 Board::play 才会抛异常，那时已经很难查了。
            if not d.get("selfplay", True):
                raise ValueError("评测模式的对局记录不能进 replay buffer："
                                 "它只记了网络方的着法，无法回放重建局面")
            g = Game.from_record(d, main_player_only=main_player_only)
            if len(g) == 0:
                continue
            self.games.append(g)
            self.n_positions += g.n_trainable
            self.total_games_seen += 1
            added += 1
        while self.n_positions > self.capacity and len(self.games) > 1:
            old = self.games.popleft()
            self.n_positions -= old.n_trainable
        self._cum = None
        return added

    def _cumulative(self) -> np.ndarray:
        if self._cum is None:
            lens = np.fromiter((g.n_trainable for g in self.games), dtype=np.int64,
                               count=len(self.games))
            self._cum = np.cumsum(lens)
        return self._cum

    def sample(self, batch: int, rng: np.random.Generator, threads: int = 8,
               augment: bool = True) -> dict[str, np.ndarray]:
        if self.n_positions == 0:
            raise RuntimeError("replay buffer 是空的")

        cum = self._cumulative()
        flat = rng.integers(0, self.n_positions, size=batch)
        gi = np.searchsorted(cum, flat, side="right")
        k = (flat - np.where(gi > 0, cum[gi - 1], 0)).astype(np.int32)
        # k 是「第几个可训练手」，要映射成绝对 ply —— build_batch 回放时用的是绝对下标
        ply = np.array([self.games[int(g)].trainable_plies()[int(j)]
                        for g, j in zip(gi, k)], dtype=np.int32)

        # C++ 侧要求每局的 ply 递增（一次回放吐出该局全部样本），先按 (局, 手) 排序
        order = np.lexsort((ply, gi))
        gi, ply = gi[order], ply[order]

        uniq, inv = np.unique(gi, return_inverse=True)
        sel = [self.games[int(k)] for k in uniq]

        acts = np.concatenate([g.actions for g in sel]) if sel else np.zeros(0, np.int32)
        game_off = np.zeros(len(sel) + 1, dtype=np.int32)
        np.cumsum([len(g) for g in sel], out=game_off[1:])

        want_off = np.zeros(len(sel) + 1, dtype=np.int32)
        np.cumsum(np.bincount(inv, minlength=len(sel)), out=want_off[1:])

        syms = rng.integers(0, E.NUM_SYM, size=batch).astype(np.int8) if augment \
            else np.zeros(batch, dtype=np.int8)

        planes = np.empty((batch, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), dtype=np.float32)
        scalars = np.empty((batch, E.NUM_SCALARS), dtype=np.float32)
        legal = np.empty((batch, E.NUM_ACTIONS), dtype=np.uint8)
        E.build_batch(acts, game_off, ply, want_off, syms, planes, scalars, legal, threads)

        # 稀疏策略目标：动作编号按对称变换重映射即可，概率不变
        top_a = np.stack([sel[i].top_actions[p] for i, p in zip(inv, ply)])
        top_p = np.stack([sel[i].top_probs[p] for i, p in zip(inv, ply)])
        n_top = np.array([sel[i].n_top[p] for i, p in zip(inv, ply)], dtype=np.int32)
        n_legal = np.array([sel[i].n_legal[p] for i, p in zip(inv, ply)], dtype=np.int32)
        rest = np.array([sel[i].rest_prob[p] for i, p in zip(inv, ply)], dtype=np.float32)
        players = np.array([sel[i].players[p] for i, p in zip(inv, ply)], dtype=np.int8)

        sym_tab = _sym_table()
        valid = np.arange(TOP_K)[None, :] < n_top[:, None]
        remapped = sym_tab[syms.astype(np.int64)[:, None], top_a]
        # 补位槽的概率是 0，但动作编号必须仍然合法：否则 gather 到非法位置会取到
        # -inf 的 log 概率，再乘 0 就是 NaN。用第 0 个（一定合法）的动作填充。
        top_a = np.where(valid, remapped, remapped[:, :1])
        top_p = np.where(valid, top_p, 0.0)

        result0 = np.array([sel[i].result0 for i in inv], dtype=np.int8)
        s0 = np.array([sel[i].score0 for i in inv], dtype=np.float32)
        s1 = np.array([sel[i].score1 for i in inv], dtype=np.float32)

        # 目标一律换算到「该局面行棋方」的视角
        sign = np.where(players == 0, 1, -1).astype(np.int8)
        result = (result0 * sign).astype(np.int64)          # +1 胜 / 0 和 / -1 负
        wdl = (1 - result).astype(np.int64)                 # 0 胜 / 1 和 / 2 负
        score_diff = np.where(players == 0, s0 - s1, s1 - s0) / E.TOTAL_SQUARES

        return {
            "planes": planes,
            "scalars": scalars,
            "legal": legal,
            "top_actions": top_a.astype(np.int64),
            "top_probs": top_p,
            "rest_prob": rest,
            "n_top": n_top,
            "n_legal": n_legal,
            "wdl": wdl,
            "score_diff": score_diff.astype(np.float32),
        }

    # ---- 持久化 ----

    def save_shard(self, path: str) -> None:
        """先写临时文件再原子重命名。

        直接写最终路径的话，进程在写到一半被强杀（开发机会话到期就是这样）
        会留下一个半截的 npz，下次续训直接崩在 zlib 解压上 ——
        而且是崩在「恢复」这一步，等于把整个 checkpoint 也一起废掉了。
        """
        if not self.games:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        lens = np.array([len(g) for g in self.games], dtype=np.int32)
        np.savez_compressed(
            tmp,
            lens=lens,
            actions=np.concatenate([g.actions for g in self.games]),
            players=np.concatenate([g.players for g in self.games]),
            n_legal=np.concatenate([g.n_legal for g in self.games]),
            trainable=np.concatenate([
                (np.ones(len(g), np.uint8) if g.trainable is None
                 else g.trainable.astype(np.uint8)) for g in self.games]),
            n_top=np.concatenate([g.n_top for g in self.games]),
            rest_prob=np.concatenate([g.rest_prob for g in self.games]),
            top_actions=np.concatenate([g.top_actions for g in self.games]),
            top_probs=np.concatenate([g.top_probs for g in self.games]),
            result0=np.array([g.result0 for g in self.games], dtype=np.int8),
            score0=np.array([g.score0 for g in self.games], dtype=np.int16),
            score1=np.array([g.score1 for g in self.games], dtype=np.int16),
        )
        os.replace(tmp, path)

    def load_shard(self, path: str) -> int:
        with np.load(path) as z:
            # 必须先把每个数组整体取出来一次。NpzFile 是惰性的：**每次 z["k"] 都会
            # 重新解压整个数组**。写成在循环里 z["actions"][a:b] 的话，
            # 11 万局 x 9 个数组 = 上百万次全量解压，表现为进程直接挂死。
            # 小规模（几千局）完全看不出来，一上真实规模就废。
            arr = {k: z[k] for k in ("lens", "actions", "players", "n_legal", "n_top",
                                     "rest_prob", "top_actions", "top_probs",
                                     "result0", "score0", "score1")}
            try:
                # 旧快照没有这个字段，缺了就当全部可训练（纯自博弈）。
                # 用 try 而不是查 z.files —— 依然是「每个数组只取一次」。
                arr["trainable"] = z["trainable"]
            except KeyError:
                pass
        lens = arr["lens"]
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        for i in range(len(lens)):
            a, b = off[i], off[i + 1]
            self.games.append(Game(
                actions=arr["actions"][a:b], players=arr["players"][a:b],
                n_legal=arr["n_legal"][a:b], n_top=arr["n_top"][a:b],
                rest_prob=arr["rest_prob"][a:b], top_actions=arr["top_actions"][a:b],
                top_probs=arr["top_probs"][a:b], result0=int(arr["result0"][i]),
                score0=int(arr["score0"][i]), score1=int(arr["score1"][i]),
                trainable=(arr["trainable"][a:b].astype(bool)
                           if "trainable" in arr else None),
            ))
            self.n_positions += self.games[-1].n_trainable
            self.total_games_seen += 1
        while self.n_positions > self.capacity and len(self.games) > 1:
            self.n_positions -= self.games.popleft().n_trainable
        self._cum = None
        return int(len(lens))


_SYM: np.ndarray | None = None


def _sym_table() -> np.ndarray:
    global _SYM
    if _SYM is None:
        _SYM = E.sym_action_table().astype(np.int64)
    return _SYM
