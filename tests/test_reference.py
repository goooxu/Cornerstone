"""与朴素参考实现交叉比对。

Board 用位棋盘 + 角点前沿枚举，reference.cpp 用字符网格盲扫全部 17836 个组合，
两者除了朝向表以外没有共享逻辑。默认比对 5 万个局面；

    CORNERSTONE_XCHECK=1000000 pytest tests/test_reference.py

可以拉到里程碑要求的百万级。
"""

import os
import time

import numpy as np
import pytest

import cornerstone as cs
from conftest import make_rng

TARGET = int(os.environ.get("CORNERSTONE_XCHECK", "50000"))


def test_cross_check_random_positions():
    positions = 0
    mismatches = []
    t0 = time.time()
    seed = 0
    while positions < TARGET:
        b = cs.Board()
        rng = make_rng(0xC0FFEE + seed)
        seed += 1
        while not b.terminal:
            fast = np.sort(b.legal_moves())
            ref = cs.legal_moves_reference(b)
            if not np.array_equal(fast, ref):
                mismatches.append((b.history().tolist(), fast.tolist(), ref.tolist()))
                if len(mismatches) > 3:
                    break
            positions += 1
            b.play(int(fast[rng.integers(fast.size)]))
        if mismatches:
            break

    assert not mismatches, f"着法生成与参考实现不一致，前几例: {mismatches[:2]}"
    print(f"\n交叉比对 {positions} 个局面，耗时 {time.time() - t0:.1f}s")


def test_cross_check_greedy_positions():
    """随机对局的局面分布偏均匀，再补一批「总下最大棋子」的对局，
    这类局面棋子早早用完、停手更常见，能覆盖到不同的分支。"""
    sizes = np.array(cs.piece_sizes())
    for seed in range(20):
        rng = make_rng(seed)
        b = cs.Board()
        while not b.terminal:
            fast = np.sort(b.legal_moves())
            assert np.array_equal(fast, cs.legal_moves_reference(b))
            areas = np.array([sizes[cs.decode_action(int(a))["piece"]] for a in fast])
            best = np.flatnonzero(areas == areas.max())
            b.play(int(fast[best[rng.integers(best.size)]]))


def test_cross_check_terminal_positions():
    """终局局面双方都应该无着法，参考实现也必须给出空集。"""
    for seed in range(40):
        rng = make_rng(900 + seed)
        b = cs.Board()
        while not b.terminal:
            mv = b.legal_moves()
            b.play(int(mv[rng.integers(mv.size)]))
        assert b.legal_moves().size == 0
        assert cs.legal_moves_reference(b).size == 0
