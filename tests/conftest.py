import numpy as np
import pytest

import cornerstone as cs


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_game(rng, on_position=None, board=None):
    """随机合法着法走完一局。

    on_position(board, legal_moves) 会在每个待走局面上被调用一次。
    返回终局的 Board。
    """
    b = board if board is not None else cs.Board()
    guard = 0
    while not b.terminal:
        mv = b.legal_moves()
        assert mv.size > 0, "非终局却没有合法着法，说明停手/终局判定有问题"
        if on_position is not None:
            on_position(b, mv)
        b.play(int(mv[rng.integers(mv.size)]))
        guard += 1
        assert guard <= cs.MAX_PLIES, "一局超过 42 手，不可能"
    return b


def grid_of(board):
    """把局面摊成 14x14 的 numpy 网格：-1 空，0/1 为对应玩家。"""
    g = np.full((cs.BOARD_N, cs.BOARD_N), -1, dtype=np.int8)
    for p in (0, 1):
        for r, c in board.occupancy_cells(p):
            g[r, c] = p
    return g


@pytest.fixture
def rng():
    return make_rng(20260806)
