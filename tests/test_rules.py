"""规则：合法性、停手、终局、计分与胜负判定。"""

import numpy as np
import pytest

import cornerstone as cs
from conftest import grid_of, make_rng, random_game

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def test_initial_state():
    b = cs.Board()
    assert b.current_player == 0
    assert b.ply == 0
    assert not b.terminal
    assert b.scores() == (0, 0)
    assert b.remaining_mask(0) == b.remaining_mask(1) == (1 << 21) - 1
    # 首手只需盖住起始格，(4,4) 离边界足够远，91 个朝向的全部 414 个
    # (朝向, 覆盖格) 组合都放得下
    assert b.legal_count() == 414


def test_first_move_must_cover_start_square():
    b = cs.Board()
    for a in b.legal_moves():
        assert (4, 4) in cs.decode_action(int(a))["cells"]

    b.play(int(b.legal_moves()[0]))
    assert b.current_player == 1
    for a in b.legal_moves():
        assert (9, 9) in cs.decode_action(int(a))["cells"]


def test_play_updates_score_and_inventory():
    b = cs.Board()
    a = int(b.legal_moves()[100])
    d = cs.decode_action(a)
    assert b.piece_remaining(0, d["piece"])
    b.play(a)
    assert not b.piece_remaining(0, d["piece"])
    assert b.score(0) == d["size"]
    assert b.ply == 1
    assert list(b.history()) == [a]


def test_illegal_moves_raise():
    b = cs.Board()
    legal = set(int(x) for x in b.legal_moves())
    # 不盖起始格的着法必须非法
    bad = next(a for a in range(cs.NUM_ACTIONS) if cs.action_in_bounds()[a] and a not in legal)
    assert not b.is_legal(bad)
    with pytest.raises(ValueError):
        b.play(bad)
    with pytest.raises(ValueError):
        b.play(-1)
    with pytest.raises(ValueError):
        b.play(cs.NUM_ACTIONS + 5)


def test_legal_mask_matches_legal_moves():
    rng = make_rng(11)
    b = cs.Board()
    for _ in range(8):
        mv = b.legal_moves()
        mask = b.legal_mask()
        assert mask.shape == (cs.NUM_ACTIONS,)
        assert set(np.flatnonzero(mask).tolist()) == set(mv.tolist())
        assert b.legal_count() == mv.size
        b.play(int(mv[rng.integers(mv.size)]))


def test_placement_rules_hold_for_every_generated_move():
    """对随机局面，逐格验证生成的着法确实满足「不与己方边相邻、必与己方角相邻」。"""
    rng = make_rng(3)
    checked = 0

    def check(board, moves):
        nonlocal checked
        g = grid_of(board)
        p = board.current_player
        first = board.score(p) == 0
        # 抽样验证，全查太慢
        for a in rng.choice(moves, size=min(25, moves.size), replace=False):
            cells = cs.decode_action(int(a))["cells"]
            corner = False
            covers_start = False
            for r, c in cells:
                assert g[r, c] == -1, "落在已占格上"
                if (r, c) == cs.START_CELLS[p]:
                    covers_start = True
                for dr, dc in EDGE:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cs.BOARD_N and 0 <= nc < cs.BOARD_N:
                        assert g[nr, nc] != p, "与己方棋子边相邻"
                for dr, dc in DIAG:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cs.BOARD_N and 0 <= nc < cs.BOARD_N and g[nr, nc] == p:
                        corner = True
            assert covers_start if first else corner
            checked += 1

    for s in range(30):
        random_game(make_rng(1000 + s), on_position=check)
    assert checked > 1000


def test_pass_and_terminal_semantics():
    """一方无着法时停手、另一方继续；双方都无着法才终局。"""
    rng = make_rng(5)
    passes = 0
    for s in range(50):
        b = cs.Board()
        while not b.terminal:
            before = b.current_player
            b.play(int(b.legal_moves()[rng.integers(b.legal_count())]))
            if not b.terminal and b.current_player == before:
                # 同一方连走 => 对方必须确实无着法
                passes += 1
                assert not b.has_any_move(1 - before)
        # 终局 <=> 双方都无着法
        assert not b.has_any_move(0) and not b.has_any_move(1)
        assert b.legal_moves().size == 0
        with pytest.raises(Exception):
            b.play(0)
    assert passes > 0, "50 局里一次停手都没出现，停手逻辑可能没被触发"


def test_scoring_and_result():
    """占格数多者胜、相同为和局；没有 +15/+20 之类的加分。"""
    rng = make_rng(9)
    draws = 0
    for s in range(300):
        b = random_game(make_rng(5000 + s))
        s0, s1 = b.scores()
        # 占格数 = 已落棋子的格数之和，上限 89
        assert 0 < s0 <= cs.TOTAL_SQUARES and 0 < s1 <= cs.TOTAL_SQUARES
        placed0 = sum(sz for i, sz in enumerate(cs.piece_sizes()) if not b.piece_remaining(0, i))
        placed1 = sum(sz for i, sz in enumerate(cs.piece_sizes()) if not b.piece_remaining(1, i))
        assert (s0, s1) == (placed0, placed1)

        assert b.result_for(0) == (s0 > s1) - (s0 < s1)
        assert b.result_for(1) == -b.result_for(0)
        if s0 == s1:
            draws += 1
            assert b.result_for(0) == b.result_for(1) == 0
    assert draws > 0, "300 局随机对局里没有出现和局，和局判定没被覆盖到"


def test_occupancy_matches_history():
    rng = make_rng(13)
    b = random_game(rng)
    g = grid_of(b)
    replay = np.full((cs.BOARD_N, cs.BOARD_N), -1, dtype=np.int8)
    r2 = cs.Board()
    for a in b.history():
        p = r2.current_player
        for r, c in cs.decode_action(int(a))["cells"]:
            assert replay[r, c] == -1
            replay[r, c] = p
        r2.play(int(a))
    assert np.array_equal(g, replay)
    assert r2.scores() == b.scores()
    assert r2.terminal


def test_features_shape_and_content():
    b = cs.Board()
    b.play(int(b.legal_moves()[0]))
    planes, scalars = b.features()
    assert planes.shape == (cs.NUM_PLANES, cs.BOARD_N, cs.BOARD_N)
    assert scalars.shape == (cs.NUM_SCALARS,)
    assert planes.dtype == np.float32 and scalars.dtype == np.float32

    me = b.current_player           # 轮到玩家 1
    op = 1 - me
    g = grid_of(b)
    assert np.array_equal(planes[0], (g == me).astype(np.float32))
    assert np.array_equal(planes[1], (g == op).astype(np.float32))
    assert np.all(planes[8] == 1.0)
    # 起始格平面按行棋方视角区分己方/对方
    assert planes[6][cs.START_CELLS[me]] == 1.0 and planes[6].sum() == 1.0
    assert planes[7][cs.START_CELLS[op]] == 1.0 and planes[7].sum() == 1.0
    # 标量：己方/对方剩余棋子 + 己方/对方占格数
    assert scalars[:21].sum() == 21          # 玩家 1 还没落子
    assert scalars[21:42].sum() == 20        # 玩家 0 落了一枚
    assert scalars[42] == pytest.approx(b.score(me) / cs.TOTAL_SQUARES)
    assert scalars[43] == pytest.approx(b.score(op) / cs.TOTAL_SQUARES)


def test_board_copy_is_independent():
    b = cs.Board()
    b.play(int(b.legal_moves()[0]))
    c = b.copy()
    c.play(int(c.legal_moves()[0]))
    assert c.ply == 2 and b.ply == 1
    assert b.scores() != c.scores() or b.current_player != c.current_player
