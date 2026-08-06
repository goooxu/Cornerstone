"""对称变换。

可用的 4 个变换构成克莱因四元群：
    ID, TRANSPOSE(主对角), ROT180, ANTIDIAG(副对角)
起始格集合 {(4,4),(9,9)} 在这 4 个变换下都映射到自身（TRANSPOSE 保持不动，
ROT180/ANTIDIAG 互换两者）。因为特征是「行棋方视角 + 显式区分己方/对方起始格平面」，
互换起始格只是换了个座位看同一盘棋，所以 4 个都是合法的数据增广。
"""

import numpy as np
import pytest

import cornerstone as cs
from conftest import make_rng, random_game

SYM_ID, SYM_TRANSPOSE, SYM_ROT180, SYM_ANTIDIAG = range(4)
SYM_NAMES = ["ID", "TRANSPOSE", "ROT180", "ANTIDIAG"]


@pytest.fixture(scope="module")
def tables():
    return cs.sym_action_table(), cs.sym_cell_table()


def test_cell_table_is_the_expected_geometry(tables):
    _, scell = tables
    n = cs.BOARD_N
    for cell in range(cs.NUM_CELLS):
        r, c = divmod(cell, n)
        expect = {
            SYM_ID: (r, c),
            SYM_TRANSPOSE: (c, r),
            SYM_ROT180: (n - 1 - r, n - 1 - c),
            SYM_ANTIDIAG: (n - 1 - c, n - 1 - r),
        }
        for s in range(4):
            er, ec = expect[s]
            assert scell[s, cell] == er * n + ec, SYM_NAMES[s]


def test_start_squares_map_to_start_squares(tables):
    _, scell = tables
    starts = {r * cs.BOARD_N + c for r, c in cs.START_CELLS}
    for s in range(4):
        assert {int(scell[s, x]) for x in starts} == starts


def test_action_table_matches_cell_table(tables):
    """核心性质：变换后的动作，其格集合正好等于原格集合逐格变换的结果。"""
    saction, scell = tables
    ib = cs.action_in_bounds()
    for a in range(cs.NUM_ACTIONS):
        if not ib[a]:
            for s in range(4):
                assert saction[s, a] == -1
            continue
        cells = cs.decode_action(a)["cells"]
        for s in range(4):
            t = int(saction[s, a])
            assert t >= 0 and ib[t]
            want = {int(scell[s, r * cs.BOARD_N + c]) for r, c in cells}
            got = {r * cs.BOARD_N + c for r, c in cs.decode_action(t)["cells"]}
            assert got == want, (a, SYM_NAMES[s])


def test_identity_and_bijection(tables):
    saction, _ = tables
    ib = cs.action_in_bounds().astype(bool)
    assert np.array_equal(saction[SYM_ID][ib], np.flatnonzero(ib))
    for s in range(4):
        mapped = saction[s][ib]
        assert len(np.unique(mapped)) == int(ib.sum()), f"{SYM_NAMES[s]} 不是双射"


def test_klein_four_group(tables):
    """每个元素自逆，任意两个非单位元之积是第三个非单位元。"""
    saction, _ = tables
    ib = cs.action_in_bounds().astype(bool)
    idx = np.flatnonzero(ib)

    for s in range(4):
        assert np.array_equal(saction[s][saction[s][idx]], idx), f"{SYM_NAMES[s]} 不自逆"

    compose = {
        (SYM_TRANSPOSE, SYM_ROT180): SYM_ANTIDIAG,
        (SYM_ROT180, SYM_TRANSPOSE): SYM_ANTIDIAG,
        (SYM_TRANSPOSE, SYM_ANTIDIAG): SYM_ROT180,
        (SYM_ANTIDIAG, SYM_TRANSPOSE): SYM_ROT180,
        (SYM_ROT180, SYM_ANTIDIAG): SYM_TRANSPOSE,
        (SYM_ANTIDIAG, SYM_ROT180): SYM_TRANSPOSE,
    }
    for (s1, s2), want in compose.items():
        assert np.array_equal(saction[s2][saction[s1][idx]], saction[want][idx]), (
            f"{SYM_NAMES[s1]} 后接 {SYM_NAMES[s2]} 应该等于 {SYM_NAMES[want]}"
        )


def test_piece_identity_preserved(tables):
    """变换只改朝向和位置，不能把一枚棋子变成另一枚。"""
    saction, _ = tables
    ib = cs.action_in_bounds()
    rng = make_rng(31)
    for a in rng.choice(np.flatnonzero(ib), size=3000, replace=False):
        p = cs.decode_action(int(a))["piece"]
        for s in range(4):
            assert cs.decode_action(int(saction[s, a]))["piece"] == p


def test_transpose_replay_is_a_legal_game(tables):
    """TRANSPOSE 保持两个起始格各自不动，因此把整局着法逐手变换后
    仍是一局合法对局，且双方比分不变。"""
    saction, _ = tables
    for seed in range(25):
        b = random_game(make_rng(400 + seed))
        t = cs.Board()
        for a in b.history():
            ta = int(saction[SYM_TRANSPOSE, int(a)])
            assert t.is_legal(ta), "变换后的着法不合法"
            t.play(ta)
        assert t.terminal
        assert t.scores() == b.scores()
        assert t.result_for(0) == b.result_for(0)


def test_transpose_feature_equivariance(tables):
    """变换后局面的特征 = 原特征逐平面转置；标量不变。"""
    saction, _ = tables
    rng = make_rng(77)
    b, t = cs.Board(), cs.Board()
    for _ in range(14):
        if b.terminal:
            break
        pb, sb = b.features()
        pt, st = t.features()
        assert np.array_equal(pt, pb.transpose(0, 2, 1)), "平面不满足转置等变"
        assert np.array_equal(st, sb), "标量不该随几何变换改变"

        a = int(b.legal_moves()[rng.integers(b.legal_count())])
        b.play(a)
        t.play(int(saction[SYM_TRANSPOSE, a]))


def test_transformed_legal_set_is_consistent(tables):
    """TRANSPOSE 下：变换后局面的合法着法集合 = 原合法集合逐个变换。"""
    saction, _ = tables
    rng = make_rng(101)
    b, t = cs.Board(), cs.Board()
    for _ in range(16):
        if b.terminal:
            break
        want = np.sort(saction[SYM_TRANSPOSE][b.legal_moves()])
        assert np.array_equal(np.sort(t.legal_moves()), want)
        a = int(b.legal_moves()[rng.integers(b.legal_count())])
        b.play(a)
        t.play(int(saction[SYM_TRANSPOSE, a]))
