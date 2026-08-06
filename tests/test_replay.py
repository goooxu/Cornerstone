"""Replay buffer 与批构造。

最关键的一条：C++ 的 build_batch 是从着法序列**重建**局面的，
必须逐字节对上「用 Board 老老实实回放一遍」的结果，否则训练数据是错的
而且很难从 loss 曲线上看出来。
"""

import numpy as np
import pytest

import cornerstone as cs
from cornerstone import _engine as E
from cornerstone.replay import ReplayBuffer
from conftest import make_rng
from test_selfplay import drive


@pytest.fixture(scope="module")
def games():
    eng = E.SelfPlayEngine(16, E.MctsConfig(simulations=16, max_considered=8), 21)
    return drive(eng, 24)


def reference_features(actions, ply):
    """用 Board 逐手回放，取第 ply 手待走时的特征与合法掩码。"""
    b = cs.Board()
    for a in actions[:ply]:
        b.play(int(a))
    planes, scalars = b.features()
    return planes, scalars, b.legal_mask()


def test_build_batch_matches_plain_replay(games):
    g = games[0]
    acts = np.asarray(g["actions"], dtype=np.int32)
    t = len(acts)
    ply = np.arange(t, dtype=np.int32)

    planes = np.empty((t, cs.NUM_PLANES, 14, 14), dtype=np.float32)
    scal = np.empty((t, cs.NUM_SCALARS), dtype=np.float32)
    legal = np.empty((t, cs.NUM_ACTIONS), dtype=np.uint8)
    E.build_batch(acts, np.array([0, t], np.int32), ply, np.array([0, t], np.int32),
                  None, planes, scal, legal, 1)

    for i in range(t):
        rp, rs, rl = reference_features(acts, i)
        assert np.array_equal(planes[i], rp), f"第 {i} 手的特征平面对不上"
        assert np.array_equal(scal[i], rs), f"第 {i} 手的标量对不上"
        assert np.array_equal(legal[i], rl), f"第 {i} 手的合法掩码对不上"


def test_build_batch_multithreaded_matches_single_thread(games):
    sel = games[:8]
    acts = np.concatenate([np.asarray(g["actions"], np.int32) for g in sel])
    off = np.zeros(len(sel) + 1, np.int32)
    np.cumsum([len(g["actions"]) for g in sel], out=off[1:])
    ply = np.concatenate([np.arange(len(g["actions"]), dtype=np.int32) for g in sel])

    n = len(ply)
    outs = []
    for threads in (1, 8):
        p = np.empty((n, cs.NUM_PLANES, 14, 14), np.float32)
        s = np.empty((n, cs.NUM_SCALARS), np.float32)
        l = np.empty((n, cs.NUM_ACTIONS), np.uint8)
        E.build_batch(acts, off, ply, off, None, p, s, l, threads)
        outs.append((p, s, l))
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a, b), "多线程与单线程结果不一致"


@pytest.mark.parametrize("sym", [1, 2, 3])
def test_build_batch_symmetry_is_the_declared_geometry(games, sym):
    """增广后的特征必须正好是原特征按 sym_cell 置换的结果，
    合法掩码必须是按 sym_action 置换的结果。"""
    g = games[1]
    acts = np.asarray(g["actions"], np.int32)
    t = min(6, len(acts))
    ply = np.arange(t, dtype=np.int32)
    off = np.array([0, len(acts)], np.int32)
    woff = np.array([0, t], np.int32)

    def run(syms):
        p = np.empty((t, cs.NUM_PLANES, 14, 14), np.float32)
        s = np.empty((t, cs.NUM_SCALARS), np.float32)
        l = np.empty((t, cs.NUM_ACTIONS), np.uint8)
        E.build_batch(acts, off, ply, woff, syms, p, s, l, 1)
        return p, s, l

    p0, s0, l0 = run(None)
    p1, s1, l1 = run(np.full(t, sym, dtype=np.int8))

    scell = cs.sym_cell_table()[sym]
    sact = cs.sym_action_table()[sym]

    assert np.array_equal(s0, s1), "标量是座位无关量，不该随几何变换改变"
    for i in range(t):
        want = np.zeros_like(p0[i])
        flat_src = p0[i].reshape(cs.NUM_PLANES, -1)
        flat_dst = want.reshape(cs.NUM_PLANES, -1)
        flat_dst[:, scell] = flat_src
        assert np.array_equal(p1[i], want), f"第 {i} 手的平面变换不符"

        want_l = np.zeros(cs.NUM_ACTIONS, np.uint8)
        src = np.flatnonzero(l0[i])
        want_l[sact[src]] = 1
        assert np.array_equal(l1[i], want_l), f"第 {i} 手的合法掩码变换不符"


def test_buffer_sample_shapes_and_targets(games):
    buf = ReplayBuffer(capacity_positions=100_000)
    buf.add_records(games)
    assert len(buf) == sum(len(g["actions"]) for g in games)

    rng = make_rng(1)
    b = buf.sample(64, rng, threads=4, augment=True)
    assert b["planes"].shape == (64, cs.NUM_PLANES, 14, 14)
    assert b["scalars"].shape == (64, cs.NUM_SCALARS)
    assert b["legal"].shape == (64, cs.NUM_ACTIONS)
    assert b["top_actions"].shape == (64, E.MAX_TOPK)
    assert set(np.unique(b["wdl"])) <= {0, 1, 2}
    assert np.all(np.abs(b["score_diff"]) <= 1.0)
    # 每个样本至少有一个合法着法
    assert (b["legal"].sum(axis=1) > 0).all()
    # 策略目标里的动作必须在该样本的合法集合内
    rows = np.repeat(np.arange(64), E.MAX_TOPK)
    assert b["legal"][rows, b["top_actions"].ravel()].all()


def test_targets_use_the_side_to_move_perspective(games):
    buf = ReplayBuffer(capacity_positions=100_000)
    buf.add_records(games)
    rng = make_rng(2)
    b = buf.sample(256, rng, threads=4, augment=False)
    # wdl 编码：0 胜 / 1 和 / 2 负；score_diff 是行棋方减对方
    for w, d in zip(b["wdl"], b["score_diff"]):
        if w == 0:
            assert d > 0
        elif w == 2:
            assert d < 0
        else:
            assert abs(d) < 1e-6


def test_capacity_evicts_oldest(games):
    buf = ReplayBuffer(capacity_positions=200)
    buf.add_records(games)
    assert len(buf) <= 200 + max(len(g["actions"]) for g in games)
    assert len(buf.games) >= 1


def test_save_and_load_roundtrip(games, tmp_path):
    buf = ReplayBuffer(capacity_positions=100_000)
    buf.add_records(games)
    path = str(tmp_path / "shard.npz")
    buf.save_shard(path)

    other = ReplayBuffer(capacity_positions=100_000)
    n = other.load_shard(path)
    assert n == len(buf.games)
    assert len(other) == len(buf)
    for a, b in zip(buf.games, other.games):
        assert np.array_equal(a.actions, b.actions)
        assert np.array_equal(a.top_probs, b.top_probs)
        assert (a.result0, a.score0, a.score1) == (b.result0, b.score0, b.score1)
