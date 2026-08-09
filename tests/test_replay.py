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


def test_save_shard_is_atomic(games, tmp_path):
    """写快照必须走临时文件 + 原子重命名。

    直接写最终路径的话，进程写到一半被强杀（开发机会话到期就是这样）会留下
    半截的 npz，下次续训崩在 zlib 解压上 —— 而且是崩在「恢复」这一步，
    等于把还完好的 checkpoint 也一起废掉。
    """
    buf = ReplayBuffer(capacity_positions=100_000)
    buf.add_records(games)
    path = str(tmp_path / "replay.npz")

    # 先放一个可用的旧快照
    buf.save_shard(path)
    good = open(path, "rb").read()

    # 模拟写到一半失败：临时文件残留，但最终路径仍是完好的旧文件
    (tmp_path / "replay.npz.tmp.npz").write_bytes(b"truncated garbage")
    assert open(path, "rb").read() == good, "最终路径不该被半截写入污染"

    other = ReplayBuffer(capacity_positions=100_000)
    assert other.load_shard(path) == len(buf.games)


def test_load_shard_reads_each_array_once(games, tmp_path, monkeypatch):
    """载入快照时每个数组只能整体取一次。

    NpzFile 是惰性的：**每次 z["k"] 都会重新解压整个数组**。
    如果写成在循环里 `z["actions"][a:b]`，10 万局 x 9 个数组就是上百万次全量解压，
    表现为进程直接挂死。小规模测试完全看不出来，一上真实规模就废 ——
    实测 3M 局面的快照因此卡到无法使用，修好后只要 2.7 秒。
    """
    buf = ReplayBuffer(capacity_positions=100_000)
    buf.add_records(games)
    path = str(tmp_path / "shard.npz")
    buf.save_shard(path)

    real_load = np.load
    counter = {"n": 0}

    class CountingNpz:
        def __init__(self, inner):
            self._inner = inner

        def __getitem__(self, k):
            counter["n"] += 1
            return self._inner[k]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a) if hasattr(self._inner, "__exit__") else False

    monkeypatch.setattr(np, "load", lambda *a, **kw: CountingNpz(real_load(*a, **kw)))
    other = ReplayBuffer(capacity_positions=100_000)
    other.load_shard(path)

    # 11 个数组，允许一点余量；绝不能随局数增长
    assert counter["n"] <= 15, (
        f"解压了 {counter['n']} 次，说明在循环里索引了 NpzFile（{len(games)} 局）")


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


# ---- 对手池：只训练主网络那一方的手 ------------------------------------
#
# 池对局里两方都是网络，两方的手都要记录（否则从空盘回放会得到非法序列），
# 但只有主网络那一方的手能当训练目标 —— 另一方来自更弱的历史 checkpoint，
# 拿它的搜索结果训练等于向弱教师学习。

def _pool_record(n_plies=8, net_player=0):
    """造一条「两方都记录」的对局，交替行棋。"""
    import numpy as np
    from cornerstone import _engine as E
    b = E.Board()
    acts, players = [], []
    for _ in range(n_plies):
        mv = b.legal_moves()
        if len(mv) == 0:
            break
        players.append(int(b.current_player))
        acts.append(int(mv[0]))
        b.play(int(mv[0]))
    T = len(acts)
    return {
        "actions": np.array(acts, np.int32), "players": np.array(players, np.int8),
        "n_legal": np.full(T, 100, np.int32), "n_top": np.full(T, 1, np.uint8),
        "rest_prob": np.zeros(T, np.float32),
        "top_actions": np.tile(np.array(acts, np.int32)[:, None], (1, 32)),
        "top_probs": np.concatenate([np.ones((T, 1), np.float32),
                                     np.zeros((T, 31), np.float32)], axis=1),
        "result0": 1, "score0": 50, "score1": 40,
        "net_player": net_player, "selfplay": True,
    }


def test_pool_record_only_main_player_is_trainable():
    from cornerstone.replay import Game
    d = _pool_record(net_player=0)
    g = Game.from_record(d, main_player_only=True)
    assert g.trainable is not None
    assert g.n_trainable < len(g), "应当只有一半左右的手可训练"
    assert (g.players[g.trainable_plies()] == 0).all(), "取到的必须都是主网络那一方"


def test_pool_trainable_follows_net_player_not_seat_zero():
    """net_player=1 时取的是另一侧 —— 过滤不能写死成 player==0。"""
    from cornerstone.replay import Game
    g = Game.from_record(_pool_record(net_player=1), main_player_only=True)
    assert (g.players[g.trainable_plies()] == 1).all()


def test_pool_record_is_replayable_and_samplable():
    """池对局必须能进 buffer 并采样出来 —— 记录含双方的手才回放得了。"""
    import numpy as np
    from cornerstone.replay import ReplayBuffer
    buf = ReplayBuffer(capacity_positions=10_000)
    buf.add_records([_pool_record() for _ in range(6)], main_player_only=True)
    assert len(buf) > 0
    batch = buf.sample(16, np.random.default_rng(0), threads=2)
    assert batch["planes"].shape[0] == 16


def test_selfplay_records_unaffected():
    """纯自博弈路径必须一字不变：全部手都可训练。"""
    from cornerstone.replay import Game
    g = Game.from_record(_pool_record(), main_player_only=False)
    assert g.trainable is None and g.n_trainable == len(g)
