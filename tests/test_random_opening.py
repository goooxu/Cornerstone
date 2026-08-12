"""自博弈的随机开局注入。

为什么加这个功能：实测（`tools/diag_diversity.py`）训练到第 8 万步时，
512 局自博弈的**首手全部相同** —— 而首手有 414 种合法着法。到 15 万步只剩
5 种前两手、15 种前四手，三分之一的对局逐手重复。replay 名义上压着 300 万个
局面（约 49 轮历史），有效多样性远小于此。

这里守的都是「不报错但不对」那一类：随机手忘了进棋谱 → 回放非法；
忘了排除出训练 → 教网络模仿均匀随机着法；下标映射写错 → 静默训到错误的局面。
"""

import numpy as np
import pytest

import cornerstone as cs
from cornerstone import _engine as E
from cornerstone.replay import Game, ReplayBuffer
from conftest import make_rng
from test_selfplay import drive


def run(prob: float, max_plies: int, n_games: int = 24, seed: int = 7):
    cfg = E.MctsConfig(simulations=16, max_considered=8, temperature_plies=4,
                       random_opening_prob=prob, random_opening_max_plies=max_plies)
    return drive(E.SelfPlayEngine(16, cfg, seed), n_games)


def test_default_is_off():
    """默认必须逐位等同于改动前 —— 每一手都有搜索目标。"""
    for rec in run(0.0, 0):
        assert (np.asarray(rec["n_top"]) > 0).all()


def test_random_opening_moves_are_recorded_but_not_trainable():
    """随机手照常进棋谱（回放要用），但 n_top = 0 表示没有搜索目标。

    忘了进棋谱的后果不是报错，是**从空盘回放会得到非法序列** ——
    评测那条随机开局路径（EvalConfig.opening_plies）正是这么做的，
    所以它的记录带着 selfplay=False 的标记，进不了 replay。
    """
    recs = run(1.0, 6, n_games=32)
    n_top = [np.asarray(r["n_top"]) for r in recs]
    lead = [int(np.argmax(t > 0)) for t in n_top]        # 开头连续几个 0

    assert max(lead) > 0, "prob=1.0 却一局都没注入随机开局"
    assert all(k % 2 == 0 for k in lead), \
        "随机开局手数必须是偶数 —— 否则随机开局本身就给某一方送了先手"
    assert max(lead) <= 6
    for t, k in zip(n_top, lead):
        assert (t[k:] > 0).all(), "搜索段里不该出现 n_top=0"


def test_random_opening_records_replay_from_empty_board():
    """带随机开局的记录必须能从**空盘**逐手回放 —— replay buffer 的根本前提。"""
    for rec in run(1.0, 6, n_games=16):
        b = cs.Board()
        for a in rec["actions"]:
            assert b.is_legal(int(a)), "随机开局的着法没进棋谱，回放序列非法"
            b.play(int(a))
        assert b.terminal


def test_opening_diversity_actually_improves():
    """功能要真的起作用：注入之后首手不再由网络决定。

    必须喂一个**尖峰**先验才测得出来 —— 默认的均匀假网络配上 Gumbel 噪声本来就
    很分散，那种设置下开不开注入都一样，等于没测。真实训练里塌缩的正是先验：
    实测第 8 万步的网络 512 局自博弈只走一种首手。

    这条同时守「注入点在 start_game」：放错地方（比如放进 reset_tree）
    会让每局开头照样相同、只是树被重建。
    """
    # 不能只把某一个动作抬高 —— 首手必须盖住起始格，动作 0 根本不合法，
    # 抬了等于没抬（实测那样写首手仍有 31 种）。改成给整条动作轴一个陡坡：
    # 无论合法集是什么，**下标最小的那个合法动作**总是遥遥领先。
    ramp = -5.0 * np.arange(cs.NUM_ACTIONS, dtype=np.float32)

    def peaked(planes, n):
        return np.tile(ramp, (n, 1))

    def firsts(prob):
        cfg = E.MctsConfig(simulations=16, max_considered=8, temperature_plies=4,
                           random_opening_prob=prob, random_opening_max_plies=6)
        recs = drive(E.SelfPlayEngine(16, cfg, 3), 32, logit_fn=peaked)
        return {int(r["actions"][0]) for r in recs}

    off, on = firsts(0.0), firsts(1.0)
    assert len(off) <= 3, f"尖峰先验下首手本该高度集中，实际 {len(off)} 种"
    assert len(on) >= 4 * len(off)


def test_temperature_window_counts_searched_plies_only():
    """温度窗口按**搜索过的手数**算，不是绝对手数。

    按绝对手数的话，k 越大能享受带噪 argmax 的搜索手就越少 ——
    随机开局的剂量会悄悄变成两个变量。这里用「搜索段前若干手的分散度」
    间接守它：注入 6 手随机开局后，搜索段仍应保有温度带来的分散。
    """
    recs = run(1.0, 6, n_games=48, seed=11)
    firsts = set()
    for r in recs:
        n_top = np.asarray(r["n_top"])
        k = int(np.argmax(n_top > 0))
        # 搜索段的第一手：若温度窗口被随机开局吃掉，它会退化成确定性 argmax
        firsts.add((int(r["actions"][k]), k))
    assert len(firsts) > 8


# ---- replay 侧的可训练掩码 ----

def _game(n_top_seq):
    t = len(n_top_seq)
    return Game(
        actions=np.arange(t, dtype=np.int32),
        players=np.zeros(t, np.int8),
        n_legal=np.full(t, 5, np.int32),
        n_top=np.asarray(n_top_seq, dtype=np.uint8),
        rest_prob=np.zeros(t, np.float32),
        top_actions=np.zeros((t, 32), np.int32),
        top_probs=np.zeros((t, 32), np.float32),
        result0=1, score0=50, score1=40,
    )


def test_train_idx_skips_sentinel_moves():
    g = _game([0, 0, 3, 4, 0, 5])
    assert g.n_train() == 3
    assert list(g.train_idx) == [2, 3, 5]
    assert len(g) == 6, "__len__ 必须仍是全部着法数 —— 回放要用整条序列"


def test_buffer_counts_and_evicts_by_trainable():
    """容量与采样都按可训练手算。按全部手算的话，容量的语义会随随机开局剂量漂移。"""
    buf = ReplayBuffer(capacity_positions=4)
    for _ in range(3):
        buf.games.append(_game([0, 0, 1, 1]))
        buf.n_positions += buf.games[-1].n_train()
    assert len(buf) == 6
    while buf.n_positions > buf.capacity and len(buf.games) > 1:
        buf.n_positions -= buf.games.popleft().n_train()
    assert len(buf) == 4 and len(buf.games) == 2


def test_sample_never_returns_a_sentinel_position():
    """最关键的一条：采样出来的 ply 必须落在有搜索目标的手上。

    映射写错（比如直接拿「第几个可训练手」当绝对下标）不会报错，
    只会静默地训到错误的局面 —— 特征取自 A 手，目标取自 B 手。
    """
    recs = run(1.0, 6, n_games=24, seed=5)
    buf = ReplayBuffer(capacity_positions=1_000_000)
    buf.add_records(recs)
    assert len(buf) < sum(len(r["actions"]) for r in recs), "掩码没生效"

    rng = make_rng(3)
    batch = buf.sample(256, rng, threads=1, augment=False)
    # n_top 是从**绝对下标**取出来的；只要有一个 0 就说明映射错了
    assert (batch["n_top"] > 0).all()
    assert (batch["n_legal"] > 0).all()


def test_sampled_features_match_plain_replay():
    """特征与目标必须取自**同一手**。

    拿 Board 老老实实回放到那一手，逐字节比对 build_batch 的输出。
    """
    recs = run(1.0, 6, n_games=8, seed=9)
    buf = ReplayBuffer(capacity_positions=1_000_000)
    buf.add_records(recs)
    rng = make_rng(4)
    batch = buf.sample(64, rng, threads=1, augment=False)

    # 反查：对每个样本，在 buffer 里找出唯一匹配的 (局, 手) 并比对特征
    seen = 0
    for g in buf.games:
        for ply in g.train_idx:
            b = cs.Board()
            for a in g.actions[:ply]:
                b.play(int(a))
            planes, scal = b.features()
            hit = np.flatnonzero((batch["planes"] == planes).all(axis=(1, 2, 3)))
            for i in hit:
                assert np.array_equal(batch["scalars"][i], scal)
                assert int(batch["n_top"][i]) == int(g.n_top[ply])
                seen += 1
    assert seen >= 32, "对不上的样本太多，回放与采样的下标口径不一致"


@pytest.mark.parametrize("max_plies", [2, 4, 8])
def test_injected_length_never_exceeds_max(max_plies):
    for rec in run(1.0, max_plies, n_games=16, seed=13):
        k = int(np.argmax(np.asarray(rec["n_top"]) > 0))
        assert 0 < k <= max_plies
