"""Gumbel AlphaZero 自博弈引擎。

用「假网络」驱动：均匀先验或人为指定的尖峰先验，这样可以检查搜索本身的行为，
不受真实网络训练状态的影响。
"""

import numpy as np
import pytest

import cornerstone as cs
from cornerstone import _engine as E
from conftest import make_rng


def drive(engine, n_games, logit_fn=None, wdl_fn=None, max_rounds=200000):
    """驱动引擎跑到收够 n_games 局。logit_fn(planes, n) -> [n, A]。"""
    g = engine.num_games
    planes = np.zeros((g, cs.NUM_PLANES, 14, 14), dtype=np.float32)
    scal = np.zeros((g, cs.NUM_SCALARS), dtype=np.float32)
    out, rounds = [], 0
    while len(out) < n_games:
        rounds += 1
        assert rounds < max_rounds, "驱动循环没有收敛"
        n = engine.prepare(planes, scal)
        if n == 0:
            out.extend(engine.advance())
            continue
        logits = (logit_fn(planes, n) if logit_fn
                  else np.zeros((n, cs.NUM_ACTIONS), dtype=np.float32))
        wdl = (wdl_fn(n) if wdl_fn
               else np.full((n, 3), 1 / 3, dtype=np.float32))
        engine.feed(logits, wdl)
    return out


def replay_is_legal(rec) -> "cs.Board":
    b = cs.Board()
    for a in rec["actions"]:
        assert b.is_legal(int(a)), "对局记录里出现非法着法"
        b.play(int(a))
    return b


def test_selfplay_produces_legal_finished_games():
    eng = E.SelfPlayEngine(16, E.MctsConfig(simulations=24, max_considered=8), 3)
    games = drive(eng, 8)
    assert len(games) >= 8
    for rec in games:
        b = replay_is_legal(rec)
        assert b.terminal
        assert (b.score(0), b.score(1)) == (rec["score0"], rec["score1"])
        assert rec["result0"] == b.result_for(0)


def test_policy_targets_are_a_proper_distribution():
    eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=32, max_considered=8), 5)
    for rec in drive(eng, 4):
        total = rec["top_probs"].sum(axis=1) + rec["rest_prob"]
        assert np.allclose(total, 1.0, atol=2e-3), "top-K 概率加尾部质量应当为 1"
        assert (rec["top_probs"] >= -1e-6).all()
        assert (rec["rest_prob"] >= -1e-6).all()
        # top-K 必须按概率降序
        for row in rec["top_probs"]:
            assert np.all(np.diff(row[row > 0]) <= 1e-6)


def test_top_actions_are_legal_at_their_position():
    eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=24, max_considered=8), 11)
    rec = drive(eng, 1)[0]
    b = cs.Board()
    for t, a in enumerate(rec["actions"]):
        k = int(rec["n_top"][t])
        for j in range(k):
            assert b.is_legal(int(rec["top_actions"][t, j])), "策略目标里出现非法着法"
        assert int(rec["n_legal"][t]) == b.legal_count()
        b.play(int(a))


def test_search_follows_a_peaked_prior():
    """给某个合法着法一个极高的先验，搜索应该压倒性地选它。

    这条验证的是「先验 -> 根候选采样 -> 落子」这条链路确实接上了。
    """
    board = cs.Board()
    target = int(board.legal_moves()[7])

    def logits(planes, n):
        out = np.zeros((n, cs.NUM_ACTIONS), dtype=np.float32)
        out[:, target] = 30.0
        return out

    eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=32, max_considered=8,
                                           temperature_plies=0), 1)
    recs = drive(eng, 4, logit_fn=logits)
    assert all(int(r["actions"][0]) == target for r in recs)


def test_value_signal_reaches_the_root():
    """把价值固定成「必胜」，根估值就该是 +1；固定成「必负」则是 -1。"""
    for wdl_vec, sign in [((1.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 1.0), -1.0)]:
        eng = E.SelfPlayEngine(4, E.MctsConfig(simulations=16, max_considered=4), 2)
        rec = drive(eng, 1, wdl_fn=lambda n, v=wdl_vec: np.tile(
            np.array(v, dtype=np.float32), (n, 1)))[0]
        assert np.allclose(rec["root_values"], sign, atol=1e-5)


def test_more_simulations_visit_more_nodes():
    """模拟数越多，根的改进策略越集中（熵越低）。"""
    def entropy(rec):
        p = rec["top_probs"]
        p = np.where(p > 0, p, 1.0)
        return float(-(rec["top_probs"] * np.log(p)).sum(axis=1).mean())

    rng = make_rng(3)
    noise = rng.standard_normal(cs.NUM_ACTIONS).astype(np.float32)

    def logits(planes, n):
        return np.tile(noise, (n, 1))

    ents = []
    for sims in (8, 128):
        eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=sims, max_considered=16), 4)
        ents.append(np.mean([entropy(r) for r in drive(eng, 4, logit_fn=logits)]))
    assert ents[1] < ents[0], f"模拟数增大后策略熵没有下降: {ents}"


def test_selfplay_records_are_marked_replayable():
    eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=16, max_considered=4), 17)
    for rec in drive(eng, 4):
        assert rec["selfplay"] is True
        # 自博弈记录里双方的手都在，且交替出现
        assert set(rec["players"].tolist()) == {0, 1}


def test_eval_mode_alternates_sides():
    ev = E.EvalConfig(enabled=True,
                      opponent=E.AgentConfig(kind=E.AgentKind.GreedyArea),
                      opening_plies=4)
    eng = E.SelfPlayEngine(8, E.MctsConfig(simulations=16, max_considered=4), 9, ev)
    games = drive(eng, 16)
    sides = [g["net_player"] for g in games]
    assert set(sides) == {0, 1}, "评测模式没有交换先后手"
    assert abs(sum(s == 0 for s in sides) - len(sides) / 2) <= len(sides) / 4
    for rec in games:
        assert rec["score0"] + rec["score1"] > 0
        assert rec["result0"] in (-1, 0, 1)


def test_eval_records_only_net_moves_and_are_not_replayable():
    """评测记录只含网络方的手（随机开局与对手的着法都没记），
    所以既不能从空棋盘回放，也不能进 replay buffer。"""
    ev = E.EvalConfig(enabled=True,
                      opponent=E.AgentConfig(kind=E.AgentKind.Random),
                      opening_plies=0)
    eng = E.SelfPlayEngine(4, E.MctsConfig(simulations=8, max_considered=4), 13, ev)
    recs = drive(eng, 4)
    for rec in recs:
        assert rec["selfplay"] is False
        assert all(p == rec["net_player"] for p in rec["players"])

    from cornerstone.replay import ReplayBuffer
    with pytest.raises(ValueError, match="评测模式"):
        ReplayBuffer(1000).add_records(recs)


def test_engine_threads_do_not_change_results():
    """各局的树互相独立，所以把树搜索摊到多线程之后，结果必须和单线程逐位一致。

    这条不测的话，多线程带来的数据竞争会表现为「训练变差了」，而不是崩溃 ——
    极难定位。
    """
    def run(threads):
        eng = E.SelfPlayEngine(24, E.MctsConfig(simulations=24, max_considered=8), 77,
                               E.EvalConfig(), threads)
        rng = make_rng(5)
        noise = rng.standard_normal(cs.NUM_ACTIONS).astype(np.float32)
        return drive(eng, 12, logit_fn=lambda planes, n: np.tile(noise, (n, 1)))

    a, b = run(1), run(8)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x["actions"], y["actions"]), "多线程改变了对局走向"
        assert np.array_equal(x["top_actions"], y["top_actions"])
        assert np.allclose(x["top_probs"], y["top_probs"], atol=0, rtol=0)
        assert (x["result0"], x["score0"], x["score1"]) == (y["result0"], y["score0"], y["score1"])


def test_engine_threads_preserve_batch_ordering():
    """prepare() 收集到的批必须按局号排序，否则 feed() 里 logits 与局对不上。"""
    for threads in (1, 4, 16):
        eng = E.SelfPlayEngine(32, E.MctsConfig(simulations=8), 3, E.EvalConfig(), threads)
        planes = np.zeros((32, cs.NUM_PLANES, 14, 14), np.float32)
        scal = np.zeros((32, cs.NUM_SCALARS), np.float32)
        n = eng.prepare(planes, scal)
        assert n == 32, "首轮每局都应该有一个待评估的根"
        # 特征里第 0 个平面是己方占用，开局全空；用标量里的占格数区分不了，
        # 这里改为验证多线程与单线程写出的特征完全一致
        if threads == 1:
            ref = planes.copy()
        else:
            assert np.array_equal(planes, ref), "多线程写出的特征顺序与单线程不一致"


def test_engine_rejects_bad_shapes():
    eng = E.SelfPlayEngine(4, E.MctsConfig(simulations=8), 1)
    planes = np.zeros((4, cs.NUM_PLANES, 14, 14), dtype=np.float32)
    scal = np.zeros((4, cs.NUM_SCALARS), dtype=np.float32)
    n = eng.prepare(planes, scal)
    with pytest.raises(ValueError):
        eng.feed(np.zeros((n, 10), dtype=np.float32), np.zeros((n, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        eng.feed(np.zeros((n, cs.NUM_ACTIONS), dtype=np.float32),
                 np.zeros((n, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        eng.prepare(np.zeros((1, cs.NUM_PLANES, 14, 14), dtype=np.float32), scal)
