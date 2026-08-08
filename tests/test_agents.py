"""基线智能体与对局框架。"""

import numpy as np
import pytest

import cornerstone as cs
from cornerstone import _engine as E
from cornerstone.arena import BASELINES, play_pair
from conftest import make_rng

ALL_KINDS = [
    E.AgentKind.Random,
    E.AgentKind.GreedyArea,
    E.AgentKind.GreedyMobility,
    E.AgentKind.FlatMCTS,
]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_select_move_is_always_legal(kind):
    cfg = E.AgentConfig(kind=kind, rollouts=64)
    rng = make_rng(4)
    b = cs.Board()
    steps = 0
    while not b.terminal:
        a = E.select_move(b, cfg, int(rng.integers(1 << 62)))
        assert b.is_legal(a), f"{kind} 给出了非法着法"
        b.play(a)
        steps += 1
    assert steps > 0 and b.terminal


@pytest.mark.parametrize("name", list(BASELINES))
def test_registry_agents_play_a_full_game(name):
    cfg = BASELINES[name]
    if cfg.kind == E.AgentKind.FlatMCTS and cfg.rollouts > 1024:
        pytest.skip("预算太大，单测里不跑")
    rng = make_rng(6)
    b = cs.Board()
    while not b.terminal:
        b.play(E.select_move(b, cfg, int(rng.integers(1 << 62))))
    assert b.scores()[0] + b.scores()[1] > 0


def test_greedy_area_picks_largest_piece():
    cfg = E.AgentConfig(kind=E.AgentKind.GreedyArea)
    sizes = cs.piece_sizes()
    b = cs.Board()
    for _ in range(6):
        if b.terminal:
            break
        best = max(sizes[cs.decode_action(int(a))["piece"]] for a in b.legal_moves())
        a = E.select_move(b, cfg, 123)
        assert sizes[cs.decode_action(a)["piece"]] == best
        b.play(a)


def test_match_bookkeeping_is_consistent():
    a = BASELINES["greedy-area"]
    b = BASELINES["random"]
    r = E.play_match(a, b, 200, 42, 8, 4)
    assert r["games"] == 200
    assert r["wins_a"] + r["wins_b"] + r["draws"] == 200
    assert r["a_as_first"] == 100, "先后手必须逐局交换"
    assert r["a_wins_as_first"] + r["a_wins_as_second"] == r["wins_a"]
    assert 0 < r["score_a"] and 0 < r["score_b"]
    assert 0 < r["plies"] <= 200 * cs.MAX_PLIES


def test_match_is_reproducible_and_thread_count_invariant():
    a, b = BASELINES["greedy-mobility"], BASELINES["greedy-area"]
    r1 = E.play_match(a, b, 120, 99, 1, 4)
    r2 = E.play_match(a, b, 120, 99, 1, 4)
    assert r1 == r2, "同种子同线程数结果必须一致"
    # 每局种子只由局号决定，所以换线程数结果也不该变
    r3 = E.play_match(a, b, 120, 99, 16, 4)
    assert r1 == r3, "结果不该依赖线程划分"


def test_ladder_is_monotone_at_the_ends():
    """只验证差距足够大的几对，避免单测因为统计涨落而不稳。"""
    pairs = [
        ("greedy-area", "random", 0.80),
        ("greedy-mobility", "greedy-area", 0.70),
        ("flat-mcts-256", "random", 0.85),
    ]
    for strong, weak, floor in pairs:
        r = play_pair(strong, weak, BASELINES[strong], BASELINES[weak],
                      games=200, seed=5, threads=8, opening_plies=4)
        assert r.score_rate_a > floor, f"{strong} 对 {weak} 得分率只有 {r.score_rate_a:.3f}"


def test_flat_mcts_improves_with_budget():
    weak = E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=64)
    strong = E.AgentConfig(kind=E.AgentKind.FlatMCTS, rollouts=2048)
    r = E.play_match(strong, weak, 200, 11, 16, 4)
    rate = (r["wins_a"] + 0.5 * r["draws"]) / r["games"]
    assert rate > 0.60, f"预算大 32 倍却只有 {rate:.3f} 得分率"


def test_play_pair_rejects_odd_game_count():
    with pytest.raises(ValueError):
        play_pair("random", "random", BASELINES["random"], BASELINES["random"],
                  games=101, seed=1, threads=1, opening_plies=0)


def test_tie_breaking_is_randomized():
    """同分着法之间必须随机选。否则贪心智能体永远下出同一局，
    arena 里跑几百局等于只跑了一局。"""
    cfg = BASELINES["greedy-area"]
    b = cs.Board()
    picks = {E.select_move(b, cfg, s) for s in range(200)}
    assert len(picks) > 1, "同分着法没有被打散"
    # 而且每个都得是最大棋子
    sizes = cs.piece_sizes()
    assert {sizes[cs.decode_action(a)["piece"]] for a in picks} == {5}


def test_opening_randomization_changes_the_distribution():
    a = BASELINES["greedy-mobility"]
    b = BASELINES["greedy-area"]
    r0 = E.play_match(a, b, 200, 1, 1, 0)
    r6 = E.play_match(a, b, 200, 1, 1, 6)
    for r in (r0, r6):
        assert r["wins_a"] + r["wins_b"] + r["draws"] == 200
    assert r0["plies"] != r6["plies"]
    # 随机开局会削弱强者的优势
    rate0 = (r0["wins_a"] + 0.5 * r0["draws"]) / 200
    rate6 = (r6["wins_a"] + 0.5 * r6["draws"]) / 200
    assert rate0 > 0.5 and rate6 > 0.5 and rate0 != rate6


# ---- 开局配对 ----------------------------------------------------------
#
# 第 2k 与第 2k+1 局用**同一个随机开局**，只是执先方相反。
# 不这么做的话，某个碰巧利于一方的开局只会被一方碰上 —— 400 局就是 400 个
# 互不相干的开局，公平性只在期望意义上成立。

def test_self_match_scores_exactly_half():
    """同一个智能体自我对局，得分率必须**精确**是 0.5。

    这是开局配对最强的判据：一对里两局开局相同、rng 流相同、双方策略相同，
    所以两局逐手完全一样，只是执先方相反 —— 一胜一负必然抵消。
    没有配对的话这里只会是 0.5 ± 0.025，而且胜负数不对称。
    """
    from cornerstone.arena import BASELINES, play_pair
    for name in ("random", "greedy-area", "greedy-mobility"):
        cfg = BASELINES[name]
        r = play_pair(name, name, cfg, cfg, 200, 11, 4, 4)
        assert r.score_rate_a == 0.5, f"{name} 自我对局得分率 {r.score_rate_a}，开局配对没生效"
        assert r.wins_a == r.wins_b, f"{name} 胜负数不对称：{r.wins_a} vs {r.wins_b}"


def test_paired_openings_still_differ_between_pairs():
    """配对是「相邻两局相同」，不是「所有局都相同」。

    如果把种子写成常数，400 局会变成同一局打 400 遍 —— 得分率同样是 0.5，
    上一个测试抓不到。这里用两个强弱悬殊的智能体：真的只打一个开局的话，
    得分率会钉在 0 或 1，不可能落在中间。
    """
    from cornerstone.arena import BASELINES, play_pair
    r = play_pair("greedy-area", "greedy-mobility",
                  BASELINES["greedy-area"], BASELINES["greedy-mobility"], 200, 5, 4, 4)
    assert 0.02 < r.score_rate_a < 0.98, f"得分率 {r.score_rate_a} 像是所有局共用一个开局"
    assert r.mean_plies > 20
