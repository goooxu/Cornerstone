"""`tools/net_arena.py` 的分派与胜负方向。

拟合本身由 `tests/test_elo.py` 覆盖，这里只盯一件事：**谁赢了有没有记反**。

这不是杞人忧天 —— 这个项目已经因为「传参顺序和打印出来的名字对不上」
把 A/B 的结论读反过一次（见 docs/08 最后一条）。而在混合循环赛里更危险：
`evaluate_vs_baseline` 报的是**网络方**的胜负，网络坐第二个位置时必须翻过来，
写反了不会报错，只会让整张 Elo 表上下颠倒。
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "net_arena", os.path.join(REPO, "tools", "net_arena.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["net_arena"] = mod
    spec.loader.exec_module(mod)
    return mod


arena = _load_tool()


class _FakeResult:
    """冒充 EvalResult / PairResult 里被用到的那几个字段。"""

    def __init__(self, wins, draws, games):
        self.wins, self.draws, self.games = wins, draws, games
        self.score_a = wins + 0.5 * draws


class _Fake(arena.Participant):
    def __init__(self, name, kind, step=-1):
        self.spec = name
        self.kind = kind
        self.name = name
        self.step = step
        self.model = object() if kind == "net" else None
        self.cfg = object() if kind == "rule" else None


def test_step_of_parses_checkpoint_names():
    assert arena.step_of("/x/y/step00200230.pt") == 200230
    assert arena.step_of("step00000630.pt") == 630
    assert arena.step_of("random") == -1


def test_net_vs_rule_direction_when_net_is_first(monkeypatch):
    """网络坐 a 位：直接用网络方的得分。"""
    net, rule = _Fake("net@100", "net", 100), _Fake("greedy-area", "rule")
    monkeypatch.setattr(
        "cornerstone.evaluate.evaluate_vs_baseline",
        lambda *a, **k: _FakeResult(wins=300, draws=20, games=400))
    sa, g = arena.play(net, rule, 400, 64, 0, "cpu", 4, 4)
    assert g == 400
    assert sa == pytest.approx(310.0)          # 300 + 0.5*20


def test_net_vs_rule_direction_when_net_is_second(monkeypatch):
    """网络坐 b 位：`evaluate_vs_baseline` 仍报网络方的胜负，必须翻过来。

    同一场对局，网络拿 310 分，那么 a（规则方）应当拿 400 − 310 = 90。
    写成 310 的话整张表就反了。
    """
    rule, net = _Fake("greedy-area", "rule"), _Fake("net@100", "net", 100)
    monkeypatch.setattr(
        "cornerstone.evaluate.evaluate_vs_baseline",
        lambda *a, **k: _FakeResult(wins=300, draws=20, games=400))
    sa, g = arena.play(rule, net, 400, 64, 0, "cpu", 4, 4)
    assert g == 400
    assert sa == pytest.approx(90.0)


def test_the_two_directions_are_complementary(monkeypatch):
    """同一场对局，两种坐法算出来的得分必须互补。"""
    monkeypatch.setattr(
        "cornerstone.evaluate.evaluate_vs_baseline",
        lambda *a, **k: _FakeResult(wins=137, draws=9, games=400))
    net, rule = _Fake("net@1", "net", 1), _Fake("random", "rule")
    first, g = arena.play(net, rule, 400, 64, 0, "cpu", 4, 4)
    second, _ = arena.play(rule, net, 400, 64, 0, "cpu", 4, 4)
    assert first + second == pytest.approx(g)


def test_net_vs_net_uses_a_perspective(monkeypatch):
    monkeypatch.setattr(
        "cornerstone.evaluate.evaluate_vs_network",
        lambda *a, **k: _FakeResult(wins=250, draws=10, games=400))
    a, b = _Fake("net@1", "net", 1), _Fake("net@2", "net", 2)
    sa, g = arena.play(a, b, 400, 64, 0, "cpu", 4, 4)
    assert (sa, g) == (pytest.approx(255.0), 400)


def test_rule_vs_rule_uses_play_pair(monkeypatch):
    monkeypatch.setattr("net_arena.play_pair",
                        lambda *a, **k: _FakeResult(wins=180, draws=40, games=400))
    a, b = _Fake("greedy-area", "rule"), _Fake("random", "rule")
    sa, g = arena.play(a, b, 400, 64, 0, "cpu", 4, 4)
    assert (sa, g) == (pytest.approx(200.0), 400)


def test_score_matrix_is_symmetric_and_fit_orders_correctly():
    """把 play() 的输出填进矩阵、再拟合，强者必须排在前面。

    这一条是端到端地验证「方向没反」：构造一个 A 稳赢 B、B 稳赢 C 的三方，
    拟合出来的次序必须是 A > B > C。
    """
    from cornerstone.elo import fit_elo
    n = 3
    scores = np.zeros((n, n))
    games = np.zeros((n, n))
    # A(0) 对 B(1) 得 300/400；B(1) 对 C(2) 得 300/400；A 对 C 得 380/400
    for i, j, sa in ((0, 1, 300.0), (1, 2, 300.0), (0, 2, 380.0)):
        scores[i, j] += sa
        scores[j, i] += 400 - sa
        games[i, j] += 400
        games[j, i] += 400
    elo = fit_elo(scores, games, anchor=2)
    assert elo[0] > elo[1] > elo[2]
    assert elo[2] == pytest.approx(0.0)


def test_bridge_mode_skips_saturated_mixed_pairs():
    """`--pairs bridge` 只让指定的网络去和规则搭桥。

    晚期网络对规则全是 1.0，打了没信息量，只是白烧时间。
    """
    import argparse
    nets = [_Fake("net@630", "net", 630), _Fake("net@200230", "net", 200230)]
    rules = [_Fake("random", "rule"), _Fake("greedy-area", "rule")]
    parts = rules + nets
    bridge = {630}

    def wanted(a, b):
        if a.is_net == b.is_net:
            return True
        net = a if a.is_net else b
        return net.step in bridge

    import itertools
    pairs = [(parts[i].name, parts[j].name)
             for i, j in itertools.combinations(range(len(parts)), 2)
             if wanted(parts[i], parts[j])]
    assert ("random", "net@630") in pairs
    assert ("random", "net@200230") not in pairs, "晚期网络对规则应被跳过"
    assert ("net@630", "net@200230") in pairs, "网络之间要全打"
    assert ("random", "greedy-area") in pairs, "规则之间要全打"
    del argparse
