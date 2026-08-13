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


def test_default_simulations_is_the_product_setting():
    """`--simulations` 默认必须是 64（带搜索），不是 0（纯策略）。

    移除训练期周期评测之后，`net_arena.py` 是棋力结论的**唯一**来源，而且没有
    任何代码调用它 —— 全是人手敲命令行，所以「默认值」实际上就是大多数调用的
    取值。两种口径会给出**相反**的排名（v2-bf16 的终点档纯策略下排第 8、
    带搜索下排第 1），而打印出来的表长得一模一样，事后认不出手里这张是哪一种。
    """
    import re
    src = open(os.path.join(REPO, "tools", "net_arena.py")).read()
    m = re.search(r'"--simulations",\s*type=int,\s*default=(\d+)', src)
    assert m, "找不到 --simulations 的定义，这条测试本身该更新了"
    assert int(m.group(1)) == 64, "默认口径退回纯策略了 —— 那会让排名反过来"


# ---- net_arena 的方向自检：该拦的要拦，不该拦的不能拦，而且不能吃掉数据 ----

def test_direction_check_is_scoped_to_a_single_run():
    """「晚的档打得过早的档」只在**同一条跑内部**才是不变量。

    跨跑比较时它恰恰是被测的假设：`v4-bf16-long`（从第 10 万步续训到 22 万）
    的终点实测**打不过** `v4-bf16` 第 9 万步那档 —— 那是真结果，不是表颠倒。
    原实现不分跑，把这次完全正确的比较判成了「表颠倒」。
    """
    src = open(os.path.join(REPO, "tools", "net_arena.py"), encoding="utf-8").read()
    i = src.index("SPAN = 100_000")
    blk = src[i:i + 600]
    assert "len(runs) == 1" in blk, "步数跨度那条自检没有限定在同一条跑内"
    assert "os.path.dirname" in blk, "没有从 spec 推出跑名"


def test_results_are_written_before_the_direction_check():
    """自检必须排在**写盘之后**。

    它原先在写盘前 `SystemExit`，一次误报就把 28 对、1.5 小时的对局数据一起
    带走了 —— 而那些数据本身是好的。落盘是廉价且不可再生的，自检是可再议的，
    次序不能反。
    """
    src = open(os.path.join(REPO, "tools", "net_arena.py"), encoding="utf-8").read()
    assert src.index('已写入') < src.index("方向自检失败"), \
        "方向自检出现在写盘之前 —— 一次误报会把整场对局数据吃掉"
