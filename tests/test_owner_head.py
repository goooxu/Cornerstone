"""逐格归属头（`ModelConfig.owner_head`）。

分两组：**目标算得对不对**（引擎侧）和**头接得对不对**（模型侧）。

前一组是重点。归属目标算错了训练照样跑、loss 照样降，从任何曲线上都看不出来 ——
和特征里放错 `ply` 是同一类故障。所以这里不只测「能跑」，而是拿三条**互相独立的
不变量**去卡它：与终局比分一致、与当前局面自洽、在四个对称变换下同变。
"""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone import _engine as E          # noqa: E402


# ------------------------------------------------------------------ 造一局棋

def _random_game(seed: int = 7):
    rng = np.random.default_rng(seed)
    b = E.Board()
    acts = []
    while not b.terminal:
        mv = b.legal_moves()
        acts.append(int(rng.choice(mv)) if len(mv) else -1)
        b.play(acts[-1])
    return np.asarray(acts, dtype=np.int32), b.score(0), b.score(1)


def _build(acts, sym=None, mobility=True):
    """可达度平面默认**关**（算它要两次全量走法生成，自博弈慢 27%），
    所以要测它就得显式打开。这里默认 True 是因为本文件的可达度那几条都要它；
    `test_mobility_is_off_by_default` 专门测关掉时的行为。"""
    t = len(acts)
    off = np.array([0, t], dtype=np.int32)
    ply = np.arange(t, dtype=np.int32)
    planes = np.empty((t, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), np.float32)
    scalars = np.empty((t, E.NUM_SCALARS), np.float32)
    legal = np.empty((t, E.NUM_ACTIONS), np.uint8)
    owner = np.empty((t, E.NUM_CELLS), np.int8)
    syms = None if sym is None else np.full(t, sym, np.int8)
    E.build_batch(acts, off, ply, off, syms, planes, scalars, legal, 1, owner=owner,
                  with_mobility=mobility)
    return planes, owner


# ------------------------------------------------------------- 目标算得对不对

def test_owner_counts_match_the_final_score():
    """己方/对方的格数必须正好是终局比分，空格数是剩下的那些。

    这条把归属头和已有的 `score_diff` 目标绑在一起 —— 占格数差正是逐格归属的求和，
    一个算错另一个会露馅。
    """
    acts, s0, s1 = _random_game()
    _, owner = _build(acts)
    mine = (owner == 1).sum(1)
    theirs = (owner == 2).sum(1)
    empty = (owner == 0).sum(1)

    assert set(mine.tolist()) == {s0, s1}, "己方格数没有在两个座位的比分之间取值"
    assert set((mine + theirs).tolist()) == {s0 + s1}
    assert set(empty.tolist()) == {E.NUM_CELLS - s0 - s1}


def test_owner_is_seat_relative_not_absolute():
    """**座位相对**：以该样本当时的行棋方为「己方」，与 9 个输入平面同一口径。

    写成绝对座位的话，rot180 与反对角变换会把己方起始格映射到另一个座位，
    增广就产生了分布外样本 —— 特征里不放 `ply` 正是同一个理由。

    判据：相邻两手要么标签不变（有人停手），要么 1 与 2 整体互换。
    """
    acts, _, _ = _random_game()
    _, owner = _build(acts)
    for t in range(len(owner) - 1):
        a, b = owner[t], owner[t + 1]
        swapped = np.where(a == 0, 0, 3 - a)
        assert np.array_equal(a, b) or np.array_equal(swapped, b), \
            f"第 {t} 手到第 {t+1} 手既不是不变也不是 1<->2 互换"


def test_already_placed_stones_own_their_cells():
    """当下已经落子的格，终局必然还归那一方 —— 棋子落下就不会再动。

    这条把归属目标和**同一批样本的输入平面**对上了：平面 0 是己方占格。
    如果归属是从别的时刻、别的局面算出来的，这里立刻炸。
    """
    acts, _, _ = _random_game()
    planes, owner = _build(acts)
    for t in range(len(owner)):
        mine_now = planes[t, 0].reshape(-1) > 0.5
        opp_now = planes[t, 1].reshape(-1) > 0.5
        assert np.all(owner[t][mine_now] == 1), f"第 {t} 手：己方已落的子终局不归己方"
        assert np.all(owner[t][opp_now] == 2), f"第 {t} 手：对方已落的子终局不归对方"


@pytest.mark.parametrize("sym", [1, 2, 3])
def test_owner_follows_the_same_cell_permutation_as_the_planes(sym):
    """增广时归属目标必须跟着**同一张** cell 置换表搬运。

    走两张不同的表（哪怕只差一个转置）不会报错，只会让标签和棋盘对不上，
    而训练曲线完全正常。
    """
    acts, _, _ = _random_game()
    planes0, owner0 = _build(acts)
    planes_s, owner_s = _build(acts, sym=sym)

    cmap = np.asarray(E.sym_cell_table(), dtype=np.int64)[sym]
    want = np.empty_like(owner0)
    want[:, cmap] = owner0
    assert np.array_equal(want, owner_s), "归属没按 sym_cell_table 搬运"

    # 与平面走同一张表 —— 只测归属自己自洽是不够的
    p0 = planes0[:, 0].reshape(len(acts), -1)
    ps = planes_s[:, 0].reshape(len(acts), -1)
    want_p = np.empty_like(p0)
    want_p[:, cmap] = p0
    assert np.allclose(want_p, ps), "归属与特征平面用了不同的置换表"


def test_owner_is_optional_and_old_positional_calls_still_work():
    """`owner` 是可选出参，且排在 `threads` 之后 —— 老的位置调用一个字都不用改。

    第一版把它插在 `threads` 前面，`build_batch(..., legal, 8)` 于是把 8 当成
    owner 数组去 cast，5 个现存单测当场全红。
    """
    acts, _, _ = _random_game()
    t = len(acts)
    off = np.array([0, t], dtype=np.int32)
    ply = np.arange(t, dtype=np.int32)
    planes = np.empty((t, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), np.float32)
    scalars = np.empty((t, E.NUM_SCALARS), np.float32)
    legal = np.empty((t, E.NUM_ACTIONS), np.uint8)
    E.build_batch(acts, off, ply, off, None, planes, scalars, legal, 4)   # 不要 owner


# ------------------------------------------------------------- 头接得对不对

def _cfg(**kw):
    from cornerstone.model import ModelConfig
    base = dict(dim=64, blocks=2, heads=4, attn_every=2)
    base.update(kw)
    return ModelConfig(**base)


def test_default_config_is_bit_identical_to_before():
    """**默认必须关。**

    `load_weights` 对缺键和多键都抛 KeyError，`build_model_from_release` 也拿
    missing 去和 quant 比对 —— 无条件加一个头会让三把尺子和所有历史 checkpoint
    立刻装不进去。
    """
    from cornerstone.model import CornerNet, ModelConfig
    assert ModelConfig().owner_head is False
    m = CornerNet(_cfg())
    assert m.owner is None
    assert not [k for k in m.state_dict() if k.startswith("owner.")]


def test_enabling_adds_only_the_head():
    from cornerstone.model import CornerNet
    off = set(CornerNet(_cfg()).state_dict())
    on = set(CornerNet(_cfg(owner_head=True)).state_dict())
    assert on - off == {"owner.weight", "owner.bias"}
    assert not off - on


def test_forward_contract_is_unchanged_by_default():
    """`pol, wdl, sc = model(...)` 这个三元组解包散布在 4 个推理调用点上
    （pool / evaluate / web / requantize）。默认必须仍是三项。
    """
    import torch
    from cornerstone.model import CornerNet
    x = torch.randn(2, E.NUM_PLANES, E.BOARD_N, E.BOARD_N)
    s = torch.randn(2, E.NUM_SCALARS)

    m = CornerNet(_cfg(owner_head=True)).eval()
    with torch.no_grad():
        assert len(m(x, s)) == 3, "开了头也不该改变默认返回"
        out = m(x, s, with_owner=True)
    assert len(out) == 4
    assert out[3].shape == (2, E.NUM_CELLS, 3)


def test_owner_loss_is_finite_and_scales_with_weight():
    import torch
    from cornerstone.losses import owner_loss
    torch.manual_seed(0)
    logits = torch.randn(3, E.NUM_CELLS, 3)
    target = torch.randint(0, 3, (3, E.NUM_CELLS))
    lo = owner_loss(logits, target)
    assert torch.isfinite(lo) and lo > 0
    # 对格取平均 —— 量级应当和一个普通三分类交叉熵可比，w_owner 才好当权重调
    assert 0.5 < float(lo) < 3.0


def test_both_training_steps_pass_with_owner():
    """**训练步有两份实现**：单卡在 `train.py`，多卡在 `pool.py`。

    加这个头时只改了 `train.py` 那份，于是 `own-bf16` 起跑后指标里根本没有
    owner 项 —— 而模型是**建对了**的（`spec["model_cfg"]` 带着 `owner_head`，
    那个 Linear 存在、也在优化器里），只是从来没被要求输出。
    训练照跑、loss 照降、参数量和 state_dict 的键都对，
    从任何中间产物上都看不出来；跑完 3 小时对上尺子只会得到
    「归属头没有效果」—— 而它根本没生效过。

    所以这条不测行为，测两份实现都把开关透传下去了。
    """
    import re
    for name in ("cornerstone/train.py", "cornerstone/pool.py"):
        src = open(os.path.join(REPO, name), encoding="utf-8").read()
        body = src[src.index("total_loss(") - 2000:src.index("total_loss(") + 200]
        assert "with_owner" in body, f"{name} 的训练步没有把 with_owner 传下去"


# ------------------------------------------------------------- 策略头隐藏层

def test_policy_hidden_defaults_off_and_keeps_key_names():
    """默认 0：键名必须还是 `policy.weight` —— 一改成 Sequential 就变成
    `policy.0.weight`，三把尺子和所有历史 checkpoint 立刻装不进去。
    """
    from cornerstone.model import CornerNet, ModelConfig
    assert ModelConfig().policy_hidden == 0
    keys = set(CornerNet(_cfg()).state_dict())
    assert "policy.weight" in keys and "policy.0.weight" not in keys


def test_policy_hidden_preserves_the_uniform_prior():
    """**零初始化的语义不能丢。**

    策略头零初始化是为了训练一开始就是均匀先验，不给 MCTS 一个随机的强先验。
    加隐藏层后 `self.policy.weight` 不再存在，`reset_parameters` 必须改成
    只零**最后一层** —— 漏掉的话初始先验变成随机的强先验，
    而这件事不报错、只是让早期自博弈被带偏，从 loss 上看不出来。
    """
    import torch
    from cornerstone.model import CornerNet
    for hidden in (0, 128):
        m = CornerNet(_cfg(policy_hidden=hidden)).eval()
        x = torch.randn(2, E.NUM_PLANES, E.BOARD_N, E.BOARD_N)
        s = torch.randn(2, E.NUM_SCALARS)
        with torch.no_grad():
            pol = m(x, s)[0]
        assert torch.all(pol == 0), f"policy_hidden={hidden} 的初始先验不是均匀的"


def test_policy_hidden_grows_only_the_policy_head():
    from cornerstone.model import CornerNet
    a = CornerNet(_cfg())
    b = CornerNet(_cfg(policy_hidden=128))
    def by_head(m):
        g = {}
        for n, p in m.named_parameters():
            g[n.split(".")[0]] = g.get(n.split(".")[0], 0) + p.numel()
        return g
    ga, gb = by_head(a), by_head(b)
    assert gb["policy"] > ga["policy"]
    for k in ga:
        if k != "policy":
            assert ga[k] == gb[k], f"{k} 也跟着变了，应该只动策略头"


# ------------------------------------------------------------- 可达度输入平面

def _mobility_by_hand(board):
    """独立重算「当前行棋方的可达度」—— 不碰 features()，只用公开的着法 API。"""
    n = E.BOARD_N
    cnt = np.zeros(n * n, np.int64)
    for a in board.legal_moves():
        for (r, c) in E.decode_action(a)["cells"]:
            cnt[r * n + c] += 1
    return (np.minimum(cnt, 64) / 64.0).astype(np.float32)


def test_mobility_plane_matches_an_independent_recount():
    """平面 9 必须与「枚举合法着法、逐格累加」逐位一致。

    这是唯一真正能证伪计数逻辑的测试：漏一类着法、把 anchor 当成覆盖格、
    归一化写错——三种都不会报错，只会喂给网络一个悄悄失真的场。
    """
    rng = np.random.default_rng(11)
    b = E.Board()
    acts, want = [], []
    while not b.terminal:
        want.append(_mobility_by_hand(b))
        mv = b.legal_moves()
        acts.append(int(rng.choice(mv)) if len(mv) else -1)
        b.play(acts[-1])
    planes, _ = _build(np.asarray(acts, np.int32))
    got = planes[:, E.NUM_PLANES - 2].reshape(len(want), -1)
    assert np.allclose(got, np.asarray(want)), "可达度平面与独立重算不一致"


def test_legacy_planes_are_untouched_by_the_extension():
    """**平面顺序是兼容性契约。**

    常数平面必须还钉在下标 8。它原先写成 `NUM_PLANES-1`，扩容时会跟着漂到
    下标 10 —— 而这不报任何错：老 checkpoint 照样加载、照样推理，
    只是切出来的第 9 个平面从恒 1 变成恒 0，棋力莫名其妙掉一截。
    """
    rng = np.random.default_rng(3)
    b = E.Board()
    acts = []
    while not b.terminal:
        mv = b.legal_moves()
        acts.append(int(rng.choice(mv)) if len(mv) else -1)
        b.play(acts[-1])
    planes, _ = _build(np.asarray(acts, np.int32))
    assert np.all(planes[:, 8] == 1.0), "常数平面不在下标 8 了"
    # 前两个平面就是双方占格，拿 legal/occupancy 之外的独立口径核一下形状即可
    assert planes.shape[1] == E.NUM_PLANES >= 11


def test_mobility_support_is_inside_the_allowed_region():
    """可达度的支撑集必须落在该方的可落区内 —— 放不下的格不可能被覆盖。

    这条对两个平面都成立，所以能同时守住「传错玩家」这种错法。
    """
    rng = np.random.default_rng(5)
    b = E.Board()
    acts = []
    while not b.terminal:
        mv = b.legal_moves()
        acts.append(int(rng.choice(mv)) if len(mv) else -1)
        b.play(acts[-1])
    planes, _ = _build(np.asarray(acts, np.int32))
    t = planes.shape[0]
    for mob, allowed, who in ((9, 2, "己方"), (10, 4, "对方")):
        sup = planes[:, mob].reshape(t, -1) > 0
        ok = planes[:, allowed].reshape(t, -1) > 0.5
        assert not (sup & ~ok).any(), f"{who}可达度落到了可落区之外"


def test_old_models_ignore_the_new_planes():
    """`in_planes` 默认 9：老模型拿到 11 个平面时必须切掉多的两个，
    且结果与只喂 9 个平面**逐位相同**。三把尺子靠这条继续可用。
    """
    import torch
    from cornerstone.model import CornerNet, ModelConfig
    assert ModelConfig().in_planes == 9
    m = CornerNet(_cfg()).eval()
    assert m.stem.in_channels == 9
    x = torch.randn(2, E.NUM_PLANES, E.BOARD_N, E.BOARD_N)
    s = torch.randn(2, E.NUM_SCALARS)
    with torch.no_grad():
        assert torch.equal(m(x, s)[0], m(x[:, :9], s)[0])


def test_mobility_is_off_by_default_and_costs_nothing_when_off():
    """**默认必须关。**

    可达度平面实测只值 +7.4 ± 14.3（不显著），却让自博弈慢 27% ——
    features() 无条件多算两遍全量走法生成，而每个待评估叶子都要调它一次。
    这个回归在 mobp/data2/data4 三条跑上白付了几小时机时才被查出来。

    关掉时前 9 个平面必须与开着时**逐位相同** —— 否则就不是「省掉多余计算」，
    而是悄悄改变了模型的输入。
    """
    acts, _, _ = _random_game()
    on, _ = _build(acts, mobility=True)
    off, _ = _build(acts, mobility=False)
    assert np.array_equal(on[:, :9], off[:, :9]), "关掉可达度改变了前 9 个平面"
    assert not off[:, 9].any() and not off[:, 10].any(), "关掉时平面 9/10 应为全零"
    assert on[:, 9].any(), "开启时平面 9 应非零"

    # 不传参数 = 关。引擎默认产 9 平面的语义，老模型才不会被静默喂零。
    t = len(acts)
    o = np.array([0, t], dtype=np.int32)
    pl = np.empty((t, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), np.float32)
    sc = np.empty((t, E.NUM_SCALARS), np.float32)
    lg = np.empty((t, E.NUM_ACTIONS), np.uint8)
    E.build_batch(acts, o, np.arange(t, dtype=np.int32), o, None, pl, sc, lg, 1)
    assert not pl[:, 9].any(), "默认应当是关的"
