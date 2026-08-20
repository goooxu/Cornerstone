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


def _build(acts, sym=None):
    t = len(acts)
    off = np.array([0, t], dtype=np.int32)
    ply = np.arange(t, dtype=np.int32)
    planes = np.empty((t, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), np.float32)
    scalars = np.empty((t, E.NUM_SCALARS), np.float32)
    legal = np.empty((t, E.NUM_ACTIONS), np.uint8)
    owner = np.empty((t, E.NUM_CELLS), np.int8)
    syms = None if sym is None else np.full(t, sym, np.int8)
    E.build_batch(acts, off, ply, off, syms, planes, scalars, legal, 1, owner=owner)
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
