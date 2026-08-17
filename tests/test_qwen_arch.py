"""Qwen3 形状骨干（`arch="qwen"`）。

分两组：**纯数学的 RoPE 部分不依赖 ModelConfig**，任何时候都能跑；
整网那组要等 `ModelConfig.arch` 落地。

每条都盯一个「不报错但不对」——这套骨干里位置编码和注意力方向都属于
「配错了照样能训、只是训出来的东西不对」，从 loss 上完全看不出来。
"""

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cornerstone.qwen_block import (          # noqa: E402
    apply_rope2d, rope2d_tables,
)

BOARD, CELLS, HEAD_DIM, EXTRA = 14, 196, 128, 44


# --------------------------------------------------------------- 2D RoPE

@pytest.fixture(scope="module")
def tables():
    return rope2d_tables(CELLS, BOARD, HEAD_DIM, EXTRA)


def _dot(base, cos, sin, i, j):
    """同一个向量放在格 i 和格 j 上、旋转后的内积 —— RoPE 表达的相对位置信号。"""
    qi = apply_rope2d(base, cos[i:i + 1], sin[i:i + 1])
    kj = apply_rope2d(base, cos[j:j + 1], sin[j:j + 1])
    return float((qi * kj).sum())


def test_rope_tables_cover_board_plus_extra_tokens(tables):
    cos, sin = tables
    assert cos.shape == sin.shape == (CELLS + EXTRA, HEAD_DIM)


def test_rope_is_translation_invariant(tables):
    """位移相同 -> 内积相同。这是 RoPE 的定义性质，劈成两半之后仍须成立。"""
    cos, sin = tables
    torch.manual_seed(0)
    base = torch.randn(1, 1, 1, HEAD_DIM)
    vals = [_dot(base, cos, sin, a, b) for a, b in ((0, 1), (5, 6), (100, 101))]
    assert max(vals) - min(vals) < 1e-3, f"同一位移给出不同内积：{vals}"


def test_rope_is_two_dimensional_not_raster(tables):
    """**这条是选 2D 而不是 1D 的全部理由。**

    按行优先展开时，(0,13) 与 (1,0) 的序列下标只差 1，棋盘上却隔着整整一行。
    1D RoPE 会把这两个当成邻居 —— 而且照样能训，只是位置编码是错的。
    """
    cos, sin = tables
    torch.manual_seed(0)
    base = torch.randn(1, 1, 1, HEAD_DIM)
    across_row = _dot(base, cos, sin, 13, 14)      # (0,13) -> (1,0)
    true_adj = _dot(base, cos, sin, 0, 1)          # (0,0)  -> (0,1)
    assert abs(across_row - true_adj) > 1e-2, "位置编码退化成一维了"


def test_rope_does_not_mix_row_and_col_axes(tables):
    """行、列各占 head_dim 的一半，两半必须各自配对旋转。

    整条 head_dim 一起转的话，行的后半会和列的前半配成一对，两个坐标轴就串了。
    """
    cos, sin = tables
    torch.manual_seed(0)
    base = torch.randn(1, 1, 1, HEAD_DIM)
    assert abs(_dot(base, cos, sin, 0, 14) - _dot(base, cos, sin, 0, 1)) > 1e-3, \
        "只改行与只改列给出了相同的编码"


def test_rope_is_identity_off_board(tables):
    """棋子/标量 token 不在棋盘上，硬塞一个坐标是假的 —— 角度取 0，RoPE 退化成恒等。"""
    cos, sin = tables
    torch.manual_seed(0)
    base = torch.randn(1, 1, 1, HEAD_DIM)
    off = CELLS + 3
    assert torch.allclose(apply_rope2d(base, cos[off:off + 1], sin[off:off + 1]), base)


def test_rope_requires_head_dim_divisible_by_four():
    """劈两半、每半再两两配对 —— 不整除的话会静默切出错位的张量。"""
    with pytest.raises(AssertionError):
        rope2d_tables(CELLS, BOARD, 130, EXTRA)


# ------------------------------------------------------------------ 整网
#
# 下面这组要 `ModelConfig.arch` 落地之后才能跑。

def _qwen_cfg(**kw):
    from cornerstone.model import ModelConfig
    if "arch" not in ModelConfig.__dataclass_fields__:
        pytest.skip("ModelConfig 还没有 arch 字段")
    base = dict(arch="qwen", dim=128, blocks=2, heads=4, kv_heads=2,
                head_dim=32, intermediate=256)
    base.update(kw)
    return ModelConfig(**base)


def _inputs(b=8):
    from cornerstone import _engine as E
    torch.manual_seed(7)
    return (torch.randn(b, E.NUM_PLANES, E.BOARD_N, E.BOARD_N),
            torch.randn(b, E.NUM_SCALARS))


def test_qwen_head_shapes_match_the_poly_backbone():
    """换骨干不改接口：三个头的形状必须逐字不变，否则下游全要跟着改。"""
    from cornerstone import _engine as E
    from cornerstone.model import CornerNet
    m = CornerNet(_qwen_cfg()).eval()
    with torch.no_grad():
        pol, wdl, sc = m(*_inputs())
    assert pol.shape == (8, E.NUM_ACTIONS)
    assert wdl.shape == (8, 3)
    assert sc.shape == (8,)


def test_attention_is_bidirectional():
    """因果 mask 会让第 k 个格看不到它右边和下边的棋盘 —— 纯粹的信息损失。

    判据：改一个格的输入，别的格的输出也该跟着变（信息传得过去）。

    **不能用 policy 头判**：它是零初始化的（给 MCTS 均匀先验），输出恒为 0，
    比什么都相等 —— 这条测试第一版就是这么写的，测了个寂寞。用 wdl 头。
    """
    from cornerstone.model import CornerNet
    m = CornerNet(_qwen_cfg()).eval()
    planes, scalars = _inputs(b=8)
    with torch.no_grad():
        a = m(planes, scalars)[1]                      # wdl，不是 policy
        sw = planes.clone()
        sw[:, :, 0, 0] += 3.0                          # 只动左上角那一个格
        b = m(sw, scalars)[1]
    assert not torch.allclose(a, b), "改了一个格，输出没变 —— 信息没传出去"


def test_rope_tables_are_not_persistent_buffers():
    """持久 buffer 两头出事：`pool._layout()` 只遍历参数、`load_weights()` 又对
    未知键抛错。RoPE 表是 cfg 的确定性函数，每个 worker 各建一份即可。
    """
    from cornerstone.model import CornerNet
    m = CornerNet(_qwen_cfg())
    keys = [k for k in m.state_dict() if "cos" in k or "sin" in k or "rope" in k]
    assert not keys, f"RoPE 表进了 state_dict：{keys}"


def test_layout_covers_every_parameter():
    """`pool._layout()` 靠「模型没有 buffer」这个前提把权重打平分发。"""
    from cornerstone.model import CornerNet
    from cornerstone.pool import _layout
    m = CornerNet(_qwen_cfg())
    assert sum(x[3] for x in _layout(m)) == sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
def test_every_precision_builds_and_runs(precision):
    """量化 GEMM 要求两个维都被 32 整除 —— 序列长 240、hidden/intermediate 都要过关。"""
    if precision != "bf16" and not torch.cuda.is_available():
        pytest.skip("量化路径要 GPU")
    from cornerstone.model import CornerNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CornerNet(_qwen_cfg(dim=256, heads=4, kv_heads=2, head_dim=64,
                            intermediate=512, precision=precision)).to(dev)
    m.to_param_dtype().eval()
    planes, scalars = _inputs()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        pol, _, _ = m(planes.to(dev), scalars.to(dev))
    assert torch.isfinite(pol).all()
