"""CornerNet 与损失。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cornerstone as cs
from cornerstone.losses import policy_loss, score_loss, total_loss, value_loss
from cornerstone.model import ACTIONS, CornerNet, ModelConfig

SMALL = ModelConfig(dim=48, blocks=3, attn_every=2, heads=4)


@pytest.fixture(scope="module")
def net():
    return CornerNet(SMALL).eval()


def test_output_shapes(net):
    p = torch.randn(5, cs.NUM_PLANES, 14, 14)
    s = torch.randn(5, cs.NUM_SCALARS)
    pol, wdl, sc = net(p, s)
    assert pol.shape == (5, ACTIONS)
    assert wdl.shape == (5, 3)
    assert sc.shape == (5,)


def test_policy_head_zero_init_gives_uniform_prior():
    """策略头零初始化很重要：否则训练一开始 MCTS 就被一个随机的强先验带偏。"""
    m = CornerNet(SMALL).eval()
    pol, _, _ = m(torch.randn(3, cs.NUM_PLANES, 14, 14), torch.randn(3, cs.NUM_SCALARS))
    assert torch.equal(pol, torch.zeros_like(pol))


def test_token_view_is_really_zero_copy():
    """(B,196,D) 连续张量的 NCHW 视图必须是 channels_last，否则每个 block 要多两次大拷贝。"""
    b, d = 4, 48
    x = torch.randn(b, cs.NUM_CELLS, d)
    xc = x.view(b, 14, 14, d).permute(0, 3, 1, 2)
    assert xc.shape == (b, d, 14, 14)
    assert xc.is_contiguous(memory_format=torch.channels_last)
    assert xc.data_ptr() == x.data_ptr(), "产生了拷贝"
    back = xc.permute(0, 2, 3, 1).reshape(b, cs.NUM_CELLS, d)
    assert back.data_ptr() == x.data_ptr()
    assert torch.equal(back, x)


def test_action_layout_matches_engine_encoding(net):
    """策略头输出 (B,196,91) 后转置展平，必须对上 ori*196+cell 的编码。"""
    b = 2
    h = torch.randn(b, cs.NUM_CELLS, SMALL.dim)
    per_cell = net.policy(h)                       # (B, 196, 91)
    flat = per_cell.transpose(1, 2).reshape(b, ACTIONS)
    for ori in (0, 17, 90):
        for cell in (0, 95, 195):
            assert torch.allclose(flat[:, ori * cs.NUM_CELLS + cell], per_cell[:, cell, ori])


def test_forward_is_deterministic_in_eval(net):
    p = torch.randn(3, cs.NUM_PLANES, 14, 14)
    s = torch.randn(3, cs.NUM_SCALARS)
    with torch.no_grad():
        a = net(p, s)
        b = net(p, s)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_gradients_reach_every_parameter():
    m = CornerNet(SMALL).train()
    pol, wdl, sc = m(torch.randn(4, cs.NUM_PLANES, 14, 14), torch.randn(4, cs.NUM_SCALARS))
    (pol.sum() + wdl.sum() + sc.sum()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    # 策略头是零初始化的，但它的梯度依然应该非零
    assert not missing, f"这些参数没拿到梯度: {missing}"


# ---- 损失 ----

def make_batch(b=6, n_legal=50, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    legal = torch.zeros(b, ACTIONS, dtype=torch.bool)
    idx = torch.stack([torch.randperm(ACTIONS, generator=g)[:n_legal] for _ in range(b)])
    legal.scatter_(1, idx, True)

    top_actions = idx[:, :k].contiguous()
    raw = torch.rand(b, k, generator=g)
    probs = raw / raw.sum(dim=1, keepdim=True) * 0.7      # top-K 占 70%
    return {
        "legal": legal,
        "top_actions": top_actions,
        "top_probs": probs,
        "rest_prob": torch.full((b,), 0.3),
        "n_top": torch.full((b,), k, dtype=torch.long),
        "n_legal": torch.full((b,), n_legal, dtype=torch.long),
        "wdl": torch.randint(0, 3, (b,), generator=g),
        "score_diff": torch.randn(b, generator=g) * 0.2,
    }


def test_policy_loss_is_minimized_by_the_target_distribution():
    batch = make_batch()
    b, k = batch["top_probs"].shape
    n_legal = int(batch["n_legal"][0])

    # 精确复现目标分布：top-K 用给定概率，其余合法着法均摊 rest_prob
    target = torch.zeros(b, ACTIONS)
    target.scatter_(1, batch["top_actions"], batch["top_probs"])
    tail = batch["rest_prob"] / (n_legal - k)
    target += batch["legal"].float() * tail[:, None]
    target.scatter_(1, batch["top_actions"], batch["top_probs"])

    exact = torch.log(target.clamp_min(1e-30))
    loss_exact = policy_loss(exact, batch["legal"], batch["top_actions"],
                             batch["top_probs"], batch["rest_prob"],
                             batch["n_top"], batch["n_legal"])
    # 目标分布的熵就是交叉熵的下界
    ent = -(target * torch.log(target.clamp_min(1e-30))).sum(dim=1).mean()
    assert torch.allclose(loss_exact, ent, atol=1e-4)

    worse = policy_loss(torch.zeros(b, ACTIONS), batch["legal"], batch["top_actions"],
                        batch["top_probs"], batch["rest_prob"],
                        batch["n_top"], batch["n_legal"])
    assert worse > loss_exact + 1e-3


def test_policy_loss_has_no_nan_with_padded_slots():
    """补位槽的概率是 0，但动作编号必须仍指向合法着法，否则 0 * -inf = NaN。"""
    batch = make_batch(k=8)
    batch["n_top"] = torch.full((6,), 3, dtype=torch.long)   # 后 5 个槽是补位
    batch["top_probs"][:, 3:] = 0.0
    logits = torch.randn(6, ACTIONS)
    loss = policy_loss(logits, batch["legal"], batch["top_actions"], batch["top_probs"],
                       batch["rest_prob"], batch["n_top"], batch["n_legal"])
    assert torch.isfinite(loss)


def test_policy_loss_ignores_illegal_logits():
    """非法着法的 logit 无论多大都不该影响损失。"""
    batch = make_batch()
    logits = torch.randn(6, ACTIONS)
    args = (batch["legal"], batch["top_actions"], batch["top_probs"],
            batch["rest_prob"], batch["n_top"], batch["n_legal"])
    a = policy_loss(logits, *args)
    bumped = logits.masked_fill(~batch["legal"], 1e4)
    b = policy_loss(bumped, *args)
    assert torch.allclose(a, b, atol=1e-5)


def test_value_and_score_losses():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    assert value_loss(logits, torch.tensor([0, 1])) < 1e-3
    assert value_loss(logits, torch.tensor([2, 2])) > 5.0
    assert score_loss(torch.zeros(4), torch.zeros(4)) == 0.0
    assert score_loss(torch.zeros(4), torch.ones(4)) > 0


def test_total_loss_reports_all_parts():
    batch = make_batch()
    out = (torch.randn(6, ACTIONS), torch.randn(6, 3), torch.randn(6))
    loss, parts = total_loss(out, batch)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "policy", "value", "score", "wdl_acc", "policy_entropy"}
    # 策略熵不会超过 log(合法着法数)
    assert parts["policy_entropy"] <= np.log(int(batch["n_legal"][0])) + 1e-4


def test_to_param_dtype_preserves_parameter_objects():
    """降精度必须原地改 `param.data`，不能替换 Parameter 对象。

    换了对象的话，优化器持有的那批 fp32 master 就和模型脱钩了 —— 不报错，
    只是训练照跑而权重永远不动。这是低精度训练里最难查的一类失效。
    """
    net = CornerNet(SMALL)
    before = [p for p in net.parameters()]
    net.to_param_dtype()
    assert all(a is b for a, b in zip(before, net.parameters()))
    assert all(p.dtype is torch.bfloat16 for p in net.parameters())


def test_bf16_model_accepts_fp32_input_without_autocast():
    """CPU 上 autocast 是关的，而 web 有真实的 CPU 回退路径。

    没有 forward 入口那条 dtype 护栏的话，这里会直接报
    `Input type (float) and bias type (c10::BFloat16) should be the same`。
    """
    net = CornerNet(SMALL).eval().to_param_dtype()
    p = torch.randn(4, cs.NUM_PLANES, 14, 14)          # fp32 输入
    s = torch.randn(4, cs.NUM_SCALARS)
    with torch.no_grad():
        pol, wdl, sc = net(p, s)
    assert pol.shape == (4, ACTIONS)
    assert torch.isfinite(pol.float()).all()
