"""门控冠军：自博弈的生成器是**被实测出来的**最强档。

裸自博弈里没有任何机制保证生成数据的那份权重不比历史最优差 —— 实测两条腿都在
训练量的 73~80% 处见顶，之后终点比峰值低 19~22 Elo，而那 20~27% 的训练数据
全是由一个正在退化的策略产出的。门控给这条链加一个棘轮。

这里守的都是「不报错但不对」那一类，其中最凶的一条是**各 rank 的 learner
必须逐位相同** —— 晋升是各工作进程本地做的，一旦不成立，四张卡的冠军会各走各的，
表现只是「自博弈数据来自四个不同的网络」，从任何日志上都看不出来。
"""

import os

import numpy as np
import pytest
import torch

from cornerstone.model import CornerNet, ModelConfig
from cornerstone.train import TrainConfig, Trainer

SMALL = ModelConfig(dim=32, blocks=2, attn_every=0)


def _cfg(tmp_path, **kw) -> TrainConfig:
    # selfplay_devices 必须显式给 "cpu"：留空的话 visible_devices 会把机器上
    # 所有可见 GPU 都算进来，于是单卡用例在 4 卡机上悄悄走了工作池路径
    base = dict(exp="t", run_dir=str(tmp_path / "run"), hot_dir=str(tmp_path / "hot"),
                dim=32, blocks=2, attn_every=0, device="cpu", selfplay_devices="cpu",
                replay_capacity=1000, batch_size=8, parallel_games=8)
    base.update(kw)
    return TrainConfig(**base)


# ---- 晋升的原地性 ----

def test_promote_keeps_parameter_objects():
    """晋升必须**原地**拷贝。

    重新赋值模块（champ = copy.deepcopy(model) 之类）会让 champ 那份编译图失效 ——
    下一轮自博弈重编译几十秒 —— 并且把 state_dict 的键变成 `_orig_mod.*`，
    当场破坏 checkpoint 契约。`load_state_dict` 是就地 copy_，Parameter 对象不变。
    """
    torch.manual_seed(0)
    learner = CornerNet(SMALL)
    champ = CornerNet(SMALL)
    before = [id(p) for p in champ.parameters()]

    with torch.no_grad():
        for p in learner.parameters():
            p.add_(1.0)
    champ.load_state_dict(learner.state_dict())

    assert [id(p) for p in champ.parameters()] == before
    for a, b in zip(champ.parameters(), learner.parameters()):
        assert torch.equal(a, b)
    assert not any(k.startswith("_orig_mod") for k in champ.state_dict())


def test_promote_copies_buffers_too():
    """位置嵌入之外还有 buffer（各 RMSNorm 没有，但结构一变就可能有）。

    只拷 parameters 不拷 buffers 是个典型的静默错法。
    """
    torch.manual_seed(0)
    learner, champ = CornerNet(SMALL), CornerNet(SMALL)
    with torch.no_grad():
        for b in learner.buffers():
            b.add_(1)
    champ.load_state_dict(learner.state_dict())
    for a, b in zip(champ.buffers(), learner.buffers()):
        assert torch.equal(a, b)


# ---- 生成器是谁 ----

def test_champion_is_the_selfplay_generator(tmp_path):
    """开门控时驱动必须绑在 champ 上，不是 learner。

    绑反了不会报错 —— 只是门控完全失效，训练退化成裸自博弈，
    而 gate_score 照样有数（learner 打 champ，两者是同一个对象，恒 0.5）。
    """
    tr = Trainer(_cfg(tmp_path, gate_enabled=True))
    drv = tr.make_driver()
    assert tr._champ is not None
    assert drv.model is tr._champ
    assert drv.model is not tr.model


def test_no_champion_when_gate_disabled(tmp_path):
    """不开门控时不该多出一份模型 —— 行为要与改动前逐位相同。"""
    tr = Trainer(_cfg(tmp_path, gate_enabled=False))
    drv = tr.make_driver()
    assert tr._champ is None
    assert drv.model is tr.model


def test_promote_updates_the_generator(tmp_path):
    """晋升之后，自博弈驱动看到的权重必须真的变了。"""
    tr = Trainer(_cfg(tmp_path, gate_enabled=True))
    drv = tr.make_driver()
    with torch.no_grad():
        for p in tr.model.parameters():
            p.add_(0.5)
    assert not torch.equal(next(drv.model.parameters()), next(tr.model.parameters()))
    tr.promote()
    for a, b in zip(drv.model.parameters(), tr.model.parameters()):
        assert torch.equal(a, b)


# ---- 门控本身 ----

def test_gate_score_is_exactly_half_against_itself(tmp_path):
    """同一份权重对自己，成对开局下得分率必须**精确**等于 0.5。

    一对里两局是同一个开局、执先方相反，且双方是同一个网络 —— 逐手完全一样，
    必然一胜一负抵消。这条判据和规则基线那条同源（见 docs/03），
    它一旦不成立，说明开局配对或先后手分配坏了，整个门控的数就没有意义。
    """
    tr = Trainer(_cfg(tmp_path, gate_enabled=True, gate_games=16,
                      gate_simulations=4, engine_threads_per_gpu=2))
    tr.make_driver()
    score, n = tr.gate(seed=3)
    assert n == 16
    assert score == pytest.approx(0.5, abs=1e-9)


def test_gate_compares_learner_against_champion(tmp_path, monkeypatch):
    """门控必须拿 **learner 打 champion**，而且 learner 在第一个位置。

    这条不看胜负数，直接盯参数身份 —— 因为「传错对象」的失效是完全静默的：
    误传成 champ vs champ，得分率会恒等于 0.5，看起来就像「learner 还没练出来」，
    可以一直骗到训练结束。而位置参数写反（A/B 顺序）这个项目真犯过一次，
    当时把整个 A/B 实验的结论读反了（见 docs/08 最后一条）。
    """
    tr = Trainer(_cfg(tmp_path, gate_enabled=True, gate_games=8, gate_simulations=2))
    tr.make_driver()

    seen = {}

    class _R:
        score_rate, games = 0.62, 8

    def fake(a, b, dev, **kw):
        seen["a"], seen["b"], seen["kw"] = a, b, kw
        return _R()

    monkeypatch.setattr("cornerstone.evaluate.evaluate_vs_network", fake)
    score, n = tr.gate(seed=3)

    assert seen["a"] is tr.model, "第一个位置必须是 learner"
    assert seen["b"] is tr._champ, "第二个位置必须是 champion"
    assert (score, n) == (0.62, 8)
    assert seen["kw"]["opening_plies"] == tr.cfg.gate_opening_plies
    assert seen["kw"]["simulations"] == tr.cfg.gate_simulations


def test_gate_restores_training_mode(tmp_path):
    """门控内部会把模型切到 eval，跑完必须切回来。

    忘了切回来 = 后续训练步全程在 eval 模式下跑。本模型没有 dropout/BN，
    所以**不会有任何症状** —— 正是因此才要有测试盯着。
    """
    tr = Trainer(_cfg(tmp_path, gate_enabled=True, gate_games=8, gate_simulations=2))
    tr.make_driver()
    tr.model.train()
    tr.gate(seed=3)
    assert tr.model.training


# ---- 多卡：本地晋升的前提 ----

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少 2 张 GPU")
def test_all_ranks_hold_identical_learner_weights(tmp_path):
    """**本地晋升成立的唯一前提。**

    晋升不跨进程传权重，靠的是 DDP 让各 rank 的 learner 逐位相同
    （相同初值 + 相同优化器状态 + all_reduce 的结果各 rank 一致）。
    这条一旦破了，四张卡的冠军会分叉，而它**不报错**。
    """
    from cornerstone.pool import WorkerPool
    from cornerstone import _engine as E

    cfg = _cfg(tmp_path, gate_enabled=True, device="cuda:0",
               selfplay_devices="cuda:0,cuda:1", batch_size=16, parallel_games=8)
    torch.manual_seed(0)
    model = CornerNet(ModelConfig(dim=cfg.dim, blocks=cfg.blocks,
                                  attn_every=cfg.attn_every)).cuda()
    model.to_param_dtype()
    pool = WorkerPool(model, ["cuda:0", "cuda:1"], cfg,
                      E.MctsConfig(simulations=4, max_considered=4), seed=1)
    try:
        h0 = pool.weight_hashes()
        assert len(set(h0)) == 1, f"起步就不一致：{h0}"

        rng = np.random.default_rng(0)
        sampler = _ConstBatch(cfg.batch_size)
        pool.train_steps(3, sampler, lambda _s: 1e-3)
        h1 = pool.weight_hashes()
        assert len(set(h1)) == 1, f"训练 3 步后各 rank 的 learner 分叉了：{h1}"
        assert h1[0] != h0[0], "权重压根没动，这条测试就没在测东西"
        del rng
    finally:
        pool.close()


class _ConstBatch:
    """给 pool.train_steps 用的最小采样器：形状对就行，数值不重要。"""

    def __init__(self, n: int):
        from cornerstone import _engine as E
        self.n = n
        self.b = {
            "planes": np.zeros((n, E.NUM_PLANES, E.BOARD_N, E.BOARD_N), np.float32),
            "scalars": np.zeros((n, E.NUM_SCALARS), np.float32),
            "legal": np.ones((n, E.NUM_ACTIONS), np.uint8),
            "top_actions": np.zeros((n, 32), np.int64),
            "top_probs": np.full((n, 32), 1 / 32, np.float32),
            "rest_prob": np.zeros(n, np.float32),
            "n_top": np.full(n, 32, np.int32),
            "n_legal": np.full(n, E.NUM_ACTIONS, np.int32),
            "wdl": np.zeros(n, np.int64),
            "score_diff": np.zeros(n, np.float32),
        }

    def peek(self):
        return self.b

    def __call__(self):
        return self.b


def test_gate_config_defaults_are_off():
    """默认必须关着 —— 已发布的两条腿是没有门控的，默认一变对照就断了。"""
    c = TrainConfig()
    assert c.gate_enabled is False
    assert c.random_opening_prob == 0.0
    assert c.random_opening_max_plies == 0
    # 这三个是从 C++ 里暴露出来的，默认值必须与引擎一致，否则等于悄悄改了实验
    assert (c.c_visit, c.c_scale, c.value_from_score) == (50.0, 1.0, 0.0)
    assert os.environ.get("CORNERSTONE_GATE") is None
