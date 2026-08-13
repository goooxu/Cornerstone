"""工作池的核心不变量。

这里只有一条，但它是 DDP 成立的地基：**各 rank 的 learner 必须逐位相同**。
破了不会报错，也不会让 loss 变难看 —— 只表现为四张卡在训练四个略微不同的网络，
而落盘的是 rank 0 那一份。这类失效从任何日志上都看不出来。

（这条测试原先长在门控那套代码里。门控 = 自博弈不用最新权重，而是另存一份
「当前最强」当冠军，新模型赢过它才能接替；它靠「各 rank 本地晋升」省掉跨进程传权重，
所以它把这个不变量顶到了台前。门控已被实测证伪并移除，见 docs/08，
但不变量本身和门控无关，留下来。）
"""

import numpy as np
import pytest
import torch

from cornerstone import _engine as E
from cornerstone.model import CornerNet, ModelConfig
from cornerstone.train import TrainConfig


class _ConstBatch:
    """最小采样器：形状与 dtype 对得上就行，数值不重要。"""

    def __init__(self, n: int):
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少 2 张 GPU")
def test_all_ranks_hold_identical_weights(tmp_path):
    from cornerstone.pool import WorkerPool

    cfg = TrainConfig(exp="t", run_dir=str(tmp_path / "run"), hot_dir=str(tmp_path / "hot"),
                      dim=32, blocks=2, attn_every=0, device="cuda:0",
                      selfplay_devices="cuda:0,cuda:1", replay_capacity=1000,
                      batch_size=16, parallel_games=8)
    torch.manual_seed(0)
    model = CornerNet(ModelConfig(dim=cfg.dim, blocks=cfg.blocks,
                                  attn_every=cfg.attn_every)).cuda()
    model.to_param_dtype()
    pool = WorkerPool(model, ["cuda:0", "cuda:1"], cfg,
                      E.MctsConfig(simulations=4, max_considered=4), seed=1)
    try:
        h0 = pool.weight_hashes()
        assert len(set(h0)) == 1, f"起步就不一致：{h0}"

        pool.train_steps(3, _ConstBatch(cfg.batch_size), lambda _s: 1e-3)
        h1 = pool.weight_hashes()
        assert len(set(h1)) == 1, f"训练 3 步后各 rank 分叉了：{h1}"
        assert h1[0] != h0[0], "权重压根没动，这条测试就没在测东西"
    finally:
        pool.close()


def test_selfplay_defaults_are_on():
    """随机开局注入是默认开的 —— 它治的开局塌缩是实测到的（首手 414 选 1）。

    默认值一改，`docs/08` 里那组对照数字（+92~+149 中途、终点打平）就不再描述
    默认配置了，所以这条钉住它。
    """
    c = TrainConfig()
    assert (c.random_opening_prob, c.random_opening_max_plies) == (0.5, 6)
    # 这三个是从 C++ 暴露出来的，默认值必须与引擎一致，否则等于悄悄改了实验
    assert (c.c_visit, c.c_scale, c.value_from_score) == (50.0, 1.0, 0.0)
