"""checkpoint 的存读契约。

这一层此前**完全没有测试**，而它同时承担三件事：续训不能丢精度、
历史 checkpoint 要能继续读、旧格式的优化器状态要能迁移。
三者任何一条断了，症状都出现在几小时后的续训里，而不是当场。

契约是：`blob["model"]` 里的参数条目**永远是 fp32 主权重**；
计算权重（bf16）不存，因为它精确等于 `master.bfloat16()`。
"""

import os

import pytest

torch = pytest.importorskip("torch")

from cornerstone import _engine as E  # noqa: E402
from cornerstone.model import load_checkpoint  # noqa: E402
from cornerstone.train import TrainConfig, Trainer  # noqa: E402


def _cfg(tmp_path, **kw) -> TrainConfig:
    return TrainConfig(
        exp="t", run_dir=str(tmp_path / "run"), hot_dir=str(tmp_path / "hot"),
        dim=32, blocks=2, attn_every=0, device="cpu",
        replay_capacity=1000, batch_size=8, **kw)


def _trainer(tmp_path, **kw) -> Trainer:
    torch.manual_seed(0)
    return Trainer(_cfg(tmp_path, **kw))


def _step(tr: Trainer, n: int = 3, grad: float = 0.02) -> None:
    for _ in range(n):
        for p in tr.model.parameters():
            p.grad = torch.full_like(p, grad)
        tr.opt.step(grad_clip=1.0)
        tr.step += 1


def _inputs(b: int = 2):
    return (torch.zeros(b, E.NUM_PLANES, E.BOARD_N, E.BOARD_N),
            torch.zeros(b, E.NUM_SCALARS))


# ---- 契约 ----

def test_model_entry_is_fp32_master(tmp_path):
    tr = _trainer(tmp_path)
    _step(tr)
    blob = torch.load(tr.save_checkpoint(), map_location="cpu", weights_only=False)

    master = tr.opt.master_state_dict()
    assert master, "优化器没有主权重，契约无从谈起"
    for name, m in master.items():
        assert blob["model"][name].dtype is torch.float32, f"{name} 存成了低精度"
        assert torch.equal(blob["model"][name], m), f"{name} 存的不是主权重"


def test_compute_weights_are_exactly_master_rounded(tmp_path):
    tr = _trainer(tmp_path)
    _step(tr)
    params = dict(tr.model.named_parameters())
    for name, m in tr.opt.master_state_dict().items():
        assert params[name].dtype is torch.bfloat16
        assert torch.equal(params[name].detach(), m.bfloat16())


# ---- 往返 ----

def test_save_load_roundtrip_preserves_master_and_progress(tmp_path):
    tr = _trainer(tmp_path)
    _step(tr, 5)
    tr.iteration, tr.games_played = 7, 1234
    path = tr.save_checkpoint()

    tr2 = _trainer(tmp_path / "b")
    tr2.load_checkpoint(path)

    assert tr2.step == tr.step
    assert tr2.iteration == 7 and tr2.games_played == 1234
    for a, b in zip(tr.opt._masters, tr2.opt._masters):
        assert torch.equal(a, b), "主权重没有逐位恢复"
    for a, b in zip(tr.model.parameters(), tr2.model.parameters()):
        assert torch.equal(a.detach(), b.detach())

    # 续训必须走同一条轨迹 —— 动量没恢复的话这里立刻分叉
    _step(tr, 3)
    _step(tr2, 3)
    for a, b in zip(tr.model.parameters(), tr2.model.parameters()):
        assert torch.equal(a.detach(), b.detach()), "续训后分叉，动量没恢复"


# ---- 旧格式 ----

def _to_legacy(path: str) -> str:
    """把新格式的 checkpoint 改写成改造前的样子。

    三处差异：优化器状态是 `{step:int, m, v}`（bf16）、model 里带 TE 的
    `_extra_state` 键、model_config 里没有 `param_dtype`。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob.pop("opt_format", None)
    for st in blob["optimizer"]["state"].values():
        st["m"] = st.pop("exp_avg").bfloat16()
        st["v"] = st.pop("exp_avg_sq").bfloat16()
        st["step"] = int(st["step"].item())
    blob["model"]["blocks.0.mlp.up._extra_state"] = torch.zeros(16, dtype=torch.uint8)
    blob["model_config"].pop("param_dtype", None)
    out = path + ".legacy"
    torch.save(blob, out)
    return out


def test_legacy_checkpoint_loads_for_inference(tmp_path):
    tr = _trainer(tmp_path)
    _step(tr)
    legacy = _to_legacy(tr.save_checkpoint())

    model, step = load_checkpoint(legacy, device="cpu")
    assert step == tr.step
    # 推理一律用计算权重，即使 checkpoint 里存的是 fp32
    assert all(p.dtype is torch.bfloat16 for p in model.parameters())
    with torch.no_grad():
        pol, wdl, sc = model(*_inputs())
    assert torch.isfinite(pol).all() and torch.isfinite(wdl).all()


def test_legacy_checkpoint_can_be_resumed(tmp_path, capsys):
    tr = _trainer(tmp_path)
    _step(tr)
    legacy = _to_legacy(tr.save_checkpoint())

    tr2 = _trainer(tmp_path / "b")
    tr2.load_checkpoint(legacy)
    assert "已迁移" in capsys.readouterr().out
    for a, b in zip(tr.opt._masters, tr2.opt._masters):
        assert torch.equal(a, b)
    _step(tr2, 1)          # 关键：续训第一步不能抛 KeyError: 'exp_avg'


# ---- 真实产物 ----

_RUNS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")


@pytest.mark.parametrize("run", ["v2-bf16", "v2-fp8"])
def test_published_checkpoints_are_still_readable(run):
    """两份已发布报告的 checkpoint 必须还能读出来做推理。

    没有这些产物的环境自动跳过 —— 它们不在 git 里。
    """
    ckpt_dir = os.path.join(_RUNS, run, "ckpt")
    latest = os.path.join(ckpt_dir, "latest")
    if not os.path.exists(latest):
        pytest.skip(f"没有 {run} 的训练产物")
    if run == "v2-fp8" and not torch.cuda.is_available():
        pytest.skip("FP8 模型需要 GPU")
    with open(latest) as f:
        path = os.path.join(ckpt_dir, f.read().strip())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, step = load_checkpoint(path, device=dev)
    assert isinstance(step, int) and step > 0
    assert all(p.dtype is torch.bfloat16 for p in model.parameters())

    # **精度必须原样读回来。** 这两份 blob 的 model_config 里只有老的
    # `fp8: bool`，没有 `precision` —— 而 `load_checkpoint` 对未知键做静默过滤、
    # te.Linear 与 nn.Linear 的 state_dict 键名又都是 `weight`。
    # 兼容别名一旦断掉，v2-fp8 会**静默降级成 BF16 模型**，不抛任何异常，
    # 只是「复现 FP8 那条腿」悄悄变成了复现 BF16。
    want = "fp8" if run.endswith("fp8") else "bf16"
    assert model.cfg.precision == want, f"{run} 读回来是 {model.cfg.precision}"
    if want == "fp8":
        import transformer_engine.pytorch as te
        assert any(isinstance(m, te.Linear) for m in model.modules()), \
            "FP8 checkpoint 读出来一个 te.Linear 都没有 —— 静默降级了"
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16,
                                         enabled=dev == "cuda"):
        pol, wdl, _ = model(*[t.to(dev) for t in _inputs(8)])
    assert torch.isfinite(pol.float()).all()
