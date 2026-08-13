"""训练档 → 发布包。

这一层的全部价值建立在一条性质上：**发布包的前向与训练档的前向逐位相同**。
它成立，「导出」就只是换了个存法；它不成立，我们就是在发布另一个模型，
而且没有任何症状 —— 棋力照样能测出来，只是测的不是要交付的那个东西。

所以这里第一条用例就是逐位比对，其余每一条也都盯着一个「不报错但不对」。
"""

import os
import sys
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cornerstone import _engine as E                     # noqa: E402
from cornerstone import export as X                      # noqa: E402
from cornerstone.model import (                          # noqa: E402
    CornerNet, ModelConfig, load_checkpoint, load_weights,
)

_CUDA = torch.cuda.is_available()


def _avail(precision: str) -> bool:
    if precision == "bf16":
        return True
    if not _CUDA:
        return False
    try:
        from cornerstone.fp8 import is_available
    except Exception:                                     # noqa: BLE001
        return False
    a = is_available(precision)
    return a[0] if isinstance(a, tuple) else bool(a)


ALL = [pytest.param(p, marks=pytest.mark.skipif(
    not _avail(p), reason=f"当前环境不支持 {p}")) for p in ("bf16", "fp8", "fp4")]
QUANT = ALL[1:]


def _dev() -> str:
    return "cuda" if _CUDA else "cpu"


def _cfg(precision: str) -> ModelConfig:
    # **blocks 至少 4**：首尾 block 被强制走高精度，blocks=2 的模型里一个
    # te.Linear 都没有，量化相关的断言会静默变成空断言
    return ModelConfig(dim=64, blocks=4, attn_every=2, heads=4, precision=precision)


def _training_blob(precision: str) -> dict:
    """造一份**训练档形状**的 blob：参数是 fp32 主权重，外加 TE 的 _extra_state。"""
    torch.manual_seed(0)
    cfg = _cfg(precision)
    m = CornerNet(cfg).to(_dev())                # 构造时就是 fp32，即 master
    sd = {}
    for k, v in m.state_dict().items():
        sd[k] = v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
    return {
        "step": 1234, "iteration": 3, "games_played": 9,
        "opt_format": "master-adamw-v1",
        "config": {"exp": "t"}, "model_config": asdict(cfg),
        "model": sd,
        "optimizer": {"state": {}, "param_groups": []},
        "rng": None, "torch_rng": torch.get_rng_state(),
    }


def _write_ckpt(tmp_path, precision: str) -> str:
    blob = _training_blob(precision)
    p = str(tmp_path / f"step00001234-{precision}.pt")
    torch.save(blob, p)
    return p


def _model_the_training_way(ckpt: str):
    """完全照旧路径建模型：fp32 master -> load_weights -> to_param_dtype。

    这就是改动之前 `load_checkpoint` 做的事，也是训练与评测时真正在用的那个模型。
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in blob["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    dev = torch.device(_dev())
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else X._null()
    with ctx:
        m = CornerNet(cfg).to(dev)
        load_weights(m, blob["model"])
        m.to_param_dtype()
    return m.eval()


def _inputs(b: int = 8):
    torch.manual_seed(7)
    d = _dev()
    return (torch.randn(b, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=d),
            torch.randn(b, E.NUM_SCALARS, device=d))


def _forward(m, inp):
    dev = next(m.parameters()).device
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else X._null()
    with ctx, torch.no_grad(), torch.autocast(
            dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        return [t.float().cpu() for t in m(*inp)]


# ---- 地基：导出不改变数值 ----

@pytest.mark.parametrize("precision", ALL)
def test_exported_forward_is_bit_identical(tmp_path, precision):
    """发布包与训练档必须给出**逐位相同**的前向。

    这条撑着整个方案。它一旦红，就说明发布出去的模型和阶梯上量到的不是同一个，
    而这种偏差在棋力数字上是看不出来的 —— 只会表现为「线上比线下弱一点」，
    然后没人知道为什么。

    量化那两种精度里，这条同时验着一件更细的事：导出必须从 **bf16 计算权重**
    量化，而不是从 fp32 master。后者会得到另一组量化值，前向就不再逐位相同。
    """
    ckpt = _write_ckpt(tmp_path, precision)
    out = str(tmp_path / "release.pt")
    X.write_release(ckpt, out, device=_dev())

    ref = _model_the_training_way(ckpt)
    got, step = load_checkpoint(out, device=_dev())
    assert step == 1234
    assert got.cfg.precision == precision

    inp = _inputs()
    for name, a, b in zip(("policy", "wdl", "score"), _forward(ref, inp), _forward(got, inp)):
        assert torch.equal(a, b), \
            f"{precision}/{name} 不逐位相同，最大绝对差 {(a - b).abs().max():.3e}"


@pytest.mark.parametrize("precision", QUANT)
def test_release_stores_rowwise_only(tmp_path, precision):
    """发布包里不许出现 columnwise 数据。

    columnwise 是**反向传播**要用的，推理一个字节都用不上。留着它
    MXFP8 发布包会比 bf16 还大（1.03×）—— 而前向照样正确，
    所以这是个纯靠体积才能发现的错误。
    """
    ckpt = _write_ckpt(tmp_path, precision)
    blob = X.build_release(ckpt, device=_dev())
    assert blob["quant"], "量化精度导出来却没有量化权重"
    for key, rec in blob["quant"].items():
        assert not any("columnwise" in k for k in rec), f"{key} 存了 columnwise"
        assert rec["data"].dtype is torch.uint8
        assert rec["scale_inv"].dtype is torch.uint8

    n_q = sum(r["data"].numel() + r["scale_inv"].numel() for r in blob["quant"].values())
    n_bf16 = sum(t.numel() for k, t in blob["bf16"].items()) * 2
    # 量化那部分必须真的比 bf16 小：fp8 约 0.52×、fp4 约 0.28×
    orig = sum(int(torch.tensor(r["shape"]).prod()) for r in blob["quant"].values()) * 2
    assert n_q < orig * 0.8, f"{precision} 量化部分 {n_q} 字节 vs bf16 {orig}，没省下来"
    assert n_bf16 > 0, "非量化的参数一个都没存？"


@pytest.mark.parametrize("precision", QUANT)
def test_release_keeps_first_last_blocks_bf16(tmp_path, precision):
    """首尾 block 与全部非 GEMM 张量在发布包里仍是 bf16。

    速查表要的「部署时也保持混合精度 checkpoint」。首尾 block 对精度最敏感，
    训练时就没走低精度，发布时更不该被顺手量化进去。
    """
    ckpt = _write_ckpt(tmp_path, precision)
    blob = X.build_release(ckpt, device=_dev())
    last = _cfg(precision).blocks - 1
    for i in (0, last):
        for k in (f"blocks.{i}.mlp.up.weight", f"blocks.{i}.mlp.down.weight"):
            assert k in blob["bf16"], f"{k} 应当留在 bf16 里"
            assert k not in blob["quant"]
            assert blob["bf16"][k].dtype is torch.bfloat16
    # 中间 block 必须是量化的，否则这条测试没在测东西
    assert f"blocks.1.mlp.up.weight" in blob["quant"]
    for k in ("stem.weight", "pos", "policy.weight", "norm_out.weight"):
        assert k in blob["bf16"] and k not in blob["quant"]


@pytest.mark.parametrize("precision", ALL)
def test_release_precision_follows_training_precision(tmp_path, precision):
    """发布格式由训练精度决定，没有开关可以选错。"""
    ckpt = _write_ckpt(tmp_path, precision)
    blob = X.build_release(ckpt, device=_dev())
    assert blob["weight_precision"] == precision
    assert blob["model_config"]["precision"] == precision
    assert bool(blob["quant"]) == (precision != "bf16")


def test_release_drops_everything_that_is_not_weights(tmp_path):
    """优化器状态、RNG、步数簿记、TE 的 amax 元数据一律不进发布包。"""
    ckpt = _write_ckpt(tmp_path, "bf16")
    out = str(tmp_path / "release.pt")
    size = X.write_release(ckpt, out, device=_dev())
    blob = torch.load(out, map_location="cpu", weights_only=False)

    for gone in ("optimizer", "config", "rng", "torch_rng", "iteration", "games_played"):
        assert gone not in blob, f"发布包里还带着 {gone}"
    assert not any(k.endswith("_extra_state") for k in blob["bf16"]), \
        "TE 的 _extra_state 是 amax 元数据，下次前向自己重建，不该存"
    # 只有权重 + 一点元信息，必须显著小于训练档
    assert size < os.path.getsize(ckpt) * 0.6
    assert blob["source"] == os.path.basename(ckpt)
    assert "te" in blob["exported_by"], "量化格式绑在 TE 版本上，要能查出来是谁导的"


# ---- 两个入口各认各的 ----

def test_inference_loader_rejects_training_checkpoint(tmp_path):
    """推理入口拿到训练档必须报人话错误，**不能将就着读**。

    将就着读的后果是：低精度那两种精度下，读到的是 fp32 master 而不是
    已量化的权重，于是评测量的模型和发布出去的不是同一个 —— 没有任何症状。
    """
    ckpt = _write_ckpt(tmp_path, "bf16")
    with pytest.raises(ValueError, match="训练 checkpoint"):
        load_checkpoint(ckpt, device=_dev())


def test_training_resume_rejects_release(tmp_path):
    """反过来同理：续训拿到发布包要说清楚，不能是 KeyError: 'optimizer'。"""
    from cornerstone.train import TrainConfig, Trainer
    ckpt = _write_ckpt(tmp_path, "bf16")
    out = str(tmp_path / "release.pt")
    X.write_release(ckpt, out, device=_dev())

    cfg = TrainConfig(exp="t", run_dir=str(tmp_path / "run"), hot_dir=str(tmp_path / "hot"),
                      dim=64, blocks=4, attn_every=2, device="cpu",
                      selfplay_devices="cpu", replay_capacity=100, batch_size=8)
    tr = Trainer(cfg.resolve(REPO))
    with pytest.raises(ValueError, match="发布包"):
        tr.load_checkpoint(out)


def test_is_release_tells_the_two_apart(tmp_path):
    ckpt = _write_ckpt(tmp_path, "bf16")
    out = str(tmp_path / "release.pt")
    X.write_release(ckpt, out, device=_dev())
    assert not X.is_release(torch.load(ckpt, map_location="cpu", weights_only=False))
    assert X.is_release(torch.load(out, map_location="cpu", weights_only=False))


# ---- 收割：先全部验证，再删 ----

def _fake_run(tmp_path, steps=(100, 200, 300), stable=True) -> str:
    """造一个跑目录：ckpt/ 里若干训练档 + latest 指针（+ stable.pt）。"""
    run = tmp_path / "run"
    ck = run / "ckpt"
    ck.mkdir(parents=True)
    blob = _training_blob("bf16")
    for s in steps:
        b = dict(blob); b["step"] = s
        torch.save(b, ck / f"step{s:08d}.pt")
    if stable:
        b = dict(blob); b["step"] = steps[0]
        torch.save(b, ck / "stable.pt")
    (ck / "latest").write_text(f"step{steps[-1]:08d}.pt")
    return str(run)


def _harvest(run_dir, do_it):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_model", os.path.join(REPO, "tools", "export_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.harvest(run_dir, _dev(), do_it)


def test_harvest_dry_run_writes_nothing(tmp_path):
    """默认是预演。**一个字节都不许写**，也不许删。"""
    run = _fake_run(tmp_path)
    before = sorted(os.listdir(os.path.join(run, "ckpt")))
    _harvest(run, do_it=False)
    assert sorted(os.listdir(os.path.join(run, "ckpt"))) == before
    assert not os.path.exists(os.path.join(run, "model")), "预演却建了 model/"


def test_harvest_keeps_exactly_the_training_entrypoints(tmp_path):
    """收割后 `ckpt/` 里只剩「为继续训练而存在」的那些，一份不多。

    保留集合由用途机械推导：`stable.pt`（WSD 分叉点）+ `latest` 指向的终点档。
    多留一份看不出问题，只是每轮悄悄多占 162.5 MiB —— 攒起来就是几十 GB。
    """
    run = _fake_run(tmp_path, steps=(100, 200, 300))
    _harvest(run, do_it=True)
    left = sorted(os.listdir(os.path.join(run, "ckpt")))
    assert left == ["latest", "stable.pt", "step00000300.pt"], left


def test_harvest_exports_every_checkpoint_including_the_kept_ones(tmp_path):
    """每一份都要有发布包 —— **包括留下来续训的那两份**。

    漏掉它们的话，终点档就参加不了评测：训练档只有续训读得了。
    """
    run = _fake_run(tmp_path, steps=(100, 200, 300))
    _harvest(run, do_it=True)
    got = sorted(os.listdir(os.path.join(run, "model")))
    assert got == ["stable.pt", "step00000100.pt", "step00000200.pt", "step00000300.pt"], got
    for f in got:
        blob = torch.load(os.path.join(run, "model", f),
                          map_location="cpu", weights_only=False)
        assert X.is_release(blob)


def test_harvest_verifies_before_deleting():
    """源码级：删除必须排在**全部**验证之后。

    反过来写的话，一旦导出有问题，等发现时原档已经没了 —— 而 runs/ 不在 git 里，
    这是不可逆的。所以这条盯的是次序本身，不是某次运行的结果。
    """
    src = open(os.path.join(REPO, "tools", "export_model.py"), encoding="utf-8").read()
    body = src[src.index("def harvest("):]
    assert body.index("verify(") < body.index("os.remove("), \
        "os.remove 出现在 verify 之前 —— 收割变成了「先删后验」"
    # 验证在循环里逐份做，删除在循环外一次性做：中途炸掉时一份都还没删
    assert body.index("os.remove(") > body.index("for n in drop"), "删除不在独立的收尾循环里"
