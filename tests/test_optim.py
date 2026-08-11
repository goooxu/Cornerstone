"""bf16 计算权重 + fp32 主权重的优化器。

这个文件守的是低精度训练里最隐蔽的一类故障：**不报错、但权重不动**。
主权重接错、Parameter 对象被替换、梯度没清干净 —— 三者的症状都是「训练照跑，
loss 曲线看着还行，模型就是不长棋力」，而且都要到训练后半程才显现。
"""

import pytest

torch = pytest.importorskip("torch")

from cornerstone.optim import MasterWeightAdamW  # noqa: E402


def _make(n: int = 256, dtype=torch.bfloat16, fill: float | None = None):
    """一个两参数的小模型 + 对应的优化器。"""
    torch.manual_seed(0)
    lin = torch.nn.Linear(n, n, bias=True)
    if fill is not None:
        with torch.no_grad():
            lin.weight.fill_(fill)
            lin.bias.fill_(fill)
    groups = [{"params": [lin.weight], "names": ["weight"], "weight_decay": 0.01},
              {"params": [lin.bias], "names": ["bias"], "weight_decay": 0.0}]
    opt = MasterWeightAdamW(groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    lin.to(dtype)                      # 顺序：先建优化器（拿 fp32 master），再降精度
    return lin, opt


def _drive(lin, opt, steps: int, grad: float, clip=None):
    for _ in range(steps):
        for p in lin.parameters():
            p.grad = torch.full_like(p, grad)
        opt.step(grad_clip=clip)


# ---- master 的基本性质 ----

def test_master_is_fp32_while_params_are_bf16():
    lin, opt = _make()
    assert all(p.dtype is torch.bfloat16 for p in lin.parameters())
    assert all(m.dtype is torch.float32 for m in opt._masters)


def test_master_holds_the_pre_downcast_values():
    """master 必须取自**降精度之前**的 fp32 初值。

    顺序写反（先降 bf16 再建优化器）的话 master 是从 bf16 值回填的，
    等于一开始就丢掉一半精度，而训练看不出任何异常。
    """
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias=False)
    fp32_init = lin.weight.detach().clone()
    opt = MasterWeightAdamW([{"params": [lin.weight], "names": ["weight"]}], lr=1e-3)
    lin.to(torch.bfloat16)

    assert torch.equal(opt._masters[0], fp32_init)
    # 而计算权重就是它舍入到 bf16 的结果
    assert torch.equal(lin.weight.detach(), fp32_init.bfloat16())


def test_downcast_preserves_parameter_objects():
    """`model.to(bf16)` 不能替换 Parameter 对象，否则优化器持有的引用整体失效。

    失效之后不会报错 —— 优化器一直在更新一批已经和模型脱钩的张量，
    表现是「训练正常、权重永远不动」，和 docs/06 记的 bf16 中点 bug 同一类。
    """
    lin, opt = _make()
    assert opt._params[0] is lin.weight
    assert opt._params[1] is lin.bias


# ---- 为什么必须有 master ----

def test_updates_below_bf16_ulp_survive_only_with_master():
    """退火尾段的更新小于 bf16 的 ulp，没有 master 就会被整片吃掉。

    权重取 0.17（实测量级），此处 bf16 的 ulp 是 2^-10 ≈ 9.8e-4；
    余弦退火终点 lr = 2e-3 * min_lr_ratio(0.1) = 2e-4，AdamW 下每步位移约等于 lr，
    只有 0.2 个 ulp。实测：有 master 走完 100.2% 的应走位移，没有只有 0.1%。
    """
    W, N, LR = 0.17, 400, 2e-4
    expected = LR * N

    def run(with_master: bool) -> float:
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.full((1024,), W, dtype=torch.bfloat16))
        if with_master:
            opt = MasterWeightAdamW([{"params": [p], "names": ["w"]}], lr=LR)
        else:                       # 直接在 bf16 上更新，m/v 也是 bf16
            opt = torch.optim.AdamW([p], lr=LR, betas=(0.9, 0.95), eps=1e-8,
                                    weight_decay=0.0)
        for _ in range(N):
            p.grad = torch.full_like(p, 1e-3)
            opt.step(grad_clip=None) if with_master else opt.step()
        return (p.detach().float() - W).abs().mean().item()

    with_m, without = run(True), run(False)
    assert with_m > 0.9 * expected, f"有 master 也只走了 {with_m:.5f}/{expected:.5f}"
    assert without < 0.05 * expected, \
        f"没有 master 却走了 {without:.5f}，这条测试失去了判别力"


# ---- 梯度：清理与裁剪 ----

def test_zero_grad_clears_model_param_grads():
    """父类只认它自己的 params（= master），模型参数的梯度必须自己清。

    漏了不会报错：`.grad` 一直累加，loss 越训越怪，但每一步看着都正常。
    """
    lin, opt = _make()
    for p in lin.parameters():
        p.grad = torch.ones_like(p)
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in lin.parameters())

    for p in lin.parameters():
        p.grad = torch.ones_like(p)
    opt.zero_grad(set_to_none=False)
    assert all(p.grad is not None and not p.grad.any() for p in lin.parameters())


def test_grad_norm_is_computed_in_fp32_and_is_pre_clip():
    """返回的是**裁剪前**的范数，且在 fp32 上算。

    bf16 只有 8 位有效数字，上万个元素的平方和直接在 bf16 里累加会明显偏低 ——
    而 grad_norm 是判断训练是否发散的主要指标，不能失真。
    """
    lin, opt = _make(n=128)
    g = 0.03
    for p in lin.parameters():
        p.grad = torch.full_like(p, g)
    n_elem = sum(p.numel() for p in lin.parameters())
    ref = (g ** 2 * n_elem) ** 0.5

    gnorm = opt.step(grad_clip=1e-4)            # 裁剪阈值远小于真实范数
    assert abs(float(gnorm) - ref) / ref < 1e-3, f"{float(gnorm)} vs 参考 {ref}"


def test_grad_norm_available_without_clipping():
    lin, opt = _make(n=64)
    for p in lin.parameters():
        p.grad = torch.full_like(p, 0.01)
    assert torch.isfinite(opt.step(grad_clip=None))


# ---- 存读 ----

def test_master_state_dict_keys_match_param_names():
    lin, opt = _make()
    assert set(opt.master_state_dict()) == {"weight", "bias"}
    assert all(v.dtype is torch.float32 for v in opt.master_state_dict().values())


def test_load_master_state_dict_syncs_params_and_rejects_gaps():
    lin, opt = _make(n=32)
    sd = {k: torch.full_like(v, 0.25) for k, v in opt.master_state_dict().items()}
    opt.load_master_state_dict(sd)
    # 装载必须**顺带**同步计算权重，不给「顺序写反」留空间
    assert all(torch.equal(p.detach(), torch.full_like(p, 0.25)) for p in lin.parameters())

    with pytest.raises(KeyError, match="主权重缺少"):
        opt.load_master_state_dict({"weight": sd["weight"]})
    opt.load_master_state_dict({"weight": sd["weight"]}, partial=True)   # 显式放行


def test_state_dict_roundtrip_is_bit_exact():
    lin, opt = _make(n=64)
    _drive(lin, opt, 3, 0.02)
    blob_opt, blob_master = opt.state_dict(), opt.master_state_dict()

    lin2, opt2 = _make(n=64)
    opt2.load_state_dict(blob_opt, master=blob_master)
    assert all(torch.equal(a, b) for a, b in zip(opt._masters, opt2._masters))

    _drive(lin, opt, 3, 0.02)
    _drive(lin2, opt2, 3, 0.02)
    for a, b in zip(lin.parameters(), lin2.parameters()):
        assert torch.equal(a.detach(), b.detach())


def test_state_dict_does_not_carry_master_and_stays_usable():
    """master 只存一份（在 `"model"` 里）。

    另外守一个 torch 的陷阱：`Optimizer.state_dict()` 返回的 per-param state
    是活对象，谁要是事后去 pop/改写，改的是运行中的优化器。
    """
    lin, opt = _make(n=32)
    _drive(lin, opt, 2, 0.02)
    sd = opt.state_dict()
    keys = {k for st in sd["state"].values() for k in st}
    assert keys <= {"step", "exp_avg", "exp_avg_sq"}
    assert all(v.dtype is torch.float32
               for st in sd["state"].values()
               for k, v in st.items() if k != "step" and torch.is_tensor(v))
    _drive(lin, opt, 1, 0.02)                   # 取过 state_dict 之后还能接着训


def test_legacy_fp8_optimizer_state_is_migrated(capsys):
    """`Fp8AdamW` 存的 `{step:int, m, v}` 要能续训。

    不翻译的话 torch 会照单收下，直到续训的**第一次 step** 才抛
    `KeyError: 'exp_avg'` —— 开发机每 8 小时回收一次，续训是最高频路径。
    """
    lin, opt = _make(n=32)
    _drive(lin, opt, 2, 0.02)
    legacy = opt.state_dict()
    for st in legacy["state"].values():
        st["m"] = st.pop("exp_avg").bfloat16()
        st["v"] = st.pop("exp_avg_sq").bfloat16()
        st["step"] = int(st["step"].item())

    lin2, opt2 = _make(n=32)
    opt2.load_state_dict(legacy)
    assert "已迁移" in capsys.readouterr().out
    for st in opt2.state.values():
        assert st["exp_avg"].dtype is torch.float32
        assert st["exp_avg_sq"].dtype is torch.float32
    _drive(lin2, opt2, 1, 0.02)                 # 关键：续训不能抛 KeyError


def test_unrecognized_optimizer_state_is_dropped_loudly(capsys):
    lin, opt = _make(n=32)
    _drive(lin, opt, 2, 0.02)
    junk = opt.state_dict()
    for st in junk["state"].values():
        st.pop("exp_avg"), st.pop("exp_avg_sq")

    lin2, opt2 = _make(n=32)
    opt2.load_state_dict(junk)
    assert "警告" in capsys.readouterr().out
    _drive(lin2, opt2, 1, 0.02)
