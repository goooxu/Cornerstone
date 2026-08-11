"""FP8 混合精度：GEMM 走 MXFP8，周边一律高精度。

核心要验证的是三件事：**FP8 只改计算不改存储**（权重是普通 bf16 张量）、
**FP8 GEMM 真的在算**（TE 静默降级是这套方案最危险的失效模式）、
以及 MXFP8 的维度约束确实存在（它一路传导到 batch size 上）。

主权重与随机舍入相关的用例已经删掉 —— 主权重现在是 fp32，由
`MasterWeightAdamW` 持有，对应的测试在 `test_optim.py`。
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")

if not torch.cuda.is_available():
    pytest.skip("FP8 需要 GPU", allow_module_level=True)

from cornerstone import fp8 as F  # noqa: E402
from cornerstone.model import CornerNet, ModelConfig  # noqa: E402

_avail = F.is_available()
if not (_avail[0] if isinstance(_avail, tuple) else _avail):
    pytest.skip(f"当前 GPU 不支持 MXFP8: {_avail}", allow_module_level=True)


@pytest.fixture
def linear():
    return F.fp8_linear(256, 512, bias=False).cuda()


def _small(fp8: bool) -> CornerNet:
    """**blocks 至少 4**：首尾 block 被强制走高精度，blocks=2 的模型里
    一个 te.Linear 都没有，所有 FP8 断言会静默变成空断言。"""
    torch.manual_seed(0)
    return CornerNet(ModelConfig(dim=64, blocks=4, attn_every=2, fp8=fp8)).cuda()


def _n_te(model) -> int:
    import transformer_engine.pytorch as te
    return sum(isinstance(m, te.Linear) for m in model.modules())


# ---- 存储：FP8 不改变权重的存储格式 ----

def test_weights_are_plain_bf16_not_quantized(linear):
    """权重必须是**普通** bf16 张量。

    这条守的是「不要退回 FP8 主权重」：`quantized_model_init` 会让权重张量本身
    变成 MXFP8（行/列两套 E4M3 数据 + 两套 E8M0 块缩放），那样就没有高精度
    master 了，退火尾段的更新会被整片吃掉（量化依据见 test_optim.py）。
    """
    linear.to(torch.bfloat16)
    w = linear.weight
    assert w.dtype is torch.bfloat16
    assert not hasattr(w, "_rowwise_data"), "权重被量化了 —— FP8 主权重方案又回来了"
    assert w.element_size() == 2


def test_both_legs_have_identical_parameter_footprint():
    """两条腿的参数量、逐参数 dtype、总字节数必须完全相同。

    这是「唯一的区别是 GEMM 开关」这句话的可执行表述。一旦哪天存储又出现差异，
    A/B 就不再是单变量对照了。
    """
    a, b = _small(False).to(torch.bfloat16), _small(True).to(torch.bfloat16)
    assert _n_te(a) == 0 and _n_te(b) > 0, "两条腿没有区别，这条测试没在测东西"
    sa = {n: (p.shape, p.dtype, p.numel() * p.element_size())
          for n, p in a.named_parameters()}
    sb = {n: (p.shape, p.dtype, p.numel() * p.element_size())
          for n, p in b.named_parameters()}
    assert sa == sb
    assert sum(v[2] for v in sa.values()) == sum(v[2] for v in sb.values())


def test_both_legs_start_from_identical_weights():
    """同一 seed 下两条腿的初始权重必须逐位相同。

    `te.Linear` 默认用 normal(0, 0.023) 初始化而 `nn.Linear` 用 kaiming_uniform，
    两者消耗的 RNG 流长度也不同 —— 不做对齐的话，126 个参数张量里有 75 个不同，
    连不含 TE 的卷积和位置嵌入都跟着错位。这个隐藏变量污染过一次 A/B。
    """
    a, b = _small(False), _small(True)
    assert _n_te(b) > 0, "FP8 那条腿里没有 te.Linear，这条测试没在测东西"
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    bad = [n for n in pa if not torch.equal(pa[n].float().cpu(), pb[n].float().cpu())]
    assert not bad, f"{len(bad)}/{len(pa)} 个张量初值不同，例如 {bad[:3]}"


# ---- 计算：FP8 GEMM 到底有没有在算 ----

def test_recipe_is_mxfp8_hybrid():
    """前向 E4M3（精度优先）/ 反向 E5M2（范围优先）。

    `MXFP8BlockScaling` 默认是纯 E4M3，必须显式指定 HYBRID —— 梯度的动态范围
    比激活宽得多，4 位指数不够用。
    """
    from transformer_engine.common import recipe
    r = F.mxfp8_recipe()
    assert isinstance(r, recipe.MXFP8BlockScaling)
    assert r.fp8_format is recipe.Format.HYBRID


def test_probe_distinguishes_enabled_from_disabled():
    """探针本身要有分辨力 —— 恒真的探针我们已经吃过一次亏。

    旧探针靠捕获 `quantized weights without quantized compute` 警告，而那条警告
    在非量化权重下永远不会出现，探针就静默退化成「总是 True」了。
    """
    m = _small(True).to(torch.bfloat16)
    assert _n_te(m) > 0
    p = torch.zeros(8, 9, 14, 14, device="cuda")
    s = torch.zeros(8, 44, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.fp8_gemm_active(m, p, s) is True          # 模型自己会开 fp8 作用域

    # 把作用域摘掉：同样的模型必须被判成「没在算 FP8」
    m.cfg.fp8 = False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.fp8_gemm_active(m, p, s) is False


def test_fp8_changes_numerics_but_stays_close(linear):
    """FP8 前向与高精度前向必须**不逐位相同**（证明量化真的发生了）且相对差 < 5%。

    这是「开了但没生效」的终极判据 —— 数值不会说谎。
    """
    torch.manual_seed(2)
    linear.to(torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(512, 256, device="cuda") * 0.05)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    with F.fp8_autocast():
        y_fp8 = linear(x)
    y_hp = linear(x)
    ref = x.float() @ linear.weight.float().T

    assert not torch.equal(y_fp8, y_hp), "FP8 与高精度输出逐位相同 —— 量化根本没发生"
    rel = (y_fp8.float() - ref).norm() / ref.norm()
    assert rel < 0.05, f"FP8 GEMM 与参考实现相对误差 {rel:.4f} 偏大"


def test_mxfp8_requires_dims_divisible_by_32():
    """MXFP8 的硬约束，会一路传导到 batch size 上，必须显式记下来。

    token 维是 B*196，而 196 mod 32 == 4，所以 **batch 必须是 8 的倍数**
    （8*196 = 1568 = 49*32）。自博弈里 prepare() 返回的批大小是变化的，
    所以送进 FP8 前向之前必须补齐。
    """
    lin = F.fp8_linear(64, 64, bias=False).cuda().to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="divisible by 32"):
        with F.fp8_autocast():
            lin(torch.randn(8, 64, device="cuda", dtype=torch.bfloat16))
    with F.fp8_autocast():
        lin(torch.randn(32, 64, device="cuda", dtype=torch.bfloat16))   # 32 就没问题

    assert F.pad_to_mxfp8(1) == 8
    assert F.pad_to_mxfp8(8) == 8
    assert F.pad_to_mxfp8(9) == 16
    assert F.pad_to_mxfp8(1000) == 1000        # 1000*196 = 196000 = 6125*32
    for b in (8, 16, 24, 512, 1000):
        assert (b * 196) % 32 == 0


# ---- 设备护栏 ----

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少两张卡")
def test_fp8_model_works_when_current_device_differs():
    """FP8 模型放在非默认卡上、且调用时当前设备不匹配 —— 必须照常工作。

    TE 按**当前 CUDA 设备**取 cuBLAS 句柄。不匹配时的表现有两种，都很难查：
    要么静默退回非量化计算，要么直接非法访存（报错点还常飘到别的算子上）。
    自博弈的多卡驱动天然会撞上这个场景。

    断言用探针而不是「没有警告」—— 后者在非量化权重下恒真，等于没测。
    """
    torch.manual_seed(4)
    with torch.cuda.device(1):
        m = CornerNet(ModelConfig(dim=64, blocks=4, attn_every=2, fp8=True))
        m = m.to("cuda:1").to(torch.bfloat16)
    assert _n_te(m) > 0
    p = torch.zeros(8, 9, 14, 14, device="cuda:1")
    s = torch.zeros(8, 44, device="cuda:1")

    torch.cuda.set_device(0)                    # 故意让当前设备与模型不一致
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.fp8_gemm_active(m, p, s) is True
    assert torch.cuda.current_device() == 0, "探针不该改变调用方的当前设备"
