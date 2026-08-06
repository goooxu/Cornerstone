"""FP8 主权重与随机舍入。

核心要验证的是：主权重**真的**是 FP8（不是高精度副本临时量化），
以及随机舍入确实把「小更新被系统性吃掉」变成了零均值噪声。
后者是整套方案能不能训起来的关键 —— 没有它，学习率一降训练就停住。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")

if not torch.cuda.is_available():
    pytest.skip("FP8 需要 GPU", allow_module_level=True)

from cornerstone import fp8 as F  # noqa: E402
from cornerstone.model import ACTIONS as ACTIONS_  # noqa: E402

_avail = F.is_available()
if not (_avail[0] if isinstance(_avail, tuple) else _avail):
    pytest.skip(f"当前 GPU 不支持 MXFP8: {_avail}", allow_module_level=True)


@pytest.fixture
def linear():
    return F.fp8_linear(256, 512, bias=False).cuda()


def test_weights_are_stored_as_mxfp8(linear):
    w = linear.weight
    assert F.is_quantized(w), "权重不是量化张量 —— fp8_model_init 没生效"
    assert w._rowwise_data.dtype == torch.uint8
    assert w._rowwise_data.shape == (512, 256)
    # 每 32 个元素一个 E8M0 缩放
    assert w._rowwise_scale_inv.shape[-1] == 256 // F.MX_BLOCK


def test_mxfp8_costs_2_06_bytes_per_param(linear):
    """MXFP8 每参数 2.06 字节：行/列两套 E4M3 数据 + 两套 E8M0 块缩放。

    比**纯 BF16 存储**（2 字节）略多 —— 因为块缩放的 FP8 没法便宜地转置，
    反向的两个 GEMM 需要不同的连续维，只能两套都留。

    但项目里真正的对照不是「纯 BF16 存储」：非 FP8 配置走的是标准混合精度，
    主权重是 **fp32**（4 字节/参数），autocast 只在计算时临时转 BF16。
    对着那个基线，FP8 是实打实省显存的 —— 见
    test_fp8_saves_memory_against_the_actual_baseline。
    """
    w = linear.weight
    fp8_bytes = (w._rowwise_data.numel() + w._columnwise_data.numel()
                 + w._rowwise_scale_inv.numel() + w._columnwise_scale_inv.numel())
    assert fp8_bytes / w.numel() == pytest.approx(2.0625, abs=0.01)
    assert fp8_bytes > w.numel() * 2          # 略多于纯 BF16 存储
    assert fp8_bytes < w.numel() * 4          # 但远少于 fp32 主权重


def test_fp8_saves_memory_against_the_actual_baseline():
    """对着项目里真实的 BF16 配置比：后者的主权重是 **fp32**（标准混合精度）。

    节省的比例取决于有多少参数被量化（首尾 block 与所有卷积/Norm/头都不量化），
    所以这里不钉总量比值，而是钉两条与规模无关的性质：
      1. 非 FP8 配置的参数确实是 fp32
      2. 被量化的那部分，2.06 字节/参数，约为 fp32 的 52%
    """
    from cornerstone.model import CornerNet, ModelConfig

    def stats(fp8: bool):
        with torch.cuda.device("cuda"):
            m = CornerNet(ModelConfig(dim=256, blocks=8, fp8=fp8)).cuda()
        total = qbytes = qparams = 0
        dtypes = set()
        for p in m.parameters():
            if F.is_quantized(p):
                b = sum(x.numel() for x in (p._rowwise_data, p._columnwise_data,
                                            p._rowwise_scale_inv, p._columnwise_scale_inv))
                qbytes += b
                qparams += p.numel()
                total += b
            else:
                dtypes.add(p.dtype)
                total += p.numel() * p.element_size()
        return total, qbytes, qparams, dtypes

    bf16_total, _, bf16_q, bf16_dtypes = stats(False)
    fp8_total, qbytes, qparams, _ = stats(True)

    assert bf16_q == 0, "非 FP8 配置不该有量化参数"
    assert bf16_dtypes == {torch.float32}, (
        f"非 FP8 配置的主权重应当是 fp32（autocast 只在计算时转 BF16），实际 {bf16_dtypes}")
    assert qparams > 0
    assert qbytes / (qparams * 4) == pytest.approx(0.5156, abs=0.02), "量化部分应约为 fp32 的一半"
    assert fp8_total < bf16_total, "整体上 FP8 配置应当更省"


def test_roundtrip_error_is_within_e4m3_resolution(linear):
    x = torch.randn(512, 256, device="cuda") * 0.05
    F.write_weight_(linear.weight, x, stochastic=False)
    back = F.read_weight(linear.weight)
    rel = (back - x).abs() / x.abs().clamp_min(1e-6)
    # E4M3 有 3 位尾数 -> 相对误差上界约 2^-4
    assert rel.median() < 2 ** -4
    assert rel.max() < 0.5


def test_ulp_matches_the_actual_quantization_grid(linear):
    """估出来的 ulp 必须和真实网格间距一致，否则随机舍入的噪声幅度就是错的。"""
    x = torch.randn(512, 256, device="cuda") * 0.1
    ulp = F.quantization_ulp(x)
    F.write_weight_(linear.weight, x, stochastic=False)
    err = (F.read_weight(linear.weight) - x).abs()
    # 舍入到最近值的误差不会超过半个 ulp（留一点余量给块缩放的边界情况）
    assert (err <= ulp * 0.75 + 1e-9).float().mean() > 0.99


def test_stochastic_rounding_is_unbiased(linear):
    """同一个值反复量化，均值应收敛到真值；确定性舍入则收敛到一个有偏的格点。"""
    torch.manual_seed(0)
    x = torch.randn(512, 256, device="cuda") * 0.05

    F.write_weight_(linear.weight, x, stochastic=False)
    rn = F.read_weight(linear.weight)
    rn_bias = (rn - x).mean().abs().item()

    acc = torch.zeros_like(x)
    trials = 200
    for _ in range(trials):
        F.write_weight_(linear.weight, x, stochastic=True)
        acc += F.read_weight(linear.weight)
    sr_bias = (acc / trials - x).mean().abs().item()

    scale = x.abs().mean().item()
    assert sr_bias < 0.02 * scale, f"随机舍入仍有偏差 {sr_bias:.3e}"
    # 单次随机舍入的方差比确定性舍入大，但均值无偏 —— 这正是我们要的
    assert sr_bias < rn_bias or rn_bias < 1e-4 * scale


def test_tiny_updates_survive_only_with_stochastic_rounding(linear):
    """这条是整套 FP8 主权重方案成立与否的关键。

    施加一个远小于 ulp 的更新：确定性舍入会把它整个吃掉（权重一动不动），
    随机舍入下权重会以正确的期望速率漂移。
    """
    torch.manual_seed(1)
    w0 = torch.full((512, 256), 0.1, device="cuda")
    steps, delta = 300, 1e-4

    for stochastic, should_move in [(False, False), (True, True)]:
        F.write_weight_(linear.weight, w0, stochastic=False)
        # 基线必须取**量化之后**的值：拿它和未量化的 w0 比，量到的是初始量化误差
        # （最大半个 ulp），不是累计移动量
        start = F.read_weight(linear.weight).mean().item()
        ulp = F.quantization_ulp(w0).mean().item()
        assert delta < ulp / 2, f"这条测试要求单步更新小于半个 ulp（ulp={ulp:.5f}）"

        for _ in range(steps):
            cur = F.read_weight(linear.weight)
            F.write_weight_(linear.weight, cur + delta, stochastic=stochastic)
        moved = F.read_weight(linear.weight).mean().item() - start
        expected = steps * delta

        if should_move:
            assert moved > 0.5 * expected, \
                f"随机舍入下权重只移动了 {moved:.5f}，期望约 {expected:.5f}"
        else:
            assert abs(moved) < 0.05 * expected, \
                f"确定性舍入本该把小于半个 ulp 的更新全部吃掉，却移动了 {moved:.5f}"


def test_fp8_linear_forward_is_close_to_bf16_reference(linear):
    torch.manual_seed(2)
    ref_w = torch.randn(512, 256, device="cuda") * 0.05
    F.write_weight_(linear.weight, ref_w, stochastic=False)

    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    with F.fp8_autocast():
        y = linear(x)
    ref = x.float() @ F.read_weight(linear.weight).T

    rel = (y.float() - ref).norm() / ref.norm()
    assert rel < 0.05, f"FP8 GEMM 与参考实现相对误差 {rel:.4f} 偏大"


def test_fp8_adamw_moves_weights_and_keeps_them_quantized():
    torch.manual_seed(3)
    lin = F.fp8_linear(128, 256, bias=False).cuda()
    before = F.read_weight(lin.weight).clone()

    opt = F.Fp8AdamW(lin.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(20):
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        with F.fp8_autocast():
            loss = lin(x).float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    after = F.read_weight(lin.weight)
    assert F.is_quantized(lin.weight), "更新之后权重必须仍然是 FP8"
    assert (after - before).abs().mean() > 0, "权重没动"
    assert torch.isfinite(after).all()


def test_mxfp8_requires_dims_divisible_by_32():
    """MXFP8 的硬约束，会一路传导到 batch size 上，必须显式记下来。

    token 维是 B*196，而 196 mod 32 == 4，所以 **batch 必须是 8 的倍数**
    （8*196 = 1568 = 49*32）。自博弈里 prepare() 返回的批大小是变化的，
    所以送进 FP8 前向之前必须补齐。
    """
    lin = F.fp8_linear(64, 64, bias=False).cuda()
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少两张卡")
def test_fp8_model_works_when_current_device_differs():
    """FP8 模型放在非默认卡上、且调用时当前设备不匹配 —— 必须照常工作。

    这是踩过两次的坑。TE 按**当前设备**取 cuBLAS 句柄，不匹配时表现有两种：
      - 静默退回非量化计算，只发一条 UserWarning（FP8 名存实亡）
      - 直接 CUDA illegal memory access，且报错点常飘到别的算子上

    第一次以为是「同进程只能用一张卡」而加了单卡护栏（误判）；
    第二次是训练步没设设备，A/B 实验一开跑就崩。
    护栏现在放在 CornerNet.forward 里，这条测试就是守它的。
    """
    import warnings

    from cornerstone.model import CornerNet, ModelConfig

    with torch.cuda.device("cuda:1"):
        m = CornerNet(ModelConfig(dim=128, blocks=4, fp8=True)).to("cuda:1").eval()
    p = torch.randn(32, 9, 14, 14, device="cuda:1")
    s = torch.randn(32, 44, device="cuda:1")

    torch.cuda.set_device(0)                     # 故意让当前设备对不上
    assert torch.cuda.current_device() == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pol, wdl, _ = m(p, s)
        torch.cuda.synchronize("cuda:1")
    assert torch.isfinite(pol).all() and pol.shape == (32, ACTIONS_)
    assert not [w for w in caught if "quantized compute" in str(w.message)], \
        "退回了非量化计算 —— FP8 名存实亡"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少两张卡")
def test_fp8_adamw_works_when_current_device_differs():
    """优化器也要能在「当前设备对不上」时正常更新。

    护栏最早只加在模型前向上，结果 A/B 实验第二次崩在 Fp8AdamW 里：
    read_weight 的 dequantize 也按当前设备分配，模型在 cuda:1 而当前设备
    是 cuda:0 就报 Expected all tensors to be on the same device。
    """
    with torch.cuda.device("cuda:1"):
        lin = F.fp8_linear(128, 256, bias=False).to("cuda:1")
    before = F.read_weight(lin.weight).clone()
    opt = F.Fp8AdamW(lin.parameters(), lr=1e-2)

    torch.cuda.set_device(0)                      # 故意错开
    for _ in range(5):
        with torch.cuda.device("cuda:1"), F.fp8_autocast():
            loss = lin(torch.randn(32, 128, device="cuda:1",
                                   dtype=torch.bfloat16)).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    after = F.read_weight(lin.weight)
    assert F.is_quantized(lin.weight)
    assert (after - before).abs().mean() > 0
    assert torch.isfinite(after).all()


def test_fp8_adamw_handles_mixed_quantized_and_plain_params():
    lin = F.fp8_linear(64, 64, bias=False).cuda()
    plain = torch.nn.Linear(64, 64).cuda()
    opt = F.Fp8AdamW(list(lin.parameters()) + list(plain.parameters()), lr=1e-3)

    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    with F.fp8_autocast():
        out = lin(x)
    loss = plain(out.float()).pow(2).mean()
    loss.backward()
    w_before = plain.weight.detach().clone()
    opt.step()
    assert not torch.equal(plain.weight.detach(), w_before), "非量化参数也应该被更新"
