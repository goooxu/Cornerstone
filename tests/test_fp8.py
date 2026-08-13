"""低精度混合训练：GEMM 走 MXFP8 / NVFP4，周边一律高精度。

核心要验证的是四件事：**低精度只改计算不改存储**（权重是普通 bf16 张量）、
**量化 GEMM 真的在算而且算的是这个精度**（TE 静默降级是这套方案最危险的失效
模式）、**三组的初值逐位相同**（否则 A/B 里混进隐藏变量），
以及维度约束确实存在（它一路传导到 batch size 上）。

主权重与随机舍入相关的用例已经删掉 —— 主权重现在是 fp32，由
`MasterWeightAdamW` 持有，对应的测试在 `test_optim.py`。
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")

if not torch.cuda.is_available():
    pytest.skip("低精度需要 GPU", allow_module_level=True)

from cornerstone import fp8 as F  # noqa: E402
from cornerstone.model import CornerNet, ModelConfig  # noqa: E402


def _ok(precision: str) -> bool:
    a = F.is_available(precision)
    return a[0] if isinstance(a, tuple) else bool(a)


_MX = _ok("fp8")
_NV = _ok("fp4")
if not _MX:
    pytest.skip(f"当前 GPU 不支持 MXFP8: {F.is_available('fp8')}", allow_module_level=True)

needs_fp4 = pytest.mark.skipif(not _NV, reason=f"当前 GPU 不支持 NVFP4: {F.is_available('fp4')}")


@pytest.fixture
def linear():
    return F.quant_linear("fp8", 256, 512, bias=False).cuda()


def _small(precision: str) -> CornerNet:
    """**blocks 至少 4**：首尾 block 被强制走高精度，blocks=2 的模型里
    一个 te.Linear 都没有，所有低精度断言会静默变成空断言。"""
    torch.manual_seed(0)
    return CornerNet(ModelConfig(dim=64, blocks=4, attn_every=2, precision=precision)).cuda()


def _n_te(model) -> int:
    import transformer_engine.pytorch as te
    return sum(isinstance(m, te.Linear) for m in model.modules())


def _dummy(dev="cuda"):
    return (torch.zeros(8, 9, 14, 14, device=dev), torch.zeros(8, 44, device=dev))


# ---- 配置：三值精度与老 checkpoint 的兼容 ----

def test_precision_field_rejects_typos():
    """CLI 是从 dataclass 自动生成的、没有 choices —— `__post_init__` 是唯一防线。

    没有它，`--precision fp16` 这种手误会安静地建出一个 `quantized == False`
    的配置：跑起来一切正常，只是那组其实是 BF16。
    """
    with pytest.raises(AssertionError):
        ModelConfig(precision="fp16")
    for p in ("bf16", "fp8", "fp4"):
        assert ModelConfig(precision=p).precision == p
    assert ModelConfig(precision="bf16").quantized is False
    assert ModelConfig(precision="fp8").quantized is True
    assert ModelConfig(precision="fp4").quantized is True


def test_legacy_model_config_without_precision_loads_as_fp8():
    """只写了 `fp8: True`、没有 `precision` 的老 blob 必须装成 fp8 模型。

    `runs/v2-fp8` 的 checkpoint 全长这样。而 `load_checkpoint` 对未知键做**静默
    过滤**，`te.Linear` 与 `nn.Linear` 的 state_dict 键名又都是 `weight`
    （`_extra_state` 被 `load_weights` 剥掉）—— 所以一旦这条兼容断了，
    老 checkpoint 会**静默降级成 BF16 模型**，不抛任何异常，
    只是「FP8 组的复现」悄悄变成了 BF16。
    """
    old = {"dim": 64, "blocks": 4, "attn_every": 2, "fp8": True}
    cfg = ModelConfig(**old)
    assert cfg.precision == "fp8" and cfg.quantized
    torch.manual_seed(0)
    assert _n_te(CornerNet(cfg).cuda()) > 0

    assert ModelConfig(**{**old, "fp8": False}).precision == "bf16"
    # asdict 往返：新配置里 precision 已经显式给了，不能被回填的 fp8 顶回去
    from dataclasses import asdict
    for p in ("bf16", "fp8", "fp4"):
        blob = asdict(ModelConfig(dim=64, blocks=4, precision=p))
        assert blob["fp8"] is (p == "fp8"), "asdict 出来的 blob 必须仍带 fp8 兼容字段"
        assert ModelConfig(**blob).precision == p, f"{p} 往返之后变了"


# ---- 存储：低精度不改变权重的存储格式 ----

def test_weights_are_plain_bf16_not_quantized(linear):
    """权重必须是**普通** bf16 张量。

    这条守的是「不要退回 FP8 主权重」：`quantized_model_init` 会让权重张量本身
    变成量化格式（行/列两套数据 + 两套块缩放），那样就没有高精度 master 了，
    退火尾段的更新会被整片吃掉（量化依据见 test_optim.py）。
    """
    linear.to(torch.bfloat16)
    w = linear.weight
    assert w.dtype is torch.bfloat16
    assert not hasattr(w, "_rowwise_data"), "权重被量化了 —— FP8 主权重方案又回来了"
    assert w.element_size() == 2


def test_three_legs_have_identical_parameter_footprint():
    """三组的参数量、逐参数 dtype、总字节数必须完全相同。

    这是「唯一的区别是 GEMM 精度」这句话的可执行表述。一旦哪天存储又出现差异，
    A/B/C 就不再是单变量对照了。
    """
    legs = {p: _small(p).to(torch.bfloat16) for p in ("bf16", "fp8", "fp4")}
    assert _n_te(legs["bf16"]) == 0, "BF16 组里出现了 te.Linear"
    assert _n_te(legs["fp8"]) > 0 and _n_te(legs["fp4"]) > 0, "量化组里没有 te.Linear"
    assert _n_te(legs["fp8"]) == _n_te(legs["fp4"]), "fp8 与 fp4 的量化层数不同"

    def shape_of(m):
        return {n: (p.shape, p.dtype, p.numel() * p.element_size())
                for n, p in m.named_parameters()}

    ref = shape_of(legs["bf16"])
    for p, m in legs.items():
        assert shape_of(m) == ref, f"{p} 组的参数存储与 bf16 组不同"


def test_three_legs_start_from_identical_weights():
    """同一 seed 下**三组两两**的初始权重必须逐位相同。

    `te.Linear` 默认用 normal(0, 0.023) 初始化而 `nn.Linear` 用 kaiming_uniform，
    两者消耗的 RNG 流长度也不同 —— 不做对齐的话，126 个参数张量里有 75 个不同，
    连不含 TE 的卷积和位置嵌入都跟着错位。这个隐藏变量污染过一次 A/B，
    `make_linear` 里那句无条件的 `ref = nn.Linear(...)` 就是唯一的对齐点，
    而它删掉之后不会报任何错。
    """
    legs = {p: _small(p) for p in ("bf16", "fp8", "fp4")}
    assert _n_te(legs["fp4"]) > 0, "FP4 组里没有 te.Linear，这条测试没在测东西"
    ref = dict(legs["bf16"].named_parameters())
    for p in ("fp8", "fp4"):
        cur = dict(legs[p].named_parameters())
        assert ref.keys() == cur.keys()
        bad = [n for n in ref if not torch.equal(ref[n].float().cpu(), cur[n].float().cpu())]
        assert not bad, f"{p} 组有 {len(bad)}/{len(ref)} 个张量初值不同，例如 {bad[:3]}"


# ---- 计算：量化 GEMM 到底有没有在算、算的是不是这个精度 ----

def test_recipe_is_mxfp8_hybrid():
    """前向 E4M3（精度优先）/ 反向 E5M2（范围优先）。

    `MXFP8BlockScaling` 默认是纯 E4M3，必须显式指定 HYBRID —— 梯度的动态范围
    比激活宽得多，4 位指数不够用。
    """
    from transformer_engine.common import recipe
    r = F.recipe_for("fp8")
    assert isinstance(r, recipe.MXFP8BlockScaling)
    assert r.fp8_format is recipe.Format.HYBRID
    assert r.mxfp8() and not r.nvfp4()


@needs_fp4
def test_fp4_recipe_keeps_rht_and_stochastic_rounding():
    """NVFP4 的默认参数就是速查表要求的配置，**不能有任何 `disable_*` 被打开**。

    FP4 只有 2 位尾数，四件事缺一不可：
      * 两层缩放（每 16 元素一个 E4M3 因子 + 整张量 fp32 因子）
      * 输入与梯度过 16×16 随机 Hadamard（把离群值摊到块内）
      * 梯度随机舍入（确定性舍入在 2 位尾数下会系统性地偏置梯度）
      * 权重按 16×16 二维块量化

    这条把它们钉住：将来谁为了提速去关一个开关，会在这里而不是在一条
    「训得动但棋力莫名其妙差」的 5 小时长跑里被发现。
    """
    from transformer_engine.common import recipe
    r = F.recipe_for("fp4")
    assert isinstance(r, recipe.NVFP4BlockScaling)
    assert r.nvfp4() and not r.mxfp8()
    on = {k: v for k, v in vars(r).items() if k.startswith("disable_") and v}
    assert not on, f"这些必需的机制被关掉了：{sorted(on)}"


@pytest.mark.parametrize("precision", ["fp8", pytest.param("fp4", marks=needs_fp4)])
def test_probe_distinguishes_enabled_from_disabled(precision):
    """探针本身要有分辨力 —— 恒真的探针我们已经吃过一次亏。

    旧探针靠捕获 `quantized weights without quantized compute` 警告，而那条警告
    在非量化权重下永远不会出现，探针就静默退化成「总是 True」了。
    """
    m = _small(precision).to(torch.bfloat16)
    assert _n_te(m) > 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.gemm_active(m, precision, *_dummy()) is True   # 模型自己会开作用域

    # 把作用域摘掉：同样的模型必须被判成「没在算低精度」
    m.cfg.precision = "bf16"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.gemm_active(m, precision, *_dummy()) is False


@needs_fp4
def test_probe_tells_fp4_apart_from_fp8():
    """探针要能分辨**精度本身**，不只是「有没有在量化」。

    fp4 被静默降级成 fp8 的话，「作用域开着 + 有 te.Linear」这两条都还成立 ——
    只有配方谓词与模块自己的量化器类型能抓到。这是本轮新增 FP4 组之后
    最可能出现、又最没有症状的失效模式。
    """
    m = _small("fp4").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.gemm_active(m, "fp4", *_dummy()) is True
        assert F.gemm_active(m, "fp8", *_dummy()) is False, "fp4 作用域被判成了 fp8"

    m8 = _small("fp8").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.gemm_active(m8, "fp8", *_dummy()) is True
        assert F.gemm_active(m8, "fp4", *_dummy()) is False, "fp8 作用域被判成了 fp4"


@pytest.mark.parametrize("precision,tol", [("fp8", 0.05),
                                           pytest.param("fp4", 0.30, marks=needs_fp4)])
def test_quant_changes_numerics_but_stays_close(linear, precision, tol):
    """量化前向与高精度前向必须**不逐位相同**（证明量化真的发生了）且相对差在容差内。

    这是「开了但没生效」的终极判据 —— 数值不会说谎。
    容差按精度分开：FP8 实测相对误差约 3.8%，FP4 约 14.6%（3.9 倍），
    所以 FP4 用 30% 的上限 —— 它抓的是「彻底坏了」，不是「精度退化了多少」。
    """
    torch.manual_seed(2)
    linear.to(torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(512, 256, device="cuda") * 0.05)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    with F.quant_autocast(precision):
        y_q = linear(x)
    y_hp = linear(x)
    ref = x.float() @ linear.weight.float().T

    assert not torch.equal(y_q, y_hp), f"{precision} 与高精度输出逐位相同 —— 量化根本没发生"
    rel = (y_q.float() - ref).norm() / ref.norm()
    assert rel < tol, f"{precision} GEMM 与参考实现相对误差 {rel:.4f} 偏大"


def test_quant_requires_dims_divisible_by_32():
    """硬约束，会一路传导到 batch size 上，必须显式记下来。

    token 维是 B*196，而 196 mod 32 == 4，所以 **batch 必须是 8 的倍数**
    （8*196 = 1568 = 49*32）。自博弈里 prepare() 返回的批大小是变化的，
    所以送进量化前向之前必须补齐。

    **NVFP4 的微块虽然是 16，整除要求仍然是 32**，所以 `pad_to_mxfp8` 通用 ——
    这一点如果记错，FP4 组会在第一批变长的自博弈批上直接炸。
    """
    lin = F.quant_linear("fp8", 64, 64, bias=False).cuda().to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="divisible by 32"):
        with F.quant_autocast("fp8"):
            lin(torch.randn(8, 64, device="cuda", dtype=torch.bfloat16))
    with F.quant_autocast("fp8"):
        lin(torch.randn(32, 64, device="cuda", dtype=torch.bfloat16))   # 32 就没问题

    assert F.pad_to_mxfp8(1) == 8
    assert F.pad_to_mxfp8(8) == 8
    assert F.pad_to_mxfp8(9) == 16
    assert F.pad_to_mxfp8(1000) == 1000        # 1000*196 = 196000 = 6125*32
    for b in (8, 16, 24, 512, 1000):
        assert (b * 196) % 32 == 0


def test_all_legs_feed_bf16_into_the_gemm():
    """三组喂给量化层的输入 dtype 必须一致，而且必须是 bf16。

    `nn.RMSNorm` 在 autocast 下按 fp32 策略跑，输出 fp32 —— 而它的下游正是
    SwiGLU 与 Attention 的投影。BF16/MXFP8 吃得下，**NVFP4 直接报
    `RHT is only supported for bfloat16 input`**（随机 Hadamard 是 FP4 收敛的
    必需件，不能靠 `disable_rht` 绕）。`model.match_dtype` 把它折了回来。

    这条同时守住对照的单变量性：不折的话 FP8 从 fp32 量化、FP4 从 bf16 量化，
    两组就多差了一个「量化源精度」。
    """
    import transformer_engine.pytorch as te
    for precision in ("fp8", "fp4"):
        if precision == "fp4" and not _NV:
            continue
        m = _small(precision).to(torch.bfloat16)
        seen = []
        for mod in m.modules():
            if isinstance(mod, te.Linear):
                mod.register_forward_pre_hook(lambda md, a: seen.append(a[0].dtype))
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            m(*_dummy())
        assert seen, "一个 te.Linear 都没被调到"
        assert set(seen) == {torch.bfloat16}, f"{precision} 组喂进了 {set(seen)}"


# ---- 设备护栏 ----

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少两张卡")
def test_quant_model_works_when_current_device_differs():
    """量化模型放在非默认卡上、且调用时当前设备不匹配 —— 必须照常工作。

    TE 按**当前 CUDA 设备**取 cuBLAS 句柄。不匹配时的表现有两种，都很难查：
    要么静默退回非量化计算，要么直接非法访存（报错点还常飘到别的算子上）。
    自博弈的多卡驱动天然会撞上这个场景。

    断言用探针而不是「没有警告」—— 后者在非量化权重下恒真，等于没测。
    """
    torch.manual_seed(4)
    with torch.cuda.device(1):
        m = CornerNet(ModelConfig(dim=64, blocks=4, attn_every=2, precision="fp8"))
        m = m.to("cuda:1").to(torch.bfloat16)
    assert _n_te(m) > 0

    torch.cuda.set_device(0)                    # 故意让当前设备与模型不一致
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert F.gemm_active(m, "fp8", *_dummy("cuda:1")) is True
    assert torch.cuda.current_device() == 0, "探针不该改变调用方的当前设备"
