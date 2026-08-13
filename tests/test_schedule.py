"""学习率日程：WSD，以及「跑满之后不许再落盘」。

这两件事凑在一起是有原因的 —— WSD 的卖点是「训多久不必一开始指定」，
而「随时能加训」这个动作**每次都会经过跑满后再启动那条路径**。
那条路径上有一个已经在磁盘上发生过的事故（见文件末尾那条用例）。

不需要 GPU：`Trainer.lr_at` 是纯函数，只读 `TrainConfig`。
"""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cornerstone.train import TrainConfig, Trainer  # noqa: E402


def lr_curve(cfg: TrainConfig, steps):
    """只借 lr_at 这个纯函数，不真的建 Trainer（那要 GPU、要建模型）。"""
    fake = object.__new__(Trainer)
    fake.cfg = cfg
    return [Trainer.lr_at(fake, s) for s in steps]


def decay_start(cfg: TrainConfig) -> int:
    fake = object.__new__(Trainer)
    fake.cfg = cfg
    return Trainer.decay_start_step(fake)


WSD = dict(lr=2e-3, min_lr_ratio=0.1, warmup_steps=500, lr_schedule="wsd",
           total_steps=111_000, lr_horizon_steps=111_000, lr_decay_steps=11_000)


def test_wsd_is_flat_then_decays():
    """warmup 线性上 → stable **恒等于** lr → 末段单调退到 lr*min_lr_ratio。

    「恒等于」是 WSD 与余弦的**唯一**区别，也是它「预算不必预先知道」的全部来源：
    stable 段上的任何一步都可以当成分叉点，再花约 10% 的步数退火收工。
    stable 段只要偷偷带一点斜率，这个性质就没了 —— 而曲线看上去还是对的。
    """
    cfg = TrainConfig(**WSD)
    assert decay_start(cfg) == 100_000

    assert lr_curve(cfg, [0])[0] == pytest.approx(2e-3 / 500)
    assert lr_curve(cfg, [499])[0] == pytest.approx(2e-3)

    stable = lr_curve(cfg, [500, 1000, 40_000, 99_999])
    assert all(v == pytest.approx(2e-3) for v in stable), f"stable 段不平：{stable}"

    decay = lr_curve(cfg, range(100_000, 111_001, 500))
    assert decay[0] == pytest.approx(2e-3)
    assert decay[-1] == pytest.approx(2e-4)          # lr * min_lr_ratio
    assert all(a > b for a, b in zip(decay, decay[1:])), "退火段不是单调下降"
    # 线性：中点应该正好在两端的算术平均上
    mid = lr_curve(cfg, [105_500])[0]
    assert mid == pytest.approx((2e-3 + 2e-4) / 2, rel=1e-6)

    # 越过 horizon 之后停在底部，不再往下（也不许回卷）
    assert lr_curve(cfg, [130_000])[0] == pytest.approx(2e-4)


def test_wsd_decay_window_does_not_move_when_total_steps_changes():
    """**这条是 `lr_horizon_steps` 存在的全部理由。**

    如果退火窗口锚在 `total_steps` 上（`decay_start = total_steps - decay`），
    那么「训完 11.1 万觉得还能再练，把 total_steps 改成 25 万」的那一刻，
    已经退到 2e-4 的学习率会**跳回 2e-3** —— 那不是「加训」，
    是「做了一次 warm restart 之后再加训」，而两者在日志上长得一模一样。

    horizon 独立之后，`total_steps` 就真的只剩「这次跑到哪停」这一个含义。
    """
    a = TrainConfig(**WSD)
    b = TrainConfig(**{**WSD, "total_steps": 250_000})       # 只改这一个
    probe = [500, 50_000, 99_999, 100_000, 105_500, 111_000, 130_000]
    assert lr_curve(a, probe) == lr_curve(b, probe)
    assert decay_start(a) == decay_start(b) == 100_000

    # 真要把曲线拉长，得**同时**动 horizon —— 这是显式的，不是副作用
    c = TrainConfig(**{**WSD, "total_steps": 250_000, "lr_horizon_steps": 250_000})
    assert decay_start(c) == 239_000
    assert lr_curve(c, [111_000])[0] == pytest.approx(2e-3), "延长后 stable 段该继续"


def test_cosine_unchanged_when_horizon_is_zero():
    """余弦分支也改用 horizon 了，`horizon=0` 时必须与改动前逐位相同。

    v2-* 那批跑用的是余弦，将来复现它们要靠这条。
    """
    import math
    c = TrainConfig(lr=2e-3, min_lr_ratio=0.1, warmup_steps=500, total_steps=150_000)
    assert c.lr_schedule == "cosine" and c.lr_horizon_steps == 0

    def old(step):                                   # 改动前的实现，逐字抄下来
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        t = min(1.0, (step - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * cos)

    probe = [0, 1, 499, 500, 1000, 75_000, 149_999, 150_000, 200_000]
    assert lr_curve(c, probe) == [old(s) for s in probe]
    assert decay_start(c) == 0, "非 wsd 不该有 stable 终点"


def test_wsd_decay_defaults_to_a_tenth_of_the_horizon():
    """`lr_decay_steps=0` 时退火段取 horizon 的 10% —— 文献上的常用值。"""
    cfg = TrainConfig(lr_schedule="wsd", total_steps=100_000, warmup_steps=500)
    assert decay_start(cfg) == 90_000
    assert lr_curve(cfg, [89_999])[0] == pytest.approx(cfg.lr)
    assert lr_curve(cfg, [100_000])[0] == pytest.approx(cfg.lr * cfg.min_lr_ratio)


def test_finished_run_does_not_overwrite_checkpoint():
    """跑满之后再被启动一次，**必须一个字节都不写**。

    这不是假想的：`runs/v2-bf16` 的磁盘上就留着后果 ——

        step00140227.pt   170 MB   有 exp_avg
        step00150227.pt    56 MB   无 exp_avg    <- latest 指向它

    成因：跑满后启动，训练循环立刻 break，一步没训就掉进收尾的
    `save_checkpoint()`，而 AdamW 的 `state` 是**懒创建**的、此时还空着。
    于是一份没有动量的档覆盖掉了同名的里程碑，replay 快照也被重写。
    守护脚本每 3 分钟查一次，配置一改就会走这条路径；
    **WSD 的「续跑下一段」更是每次都要经过它**。

    这里断言的是源码而不是行为：真跑一次要 GPU、要建模型、要几分钟，
    而这个保护的全部内容就是「早返回，别走到 break」这一个控制流。
    """
    src = open(os.path.join(REPO, "tools", "train.py"), encoding="utf-8").read()
    assert "step_at_start = trainer.step" in src, "少了区分「本次一步没训」的基准"
    lines = src.splitlines()
    i = next(n for n, ln in enumerate(lines)
             if ln.strip() == "if trainer.step >= cfg.total_steps:")
    body = lines[i:i + 40]
    assert any(ln.strip() == "if trainer.step == step_at_start:" for ln in body)

    # **`return 0` 必须排在 `break` 前面** —— 反过来就是「先 break 到收尾落盘、
    # 再判断」，那正是 56 MB 残档的成因。
    # 按整行比对，不按子串：这段的注释里就写着「若照常 break 出去…」，
    # `src.index("break")` 会先撞上那个词。
    def line_of(kw):
        return next(n for n, ln in enumerate(body) if ln.strip() == kw)

    assert line_of("return 0") < line_of("break"), \
        "一步没训时必须直接 return，不能 break 到收尾落盘"


def test_stable_checkpoint_is_preserved_on_entering_decay(tmp_path):
    """跨进退火段的那一步要留一份**永久**的 stable 档 + replay 快照。

    默认机制留不住它，两个理由各自独立：
      * replay 快照只有一份、每次 `save_snapshot()` 覆盖同一个文件
      * 里程碑保留的是「每个 milestone_every_steps 桶里的第一份」，而 100,000
        落在桶 10 里、桶 10 早被 100,0xx 之前的某一份占了

    没有这两份东西，「以后从 stable 续到 25 万」只能从零重跑 ——
    而那正是选 WSD 的全部理由。
    """
    saved = []
    fake = object.__new__(Trainer)
    # ckpt_dir 是从 cfg.run_dir 推出来的 property，不能直接赋值
    fake.cfg = TrainConfig(**WSD, run_dir=str(tmp_path))
    os.makedirs(fake.ckpt_dir, exist_ok=True)
    fake._stable_saved = False

    def _save_ckpt(tag=None, update_latest=True):
        saved.append(("ckpt", tag, update_latest))
        open(os.path.join(fake.ckpt_dir, f"{tag}.pt"), "w").close()

    fake.save_checkpoint = _save_ckpt
    fake.save_snapshot = lambda name="replay.npz": saved.append(("snap", name))

    fake.step = 99_999
    assert Trainer.maybe_save_stable(fake) is False and not saved

    fake.step = 100_400                      # 一轮 400 步，正好跨过去
    assert Trainer.maybe_save_stable(fake) is True
    # **不许动 latest**：stable 是分叉点，不是这条跑的进度
    assert ("ckpt", "stable", False) in saved and ("snap", "stable.npz") in saved

    saved.clear()
    fake.step = 104_000                      # 只留**一份**，后面的步数不再重复落
    assert Trainer.maybe_save_stable(fake) is False and not saved

    # 恢复之后进程内标志归零，但磁盘上已经有 stable.pt —— 仍然不许覆盖，
    # 否则分叉点会一路漂到退火段里去
    fake._stable_saved = False
    assert Trainer.maybe_save_stable(fake) is False and not saved

    # 余弦那组完全不该触发
    cos = object.__new__(Trainer)
    cos.cfg = TrainConfig(total_steps=150_000, run_dir=str(tmp_path))
    cos._stable_saved = False
    cos.step = 149_000
    cos.save_checkpoint = cos.save_snapshot = lambda *a, **k: saved.append("不该被调到")
    assert Trainer.maybe_save_stable(cos) is False and not saved


def test_wsd_fields_live_in_the_experiment_script_not_env():
    """WSD 的三个字段必须写在 `ab_experiment.sh` 里，而不是靠环境变量传。

    `save_checkpoint` 存了 `asdict(cfg)`，但 `load_checkpoint` **从不读它** ——
    学习率曲线的形状 100% 由本次命令行决定。而守护脚本自动恢复时只转交实验名
    与设备：任何靠环境变量传进来的旋钮都会在那一刻悄悄退回默认值。
    `v2-fp8` 的 `parallel_games` 就是这么在第 12.8 万步被从 4096 改成 2048 的。

    日程形状拆成两处：`--lr-schedule wsd` 全跑共用，放 COMMON；
    三个步数字段按实验名变（加长跑要 22 万），放 `budget_for()`。
    两处都必须真的被 `start_leg` 用上。
    """
    sh = open(os.path.join(REPO, "scripts", "ab_experiment.sh"), encoding="utf-8").read()
    common = sh[sh.index("COMMON=("):sh.index(")", sh.index("COMMON=("))]
    assert "--lr-schedule wsd" in common

    fn = sh[sh.index("budget_for()"):sh.index("}", sh.index("budget_for()"))]
    for flag in ("--total-steps", "--lr-horizon-steps", "--lr-decay-steps"):
        assert flag in fn, f"{flag} 不在 budget_for 里"
    assert "220000" in fn and "111000" in fn, "两种预算都要在"
    assert '$(budget_for "$exp")' in sh, "budget_for 定义了却没被 start_leg 用上"


def test_watchdog_asks_the_experiment_script_for_the_budget():
    """守护脚本判「跑满没有」时**调用** `ab_experiment.sh budget`，不自己解析文件。

    原来是 `sed ... | head -1` 抠第一条 `--total-steps`。所有跑共用一个预算时
    凑合能用；一旦按实验名分派，它会给出别人的数字 —— 表现为
    「`v4-bf16-long` 刚到 11.1 万就被判定跑满、不再拉起」，而日志上只写着
    「训练完成」，看不出是判据错了。同一套规则不能写在两个地方。
    """
    sh = open(os.path.join(REPO, "scripts", "watch_training.sh"), encoding="utf-8").read()
    fn = sh[sh.index("total_steps()"):sh.index("}", sh.index("total_steps()"))]
    assert "ab_experiment.sh" in fn and "budget" in fn, "还在自己解析步数"
    assert "sed" not in fn, "total_steps 里不该再有 sed 解析"
    # 调用点都要把实验名传进去，否则等于没分派
    assert 'total_steps "$exp"' in sh


def test_watchdog_refuses_to_launch_an_arm_it_cannot_classify():
    """守护脚本按实验名分派精度，**认不出来就必须拒绝拉起**。

    这里原先是 `*fp8*) start-b;; *) start-a`，于是任何不含 "fp8" 的名字都被当成
    BF16 组 —— 加 FP4 组之后，`v4-fp4` 会被静默拉成一条 BF16 的跑，
    而且只在「开发机过期后自动恢复」那一刻发生，日志上完全看不出来。
    """
    sh = open(os.path.join(REPO, "scripts", "watch_training.sh"), encoding="utf-8").read()
    body = sh[sh.index("start_training()"):]
    # 匹配 case 分支本身而不是裸模式串 —— 注释里也会出现 `*bf16*` 之类的字样，
    # 用裸串会把注释的位置当成分支的位置，序关系就判错了（写这条时踩了一次）。
    # 用正则是因为分支之间为了对齐用了不同数量的空格。
    import re
    pos = {}
    for pat, arm in (("*-long", "start-d"), ("*fp4*", "start-c"),
                     ("*fp8*", "start-b"), ("*bf16*", "start-a")):
        m = re.search(re.escape(pat) + r'\)\s+arm="' + arm + '"', body)
        assert m, f"{pat} 没有分派到 {arm}"
        pos[pat] = m.start()
    # **`-long` 必须排在 `*bf16*` 前面**：`v4-bf16-long` 两者都匹配，
    # 落到 start-a 的话预算退回 11.1 万，加长跑会在半路被判定跑满
    assert pos["*-long"] < pos["*bf16*"], "-long 没有排在 *bf16* 之前"
    assert "return 1" in body[:body.index("esac")], "认不出精度时必须拒绝拉起"


def test_experiment_script_derives_precision_from_the_run_name():
    """精度也按实验名分派、写死在脚本里，理由与 extra_for 完全相同。"""
    sh = open(os.path.join(REPO, "scripts", "ab_experiment.sh"), encoding="utf-8").read()
    fn = sh[sh.index("precision_for()"):sh.index("esac", sh.index("precision_for()"))]
    assert '*fp4*)' in fn and '*fp8*)' in fn and '*bf16*)' in fn
    # **不能后缀锚定**：`*-fp8)` 匹配不上 `v4-fp8-long`，那条跑会被静默当成 bf16
    assert "*-fp8)" not in fn and "*-fp4)" not in fn, "精度分派又用回了后缀锚定"
    assert "exit 1" in fn, "认不出精度时必须报错退出，不能默默回退到 bf16"
    assert '--precision "$(precision_for "$exp")"' in sh
    assert "--fp8" not in sh, "还留着老的 --fp8 开关，那个参数已经不存在了"

    # 种子同理：`v4-bf16-s2` 存在的唯一理由就是换种子，守护恢复时若退回
    # 默认 seed 1，它就变成 v4-bf16 的重复跑，而日志上看不出来
    sf = sh[sh.index("seed_for()"):sh.index("esac", sh.index("seed_for()"))]
    assert "*-s2) echo 2" in sf and "*) *echo 1" not in sf
    assert '--seed "$(seed_for "$exp")"' in sh, "seed_for 定义了却没被 start_leg 用上"
    assert "--seed 1\n" not in sh, "COMMON 里还写死着 --seed 1"
