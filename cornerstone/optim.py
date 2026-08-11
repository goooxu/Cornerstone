"""bf16 计算权重 + fp32 主权重的 AdamW。

低精度训练的分工是固定的：**低精度负责算得快（GEMM 输入），高精度负责记得准
（主权重、优化器状态、累加器）**。模型参数是 bf16，前向反向都用它；
但真正被优化器更新的是一份 fp32 副本（master），每步更新完再舍入回 bf16。

为什么必须有 master —— 在本项目自己的超参上实测（权重取实测量级 0.17，
恒定梯度连更 400 步）：

    学习率      每步更新      有 fp32 master     直接更新 bf16
    2e-3        2.05 ulp       99.9%              112.1%
    2e-4        0.20 ulp      100.2%              **0.1%**

`min_lr_ratio=0.1`，所以 2e-4 就是余弦退火的终点学习率。**在那里，没有 master
的话更新几乎被完整吞掉** —— 权重只走了应走位移的千分之一，而训练看起来一切正常。
峰值学习率下反而看不出问题（甚至偏大 12%，那是舍入噪声在随机游走），
所以这个故障只会在训练后半程显现，最难查。

这个文件刻意不 import transformer_engine：master 权重是两条腿共用的脚手架，
和 FP8 无关，而 BF16 那条腿必须能在没装 TE 的机器上跑。
"""

from __future__ import annotations

import copy

import torch
from torch.nn.utils import clip_grad_norm_


class MasterWeightAdamW(torch.optim.AdamW):
    """参数是 bf16，被更新的是 fp32 master；m/v 由父类持有，也是 fp32。

    **继承 AdamW 而不是重写一遍 AdamW 数学**，这样：
      - `state_dict()` 直接是 torch 标准格式（`exp_avg`/`exp_avg_sq`/`step`），
        续训、迁移、别人读都不用额外约定；
      - 拿得到 foreach/fused 内核，不会因为换成 Python 循环把步时拖慢；
      - 调用方改 `param_groups[*]["lr"]` 的写法一个字都不用动。

    传进来的 group 里除了 `params` 还要带一个等长的 `names`（参数全名）——
    `master_state_dict()` 要用它拼 checkpoint 的键。
    """

    def __init__(self, groups, lr: float, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self._params: list[torch.Tensor] = []
        self._masters: list[torch.Tensor] = []
        self._names: list[str | None] = []
        master_groups = []
        for g in groups:
            g = dict(g)
            params = list(g.pop("params"))
            # names 必须**从 group 里拿走**，不能留着：
            # `Optimizer.load_state_dict` 会用存档里的 param_groups 覆盖所有自定义键，
            # 读一份旧 checkpoint 就会让 names 凭空消失，而 master 存盘随即失去键名。
            names = list(g.pop("names", []))
            if names and len(names) != len(params):
                raise ValueError(f"names 与 params 不等长: {len(names)} vs {len(params)}")
            masters = [p.detach().float().clone() for p in params]
            self._params += params
            self._masters += masters
            self._names += names or [None] * len(params)
            master_groups.append({**g, "params": masters})
        super().__init__(master_groups, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)

    # ---- 训练循环 ----

    def zero_grad(self, set_to_none: bool = True) -> None:
        """父类只认它自己的 params（= master），**模型参数的梯度必须自己清**。

        漏了这一句不会报错：模型参数的 `.grad` 一直累加，loss 越训越怪，
        但每一步看着都正常。
        """
        super().zero_grad(set_to_none=set_to_none)
        for p in self._params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None, *, grad_clip: float | None = None) -> torch.Tensor:
        """上抬梯度 -> 裁剪 -> fp32 更新 -> 写回 bf16。返回**裁剪前**的梯度范数。

        裁剪收进 step 里而不是留给调用方，是因为它必须发生在
        「梯度已上抬到 fp32」和「AdamW 更新」之间 —— 做成相邻的两次调用，
        顺序就有写错的余地，而写错了不报错，只是裁剪在 bf16 上算得不准。
        """
        if closure is not None:
            raise NotImplementedError("MasterWeightAdamW 不支持 closure")
        for p, m in zip(self._params, self._masters):
            if p.grad is None:
                m.grad = None
                continue
            # copy=True 不能省：参数本身是 fp32 时（CPU、smoke 配置）`.to(float32)`
            # 返回的就是原张量，后面任何就地操作会直接改坏模型的梯度。
            m.grad = p.grad.detach().to(torch.float32, copy=True)
        # max_norm=inf 时不裁剪，但照样把范数算出来给 metrics 用
        gnorm = clip_grad_norm_(self._masters,
                                float("inf") if grad_clip is None else grad_clip)
        super().step()
        self.sync_params()
        return gnorm

    @torch.no_grad()
    def sync_params(self) -> None:
        """master -> bf16 参数。舍入到最近，不需要随机舍入 —— master 保着全精度。"""
        for p, m in zip(self._params, self._masters):
            p.copy_(m)

    # ---- 主权重的存读 ----

    def master_state_dict(self) -> dict[str, torch.Tensor]:
        if any(n is None for n in self._names):
            raise RuntimeError("构造优化器时没有传 names，无法导出主权重")
        return {n: m.detach().clone() for n, m in zip(self._names, self._masters)}

    @torch.no_grad()
    def load_master_state_dict(self, sd: dict, *, partial: bool = False) -> int:
        """从 fp32 张量装回 master，并**顺带同步 bf16 参数**。

        同步放在这里而不是让调用方补一句：参数与 master 必须一致，
        由一次调用保证，就不给「顺序写反」留空间。

        `sd` 里可以有多余的键（buffers、TE 的 `_extra_state`），一律忽略；
        `partial=False` 时要求每个 master 都被覆盖到。
        """
        idx = {n: i for i, n in enumerate(self._names) if n is not None}
        hit = set()
        for k, v in sd.items():
            i = idx.get(k)
            if i is None:
                continue
            self._masters[i].copy_(v.to(self._masters[i].device, torch.float32))
            hit.add(k)
        if not partial:
            missing = [n for n in idx if n not in hit]
            if missing:
                raise KeyError(f"主权重缺少 {len(missing)} 项，例如 {missing[:3]}")
        self.sync_params()
        return len(hit)

    # ---- 优化器状态的存读 ----

    def load_state_dict(self, state_dict: dict, master: dict | None = None) -> None:
        state_dict = copy.deepcopy(state_dict)
        migrated, dropped = self._normalize_legacy(state_dict)
        if migrated:
            print(f"[优化器] 旧格式动量已迁移 {migrated} 项：m/v -> exp_avg/exp_avg_sq",
                  flush=True)
        if dropped:
            print(f"[优化器] 警告：{dropped} 项优化器状态无法识别，已整条丢弃"
                  f"（这些参数的动量与步数从零重启）", flush=True)
        super().load_state_dict(state_dict)
        if master is not None:
            self.load_master_state_dict(master, partial=True)

    @staticmethod
    def _normalize_legacy(state_dict: dict) -> tuple[int, int]:
        """把 `Fp8AdamW` 存的 `{step:int, m, v}` 翻译成 AdamW 的格式。

        两者数学上等价（同 betas、同 bias correction、同解耦权重衰减），所以旧跑
        仍然能续训，不用只当权重来源。**必须在 load 之前翻译**：torch 会把 `m`/`v`
        照单收下，直到续训的第一次 `step()` 才抛 `KeyError: 'exp_avg'` ——
        开发机每 8 小时回收一次、续训是最高频路径，这个坑一定会踩到。
        """
        migrated = dropped = 0
        state = state_dict.get("state", {})
        for key in list(state):
            st = state[key]
            if not isinstance(st, dict):
                continue
            if "exp_avg" not in st:
                if "m" in st and "v" in st:
                    st["exp_avg"] = st.pop("m").float()
                    st["exp_avg_sq"] = st.pop("v").float()
                    migrated += 1
                else:
                    # **整条删掉，不能只删动量留下 step**：torch 的 AdamW 用
                    # `len(state) == 0` 判断要不要初始化，留个 step 就会让它跳过
                    # 初始化，然后原地抛 `KeyError: 'exp_avg'` —— 正是这个函数
                    # 要防的那个错。
                    del state[key]
                    dropped += 1
                    continue
            if isinstance(st.get("step"), (int, float)):
                st["step"] = torch.tensor(float(st["step"]))
        return migrated, dropped
