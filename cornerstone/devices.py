"""设备列表解析。

这里原先是 `multigpu.py`：每张卡一份模型副本 + 一个独立的 `SelfPlayDriver`，
用 Python 线程并发跑。**已被 `pool.py`（每卡一个工作进程）取代** ——
线程方案下 FP8 自博弈完全不扩展（4 卡 39k 评估/s，还不如 2 卡），
根因是同进程内的 GIL 与 TE 全局 FP8 状态争用，见 `docs/07`。

那个类整个删掉而不是留着当备选：它的权重同步语义和工作池**正好相反**
（线程方案里父进程持有真身、每轮往副本推；工作池里真身在工作进程，
父进程往下推就是拿陈旧权重覆盖训练结果）。两套相反的语义并存，
接错了不报错，只表现为「怎么训都不涨棋力」。
"""

from __future__ import annotations

import torch


def visible_devices(spec: str = "") -> list[str]:
    """把 "cuda:1,cuda:2" 之类的字符串解析成设备列表；空串表示用全部可见 GPU。"""
    if spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]
