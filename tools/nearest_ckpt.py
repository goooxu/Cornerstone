#!/usr/bin/env python3
"""在一个 ckpt 目录里找出**最接近**目标步数的 checkpoint，打印「路径 步数」。

为什么不要求精确命中：训练早期会按 replay 数据量限流（`steps_for_iteration()`），
步数计数因此错位，落盘点是 19832 这种数而不是 20000。
要求精确匹配的话，对照实验永远找不到可对齐的 checkpoint。
"""

import os
import re
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("用法: nearest_ckpt.py <ckpt目录> <目标步数>", file=sys.stderr)
        return 2
    d, target = sys.argv[1], int(sys.argv[2])
    if not os.path.isdir(d):
        print(f"目录不存在: {d}", file=sys.stderr)
        return 1
    cks = []
    for f in os.listdir(d):
        m = re.fullmatch(r"step(\d+)\.pt", f)
        if m:
            cks.append((int(m.group(1)), f))
    if not cks:
        print(f"{d} 里没有 checkpoint", file=sys.stderr)
        return 1
    step, name = min(cks, key=lambda x: abs(x[0] - target))
    print(os.path.join(d, name), step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
