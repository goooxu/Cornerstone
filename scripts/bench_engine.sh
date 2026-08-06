#!/usr/bin/env bash
# 引擎吞吐基准。在容器内跑：
#   bash scripts/devbox.sh exec bash scripts/bench_engine.sh [线程数]
#
# 关心两个数：单核每秒能生成多少次着法、每秒能跑完多少局随机对局。
# 这决定了自博弈里 CPU 侧的天花板。

set -euo pipefail
THREADS="${1:-1}"
exec python3 "$(dirname "${BASH_SOURCE[0]}")/../tools/bench_engine.py" --threads "$THREADS"
