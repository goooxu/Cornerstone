#!/usr/bin/env bash
# 资源利用率快照：GPU 与 CPU 各看一眼，用来验收「压满 CPU 和 GPU」这条要求。
#
#   bash scripts/monitor.sh [采样秒数]
#
# GPU 指标优先从常驻的 dcgm-exporter 抓（比反复起 nvidia-smi 便宜且更准），
# 抓不到就退回 nvidia-smi。

set -uo pipefail
SECS="${1:-10}"
DCGM_URL="${DCGM_URL:-http://127.0.0.1:9400/metrics}"

echo "=== 采样 ${SECS}s ==="

echo
echo "--- GPU ---"
if curl -s --max-time 3 "$DCGM_URL" >/dev/null 2>&1; then
  curl -s --max-time 3 "$DCGM_URL" | awk '
    /^DCGM_FI_DEV_GPU_UTIL\{/       { split($0,a,"} "); util[++n]=a[2] }
    /^DCGM_FI_DEV_FB_USED\{/        { split($0,a,"} "); mem[++m]=a[2] }
    /^DCGM_FI_DEV_SM_CLOCK\{/       { split($0,a,"} "); clk[++c]=a[2] }
    END {
      printf "%-6s %10s %12s %10s\n", "GPU", "利用率%", "显存MiB", "SM时钟"
      for (i = 1; i <= n; i++) printf "%-6d %10s %12s %10s\n", i-1, util[i], mem[i], clk[i]
    }'
else
  echo "（dcgm-exporter 不可达，退回 nvidia-smi）"
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
             --format=csv 2>/dev/null || echo "nvidia-smi 也用不了"
fi

echo
echo "--- CPU ---"
if command -v mpstat >/dev/null 2>&1; then
  mpstat "$SECS" 1 2>/dev/null | tail -2
else
  # 退回读 /proc/stat 自己算
  read -r _ a b c d rest < /proc/stat
  idle0=$d; tot0=$((a+b+c+d))
  sleep "$SECS"
  read -r _ a b c d rest < /proc/stat
  idle1=$d; tot1=$((a+b+c+d))
  busy=$(( (tot1-tot0) - (idle1-idle0) ))
  echo "整机 CPU 利用率: $(( busy * 100 / (tot1-tot0) ))%  （$(nproc) 核）"
fi

echo
echo "--- 本项目相关进程 ---"
ps -eo pid,pcpu,pmem,rss,etime,comm --sort=-pcpu 2>/dev/null | head -8
