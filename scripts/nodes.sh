#!/usr/bin/env bash
# 列出当前分到手的开发节点、它们的 IP 和**剩余时间**。
#
#   bash scripts/nodes.sh            # 全部
#   bash scripts/nodes.sh --free     # 只列没在跑训练的
#
# 为什么需要这个：**开发机不是固定机器，是 Slurm 上的 8 小时作业。**
# 到点节点被收回、换一台，IP 跟着变。此前多次把这件事误判成「容器被回收」
# 或「CUDA IPC 坏了」——真正的症状是 ssh 报
#
#     Access denied by pam_slurm_adopt: you have no active jobs on this node
#
# 而不是容器不见了。区别很重要：容器问题重建容器就好，作业到期得换节点，
# 并且 `.devhosts` 里的 IP 全要跟着改。
#
# 剩余时间是排任务时的硬约束：一场 66 对的 arena 要 3 小时，
# 排在只剩 1 小时的节点上必然半路作废（net_arena 中途不落盘，见 docs/03）。

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="${CORNERSTONE_SSH:-$HOME/.local/bin/sshx}"
# 随便找一个能连上的节点来跑 squeue —— 本机没有 slurm 客户端
SEED="${CORNERSTONE_SEED_NODE:-}"

pick_seed() {
  [ -n "$SEED" ] && { echo "$SEED"; return; }
  # 上一次记下来的节点优先，省一次全表扫描
  local cache="$REPO/.nodes-seed"
  [ -f "$cache" ] && { local s; s=$(cat "$cache"); \
    timeout 20 "$SSH" "$s" true 2>/dev/null && { echo "$s"; return; }; }
  local ip
  for ip in $(grep -oE '10\.[0-9.]+' "$REPO/.devhosts" 2>/dev/null | sort -u); do
    if timeout 20 "$SSH" "$ip" true 2>/dev/null; then
      echo "$ip" > "$cache"; echo "$ip"; return
    fi
  done
  return 1
}

seed="$(pick_seed)" || { echo "一个节点都连不上 —— 作业可能全过期了，先重新申请" >&2; exit 1; }

# 注意把 127.0.1.1 换回 seed 的 IP —— 节点解析自己的主机名会得到回环地址，
# 直接打出来会让「我是从哪台查的」那一行看着不可用
timeout 60 "$SSH" "$seed" "squeue -u \$USER -h -t RUNNING --format='%N %L' 2>/dev/null |
  while read -r n t; do
    ip=\$(getent hosts \"\$n.nvidia.com\" 2>/dev/null | awk '{print \$1}')
    [ -z \"\$ip\" ] && ip='?'
    [ \"\$ip\" = '127.0.1.1' ] && ip='$seed'
    printf '%-24s %-16s 剩余 %s\n' \"\$n\" \"\$ip\" \"\$t\"
  done" 2>/dev/null
