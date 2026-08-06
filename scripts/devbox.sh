#!/usr/bin/env bash
# cornerstone 开发容器管理（在开发机上运行）
#
#   bash scripts/devbox.sh up            拉起常驻容器（幂等）
#   bash scripts/devbox.sh exec <cmd..>  在容器内执行命令
#   bash scripts/devbox.sh root <cmd..>  以 root 在容器内执行（装包用）
#   bash scripts/devbox.sh shell         进交互 shell
#   bash scripts/devbox.sh down          销毁容器
#   bash scripts/devbox.sh status        查看状态
#
# 设计要点：
#  - 以调用者的 uid/gid 运行，避免在 NFS 工作目录上产生 root 属主文件（NFS 常有 root_squash）
#  - HOME 指向工作目录内，pip / triton 缓存不会写到容器临时层里，换机器后仍在
#  - --network host 便于 Web 试玩工具直接暴露 8080
#  - --ipc=host 让自博弈的共享内存环形缓冲不受默认 64MB /dev/shm 限制

set -euo pipefail

IMAGE="${CORNERSTONE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
NAME="${CORNERSTONE_CONTAINER:-cornerstone}"
# 路径全部由脚本自身位置推出来，不写死 —— 换机器/换用户都不用改，
# 也避免把开发机的目录结构带进 git。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(dirname "$REPO")"
# replay 热数据放本地盘。/raid 是 root 属主写不了，/tmp 挂在本地 NVMe（1.4T 可用）
# 且 CLAUDE.md 明确允许使用，因此用 /tmp。tmpfiles 的清理策略是 30 天龄期，
# 活跃写入的文件不会被清掉；且热数据本身可再生，快照始终回写 NFS。
HOT="${CORNERSTONE_HOT:-/tmp/cornerstone}"
CHOME="$REPO/.container-home"                      # 容器内 HOME

usage() { sed -n '2,20p' "$0"; exit 1; }

exists()  { docker ps -a  --format '{{.Names}}' | grep -qx "$NAME"; }
running() { docker ps     --format '{{.Names}}' | grep -qx "$NAME"; }

cmd_up() {
  mkdir -p "$CHOME" "$WORKDIR/runs"
  if [ ! -d "$HOT" ]; then
    mkdir -p "$HOT" 2>/dev/null || {
      echo "警告: 无法创建 $HOT，replay 热数据将退回到 $WORKDIR/runs（NFS，较慢且容量吃紧）" >&2
      HOT=""
    }
  fi

  if running; then echo "容器 $NAME 已在运行"; return 0; fi
  if exists;  then echo "启动已存在的容器 $NAME"; docker start "$NAME" >/dev/null; return 0; fi

  local mounts=(-v "$WORKDIR:$WORKDIR")
  [ -n "$HOT" ] && mounts+=(-v "$HOT:$HOT")

  echo "创建容器 $NAME（镜像 $IMAGE）"
  docker run -d --name "$NAME" \
    --gpus all \
    --network host \
    --ipc=host \
    --cap-add SYS_NICE \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -e HOME="$CHOME" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRITON_CACHE_DIR="$CHOME/.triton" \
    -e PYTHONPATH="$REPO" \
    -e USER=cornerstone \
    -e LOGNAME=cornerstone \
    "${mounts[@]}" \
    -w "$REPO" \
    "$IMAGE" sleep infinity >/dev/null
  echo "已创建"
}

cmd_exec() { [ $# -gt 0 ] || usage; cmd_up >/dev/null; docker exec -w "$REPO" "$NAME" "$@"; }
cmd_root() { [ $# -gt 0 ] || usage; cmd_up >/dev/null; docker exec -u 0 -w "$REPO" "$NAME" "$@"; }
cmd_shell(){ cmd_up >/dev/null; docker exec -it -w "$REPO" "$NAME" bash; }
cmd_down() { exists && docker rm -f "$NAME" >/dev/null && echo "已销毁 $NAME" || echo "容器 $NAME 不存在"; }
cmd_status(){
  docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  running && docker exec "$NAME" bash -lc 'echo "容器内: $(python3 -c "import torch;print(torch.__version__)" 2>/dev/null) / GPU $(nvidia-smi -L 2>/dev/null | wc -l) 张"'
}

case "${1:-}" in
  up)     shift; cmd_up "$@" ;;
  exec)   shift; cmd_exec "$@" ;;
  root)   shift; cmd_root "$@" ;;
  shell)  shift; cmd_shell "$@" ;;
  down)   shift; cmd_down "$@" ;;
  status) shift; cmd_status "$@" ;;
  *) usage ;;
esac
