#!/usr/bin/env bash
# Web 试玩服务管理（常驻）。
#
#   bash scripts/web.sh start [checkpoint]
#   bash scripts/web.sh stop
#   bash scripts/web.sh status
#   bash scripts/web.sh restart [checkpoint]
#
# 不给 checkpoint 就自动找 runs/*/ckpt/latest 里最新的一个；一个都没有就用规则基线。
#
# 用 pid 文件而不是 pkill 来停服务：`pkill -f web/server.py` 会匹配**完整命令行**，
# 任何恰好带上这个字符串的进程（比如远程执行时的 ssh 命令本身）都会被误杀。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(dirname "$REPO")"
RUNS="$WORKDIR/runs"
PIDFILE="$RUNS/web.pid"
LOGFILE="$RUNS/web.log"
PORT="${CORNERSTONE_WEB_PORT:-8080}"

mkdir -p "$RUNS"

# 不给参数时的默认对手：挑**最近还在写**的那条跑，用它的 net:<跑>/latest。
# 用 latest 而不是当时那个具体文件，是因为训练还在继续 ——
# 钉死一个文件的话，服务开着开着模型就旧了，而界面上看不出来。
default_backend() {
  local best="" bestrun=""
  for f in "$RUNS"/*/ckpt/latest; do
    [ -e "$f" ] || continue
    local ck; ck="$(dirname "$f")/$(cat "$f")"
    [ -e "$ck" ] || continue
    if [ -z "$best" ] || [ "$ck" -nt "$best" ]; then
      best="$ck"; bestrun="$(basename "$(dirname "$(dirname "$ck")")")"
    fi
  done
  [ -n "$bestrun" ] && echo "net:$bestrun/latest"
}

# 和 train.sh 一样核对 /proc/<pid>/cmdline，不能只靠 kill -0。
# 这里比 train.sh 更容易踩：开发机每 8 小时过期，容器重建后 PID 从个位数重新开始
# （实测训练进程拿到过 553、688 这种号），而 web.pid 是留在 NFS 上的旧号码。
# 只用 kill -0 的话，一个不相干的低号进程就会让 status 误报「运行中」、
# start 拒绝启动，而 stop 会去杀那个无辜的进程。
alive() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cmd
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" || return 1
  case "$cmd" in
    *web/server.py*) return 0 ;;
    *) return 1 ;;                        # PID 被复用了，当作没在跑
  esac
}

cmd_start() {
  if alive; then echo "已在运行 (pid $(cat "$PIDFILE"))"; return 0; fi
  local ck="${1:-$(default_backend)}"
  local args=(--port "$PORT")
  # 参数可以是后端 ID（rule:xxx / net:跑/文件）也可以是 checkpoint 路径。
  # 界面上无论如何都能切，这里定的只是默认值。
  case "$ck" in
    rule:*|net:*)
      args+=(--backend "$ck"); echo "默认后端: $ck" ;;
    "")
      echo "没找到 checkpoint，默认用规则基线（界面上可切换）" ;;
    *)
      if [ -e "$ck" ]; then
        args+=(--checkpoint "$ck"); echo "默认 checkpoint: $ck"
      else
        echo "给的 checkpoint 不存在：$ck —— 退回规则基线（界面上可切换）" >&2
      fi ;;
  esac

  cd "$REPO"
  nohup python3 -u web/server.py "${args[@]}" >"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  sleep 12
  if alive; then
    echo "已启动 (pid $(cat "$PIDFILE"))，监听 0.0.0.0:$PORT，日志 $LOGFILE"
  else
    echo "启动失败，日志尾部：" >&2
    tail -20 "$LOGFILE" >&2
    return 1
  fi
}

cmd_stop() {
  if ! alive; then echo "未在运行"; rm -f "$PIDFILE"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 20); do alive || break; sleep 0.5; done
  alive && kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "已停止"
}

case "${1:-}" in
  start)   shift; cmd_start "${1:-}" ;;
  stop)    cmd_stop ;;
  restart) shift; cmd_stop; cmd_start "${1:-}" ;;
  status)
    if alive; then
      echo "运行中 (pid $(cat "$PIDFILE"))，端口 $PORT"
      tail -3 "$LOGFILE" 2>/dev/null || true
    else
      echo "未运行"
    fi ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
