#!/usr/bin/env bash
# 训练服务管理（开发机单次会话有时长上限，所以做成可随时停/随时续的形态）。
#
#   bash scripts/train.sh start [exp] [额外参数...]
#   bash scripts/train.sh stop  [exp]
#   bash scripts/train.sh status [exp]
#   bash scripts/train.sh tail  [exp]
#   bash scripts/train.sh resume [exp] [额外参数...]    # 等价于 start，训练本身会自动续
#
# 训练脚本本身就是幂等的：重复执行会从 runs/<exp>/ckpt/latest 自动恢复
# （模型、优化器状态、step、RNG、replay 快照）。这个脚本只管进程生命周期。
#
# 和 web.sh 一样用 pid 文件而不是 pkill —— pkill -f 匹配完整命令行，会误杀
# 任何恰好带上该字符串的进程。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(dirname "$REPO")"
RUNS="$WORKDIR/runs"

EXP="${2:-bf16}"
PIDFILE="$RUNS/$EXP/train.pid"
LOGFILE="$RUNS/$EXP/train.log"

# 只有 kill -0 是不够的：进程崩掉之后 PID 会被系统复用，
# 那时 kill -0 依然成功，status 会误报「还在跑」，stop 更会去杀一个无关进程。
# 所以还要核对 /proc/<pid>/cmdline 确实是本实验的训练进程。
alive() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cmd
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" || return 1
  case "$cmd" in
    *tools/train.py*--exp\ "$EXP"*) return 0 ;;
    *) return 1 ;;                      # PID 被复用了，当作没在跑
  esac
}

cmd_start() {
  mkdir -p "$RUNS/$EXP"
  if alive; then echo "实验 $EXP 已在训练 (pid $(cat "$PIDFILE"))"; return 0; fi
  cd "$REPO"
  # -u 关掉 stdout 缓冲：日志重定向到文件时，print 默认是全缓冲的，
  # 训练跑了半天日志里却什么都看不到
  nohup python3 -u tools/train.py --exp "$EXP" "$@" >>"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  sleep 8
  if alive; then
    echo "已启动 $EXP (pid $(cat "$PIDFILE"))，日志 $LOGFILE"
    echo "查看进度：bash scripts/train.sh tail $EXP"
  else
    echo "启动失败，日志尾部：" >&2
    tail -30 "$LOGFILE" >&2
    return 1
  fi
}

cmd_stop() {
  if ! alive; then echo "$EXP 未在训练"; rm -f "$PIDFILE"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  echo "发送 SIGTERM，等待收尾落盘…"
  kill "$pid" 2>/dev/null || true
  # 训练进程收到 SIGTERM 后会退出当前的自博弈/训练循环并落盘。
  # 时限要给够：收尾要写 checkpoint + 一份几百 MB 的 replay 快照。
  # 超时被强杀的代价不是丢几步 —— 是**最终快照没写成**，
  # 下次续训只能用较旧的周期快照，replay 少掉几十万局面。
  # 实测就吃过一次：一组因此少了 40 万局面，另一条完好，
  # 对照实验凭空多了一个不对称。
  for _ in $(seq 900); do alive || break; sleep 1; done
  if alive; then
    echo "警告：等待 900s 仍未退出，强制结束 —— 最终 replay 快照可能没写成" >&2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  echo "已停止。下次 start 会从 checkpoint 自动续训。"
}

case "${1:-}" in
  start|resume) shift; shift 2>/dev/null || true; cmd_start "$@" ;;
  stop)   cmd_stop ;;
  tail)   tail -f "$LOGFILE" ;;
  status)
    if alive; then
      echo "$EXP 训练中 (pid $(cat "$PIDFILE"))"
    else
      echo "$EXP 未运行"
    fi
    [ -f "$RUNS/$EXP/ckpt/latest" ] && echo "最新 checkpoint: $(cat "$RUNS/$EXP/ckpt/latest")"
    [ -f "$LOGFILE" ] && { echo "--- 日志尾部 ---"; tail -5 "$LOGFILE"; }
    ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
