#!/usr/bin/env bash
# 训练守护：跑在**本机**上，开发机过期后自动把训练拉回来。
#
#   bash scripts/watch_training.sh once      跑一轮检查就退出（调试用）
#   bash scripts/watch_training.sh start     后台常驻
#   bash scripts/watch_training.sh stop
#   bash scripts/watch_training.sh status
#   bash scripts/watch_training.sh pause / resume
#
# 为什么放本机：开发机每 8 小时过期，守护进程本身不能跟着一起没。
# 本机又挂着同一个 NFS，读训练进度直接读文件，不用 ssh。
#
# 机器绑定表在 repo 根目录的 `.devhosts`（不进 git，因为含开发机地址）：
#     <主机>  <实验名>  <GPU列表>
#
# **一条实验只能绑一台机器。** runs/ 在共享 NFS 上，同一条在两台机器同时跑会
# 互相覆盖 checkpoint。守护脚本按绑定表办事，不会自作主张换机器。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(dirname "$REPO")"
RUNS="$WORKDIR/runs"
HOSTS_FILE="${CORNERSTONE_HOSTS:-$REPO/.devhosts}"
SSH="${CORNERSTONE_SSH:-$HOME/.local/bin/sshx}"
INTERVAL="${CORNERSTONE_WATCH_INTERVAL:-180}"
# 每条腿独占一台机器，引擎线程数按整机给。恢复时要和首次启动用的值一致。
ENGINE_THREADS="${CORNERSTONE_ENGINE_THREADS:-128}"

LOG="$RUNS/watchdog.log"
PIDFILE="$RUNS/watchdog.pid"
PAUSE="$RUNS/watchdog.pause"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15)

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

# 远程执行，带超时。两个细节都是必需的：
#   - timeout：开发机过期时 ssh 会挂住，没有超时整个循环就卡死
#   - </dev/null：ssh 默认从 stdin 读。在 `while read ... done < 文件` 里调 ssh，
#     它会把文件剩下的内容全吃掉，循环只跑一轮就结束 ——
#     表现是「第二台机器永远不被检查」，而且完全没有报错
rexec() {
  local host="$1"; shift
  timeout 120 "$SSH" "${SSH_OPTS[@]}" "$host" "$@" </dev/null 2>&1
}

# 训练是否在跑。用 train.sh status 而不是数进程 ——
# 它会核对 /proc/<pid>/cmdline，避免 PID 被复用导致误判。
is_training() {
  local host="$1" exp="$2"
  rexec "$host" "bash $REPO/scripts/devbox.sh exec bash scripts/train.sh status $exp 2>/dev/null | head -1" \
    | grep -q '训练中'
}

# 恢复训练时**不能自己拼参数表**，只能转交给 ab_experiment.sh。
# 这里曾经复制了一份完整的超参列表，等于同一套配置写在两个地方 ——
# 一旦哪边改了另一边没跟上，恢复出来的就是另一个实验，而对照实验最怕的正是这个：
# 第一次 A/B 就是毁在两条腿 games_per_iter 不一致（1024 vs 2048）上，
# 而且从日志表面完全看不出来，只有把两边的启动命令逐字比对才会发现。
# 现在配置只有 ab_experiment.sh 里那一份。
start_training() {
  local host="$1" exp="$2" devs="$3" arm vars
  case "$exp" in
    *fp8*) arm="start-b"; vars="B_EXP=$exp B_DEVICES=$devs" ;;
    *)     arm="start-a"; vars="A_EXP=$exp A_DEVICES=$devs" ;;
  esac
  rexec "$host" "bash $REPO/scripts/devbox.sh exec \
    env $vars ENGINE_THREADS=$ENGINE_THREADS \
    bash scripts/ab_experiment.sh $arm 2>&1 | tail -2"
}

# 配置里的总步数。从 ab_experiment.sh 里取，而不是在这儿再抄一份 ——
# 抄一份就迟早对不上（这个教训在启动参数上已经吃过一次）。
total_steps() {
  sed -n 's/.*--total-steps \([0-9]\+\).*/\1/p' "$REPO/scripts/ab_experiment.sh" | head -1
}

# 训练是不是已经跑满了。
#
# 不判这个的话，跑满之后会陷入死循环：训练进程一启动就发现
# step >= total_steps，落一份 checkpoint 加几百 MB 的 replay 快照然后退出；
# 守护看它没在跑，3 分钟后再拉一次。既白烧 NFS，日志也会被
# 「恢复失败，下一轮重试」刷满 —— 而它根本不是失败。
finished() {
  local exp="$1" f="$RUNS/$exp/logs/metrics.jsonl" total step
  total="$(total_steps)"
  [ -n "$total" ] && [ -f "$f" ] || return 1
  step="$(tail -1 "$f" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("step",0))' 2>/dev/null)"
  [ -n "$step" ] || return 1
  [ "$step" -ge "$total" ]
}

progress() {   # 直接读 NFS 上的 metrics，不用 ssh
  local exp="$1" f="$RUNS/$exp/logs/metrics.jsonl"
  [ -f "$f" ] || { echo "无数据"; return; }
  tail -1 "$f" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    loss = ("loss %.4f" % d["loss"]) if "loss" in d else "攒数据中"
    print("step %d %s" % (d["step"], loss))
except Exception:
    print("无数据")
' 2>/dev/null || echo "无数据"
}

check_once() {
  [ -f "$PAUSE" ] && { log "已暂停（存在 $PAUSE），跳过本轮"; return; }
  [ -f "$HOSTS_FILE" ] || { log "找不到机器绑定表 $HOSTS_FILE"; return; }

  while read -r host exp devs <&3; do
    case "$host" in ''|\#*) continue ;; esac
    [ -n "${devs:-}" ] || continue

    if ! rexec "$host" true >/dev/null; then
      log "[$exp] 主机不可达（过期或网络抖动），本轮跳过"
      continue
    fi

    if is_training "$host" "$exp"; then
      continue                              # 一切正常，不刷日志
    fi

    if finished "$exp"; then
      # 只记一条就够，别每 3 分钟往日志里刷一遍
      if [ ! -f "$RUNS/$exp/.done" ]; then
        log "[$exp] 已跑满 $(total_steps) 步，训练完成，不再拉起（$(progress "$exp")）"
        : >"$RUNS/$exp/.done"
      fi
      continue
    fi

    log "[$exp] 训练未在运行，尝试恢复（进度 $(progress "$exp")）"
    # 容器可能随机器一起没了，devbox.sh up 是幂等的
    rexec "$host" "bash $REPO/scripts/devbox.sh up" >/dev/null
    local out; out="$(start_training "$host" "$exp" "$devs")"
    log "[$exp] $(echo "$out" | tr '\n' ' ')"

    sleep 30
    if is_training "$host" "$exp"; then
      log "[$exp] 已恢复"
    else
      log "[$exp] 恢复失败，下一轮重试"
    fi
  done 3<"$HOSTS_FILE"
}

alive() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null &&
    grep -q watch_training "/proc/$pid/cmdline" 2>/dev/null
}

case "${1:-}" in
  once)  check_once; echo "已检查一轮，日志见 $LOG" ;;
  loop)  log "守护启动，间隔 ${INTERVAL}s"
         while true; do check_once; sleep "$INTERVAL"; done ;;
  start)
    mkdir -p "$RUNS"
    alive && { echo "已在运行 (pid $(cat "$PIDFILE"))"; exit 0; }
    nohup bash "$0" loop >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    sleep 2
    alive && echo "已启动 (pid $(cat "$PIDFILE"))，间隔 ${INTERVAL}s，日志 $LOG" \
          || { echo "启动失败，见 $LOG" >&2; exit 1; }
    ;;
  stop)
    alive || { echo "未在运行"; rm -f "$PIDFILE"; exit 0; }
    kill "$(cat "$PIDFILE")" 2>/dev/null
    rm -f "$PIDFILE"; log "守护已停止"; echo "已停止" ;;
  pause)  touch "$PAUSE"; echo "已暂停（手动停训练时不会被自动拉起）" ;;
  resume) rm -f "$PAUSE"; echo "已恢复" ;;
  status)
    alive && echo "守护运行中 (pid $(cat "$PIDFILE"))" || echo "守护未运行"
    [ -f "$PAUSE" ] && echo "当前处于暂停状态"
    while read -r host exp devs <&3; do
      case "$host" in ''|\#*) continue ;; esac
      printf '  %-10s %s%s\n' "$exp" "$(progress "$exp")" \
             "$(finished "$exp" && echo '  [已跑满，不再拉起]')"
    done 3<"$HOSTS_FILE" 2>/dev/null
    echo "--- 日志尾部 ---"; tail -8 "$LOG" 2>/dev/null
    exit 0 ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
