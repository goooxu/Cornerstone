#!/usr/bin/env bash
# cornerstone 环境探测
#
# 宿主机与容器内都可以运行，输出一份环境报告。
# 用途：确认构建链路（cmake/nvcc/pybind11/TE）与硬件（sm_100/NUMA/磁盘）是否符合预期。
#
#   bash scripts/probe.sh

set -uo pipefail

hr() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
show() { if have "$1"; then printf '%-16s %s\n' "$1" "$("${@:2}" 2>&1 | head -1)"; else printf '%-16s MISSING\n' "$1"; fi; }

hr "运行位置"
if [ -f /.dockerenv ]; then echo "容器内"; else echo "宿主机"; fi
echo "hostname: $(hostname)"
echo "uname:    $(uname -srm)"
echo "user:     $(id -un) uid=$(id -u) gid=$(id -g)"

hr "CPU / NUMA"
lscpu 2>/dev/null | grep -E '^(Architecture|Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)):'
echo "nproc:     $(nproc)"
echo "pagesize:  $(getconf PAGESIZE)"
if have numactl; then
  numactl --hardware 2>/dev/null | grep -E '^node [0-9]+ (cpus|size):' | head -8
else
  # numactl 不在时退回 sysfs
  for n in /sys/devices/system/node/node[01]; do
    [ -e "$n/cpulist" ] && echo "$(basename "$n") cpus: $(cat "$n/cpulist")"
  done
fi

hr "内存"
free -g 2>/dev/null | head -2

hr "GPU"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
  echo "-- NVLink 拓扑 --"
  nvidia-smi topo -m 2>/dev/null | awk 'NR<=5{print $1,$2,$3,$4,$5}'
else
  echo "nvidia-smi MISSING"
fi

hr "构建工具链"
show git     git --version
show g++     g++ --version
show gcc     gcc --version
show cmake   cmake --version
show ninja   ninja --version
show make    make --version
show nvcc    nvcc --version
if have nvcc; then
  echo "nvcc 完整版本: $(nvcc --version | tail -2 | head -1)"
fi

hr "Python 侧"
show python3 python3 --version
python3 - <<'PY' 2>&1
import importlib, sys

print(f"{'sys.executable':<24} {sys.executable}")

def probe(mod, attrs=("__version__",), extra=None):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"{mod:<24} MISSING ({type(e).__name__})")
        return None
    ver = next((getattr(m, a) for a in attrs if hasattr(m, a)), "?")
    print(f"{mod:<24} {ver}")
    if extra:
        try:
            extra(m)
        except Exception as e:
            print(f"{'':<24}   extra probe failed: {e}")
    return m

def torch_extra(t):
    print(f"{'':<24}   cuda={t.version.cuda} available={t.cuda.is_available()}")
    if t.cuda.is_available():
        cap = t.cuda.get_device_capability(0)
        print(f"{'':<24}   device0={t.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} count={t.cuda.device_count()}")
        # FP8 dtype 支持（MXFP8 的 GEMM 依赖这两个 dtype）
        for name in ("float8_e4m3fn", "float8_e5m2"):
            print(f"{'':<24}   torch.{name}: {'yes' if hasattr(t, name) else 'NO'}")
    print(f"{'':<24}   cmake_prefix={t.utils.cmake_prefix_path}")

def pybind11_extra(p):
    print(f"{'':<24}   include={p.get_include()}")
    print(f"{'':<24}   cmakedir={p.get_cmake_dir()}")

probe("torch", extra=torch_extra)
probe("pybind11", extra=pybind11_extra)
probe("transformer_engine")
probe("transformer_engine.pytorch")
probe("triton")
probe("numpy")
probe("fastapi")
probe("uvicorn")
probe("pytest")
probe("setuptools")
PY

hr "Transformer Engine FP8 能力"
python3 - <<'PY' 2>&1
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    print("te.pytorch 导入成功")
    print("可用 recipe:", [n for n in dir(recipe) if not n.startswith('_') and n[0].isupper()])
    for fn in ("fp8_available", "check_fp8_support"):
        f = getattr(te, fn, None)
        if f:
            print(f"te.{fn}() ->", f())
    # Blackwell 上关心 MXFP8 是否可用
    for fn in ("is_mxfp8_available", "check_mxfp8_support"):
        f = getattr(te, fn, None)
        if f:
            print(f"te.{fn}() ->", f())
except Exception as e:
    print(f"TE 探测失败: {type(e).__name__}: {e}")
PY

hr "磁盘"
# 工作目录（可能在网络盘上、容量有限）与本地盘各看一眼
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
df -h "$WORKDIR" /tmp / 2>/dev/null | sort -u

hr "完成"
