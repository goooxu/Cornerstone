#!/usr/bin/env bash
# 在容器内编译 C++ 引擎。宿主机没有 cmake/nvcc，必须在容器里跑：
#   bash scripts/devbox.sh exec bash scripts/build.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$REPO/engine/build"

command -v cmake >/dev/null || { echo "没有 cmake —— 你大概在宿主机上跑，应该用 scripts/devbox.sh exec" >&2; exit 1; }

GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)

cmake -S "$REPO/engine" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" --parallel "$(nproc)"

echo
echo "产物:"
ls -la "$REPO"/cornerstone/_engine*.so
