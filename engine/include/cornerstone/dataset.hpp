// 从着法序列重建训练批。
//
// replay buffer 只存着法序列（约 9 KB/局），局面特征在取批时重建 ——
// CPU 有 144 核而磁盘只剩百来 GB，用算力换存储是划算的。
//
// 每局只回放一次、一次性吐出该局被抽中的全部 ply，避免「每个样本各回放一次」
// 带来的平方级开销。对称增广也在这里做完：几何变换直接作用在特征平面和合法掩码上，
// 稀疏策略目标的重映射留给 Python（一次 numpy 花式索引就够）。

#pragma once

#include <cstdint>

#include "cornerstone/board.hpp"

namespace cornerstone {

// actions/game_offsets: 各局的着法序列，拼接存放，第 i 局是
//     [ actions[game_offsets[i]], actions[game_offsets[i+1]] )
// want_ply/want_offsets: 各局要抽的 ply 下标，同样拼接存放
// syms: 每个样本用哪个对称变换（长度等于样本总数），传 nullptr 表示不增广
//
// 输出按样本在 want_ply 里的顺序写：
//     planes  [n, NUM_PLANES, 14, 14]
//     scalars [n, NUM_SCALARS]
//     legal   [n, NUM_ACTIONS]  0/1
void build_batch(const int32_t* actions, const int32_t* game_offsets, int n_games,
                 const int32_t* want_ply, const int32_t* want_offsets,
                 const int8_t* syms, float* planes, float* scalars, uint8_t* legal,
                 int threads);

}  // namespace cornerstone
