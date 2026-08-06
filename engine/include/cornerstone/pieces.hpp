// 21 枚 Blokus 棋子及其全部不同朝向。
//
// 朝向表在静态初始化时由 21 个基础形状生成：每个形状套用 8 种变换
// （4 次旋转 × 是否镜像），归一化到包围盒左上角后去重。去重后总数恰为 91，
// 这个数字在初始化时会被断言检查 —— 它同时也是动作空间 91*196 的来源。

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cornerstone/bitboard.hpp"

namespace cornerstone {

inline constexpr int NUM_PIECES     = 21;
inline constexpr int NUM_ORI        = 91;                    // 去重后的朝向总数
inline constexpr int MAX_PIECE_SIZE = 5;
inline constexpr int NUM_ACTIONS    = NUM_ORI * NUM_CELLS;   // 91 * 196 = 17836
inline constexpr int TOTAL_SQUARES  = 89;                    // 单方全部棋子的格数之和

// 起始格（0-indexed）。玩家 0 用 (4,4)，玩家 1 用 (9,9)。
inline constexpr int START_R[2] = {4, 9};
inline constexpr int START_C[2] = {4, 9};

struct Orientation {
    uint8_t piece;                       // 所属棋子 0..20
    uint8_t size;                        // 格数 1..5
    uint8_t h, w;                        // 包围盒高/宽
    int8_t  dr[MAX_PIECE_SIZE];          // 相对包围盒左上角的行偏移
    int8_t  dc[MAX_PIECE_SIZE];          // 相对包围盒左上角的列偏移
};

// 全部 91 个朝向。顺序由生成过程确定且稳定 —— 动作编码依赖它，不要随意改动生成顺序。
const std::array<Orientation, NUM_ORI>& orientations();

// 每枚棋子的格数（1..5）
const std::array<uint8_t, NUM_PIECES>& piece_sizes();

// 棋子名（I1/I2/I3/V3/... 便于调试与前端展示）
const std::array<std::string, NUM_PIECES>& piece_names();

// 每枚棋子对应的朝向 id 列表
const std::array<std::vector<uint8_t>, NUM_PIECES>& piece_orientations();

// ---- 动作编码 ----
// action = ori * NUM_CELLS + anchor_cell，anchor_cell 是包围盒左上角所在的逻辑格序号。
constexpr int encode_action(int ori, int anchor_cell) noexcept {
    return ori * NUM_CELLS + anchor_cell;
}
constexpr int action_ori(int action) noexcept { return action / NUM_CELLS; }
constexpr int action_anchor(int action) noexcept { return action % NUM_CELLS; }

// 该 (朝向, 锚点) 组合是否完全落在 14x14 内。索引方式同 encode_action。
const std::vector<uint8_t>& action_in_bounds();

// 该动作对应的棋子格集合（越界的动作返回空 BB）
const std::vector<BB>& action_masks();

// ---- 对称变换 ----
// 起始格集合 {(4,4),(9,9)} 在下面 4 个变换下保持「一个起始格映射到一个起始格」，
// 因此在「以行棋方视角编码 + 显式区分己方/对方起始格平面」的表示下，这 4 个都是合法增广。
// 它们构成克莱因四元群。
enum Symmetry : int {
    SYM_ID        = 0,  // (r,c) -> (r,c)
    SYM_TRANSPOSE = 1,  // (r,c) -> (c,r)          主对角翻转，两个起始格各自不动
    SYM_ROT180    = 2,  // (r,c) -> (13-r,13-c)    两个起始格互换
    SYM_ANTIDIAG  = 3,  // (r,c) -> (13-c,13-r)    两个起始格互换
    NUM_SYM       = 4,
};

// sym_cell[s][cell] -> 变换后的逻辑格序号
const std::array<std::array<int16_t, NUM_CELLS>, NUM_SYM>& sym_cell();

// sym_action[s][action] -> 变换后的动作；越界动作为 -1
const std::array<std::vector<int32_t>, NUM_SYM>& sym_action();

}  // namespace cornerstone
