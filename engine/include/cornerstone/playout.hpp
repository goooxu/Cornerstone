// 纯 C++ 的随机对局。
//
// 两个用途：
//  1. 引擎吞吐基准 —— 全程不碰 Python，测出来的是自博弈里 CPU 侧的真实天花板
//  2. M2 的随机基线与 flat-MCTS 的 rollout 后端

#pragma once

#include <cstdint>

#include "cornerstone/board.hpp"

namespace cornerstone {

struct PlayoutStats {
    int64_t games = 0;
    int64_t plies = 0;       // 累计落子手数
    int64_t movegens = 0;    // 累计着法生成次数
    int64_t wins0 = 0;
    int64_t wins1 = 0;
    int64_t draws = 0;
    int64_t score0 = 0;      // 累计占格数，用来看平均分
    int64_t score1 = 0;
};

// 从 start 出发随机走到终局，返回终局结果（从玩家 0 视角 +1/0/-1）。
// stats 非空时累加统计量。
int random_playout(Board board, uint64_t& rng_state, PlayoutStats* stats);

// 跑 n_games 局随机对局。threads<=1 时单线程。调用方负责释放 GIL。
PlayoutStats random_playouts(int64_t n_games, uint64_t seed, int threads);

// 从给定局面出发跑 n 次 rollout，返回从 board 行棋方视角的平均结果（[-1,1]）。
double rollout_value(const Board& board, int n, uint64_t seed);

}  // namespace cornerstone
