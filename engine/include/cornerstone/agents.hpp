// 纯规则基线智能体，以及它们之间的对局。
//
// 项目不许用棋谱，所以评测基线也只能由规则本身构造。这里给出一条强度递增的阶梯：
//   随机 < 贪心最大面积 < 贪心+机动性 < flat-MCTS(预算递增)
// 网络训练出来之后，用这条阶梯把 Elo 锚定到一个有绝对意义的零点上。

#pragma once

#include <cstdint>
#include <vector>

#include "cornerstone/board.hpp"

namespace cornerstone {

enum class AgentKind : int {
    Random = 0,         // 均匀随机合法着法
    GreedyArea = 1,     // 只看棋子大小，同分随机
    GreedyMobility = 2, // 棋子大小 + 己方角点前沿 - 对方角点前沿
    FlatMCTS = 3,       // 每个候选着法平摊随机 rollout，取平均结果最好的
};

struct AgentConfig {
    AgentKind kind = AgentKind::Random;

    // FlatMCTS 的**总** rollout 预算，会平摊到当前的全部合法着法上
    int rollouts = 1000;

    // GreedyMobility 的权重。默认值由 tools/tune_mobility.py 扫出来：
    // w_opp 远大于另外两项，等于让启发式变成字典序 —— 先最小化对方角点前沿，
    // 再比棋子大小。压制对方比扩张自己值钱得多，是 Blokus 里的真实规律。
    // 对 greedy-area 的得分率 0.900（约 +382 Elo），继续加大 w_opp 不再有收益。
    double w_size = 4.0;
    double w_own_anchors = 0.5;
    double w_opp_anchors = 20.0;

    // >0 时按 softmax(score/temperature) 采样，=0 时取最优（同分均匀随机）。
    // arena 里给确定性智能体加一点温度，避免同一对手之间反复下出同一局。
    double temperature = 0.0;
};

// 选一个合法着法。board 必须非终局。
int select_move(const Board& board, const AgentConfig& cfg, uint64_t& rng_state);

struct MatchResult {
    int64_t games = 0;
    int64_t wins_a = 0;
    int64_t wins_b = 0;
    int64_t draws = 0;
    int64_t score_a = 0;      // 累计占格数
    int64_t score_b = 0;
    int64_t plies = 0;
    int64_t a_as_first = 0;   // A 执先手的局数
    int64_t a_wins_as_first = 0;
    int64_t a_wins_as_second = 0;
};

// A 与 B 对弈 games 局。
//  - 先后手逐局交换（偶数局 A 执先），所以 games 取偶数才公平
//  - opening_plies > 0 时开局若干手双方均匀随机，用来打散开局
MatchResult play_match(const AgentConfig& a, const AgentConfig& b, int64_t games,
                       uint64_t seed, int threads, int opening_plies);

}  // namespace cornerstone
