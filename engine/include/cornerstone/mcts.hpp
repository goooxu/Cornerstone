// Gumbel AlphaZero 搜索与自博弈。
//
// 为什么是 Gumbel AlphaZero 而不是原版：Blokus 的动作空间 17836，单步合法着法常有
// 几百个。原版 AlphaZero 靠 PUCT + Dirichlet 噪声探索，在这种宽度下要很大的模拟数
// 才能把根节点的访问分布打得有意义。Gumbel-AZ 在根节点用 Gumbel top-k 采样出 m 个
// 候选、再用 sequential halving 分配预算，**低模拟数下的策略改进有理论保证**，
// 正好对上这里的情况。
//
// 批量方式：同时开很多局，每局在任意时刻只有一个待评估的叶子。
// 这样既不需要 virtual loss，也不会有多线程回调 Python 的 GIL 争用 ——
// 批量大小来自「局数」而不是「单局内的并行叶子数」。
//
// 驱动循环（Python 侧）：
//     while True:
//         n = engine.prepare()          # 收集全部待评估叶子的特征
//         if n: engine.feed(logits, wdl)   # 喂网络输出
//         else: games = engine.advance()   # 本手模拟做完，各局落子

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cornerstone/agents.hpp"
#include "cornerstone/board.hpp"

namespace cornerstone {

inline constexpr int MAX_TOPK = 32;   // 训练目标保留的稀疏策略项数

struct MctsConfig {
    int simulations = 128;        // 每手的模拟数 n
    int max_considered = 16;      // Gumbel 根候选数 m
    double c_visit = 50.0;        // sigma(q) = (c_visit + max_N) * c_scale * q
    double c_scale = 1.0;
    int temperature_plies = 12;   // 前若干手按改进策略采样落子，之后取 argmax
    int top_k = MAX_TOPK;         // 稀疏策略目标保留多少项
    double value_from_score = 0.0;  // >0 时把终局占格差按此权重混进价值目标

    // 自博弈的随机开局注入。**这是状态分布的旋钮，temperature_plies 不是。**
    //
    // 实测（tools/diag_diversity.py）：训练到第 8 万步时，512 局自博弈的**首手
    // 全部相同**（首手有 414 种合法着法）；到 15 万步只剩 5 种前两手、15 种前四手，
    // 三分之一的对局逐手重复。replay 名义压着 300 万个局面，有效多样性远小于此。
    //
    // 以 random_opening_prob 的概率给一局注入 k 手均匀随机着法，
    // k 从 {2, 4, ..., random_opening_max_plies} 里均匀抽（**取偶数**，
    // 这样两个座位各走一半，先后手不会被开局本身带偏）。
    //
    // 随机手照常进 history（否则从空盘回放会断），但 n_top = 0 当哨兵，
    // 表示「没有搜索目标」—— Python 侧的 replay 靠这个把它们排除在训练之外。
    double random_opening_prob = 0.0;
    int random_opening_max_plies = 0;
};

// 一手的训练样本。特征不存 —— 回放着法序列就能重建，CPU 有的是。
//
// 策略目标只存 top-K，但**不做归一化**，另外带上尾部剩余的概率质量。
// 这一点很重要：训练早期先验接近均匀、单步合法着法常有几百个，
// 若把 top-32 归一化当目标，等于在教网络集中到一组本质上任意的着法上。
// 训练时把 rest_prob 均摊到其余合法着法上，才是对改进策略的无偏近似。
struct MoveTarget {
    int32_t action = -1;                  // 实际落的子
    int8_t  player = 0;
    uint8_t n_top = 0;
    int32_t n_legal = 0;                  // 该局面的合法着法总数
    float   rest_prob = 0.f;              // top-K 之外的概率质量
    int32_t top_action[MAX_TOPK] = {};    // 改进策略的 top-K（按概率降序）
    float   top_prob[MAX_TOPK] = {};
    float   root_value = 0.f;             // 网络对根局面的估值（行棋方视角），用于诊断
};

struct GameRecord {
    std::vector<MoveTarget> moves;
    int8_t  result0 = 0;      // 玩家 0 视角 +1/0/-1
    int16_t score0 = 0;
    int16_t score1 = 0;
    int8_t  net_player = 0;   // 评测模式下网络执哪一方；自博弈时无意义

    // 自博弈记录里 moves 是**完整**的着法序列，可以从空棋盘逐手回放重建局面 ——
    // replay buffer 正是靠这一点只存着法。
    // 评测记录里只有网络方走的手（随机开局与对手的着法都没记），回放会得到非法序列，
    // 所以**不能**进 replay buffer。这个标记就是用来拦住误用的。
    bool selfplay = true;
};

// 评测模式。两种对手：
//
//  1. 规则基线（net_opponent = false）：网络方用 Gumbel-AZ 搜索，
//     轮到对手时不建树，直接用 AgentConfig 选点。
//
//  2. 另一个网络（net_opponent = true）：两方都建树搜索。
//     此时 prepare() 会额外输出每个待评估局面该由哪个网络来算 ——
//     标记是**按局**的（谁在搜索就用谁的网络），不是按局面深度。
//     网络强过全部规则基线之后，这是唯一还能继续量 Elo 的办法。
struct EvalConfig {
    bool enabled = false;
    AgentConfig opponent;
    bool net_opponent = false;
    int opening_plies = 2;    // 开局随机手数（双方各 1 手），打散重复对局
};

class SelfPlayEngine {
public:
    // threads > 1 时把各局的树搜索分摊到多个线程。各局的树完全独立，
    // 是天然可并行的；单线程时 CPU 只用得上一个核，而着法生成与树操作
    // 恰恰是 CPU 侧的主要开销。
    SelfPlayEngine(int num_games, const MctsConfig& cfg, uint64_t seed,
                   const EvalConfig& eval = EvalConfig{}, int threads = 1);
    ~SelfPlayEngine();

    SelfPlayEngine(const SelfPlayEngine&) = delete;
    SelfPlayEngine& operator=(const SelfPlayEngine&) = delete;

    int num_games() const;
    // 一次 prepare 最多产出 num_games 个待评估局面
    int max_batch() const;

    // 把待评估叶子的特征写进 planes / scalars（容量需 >= max_batch()）。
    // which_net 非空时，写入每个局面该用哪个网络（0 = 主网络，1 = 对手网络），
    // 只在 net_opponent 模式下有意义。
    // 返回实际数量；返回 0 表示本手的模拟已经做完，该调 advance()。
    int prepare(float* planes, float* scalars, int8_t* which_net = nullptr);

    // 喂入上一次 prepare 收集到的那批局面的网络输出。
    //   logits: [n, NUM_ACTIONS]（未经 mask，引擎内部会按合法着法 mask 并归一化）
    //   wdl:    [n, 3] 胜/和/负的概率（行棋方视角）
    void feed(const float* logits, const float* wdl);

    // 各局按 Gumbel-AZ 落一子。返回本次落子后走完的对局。
    std::vector<GameRecord> advance();

    // 已经走完的总局数
    int64_t finished_games() const;

    // 把所有局重置到给定着法序列对应的局面。Web 试玩工具用它对单个局面做搜索：
    // num_games=1 + set_position + 跑完模拟 + 读 root_info，不必调 advance()。
    void set_position(const std::vector<int32_t>& actions);

    // 搜索结束后根节点的信息。probs 是改进策略（与 actions 一一对应），
    // value 是网络对根局面的估值（行棋方视角，P(胜)-P(负)）。
    struct RootInfo {
        std::vector<int32_t> actions;
        std::vector<float>   probs;
        std::vector<int32_t> visits;
        std::vector<float>   priors;
        float value = 0.f;
        bool  ready = false;    // 根还没被评估时为 false
    };
    RootInfo root_info(int game = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cornerstone
