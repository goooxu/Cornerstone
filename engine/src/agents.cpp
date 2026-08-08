#include "cornerstone/agents.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#include "cornerstone/playout.hpp"

namespace cornerstone {
namespace {

inline uint64_t next_random(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint32_t bounded(uint64_t& s, uint32_t n) {
    return uint32_t((next_random(s) >> 32) * uint64_t(n) >> 32);
}

inline double uniform01(uint64_t& s) {
    return double(next_random(s) >> 11) * (1.0 / 9007199254740992.0);
}

// 走 m 之后，双方各自还剩多少个角点前沿格。
// 不走 Board::play —— 那会顺带做停手/终局判定，对每个候选着法都做太贵。
void anchors_after(const Board& b, int me, const BB& mask, int& own_cnt, int& opp_cnt) {
    const int op = 1 - me;
    const BB new_own = b.occupancy(me) | mask;
    const BB opp = b.occupancy(op);
    const BB occupied = new_own | opp;

    const BB own_allowed = ~occupied & ~edge_dilate(new_own) & valid_mask();
    own_cnt = (diag_dilate(new_own) & own_allowed).popcount();

    if (b.score(op) > 0) {
        const BB opp_allowed = ~occupied & ~edge_dilate(opp) & valid_mask();
        opp_cnt = (diag_dilate(opp) & opp_allowed).popcount();
    } else {
        // 对方还没落过子，唯一的落点就是它的起始格
        const BB start = BB::single(bit_of(START_R[op], START_C[op]));
        opp_cnt = (start & ~occupied).popcount();
    }
}

// 按 score 选一个下标：temperature<=0 时取最大值（同分均匀随机），否则 softmax 采样
size_t pick(const std::vector<double>& score, double temperature, uint64_t& rng) {
    const size_t n = score.size();
    if (temperature <= 0.0) {
        double best = score[0];
        for (double s : score) best = std::max(best, s);
        size_t chosen = 0, seen = 0;
        for (size_t i = 0; i < n; ++i) {
            if (score[i] >= best - 1e-12) {
                // 蓄水池抽样，一遍扫完就均匀选中一个最优项
                if (bounded(rng, uint32_t(++seen)) == 0) chosen = i;
            }
        }
        return chosen;
    }

    double best = score[0];
    for (double s : score) best = std::max(best, s);
    double total = 0.0;
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = std::exp((score[i] - best) / temperature);
        total += w[i];
    }
    double r = uniform01(rng) * total;
    for (size_t i = 0; i < n; ++i) {
        r -= w[i];
        if (r <= 0.0) return i;
    }
    return n - 1;
}

}  // namespace

int select_move(const Board& board, const AgentConfig& cfg, uint64_t& rng_state) {
    std::vector<int32_t> moves;
    board.legal_moves(moves);
    if (moves.empty()) return -1;

    if (cfg.kind == AgentKind::Random)
        return moves[bounded(rng_state, uint32_t(moves.size()))];

    const int me = board.current_player();
    const auto& oris = orientations();
    const auto& sizes = piece_sizes();
    const auto& masks = action_masks();

    std::vector<double> score(moves.size());

    switch (cfg.kind) {
        case AgentKind::GreedyArea:
            for (size_t i = 0; i < moves.size(); ++i)
                score[i] = double(sizes[oris[action_ori(moves[i])].piece]);
            break;

        case AgentKind::GreedyMobility:
            for (size_t i = 0; i < moves.size(); ++i) {
                int own_cnt = 0, opp_cnt = 0;
                anchors_after(board, me, masks[moves[i]], own_cnt, opp_cnt);
                score[i] = cfg.w_size * double(sizes[oris[action_ori(moves[i])].piece]) +
                           cfg.w_own_anchors * double(own_cnt) -
                           cfg.w_opp_anchors * double(opp_cnt);
            }
            break;

        case AgentKind::FlatMCTS: {
            // 总预算平摊到每个候选着法。着法多的时候每个只分到很少的 rollout，
            // 噪声大是 flat-MCTS 的固有缺陷 —— 它本来就只是个基线。
            const int per = std::max(1, cfg.rollouts / int(moves.size()));
            for (size_t i = 0; i < moves.size(); ++i) {
                Board next = board;
                next.play(moves[i]);
                if (next.terminal()) {
                    score[i] = double(next.result_for(me));
                    continue;
                }
                // rollout_value 是从 next 行棋方视角的，换算回 me 的视角
                const double v = rollout_value(next, per, next_random(rng_state));
                score[i] = (next.current_player() == me) ? v : -v;
            }
            break;
        }

        default:
            break;
    }

    return moves[pick(score, cfg.temperature, rng_state)];
}

namespace {

// 下一局。a_is_first 决定 A 执哪一方。
void play_one(const AgentConfig& ca, const AgentConfig& cb, bool a_is_first,
              int opening_plies, uint64_t& rng, MatchResult& out) {
    Board b;
    for (int i = 0; i < opening_plies && !b.terminal(); ++i) {
        std::vector<int32_t> mv;
        b.legal_moves(mv);
        if (mv.empty()) break;
        b.play(mv[bounded(rng, uint32_t(mv.size()))]);
    }

    while (!b.terminal()) {
        const bool a_to_move = (b.current_player() == 0) == a_is_first;
        const int m = select_move(b, a_to_move ? ca : cb, rng);
        if (m < 0) break;
        b.play(m);
        ++out.plies;
    }

    const int pa = a_is_first ? 0 : 1;
    const int res = b.result_for(pa);
    ++out.games;
    out.score_a += b.score(pa);
    out.score_b += b.score(1 - pa);
    if (a_is_first) ++out.a_as_first;
    if (res > 0) {
        ++out.wins_a;
        if (a_is_first) ++out.a_wins_as_first; else ++out.a_wins_as_second;
    } else if (res < 0) {
        ++out.wins_b;
    } else {
        ++out.draws;
    }
}

void merge(MatchResult& t, const MatchResult& s) {
    t.games += s.games;
    t.wins_a += s.wins_a;
    t.wins_b += s.wins_b;
    t.draws += s.draws;
    t.score_a += s.score_a;
    t.score_b += s.score_b;
    t.plies += s.plies;
    t.a_as_first += s.a_as_first;
    t.a_wins_as_first += s.a_wins_as_first;
    t.a_wins_as_second += s.a_wins_as_second;
}

}  // namespace

MatchResult play_match(const AgentConfig& a, const AgentConfig& b, int64_t games,
                       uint64_t seed, int threads, int opening_plies) {
    threads = std::max(1, threads);
    if (games <= 0) return {};

    std::vector<MatchResult> per(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));

    for (int t = 0; t < threads; ++t) {
        const int64_t lo = games * t / threads;
        const int64_t hi = games * (t + 1) / threads;
        pool.emplace_back([&per, &a, &b, t, lo, hi, seed, opening_plies] {
            MatchResult& out = per[static_cast<size_t>(t)];
            for (int64_t i = lo; i < hi; ++i) {
                // 种子按 i/2 派生，于是第 2k 与第 2k+1 局**开局完全相同**，
                // 只是执先方相反 —— 同一个随机开局双方各走一次。
                // 不这么做的话，某个碰巧利于一方的开局只会被一方碰上。
                // 开局之后 rng 继续被两个智能体消耗，走法自然分岔，不影响独立性。
                // 仍然只依赖 i，所以结果与线程划分无关、可复现。
                uint64_t rng = seed ^ (uint64_t(i / 2) * 0x9E3779B97F4A7C15ULL);
                play_one(a, b, (i % 2) == 0, opening_plies, rng, out);
            }
        });
    }
    for (auto& th : pool) th.join();

    MatchResult total;
    for (const auto& r : per) merge(total, r);
    return total;
}

}  // namespace cornerstone
