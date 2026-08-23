#include "cornerstone/mcts.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace cornerstone {
namespace {

inline uint64_t next_random(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double uniform01(uint64_t& s) {
    // 排除 0，log 才有定义
    return (double(next_random(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

inline double gumbel(uint64_t& s) { return -std::log(-std::log(uniform01(s))); }

struct Node {
    Board board;
    std::vector<int32_t> actions;
    std::vector<float>   prior;    // 已按合法着法 mask 并归一化
    std::vector<double>  w;        // 累计价值，本节点行棋方视角
    std::vector<int32_t> n;
    std::vector<int32_t> child;    // -1 表示还没建
    float   value = 0.f;           // 网络对本节点的估值，行棋方视角
    int32_t total_n = 0;
    int8_t  player = 0;
    bool    expanded = false;
    bool    terminal = false;
};

// 根节点的 sequential halving 状态
struct RootSH {
    std::vector<int32_t> cand;      // 候选，存 Node::actions 的下标
    std::vector<double>  gscore;    // g_a + log(prior_a)，按 actions 下标存
    int rounds = 1;
    int round = 0;
    int target = 1;                 // 本轮每个候选的累计目标访问数
};

struct GameState {
    Board board;
    std::vector<Node> nodes;
    RootSH sh;
    int sims_done = 0;
    int pending = -1;                                   // 待评估的节点下标
    std::vector<std::pair<int32_t, int32_t>> path;      // (node, action_idx)
    std::vector<MoveTarget> history;
    uint64_t rng = 0;
    int8_t net_player = 0;                              // 评测模式下网络执哪一方
    uint64_t pair_rng = 0;      // 本对开局用的 rng 快照
    bool paired_second = false; // 这一局是不是「同一开局的第二遍」
    int random_plies = 0;       // 本局开头有几手是随机注入的（见 MctsConfig）
};

// 终局节点的价值，从 player 视角
float terminal_value(const Board& b, int player, double from_score) {
    const double wdl = double(b.result_for(player));
    if (from_score <= 0.0) return float(wdl);
    const double diff = double(b.score(player) - b.score(1 - player)) / double(TOTAL_SQUARES);
    return float((1.0 - from_score) * wdl + from_score * diff);
}

Node make_node(const Board& b) {
    Node nd;
    nd.board = b;
    nd.player = int8_t(b.current_player());
    // 根也可能一上来就是终局（评测模式下随机开局有可能直接走完），
    // 必须在这里标出来，否则 prepare 会把它当叶子送去评估，
    // 而它的合法着法数为 0，init_root_sh 里会除零
    nd.terminal = b.terminal();
    return nd;
}

}  // namespace

struct SelfPlayEngine::Impl {
    MctsConfig cfg;
    EvalConfig eval;
    int threads = 1;
    std::vector<GameState> games;
    std::vector<int32_t> batch;     // 本次 prepare 收集到的局下标
    std::atomic<int64_t> finished{0};

    Impl(int num_games, const MctsConfig& c, uint64_t seed, const EvalConfig& ev, int th)
        : cfg(c), eval(ev), threads(std::max(1, th)) {
        if (num_games <= 0) throw std::invalid_argument("num_games 必须为正");
        if (cfg.top_k < 1 || cfg.top_k > MAX_TOPK) throw std::invalid_argument("top_k 越界");
        games.resize(size_t(num_games));
        for (int i = 0; i < num_games; ++i) {
            GameState& g = games[size_t(i)];
            g.rng = seed + uint64_t(i) * 0x9E3779B97F4A7C15ULL + 1;
            g.net_player = int8_t(i % 2);   // 先后手逐局交换
            start_game(g);
        }
        batch.reserve(size_t(num_games));
    }

    void reset_tree(GameState& g) {
        g.nodes.clear();
        g.nodes.push_back(make_node(g.board));
        g.sh = RootSH{};
        g.sims_done = 0;
        g.pending = -1;
        g.path.clear();
    }

    // 开新的一局：评测模式下先走随机开局，再把对手方走到网络该动为止；
    // 自博弈模式下按 cfg.random_opening_* 注入随机开局
    void start_game(GameState& g) {
        g.board.reset();
        g.random_plies = 0;
        if (!eval.enabled) inject_random_opening(g);
        if (eval.enabled) {
            // 开局配对：第一遍先把 rng 存下来，第二遍恢复它，于是两局开局逐手相同，
            // 而 net_player 已经翻过边 —— 同一个随机开局双方各执先一次。
            if (g.paired_second) g.rng = g.pair_rng;
            else                 g.pair_rng = g.rng;
            std::vector<int32_t> mv;
            for (int i = 0; i < eval.opening_plies && !g.board.terminal(); ++i) {
                g.board.legal_moves(mv);
                if (mv.empty()) break;
                g.board.play(mv[next_random(g.rng) % mv.size()]);
            }
            play_opponent_until_net_turn(g);
        }
        reset_tree(g);
    }

    // 自博弈的随机开局：以 prob 的概率走 k 手均匀随机着法，k 从
    // {2, 4, ..., max_plies} 里均匀抽（偶数 —— 两个座位各走一半，
    // 否则随机开局本身就会给某一方送先手）。
    //
    // 和评测那条随机开局路径（start_game 里 eval.enabled 的分支）的**关键区别**：
    // 那条直接落在盘上、**不进 history**，所以带着它记录的着法序列从空盘回放会非法
    // （GameRecord::selfplay 那个标记就是用来拦这种记录的）。
    // 这里必须把随机手也 push 进 history，只是标成 n_top = 0。
    void inject_random_opening(GameState& g) {
        if (cfg.random_opening_prob <= 0.0 || cfg.random_opening_max_plies < 2) return;
        const double u = double(next_random(g.rng) % 1000000u) / 1000000.0;
        if (u >= cfg.random_opening_prob) return;
        const int pairs = cfg.random_opening_max_plies / 2;          // 抽 1..pairs 对
        const int k = 2 * (1 + int(next_random(g.rng) % uint64_t(pairs)));
        std::vector<int32_t> mv;
        for (int i = 0; i < k && !g.board.terminal(); ++i) {
            g.board.legal_moves(mv);
            if (mv.empty()) break;
            MoveTarget mt;
            mt.action = mv[next_random(g.rng) % mv.size()];
            mt.player = int8_t(g.board.current_player());
            mt.n_legal = int32_t(mv.size());
            mt.n_top = 0;                    // 哨兵：没有搜索目标，不参与训练
            mt.rest_prob = 0.f;
            g.history.push_back(mt);
            g.board.play(mt.action);
            ++g.random_plies;
        }
    }

    void play_opponent_until_net_turn(GameState& g) {
        if (!eval.enabled || eval.net_opponent) return;   // 网络对手也要建树，交给正常流程
        while (!g.board.terminal() && g.board.current_player() != g.net_player) {
            const int m = select_move(g.board, eval.opponent, g.rng);
            if (m < 0) break;
            g.board.play(m);
        }
    }

    // ---- 选择 ----

    // 内部节点：Gumbel-AZ 的非根选择规则
    //   argmax_a [ improved_pi(a) - N(a) / (1 + ΣN) ]
    // improved_pi = softmax(log prior + sigma(completed_q))
    int select_internal(const Node& nd, std::vector<double>& scratch) const {
        const size_t k = nd.actions.size();
        double max_n = 0.0;
        double sum_pi_visited = 0.0, sum_pi_q = 0.0;
        for (size_t i = 0; i < k; ++i) {
            max_n = std::max(max_n, double(nd.n[i]));
            if (nd.n[i] > 0) {
                sum_pi_visited += nd.prior[i];
                sum_pi_q += nd.prior[i] * (nd.w[i] / nd.n[i]);
            }
        }
        // v_mix：把根估值与已访问子节点的加权 q 混起来，作为未访问动作的 q 补全值
        double v_mix = nd.value;
        if (nd.total_n > 0 && sum_pi_visited > 1e-12)
            v_mix = (nd.value + double(nd.total_n) * (sum_pi_q / sum_pi_visited)) /
                    (1.0 + double(nd.total_n));

        const double sigma = (cfg.c_visit + max_n) * cfg.c_scale;

        scratch.resize(k);
        double best_logit = -1e300;
        for (size_t i = 0; i < k; ++i) {
            const double q = nd.n[i] > 0 ? (nd.w[i] / nd.n[i]) : v_mix;
            scratch[i] = std::log(double(nd.prior[i]) + 1e-20) + sigma * q;
            best_logit = std::max(best_logit, scratch[i]);
        }
        double total = 0.0;
        for (size_t i = 0; i < k; ++i) {
            scratch[i] = std::exp(scratch[i] - best_logit);
            total += scratch[i];
        }

        const double denom = 1.0 + double(nd.total_n);
        int best = 0;
        double best_score = -1e300;
        for (size_t i = 0; i < k; ++i) {
            const double s = scratch[i] / total - double(nd.n[i]) / denom;
            if (s > best_score) {
                best_score = s;
                best = int(i);
            }
        }
        return best;
    }

    // 根节点：sequential halving。取本轮里访问数还没达标、且访问数最少的候选。
    int select_root(GameState& g) {
        Node& nd = g.nodes[0];
        RootSH& sh = g.sh;
        for (;;) {
            int best = -1;
            int32_t best_n = 0;
            for (int32_t ci : sh.cand) {
                if (nd.n[size_t(ci)] >= sh.target) continue;
                if (best < 0 || nd.n[size_t(ci)] < best_n) {
                    best = ci;
                    best_n = nd.n[size_t(ci)];
                }
            }
            if (best >= 0) return best;

            // 本轮配额都满了
            if (sh.cand.size() > 1) {
                halve(nd, sh);
            } else {
                sh.target += 1;   // 只剩一个候选，把剩余预算全给它
            }
        }
    }

    // 按 g_a + log(prior_a) + sigma(q_a) 排序，留下前一半
    void halve(const Node& nd, RootSH& sh) const {
        double max_n = 0.0;
        for (int32_t ci : sh.cand) max_n = std::max(max_n, double(nd.n[size_t(ci)]));
        const double sigma = (cfg.c_visit + max_n) * cfg.c_scale;

        std::stable_sort(sh.cand.begin(), sh.cand.end(), [&](int32_t a, int32_t b) {
            const double qa = nd.n[size_t(a)] > 0 ? nd.w[size_t(a)] / nd.n[size_t(a)] : -1.0;
            const double qb = nd.n[size_t(b)] > 0 ? nd.w[size_t(b)] / nd.n[size_t(b)] : -1.0;
            return sh.gscore[size_t(a)] + sigma * qa > sh.gscore[size_t(b)] + sigma * qb;
        });
        sh.cand.resize(std::max<size_t>(1, sh.cand.size() / 2));
        sh.round += 1;

        const int per = std::max(1, cfg.simulations / (sh.rounds * int(sh.cand.size())));
        sh.target += per;
    }

    void init_root_sh(GameState& g) {
        Node& nd = g.nodes[0];
        RootSH& sh = g.sh;
        const size_t k = nd.actions.size();
        if (k == 0) return;

        sh.gscore.resize(k);
        for (size_t i = 0; i < k; ++i)
            sh.gscore[i] = std::log(double(nd.prior[i]) + 1e-20) + gumbel(g.rng);

        // Gumbel top-k 不放回采样：取 log prior + Gumbel 最大的 m 个
        std::vector<int32_t> order(k);
        std::iota(order.begin(), order.end(), 0);
        const size_t m = std::min<size_t>(size_t(std::max(1, cfg.max_considered)), k);
        std::partial_sort(order.begin(), order.begin() + long(m), order.end(),
                          [&](int32_t a, int32_t b) { return sh.gscore[size_t(a)] > sh.gscore[size_t(b)]; });
        sh.cand.assign(order.begin(), order.begin() + long(m));

        sh.rounds = 1;
        while ((size_t(1) << sh.rounds) < m) ++sh.rounds;   // ceil(log2(m))
        sh.round = 0;
        sh.target = std::max(1, cfg.simulations / (sh.rounds * int(m)));
    }

    // ---- 模拟 ----

    void backup(GameState& g, double value, int leaf_player) {
        for (auto it = g.path.rbegin(); it != g.path.rend(); ++it) {
            Node& nd = g.nodes[size_t(it->first)];
            const size_t ai = size_t(it->second);
            // 停手会让同一方连走，所以不能按深度奇偶定符号，必须看实际行棋方
            nd.w[ai] += (nd.player == leaf_player) ? value : -value;
            nd.n[ai] += 1;
            nd.total_n += 1;
        }
    }

    // 返回 true 表示产生了一个待评估叶子；false 表示这次模拟已在内部结算完
    bool descend(GameState& g, std::vector<double>& scratch) {
        g.path.clear();
        if (!g.nodes[0].expanded) {
            g.pending = 0;
            return true;
        }
        if (g.nodes[0].terminal) return false;   // 根就是终局，交给 advance 处理

        int idx = 0;
        for (;;) {
            const int ai = (idx == 0) ? select_root(g) : select_internal(g.nodes[size_t(idx)], scratch);
            g.path.push_back({idx, ai});

            int c = g.nodes[size_t(idx)].child[size_t(ai)];
            if (c < 0) {
                Board nb = g.nodes[size_t(idx)].board;
                nb.play(g.nodes[size_t(idx)].actions[size_t(ai)]);
                g.nodes.push_back(make_node(nb));          // 这一步会让之前取的引用失效
                c = int(g.nodes.size()) - 1;
                g.nodes[size_t(idx)].child[size_t(ai)] = c;

                if (g.nodes[size_t(c)].board.terminal()) {
                    Node& cn = g.nodes[size_t(c)];
                    cn.terminal = true;
                    cn.expanded = true;
                    cn.value = terminal_value(cn.board, cn.player, cfg.value_from_score);
                    backup(g, cn.value, cn.player);
                    return false;
                }
                g.pending = c;
                return true;
            }

            if (g.nodes[size_t(c)].terminal) {
                backup(g, g.nodes[size_t(c)].value, g.nodes[size_t(c)].player);
                return false;
            }
            idx = c;
        }
    }

    // ---- 对外接口 ----

    // 把 [0, games.size()) 按线程数切段并行执行；threads<=1 时直接串行调用
    template <typename F>
    void parallel_games(F&& fn) {
        const int t = std::min<int>(threads, int(games.size()));
        if (t <= 1) {
            fn(size_t(0), games.size(), 0);
            return;
        }
        std::vector<std::thread> pool;
        pool.reserve(size_t(t));
        for (int i = 0; i < t; ++i) {
            const size_t lo = games.size() * size_t(i) / size_t(t);
            const size_t hi = games.size() * size_t(i + 1) / size_t(t);
            pool.emplace_back([&fn, lo, hi, i] { fn(lo, hi, i); });
        }
        for (auto& th : pool) th.join();
    }

    int prepare(float* planes, float* scalars, int8_t* which_net) {
        const int t = std::max(1, std::min<int>(threads, int(games.size())));
        std::vector<std::vector<int32_t>> local(static_cast<size_t>(t));

        parallel_games([&](size_t lo, size_t hi, int tid) {
            std::vector<double> scratch;              // 每线程一份，不能共享
            auto& out = local[size_t(tid)];
            for (size_t gi = lo; gi < hi; ++gi) {
                GameState& g = games[gi];
                if (g.nodes[0].terminal) continue;    // 根终局，等 advance 收尾
                // 根的评估本来就不计入模拟预算（见 expand 里 ni==0 的分支），
                // 但原先它藏在下面这个循环的第一次 descend 里 —— 于是 simulations=0
                // 时循环一次都不进，根永远等不到评估，advance 又因为根没展开直接跳过，
                // 整个驱动循环空转。提到外面之后 simulations=0 才有意义：
                // 没有任何子节点被访问 => max_n=0、q 全等于 v_mix =>
                // improved_policy = log(prior) + 常数 => argmax 就是纯策略。
                //
                // simulations >= 1 的行为不变：原来第一次 descend 处理的也是未展开的根，
                // 这里的 continue 和那边的 break 一样都是跳过本局的后续。
                if (!g.nodes[0].expanded) {
                    if (descend(g, scratch)) {
                        out.push_back(int32_t(gi));
                        continue;
                    }
                }
                while (g.sims_done < cfg.simulations) {
                    if (descend(g, scratch)) {
                        out.push_back(int32_t(gi));
                        break;
                    }
                    ++g.sims_done;
                }
            }
        });

        // 按局号顺序合并，保证批内顺序与线程划分无关（结果可复现）
        batch.clear();
        for (auto& v : local) batch.insert(batch.end(), v.begin(), v.end());

        // 写特征也并行：一个局面 9*196 个 float，几百上千个局面时不算白给
        const size_t n = batch.size();
        const int ft = std::max(1, std::min<int>(threads, int(n)));
        if (which_net) {
            // 标记按局：谁在搜索就用谁的网络。同一棵树里所有叶子都归搜索方，
            // 不能按叶子局面的行棋方来分。
            for (size_t k = 0; k < n; ++k) {
                const GameState& g = games[size_t(batch[k])];
                which_net[k] = int8_t(g.nodes[0].player == g.net_player ? 0 : 1);
            }
        }
        if (ft <= 1) {
            for (size_t k = 0; k < n; ++k) write_features(k, planes, scalars);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(size_t(ft));
            for (int i = 0; i < ft; ++i) {
                const size_t lo = n * size_t(i) / size_t(ft);
                const size_t hi = n * size_t(i + 1) / size_t(ft);
                pool.emplace_back([this, lo, hi, planes, scalars] {
                    for (size_t k = lo; k < hi; ++k) write_features(k, planes, scalars);
                });
            }
            for (auto& th : pool) th.join();
        }
        return int(n);
    }

    void write_features(size_t k, float* planes, float* scalars) const {
        const GameState& g = games[size_t(batch[k])];
        g.nodes[size_t(g.pending)].board.features(
            planes + k * NUM_PLANES * PLANE_SIZE, scalars + k * NUM_SCALARS,
            cfg.with_mobility);
    }

    void feed(const float* logits, const float* wdl) {
        // 每个 batch 项对应不同的局，互不相干，可以直接并行。
        // 这里最贵的是 legal_moves()（一次完整的着法生成）。
        const size_t n = batch.size();
        const int t = std::max(1, std::min<int>(threads, int(n)));
        if (t <= 1) {
            feed_range(0, n, logits, wdl);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(size_t(t));
            for (int i = 0; i < t; ++i) {
                const size_t lo = n * size_t(i) / size_t(t);
                const size_t hi = n * size_t(i + 1) / size_t(t);
                pool.emplace_back([this, lo, hi, logits, wdl] { feed_range(lo, hi, logits, wdl); });
            }
            for (auto& th : pool) th.join();
        }
        batch.clear();
    }

    void feed_range(size_t from, size_t to, const float* logits, const float* wdl) {
        std::vector<int32_t> moves;
        for (size_t k = from; k < to; ++k) {
            GameState& g = games[size_t(batch[k])];
            const int ni = g.pending;
            const float* lg = logits + k * NUM_ACTIONS;

            Node& nd = g.nodes[size_t(ni)];
            nd.board.legal_moves(moves);
            const size_t na = moves.size();

            nd.actions.assign(moves.begin(), moves.end());
            nd.prior.resize(na);
            nd.w.assign(na, 0.0);
            nd.n.assign(na, 0);
            nd.child.assign(na, -1);

            float best = -1e30f;
            for (size_t i = 0; i < na; ++i) best = std::max(best, lg[nd.actions[i]]);
            double total = 0.0;
            for (size_t i = 0; i < na; ++i) {
                const double e = std::exp(double(lg[nd.actions[i]] - best));
                nd.prior[i] = float(e);
                total += e;
            }
            for (size_t i = 0; i < na; ++i) nd.prior[i] = float(double(nd.prior[i]) / total);

            const float* p = wdl + k * 3;
            nd.value = p[0] - p[2];                 // P(胜) - P(负)
            nd.expanded = true;
            nd.total_n = 0;

            g.pending = -1;
            if (ni == 0) {
                init_root_sh(g);                    // 根的评估不计入模拟预算
            } else {
                backup(g, nd.value, nd.player);
                ++g.sims_done;
            }
        }
    }

    // 完整的改进策略（长度为该局面的合法着法数）
    void improved_policy(const Node& nd, std::vector<double>& out) const {
        const size_t k = nd.actions.size();
        double max_n = 0.0, sum_pi_visited = 0.0, sum_pi_q = 0.0;
        for (size_t i = 0; i < k; ++i) {
            max_n = std::max(max_n, double(nd.n[i]));
            if (nd.n[i] > 0) {
                sum_pi_visited += nd.prior[i];
                sum_pi_q += nd.prior[i] * (nd.w[i] / nd.n[i]);
            }
        }
        double v_mix = nd.value;
        if (nd.total_n > 0 && sum_pi_visited > 1e-12)
            v_mix = (nd.value + double(nd.total_n) * (sum_pi_q / sum_pi_visited)) /
                    (1.0 + double(nd.total_n));

        const double sigma = (cfg.c_visit + max_n) * cfg.c_scale;
        out.resize(k);
        double best = -1e300;
        for (size_t i = 0; i < k; ++i) {
            const double q = nd.n[i] > 0 ? (nd.w[i] / nd.n[i]) : v_mix;
            out[i] = std::log(double(nd.prior[i]) + 1e-20) + sigma * q;
            best = std::max(best, out[i]);
        }
        double total = 0.0;
        for (size_t i = 0; i < k; ++i) {
            out[i] = std::exp(out[i] - best);
            total += out[i];
        }
        for (size_t i = 0; i < k; ++i) out[i] /= total;
    }

    std::vector<GameRecord> advance() {
        const int t = std::max(1, std::min<int>(threads, int(games.size())));
        std::vector<std::vector<GameRecord>> local(static_cast<size_t>(t));
        parallel_games([&](size_t lo, size_t hi, int tid) {
            advance_range(lo, hi, local[size_t(tid)]);
        });
        std::vector<GameRecord> done;
        for (auto& v : local)
            done.insert(done.end(), std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
        return done;
    }

    void advance_range(size_t lo, size_t hi, std::vector<GameRecord>& done) {
        std::vector<double> pi;
        std::vector<int32_t> order;

        for (size_t gi = lo; gi < hi; ++gi) {
            GameState& g = games[gi];
            Node& root = g.nodes[0];
            if (!root.expanded || root.actions.empty()) {
                // 根还没评估过（首次调用 advance 前必须先 prepare/feed），或者已终局
                if (g.board.terminal()) finish(g, done);
                continue;
            }

            improved_policy(root, pi);
            const size_t k = pi.size();

            // 落子：前若干手用 Gumbel-AZ 的带噪 argmax（Gumbel 噪声即随机性来源），
            // 之后改用去噪的改进策略 argmax
            // 温度窗口按**搜索过的手数**算，不是绝对手数 —— 否则注入 k 手随机开局
            // 之后，k 越大能享受带噪 argmax 的搜索手就越少，随机开局的剂量就悄悄
            // 变成了两个变量。减掉 random_plies，各 k 的搜索段口径一致。
            size_t chosen = 0;
            if (g.board.ply() - g.random_plies < cfg.temperature_plies && !g.sh.cand.empty()) {
                double max_n = 0.0;
                for (size_t i = 0; i < k; ++i) max_n = std::max(max_n, double(root.n[i]));
                const double sigma = (cfg.c_visit + max_n) * cfg.c_scale;
                double best = -1e300;
                for (int32_t ci : g.sh.cand) {
                    const size_t i = size_t(ci);
                    const double q = root.n[i] > 0 ? root.w[i] / root.n[i] : -1.0;
                    const double s = g.sh.gscore[i] + sigma * q;
                    if (s > best) {
                        best = s;
                        chosen = i;
                    }
                }
            } else {
                double best = -1e300;
                for (size_t i = 0; i < k; ++i)
                    if (pi[i] > best) {
                        best = pi[i];
                        chosen = i;
                    }
            }

            // 训练目标：改进策略的 top-K，不归一化，另存尾部质量
            order.resize(k);
            std::iota(order.begin(), order.end(), 0);
            const size_t topk = std::min<size_t>(size_t(cfg.top_k), k);
            std::partial_sort(order.begin(), order.begin() + long(topk), order.end(),
                              [&](int32_t a, int32_t b) { return pi[size_t(a)] > pi[size_t(b)]; });

            MoveTarget mt;
            mt.action = root.actions[chosen];
            mt.player = root.player;
            mt.n_legal = int32_t(k);
            mt.n_top = uint8_t(topk);
            mt.root_value = root.value;
            double head = 0.0;
            for (size_t i = 0; i < topk; ++i) {
                mt.top_action[i] = root.actions[size_t(order[i])];
                mt.top_prob[i] = float(pi[size_t(order[i])]);
                head += pi[size_t(order[i])];
            }
            mt.rest_prob = float(std::max(0.0, 1.0 - head));
            g.history.push_back(mt);

            g.board.play(mt.action);
            play_opponent_until_net_turn(g);      // 评测模式下由基线接着走
            if (g.board.terminal()) {
                finish(g, done);
            } else {
                reset_tree(g);
            }
        }
    }

    void finish(GameState& g, std::vector<GameRecord>& done) {
        GameRecord rec;
        rec.moves = std::move(g.history);
        rec.result0 = int8_t(g.board.result_for(0));
        rec.score0 = int16_t(g.board.score(0));
        rec.score1 = int16_t(g.board.score(1));
        rec.net_player = g.net_player;
        rec.selfplay = !eval.enabled;
        done.push_back(std::move(rec));
        ++finished;

        g.history.clear();
        // 下一局换边，并且重放同一个开局（见 start_game 里的配对逻辑）
        g.net_player = int8_t(1 - g.net_player);
        g.paired_second = !g.paired_second;
        start_game(g);
    }
};

void SelfPlayEngine::set_position(const std::vector<int32_t>& actions) {
    for (GameState& g : impl_->games) {
        g.board.reset();
        for (int32_t a : actions) g.board.play(a);
        g.history.clear();
        // 外部指定的局面不算随机开局 —— 忘了清零的话温度窗口会按上一局的 k 偏移
        g.random_plies = 0;
        impl_->reset_tree(g);
    }
}

SelfPlayEngine::RootInfo SelfPlayEngine::root_info(int game) const {
    RootInfo info;
    const GameState& g = impl_->games.at(size_t(game));
    const Node& root = g.nodes[0];
    if (!root.expanded || root.actions.empty()) return info;

    std::vector<double> pi;
    impl_->improved_policy(root, pi);
    info.actions = root.actions;
    info.visits = root.n;
    info.probs.assign(pi.begin(), pi.end());
    info.priors = root.prior;
    info.value = root.value;
    info.ready = true;
    return info;
}

SelfPlayEngine::SelfPlayEngine(int num_games, const MctsConfig& cfg, uint64_t seed,
                               const EvalConfig& eval, int threads)
    : impl_(std::make_unique<Impl>(num_games, cfg, seed, eval, threads)) {}
SelfPlayEngine::~SelfPlayEngine() = default;

int SelfPlayEngine::num_games() const { return int(impl_->games.size()); }
int SelfPlayEngine::max_batch() const { return int(impl_->games.size()); }
int SelfPlayEngine::prepare(float* planes, float* scalars, int8_t* which_net) {
    return impl_->prepare(planes, scalars, which_net);
}
void SelfPlayEngine::feed(const float* logits, const float* wdl) { impl_->feed(logits, wdl); }
std::vector<GameRecord> SelfPlayEngine::advance() { return impl_->advance(); }
int64_t SelfPlayEngine::finished_games() const { return impl_->finished.load(); }

}  // namespace cornerstone
