#include "cornerstone/playout.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace cornerstone {
namespace {

// splitmix64：状态小、质量够用，每个线程一份，互不干扰
inline uint64_t next_random(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint32_t bounded(uint64_t& s, uint32_t n) {
    // Lemire 的无偏界内取值
    uint32_t x = uint32_t(next_random(s) >> 32);
    uint64_t m = uint64_t(x) * uint64_t(n);
    uint32_t l = uint32_t(m);
    if (l < n) {
        uint32_t t = uint32_t(-int32_t(n)) % n;
        while (l < t) {
            x = uint32_t(next_random(s) >> 32);
            m = uint64_t(x) * uint64_t(n);
            l = uint32_t(m);
        }
    }
    return uint32_t(m >> 32);
}

}  // namespace

int random_playout(Board board, uint64_t& rng_state, PlayoutStats* stats) {
    std::vector<int32_t> moves;
    moves.reserve(1024);

    while (!board.terminal()) {
        board.legal_moves(moves);
        if (stats) ++stats->movegens;
        if (moves.empty()) break;  // 正常情况下 terminal 已经拦住了，这里只是兜底
        board.play(moves[bounded(rng_state, uint32_t(moves.size()))]);
        if (stats) ++stats->plies;
    }

    if (stats) {
        ++stats->games;
        stats->score0 += board.score(0);
        stats->score1 += board.score(1);
        const int r = board.result_for(0);
        if (r > 0) ++stats->wins0;
        else if (r < 0) ++stats->wins1;
        else ++stats->draws;
    }
    return board.result_for(0);
}

PlayoutStats random_playouts(int64_t n_games, uint64_t seed, int threads) {
    threads = std::max(1, threads);
    if (n_games <= 0) return {};

    // static_cast 不能省：写成 per(size_t(threads)) 会被解析成函数声明
    std::vector<PlayoutStats> per(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    pool.reserve(size_t(threads));

    for (int t = 0; t < threads; ++t) {
        const int64_t lo = n_games * t / threads;
        const int64_t hi = n_games * (t + 1) / threads;
        pool.emplace_back([&per, t, lo, hi, seed] {
            uint64_t s = seed + uint64_t(t) * 0x1234567890ABCDEFULL + 1;
            PlayoutStats& st = per[size_t(t)];
            for (int64_t i = lo; i < hi; ++i) random_playout(Board(), s, &st);
        });
    }
    for (auto& th : pool) th.join();

    PlayoutStats total;
    for (const auto& st : per) {
        total.games += st.games;
        total.plies += st.plies;
        total.movegens += st.movegens;
        total.wins0 += st.wins0;
        total.wins1 += st.wins1;
        total.draws += st.draws;
        total.score0 += st.score0;
        total.score1 += st.score1;
    }
    return total;
}

double rollout_value(const Board& board, int n, uint64_t seed) {
    if (n <= 0 || board.terminal()) {
        return board.terminal() ? double(board.result_for(board.current_player())) : 0.0;
    }
    const int me = board.current_player();
    uint64_t s = seed + 1;
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        const int r0 = random_playout(board, s, nullptr);
        sum += (me == 0) ? r0 : -r0;
    }
    return double(sum) / double(n);
}

}  // namespace cornerstone
