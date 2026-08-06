#include "cornerstone/dataset.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cornerstone/pieces.hpp"

namespace cornerstone {
namespace {

void build_range(int lo, int hi, const int32_t* actions, const int32_t* game_offsets,
                 const int32_t* want_ply, const int32_t* want_offsets, const int8_t* syms,
                 float* planes, float* scalars, uint8_t* legal) {
    const auto& scell = sym_cell();
    const auto& saction = sym_action();

    std::vector<float> tmp_planes(size_t(NUM_PLANES) * PLANE_SIZE);
    std::vector<uint8_t> tmp_legal(NUM_ACTIONS);
    std::vector<int32_t> moves;

    for (int g = lo; g < hi; ++g) {
        const int32_t w0 = want_offsets[g], w1 = want_offsets[g + 1];
        if (w0 == w1) continue;

        // 该局要抽的 ply 必须递增，才能一次回放全部拿到
        int32_t next = w0;
        Board b;
        const int32_t a0 = game_offsets[g], a1 = game_offsets[g + 1];

        for (int32_t step = 0; step <= a1 - a0 && next < w1; ++step) {
            while (next < w1 && want_ply[next] == step) {
                const int out = next;
                const int8_t s = syms ? syms[out] : 0;

                float* dp = planes + size_t(out) * NUM_PLANES * PLANE_SIZE;
                float* ds = scalars + size_t(out) * NUM_SCALARS;
                uint8_t* dl = legal + size_t(out) * NUM_ACTIONS;

                if (s == 0) {
                    b.features(dp, ds);
                    b.legal_mask(dl);
                } else {
                    b.features(tmp_planes.data(), ds);
                    b.legal_mask(tmp_legal.data());
                    const auto& cmap = scell[size_t(s)];
                    for (int pl = 0; pl < NUM_PLANES; ++pl) {
                        const float* src = tmp_planes.data() + pl * PLANE_SIZE;
                        float* dst = dp + pl * PLANE_SIZE;
                        for (int cell = 0; cell < NUM_CELLS; ++cell)
                            dst[cmap[size_t(cell)]] = src[cell];
                    }
                    std::memset(dl, 0, NUM_ACTIONS);
                    const auto& amap = saction[size_t(s)];
                    for (int a = 0; a < NUM_ACTIONS; ++a)
                        if (tmp_legal[size_t(a)]) dl[amap[size_t(a)]] = 1;
                }
                ++next;
            }
            if (step < a1 - a0) b.play(actions[a0 + step]);
        }
    }
}

}  // namespace

void build_batch(const int32_t* actions, const int32_t* game_offsets, int n_games,
                 const int32_t* want_ply, const int32_t* want_offsets, const int8_t* syms,
                 float* planes, float* scalars, uint8_t* legal, int threads) {
    threads = std::max(1, std::min(threads, n_games));
    if (n_games <= 0) return;

    if (threads == 1) {
        build_range(0, n_games, actions, game_offsets, want_ply, want_offsets, syms,
                    planes, scalars, legal);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t) {
        const int lo = n_games * t / threads;
        const int hi = n_games * (t + 1) / threads;
        pool.emplace_back(build_range, lo, hi, actions, game_offsets, want_ply, want_offsets,
                          syms, planes, scalars, legal);
    }
    for (auto& th : pool) th.join();
}

}  // namespace cornerstone
