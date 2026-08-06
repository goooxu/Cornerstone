// 朴素参考实现：与 Board 的着法生成完全独立的第二条代码路径。
//
// 刻意不使用位棋盘、不使用角点前沿枚举，而是把局面摊成 14x14 的字符网格，
// 盲扫全部 17836 个 (朝向, 锚点) 组合，对每个候选逐格检查规则。
// 慢得多，但逻辑上和 Board::generate 没有共享代码，适合拿来交叉比对。

#include "cornerstone/reference.hpp"

#include <algorithm>

namespace cornerstone {

std::vector<int32_t> legal_moves_reference(const Board& b) {
    std::vector<int32_t> out;
    if (b.terminal()) return out;

    const int p = b.current_player();

    // 摊成网格：-1 空，0/1 为对应玩家
    int8_t grid[BOARD_N][BOARD_N];
    for (int r = 0; r < BOARD_N; ++r)
        for (int c = 0; c < BOARD_N; ++c) {
            const int bit = bit_of(r, c);
            grid[r][c] = b.occupancy(0).test(bit) ? 0 : (b.occupancy(1).test(bit) ? 1 : -1);
        }

    // 尚未落过子 <=> 占格数为 0
    const bool first = (b.score(p) == 0);
    const int sr = START_R[p], sc = START_C[p];

    const auto& oris = orientations();
    static const int E[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    static const int D[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    for (int o = 0; o < NUM_ORI; ++o) {
        const Orientation& od = oris[o];
        if (!b.piece_remaining(p, od.piece)) continue;

        for (int cell = 0; cell < NUM_CELLS; ++cell) {
            const int ar = cell / BOARD_N, ac = cell % BOARD_N;

            int rr[MAX_PIECE_SIZE], cc[MAX_PIECE_SIZE];
            bool inside = true;
            for (int i = 0; i < od.size; ++i) {
                rr[i] = ar + od.dr[i];
                cc[i] = ac + od.dc[i];
                if (rr[i] < 0 || rr[i] >= BOARD_N || cc[i] < 0 || cc[i] >= BOARD_N) {
                    inside = false;
                    break;
                }
            }
            if (!inside) continue;

            bool ok = true, corner = false, covers_start = false;
            for (int i = 0; i < od.size && ok; ++i) {
                const int r = rr[i], c = cc[i];
                if (grid[r][c] != -1) { ok = false; break; }
                if (r == sr && c == sc) covers_start = true;
                for (auto& d : E) {
                    const int nr = r + d[0], nc = c + d[1];
                    if (nr < 0 || nr >= BOARD_N || nc < 0 || nc >= BOARD_N) continue;
                    if (grid[nr][nc] == p) { ok = false; break; }
                }
                for (auto& d : D) {
                    const int nr = r + d[0], nc = c + d[1];
                    if (nr < 0 || nr >= BOARD_N || nc < 0 || nc >= BOARD_N) continue;
                    if (grid[nr][nc] == p) corner = true;
                }
            }
            if (!ok) continue;

            if (first ? covers_start : corner) out.push_back(encode_action(o, cell));
        }
    }

    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace cornerstone
